"""Qwen3.5/3.8 block-FP8 safetensors, re-laid out for the Gaudi2 graphs.

Layout rules:

* bf16 GEMM weights have K in "sigma order" inside each 128-element block (``kperm``).
* FP8 GEMM weights have K in "tau" order (``tperm``).
* The attention output-gate rows are sigma-permuted, and ``o_proj`` sees sigma then tau.
* Gated-delta-net value heads are reordered so value head ``j`` uses key head ``j % nk``.
* RMSNorm weights carry the ``+1`` (except the gated norm), ``A_log`` becomes
  ``ssm_a = -exp(A_log)``.
* OCP-range checkpoint bytes are halved and their block scales doubled; others load as-is.

Weights stream one fused tensor at a time, so host memory stays far below the model size.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Collection, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.quant import BLOCK, scale_suffix, weight_block

FP8_MAX = 240.0

_POOL = ThreadPoolExecutor(max(4, min(32, (os.cpu_count() or 8) // 2)))
_DTYPES = {
    "BF16": np.uint16,
    "F32": np.float32,
    "F8_E4M3": np.uint8,
    "F16": np.float16,
    "I64": np.int64,
}


def _e4m3_values() -> np.ndarray:
    u = np.arange(256, dtype=np.int32)
    sign = np.where(u & 0x80, -1.0, 1.0)
    exponent, mantissa = (u >> 3) & 15, u & 7
    value = sign * np.where(
        exponent == 0,
        mantissa / 8.0 * 2.0**-6,
        (1 + mantissa / 8.0) * 2.0 ** (exponent - 7.0),
    )
    value[(u & 0x7F) == 0x7F] = np.nan
    return value.astype(np.float32)


E4M3 = _e4m3_values()


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """float32 to bf16 bits, rounding to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """float32 with |x| <= 240 to e4m3 bytes, rounding to nearest even."""
    x = np.asarray(x, np.float32)
    magnitude = np.abs(x)
    _, exponent = np.frexp(magnitude)
    e = np.maximum(exponent.astype(np.int32) - 1, -6)
    quantum = np.ldexp(np.float32(1.0), e - 3).astype(np.float32)
    rounded = np.rint(magnitude / quantum) * quantum
    _, exponent2 = np.frexp(rounded)
    e2 = exponent2.astype(np.int32) - 1
    normal = rounded >= np.float32(2.0**-6)
    field = np.where(normal, e2 + 7, 0)
    mantissa = np.where(
        normal, rounded / np.ldexp(np.float32(1.0), e2) * 8 - 8, rounded * 512
    )
    sign = np.where(np.signbit(x), 0x80, 0)
    return ((field << 3) | mantissa.astype(np.int32) | sign).astype(np.uint8)


def _halve_table() -> np.ndarray:
    values = E4M3.copy()
    values[np.isnan(values)] = 0
    return e4m3_encode(values / 2)


HALVE = _halve_table()


def _row_chunks(n: int, parts: int = 32) -> tuple[range, int]:
    step = max(1, -(-n // parts))
    return range(0, n, step), step


def halve(w: np.ndarray) -> np.ndarray:
    """``HALVE[w]`` in row chunks on many threads. A byte with an exponent field of 2 or more halves
    by taking 1 from the exponent (8 from the byte); subnormals, zero and NaN use the table."""
    out = np.empty(w.shape, np.uint8)
    rows, step = _row_chunks(w.shape[0])

    def run(i: int) -> None:
        src, dst = w[i : i + step], out[i : i + step]
        np.subtract(src, np.uint8(8), out=dst)
        rest = ((src & 0x70) == 0) | ((src & 0x7F) == 0x7F)
        dst[rest] = HALVE[src[rest]]

    list(_POOL.map(run, rows))
    return out


def pcopy(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Row-chunked parallel copy (pages memory-mapped data in with many threads)."""
    rows, step = _row_chunks(dst.shape[0])
    list(_POOL.map(lambda i: np.copyto(dst[i : i + step], src[i : i + step]), rows))
    return dst


def pcat(parts: list[np.ndarray]) -> np.ndarray:
    out = np.empty(
        (sum(p.shape[0] for p in parts),) + parts[0].shape[1:], parts[0].dtype
    )
    offset = 0
    for part in parts:
        pcopy(out[offset : offset + part.shape[0]], part)
        offset += part.shape[0]
    return out


def sigma_perm(n: int) -> np.ndarray:
    """perm[s] = source index of stored slot s, per 128-element block."""
    s = np.arange(128)
    base = np.where(s % 2 == 0, s // 2, 64 + s // 2)
    return (np.arange(n // 128)[:, None] * 128 + base[None, :]).reshape(-1)


def sigma_inv(p: np.ndarray) -> np.ndarray:
    """Stored slot of logical index p (inverse of ``sigma_perm``)."""
    p = np.asarray(p)
    block, r = p & ~127, p & 127
    return block + np.where(r < 64, 2 * r, 2 * (r - 64) + 1)


def tau_perm(n: int) -> np.ndarray:
    """perm[s] = source index of stored slot s, per 256-element block of tau order."""
    s = np.arange(256)
    return (
        np.arange(n // 256)[:, None] * 256 + ((s % 4) * 64 + s // 4)[None, :]
    ).reshape(-1)


def _permute_k(w: np.ndarray, groups: int, inner: int) -> np.ndarray:
    m, k = w.shape
    if k % (groups * inner):
        raise ValueError(f"K {k} is not a multiple of {groups * inner}: pad it first")
    out = np.empty((m, k), w.dtype)
    rows, step = _row_chunks(m)

    def run(i: int) -> None:
        out[i : i + step] = (
            w[i : i + step]
            .reshape(-1, k // (groups * inner), groups, inner)
            .transpose(0, 1, 3, 2)
            .reshape(-1, k)
        )

    list(_POOL.map(run, rows))
    return out


def kperm(w: np.ndarray) -> np.ndarray:
    """K axis into sigma order."""
    return _permute_k(w, 2, 64)


def tperm(w: np.ndarray) -> np.ndarray:
    """K axis into tau order."""
    return _permute_k(w, 4, 64)


class Safetensors:
    """Zero-copy NumPy views of a (possibly sharded) safetensors checkpoint."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.path.glob("*.safetensors"))
        self.meta: dict[str, tuple[Path, int, str, tuple[int, ...]]] = {}
        for name in files:
            file = self.path / name
            with open(file, "rb") as handle:
                (size,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(size))
            header.pop("__metadata__", None)
            for tensor, entry in header.items():
                self.meta[tensor] = (
                    file,
                    8 + size + entry["data_offsets"][0],
                    entry["dtype"],
                    tuple(entry["shape"]),
                )
        self._maps: dict[Path, np.memmap] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.meta

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        file, offset, dtype, shape = self.meta[name]
        if file not in self._maps:
            self._maps[file] = np.memmap(file, np.uint8, "r")
        count = int(np.prod(shape)) * np.dtype(_DTYPES[dtype]).itemsize
        data = self._maps[file][offset : offset + count]
        return data.view(_DTYPES[dtype]).reshape(shape), dtype


@dataclass(frozen=True, slots=True)
class Fp8Weight:
    """Block-FP8 bytes (M, K), K tau-permuted, with f32 block scales (M/128, K/128)."""

    bytes: np.ndarray
    block_scales: np.ndarray


@dataclass(frozen=True, slots=True)
class DenseWeight:
    """A bf16 (uint16 bits) or f32 tensor uploaded as-is."""

    array: np.ndarray


class Qwen35Checkpoint:
    """Hugging Face Qwen3.5/3.8 block-FP8 checkpoint in the Gaudi kernel layouts."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        top = json.loads((self.path / "config.json").read_text())
        self.config = Qwen35Config.from_hugging_face(top, allow_tied_embeddings=True)
        text = top.get("text_config", top)
        quantization = top.get("quantization_config") or text.get("quantization_config")
        if weight_block(quantization) != [BLOCK, BLOCK]:
            raise ValueError(f"{self.path} is not a 128x128 block-FP8 checkpoint")
        self.files = Safetensors(self.path)
        prefix = "model.language_model."
        self.prefix = (
            prefix if any(k.startswith(prefix) for k in self.files.meta) else "model."
        )
        c = self.config
        r = c.linear_num_value_heads // c.linear_num_key_heads
        nk = c.linear_num_key_heads
        self.value_head_order = np.array(
            [(j % nk) * r + j // nk for j in range(c.linear_num_value_heads)]
        )
        self._halve: bool | None = None

    # ---- raw tensors
    def _layer(self, index: int) -> str:
        if index == self.config.num_hidden_layers:
            return "mtp.layers.0."
        return f"{self.prefix}layers.{index}."

    def _f32(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype == "BF16":
            return bf16_to_f32(array)
        if dtype == "F32":
            return np.array(array)
        raise ValueError(f"{name}: expected bf16 or f32, got {dtype}")

    def _bf16(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype != "BF16":
            raise ValueError(f"{name}: expected bf16, got {dtype}")
        return array

    def _fp8(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        array, dtype = self.files.raw(name)
        if dtype != "F8_E4M3":
            raise ValueError(f"{name}: expected block FP8, got {dtype}")
        scales = self._f32(scale_suffix(self.files, name)).copy()
        weight = np.asarray(array)
        if self._halve is None:  # OCP-range bytes (the original release)?
            self._halve = bool(np.nanmax(np.abs(E4M3[weight[:256]])) > FP8_MAX)
        if self._halve:
            weight = halve(weight)
            scales *= 2
        return weight, scales

    def _value_rows(self, w: np.ndarray, scales: np.ndarray, lo: int):
        """Reorder value-head row blocks (128 rows each) starting at row ``lo``."""
        nv = self.config.linear_num_value_heads
        rows = np.arange(w.shape[0])
        heads = rows[lo : lo + nv * BLOCK].reshape(nv, BLOCK)[self.value_head_order]
        rows = np.concatenate([rows[:lo], heads.reshape(-1), rows[lo + nv * BLOCK :]])
        blocks = np.arange(scales.shape[0])
        b0 = lo // BLOCK
        blocks = np.concatenate(
            [
                blocks[:b0],
                blocks[b0 : b0 + nv][self.value_head_order],
                blocks[b0 + nv :],
            ]
        )
        return w[rows], scales[blocks]

    def _fp8_cat(self, parts: list[tuple[np.ndarray, np.ndarray]]) -> Fp8Weight:
        w = pcat([p[0] for p in parts])
        return Fp8Weight(
            tperm(w), np.ascontiguousarray(np.concatenate([p[1] for p in parts], 0))
        )

    def _norm(self, name: str) -> DenseWeight:
        return DenseWeight(np.ascontiguousarray(self._f32(name) + np.float32(1.0)))

    # ---- prepared weights
    def weights(
        self, *, mtp: bool = False, layers: Collection[int] | None = None
    ) -> Iterator[tuple[str, Fp8Weight | DenseWeight]]:
        """Yield ``(name, weight)`` one fused device tensor at a time; ``layers`` limits the
        decoder layers prepared (the embeddings and output norm always are)."""
        c = self.config
        p = self.prefix
        embed = self._bf16(p + "embed_tokens.weight")
        yield "E_gather", DenseWeight(pcopy(np.empty(embed.shape, embed.dtype), embed))
        head = embed if c.tie_word_embeddings else self._bf16("lm_head.weight")
        yield "E_head", DenseWeight(kperm(head))
        yield "out_norm", self._norm(p + "norm.weight")
        gate_perm = sigma_perm(c.head_dim)
        nv, nk, dk = (
            c.linear_num_value_heads,
            c.linear_num_key_heads,
            c.linear_key_head_dim,
        )
        indices = list(range(c.num_hidden_layers))
        if mtp:
            c.require_mtp()
            indices.append(c.num_hidden_layers)
        if layers is not None:
            indices = [i for i in indices if i in layers]
        full = set(c.full_layer_ids)
        for i in indices:
            layer = self._layer(i)
            yield f"l{i}.attn_norm", self._norm(layer + "input_layernorm.weight")
            yield (
                f"l{i}.post_norm",
                self._norm(layer + "post_attention_layernorm.weight"),
            )
            if i in full or i == c.num_hidden_layers:
                q, qs = self._fp8(layer + "self_attn.q_proj.weight")
                q = q.copy()
                for h in range(c.num_attention_heads):  # gate rows sigma-permuted
                    g0 = 2 * c.head_dim * h + c.head_dim
                    q[g0 : g0 + c.head_dim] = q[g0 : g0 + c.head_dim][gate_perm]
                yield (
                    f"l{i}.w_qkv",
                    self._fp8_cat(
                        [
                            (q, qs),
                            self._fp8(layer + "self_attn.k_proj.weight"),
                            self._fp8(layer + "self_attn.v_proj.weight"),
                        ]
                    ),
                )
                o, os_ = self._fp8(layer + "self_attn.o_proj.weight")
                yield f"l{i}.w_o", Fp8Weight(tperm(kperm(o)), os_)
                qk = np.concatenate(
                    [
                        self._f32(layer + "self_attn.q_norm.weight"),
                        self._f32(layer + "self_attn.k_norm.weight"),
                    ]
                )
                yield (
                    f"l{i}.qk_norm",
                    DenseWeight(np.ascontiguousarray(qk + np.float32(1.0))),
                )
            else:
                a = layer + "linear_attn."
                qkv = self._value_rows(
                    *self._fp8(a + "in_proj_qkv.weight"), 2 * nk * dk
                )
                z = self._value_rows(*self._fp8(a + "in_proj_z.weight"), 0)
                yield f"l{i}.w_in", self._fp8_cat([qkv, z])
                order = self.value_head_order
                ba = pcat(
                    [
                        self._bf16(a + "in_proj_b.weight")[order],
                        self._bf16(a + "in_proj_a.weight")[order],
                    ]
                )
                yield f"l{i}.w_ba", DenseWeight(kperm(ba))
                o, os_ = self._fp8(a + "out_proj.weight")
                m = o.shape[0]
                o = np.ascontiguousarray(o.reshape(m, nv, dk)[:, order].reshape(m, -1))
                yield (
                    f"l{i}.w_out",
                    Fp8Weight(tperm(o), np.ascontiguousarray(os_[:, order])),
                )
                conv = self._f32(a + "conv1d.weight").reshape(c.linear_conv_dim, -1)
                lo = 2 * nk * dk
                conv = np.concatenate(
                    [
                        conv[:lo],
                        conv[lo:].reshape(nv, dk, -1)[order].reshape(nv * dk, -1),
                    ]
                )
                yield f"l{i}.conv_w", DenseWeight(np.ascontiguousarray(conv.T))
                a_log = self._f32(a + "A_log").astype(np.float64)
                yield (
                    f"l{i}.ssm_a",
                    DenseWeight((-np.exp(a_log))[order].astype(np.float32)),
                )
                yield (
                    f"l{i}.dt_bias",
                    DenseWeight(np.ascontiguousarray(self._f32(a + "dt_bias")[order])),
                )
                yield (
                    f"l{i}.ssm_norm",
                    DenseWeight(np.ascontiguousarray(self._f32(a + "norm.weight"))),
                )
            yield (
                f"l{i}.w_gu",
                self._fp8_cat(
                    [
                        self._fp8(layer + "mlp.gate_proj.weight"),
                        self._fp8(layer + "mlp.up_proj.weight"),
                    ]
                ),
            )
            d, ds = self._fp8(layer + "mlp.down_proj.weight")
            yield f"l{i}.w_down", Fp8Weight(tperm(d), ds)
        if mtp:
            fc = self._bf16("mtp.fc.weight")
            yield "mtp.fc", DenseWeight(kperm(fc))
            for name, source in (
                ("enorm", "mtp.pre_fc_norm_embedding.weight"),
                ("hnorm", "mtp.pre_fc_norm_hidden.weight"),
                ("norm", "mtp.norm.weight"),
            ):
                yield f"mtp.{name}", self._norm(source)


def rope_table(positions: int, config: Qwen35Config) -> np.ndarray:
    """(positions, 128) f32 rows ``[cos | cos | -sin | sin]`` (NeoX, rotary dim 64)."""
    rotary = config.rotary_dim
    half = rotary // 2
    inverse = config.rope_theta ** (-np.arange(half, dtype=np.float64) * 2.0 / rotary)
    angle = np.arange(positions, dtype=np.float64)[:, None] * inverse[None, :]
    c, s = np.cos(angle), np.sin(angle)
    return np.concatenate([c, c, -s, s], 1).astype(np.float32)
