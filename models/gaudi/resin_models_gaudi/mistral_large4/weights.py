"""Mistral-Large-4 device weights for one tensor-parallel rank, prepared from a GGUF.

``prepare`` yields ``(name, array, kind)`` per device tensor. Kinds are ``f32``, ``bf16`` (u16
bits), ``i8`` and ``planar:<qtype>`` (the ``lq_qmv8`` layout as u8, ``[experts, bytes]`` for
stacked experts). The model dispatches GEMVs by kind only, so another checkpoint format (a
safetensors quant) is another ``prepare`` with its own kinds.

Rank r of n holds:
- attention heads ``[r H/n, (r+1) H/n)``: wq rows, wo input columns, k_b / v_b and SSMax;
- every routed expert's intermediate slice (gate / up rows, down input columns), the shared
  expert's and the dense FFN's widths;
- the vocabulary slice of the head.
Replicated: norms, wkv_a (the latent is shared by all heads), the router, the DSA indexer and
the Q8_0 embedding (looked up on the device).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from resin.models.mistral_large4.config import MistralLarge4Config
from resin_models_gaudi.gguf import pack
from resin_models_gaudi.gguf.planar import LAYOUT
from resin_models_gaudi.gguf.reader import GgufModel
from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne

try:
    np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
except AttributeError:  # numpy < 2
    np.core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class RankSlices:
    heads: slice
    expert: slice
    """each routed expert's intermediate columns"""
    shared: slice
    dense: slice
    vocab: slice

    @classmethod
    def of(cls, cfg: MistralLarge4Config, rank: int, nranks: int) -> RankSlices:
        def part(n: int, align: int) -> slice:
            s = n // nranks
            if s * nranks != n or s % align:
                raise ValueError(f"{n} does not split into {nranks} parts of {align}s")
            return slice(rank * s, (rank + 1) * s)

        if not 0 <= rank < nranks:
            raise ValueError(f"rank {rank} of {nranks}")
        return cls(
            heads=part(cfg.num_attention_heads, 1),
            expert=part(cfg.moe_intermediate_size, 256),
            shared=part(cfg.moe_intermediate_size, 256),
            dense=part(cfg.intermediate_size, 256),
            vocab=part(cfg.vocab_size, 128),
        )

    @property
    def num_heads(self) -> int:
        return self.heads.stop - self.heads.start


def f32(model: GgufModel, name: str) -> np.ndarray:
    """any tensor (F32 or a GEMV block type) -> f32 in its numpy shape"""
    t = model.tensor(name)
    if t.qtype == "F32":
        return model.read(name)
    B = model.read(name)
    return pack.dequantize(B.reshape(-1, t.row_bytes), t.qtype).reshape(t.shape)


def _kind(model: GgufModel, name: str) -> str:
    return "planar:" + model.tensor(name).qtype


def _rows(model: GgufModel, name: str, rows: slice | None = None) -> np.ndarray:
    return pack.planar_tensor(model, name, rows=rows)


def _cols(model: GgufModel, name: str, cols: slice) -> np.ndarray:
    return pack.planar_tensor(model, name, inputs=cols)


def embedding_q8(model: GgufModel) -> Iterator[tuple[str, np.ndarray, str]]:
    """token_embd (Q8_0) for device gathers: ``emb_q`` int8 [V, K] and ``emb_s`` f32
    [V, K/32] block scales (value = q * d, as ggml)"""
    name = "token_embd.weight"
    t = model.tensor(name)
    if t.qtype != "Q8_0":
        raise ValueError(f"{name}: expected Q8_0, got {t.qtype}")
    V, K = t.shape
    nb = K // 32
    q, d = np.empty((V, K), np.int8), np.empty((V, nb), np.float32)
    step = 2048

    def one(r0: int) -> None:
        b = model.read_rows(name, slice(r0, min(V, r0 + step))).reshape(-1, nb, 34)
        q[r0 : r0 + len(b)] = b[:, :, 2:].view(np.int8).reshape(len(b), K)
        d[r0 : r0 + len(b)] = np.ascontiguousarray(b[:, :, :2]).view(np.float16)[..., 0]

    list(pack.pool().map(one, range(0, V, step)))
    yield "emb_q", q, "i8"
    yield "emb_s", d, "f32"


def prepare(
    model: GgufModel,
    cfg: MistralLarge4Config,
    rank: int = 0,
    nranks: int = 1,
    layers: range | None = None,
) -> Iterator[tuple[str, np.ndarray, str]]:
    sl = RankSlices.of(cfg, rank, nranks)
    layers = range(cfg.num_hidden_layers) if layers is None else layers
    qk = cfg.qk_head_dim
    hr = slice(sl.heads.start * qk, sl.heads.stop * qk)  # wq rows, wo inputs (v = qk)
    yield "out_norm", f32(model, "output_norm.weight"), "f32"
    yield "head", _rows(model, "output.weight", sl.vocab), _kind(model, "output.weight")
    yield from embedding_q8(model)
    for i in layers:
        p, q = f"blk.{i}.", f"l{i}."
        yield q + "attn_norm", f32(model, p + "attn_norm.weight"), "f32"
        yield q + "ffn_norm", f32(model, p + "ffn_norm.weight"), "f32"
        yield (
            q + "wq",
            _rows(model, p + "attn_q.weight", hr),
            _kind(model, p + "attn_q.weight"),
        )
        if cfg.owns_kv(i):
            n = p + "attn_kv_a_mqa.weight"
            yield q + "wkv", _rows(model, n), _kind(model, n)
            yield q + "kv_norm", f32(model, p + "attn_kv_a_norm.weight"), "f32"
        heads, lat = cfg.num_attention_heads, cfg.kv_lora_rank
        kb = f32(model, p + "attn_k_b.weight").reshape(heads, lat, -1)[sl.heads]
        vb = f32(model, p + "attn_v_b.weight").reshape(heads, cfg.v_head_dim, lat)[
            sl.heads
        ]
        yield q + "kb", bf16_rne(kb), "bf16"  # (h, 512, nope)
        yield q + "vb", bf16_rne(vb), "bf16"  # (h, v, 512)
        yield (
            q + "ss_s",
            np.ascontiguousarray(f32(model, p + "attn_ssmax.weight")[sl.heads]),
            "f32",
        )
        yield (
            q + "ss_b",
            np.ascontiguousarray(f32(model, p + "attn_ssmax.bias")[sl.heads]),
            "f32",
        )
        n = p + "attn_output.weight"
        yield q + "wo", _cols(model, n, hr), _kind(model, n)
        if cfg.index_owners[i]:
            x = p + "indexer."
            yield q + "ix_qn", f32(model, x + "q_norm.weight"), "f32"
            yield q + "ix_qnb", f32(model, x + "q_norm.bias"), "f32"
            yield (
                q + "ix_wq",
                _rows(model, x + "attn_q_b.weight"),
                _kind(model, x + "attn_q_b.weight"),
            )
            yield (
                q + "ix_wk",
                _rows(model, x + "attn_k.weight"),
                _kind(model, x + "attn_k.weight"),
            )
            yield q + "ix_kn", f32(model, x + "k_norm.weight"), "f32"
            yield q + "ix_knb", f32(model, x + "k_norm.bias"), "f32"
            yield q + "ix_proj", f32(model, x + "proj.weight"), "f32"
            yield q + "ix_ss_s", f32(model, x + "ssmax.weight"), "f32"
            yield q + "ix_ss_b", f32(model, x + "ssmax.bias"), "f32"
        if cfg.is_moe(i):
            yield q + "router", f32(model, p + "ffn_gate_inp.weight"), "f32"
            g, u, d = (p + f"ffn_{n}_shexp.weight" for n in ("gate", "up", "down"))
            gu = np.concatenate(
                [_rows(model, g, sl.shared), _rows(model, u, sl.shared)]
            )
            yield q + "shu", gu, _kind(model, g)
            yield q + "shd", _cols(model, d, sl.shared), _kind(model, d)
            g, u, d = (p + f"ffn_{n}_exps.weight" for n in ("gate", "up", "down"))
            if model.tensor(g).qtype != model.tensor(u).qtype:
                raise ValueError(f"{g} and {u} differ in quantization")
            # per expert: gate rows | up rows (one planar matrix of 2 x the slice width)
            xu = pack.planar_experts(model, g, [(sl.expert, None)])
            yield (
                q + "xu",
                np.concatenate(
                    [xu, pack.planar_experts(model, u, [(sl.expert, None)])], 1
                ),
                _kind(model, g),
            )
            del xu
            yield (
                q + "xd",
                pack.planar_experts(model, d, [(None, sl.expert)]),
                _kind(model, d),
            )
        else:
            g, u, d = (p + f"ffn_{n}.weight" for n in ("gate", "up", "down"))
            gu = np.concatenate([_rows(model, g, sl.dense), _rows(model, u, sl.dense)])
            yield q + "du", gu, _kind(model, g)
            yield q + "dd", _cols(model, d, sl.dense), _kind(model, d)


# ------------------------------------------------------------------ host tables
def rope_frequencies(
    cfg: MistralLarge4Config, alpha: np.ndarray | None = None
) -> np.ndarray:
    """inverse frequencies of the rotary dims: transformers' YaRN (truncated correction
    range, attention factor 1) times LeRoPE's learned ``exp(alpha)`` per frequency"""
    dim, base = cfg.qk_rope_head_dim, cfg.rope_theta
    pos = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    extra, inter = 1.0 / pos, 1.0 / (cfg.yarn_factor * pos)

    def correction(rot: float) -> float:
        return (
            dim
            * math.log(cfg.yarn_original_max_position / (rot * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(math.floor(correction(cfg.yarn_beta_fast)), 0)
    high = min(math.ceil(correction(cfg.yarn_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    keep = 1 - ramp
    f = (inter * (1 - keep) + extra * keep).astype(np.float32)
    if alpha is not None:
        f = (f * np.exp(np.asarray(alpha, np.float32))).astype(np.float32)
    return f


def lerope_alpha(model: GgufModel) -> np.ndarray:
    """LeRoPE's per-frequency alpha (the GGUF stores ``rope_freqs = exp(-alpha)``)"""
    return -np.log(f32(model, "rope_freqs.weight").astype(np.float64)).astype(
        np.float32
    )


def rope_table(
    cfg: MistralLarge4Config, alpha: np.ndarray, positions: int
) -> np.ndarray:
    """(positions, 128) f32 rows for ``ml4_rope`` (adjacent pairs, as
    apply_rotary_pos_emb_interleave): ``[cos_(l>>1), l < 64 | (l odd ? +1 : -1) sin_(l>>1)]``"""
    f = rope_frequencies(cfg, alpha)
    ang = np.outer(np.arange(positions, dtype=np.float32), f).astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    half = cfg.qk_rope_head_dim
    sign = np.where(np.arange(half) % 2 == 1, 1.0, -1.0).astype(np.float32)
    return np.ascontiguousarray(
        np.concatenate([np.repeat(c, 2, 1), np.repeat(s, 2, 1) * sign], 1), np.float32
    )


def planar_sb(qtype: str) -> int:
    return LAYOUT[qtype][0]
