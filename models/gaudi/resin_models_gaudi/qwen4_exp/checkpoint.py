"""Qwen3.8-Flash-Next (qwen4_exp) weights in the Gaudi kernels' layouts, one rank's share at a time.

The checkpoint (``Qwen/Qwen3.8-Flash-Next-FP8``) holds the routed experts in block FP8 (128 x 128 scales,
OCP e4m3) and everything else in BF16. Per rank (expert parallel over ``nranks``, see ``Shard``):

- dense projections: BF16 with the K axis in sigma order (``kperm``) for the bf16 GEMMs;
- hyper connections, per module: ``w`` (4H) f32 (1 + gamma), ``dn`` (384, 4H) = [down (320 rows) | inject
  (4 rows) | 0] and ``up`` (4H, 384) with zero K columns 320 .. 383, both BF16 sigma-K;
- delta net: ``w_in`` [qkv | z], ``w_ba`` [b | a], ``w_out``, ``conv_w`` (4, C), ``ssm_a`` = -exp(A_log), ``dt_bias``,
  ``ssm_norm`` with the value heads in the kernels' order (value head h reads key head h % nk), this rank's
  key heads and the value heads that read them;
- attention: ``w_qkv`` [[q_h | gate_h] x heads, k x groups, v x groups, indexer [q (4 x 128) | k (128)]] with the
  gate rows sigma-permuted (the gate kernel reads them against sigma-ordered attention), ``w_o``, ``qk_norm``,
  ``ix_norm``; this rank's heads and their KV groups, the indexer rows replicated;
- MoE: ``router`` [gate (E rows) | shared gate (1 row)], ``sh_gu``, ``sh_down``, and the local experts packed for
  ``moe_up_w`` / ``moe_down_w`` (``pack_experts``: input-major block-contiguous FP8 bytes, Gaudi ±240 range,
  block scales replicated over 64 lanes);
- the PLE: ``ple.kv`` (5H, 256 x heads) key / value projection in ``ple_cvt``'s column layout with the table's
  scale folded in, norms, ``ple.conv`` (K, 4H), this rank's slice of the n-gram table as FP8 bytes with one zero
  row appended (the gather target of other ranks' rows), and the hashing tables for ``ple_hash``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resin.models.qwen4_exp.config import Qwen4ExpConfig
from resin_models_gaudi.loading.convert import (
    bf16_rne,
    bf16_to_f32,
    halve,
    kperm,
    needs_halving,
    pcat,
    pcopy,
    sigma_perm,
)
from resin_models_gaudi.loading.source import Safetensors
from resin_models_gaudi.qwen4_exp.ngram import NGram

HCR = 384  # padded hyper-connection low rank (320 + 4 inject rows + zeros)
BLOCK = 128
_POOL = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8)))


def _tau(a: np.ndarray) -> np.ndarray:
    """Last axis (256) into tau order: out[4q + j] = a[64 j + q]."""
    s = a.shape
    return np.ascontiguousarray(a.reshape(*s[:-1], 4, 64).swapaxes(-1, -2).reshape(s))


def pack_experts(
    gate, gs, up, us, down, ds
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stacked expert tensors (E, ...) -> (W12, S12, W3, S3) for ``moe_up_w`` / ``moe_down_w``.
    gate / up: (E, F, H) e4m3 bytes (Gaudi range), gs / us: (E, F/128, H/128) f32 block scales; down: (E, H, F)
    bytes, ds: (E, H/128, F/128)."""
    E, F, H = gate.shape
    NB, NO = F // 128, H // 256
    nat = np.concatenate(
        [
            gate.transpose(0, 2, 1).reshape(E, H, NB, 128),
            up.transpose(0, 2, 1).reshape(E, H, NB, 128),
        ],
        3,
    )  # (E, H, NB, 256)
    # block-contiguous: [e][ib][b][i % 128][256]
    W12 = _tau(nat.reshape(E, H // 128, 128, NB, 256).transpose(0, 1, 3, 2, 4))
    S12 = np.empty((E, H // 128, NB, 2, 64), np.float32)
    S12[:, :, :, 0, :] = gs.transpose(0, 2, 1)[..., None]
    S12[:, :, :, 1, :] = us.transpose(0, 2, 1)[..., None]
    W3 = _tau(
        down.transpose(0, 2, 1)
        .reshape(E, F // 128, 128, NO, 256)
        .transpose(0, 1, 3, 2, 4)
    )
    S3 = np.empty((E, F // 128, NO, 2, 64), np.float32)
    S3[...] = ds.transpose(0, 2, 1).reshape(E, F // 128, NO, 2)[..., None]
    return W12, S12, W3, S3


def ple_cvt_source(D: int = 160) -> np.ndarray:
    """Byte of a table row held by each of the 256 bf16 rows ``ple_cvt`` writes per head (-1: padding)."""
    p = np.arange(256)
    q = p % 128
    b = 4 * (q // 2) + q % 2 + 2 * (p >= 128)
    return np.where(b < D, b, -1)


def ple_kv_cols(w: np.ndarray, heads: int, D: int) -> np.ndarray:
    """(M, heads D) f32 key / value weights -> (M, 256 heads) bf16 bits with the columns in ``ple_cvt``'s layout."""
    src = ple_cvt_source(D)
    out = np.zeros((w.shape[0], 256 * heads), np.float32)
    ok = src >= 0
    for h in range(heads):
        out[:, 256 * h + np.nonzero(ok)[0]] = w[:, h * D + src[ok]]
    return bf16_rne(out)


@dataclass(frozen=True, slots=True)
class Shard:
    """Rank ``r`` of ``n``: its experts and its vocabulary slice (always split over the ranks), and its key heads,
    the kernel-order value heads that read them, its attention heads and their KV groups: split over
    ``dense_ranks`` (= n: tensor parallel; 1: the dense weights replicated on every rank, so a layer needs one
    collective, after its experts, and runs as one launch)."""

    rank: int
    nranks: int
    nk: int
    nv: int
    heads: int
    groups: int
    experts: int
    vocab: int
    dense_ranks: int = 0  # 0: nranks

    def __post_init__(self) -> None:
        if not self.dense_ranks:
            object.__setattr__(self, "dense_ranks", self.nranks)
        n, dn = self.nranks, self.dense_ranks
        if self.experts % n or self.vocab % n:
            raise ValueError(f"{n} ranks must divide the experts and the vocabulary")
        if dn not in (1, n):
            raise ValueError("the dense weights are split over every rank or over none")
        if self.nk % dn or self.heads % dn:
            raise ValueError(
                f"{dn} ranks must divide the key heads and the attention heads"
            )
        if not (self.groups % dn == 0 or dn % self.groups == 0):
            raise ValueError("ranks and KV groups must divide one another")

    @property
    def dense_rank(self) -> int:
        return self.rank if self.dense_ranks > 1 else 0

    # ---- delta net (kernel order: value head h reads key head h % nk)
    @property
    def key_heads(self) -> np.ndarray:
        kl = self.nk // self.dense_ranks
        return np.arange(self.dense_rank * kl, (self.dense_rank + 1) * kl)

    @property
    def value_heads(self) -> np.ndarray:
        """kernel-order value heads of this rank, in local kernel order"""
        r = self.nv // self.nk
        return np.concatenate([m * self.nk + self.key_heads for m in range(r)])

    @property
    def local_nk(self) -> int:
        return self.nk // self.dense_ranks

    @property
    def local_nv(self) -> int:
        return self.nv // self.dense_ranks

    # ---- attention
    @property
    def attn_heads(self) -> np.ndarray:
        hl = self.heads // self.dense_ranks
        return np.arange(self.dense_rank * hl, (self.dense_rank + 1) * hl)

    @property
    def attn_groups(self) -> np.ndarray:
        hpg = self.heads // self.groups
        return np.unique(self.attn_heads // hpg)

    # ---- experts, vocabulary
    @property
    def expert_range(self) -> tuple[int, int]:
        e = self.experts // self.nranks
        return self.rank * e, (self.rank + 1) * e

    @property
    def vocab_range(self) -> tuple[int, int]:
        v = self.vocab // self.nranks
        return self.rank * v, (self.rank + 1) * v


def _channels(heads: np.ndarray, d: int) -> np.ndarray:
    return (heads[:, None] * d + np.arange(d)[None, :]).reshape(-1)


class PleTable:
    """This rank's rows of the n-gram table: ``shape`` (rows + 1, D) FP8 bytes (Gaudi range; a zero row last),
    streamed to the device a checkpoint shard at a time by ``upload``."""

    def __init__(
        self, files: Safetensors, names: Sequence[str], first_row: int
    ) -> None:
        self.files, self.names, self.first_row = files, list(names), first_row
        self.rows = sum(files.meta[n][3][0] for n in names)
        self.D = files.meta[names[0]][3][1]
        self.shape = (self.rows + 1, self.D)
        self.nbytes = self.shape[0] * self.D
        self.dtype = np.dtype(np.uint8)

    def upload(
        self,
        address: int,
        h2d: Callable[[int, np.ndarray], None],
        memset: Callable[[int, int], None],
    ) -> None:
        """OCP e4m3 shards halved into Gaudi's range (the key / value projection carries the 2), then the zero row."""
        o = 0
        for n in self.names:
            a = np.asarray(self.files.raw(n)[0])
            b = halve(a)
            h2d(address + o, b)
            o += b.nbytes
            del a, b
        memset(address + o, self.D)


class Qwen4ExpCheckpoint:
    """Hugging Face Qwen3.8-Flash-Next checkpoint in the Gaudi kernel layouts."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        top = json.loads((self.path / "config.json").read_text())
        self.config = Qwen4ExpConfig.from_hugging_face(top)
        quantization = top.get("quantization_config") or {}
        if quantization.get("quant_method") != "fp8" or list(
            quantization.get("weight_block_size", [])
        ) != [BLOCK, BLOCK]:
            raise ValueError(
                f"{self.path} is not the block-FP8 release (128 x 128 scales)"
            )
        self.files = Safetensors(self.path, missing_ok=True)
        self.prefix = "model.language_model."
        c = self.config
        r = c.linear_num_value_heads // c.linear_num_key_heads
        nk = c.linear_num_key_heads
        # kernel position j holds HF value head (j % nk) r + j // nk, which reads key head j % nk
        self.value_head_order = np.array(
            [(j % nk) * r + j // nk for j in range(c.linear_num_value_heads)]
        )
        self._halve: bool | None = None

    # ---- raw tensors
    def layer(self, index: int) -> str:
        if index == self.config.num_hidden_layers:
            return "mtp.layers.0."
        return f"{self.prefix}layers.{index}."

    def f32(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype == "BF16":
            return bf16_to_f32(array)
        if dtype == "F32":
            return np.array(array)
        raise ValueError(f"{name}: expected bf16 or f32, got {dtype}")

    def bf16(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype != "BF16":
            raise ValueError(f"{name}: expected bf16, got {dtype}")
        return array

    def i64(self, name: str) -> np.ndarray:
        array, dtype = self.files.raw(name)
        if dtype != "I64":
            raise ValueError(f"{name}: expected i64, got {dtype}")
        return np.asarray(array)

    def fp8(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """-> (e4m3 bytes in Gaudi's ±240 range, f32 block scales)"""
        array, dtype = self.files.raw(name)
        if dtype != "F8_E4M3":
            raise ValueError(f"{name}: expected block FP8, got {dtype}")
        scales = self.f32(name + "_scale_inv").copy()
        weight = np.asarray(array)
        if self._halve is None:
            self._halve = needs_halving(weight)
        if self._halve:
            weight = halve(weight)
            scales *= 2
        return weight, scales

    def norm(self, name: str) -> np.ndarray:
        return np.ascontiguousarray(self.f32(name) + np.float32(1.0))

    def shard(self, rank: int, nranks: int, *, dense_ranks: int = 0) -> Shard:
        """``dense_ranks`` 0: the dense weights split over every rank (tensor parallel); 1: replicated"""
        c = self.config
        return Shard(
            rank,
            nranks,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.num_experts,
            c.vocab_size,
            dense_ranks=dense_ranks,
        )

    # ---- the n-gram table
    def table_shards(self) -> list[str]:
        q = f"{self.layer(self.config.ple_layers[0])}ple.ple_embedding.ngram_embedding.shard_"
        names = [k for k in self.files.meta if k.startswith(q)]
        return sorted(names, key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]))

    def table_rows_per_rank(self, nranks: int) -> list[tuple[int, int]]:
        """Row range [lo, hi) held by each rank: the checkpoint shards split evenly, on shard boundaries."""
        names = self.table_shards()
        sizes = [self.files.meta[n][3][0] for n in names]
        per = len(names) // nranks
        bounds = [sum(sizes[: r * per]) for r in range(nranks)] + [sum(sizes)]
        return [(bounds[r], bounds[r + 1]) for r in range(nranks)]

    def table(self, rank: int, nranks: int) -> PleTable:
        names = self.table_shards()
        per = len(names) // nranks
        lo, hi = rank * per, (rank + 1) * per if rank < nranks - 1 else len(names)
        first = sum(self.files.meta[n][3][0] for n in names[:lo])
        return PleTable(self.files, names[lo:hi], first)

    def ngram(self) -> NGram:
        c = self.config
        q = f"{self.layer(c.ple_layers[0])}ple.ple_embedding."
        mult = [int(v) for v in self.i64(q + "layer_multipliers")]
        ng = NGram(
            c.vocab_size,
            c.ngram_size,
            c.heads_per_ngram,
            c.ngram_vocab_size_base,
            c.eos_token_id,
            mult=mult,
        )
        sizes, offs = (
            self.i64(q + "ngram_heads_vocab_sizes"),
            self.i64(q + "ngram_heads_offsets"),
        )
        if not (
            np.array_equal(sizes, ng.sizes.astype(np.int64))
            and np.array_equal(offs, ng.offs)
        ):
            raise ValueError("n-gram head sizes / offsets differ from the checkpoint")
        return ng

    # ---- prepared weights
    def _hc(
        self, pre: str, inject: bool = True
    ) -> Iterator[tuple[str, np.ndarray, str]]:
        w = self.norm(pre + "hc_norm.weight")
        dn = self.bf16(pre + "input_mix_weight_down.weight")  # (320, 4H)
        up = self.bf16(pre + "input_mix_weight_up.weight")  # (4H, 320)
        D = np.zeros((HCR, dn.shape[1]), np.uint16)
        D[: dn.shape[0]] = dn
        if inject:
            D[dn.shape[0] : dn.shape[0] + 4] = self.bf16(
                pre + "block_inject_weight.weight"
            )
        U = np.zeros((up.shape[0], HCR), np.uint16)
        U[:, : up.shape[1]] = up
        yield "w", w, "f32"
        yield "dn", kperm(D), "bf16"
        yield "up", kperm(U), "bf16"

    def weights(
        self,
        rank: int = 0,
        nranks: int = 1,
        *,
        layers: Collection[int] | None = None,
        table: bool = True,
        mtp: bool = False,
        full_head: bool = False,
        dense_ranks: int = 0,
    ) -> Iterator[tuple[str, object, str]]:
        """Yield ``(name, array | PleTable, kind)`` one device tensor at a time; kinds ``bf16`` / ``f32`` / ``i32`` /
        ``u8`` (``PleTable`` streams itself). ``layers`` limits the decoder layers (the embeddings, final mixer
        and head always come); ``full_head``: the whole LM head on every rank (vocab-parallel otherwise)."""
        c, p, sh = (
            self.config,
            self.prefix,
            self.shard(rank, nranks, dense_ranks=dense_ranks),
        )
        emb = self.bf16(p + "embed_tokens.weight")
        yield "E_gather", pcopy(np.empty(emb.shape, emb.dtype), emb), "bf16"
        head = self.bf16("lm_head.weight")
        if not full_head and nranks > 1:
            lo, hi = sh.vocab_range
            head = head[lo:hi]
        yield "E_head", kperm(pcopy(np.empty(head.shape, head.dtype), head)), "bf16"
        for n, a, k in self._hc(p + "hyper_connection_mixer.", inject=False):
            yield f"hch.{n}", a, k
        indices = list(range(c.num_hidden_layers))
        if mtp:
            indices.append(c.num_hidden_layers)
        if layers is not None:
            indices = [i for i in indices if i in layers]
        for i in indices:
            L = self.layer(i)
            for tag, mod in (
                ("hca", "attn_hyper_connection."),
                ("hcm", "mlp_hyper_connection."),
            ):
                for n, a, k in self._hc(L + mod):
                    yield f"l{i}.{tag}.{n}", a, k
            if c.is_full(i):
                yield from self._attention(i, L, sh)
            else:
                yield from self._delta_net(i, L, sh)
            yield from self._moe(i, L, sh)
            if i in c.ple_layers:
                yield from self._ple(L, rank, nranks, table)
        if mtp:
            yield "mtp.norm_e", self.norm("mtp.pre_fc_norm_embedding.weight"), "f32"
            yield "mtp.norm_h", self.norm("mtp.pre_fc_norm_hidden.weight"), "f32"
            for n, k in (
                ("fc_e", "mtp.fc_embedding.weight"),
                ("fc_h", "mtp.fc_hidden.weight"),
            ):
                a = self.bf16(k)
                yield f"mtp.{n}", kperm(pcopy(np.empty(a.shape, a.dtype), a)), "bf16"
            for n, a, k in self._hc("mtp.hyper_connection_mixer.", inject=False):
                yield f"mtph.{n}", a, k

    def _attention(
        self, i: int, L: str, sh: Shard
    ) -> Iterator[tuple[str, np.ndarray, str]]:
        c = self.config
        A, hpg = c.head_dim, c.heads_per_group
        gate_perm = sigma_perm(A)
        q = self.bf16(L + "self_attn.q_proj.weight")
        q = pcopy(np.empty(q.shape, q.dtype), q)
        for h in range(c.num_attention_heads):  # gate rows sigma-permuted
            g0 = 2 * A * h + A
            q[g0 : g0 + A] = q[g0 : g0 + A][gate_perm]
        k, v = (
            self.bf16(L + "self_attn.k_proj.weight"),
            self.bf16(L + "self_attn.v_proj.weight"),
        )
        ix = self.bf16(L + "self_attn.indexer.index_qk_proj.weight")
        if sh.dense_ranks > 1:
            heads, groups = sh.attn_heads, sh.attn_groups
            q = q[_channels(heads, 2 * A)]
            k, v = k[_channels(groups, A)], v[_channels(groups, A)]
            hpg = len(heads) // len(groups)
        qkv = pcat([q, k, v, ix])
        del q
        yield f"l{i}.w_qkv", kperm(qkv), "bf16"
        o = self.bf16(L + "self_attn.o_proj.weight")
        if sh.dense_ranks > 1:
            o = o[:, _channels(sh.attn_heads, A)]
        # the gate kernel writes the attention output in sigma order per head: K permuted twice
        yield f"l{i}.w_o", kperm(kperm(pcopy(np.empty(o.shape, o.dtype), o))), "bf16"
        qk = np.concatenate(
            [
                self.f32(L + "self_attn.q_norm.weight"),
                self.f32(L + "self_attn.k_norm.weight"),
            ]
        )
        yield f"l{i}.qk_norm", np.ascontiguousarray(qk + np.float32(1.0)), "f32"
        ixn = np.concatenate(
            [
                self.f32(L + "self_attn.indexer.q_layernorm.weight"),
                self.f32(L + "self_attn.indexer.k_layernorm.weight"),
            ]
        )
        yield f"l{i}.ix_norm", np.ascontiguousarray(ixn + np.float32(1.0)), "f32"
        del hpg

    def _delta_net(
        self, i: int, L: str, sh: Shard
    ) -> Iterator[tuple[str, np.ndarray, str]]:
        c = self.config
        a = L + "linear_attn."
        nk, nv, dk = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
        )
        order = self.value_head_order  # kernel position -> HF value head
        vheads = (
            order[sh.value_heads] if sh.dense_ranks > 1 else order
        )  # this rank's HF value heads, kernel order
        kheads = sh.key_heads if sh.dense_ranks > 1 else np.arange(nk)
        qkv = self.bf16(
            a + "in_proj_qkv.weight"
        )  # rows: q (nk dk) | k (nk dk) | v (nv dk)
        z = self.bf16(a + "in_proj_z.weight")  # (nv dk, H)
        rows_q, rows_k = _channels(kheads, dk), nk * dk + _channels(kheads, dk)
        rows_v = 2 * nk * dk + _channels(vheads, dk)
        win = pcat([qkv[rows_q], qkv[rows_k], qkv[rows_v], z[_channels(vheads, dk)]])
        yield f"l{i}.w_in", kperm(win), "bf16"
        del win
        b, al = self.bf16(a + "in_proj_b.weight"), self.bf16(a + "in_proj_a.weight")
        yield f"l{i}.w_ba", kperm(pcat([b[vheads], al[vheads]])), "bf16"
        o = self.bf16(a + "out_proj.weight")
        o = np.ascontiguousarray(o[:, _channels(vheads, dk)])
        # the delta-net kernels write their output in natural order (f32, cast to bf16): natural K
        yield f"l{i}.w_out", kperm(pcopy(np.empty(o.shape, o.dtype), o)), "bf16"
        conv = self.f32(a + "conv1d.weight").reshape(
            c.linear_conv_dim, -1
        )  # (C, 4) rows: q | k | v channels
        conv = np.concatenate([conv[rows_q], conv[rows_k], conv[rows_v]])
        yield f"l{i}.conv_w", np.ascontiguousarray(conv.T), "f32"
        a_log = self.f32(a + "A_log").astype(np.float64)
        yield f"l{i}.ssm_a", (-np.exp(a_log))[vheads].astype(np.float32), "f32"
        yield (
            f"l{i}.dt_bias",
            np.ascontiguousarray(self.f32(a + "dt_bias")[vheads]),
            "f32",
        )
        yield f"l{i}.ssm_norm", np.ascontiguousarray(self.f32(a + "norm.weight")), "f32"
        del nv

    def _moe(self, i: int, L: str, sh: Shard) -> Iterator[tuple[str, np.ndarray, str]]:
        c = self.config
        m = L + "mlp."
        yield (
            f"l{i}.router",
            kperm(
                pcat(
                    [
                        self.bf16(m + "gate.weight"),
                        self.bf16(m + "shared_expert_gate.weight"),
                    ]
                )
            ),
            "bf16",
        )
        yield (
            f"l{i}.sh_gu",
            kperm(
                pcat(
                    [
                        self.bf16(m + "shared_expert.gate_proj.weight"),
                        self.bf16(m + "shared_expert.up_proj.weight"),
                    ]
                )
            ),
            "bf16",
        )
        sd = self.bf16(m + "shared_expert.down_proj.weight")
        yield f"l{i}.sh_down", kperm(pcopy(np.empty(sd.shape, sd.dtype), sd)), "bf16"
        e0, e1 = sh.expert_range
        E_l, H, F = e1 - e0, c.hidden_size, c.moe_intermediate_size
        W12 = np.empty((E_l, H // 128, F // 128, 128, 256), np.uint8)
        S12 = np.empty((E_l, H // 128, F // 128, 2, 64), np.float32)
        W3 = np.empty((E_l, F // 128, H // 256, 128, 256), np.uint8)
        S3 = np.empty((E_l, F // 128, H // 256, 2, 64), np.float32)

        def load(
            k: int,
        ) -> None:  # read + pack one expert (threads: parallel disk reads)
            x = m + f"experts.{e0 + k}."
            gw, gs = self.fp8(x + "gate_proj.weight")
            uw, us = self.fp8(x + "up_proj.weight")
            dw, ds = self.fp8(x + "down_proj.weight")
            r = pack_experts(gw[None], gs[None], uw[None], us[None], dw[None], ds[None])
            W12[k], S12[k], W3[k], S3[k] = (part[0] for part in r)

        list(_POOL.map(load, range(E_l)))
        yield f"l{i}.w12", W12, "u8"
        yield f"l{i}.s12", S12, "f32"
        yield f"l{i}.w3", W3, "u8"
        yield f"l{i}.s3", S3, "f32"

    def _ple(
        self, L: str, rank: int, nranks: int, table: bool
    ) -> Iterator[tuple[str, object, str]]:
        c = self.config
        q = L + "ple."
        scale = float(self.f32(q + "ple_embedding.ngram_embedding.weight_scale")[0])
        if table:
            scale *= 2.0  # the table bytes are halved into Gaudi's range
            tab = self.table(rank, nranks)
            yield "ple.tab", tab, "stream"  # uploads itself, one shard at a time
            A, HP = self.ngram().device_tables(c.vocab_size, tab.first_row)
            yield "ple.A", A, "i32"
            yield "ple.HP", HP, "i32"
        kv = pcat(
            [self.bf16(q + "key_proj.weight"), self.bf16(q + "value_proj.weight")]
        )  # (5H, H)
        yield (
            "ple.kv",
            ple_kv_cols(
                bf16_to_f32(kv) * np.float32(scale), c.ple_heads, c.ple_head_dim
            ),
            "bf16",
        )
        for n in ("key", "query", "conv"):
            yield f"ple.n{n[0]}", self.norm(q + f"norm_{n}.weight"), "f32"
        cw = self.f32(q + "conv1d.weight")  # (4H, 1, K)
        yield (
            "ple.conv",
            np.ascontiguousarray(cw.reshape(cw.shape[0], -1).T),
            "f32",
        )  # (K, 4H)


__all__ = [
    "HCR",
    "PleTable",
    "Qwen4ExpCheckpoint",
    "Shard",
    "pack_experts",
    "ple_cvt_source",
    "ple_kv_cols",
]
