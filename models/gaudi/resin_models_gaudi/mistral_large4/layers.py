"""Per-layer Synapse graphs of Mistral-Large-4. Each graph is one layer kind reused for every
layer of its kind: weights are persistent inputs bound per launch under generic names.

The residual stream crosses launches as ``x`` (f32 (H, T)) plus the previous segment's partial
sum ``p``: every residual graph starts with ``x_out = x + p`` and ends with this rank's partial
of its block in ``p_out``, which the host all-reduces across ranks between launches (one rank:
nothing to reduce). A step is

    embed, then per layer: attn_pre, attn_slots x n, attn_post, ffn; then head.

Attention is absorbed MLA as one KV group of the rank's ``nh`` heads: queries
``[k_b q_nope | rope(q_rope)]`` (576 dims; 512 in NoPE layers) against cached keys
``[latent | rope(k_rope)]``, values the latent (``attn_slots``: paged slots),
then ``v_b`` per head and ``wo``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from resin.models.mistral_large4.config import MistralLarge4Config
from resin_models_gaudi.gguf.planar import LAYOUT, QMV8, nmem_for
from resin_models_gaudi.gguf.planar import size as planar_size
from resin_models_gaudi.mistral_large4 import lq
from resin_models_gaudi.mistral_large4.cache import (
    ARENA_BLOCK_ROWS,
    ROW_BYTES,
    STORE_FIELDS,
)
from resin_models_gaudi.mistral_large4.weights import RankSlices
from resin_models_gaudi.qwen3_5.sampling import PARAMS, sampler
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F32, I8, I32, U8, Device

GEMM_NT = struct.pack("??", False, True)  # y(M, N) = x(K, N) . w(K, M)^T
GEMM_NN = struct.pack("??", False, False)  # y(M, N) = x(K, N) . w(M, K)
SLOT_KEYS = 2048
ROUTER_LANES = 512  # moe_route reduces 512 logits per token


@dataclass(frozen=True, slots=True)
class Dims:
    """One rank's model dimensions."""

    hidden: int
    heads: int
    """this rank's attention heads"""
    qk: int
    latent: int
    rope: int
    v: int
    dense: int
    expert: int
    shared: int
    experts: int
    top: int
    vocab: int
    """this rank's vocabulary slice"""
    index_heads: int
    index_dim: int
    eps: float
    index_eps: float
    scale: float
    ranks: int = 1
    """tensor-parallel ranks whose partials make the residual"""
    gather_tokens: int = 0
    """launches of at most this many tokens take the ranks' partials all-gathered, (H, T, ranks)"""

    exact: bool = False
    """launches of at most 16 columns compute each column as a one-token launch does, to the bit"""

    def gathered(self, tokens: int) -> bool:
        return self.ranks > 1 and tokens <= self.gather_tokens

    def index_tokens(self, tokens: int) -> int:
        """the columns this rank's indexer scores in a launch of ``tokens``"""
        return tokens // self.ranks if tokens > lq.MAX_COLUMNS else tokens

    @classmethod
    def of(cls, cfg: MistralLarge4Config, sl: RankSlices) -> Dims:
        if cfg.routed_scale != 1.0:
            raise ValueError("routed_scale != 1 is not implemented")
        width = lambda s: s.stop - s.start  # noqa: E731
        return cls(
            hidden=cfg.hidden_size,
            heads=sl.num_heads,
            qk=cfg.qk_head_dim,
            latent=cfg.kv_lora_rank,
            rope=cfg.qk_rope_head_dim,
            v=cfg.v_head_dim,
            dense=width(sl.dense),
            expert=width(sl.expert),
            shared=width(sl.shared),
            experts=cfg.num_experts,
            top=cfg.num_experts_per_tok,
            vocab=width(sl.vocab),
            index_heads=cfg.index_heads,
            index_dim=cfg.index_head_dim,
            eps=cfg.rms_norm_eps,
            index_eps=cfg.index_norm_eps,
            scale=cfg.softmax_scale,
        )


@dataclass(frozen=True, slots=True)
class AttnKind:
    """attn_pre variants: rotary dims, owns a KV cache, owns an indexer, computes indexer
    queries (only steps where some column sees more than top-k keys)."""

    rope: bool
    owns_kv: bool
    owns_index: bool
    index_queries: bool = False

    @property
    def key_dim(self) -> int:
        return 576 if self.rope else 512

    @property
    def tag(self) -> str:
        return f"r{int(self.rope)}k{int(self.owns_kv)}i{int(self.owns_index)}q{int(self.index_queries)}"


class LayerGraph:
    """A Graph with the decoder's building blocks; weights are declared by shape and kind."""

    def __init__(
        self, device: Device, name: str, dims: Dims, tokens: int, mme: bool = False
    ) -> None:
        gathered = dims.gathered(tokens)
        self.g = Graph(
            device,
            name + ("_mme" if mme else "") + (f"_g{dims.ranks}" if gathered else ""),
        )
        self.d = dims
        self.T = tokens
        self.mme = mme
        self._inputs: dict[str, Tensor] = {}

    def input(self, name: str, shape: tuple[int, ...], dtype: int) -> Tensor:
        t = self._inputs.get(name)
        if t is None:
            t = self._inputs[name] = self.g.input(name, shape, dtype)
        return t

    def tensor(self, shape: tuple[int, ...], dtype: int) -> Tensor:
        return self.g.tensor(shape, dtype)

    def node(self, guid: str, ins, outs, params: bytes = b"") -> None:
        self.g.node(guid, ins, outs, params)

    def dep(self) -> Tensor:
        return self.input("dep", (64,), F32)

    def view(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        y = self.tensor(shape, x.dtype)
        self.node("reshape", [x], [y])
        return y

    def flat(self, x: Tensor) -> Tensor:
        n = 1
        for s in x.shape:
            n *= s
        return self.view(x, (n,)) if len(x.shape) > 1 else x

    def cast16(self, x: Tensor) -> Tensor:
        y = self.tensor(x.shape, BF16)
        self.node("cast_f32_to_bf16", [x], [y])
        return y

    def add(self, a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
        y = out if out is not None else self.tensor(a.shape, F32)
        self.node("add_fwd_f32", [a, b], [y])
        return y

    def concat(self, xs: list[Tensor], axis: int) -> Tensor:
        shape = list(xs[0].shape)
        shape[axis] = sum(x.shape[axis] for x in xs)
        y = self.tensor(tuple(shape), xs[0].dtype)
        self.node("concat", xs, [y], struct.pack("I", axis))
        return y

    def split0(self, x: Tensor, sizes: list[int]) -> list[Tensor]:
        outs = [self.tensor((s, *x.shape[1:]), x.dtype) for s in sizes]
        self.node("split", [x], outs, struct.pack("I", 0))
        return outs

    def perm(self, x: Tensor) -> Tensor:
        """(D, A, B) -> (D, B, A)"""
        y = self.tensor((x.shape[0], x.shape[2], x.shape[1]), F32)
        self.node("rg_ml4_perm_f32", [x], [y])
        return y

    def norm(
        self,
        x: Tensor,
        w: str,
        n: int | None = None,
        *,
        layer_norm: bool = False,
        bias: str | None = None,
        eps: float | None = None,
    ) -> Tensor:
        """RMSNorm (or LayerNorm with bias) over the first n rows of each column -> f32"""
        n = n or x.shape[0]
        y = self.tensor((n, x.shape[1]), F32)
        ins = [
            x,
            self.input(w, (n,), F32),
            self.input(bias, (n,), F32) if bias else self.dep(),
        ]
        prm = struct.pack("fi", self.d.eps if eps is None else eps, int(layer_norm))
        self.node("rg_ml4_norm_f32", ins, [y], prm)
        return y

    def planar(
        self, name: str, qtype: str, rows: int, inputs: int, experts: int = 1
    ) -> Tensor:
        return self.input(name, (planar_size(qtype, rows, inputs), experts), U8)

    def gemv(
        self, name: str, qtype: str, x: Tensor, K: int, M: int, ncol: int, **kw
    ) -> Tensor:
        """flat (ncol * M,) f32 of the planar weight over x's columns"""
        experts = kw.pop("experts", 1)
        W = self.planar(name, qtype, M, K, experts)
        if (
            not self.mme or experts != 1 or kw.get("ids") is not None
        ):  # routed: always GEMVs
            kw.setdefault("exact", self.d.exact)
            return lq.gemv(self.g, qtype, W, self.flat(x), self.dep(), K, M, ncol, **kw)
        y = self.tensor((M, ncol), F32)
        self.node(
            "gemm",
            [self.cast16(self.view(x, (K, ncol))), self.dequant(W, qtype, M, K)],
            [y],
            GEMM_NN,
        )
        out = kw.get("out")
        if out is not None:
            self.node("reshape", [y], [out])
            return out
        return self.flat(y)

    def dequant(self, W: Tensor, qtype: str, M: int, K: int) -> Tensor:
        """bf16 W^T (M fastest, K) of a dense planar matrix"""
        if qtype != "Q8_0":
            raise ValueError(f"no dense dequantization kernel for {qtype}")
        sb = LAYOUT[qtype][0]
        nrb = -(-M // 128)
        flat = self.tensor(
            (M * K,), BF16
        )  # the kernel stores at flat offsets k M + row
        prm = struct.pack(
            "iiiiiii", nmem_for(nrb * (K // sb), 2), K, M, nrb, 1, M, M * K
        )
        self.node("rg_lq_qdq_q8_0_f32", [W, self.dep()], [flat], prm)
        return self.view(flat, (M, K))

    def rope(
        self,
        x: Tensor,
        heads: int,
        r0: int,
        s: Tensor,
        b: Tensor,
        head_major: bool,
        columns: Tensor | None = None,
    ) -> Tensor:
        """rotary dims r0 .. r0 + 63 (r0 < 0: none) at each column's position, times the SSMax
        scale s[h] log(pos + 1) + b[h]; columns c = t * heads + h. ``columns``: which of the
        launch's T columns x holds (default: all of them)"""
        D, N = x.shape
        T = N // heads
        y = self.tensor((D, T, heads) if head_major else (D, N), F32)
        cs = self.input("cs", (128, self.T), F32)
        lp = self.input("lp", (self.T,), F32)
        if columns is not None:
            cs, lp = self.pick(cs, columns), self.pick(lp, columns)
        ins = [x, cs, s, b, lp]
        self.node(
            "rg_ml4_rope_f32", ins, [y], struct.pack("iii", heads, r0, int(head_major))
        )
        return y

    def pick(self, x: Tensor, columns: Tensor) -> Tensor:
        """``columns`` of x along its last dimension"""
        y = self.tensor((*x.shape[:-1], columns.shape[0]), x.dtype)
        self.node(
            "gather_fwd_f32", [x, columns], [y], struct.pack("i", len(x.shape) - 1)
        )
        return y

    def no_ssmax(self) -> tuple[Tensor, Tensor]:
        return self.input("zero1", (1,), F32), self.input("one1", (1,), F32)

    def swiglu(self, gu: Tensor) -> Tensor:
        n = gu.shape[1]
        y = self.tensor((gu.shape[0] // 2, n), F32)
        self.node("rg_swiglu_f8_f32", [gu, self.input(f"ones{n}", (1, n), F32)], [y])
        return y

    def residual(self) -> Tensor:
        """x_out = x + p, the previous segment's partials summed over the ranks: all-reduced,
        or all-gathered (H, T, ranks) for small launches (``Dims.gathered``) and summed here"""
        H, T = self.d.hidden, self.T
        if self.d.gathered(T):
            p = self.tensor((H, T), F32)
            self.node("rg_rsum_f32", [self.input("p", (H, T, self.d.ranks), F32)], [p])
        else:
            p = self.input("p", (H, T), F32)
        return self.add(
            self.input("x", (H, T), F32), p, out=self.input("x_out", (H, T), F32)
        )

    def arena(self, blocks: int) -> Tensor:
        """the cache arena as bytes (FP8 latent rows, bf16 rope and index units, f32 scales)"""
        return self.input("arena", (ROW_BYTES, ARENA_BLOCK_ROWS, blocks), U8)


def embed_graph(device: Device, dims: Dims, tokens: int, vocab: int) -> Graph:
    """Q8_0 embedding rows on the device: x_out (H, T) = emb_q[tok] * emb_s[tok] per 32"""
    lg = LayerGraph(device, f"ml4_embed_T{tokens}", dims, tokens)
    H, T = dims.hidden, tokens
    tok = lg.input("tok", (T,), I32)
    e8, es, e32 = (
        lg.tensor((H, T), I8),
        lg.tensor((H // 32, T), F32),
        lg.tensor((H, T), F32),
    )
    lg.node(
        "gather_fwd_i8",
        [lg.input("emb_q", (H, vocab), I8), tok],
        [e8],
        struct.pack("i", 1),
    )
    lg.node(
        "gather_fwd_f32",
        [lg.input("emb_s", (H // 32, vocab), F32), tok],
        [es],
        struct.pack("i", 1),
    )
    lg.node("cast_i8_to_f32", [e8], [e32])
    x = lg.tensor((32, H // 32, T), F32)
    lg.node(
        "mult_fwd_f32",
        [lg.view(e32, (32, H // 32, T)), lg.view(es, (1, H // 32, T))],
        [x],
    )
    lg.node("reshape", [x], [lg.input("x_out", (H, T), F32)])
    return lg.g


def attn_pre_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    kind: AttnKind,
    qtypes: dict[str, str],
    blocks: int,
    mme: bool = False,
) -> Graph:
    """Norm, queries for the slots (``q_all`` (D, nh T) bf16, column h + nh t), the chunk's new
    keys stored into their pages (``kv_rows`` (4, T): latent row, rotary unit, index unit), the
    indexer's queries and head weights when asked, and the accumulator reset.
    ``qtypes``: GEMV kind of wq / wkv / ix_wq / ix_wk."""
    name = f"ml4_attn_pre_{kind.tag}_T{tokens}_NB{blocks}"
    lg = LayerGraph(device, name, dims, tokens, mme)
    attn_pre(lg, kind, qtypes, blocks)
    return lg.g


def attn_pre(
    lg: LayerGraph,
    kind: AttnKind,
    qtypes: dict[str, str],
    blocks: int,
    reset: bool = True,
    merged: bool = False,
) -> Tensor | None:
    """``attn_pre_graph``'s nodes. ``reset``: the accumulator reset for slot merges.
    ``merged``: returns the cache store's sink (None: nothing stored)."""
    d, T = lg.d, lg.T
    H, nh, D = d.hidden, d.heads, kind.key_dim
    nope = d.qk - (d.rope if kind.rope else 0)
    x = lg.residual()
    h = lg.norm(x, "attn_norm")
    q = lg.view(lg.gemv("wq", qtypes["wq"], h, H, nh * d.qk, T), (d.qk, nh * T))
    qr = lg.rope(
        q,
        nh,
        nope if kind.rope else -1,
        lg.input("ss_s", (nh,), F32),
        lg.input("ss_b", (nh,), F32),
        True,
    )  # (qk, T, nh)
    if kind.rope:
        qn, qp = lg.split0(qr, [nope, d.rope])
    else:
        qn, qp = qr, None
    qlat = lg.tensor((d.latent, T, nh), F32)
    lg.node(
        "batch_gemm",
        [lg.cast16(qn), lg.input("kb", (nope, d.latent, nh), BF16)],
        [qlat],
        GEMM_NT,
    )
    qfull = lg.concat([qlat, qp], 0) if qp is not None else qlat  # (D, T, nh)
    lg.node(
        "cast_f32_to_bf16",
        [lg.view(lg.perm(qfull), (D, nh * T))],
        [lg.input("q_all", (D, nh * T), BF16)],
    )
    kn = ix = None
    if kind.owns_kv:
        kv = lg.view(lg.gemv("wkv", qtypes["wkv"], h, H, D, T), (D, T))
        lat = lg.norm(kv, "kv_norm", d.latent)
        if kind.rope:
            _, kr = lg.split0(kv, [d.latent, d.rope])
            kr = lg.rope(kr, 1, 0, *lg.no_ssmax(), False)
            kn = lg.concat([lat, kr], 0)  # f32: mla_store quantizes the latent
        else:
            kn = lat
    if kind.owns_index:
        Di = d.index_dim
        kx = lg.view(lg.gemv("ix_wk", qtypes["ix_wk"], h, H, Di, T), (Di, T))
        kx = lg.norm(kx, "ix_kn", layer_norm=True, bias="ix_knb", eps=d.index_eps)
        if kind.rope:
            kx = lg.rope(kx, 1, 0, *lg.no_ssmax(), False)
        ix = lg.cast16(kx)
        if kind.index_queries:
            NI, Tq = d.index_heads, d.index_tokens(T)
            # this rank's columns of a prefill chunk
            cols = lg.input("ix_cols", (Tq,), I32) if Tq < T else None
            hq = lg.pick(h, cols) if cols is not None else h
            qi_in = lg.norm(
                hq, "ix_qn", layer_norm=True, bias="ix_qnb", eps=d.index_eps
            )
            qi = lg.view(
                lg.gemv("ix_wq", qtypes["ix_wq"], qi_in, H, NI * Di, Tq), (Di, NI * Tq)
            )
            qi = lg.rope(
                qi,
                NI,
                0 if kind.rope else -1,
                lg.input("ix_ss_s", (NI,), F32),
                lg.input("ix_ss_b", (NI,), F32),
                False,
                cols,
            )  # (Di, NI Tq), column h + NI t: token-major, as the slot queries
            lg.node("cast_f32_to_bf16", [qi], [lg.input("ix_q", (Di, NI * Tq), BF16)])
            lg.node(
                "gemm",
                [hq, lg.input("ix_proj", (H, NI), F32)],
                [lg.input("ix_w", (NI, Tq), F32)],
                GEMM_NT,
            )
    sink = None
    if kn is not None or ix is not None:
        sink = lg.tensor((64, 1), I32) if merged else lg.input("sink_kv", (64, 1), I32)
        lg.node(
            "rg_mla_store_f32",
            [
                lg.arena(blocks),
                lg.input("kv_rows", (STORE_FIELDS, T), I32),
                kn if kn is not None else x,  # (unread without a latent row)
                ix if ix is not None else lg.cast16(x),
            ],
            [sink],
        )
    if reset:
        W = nh * T
        acc = [
            lg.input("acc_o", (d.latent, W, 1), F32),
            lg.input("acc_m", (W, 1), F32),
            lg.input("acc_l", (W, 1), F32),
        ]
        lg.node("rg_mla_acc_init_f32", acc, [lg.input("sink_acc", (64, 1), I32)])
    return sink


def attn_slots_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    query_block: int,
    slots: int,
    key_dim: int,
    blocks: int,
    keep_keys: int = 0,
) -> Graph:
    """One launch of ``slots`` MLA slots of ``query_block`` tokens each, merged into the running
    accumulators of the T columns. Per launch: ``slots`` (8, S) i32 [group, qcol0, qpos0, key0,
    klen, qtok0, ...] (group -1: padding), ``lat_pages`` / ``rope_pages`` (C / 128, S) i32 first
    rows (-1: none). ``keep_keys`` > 0: the DSA selection of the step's latest indexer applies
    (``ix_score`` (keep_keys, T) and ``ix_thr`` (64, T), or a prefill chunk's ``ix_mask``
    (keep_keys, T))."""
    name = f"ml4_attn_slots_T{tokens}_Q{query_block}_S{slots}_D{key_dim}_K{keep_keys}_NB{blocks}"
    lg = LayerGraph(device, name, dims, tokens)
    attn_slots(lg, query_block, slots, key_dim, blocks, keep_keys)
    return lg.g


def _values(lg: LayerGraph, ks: Tensor) -> Tensor:
    """P.V's values: the key slab's latent dims (P is in natural key order)."""
    D, latent = ks.shape[0], lg.d.latent
    return ks if D == latent else lg.split0(ks, [latent, D - latent])[0]


def _slot_merge(
    lg: LayerGraph,
    pv: Tensor,
    mx: Tensor,
    ls: Tensor,
    table: Tensor,
    out: Tensor | None,
) -> None:
    """A slot launch's results into the step's attention output: merged into the running
    accumulators, or with ``out`` (latent, nh T, 1) f32 as that output directly (a launch whose
    columns are each in one slot)."""
    if out is not None:
        lg.node("rg_mla_slot_out_f32", [pv, mx, ls, table], [out])
        return
    W = lg.d.heads * lg.T
    acc = [
        lg.input("acc_o", (lg.d.latent, W, 1), F32),
        lg.input("acc_m", (W, 1), F32),
        lg.input("acc_l", (W, 1), F32),
    ]
    guid = "am" if lg.T > lq.MAX_COLUMNS else "acc"  # a prefill chunk: one slot per launch
    lg.node(
        f"rg_mla_slot_{guid}_f32",
        [pv, mx, ls, table, *acc],
        [lg.input("sink_slots", (64, 1), I32)],
    )


def attn_slots(
    lg: LayerGraph,
    query_block: int,
    slots: int,
    key_dim: int,
    blocks: int,
    keep_keys: int = 0,
    out: Tensor | None = None,
    after: Tensor | None = None,
) -> None:
    """``attn_slots_graph``'s nodes (``out``: see ``_slot_merge``; ``after``: see
    ``attn_pre``)"""
    d, T, S, C, D = lg.d, lg.T, slots, SLOT_KEYS, key_dim
    W, M = d.heads * T, d.heads * query_block
    Mp = -(-M // 64) * 64
    arena = lg.arena(blocks)
    table = lg.input("slots", (8, S), I32)
    lp, rp = (
        lg.input("lat_pages", (C // 128, S), I32),
        lg.input("rope_pages", (C // 128, S), I32),
    )
    q = lg.tensor((D, M, S), BF16)
    lg.node("rg_mla_slot_q_f32", [lg.input("q_all", (D, W), BF16), table], [q])
    ks = lg.tensor((D, C, S), BF16)
    ins = [arena, lp, rp, table, after if after is not None else lg.dep()]
    lg.node("rg_mla_gather_f32", ins, [ks], struct.pack("i", 0))
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, S), BF16)
    mx, ls = lg.tensor((Mp, S), F32), lg.tensor((Mp, S), F32)
    prm = struct.pack("fi", d.scale, d.heads)
    if keep_keys and T > lq.MAX_COLUMNS:  # a prefill chunk: the ranks' gathered masks
        mask = lg.input("ix_mask", (keep_keys, T), F32)
        lg.node("rg_mla_slot_pm_f32", [sc, table, mask], [P, mx, ls], prm)
    elif keep_keys:
        score = lg.input("ix_score", (keep_keys, T), F32)
        thr = lg.input("ix_thr", (64, T), I32)
        lg.node("rg_mla_slot_pk_f32", [sc, table, score, thr], [P, mx, ls], prm)
    else:
        lg.node("rg_mla_slot_p_f32", [sc, table], [P, mx, ls], prm)
    pv = lg.tensor((d.latent, M, S), F32)
    lg.node("batch_gemm", [P, _values(lg, ks)], [pv], GEMM_NN)
    _slot_merge(lg, pv, mx, ls, table, out)


def index_slots_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    query_block: int,
    slots: int,
    keys: int,
    blocks: int,
) -> Graph:
    """One launch of DSA indexer slots: the block's index queries (``ix_q`` (I, NI T) bf16,
    token-major) against a 2048-key chunk of index keys (``ix_pages`` (C / 128, S)), weighted
    relu scores into the step's scratch ``ix_score`` (keys, T)."""
    d, T, S, C = dims, tokens, slots, SLOT_KEYS
    Di, NI = d.index_dim, d.index_heads
    name = f"ml4_index_slots_T{T}_Q{query_block}_S{S}_K{keys}_NB{blocks}"
    lg = LayerGraph(device, name, dims, T)
    M = NI * query_block
    table = lg.input("slots", (8, S), I32)
    pages = lg.input("ix_pages", (C // 128, S), I32)
    q = lg.tensor((Di, M, S), BF16)
    lg.node("rg_mla_slot_q_f32", [lg.input("ix_q", (Di, NI * T), BF16), table], [q])
    ks = lg.tensor((Di, C, S), BF16)
    lg.node(
        "rg_mla_gather_f32",
        [lg.arena(blocks), pages, pages, table, lg.dep()],
        [ks],
        struct.pack("i", 2),
    )
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    lg.node(
        "rg_ix_slot_acc_f32",
        [
            sc,
            lg.input("ix_w", (NI, T), F32),
            table,
            lg.input("ix_score", (keys, T), F32),
        ],
        [lg.input("sink_ix", (64, 1), I32)],
        struct.pack("f", Di**-0.5),
    )
    return lg.g


def index_select_graph(
    device: Device, dims: Dims, tokens: int, keys: int, top: int
) -> Graph:
    """Per column, the top-k threshold over its visible keys (``vis`` (T,) i32) of the scratch"""
    T = tokens
    lg = LayerGraph(device, f"ml4_index_select_T{T}_K{keys}_top{top}", dims, T)
    lg.node(
        "rg_ix_thr_f32",
        [lg.input("ix_score", (keys, T), F32), lg.input("vis", (T,), I32)],
        [lg.input("ix_thr", (64, T), I32)],
        struct.pack("i", top),
    )
    return lg.g


def index_mask_graph(
    device: Device, dims: Dims, tokens: int, keys: int, top: int
) -> Graph:
    """A prefill chunk's DSA selection for this rank's columns: the threshold, then its bits
    (``ix_bits`` (keys / 32, T), ``rg_ix_mask``), which the ranks all-gather"""
    T = tokens
    lg = LayerGraph(device, f"ml4_index_mask_T{T}_K{keys}_top{top}", dims, T)
    score, vis = lg.input("ix_score", (keys, T), F32), lg.input("vis", (T,), I32)
    thr = lg.input("ix_thr", (64, T), I32)
    lg.node("rg_ix_thr_f32", [score, vis], [thr], struct.pack("i", top))
    lg.node("rg_ix_mask_f32", [score, vis, thr], [lg.input("ix_bits", (keys // 32, T), I32)])
    return lg.g


def index_unmask_graph(device: Device, dims: Dims, tokens: int, keys: int) -> Graph:
    """Every rank's gathered DSA bits as ``mla_slot_pm``'s mask ``ix_mask`` (keys, T)"""
    T = tokens
    lg = LayerGraph(device, f"ml4_index_unmask_T{T}_K{keys}", dims, T)
    lg.node(
        "rg_ix_unmask_f32",
        [lg.input("ix_bits", (keys // 32, T), I32), lg.input("vis", (T,), I32)],
        [lg.input("ix_mask", (keys, T), F32)],
    )
    return lg.g


def index_list_graph(
    device: Device, dims: Dims, tokens: int, keys: int, top: int
) -> Graph:
    """A decode step's DSA selection as key lists: the threshold (``index_select_graph``), the
    per-block kept counts, then per column the kept positions ``ix_list`` (top, T) and the
    sparse attention slot record ``ix_slots`` (8, T) (``rg_ix_list``)."""
    lg = LayerGraph(device, f"ml4_index_list_T{tokens}_K{keys}_top{top}", dims, tokens)
    index_list(lg, keys, top)
    return lg.g


def index_list(lg: LayerGraph, keys: int, top: int) -> None:
    """``index_list_graph``'s nodes"""
    T, dims = lg.T, lg.d
    score, vis = lg.input("ix_score", (keys, T), F32), lg.input("vis", (T,), I32)
    thr = lg.input("ix_thr", (64, T), I32)
    lg.node("rg_ix_thr_f32", [score, vis], [thr], struct.pack("i", top))
    cnt, gsum = lg.tensor((keys // 64, T), I32), lg.tensor((keys // 4096, T), I32)
    lg.node("rg_ix_bcnt_f32", [score, vis, thr], [cnt, gsum])
    members = max(1, 48 // T)
    lg.node(
        "rg_ix_list_f32",
        [score, vis, thr, cnt, gsum],
        [lg.input("ix_list", (top, T), I32), lg.input("ix_slots", (8, T), I32)],
        struct.pack("iii", members, top, dims.heads),
    )


def attn_sparse_graph(
    device: Device, dims: Dims, tokens: int, key_dim: int, blocks: int, pages: int
) -> Graph:
    """Sparse decode attention: one slot per column over its DSA-listed keys (``ix_list``,
    ``ix_slots`` from the latest indexer owner's ``index_list_graph``), gathered through the
    step's block table ``kv_blocks`` (pages, T) and this layer's page offsets ``loff``, merged
    into the column's accumulators. One launch per layer at any context."""
    name = f"ml4_attn_sparse_T{tokens}_D{key_dim}_NB{blocks}_P{pages}"
    lg = LayerGraph(device, name, dims, tokens)
    attn_sparse(lg, key_dim, blocks, pages)
    return lg.g


def attn_sparse(
    lg: LayerGraph,
    key_dim: int,
    blocks: int,
    pages: int,
    out: Tensor | None = None,
    after: Tensor | None = None,
) -> None:
    """``attn_sparse_graph``'s nodes (``out``: see ``_slot_merge``; ``after``: see
    ``attn_pre``)"""
    d, T, C, D = lg.d, lg.T, SLOT_KEYS, key_dim
    W, M = d.heads * T, d.heads
    Mp = -(-M // 64) * 64
    table = lg.input("ix_slots", (8, T), I32)
    q = lg.tensor((D, M, T), BF16)
    lg.node("rg_mla_slot_q_f32", [lg.input("q_all", (D, W), BF16), table], [q])
    ins = [
        lg.arena(blocks),
        lg.input("ix_list", (C, T), I32),
        lg.input("kv_blocks", (pages, T), I32),
        lg.input("loff", (64,), I32),
        table,
        after if after is not None else lg.dep(),
    ]
    ks = lg.tensor((D, C, T), BF16)
    lg.node("rg_mla_gather_list_f32", ins, [ks])
    sc = lg.tensor((C, M, T), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, T), BF16)
    mx, ls = lg.tensor((Mp, T), F32), lg.tensor((Mp, T), F32)
    prm = struct.pack("fi", d.scale, d.heads)
    lg.node("rg_mla_slot_p_f32", [sc, table], [P, mx, ls], prm)
    pv = lg.tensor((d.latent, M, T), F32)
    lg.node("batch_gemm", [P, _values(lg, ks)], [pv], GEMM_NN)
    _slot_merge(lg, pv, mx, ls, table, out)


def attn_post_graph(
    device: Device, dims: Dims, tokens: int, wo_qtype: str, mme: bool = False
) -> Graph:
    """v_b per head on the accumulated latent outputs, then wo -> this rank's partial p_out"""
    lg = LayerGraph(device, f"ml4_attn_post_T{tokens}_{wo_qtype}", dims, tokens, mme)
    attn_post(lg, wo_qtype)
    return lg.g


def attn_post(lg: LayerGraph, wo_qtype: str, acc: Tensor | None = None) -> None:
    """``attn_post_graph``'s nodes, over ``acc`` (latent, nh T, 1) or the accumulators"""
    d, T = lg.d, lg.T
    nh = d.heads
    if acc is None:
        acc = lg.input("acc_o", (d.latent, nh * T, 1), F32)
    olat = lg.perm(lg.view(acc, (d.latent, nh, T)))  # (512, T, nh)
    ov = lg.tensor((d.v, T, nh), F32)
    lg.node(
        "batch_gemm",
        [lg.cast16(olat), lg.input("vb", (d.latent, d.v, nh), BF16)],
        [ov],
        GEMM_NT,
    )
    ot = lg.perm(ov)  # (v, nh, T): token-major columns
    lg.gemv(
        "wo",
        wo_qtype,
        ot,
        nh * d.v,
        d.hidden,
        T,
        out=lg.input("p_out", (d.hidden * T,), F32),
    )


def attn_block_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    kind: AttnKind,
    qtypes: dict[str, str],
    wo_qtype: str,
    blocks: int,
    slots: int,
    pages: int = 0,
    mme: bool = False,
) -> Graph:
    """A decode step's attention half of a layer in one launch: ``attn_pre``, then either one
    launch of ``slots`` dense slots (``pages`` 0: every column sees at most SLOT_KEYS keys, so
    one slot per row covers the step) or sparse attention over the latest indexer owner's key
    lists (``pages``: block-table rows), then ``attn_post``."""
    path = f"sparse_P{pages}" if pages else f"dense_S{slots}"
    name = f"ml4_attn_block_{kind.tag}_T{tokens}_{path}_{wo_qtype}_NB{blocks}"
    lg = LayerGraph(device, name, dims, tokens, mme)
    sink = attn_pre(lg, kind, qtypes, blocks, reset=False, merged=True)
    out = lg.tensor((dims.latent, dims.heads * tokens, 1), F32)
    if pages:
        attn_sparse(lg, kind.key_dim, blocks, pages, out, sink)
    else:
        attn_slots(lg, 1, slots, kind.key_dim, blocks, out=out, after=sink)
    attn_post(lg, wo_qtype, out)
    return lg.g


def attn_tail_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    key_dim: int,
    wo_qtype: str,
    blocks: int,
    pages: int,
    keys: int,
    top: int,
    mme: bool = False,
) -> Graph:
    """An indexer owner's attention after its index slot launches on a sparse decode step, in
    one launch: the key lists (``index_list``), sparse attention over them, ``attn_post``. Its
    ``attn_pre`` (with the index queries) runs before the index slots."""
    name = f"ml4_attn_tail_T{tokens}_D{key_dim}_{wo_qtype}_NB{blocks}_P{pages}_K{keys}"
    lg = LayerGraph(device, name, dims, tokens, mme)
    index_list(lg, keys, top)
    out = lg.tensor((dims.latent, dims.heads * tokens, 1), F32)
    attn_sparse(lg, key_dim, blocks, pages, out)
    attn_post(lg, wo_qtype, out)
    return lg.g


def ffn_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    moe: bool,
    qtypes: dict[str, str],
    mme: bool = False,
    paged: bool = False,
) -> Graph:
    """x_out = x + p, then this rank's FFN partial into p_out: a dense SwiGLU, or the top-k
    routed experts' slices plus the shared expert's. ``qtypes``: GEMV kinds of du / dd or
    xu / xd / shu / shd. mme: dense weights on the MME; paged: the routed experts on the MME
    (``paged_experts``)."""
    d, T = dims, tokens
    tag = "_".join(f"{k}{v}" for k, v in sorted(qtypes.items()))
    name = f"ml4_ffn_{'moe' if moe else 'dense'}_T{T}_{tag}" + (
        "_paged" if paged else ""
    )
    lg = LayerGraph(device, name, dims, T, mme)
    H = d.hidden
    x = lg.residual()
    h2 = lg.norm(x, "ffn_norm")
    if not moe:
        F = d.dense
        gu = lg.view(lg.gemv("du", qtypes["du"], h2, H, 2 * F, T), (2 * F, T))
        lg.gemv(
            "dd",
            qtypes["dd"],
            lg.swiglu(gu),
            F,
            H,
            T,
            out=lg.input("p_out", (H * T,), F32),
        )
        return lg.g
    E, K = d.experts, d.top
    logits = lg.tensor((E, T), F32)
    lg.node("gemm", [h2, lg.input("router", (H, E), F32)], [logits], GEMM_NT)
    padded = lg.concat([logits, lg.input("negpad", (ROUTER_LANES - E, T), F32)], 0)
    ids, wts = lg.tensor((64, T), I32), lg.tensor((64, T), F32)
    lg.node(
        "rg_moe_route_f32",
        [padded, lg.input("meta", (4,), I32)],
        [ids, wts],
        struct.pack("i", K),
    )
    idf = lg.view(ids, (64 * T,))
    F, S = d.expert, K * T
    if paged:
        yd = paged_experts(lg, h2, idf, qtypes)
    else:
        yu = lg.view(
            lg.gemv(
                "xu",
                qtypes["xu"],
                h2,
                H,
                2 * F,
                S,
                ids=idf,
                nu=K,
                bcast=True,
                experts=E,
            ),
            (2 * F, S),
        )
        yd = lg.view(
            lg.gemv(
                "xd", qtypes["xd"], lg.swiglu(yu), F, H, S, ids=idf, nu=K, experts=E
            ),
            (H, S),
        )
    Fs = d.shared
    sgu = lg.view(lg.gemv("shu", qtypes["shu"], h2, H, 2 * Fs, T), (2 * Fs, T))
    sd = lg.view(lg.gemv("shd", qtypes["shd"], lg.swiglu(sgu), Fs, H, T), (H, T))
    lg.node(
        "rg_ml4_moe_sum_f32",
        [yd, wts, lg.input("zeroH", (H, T), F32), sd],
        [lg.input("p_out", (H, T), F32)],
        struct.pack("i", K),
    )
    return lg.g


def paged_shape(dims: Dims, tokens: int) -> tuple[int, int, int, int]:
    """(psh, P, Ep, tc) of the paged experts of T tokens (pages of at least twice the mean rows
    per expert, at most 128; one partial page per expert)"""
    pairs, E = dims.top * tokens, dims.experts
    psh = 6
    while psh < 7 and (1 << psh) < 2 * pairs // E:
        psh += 1
    P = -(-pairs // (1 << psh)) + min(E, pairs)
    tc = 32
    while -(-tokens // tc) > 64:
        tc *= 2
    return psh, P, -(-E // 64) * 64, tc


def row_chunks(rows: int, cols: int, P: int) -> list[tuple[int, int]]:
    """(first row, rows) of an expert matrix's row chunks for its dequantized (rows, cols, P)
    bf16 operand (whole 128-row blocks)"""
    fit = ((1 << 31) - 1) // (2 * cols * P) // 128 * 128
    assert fit >= 128, (rows, cols, P)
    step = min(rows, fit)
    return [(r, min(step, rows - r)) for r in range(0, rows, step)]


def paged_experts(
    lg: LayerGraph, h2: Tensor, ids: Tensor, qtypes: dict[str, str]
) -> Tensor:
    """The routed experts of T tokens: the (token, slot) pairs grouped by expert into
    pages of PS rows, each page's expert slice dequantized to bf16 into the page's weight slot
    (``lq_qdqp``), one batch_gemm over the pages for gate, up and down, then the rows scattered
    back -> (H, K T) f32, column t K + s."""
    d, T = lg.d, lg.T
    H, E, K, F = d.hidden, d.experts, d.top, d.expert
    pairs = K * T
    psh, P, Ep, tc = paged_shape(d, T)
    PS = 1 << psh
    NC = -(-T // tc)
    dep, t = lg.dep(), lg.tensor
    pk = lambda *v: struct.pack("i" * len(v), *v)  # noqa: E731
    cnt, tot, pgb = t((NC * Ep,), I32), t((Ep,), I32), t((Ep,), I32)
    pe, base, cur = t((P,), I32), t((NC * Ep,), I32), t((NC * Ep,), I32)
    pairrow = t((-(-pairs // 32) * 32,), I32)
    lg.node("rg_lq_moe_cnt_f32", [ids, dep], [cnt], pk(NC, K, T, E, Ep, tc, 0, 1, 64))
    lg.node("rg_lq_moe_tot_f32", [cnt, dep], [tot], pk(Ep // 64, Ep, NC))
    lg.node("rg_lq_moe_plan_f32", [tot, dep], [pe, pgb], pk(1, E, psh, P))
    lg.node("rg_lq_moe_base_f32", [cnt, pgb, dep], [base], pk(Ep // 64, Ep, NC))
    lg.node(
        "rg_lq_moe_asg_f32",
        [ids, base, dep],
        [pairrow, cur],
        pk(NC, K, T, E, Ep, tc, 0, 1, 64),
    )
    xp = t((H, PS, P), BF16)
    lg.node(
        "rg_lq_moe_gx_f32",
        [lg.flat(h2), pairrow, dep],
        [xp],
        pk(nmem_for(pairs * -(-H // 128), 8), H, K, T, psh, 2, 0, H, H),
    )

    def expert_gemm(
        name: str, qtype: str, x: Tensor, first: int, rows: int, cols: int, total: int
    ):
        """y (rows, PS, P) = rows first .. first + rows of each page's expert (matrices of
        ``total`` rows) times the page's inputs x (cols, PS, P)"""
        guid = f"rg_lq_qdqp_{QMV8[qtype][0]}_f32"
        parts = []
        W = lg.planar(name, qtype, total, cols, E)
        for r0, n in row_chunks(rows, cols, P):
            w = t((n, cols, P), BF16)
            lg.node(
                guid,
                [W, pe, dep],
                [w],
                pk(0, cols, n, LAYOUT[qtype][0], (first + r0) // 128),
            )
            y = t((n, PS, P), F32)
            lg.node("batch_gemm", [x, w], [y], GEMM_NN)
            parts.append(y)
        return parts[0] if len(parts) == 1 else lg.concat(parts, 0)

    yg = expert_gemm("xu", qtypes["xu"], xp, 0, F, H, 2 * F)
    yu = expert_gemm("xu", qtypes["xu"], xp, F, F, H, 2 * F)
    xp2 = t((F, PS, P), BF16)
    lg.node(
        "rg_lq_moe_glu_f32",
        [lg.flat(yg), lg.flat(yu), pe, dep],
        [xp2],
        pk(nmem_for(P << psh, 4), F, psh, P, 0, F, 0, F),
    )
    y2 = expert_gemm("xd", qtypes["xd"], xp2, 0, H, F, H)
    yd = t((H * K * T,), F32)  # the kernel stores at flat offsets (t K + s) H + k
    lg.node(
        "rg_lq_moe_scatter_f32",
        [y2, pairrow, dep],
        [yd],
        pk(nmem_for(pairs * -(-H // 64), 8), H, K, T, psh),
    )
    return lg.view(yd, (H, K * T))


def head_graph(
    device: Device, dims: Dims, tokens: int, qtype: str, mme: bool = False
) -> Graph:
    """x + p, the final norm and this rank's vocabulary slice of the head -> logits (V_r, T)
    (raw: the softcap is applied by the consumer)"""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ml4_head_T{T}_{qtype}", dims, T, mme)
    x = lg.residual()
    hn = lg.norm(x, "out_norm")
    lg.gemv(
        "head",
        qtype,
        hn,
        d.hidden,
        d.vocab,
        T,
        out=lg.input("logits", (d.vocab * T,), F32),
    )
    return lg.g


def head_columns_graph(
    device: Device, dims: Dims, tokens: int, columns: int, qtype: str, mme: bool
) -> Graph:
    """The head of a prefill chunk at ``columns`` of its T columns (``cols`` (columns,) i32):
    the sampled column, or the positions prompt logprobs score -> logits (V_r, columns)."""
    d, T, C = dims, tokens, columns
    lg = LayerGraph(device, f"ml4_head_T{T}_C{C}_{qtype}", dims, C, mme)
    cols = lg.input("cols", (C,), I32)
    picked = []
    for name in ("x", "p"):
        out = lg.tensor((d.hidden, C), F32)
        lg.node(
            "gather_fwd_f32",
            [lg.input(name, (d.hidden, T), F32), cols],
            [out],
            struct.pack("i", 1),
        )
        picked.append(out)
    hn = lg.norm(lg.add(*picked), "out_norm")
    lg.gemv(
        "head",
        qtype,
        hn,
        d.hidden,
        d.vocab,
        C,
        out=lg.input("logits", (d.vocab * C,), F32),
    )
    return lg.g


def sample_graph(
    device: Device, dims: Dims, tokens: int, rows: int, nranks: int
) -> Graph:
    """The device sampler (``qwen3_5.sampling``) over the softcapped logits
    ``cap * tanh(z / cap)`` of every column (rows = T) or of column ``last`` (rows = 1).
    ``logits_all`` is the ranks' all-gathered vocabulary slices, (V_r, T, nranks)."""
    d, T, R = dims, tokens, rows
    lg = LayerGraph(device, f"ml4_sample_T{T}_R{R}_n{nranks}", dims, T)
    V = d.vocab * nranks
    if nranks > 1:
        full = lg.view(
            lg.perm(lg.input("logits_all", (d.vocab, T, nranks), F32)), (V, T)
        )
    else:
        full = lg.input("logits_all", (V, T), F32)
    if R < T:
        picked = lg.tensor((V, 1), F32)
        lg.node(
            "gather_fwd_f32",
            [full, lg.input("last", (1,), I32)],
            [picked],
            struct.pack("i", 1),
        )
        full = picked
    z = lg.tensor((V, R), F32)
    lg.node("mult_fwd_f32", [full, lg.input("inv_cap", (1, 1), F32)], [z])
    th = lg.tensor((V, R), F32)
    lg.node("tanh_fwd_f32", [z], [th])
    capped = lg.tensor((V, R), F32)
    lg.node("mult_fwd_f32", [th, lg.input("cap", (1, 1), F32)], [capped])
    sampler(lg.g, capped, lg.input("samp_params", (PARAMS, R), F32))
    return lg.g
