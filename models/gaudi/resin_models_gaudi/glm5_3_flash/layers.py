"""Per-layer Synapse graphs of GLM-5.3-Flash, with FP8 weights. Each graph is one layer kind reused for every layer of its kind: weights are persistent
inputs bound per launch under generic names.

The residual is 4 mHC streams, which cross launches as ``X`` (f32 (4H, T)) with the previous sub-layer's
coefficients ``c`` (T, 24) and its output partial ``p``, summed over the ranks between launches: every graph
starts ``X_out = post y + comb^T X`` (``mhc_res``), computes its site's coefficients ``c_out`` and collapse
(``mhc_mix``, ``mhc_coef``, ``mhc_pre``) and ends with this rank's partial in ``p_out``. A step is

    embed, then per layer: attention (kda, or mla_pre, slots x n, mla_post), ffn; then head.

MTP (``mtp_*``, a speculative step's drafts): eh_proj [enorm(embedding of the next token) ; hnorm(a normed hidden)],
then one plain residual layer (MLA + indexer over its own cache, MoE), ``shared_head.norm`` and the shared head.
Its residual is one stream ``x``; the collapse runs on 4 equal streams with the head mean (mode 1), which is x.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi.glm5_3_flash.cache import ARENA_BLOCK_ROWS, ROW_BYTES
from resin_models_gaudi.glm5_3_flash.weights import NS, RankSlices
from resin_models_gaudi.qwen3_5.sampling import PARAMS, sampler
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, U8, Device

GEMM_NT = struct.pack("??", False, True)  # y(M, N) = x(K, N) . w(K, M)^T
GEMM_NN = struct.pack("??", False, False)  # y(M, N) = x(K, N) . w(M, K)
SLOT_KEYS = 2048
SLOT_POOLS = 2048  # pools per indexer slot: 64 index pages
PAGE_POOLS = 32
LIST_KEYS = 2112  # a sparse slot: 512 pools of 4 keys and the tail, which an index never cuts
KDA_CHUNK = 64
ROWS_FIELDS = 8
LONG_T = 64  # launches wider than this: prefill chunks


@dataclass(frozen=True, slots=True)
class Dims:
    """One rank's model dimensions."""

    hidden: int
    kda_heads: int
    small: int
    """rows of [f_a | g_a | b | 0]: what the KDA input GEMM adds to [q | k | v]"""
    mla_heads: int
    qk: int
    latent: int
    q_lora: int
    dense: int
    expert: int
    experts: int
    top: int
    vocab: int
    """this rank's vocabulary slice"""
    index_heads: int
    index_dim: int
    top_pools: int
    eps: float
    hc_iters: int
    hc_eps: float
    kda_lb: float
    limit: float
    route_scale: float
    conv_rows: int
    ranks: int = 1
    gather_tokens: int = 0

    @classmethod
    def of(cls, cfg: Glm53Config, sl: RankSlices, conv_rows: int) -> Dims:
        width = lambda s: s.stop - s.start  # noqa: E731
        nk = width(sl.kda_heads)
        hd = cfg.linear_head_dim
        return cls(
            hidden=cfg.hidden_size,
            kda_heads=nk,
            small=2 * hd + nk + (-(2 * hd + nk) % 16),
            mla_heads=width(sl.mla_heads),
            qk=cfg.qk_nope_head_dim,
            latent=cfg.kv_lora_rank,
            q_lora=cfg.q_lora_rank,
            dense=width(sl.dense),
            expert=width(sl.expert),
            experts=cfg.n_routed_experts,
            top=cfg.num_experts_per_tok,
            vocab=width(sl.vocab),
            index_heads=cfg.index_n_heads,
            index_dim=cfg.index_head_dim,
            top_pools=cfg.index_topk // cfg.index_kpool,
            eps=cfg.rms_norm_eps,
            hc_iters=cfg.hc_sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            kda_lb=cfg.linear_lower_bound,
            limit=cfg.swiglu_limit,
            route_scale=cfg.routed_scaling_factor,
            conv_rows=conv_rows,
        )

    @property
    def C(self) -> int:
        return self.kda_heads * 128

    @property
    def scale(self) -> float:
        return self.qk**-0.5

    def gathered(self, tokens: int) -> bool:
        return self.ranks > 1 and tokens <= self.gather_tokens


class LayerGraph:
    """A Graph with the decoder's building blocks."""

    def __init__(self, device: Device, name: str, dims: Dims, tokens: int) -> None:
        gathered = dims.gathered(tokens)
        self.g = Graph(device, f"glm_{name}" + (f"_g{dims.ranks}" if gathered else ""))
        self.d = dims
        self.T = tokens
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

    def view(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        y = self.tensor(shape, x.dtype)
        self.node("reshape", [x], [y])
        return y

    def cast16(self, x: Tensor) -> Tensor:
        y = self.tensor(x.shape, BF16)
        self.node("cast_f32_to_bf16", [x], [y])
        return y

    def split0(self, x: Tensor, sizes: list[int]) -> list[Tensor]:
        outs = [self.tensor((s, *x.shape[1:]), x.dtype) for s in sizes]
        self.node("split", [x], outs, struct.pack("I", 0))
        return outs

    def dep(self) -> Tensor:
        return self.input("dep", (64,), F32)

    def meta(self) -> Tensor:
        return self.input("meta", (4,), I32)

    def arena8(self, blocks: int) -> Tensor:
        """the cache arena as bytes (FP8 latent rows, bf16 index units, f32 scales and tails)"""
        return self.input("arena", (ROW_BYTES, ARENA_BLOCK_ROWS, blocks), U8)

    def arena32(self, blocks: int) -> Tensor:
        """the cache arena as f32 rows (recurrent and conv state)"""
        return self.input("arena", (ROW_BYTES // 4, ARENA_BLOCK_ROWS, blocks), F32)

    def gemm(self, x: Tensor, name: str, M: int, *, out: Tensor | None = None) -> Tensor:
        """bf16 x bf16 -> f32"""
        y = out if out is not None else self.tensor((M, x.shape[1]), F32)
        w = self.input(name, (x.shape[0], M), BF16)
        self.node("gemm", [x, w], [y], GEMM_NT)
        return y

    def gemm8(self, q: Tensor, name: str, M: int) -> Tensor:
        """FP8 x FP8 -> f32"""
        y = self.tensor((M, q.shape[1]), F32)
        w = self.input(name, (q.shape[0], M), F8)
        s = self.input(name + ".s", (M,), F32)
        self.node("fp8_gemm_f32", [q, w, self.input("one", (1,), F32), s], [y], GEMM_NT)
        return y

    def q8(self, x: Tensor, sin: Tensor, *, swiglu: bool = False) -> tuple[Tensor, Tensor]:
        """Per-token FP8 activation and its scale of x * sin, or of the clamped
        SwiGLU of x's halves"""
        K = x.shape[0] // 2 if swiglu else x.shape[0]
        T = x.shape[1]
        q, s = self.tensor((-(-K // 256) * 256, T), F8), self.tensor((1, T), F32)
        mode = 1 | (4 if swiglu else 0) | (8 if T >= 24 else 0)
        self.node(
            "rg_ds_q8_f32", [x, sin, x], [q, s], struct.pack("iff", mode, self.d.eps, self.d.limit)
        )
        return q, s

    def tscale(self, y: Tensor, s: Tensor, out: Tensor | None = None) -> Tensor:
        o = out if out is not None else self.tensor(y.shape, F32)
        self.node("rg_ds_tscale_f32", [y, s], [o])
        return o

    def residual(self) -> Tensor:
        """``X_out``: the sub-layer before this graph applied to the streams, by ``mhc_post`` in a long launch,
        else ``mhc_res``."""
        H, T = self.d.hidden, self.T
        ranks = self.d.ranks if self.d.gathered(T) else 1
        y = self.input("p", (H, T, ranks), F32)
        X = self.input("X", (NS * H, T), F32)
        coef = self.input("c", (T, 24), F32)
        Xo = self.input("X_out", (NS * H, T), F32)
        self._rt = None
        if T > LONG_T:
            self._rt = self.tensor((64, T), F32)
            self.node("rg_mhc_post_f32", [X, y, coef], [Xo, self._rt], struct.pack("fi", self.d.eps, 0))
        else:
            self.node("rg_mhc_res_f32", [X, y, coef], [Xo])
        return Xo

    def site(self, X: Tensor, tag: str) -> Tensor:
        """this sub-layer's mHC coefficients ``c_out`` (T, 24)"""
        H, T = self.d.hidden, self.T
        mix = self.tensor((T, 32), F32)
        fn = self.input(tag + "_fn", (NS * H, 32), F32)
        if self._rt is not None:
            rt = self._rt
            self.node("gemm", [fn, X], [mix], GEMM_NT)
        else:
            rt = self.tensor((64, T), F32)
            self.node("rg_mhc_mix_f32", [fn, X], [mix, rt], struct.pack("f", self.d.eps))
        coef = self.input("c_out", (T, 24), F32)
        self.node(
            "rg_mhc_coef_f32",
            [mix, rt, self.input(tag + "_base", (24,), F32), self.input(tag + "_scale", (3,), F32)],
            [coef],
            struct.pack("if", self.d.hc_iters, self.d.hc_eps),
        )
        return coef

    def collapse(self, X: Tensor, coef: Tensor, ln: str, q8: int = 0, mode: int = 0):
        """mHC collapse + RMSNorm -> h16 (sigma), h32 [, h8 (tau), hs [, per-slot h8 in pair order]]"""
        H, T = self.d.hidden, self.T
        w = self.input(ln, (H,), F32)
        h16, h32 = self.tensor((H, T), BF16), self.tensor((H, T), F32)
        if not q8:
            self.node(
                "rg_mhc_pre_f32", [X, coef, w], [h16, h32], struct.pack("fi", self.d.eps, mode)
            )
            return h16, h32
        h8, hs = self.tensor((H, T), F8), self.tensor((1, T), F32)
        if q8 == 1:
            self.node(
                "rg_mhc_pre8_f32",
                [X, coef, w],
                [h16, h32, h8, hs],
                struct.pack("fi", self.d.eps, mode),
            )
            return h16, h32, h8, hs
        K = self.d.top
        xq = self.tensor((H, 1, K * T), F8)
        self.node(
            "rg_mhc_pre8x_f32",
            [X, coef, w],
            [h16, h32, h8, hs, xq],
            struct.pack("fii", self.d.eps, mode, K),
        )
        return h16, h32, h8, hs, xq

    def p_out(self) -> Tensor:
        return self.input("p_out", (self.d.hidden, self.T), F32)


def embed_graph(device: Device, dims: Dims, tokens: int, vocab: int) -> Graph:
    """The embedding as 4 identical streams"""
    lg = LayerGraph(device, f"embed_T{tokens}", dims, tokens)
    H, T = dims.hidden, tokens
    e = lg.tensor((H, T), BF16)
    lg.node(
        "gather_fwd_bf16",
        [lg.input("emb", (H, vocab), BF16), lg.input("tok", (T,), I32)],
        [e],
        struct.pack("i", 1),
    )
    e32 = lg.tensor((H, T), F32)
    lg.node("cast_bf16_to_f32", [e], [e32])
    rt = lg.tensor((64, T), F32)
    lg.node(
        "rg_mhc_post_f32",
        [lg.input("X", (NS * H, T), F32), lg.view(e32, (H, T, 1)), lg.input("c", (T, 24), F32)],
        [lg.input("X_out", (NS * H, T), F32), rt],
        struct.pack("fi", dims.eps, 1),
    )
    return lg.g


# ------------------------------------------------------------------ KDA
def kda_common(lg: LayerGraph):
    """Residual, site, collapse and the KDA inputs from one GEMM"""
    d = lg.d
    C = d.C
    X = lg.residual()
    coef = lg.site(X, "hca")
    h16, _ = lg.collapse(X, coef, "ln_a")
    qkv, small = lg.split0(lg.gemm(h16, "w_in", 3 * C + d.small), [3 * C, d.small])
    w_fg = lg.input("w_fg", (d.small, 2 * C), F32)
    fgt = lg.tensor((2 * C, lg.T), F32)
    lg.node("gemm", [small, w_fg], [fgt], GEMM_NT)
    fg, gate = lg.split0(fgt, [C, C])
    par = lg.tensor((C + d.kda_heads,), F32)
    lg.node(
        "concat",
        [lg.input("dt_bias", (C,), F32), lg.input("A_log", (d.kda_heads,), F32)],
        [par],
        struct.pack("I", 0),
    )
    return qkv, small, fg, gate, par


def kda_decode_graph(device: Device, dims: Dims, tokens: int, blocks: int) -> Graph:
    """A KDA layer for T = B sequences, one token each: the recurrent state stays in the arena
    (``kda_decode``); ``kda_bases`` (4, B) [conv_src, ssm_src, -, ssm_dst] and ``conv_dst`` (B,)."""
    lg = LayerGraph(device, f"kda_dec_T{tokens}_NB{blocks}", dims, tokens)
    d, T = dims, tokens
    qkv, small, fg, gate, par = kda_common(lg)
    arena = lg.arena32(blocks)
    out = lg.tensor((d.C, T), BF16)
    cst = lg.tensor((128, d.conv_rows, T), F32)
    lg.node(
        "rg_kda_decode_f32",
        [
            qkv,
            fg,
            small,
            lg.input("conv_w", (3 * d.C, 4), F32),
            arena,
            lg.input("kda_bases", (4, T), I32),
            par,
            gate,
            lg.input("o_norm", (128,), F32),
        ],
        [out, cst],
        struct.pack("ff", d.kda_lb, d.eps),
    )
    lg.node(
        "rg_state_store_f32",
        [arena, lg.input("conv_dst", (T,), I32), cst],
        [lg.input("sink_conv", (64, 1), I32)],
    )
    lg.gemm(out, "o", d.hidden, out=lg.p_out())
    return lg.g


def kda_prefill_graph(device: Device, dims: Dims, tokens: int, blocks: int) -> Graph:
    """A KDA layer for one sequence's T tokens (a multiple of 64) through the chunked rule on dense
    state copies: ``conv_src``, ``ssm_src``, ``conv_dst``, ``ssm_dst`` (1,)."""
    lg = LayerGraph(device, f"kda_pre_T{tokens}_NB{blocks}", dims, tokens)
    d, T = dims, tokens
    assert T % KDA_CHUNK == 0
    C, nh = d.C, d.kda_heads
    qkv, small, fg, gate, par = kda_common(lg)
    arena = lg.arena32(blocks)

    def load(name: str, rows: int) -> Tensor:
        out = lg.tensor((128, rows, 1), F32)
        lg.node("rg_state_load_f32", [arena, lg.input(name, (1,), I32)], [out])
        return out

    def store(name: str, vals: Tensor, sink: str) -> None:
        lg.node(
            "rg_state_store_f32",
            [arena, lg.input(name, (1,), I32), vals],
            [lg.input(sink, (64, 1), I32)],
        )

    conv_real = 9 * nh  # 3 C * 3 / 128 rows
    pad = d.conv_rows - conv_real
    cst_full = load("conv_src", d.conv_rows)
    if pad:
        kept = lg.tensor((128, conv_real, 1), F32)
        rest = lg.tensor((128, pad, 1), F32)
        lg.node("split", [cst_full], [kept, rest], struct.pack("I", 1))
    else:
        kept = cst_full
    cst_in = lg.view(kept, (3 * C, 3))
    sst_in = lg.view(load("ssm_src", 128 * nh), (128, 128, nh, 1))
    cst_out, sst_out = lg.tensor((3 * C, 3), F32), lg.tensor((128, 128, nh, 1), F32)
    out = lg.tensor((C, T), BF16)
    kda_blocks(lg, qkv, fg, small, gate, par, cst_in, sst_in, out, cst_out, sst_out)
    vals = lg.view(cst_out, (128, conv_real, 1))
    if pad:
        full = lg.tensor((128, d.conv_rows, 1), F32)
        lg.node("concat", [vals, rest], [full], struct.pack("I", 1))
        vals = full
    store("conv_dst", vals, "sink_conv")
    store("ssm_dst", lg.view(sst_out, (128, 128 * nh, 1)), "sink_ssm")
    lg.gemm(out, "o", d.hidden, out=lg.p_out())
    return lg.g


def kda_blocks(lg: LayerGraph, qkv, fg, small, gate, par, cst_in, sst_in, out, cst_out, sst_out):
    """The chunked KDA of one prefill launch (math ``tools/glm_np.py``).
    Per (head, 64-token chunk) blocks live in (cols, rows, nh, NC) tensors, so batch_gemm(x, w, (ta, tb))
    computes op(x) . op(w) per block. kda_ck_in (conv, norms, gates, decay-scaled operands) -> Gram products
    per 16-row sub-chunk -> kda_ck_solve (T = (I + beta KK)^-1, P) -> U = T vb, W = T kbg, B = kL^T U,
    KW = kL^T W -> sequential chain S' = e^G63 S - KW S + B -> o = qG S + P (U - W S) -> kda_ck_out."""
    d, T = lg.d, lg.T
    nh, CH, NC = d.kda_heads, KDA_CHUNK, T // KDA_CHUNK
    blk, sq, gram = (128, CH, nh, NC), (128, 128, nh, NC), (CH, CH, nh, NC)

    def t(shape: tuple[int, ...]) -> Tensor:
        return lg.tensor(shape, F32)

    def bg(x: Tensor, w: Tensor, ta: bool, tb: bool, shape) -> Tensor:
        y = t(shape)
        lg.node("batch_gemm", [x, w], [y], struct.pack("??", ta, tb))
        return y

    Qr, Kr, qG, kbg, kL, vb = (t(blk) for _ in range(6))
    Kt, cs = t((128, CH, 4, nh, NC)), t((192, nh, NC))
    conv_w = lg.input("conv_w", (3 * d.C, 4), F32)
    meta = lg.meta()
    lg.node(
        "rg_kda_ck_in_f32",
        [qkv, fg, small, conv_w, cst_in, par, meta],
        [Qr, Kr, qG, kbg, kL, vb, Kt, cs, cst_out],
        struct.pack("f", d.kda_lb),
    )
    B4 = 4 * nh * NC
    Kr4, Qr4, Kt4 = lg.view(Kr, (128, 16, B4)), lg.view(Qr, (128, 16, B4)), lg.view(Kt, (128, CH, B4))
    KK = lg.view(bg(Kr4, Kt4, False, True, (CH, 16, B4)), gram)
    QK = lg.view(bg(Qr4, Kt4, False, True, (CH, 16, B4)), gram)
    P, Tm = t(gram), t(gram)
    lg.node("rg_kda_ck_solve_f32", [KK, QK, cs], [P, Tm])
    U = bg(Tm, vb, False, False, blk)
    W = bg(Tm, kbg, False, False, blk)
    B = bg(kL, U, True, False, sq)
    KW = bg(kL, W, True, False, sq)
    one = (128, 128, nh, 1)

    def chunks(x: Tensor, shape: tuple[int, ...]) -> list[Tensor]:
        if NC == 1:
            return [x]
        parts = [t(shape) for _ in range(NC)]
        lg.node("split", [x], parts, struct.pack("I", 3 if len(shape) == 4 else 2))
        return parts

    KWc, Bc, csc = chunks(KW, one), chunks(B, one), chunks(cs, (192, nh, 1))
    S, Ss = sst_in, [sst_in]
    for c in range(NC):
        X = bg(KWc[c], S, False, False, one)
        y = t(one)
        lg.node("rg_kda_ck_step_f32", [S, X, Bc[c], csc[c]], [y])
        S = y
        if c < NC - 1:
            Ss.append(y)
    lg.node("memcpy", [S], [sst_out])
    if NC > 1:
        Sall = t(sq)
        lg.node("concat", Ss, [Sall], struct.pack("I", 3))
    else:
        Sall = sst_in
    QS = bg(qG, Sall, False, False, blk)
    D = t(blk)
    lg.node("sub_fwd_f32", [U, bg(W, Sall, False, False, blk)], [D])
    PD = bg(P, D, False, False, blk)
    lg.node(
        "rg_kda_ck_out_f32",
        [QS, PD, gate, lg.input("o_norm", (128,), F32)],
        [out],
        struct.pack("f", d.eps),
    )


def kda_verify_graph(device: Device, dims: Dims, tokens: int, verify: int, blocks: int) -> Graph:
    """A KDA layer for a verify step: ``verify`` = Tq tokens of each of T // Tq sequences (columns b Tq + t, the
    rest padding). ``kda_verify`` reads each sequence's state without writing it and records the tokens' updates in
    ``kda_pend`` (128, 3, nh, Tq, B) and ``kda_pconv`` (3C, Tq, B) for ``kda_commit``."""
    lg = LayerGraph(device, f"kda_ver{verify}_T{tokens}_NB{blocks}", dims, tokens)
    d, T = dims, tokens
    B = T // verify
    qkv, small, fg, gate, par = kda_common(lg)
    out = lg.tensor((d.C, T), BF16)
    lg.node(
        "rg_kda_verify_f32",
        [
            qkv,
            fg,
            small,
            lg.input("conv_w", (3 * d.C, 4), F32),
            lg.arena32(blocks),
            lg.input("kda_bases", (4, B), I32),
            par,
            gate,
            lg.input("o_norm", (128,), F32),
        ],
        [
            out,
            lg.input("kda_pend", (128, 3, d.kda_heads, verify, B), F32),
            lg.input("kda_pconv", (3 * d.C, verify, B), F32),
        ],
        struct.pack("ff", d.kda_lb, d.eps),
    )
    lg.gemm(out, "o", d.hidden, out=lg.p_out())
    return lg.g


def commit_graph(device: Device, dims: Dims, rows: int, verify: int, layers: int, blocks: int) -> Graph:
    """The accepted tokens of a verify step: ``kda_commit`` over every KDA layer's entries (``commit_bases``
    (4, B L), ``commit_counts`` (B L,), the layers' pend and pconv runs back to back), and each row's accepted
    column of the normed hidden ``hf`` into its STATE block's hidden slot (``hidden_table`` (2, B))."""
    lg = LayerGraph(device, f"commit_B{rows}_Q{verify}_L{layers}_NB{blocks}", dims, 1)
    d, BL = dims, rows * layers
    arena = lg.arena32(blocks)
    lg.node(
        "rg_kda_commit_f32",
        [
            arena,
            lg.input("kda_pend", (128, 3, d.kda_heads, verify, BL), F32),
            lg.input("kda_pconv", (3 * d.C, verify, BL), F32),
            lg.input("commit_bases", (4, BL), I32),
            lg.input("commit_counts", (BL,), I32),
        ],
        [lg.input("sink_commit", (64, 1), I32)],
    )
    lg.node(
        "rg_hidden_store_f32",
        [arena, lg.input("hf", (d.hidden, rows * verify), F32), lg.input("hidden_table", (2, rows), I32)],
        [lg.input("sink_hidden", (64, 1), I32)],
    )
    return lg.g


def hidden_store_graph(device: Device, dims: Dims, tokens: int, blocks: int) -> Graph:
    """One prefill chunk's last real column of the normed hidden into its STATE block's hidden slot"""
    lg = LayerGraph(device, f"hidden_T{tokens}_NB{blocks}", dims, 1)
    lg.node(
        "rg_hidden_store_f32",
        [lg.arena32(blocks), lg.input("hf", (dims.hidden, tokens), F32), lg.input("hidden_table", (2, 1), I32)],
        [lg.input("sink_hidden", (64, 1), I32)],
    )
    return lg.g


# ------------------------------------------------------------------ MLA + DSA
def mla_pre_graph(
    device: Device, dims: Dims, tokens: int, blocks: int, index: bool, mtp: bool = False
) -> Graph:
    """Residual and site, queries for the slots (``q_all`` (R, nh T) bf16, column h + nh t), the new keys
    into their pages (``kv_rows``: latent row and scale unit; ``pool_rows``: pool units and tails), the
    indexer's queries and head weights when asked, and the accumulator reset."""
    d, T = dims, tokens
    tag = "mtp_" if mtp else ""
    lg = LayerGraph(device, f"{tag}mla_pre_i{int(index)}_T{T}_NB{blocks}", d, T)
    R, nh = d.latent, d.mla_heads
    if mtp:
        h16, _, h8, _ = mtp_collapse(lg, lg.input("x", (d.hidden, T), F32), "ln_a", q8=1)
    else:
        X = lg.residual()
        coef = lg.site(X, "hca")
        h16, _, h8, _ = lg.collapse(X, coef, "ln_a", q8=1)
    qa, ca = lg.split0(lg.gemm8(h8, "w_a8", d.q_lora + R), [d.q_lora, R])
    xo, qr16 = lg.tensor((d.q_lora, T), F32), lg.tensor((d.q_lora, T), BF16)
    lg.node(
        "rg_add_rms_f32",
        [qa, qa, lg.input("q_a_ln", (d.q_lora,), F32)],
        [xo, qr16],
        struct.pack("fi", d.eps, 0),
    )
    lat = lg.tensor((R, T), F32)
    lg.node(
        "rg_ml4_norm_f32",
        [ca, lg.input("kv_a_ln", (R,), F32), lg.dep()],
        [lat],
        struct.pack("fi", d.eps, 0),
    )
    arena = lg.arena8(blocks)
    lg.node(
        "rg_mla_store_f32",
        [arena, lg.input("kv_rows", (ROWS_FIELDS, T), I32), lat, h16],
        [lg.input("sink_kv", (64, 1), I32)],
    )
    kgw = lg.gemm(h16, "w_ix", 288)
    kg = lg.tensor((256, T), F32)
    lg.node(
        "rg_ix_prep_f32",
        [kgw, lg.input("ix_kn_w", (128,), F32), lg.input("ix_kn_b", (128,), F32)],
        [kg, lg.input("ix_w", (32, T), F32)],
        struct.pack("f", 1e-6),
    )
    lg.node(
        "rg_gx_pool_f32",
        [arena, lg.input("pool_rows", (ROWS_FIELDS, T), I32), kg, lg.input("ix_ape", (128, 4), F32)],
        [lg.input("sink_pool", (64, 1), I32)],
    )
    ql = lg.view(lg.gemm(qr16, "wqlat", R * nh), (R, nh * T))
    lg.node("cast_f32_to_bf16", [ql], [lg.input("q_all", (R, nh * T), BF16)])
    if index:
        NI, Di = d.index_heads, d.index_dim
        qi = lg.view(lg.gemm(qr16, "ix_wq", NI * Di), (Di, NI * T))
        lg.node("cast_f32_to_bf16", [qi], [lg.input("ix_q", (Di, NI * T), BF16)])
    W = nh * T
    lg.node(
        "rg_mla_acc_init_f32",
        [
            lg.input("acc_o", (R, W, 1), F32),
            lg.input("acc_m", (W, 1), F32),
            lg.input("acc_l", (W, 1), F32),
        ],
        [lg.input("sink_acc", (64, 1), I32)],
    )
    return lg.g


def _merge(lg: LayerGraph, pv: Tensor, mx: Tensor, ls: Tensor, table: Tensor, chunk: bool) -> None:
    W = lg.d.mla_heads * lg.T
    R = lg.d.latent
    acc = [
        lg.input("acc_o", (R, W, 1), F32),
        lg.input("acc_m", (W, 1), F32),
        lg.input("acc_l", (W, 1), F32),
    ]
    lg.node(
        f"rg_mla_slot_{'am' if chunk else 'acc'}_f32",
        [pv, mx, ls, table, *acc],
        [lg.input("sink_slots", (64, 1), I32)],
    )


def attn_slots_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    query_block: int,
    slots: int,
    blocks: int,
    keep_keys: int = 0,
) -> Graph:
    """One launch of ``slots`` MLA slots of ``query_block`` tokens each, merged into the T columns' running
    accumulators. ``slots`` (8, S) i32, ``lat_pages`` (C / 128, S) i32 first rows (-1: none).
    ``keep_keys`` > 0: a prefill chunk's DSA mask ``ix_mask`` (keep_keys, T)."""
    d, T, S, C, R = dims, tokens, slots, SLOT_KEYS, dims.latent
    name = f"attn_slots_T{T}_Q{query_block}_S{S}_K{keep_keys}_NB{blocks}"
    lg = LayerGraph(device, name, d, T)
    W, M = d.mla_heads * T, d.mla_heads * query_block
    Mp = -(-M // 64) * 64
    table = lg.input("slots", (8, S), I32)
    pages = lg.input("lat_pages", (C // 128, S), I32)
    q = lg.tensor((R, M, S), BF16)
    lg.node("rg_mla_slot_q_f32", [lg.input("q_all", (R, W), BF16), table], [q])
    ks = lg.tensor((R, C, S), BF16)
    lg.node(
        "rg_mla_gather_f32",
        [lg.arena8(blocks), pages, pages, table, lg.dep()],
        [ks],
        struct.pack("i", 0),
    )
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, S), BF16)
    mx, ls = lg.tensor((Mp, S), F32), lg.tensor((Mp, S), F32)
    prm = struct.pack("fi", d.scale, d.mla_heads)
    if keep_keys:
        mask = lg.input("ix_mask", (keep_keys, T), F32)
        lg.node("rg_mla_slot_pm_f32", [sc, table, mask], [P, mx, ls], prm)
    else:
        lg.node("rg_mla_slot_p_f32", [sc, table], [P, mx, ls], prm)
    pv = lg.tensor((R, M, S), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    _merge(lg, pv, mx, ls, table, query_block > 1 and S == 1)
    return lg.g


def attn_sparse_graph(device: Device, dims: Dims, tokens: int, blocks: int, pages: int) -> Graph:
    """Sparse decode attention: one slot per column over its DSA-listed keys (``ix_keys``, ``ix_slots``, this
    layer's page offsets ``loff``, the step's block table ``kv_blocks``). One launch at any context."""
    d, T, C, R = dims, tokens, LIST_KEYS, dims.latent
    lg = LayerGraph(device, f"attn_sparse_T{T}_NB{blocks}_P{pages}", d, T)
    W, M = d.mla_heads * T, d.mla_heads
    Mp = -(-M // 64) * 64
    table = lg.input("ix_slots", (8, T), I32)
    q = lg.tensor((R, M, T), BF16)
    lg.node("rg_mla_slot_q_f32", [lg.input("q_all", (R, W), BF16), table], [q])
    ks = lg.tensor((R, C, T), BF16)
    lg.node(
        "rg_mla_gather_list_f32",
        [
            lg.arena8(blocks),
            lg.input("ix_keys", (C, T), I32),
            lg.input("kv_blocks", (pages, T), I32),
            lg.input("loff", (64,), I32),
            table,
            lg.dep(),
        ],
        [ks],
    )
    sc = lg.tensor((C, M, T), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, T), BF16)
    mx, ls = lg.tensor((Mp, T), F32), lg.tensor((Mp, T), F32)
    lg.node("rg_mla_slot_p_f32", [sc, table], [P, mx, ls], struct.pack("fi", d.scale, d.mla_heads))
    pv = lg.tensor((R, M, T), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    _merge(lg, pv, mx, ls, table, False)
    return lg.g


def index_slots_graph(
    device: Device, dims: Dims, tokens: int, query_block: int, slots: int, pools: int, blocks: int
) -> Graph:
    """One launch of DSA indexer slots: the block's index queries (``ix_q``, token-major) against a chunk
    of pool keys (``ix_pages`` (C / 32, S)), weighted relu scores into the step's scratch ``ix_score``
    (pools, T). Every pool lands in the scratch; ``ix_thr`` and what follows see only a column's visible ones.
    ``ix_w`` already folds the relu scale, so the scale here is 1."""
    d, T, S, C = dims, tokens, slots, SLOT_POOLS
    Di, NI = d.index_dim, d.index_heads
    name = f"index_slots_T{T}_Q{query_block}_S{S}_K{pools}_NB{blocks}"
    lg = LayerGraph(device, name, d, T)
    M = NI * query_block
    table = lg.input("slots", (8, S), I32)
    q = lg.tensor((Di, M, S), BF16)
    lg.node("rg_mla_slot_q_f32", [lg.input("ix_q", (Di, NI * T), BF16), table], [q])
    ks = lg.tensor((Di, C, S), BF16)
    lg.node(
        "rg_gx_gather_f32",
        [lg.arena8(blocks), lg.input("ix_pages", (C // PAGE_POOLS, S), I32), table, lg.dep()],
        [ks],
    )
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    lg.node(
        "rg_ix_slot_acc_f32",
        [sc, lg.input("ix_w", (NI, T), F32), table, lg.input("ix_score", (pools, T), F32)],
        [lg.input("sink_ix", (64, 1), I32)],
        struct.pack("f", 1.0),
    )
    return lg.g


def index_mask_graph(device: Device, dims: Dims, tokens: int, pools: int) -> Graph:
    """A prefill chunk's DSA selection: the threshold over each column's visible pools, then the additive
    per-key mask for ``mla_slot_pm``: kept pools' keys and the column's tail"""
    T = tokens
    lg = LayerGraph(device, f"index_mask_T{T}_K{pools}", dims, T)
    score, vis = lg.input("ix_score", (pools, T), F32), lg.input("vis", (T,), I32)
    thr = lg.tensor((64, T), I32)
    lg.node("rg_ix_thr_f32", [score, vis], [thr], struct.pack("i", dims.top_pools))
    lg.node(
        "rg_gx_mask_f32",
        [score, vis, lg.input("klen", (T,), I32), thr, lg.meta()],
        [lg.input("ix_mask", (4 * pools, T), F32)],
    )
    return lg.g


def index_list_graph(device: Device, dims: Dims, tokens: int, pools: int) -> Graph:
    """A decode step's DSA selection as key lists: threshold, per-block counts, the kept pools, then per
    column the key positions ``ix_keys`` (2112, T) (the pools' keys and the tail) and its slot record"""
    T, KS = tokens, dims.top_pools
    lg = LayerGraph(device, f"index_list_T{T}_K{pools}", dims, T)
    score, vis = lg.input("ix_score", (pools, T), F32), lg.input("vis", (T,), I32)
    thr = lg.tensor((64, T), I32)
    lg.node("rg_ix_thr_f32", [score, vis], [thr], struct.pack("i", KS))
    cnt, gsum = lg.tensor((pools // 64, T), I32), lg.tensor((pools // 4096, T), I32)
    lg.node("rg_ix_bcnt_f32", [score, vis, thr], [cnt, gsum])
    plist, pslots = lg.tensor((KS, T), I32), lg.tensor((8, T), I32)
    members = max(1, 48 // T)
    lg.node(
        "rg_ix_list_f32",
        [score, vis, thr, cnt, gsum],
        [plist, pslots],
        struct.pack("iii", members, KS, dims.mla_heads),
    )
    lg.node(
        "rg_gx_expand_f32",
        [plist, pslots, vis, lg.input("klen", (T,), I32)],
        [lg.input("ix_keys", (LIST_KEYS, T), I32), lg.input("ix_slots", (8, T), I32)],
        struct.pack("ii", dims.mla_heads, LIST_KEYS),
    )
    return lg.g


def mla_post_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """The folded o W_UV on the accumulated latent outputs -> this rank's partial ``p_out``."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"mla_post_T{T}", d, T)
    R, nh = d.latent, d.mla_heads
    acc = lg.view(lg.input("acc_o", (R, nh * T, 1), F32), (R * nh, T))
    o16 = lg.tensor((R * nh, T), BF16)
    lg.node("rg_pack16_f32", [acc], [o16])
    lg.gemm(o16, "wov", d.hidden, out=lg.p_out())
    return lg.g


# ------------------------------------------------------------------ MLP
def mtp_collapse(lg: LayerGraph, x: Tensor, ln: str, q8: int = 0):
    """RMSNorm of the one stream x, through ``mhc_pre``'s head mode over 4 equal streams"""
    x4 = lg.tensor((NS * lg.d.hidden, lg.T), F32)
    lg.node("concat", [x] * NS, [x4], struct.pack("I", 0))
    return lg.collapse(x4, lg.input("c", (lg.T, 24), F32), ln, q8=q8, mode=1)


def mtp_residual(lg: LayerGraph) -> Tensor:
    """``x_out`` = x + the previous sub-layer's partials summed over the ranks"""
    H, T = lg.d.hidden, lg.T
    if lg.d.gathered(T):
        p = lg.tensor((H, T), F32)
        lg.node("rg_rsum_f32", [lg.input("p", (H, T, lg.d.ranks), F32)], [p])
    else:
        p = lg.input("p", (H, T), F32)
    out = lg.input("x_out", (H, T), F32)
    lg.node("add_fwd_f32", [lg.input("x", (H, T), F32), p], [out])
    return out


def mtp_in_graph(device: Device, dims: Dims, tokens: int, columns: int, vocab: int) -> Graph:
    """The MTP layer's input for T columns: column c combines the embedding of ``mtp_tok`` [c] with hidden column
    ``mtp_cols`` [c] of [``mtp_prev`` (H, 1) | ``hf`` (H, columns)] (normed hiddens), as
    ``x_out`` = eh_proj [enorm(e) ; hnorm(h)] (both halves sigma-ordered bf16)."""
    lg = LayerGraph(device, f"mtp_in_T{tokens}_X{columns}", dims, tokens)
    H, T = dims.hidden, tokens
    both = lg.tensor((H, columns + 1), F32)
    lg.node(
        "concat",
        [lg.input("mtp_prev", (H, 1), F32), lg.input("hf", (H, columns), F32)],
        [both],
        struct.pack("I", 1),
    )
    hc = lg.tensor((H, T), F32)
    lg.node("gather_fwd_f32", [both, lg.input("mtp_cols", (T,), I32)], [hc], struct.pack("i", 1))
    e = lg.tensor((H, T), BF16)
    lg.node(
        "gather_fwd_bf16",
        [lg.input("emb", (H, vocab), BF16), lg.input("mtp_tok", (T,), I32)],
        [e],
        struct.pack("i", 1),
    )
    ef = lg.tensor((H, T), F32)
    lg.node("cast_bf16_to_f32", [e], [ef])

    def rmsn(x: Tensor, name: str) -> Tensor:
        xo, h = lg.tensor((H, T), F32), lg.tensor((H, T), BF16)
        lg.node(
            "rg_add_rms_f32", [x, x, lg.input(name, (H,), F32)], [xo, h], struct.pack("fi", dims.eps, 0)
        )
        return h

    cat = lg.tensor((2 * H, T), BF16)
    lg.node("concat", [rmsn(ef, "mtp.enorm"), rmsn(hc, "mtp.hnorm")], [cat], struct.pack("I", 0))
    lg.gemm(cat, "mtp.eh_proj", H, out=lg.input("x_out", (H, T), F32))
    return lg.g


def mtp_head_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """x + p, ``shared_head.norm``, the shared head -> logits (V_r, T); ``hf`` the next chain link's hidden"""
    d, T = dims, tokens
    lg = LayerGraph(device, f"mtp_head_T{T}", d, T)
    x = mtp_residual(lg)
    h16, h32 = mtp_collapse(lg, x, "mtp.sh_norm")
    lg.node("memcpy", [h32], [lg.input("hf_out", (d.hidden, T), F32)])
    lg.gemm(h16, "head", d.vocab, out=lg.input("logits", (d.vocab, T), F32))
    return lg.g


def final_hidden_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """Every column's normed hidden ``hf`` (H, T) of a prefill chunk, for the MTP layer's keys"""
    d, T = dims, tokens
    lg = LayerGraph(device, f"final_hidden_T{T}", d, T)
    X = lg.residual()
    _, h32 = lg.collapse(X, lg.input("c", (T, 24), F32), "norm", mode=1)
    lg.node("memcpy", [h32], [lg.input("hf", (d.hidden, T), F32)])
    return lg.g


def ffn_graph(
    device: Device, dims: Dims, tokens: int, moe: bool, paged: bool, mtp: bool = False
) -> Graph:
    """The MLP site. Dense: FP8 up, clamped SwiGLU, FP8 down. MoE: the sigmoid router, the shared expert
    (FP8, its output scale to the combine), and the routed experts through weight slots: one (M, K) matrix per
    (token, route) at decode, or per page of an expert's rows in a prefill chunk (``paged``)."""
    d, T = dims, tokens
    kind = "paged" if paged else ("moe" if moe else "dense")
    lg = LayerGraph(device, f"{'mtp_' if mtp else ''}ffn_{kind}_T{T}", d, T)
    H, K = d.hidden, d.top
    if mtp:
        X, coef = None, None
    else:
        X = lg.residual()
        coef = lg.site(X, "hcf")
    if not moe:
        _, _, h8, hs = lg.collapse(X, coef, "ln_m", q8=1)
        a8, sa = lg.q8(lg.gemm8(h8, "du8", 2 * d.dense), hs, swiglu=True)
        lg.tscale(lg.gemm8(a8, "dd8", H), sa, out=lg.p_out())
        return lg.g
    if mtp:
        pre = mtp_collapse(lg, mtp_residual(lg), "ln_m", q8=1 if paged else 2)
    else:
        pre = lg.collapse(X, coef, "ln_m", q8=1 if paged else 2)
    h32, h8, hs = pre[1], pre[2], pre[3]
    lgts = lg.tensor((d.experts, T), F32)
    lg.node("gemm", [h32, lg.input("router", (H, d.experts), F32)], [lgts], GEMM_NT)
    ids, wts = lg.tensor((64, T), I32), lg.tensor((64, T), F32)
    lg.node(
        "rg_moe_route_sg_f32",
        [lgts, lg.input("router_bias", (d.experts,), F32), lg.meta()],
        [ids, wts],
        struct.pack("iif", K, d.experts, d.route_scale),
    )
    a8, shs = lg.q8(lg.gemm8(h8, "sh_gu8", 2 * d.expert), hs, swiglu=True)
    sh = lg.gemm8(a8, "sh_down8", H)
    w, E = d.expert, d.experts
    W13 = lg.input("x13", (2 * w, H, E), F8)
    C13 = lg.input("x13.c", (2 * w, E), F32)
    W2 = lg.input("x2", (H, w, E), F8)
    C2 = lg.input("x2.c", (H, E), F32)
    if paged:
        _paged_experts(lg, h32, ids, wts, sh, shs, W13, C13, W2, C2)
        return lg.g
    xq = pre[4]
    S = K * T
    wd13, wd2 = lg.tensor((2 * w, H, S), F8), lg.tensor((H, w, S), F8)
    lg.node("rg_x8_copy_f32", [W13, ids], [wd13], struct.pack("i", K))
    lg.node("rg_x8_copy_f32", [W2, ids], [wd2], struct.pack("i", K))
    y13 = lg.tensor((2 * w, 1, S), F32)
    lg.node("batch_gemm", [xq, wd13], [y13], GEMM_NN)
    hq, hsl = lg.tensor((w, 1, S), F8), lg.tensor((1, S), F32)
    lg.node(
        "rg_ds_xact_f32",
        [y13, y13, C13, hs, ids],
        [hq, hsl],
        struct.pack("ifii", K, d.limit, 0, 0),
    )
    y2 = lg.tensor((H, 1, S), F32)
    lg.node("batch_gemm", [hq, wd2], [y2], GEMM_NN)
    lg.node("rg_ds_xsum_f32", [y2, C2, hsl, ids, wts, sh, shs], [lg.p_out()], struct.pack("i", K))
    return lg.g


def paged_shape(tokens: int, top: int, experts: int) -> tuple[int, int, int]:
    """(row shift, pages, rows): pages for every pair, ceil(T K / PS) + E"""
    psh = 3
    while (1 << psh) < 2 * tokens * top / experts:
        psh += 1
    PS = 1 << psh
    P = -(-tokens * top // PS) + experts
    return psh, P, PS * P


def _paged_experts(lg, h32, ids, wts, sh, shs, W13, C13, W2, C2) -> None:
    """Each touched expert's slice copied once per page of its (token, route) rows (``moe_cnt`` /
    ``moe_pages`` / ``moe_asg``), FP8 batch_gemms over the pages, weighted combine"""
    d, T = lg.d, lg.T
    H, K, E, w = d.hidden, d.top, d.experts, d.expert
    psh, P, S = paged_shape(T, K, E)
    PS = 1 << psh
    meta = lg.meta()
    cntm = lg.tensor((E, 24), I32)
    lg.node("rg_moe_cnt_f32", [ids, meta], [cntm], struct.pack("ii", 0, K))
    po, pe = lg.tensor((E,), I32), lg.tensor((P,), I32)
    lg.node("rg_moe_pages_f32", [cntm], [po, pe], struct.pack("i", psh))
    gidx, bm = lg.tensor((S,), I32), lg.tensor((64, T), I32)
    lg.node("rg_moe_asg_f32", [ids, meta, cntm, po, pe], [gidx, bm], struct.pack("iii", 0, K, psh))
    xq, sxr = lg.tensor((H, S), F8), lg.tensor((1, S), F32)
    lg.node("rg_ds_xqg_f32", [h32, gidx], [xq, sxr])
    xv = lg.view(xq, (H, PS, P))
    wd13, wd2 = lg.tensor((2 * w, H, P), F8), lg.tensor((H, w, P), F8)
    lg.node("rg_x8_copy_f32", [W13, pe], [wd13], struct.pack("i", 0))
    lg.node("rg_x8_copy_f32", [W2, pe], [wd2], struct.pack("i", 0))
    y13 = lg.tensor((2 * w, PS, P), F32)
    lg.node("batch_gemm", [xv, wd13], [y13], GEMM_NN)
    yf = lg.view(y13, (2 * w, 1, S))
    hq, hs = lg.tensor((w, 1, S), F8), lg.tensor((1, S), F32)
    lg.node(
        "rg_ds_xact_f32",
        [yf, yf, C13, sxr, pe],
        [hq, hs],
        struct.pack("ifii", 0, d.limit, 0, psh),
    )
    y2 = lg.tensor((H, PS, P), F32)
    lg.node("batch_gemm", [lg.view(hq, (w, PS, P)), wd2], [y2], GEMM_NN)
    lg.node(
        "rg_ds_xcomb_f32",
        [lg.view(y2, (H, S)), C2, hs, ids, bm, wts, sh, shs],
        [lg.p_out()],
        struct.pack("i", K),
    )


# ------------------------------------------------------------------ head
def head_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """The last sub-layer's update, the HC head mean, the final norm and this rank's vocabulary slice ->
    logits (V_r, T); ``hf`` (H, T) f32: the normed hidden the MTP layer reads"""
    d, T = dims, tokens
    lg = LayerGraph(device, f"head_T{T}", d, T)
    X = lg.residual()
    h16, h32 = lg.collapse(X, lg.input("c", (T, 24), F32), "norm", mode=1)
    lg.node("memcpy", [h32], [lg.input("hf", (d.hidden, T), F32)])
    lg.gemm(h16, "head", d.vocab, out=lg.input("logits", (d.vocab, T), F32))
    return lg.g


def head_columns_graph(device: Device, dims: Dims, tokens: int, columns: int) -> Graph:
    """The head of a prefill chunk at ``columns`` of its T columns (``cols`` (columns,) i32)"""
    d, T, C = dims, tokens, columns
    lg = LayerGraph(device, f"head_T{T}_C{C}", d, C)
    H = d.hidden
    cols = lg.input("cols", (C,), I32)

    def pick(name: str, shape: tuple[int, ...], outs: tuple[int, ...], axis: int) -> Tensor:
        out = lg.tensor(outs, F32)
        lg.node("gather_fwd_f32", [lg.input(name, shape, F32), cols], [out], struct.pack("i", axis))
        return out

    X = pick("X", (NS * H, T), (NS * H, C), 1)
    p = pick("p", (H, T), (H, C), 1)
    c = pick("c", (T, 24), (C, 24), 0)
    Xo = lg.tensor((NS * H, C), F32)
    lg.node("rg_mhc_res_f32", [X, lg.view(p, (H, C, 1)), c], [Xo])
    h16, h32 = lg.collapse(Xo, c, "norm", mode=1)
    lg.node("memcpy", [h32], [lg.input("hf", (H, C), F32)])
    lg.gemm(h16, "head", d.vocab, out=lg.input("logits", (d.vocab, C), F32))
    return lg.g


def sample_graph(device: Device, dims: Dims, tokens: int, rows: int, nranks: int) -> Graph:
    """The device sampler (``qwen3_5.sampling``) over every column's logits (rows = T) or column ``last``'s
    (rows = 1). ``logits_all`` is the ranks' all-gathered vocabulary slices, (V_r, T, nranks)."""
    d, T, R = dims, tokens, rows
    lg = LayerGraph(device, f"sample_T{T}_R{R}_n{nranks}", d, T)
    V = d.vocab * nranks
    if nranks > 1:
        perm = lg.tensor((d.vocab, nranks, T), F32)
        lg.node("rg_ml4_perm_f32", [lg.input("logits_all", (d.vocab, T, nranks), F32)], [perm])
        full = lg.view(perm, (V, T))
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
    sampler(lg.g, full, lg.input("samp_params", (PARAMS, R), F32))
    return lg.g
