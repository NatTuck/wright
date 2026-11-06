"""Per-layer Synapse graphs of DeepSeek-V4.1-Flash for one tensor-parallel rank.

The residual is 4 hyper-connection streams, which cross launches as ``X`` (f32 (4H, T)) with the
previous sub-layer's coefficients ``c`` (T, 24) and its output partial ``p``, summed over the ranks
between launches, as in ``glm5_3_flash.layers``. Every sub-layer graph starts ``X_out = post p +
comb^T X`` with ``c``, computes its own site's coefficients ``c_out`` from ``X_out``, and collapses
``X_out`` with the pre-mix of ``c``: a sub-layer's pre-mix collapses the next sub-layer's input. The
head collapses with the last ffn's ``c``. Layer 0 starts from the embedding with identity ``c`` (pre-mix
on stream 0, comb the identity) and a zero partial.

Activations into the FP8 GEMMs use power-of-two token scales (the ``*_p2`` / ``*_lin`` builds), which
round to the same values as the checkpoint's per-32 power-of-two scales.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from resin_models_gaudi.deepseek_v41.cache import PAGE_ROWS
from resin_models_gaudi.glm5_3_flash.layers import GEMM_NN, GEMM_NT
from resin_models_gaudi.glm5_3_flash.layers import LayerGraph as GlmLayerGraph
from resin_models_gaudi.glm5_3_flash.weights import NS
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, U8, Device

ROPE = 64
WINDOW = 128
PAGE = 128  # tokens per latent page: 128 FP8 rows and one row of their scales
QUERY_BLOCK_DENSE = 128  # prefill columns per slot of the dense compressed attention
ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024


@dataclass(frozen=True, slots=True)
class Dims:
    """One rank's model dimensions."""

    hidden: int
    heads: int
    head_dim: int
    q_lora: int
    o_lora: int
    groups: int
    """output groups of wo_a on this rank"""
    expert: int
    """each expert's intermediate width on this rank"""
    experts: int
    top: int
    vocab: int
    eps: float
    hc_iters: int
    hc_eps: float
    limit: float
    route_scale: float
    ranks: int
    gather_tokens: int = 0

    @classmethod
    def of(cls, cfg, ranks: int, vocab: int) -> Dims:
        return cls(
            hidden=cfg.hidden_size,
            heads=cfg.num_attention_heads // ranks,
            head_dim=cfg.head_dim,
            q_lora=cfg.q_lora_rank,
            o_lora=cfg.o_lora_rank,
            groups=cfg.o_groups // ranks,
            expert=cfg.moe_intermediate_size // ranks,
            experts=cfg.num_routed_experts,
            top=cfg.num_experts_per_token,
            vocab=vocab,
            eps=cfg.rms_norm_eps,
            hc_iters=cfg.hc_sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            limit=cfg.swiglu_limit,
            route_scale=cfg.routed_scaling_factor,
            ranks=ranks,
        )

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5

    def gathered(self, tokens: int) -> bool:
        return self.ranks > 1 and tokens <= self.gather_tokens


class LayerGraph(GlmLayerGraph):
    """GLM's building blocks with DeepSeek-V4.1's power-of-two activation scales."""

    def __init__(self, device: Device, name: str, dims: Dims, tokens: int) -> None:
        self.g = Graph(
            device,
            f"ds41_{name}" + (f"_g{dims.ranks}" if dims.gathered(tokens) else ""),
        )
        self.d = dims
        self.T = tokens
        self._inputs: dict[str, Tensor] = {}
        self._rt = None

    def collapse8(
        self, X: Tensor, coef: Tensor, ln: str
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collapse with ``coef``'s pre-mix, RMSNorm -> h16, h32, h8 (tau), hs"""
        H, T = self.d.hidden, self.T
        h16, h32 = self.tensor((H, T), BF16), self.tensor((H, T), F32)
        h8, hs = self.tensor((H, T), F8), self.tensor((1, T), F32)
        self.node(
            "rg_mhc_pre8_p2_f32",
            [X, coef, self.input(ln, (H,), F32)],
            [h16, h32, h8, hs],
            struct.pack("fi", self.d.eps, 0),
        )
        return h16, h32, h8, hs

    def q8p(
        self,
        x: Tensor,
        *,
        sin: Tensor | None = None,
        norm: str | None = None,
        swiglu: bool = False,
    ):
        """Power-of-two FP8 of x (times ``sin`` per token), of its RMSNorm with weight ``norm``, or of the
        clamped SwiGLU of its halves (tau order)."""
        K = x.shape[0] // 2 if swiglu else x.shape[0]
        T = x.shape[1]
        q, s = self.tensor((-(-K // 256) * 256, T), F8), self.tensor((1, T), F32)
        mode = (
            (1 if sin is not None else 0)
            | (2 if norm else 0)
            | (4 if swiglu else 0)
            | (8 if T >= 24 else 0)
        )
        nw = self.input(norm, (K,), F32) if norm else x
        build = {0: "z", 2: "n", 5: "s"}.get(
            mode & 7, ""
        )  # one-producer builds
        self.node(
            f"rg_ds_q8{build}_p2_f32",
            [x, sin if sin is not None else x, nw],
            [q, s],
            struct.pack("iff", mode, self.d.eps, self.d.limit),
        )
        return q, s

    def rope(self, x: Tensor, cs: Tensor, per_token: int) -> Tensor:
        """RoPE on the last 64 dims of each head of x (head_dim, per_token N), columns token-major; ``cs`` has one
        column per token (inputs ``rope_b{n}`` ones, ``rope_s{n}`` zeros, ``rope_lp{N}`` zeros)."""
        y = self.tensor(x.shape, F32)
        cols = cs.shape[1]
        self.node(
            "rg_ml4_rope_f32",
            [x, cs, self.input(f"rope_s{per_token}", (per_token,), F32),
             self.input(f"rope_b{per_token}", (per_token,), F32), self.input(f"rope_lp{cols}", (cols,), F32)],
            [y],
            struct.pack("iii", per_token, x.shape[0] - ROPE, 0),
        )  # fmt: skip
        return y

    def gemm8(
        self,
        q: Tensor,
        name: str,
        M: int,
        s: Tensor | None = None,
        out: Tensor | None = None,
    ) -> Tensor:
        """FP8 x FP8 -> f32 with the weight's row scales, times the per-token scale ``s`` (1, T) of q if given."""
        y = out if out is not None else self.tensor((M, q.shape[1]), F32)
        w = self.input(name, (q.shape[0], M), F8)
        sa = s if s is not None else self.input("one", (1,), F32)
        self.node(
            "fp8_gemm_f32", [q, w, sa, self.input(name + ".s", (M,), F32)], [y], GEMM_NT
        )
        return y

    def ds_rope(
        self,
        x: Tensor,
        cs: Tensor,
        per_token: int,
        *,
        ts: Tensor | None = None,
        norm: str | None = None,
        out: Tensor | None = None,
        bf16: bool = False,
    ) -> Tensor:
        """RoPE on the last 64 dims of each column of x (head_dim, per_token N), after an RMSNorm with weight
        ``norm`` and times the per-token scale ``ts`` (1, T); f32, or bf16 (``out`` given or ``bf16``)."""
        bf16 = bf16 or (out is not None and out.dtype == BF16)
        y = out if out is not None else self.tensor(x.shape, BF16 if bf16 else F32)
        T = cs.shape[1]
        mode = (1 if norm else 0) | (2 if ts is not None else 0) | (4 if bf16 else 0)
        self.node(
            "rg_ds_rope_f32",
            [x, cs, self.view(ts, (T,)) if ts is not None else cs,
             self.input(norm, (x.shape[0],), F32) if norm else cs],
            [y],
            struct.pack("iif", per_token, mode, self.d.eps),
        )  # fmt: skip
        return y

    def norm(self, x: Tensor, w: str) -> Tensor:
        """RMSNorm with weight ``w``, f32."""
        y = self.tensor(x.shape, F32)
        self.node(
            "rg_ml4_norm_f32",
            [x, self.input(w, (x.shape[0],), F32), self.dep()],
            [y],
            struct.pack("fi", self.d.eps, 0),
        )
        return y

    def fq4(self, x: Tensor, mode: int) -> Tensor:
        y = self.tensor(x.shape, F32)
        self.node("rg_ds_fq4_f32", [x], [y], struct.pack("i", mode))
        return y

    def keep(self, x: Tensor, name: str) -> Tensor:
        """Copy x into the persistent tensor ``name`` for a later launch."""
        out = self.input(name, x.shape, x.dtype)
        self.node("memcpy", [x], [out])
        return out

    def arena(self, blocks: int) -> Tensor:
        return self.input("arena", (ROW_BYTES, ARENA_BLOCK_ROWS, blocks), U8)


def embed_graph(device: Device, dims: Dims, tokens: int, vocab: int) -> Graph:
    """The embedding as 4 identical streams."""
    from resin_models_gaudi.glm5_3_flash.layers import embed_graph as glm_embed

    return glm_embed(device, dims, tokens, vocab)  # type: ignore[arg-type]


def engram_graph(device: Device, dims: Dims, tokens: int, part: str = "") -> Graph:
    """An Engram layer's residual: the previous sub-layer's update, then each stream gated toward the value
    that ``wkv`` makes of the lookup ``eg_emb`` (the hashed rows, summed over the ranks) -> ``X_eg``. The
    attention graph that follows starts from ``X_eg`` (``residual=False``).

    ``part`` "kv": the update and this rank's 5H / ranks rows of the product (``eg_wkv_r``) -> ``eg_kv``, which
    the ranks all-gather into ``eg_kv_all`` for ``part`` "gate": the gates from ``X`` -> ``X_eg``."""
    d, T, H = dims, tokens, dims.hidden
    lg = LayerGraph(device, f"engram_T{T}" + (f"_{part}" if part else ""), d, T)
    if part == "gate":
        C = (NS + 1) * H // d.ranks
        lg.node(
            "rg_eg_gate_g_f32",
            [lg.input("X", (NS * H, T), F32), lg.input("eg_kv_all", (C, T, d.ranks), F32),
             lg.input("eg_w", (NS * H,), F32)],
            [lg.input("X_eg", (NS * H, T), F32)],
            struct.pack("f", d.eps),
        )  # fmt: skip
        return lg.g
    X = lg.residual()
    e8, se = lg.q8p(lg.input("eg_emb", (6144, T), F32))
    if part == "kv":
        C = (NS + 1) * H // d.ranks
        lg.gemm8(e8, "eg_wkv_r", C, se, out=lg.input("eg_kv", (C, T), F32))
        return lg.g
    kv = lg.gemm8(e8, "eg_wkv", (NS + 1) * H, se)
    lg.node(
        "rg_eg_gate_f32",
        [X, kv, lg.input("eg_w", (NS * H,), F32)],
        [lg.input("X_eg", (NS * H, T), F32)],
        struct.pack("f", d.eps),
    )
    return lg.g


def engram_gather_graph(
    device: Device, dims: Dims, tokens: int, rows: int, row0: int, vocab: int = 0
) -> Graph:
    """This rank's share of an Engram lookup: each column's hashed rows ``eg_idx`` (24, T) of the table's rows
    row0 .. row0 + rows - 1 (``eg_table`` (256, rows) FP8, ``eg_rs`` (rows,) f32), zeros elsewhere -> ``eg_emb``
    (6144, T), summed over the ranks before ``engram_graph``. With ``vocab``, the rows are hashed here from the
    columns' tokens ``tok`` (T,) and the host's older-token XORs ``eg_xh`` / ``eg_xl`` (64, T) (``eg_a0`` (2, vocab),
    ``eg_hp`` (64, 4): see eg_hash.c)."""
    d, T = dims, tokens
    lg = LayerGraph(
        device,
        f"engram_gather_T{T}_R{rows}_{row0}" + (f"_h{vocab}" if vocab else ""),
        d,
        T,
    )
    J = 6144 // 256
    if vocab:
        idx = lg.tensor((J, T), I32)
        lg.node(
            "rg_eg_hash_f32",
            [lg.input("tok", (T,), I32), lg.input("eg_a0", (2, vocab), I32), lg.input("eg_xh", (64, T), I32),
             lg.input("eg_xl", (64, T), I32), lg.input("eg_hp", (64, 4), I32)],
            [idx],
        )  # fmt: skip
    else:
        idx = lg.input("eg_idx", (J, T), I32)
    lg.node(
        "rg_eg_gather_f32",
        [lg.input("eg_table", (256, rows), F8), lg.input("eg_rs", (rows,), F32), idx],
        [lg.input("eg_emb", (256 * J, T), F32)],
        struct.pack("i", row0),
    )
    return lg.g


def attention_window_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    keys: int,
    blocks: int,
    residual: bool = True,
    query_block: int = 0,
) -> Graph:
    """A window-only attention sub-layer (compress ratio 0): one slot launch over the step's window keys, each
    slot ``query_block`` columns (0: all T) against ``keys`` keys.

    Tables: ``rows`` (8, T) for the key store, ``slots`` (8, S) with the window in field 7, ``lat_pages``
    (keys / 128, S); ``cs`` / ``cs_inv`` (128, T) the rotary tables at each column's position. Without
    ``residual`` the graph starts from ``X`` as it is (an Engram graph applied the update)."""
    d, T, S, C = dims, tokens, slots, keys
    qb = query_block or T
    lg = LayerGraph(
        device, f"attn_win_T{T}_S{S}_C{C}_Q{qb}" + ("" if residual else "_eg"), d, T
    )
    D, nh = d.head_dim, d.heads
    W, M = nh * T, nh * qb
    Mp = -(-M // 64) * 64
    X = lg.residual() if residual else lg.input("X", (NS * d.hidden, T), F32)
    _, _, h8, _ = lg.collapse8(X, lg.input("c", (T, 24), F32), "attn_norm")
    if T == 1:  # one token: the coefficients before the first GEMM
        lg.site(X, "hca")
    qa, kva = lg.split0(lg.gemm8(h8, "w_a", d.q_lora + D), [d.q_lora, D])
    # the keys first
    cs, cs_inv = lg.input("cs", (128, T), F32), lg.input("cs_inv", (128, T), F32)
    kn = lg.ds_rope(kva, cs, 1, norm="kv_norm")
    arena = lg.arena(blocks)
    stored = lg.tensor((64, 1), I32)
    lg.node(
        "rg_mla_store_p2_f32",
        [arena, lg.input("rows", (8, T), I32), kn, lg.input("ix_none", (128, T), BF16)],
        [stored],
    )
    qr8, sq = lg.q8p(qa, norm="q_norm")
    q = lg.gemm8(qr8, "wq_b", nh * D)
    q16 = lg.ds_rope(lg.view(q, (D, W)), cs, nh, ts=sq, bf16=True)
    table = lg.input("slots", (8, S), I32)
    lp, rp = (
        lg.input("lat_pages", (C // PAGE, S), I32),
        lg.input("rope_pages", (C // PAGE, S), I32),
    )
    qs = lg.view(
        q16, (D, M, S)
    )  # slot s holds columns M s .. M s + M - 1 (the host's slot order)
    ks = lg.tensor((D, C, S), BF16)
    lg.node(
        "rg_mla_gather_f32", [arena, lp, rp, table, stored], [ks], struct.pack("i", 0)
    )
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    P, mx, ls = (
        lg.tensor((C, M, S), BF16),
        lg.tensor((Mp, S), F32),
        lg.tensor((Mp, S), F32),
    )
    lg.node(
        "rg_ds_slot_pw1_f32"
        if M <= 64
        else "rg_ds_slot_pw_f32",  # few columns: one member each
        [sc, table],
        [P, mx, ls],
        struct.pack("fi", d.scale, nh),
    )
    pv = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    o = lg.tensor((D, W, 1), F32)
    lg.node(
        "rg_ds_slot_out_f32",
        [pv, mx, ls, table, lg.input("attn_sink", (nh,), F32)],
        [o],
        struct.pack("i", nh),
    )
    _attention_out(lg, lg.rope(lg.view(o, (D, W)), cs_inv, nh))
    if (
        T > 1
    ):  # the next sub-layer's coefficients, created last
        lg.site(X, "hca")
    return lg.g


def _attention_out(lg: LayerGraph, o: Tensor) -> None:
    """o (head_dim, heads T), the heads' outputs with their rotation removed -> wo_a per group (FP8, each
    group's heads quantized together) -> wo_b -> this rank's partial ``p_out``."""
    d, T = lg.d, lg.T
    per = d.heads // d.groups * d.head_dim
    flat = lg.view(o, (d.heads * d.head_dim, T))
    parts = lg.split0(flat, [per] * d.groups) if d.groups > 1 else [flat]
    outs = []
    for g, part in enumerate(parts):
        o8, so = lg.q8p(part)
        outs.append(lg.gemm8(o8, f"wo_a.{g}", d.o_lora, so))
    if len(outs) > 1:
        cat = lg.tensor((d.o_lora * d.groups, T), F32)
        lg.node("concat", outs, [cat], struct.pack("I", 0))
    else:
        cat = outs[0]
    b8, sb = lg.q8p(cat)
    lg.gemm8(b8, "wo_b", d.hidden, sb, out=lg.p_out())


def _w13(lg: LayerGraph, x: Tensor, ids: Tensor, R: int) -> Tensor:
    """x (H, M, N) FP8 against N decoded w13 slots (``ids``, ``R`` as in mx4_decs) -> (2w, M, N) f32. Each slot's
    rows 0..w-1 and w..2w-1 are decoded and multiplied apart (mx4_decs' unit offset)."""
    H, w = lg.d.hidden, lg.d.expert
    return _halves(lg, x, ids, R, "x13", 2 * w * H, (H, w), 2)


def _halves(
    lg: LayerGraph,
    x: Tensor,
    ids: Tensor,
    R: int,
    name: str,
    size: int,
    part: tuple[int, int],
    n: int,
) -> Tensor:
    """x (K, M, N) against ``n`` consecutive parts (K, rows) of each decoded slot of ``name`` (``size`` elements a
    slot), each decoded and multiplied apart -> (n rows, M, N) f32."""
    E = lg.d.experts
    K, rows = part
    M, N = x.shape[1], x.shape[2]
    unit = K * rows // 2048
    outs = []
    for k in range(n):
        dec = lg.tensor((K * rows, N), F8)
        lg.node(
            "rg_mx4_decs_f32",
            [lg.input(name, (size // 2, E), U8), lg.input(name + ".s", (size // 32, E), U8), ids,
             lg.input(name + ".k", (E,), I32)],
            [dec],
            struct.pack("ii", R, k * unit),
        )  # fmt: skip
        y = lg.tensor((rows, M, N), F32)
        lg.node("batch_gemm", [x, lg.view(dec, (K, rows, N))], [y], GEMM_NT)
        outs.append(y)
    if n == 1:
        return outs[0]
    y = lg.tensor((n * rows, M, N), F32)
    lg.node("concat", outs, [y], struct.pack("I", 0))
    return y


def _w2(lg: LayerGraph, h: Tensor, ids: Tensor, R: int) -> Tensor:
    """h (w, M, N) FP8 against N decoded w2 slots -> (H, M, N) f32; at T <= 4 as two halves of the rows, each
    decoded and multiplied apart."""
    H, w = lg.d.hidden, lg.d.expert
    n = 2 if lg.T <= 4 else 1
    return _halves(lg, h, ids, R, "x2", H * w, (w, H // n), n)


def ffn_graph(device: Device, dims: Dims, tokens: int, paged: bool = False) -> Graph:
    """The MoE sub-layer: the sqrt(softplus) router, the shared expert (FP8, clamped SwiGLU) and the routed
    experts decoded from their MXFP4 slots, combined into ``p_out``: one slot per (token, route), or with
    ``paged`` one per page of an expert's (token, route) rows (``moe_cnt`` / ``moe_pages`` / ``moe_asg``)."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ffn_{'paged_' if paged else ''}T{T}", d, T)
    H, K, E, w = d.hidden, d.top, d.experts, d.expert
    w0 = w // 128 * 128
    S = K * T
    X = lg.residual()
    h16, h32, h8, hs = lg.collapse8(X, lg.input("c", (T, 24), F32), "ffn_norm")
    if (
        T == 1 and not paged
    ):  # one token: the coefficients before the router and shared-expert GEMMs
        lg.site(X, "hcf")
    logits = lg.gemm(h16, "router", E)
    g13 = lg.gemm8(h8, "sh_w13", 2 * w)
    if not paged:
        xq, sx = lg.tensor((H, 1, S), F8), lg.tensor((1, T), F32)
        lg.node("rg_ds_xq_lin_f32", [h32], [xq, sx], struct.pack("i", K))
    ids, wts = lg.tensor((64, T), I32), lg.tensor((64, T), F32)
    lg.node(
        "rg_moe_route_ssp_f32",
        [logits, lg.input("router_bias", (E,), F32), lg.meta()],
        [ids, wts],
        struct.pack("iif", K, E, d.route_scale),
    )
    a8, shs = lg.q8p(g13, sin=hs, swiglu=True)
    sh = lg.gemm8(a8, "sh_w2", H)
    if paged:
        _paged_experts(lg, h32, ids, wts, sh, shs)
        lg.site(X, "hcf")
        return lg.g
    y13 = _w13(lg, xq, ids, K)
    hq, hsl = lg.tensor((w, 1, S), F8), lg.tensor((1, S), F32)
    lg.node(
        "rg_ds_xact_lin_f32",
        [y13, y13, lg.input("x13.c", (2 * w, E), F32), sx, ids],
        [hq, hsl],
        struct.pack("ifii", K, d.limit, w - w0, 0),
    )
    y2 = _w2(lg, hq, ids, K)
    if (
        T > 1
    ):  # the next sub-layer's coefficients, created late
        lg.site(X, "hcf")
    lg.node(
        "rg_ds_xsum_f32",
        [y2, lg.input("x2.c", (H, E), F32), hsl, ids, wts, sh, shs],
        [lg.p_out()],
        struct.pack("i", K),
    )
    return lg.g


def _paged_experts(
    lg: LayerGraph, h32: Tensor, ids: Tensor, wts: Tensor, sh: Tensor, shs: Tensor
) -> None:
    """Each expert a step routes to, decoded once per page of its (token, route) rows; FP8 batch_gemms over
    the pages; the weighted combine with the shared expert."""
    from resin_models_gaudi.glm5_3_flash.layers import paged_shape

    d, T = lg.d, lg.T
    H, K, E, w = d.hidden, d.top, d.experts, d.expert
    w0 = w // 128 * 128
    psh, P, S = paged_shape(T, K, E)
    PS = 1 << psh
    meta = lg.meta()
    cntm = lg.tensor((E, 24), I32)
    lg.node("rg_moe_cnt_f32", [ids, meta], [cntm], struct.pack("ii", 0, K))
    po, pe = lg.tensor((E,), I32), lg.tensor((P,), I32)
    lg.node("rg_moe_pages_f32", [cntm], [po, pe], struct.pack("i", psh))
    gidx, bm = lg.tensor((S,), I32), lg.tensor((64, T), I32)
    lg.node(
        "rg_moe_asg_f32",
        [ids, meta, cntm, po, pe],
        [gidx, bm],
        struct.pack("iii", 0, K, psh),
    )
    # each token quantized once, then its FP8 row copied to its pooled rows
    q1, s1 = lg.tensor((H, T), F8), lg.tensor((1, T), F32)
    lg.node("rg_ds_xq1_lin_f32", [h32], [q1, s1])
    xq, sxr = lg.tensor((H, S), F8), lg.tensor((1, S), F32)
    lg.node("rg_ds_xg_f32", [q1, s1, gidx], [xq, sxr])
    y13 = _w13(lg, lg.view(xq, (H, PS, P)), pe, 0)
    yf = lg.view(y13, (2 * w, 1, S))
    hq, hs = lg.tensor((w, 1, S), F8), lg.tensor((1, S), F32)
    if w == 288:  # one member per page
        lg.node(
            "rg_ds_xact_pg_f32",
            [yf, lg.input("x13.c", (2 * w, E), F32), sxr, pe],
            [hq, hs],
            struct.pack("fi", d.limit, psh),
        )
    else:
        lg.node(
            "rg_ds_xact_lin_f32",
            [yf, yf, lg.input("x13.c", (2 * w, E), F32), sxr, pe],
            [hq, hs],
            struct.pack("ifii", 0, d.limit, w - w0, psh),
        )
    d2 = lg.tensor((H * w, P), F8)
    lg.node(
        "rg_mx4_decs_f32",
        [
            lg.input("x2", (H * w // 2, E), U8),
            lg.input("x2.s", (H * w // 32, E), U8),
            pe,
            lg.input("x2.k", (E,), I32),
        ],
        [d2],
        struct.pack("ii", 0, 0),
    )
    y2 = lg.tensor((H, PS, P), F32)
    lg.node(
        "batch_gemm", [lg.view(hq, (w, PS, P)), lg.view(d2, (w, H, P))], [y2], GEMM_NT
    )
    lg.node(
        "rg_ds_xcomb_f32",
        [lg.view(y2, (H, S)), lg.input("x2.c", (H, E), F32), hs, ids, bm, wts, sh, shs],
        [lg.p_out()],
        struct.pack("i", K),
    )


def draft_attention_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    keys: int,
    arena_blocks: int,
    block: int,
    *,
    residual: bool = True,
) -> Graph:
    """A DSpark stage's attention over draft blocks of ``block`` columns: each column's queries against its
    window part (slots of one column, the stage's window pages up to the block's anchor) and against its
    block's ``block`` keys (``dsp_bidx`` (block, T): the columns of each column's block; ``ix_lslots`` the
    count per column), in one softmax with the sink; the step's keys are not stored (``rows`` all -1). Without
    ``residual`` the graph starts from the embedded block ``X`` (the first stage)."""
    T = tokens
    lg = LayerGraph(
        device,
        f"dsp_attn_T{T}_S{slots}_C{keys}_K{block}_A{arena_blocks}"
        + ("" if residual else "_x0"),
        dims,
        T,
    )
    got: list[Tensor] = []
    attention_pre_graph(
        device, dims, T, slots, keys, arena_blocks, ratio=0, owner=False, source=False, groups=T,
        residual=residual, query_block=1, into=lg, keys_out=got.append,
    )  # fmt: skip
    D = dims.head_dim
    kb = lg.tensor((D, block * T), BF16)
    lg.node(
        "gather_fwd_bf16",
        [lg.cast16(got[0]), lg.input("dsp_bidx", (block * T,), I32)],
        [kb],
        struct.pack("i", 1),
    )
    attention_out_graph(
        device, dims, T, arena_blocks, 0, into=lg, ks=lg.view(kb, (D, block, T))
    )
    return lg.g


def inject_graph(
    device: Device, dims: Dims, tokens: int, stages: int, arena_blocks: int, taps: int
) -> Graph:
    """The DSpark stages' window keys at the step's columns: main_x = main_norm(main_proj([the ``taps`` target
    layers' stream means ``dsp_main{k}`` (H, T)])), then per stage s its key kv_norm{s}(wkv{s}(main_x)) with
    RoPE at each column's position (``cs``), stored by ``rows{s}``."""
    d, T, H, D = dims, tokens, dims.hidden, dims.head_dim
    lg = LayerGraph(device, f"dsp_inject_T{T}_A{arena_blocks}", d, T)
    cat = lg.tensor((taps * H, T), F32)
    lg.node(
        "concat",
        [lg.input(f"dsp_main{k}", (H, T), F32) for k in range(taps)],
        [cat],
        struct.pack("I", 0),
    )
    x8, sx = lg.q8p(cat)
    main = lg.norm(lg.gemm8(x8, "main_proj", H, sx), "main_norm")
    m8, sm = lg.q8p(main)
    cs, arena = lg.input("cs", (128, T), F32), lg.arena(arena_blocks)
    for k in range(stages):
        kv = lg.ds_rope(lg.gemm8(m8, f"wkv{k}", D, sm), cs, 1, norm=f"kv_norm{k}")
        lg.node(
            "rg_mla_store_p2_f32",
            [arena, lg.input(f"rows{k}", (8, T), I32), kv, lg.input("ix_none", (128, T), BF16)],
            [lg.input(f"dsp_sink{k}", (64, 1), I32)],  # a persistent sink
        )  # fmt: skip
    return lg.g


def markov_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    rows: int,
    block: int,
    ranks: int,
    vocab: int,
) -> Graph:
    """DSpark's draft tokens from the stage head's logits ``logits_all`` (V_r, T, ranks) (row b's block in
    columns b block ..): ``block`` steps, each adding the Markov head's bias for the previous token
    (``mk_head`` (V, rank) x ``mk_embed`` row; the first previous token is ``dsp_anchor`` (rows)) to the
    block's next column and taking the argmax -> ``dsp_out`` (block rows)."""
    d, V = dims, vocab
    lg = LayerGraph(
        device, f"dsp_markov_T{tokens}_R{rows}_K{block}_N{ranks}", d, tokens
    )
    logits = lg.input("logits_all", (d.vocab, tokens, ranks), F32)
    emb = lg.input("mk_embed", (MARKOV_RANK, V), BF16)
    ids = lg.input("dsp_anchor", (rows,), I32)
    outs = []
    for i in range(block):
        e = lg.tensor((MARKOV_RANK, rows), BF16)
        lg.node("gather_fwd_bf16", [emb, ids], [e], struct.pack("i", 1))
        z = lg.tensor((V, rows), F32)
        lg.node(
            "rg_ds_mk_add_f32",
            [logits, lg.gemm(e, "mk_head", V)],
            [z],
            struct.pack("ii", i, block),
        )
        lanes, ids = lg.tensor((64, rows), I32), lg.tensor((rows,), I32)
        lg.node("rg_ds_argmax_f32", [z], [lanes, ids], struct.pack("i", V))
        outs.append(ids)
    lg.node(
        "concat", outs, [lg.input("dsp_out", (block * rows,), I32)], struct.pack("I", 0)
    )
    return lg.g


def head_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """The last ffn's update, the collapse with its pre-mix, the final norm and this rank's vocabulary slice ->
    logits (V_r, T)."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"head_T{T}", d, T)
    X = lg.residual()
    h16, _ = lg.collapse(X, lg.input("c", (T, 24), F32), "norm", mode=0)
    lg.gemm(h16, "head", d.vocab, out=lg.input("logits", (d.vocab, T), F32))
    return lg.g


__all__ = [
    "Dims",
    "LayerGraph",
    "NS",
    "attention_out_graph",
    "attention_pre_graph",
    "attention_window_graph",
    "engram_gather_graph",
    "engram_graph",
    "embed_graph",
    "ffn_graph",
    "head_graph",
    "index_score_graph",
    "index_select_graph",
]


# ------------------------------------------------------------------ compressed layers (ratio 1 or 2)
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOP = 512
MARKOV_RANK = 256  # DSpark's Markov head width
INDEX_SCALE = 1.0 / 64  # index_head_dim^-0.5 * index_heads^-0.5


def attention_pre_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    keys: int,
    blocks: int,
    *,
    ratio: int,
    owner: bool,
    source: bool,
    groups: int = 0,
    state_rows: int = 1,
    residual: bool = True,
    query_block: int = 0,
    into: LayerGraph | None = None,
    after_collapse=None,
    capture: bool = False,
    keys_out=None,
) -> Graph:
    """A compressed layer's first launch: the residual, site and collapse, the queries (``q_all``), the window keys'
    store, the window part of the attention (``o1``, ``m1``, ``l1``), and

    - on a KV owner, the compressor: ``groups`` latents (ratio 2: pooled pairs, ``cmp_groups`` (4, G), with the
      pending tokens in ``cmp_state`` (1024, R) and ``cmp_rows`` (T); ratio 1: one per column), normed, their
      index keys (``ix_wk``, ``ix_knorm``, RoPE, FP4) and the latents (RoPE, FP4) stored with ``cmp_store`` (8, G);
      ``cs_g`` (128, G) holds each latent's rotary table (the position of its first token);
    - on an index source, the index queries ``ix_q`` (128, 32 T) bf16 (RoPE, FP4) and head weights ``ix_w`` (32, T).

    With ``capture``, the mean of the incoming streams goes to ``dsp_main`` (H, T) (a DSpark target layer);
    ``keys_out`` receives the step's keys (head_dim, T) f32.
    """
    d, T, S, C, G = dims, tokens, slots, keys, groups
    qb = query_block or T
    tag = ("o" if owner else "") + ("i" if source else "")
    lg = into or LayerGraph(
        device,
        f"attn_pre_T{T}_S{S}_C{C}_Q{qb}_r{ratio}{tag}_G{G}_R{state_rows}"
        + ("" if residual else "_eg"),
        d,
        T,
    )
    D, nh, H = d.head_dim, d.heads, d.hidden
    W, M = nh * T, nh * qb
    Mp = -(-M // 64) * 64
    X = lg.residual() if residual else lg.input("X", (NS * H, T), F32)
    if capture:
        lg.node("rg_ds_smean_f32", [X], [lg.input("dsp_main", (H, T), F32)])
    h16, h32, h8, _ = lg.collapse8(X, lg.input("c", (T, 24), F32), "attn_norm")
    if T == 1:  # one token: the coefficients before the first GEMM
        lg.site(X, "hca")
    if after_collapse is not None:  # nodes before the first GEMM
        after_collapse(lg)
    qa, kva = lg.split0(lg.gemm8(h8, "w_a", d.q_lora + D), [d.q_lora, D])
    # the keys first
    cs = lg.input("cs", (128, T), F32)
    kn = lg.ds_rope(kva, cs, 1, norm="kv_norm")
    if keys_out is not None:
        keys_out(kn)
    arena = lg.arena(blocks)
    stored = lg.tensor((64, 1), I32)
    lg.node(
        "rg_mla_store_p2_f32",
        [arena, lg.input("rows", (8, T), I32), kn, lg.input("ix_none", (128, T), BF16)],
        [stored],
    )
    qr8, sq = lg.q8p(qa, norm="q_norm")
    q = lg.gemm8(qr8, "wq_b", nh * D)
    q16 = lg.ds_rope(
        lg.view(q, (D, W)), cs, nh, ts=sq, out=lg.input("q_all", (D, W), BF16)
    )
    table = lg.input("slots", (8, S), I32)
    lp, rp = (
        lg.input("lat_pages", (C // PAGE, S), I32),
        lg.input("rope_pages", (C // PAGE, S), I32),
    )
    qs = lg.view(
        q16, (D, M, S)
    )  # slot s holds columns M s .. M s + M - 1 (the host's slot order)
    ks = lg.tensor((D, C, S), BF16)
    lg.node(
        "rg_mla_gather_f32", [arena, lp, rp, table, stored], [ks], struct.pack("i", 0)
    )
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    P, mx, ls = (
        lg.tensor((C, M, S), BF16),
        lg.tensor((Mp, S), F32),
        lg.tensor((Mp, S), F32),
    )
    lg.node(
        "rg_ds_slot_pw1_f32"
        if M <= 64
        else "rg_ds_slot_pw_f32",  # few columns: one member each
        [sc, table],
        [P, mx, ls],
        struct.pack("fi", d.scale, nh),
    )
    pv = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    lg.node(
        "rg_ds_slot_ml1_f32"
        if W <= 32
        else "rg_ds_slot_ml_f32",  # few columns: one member each
        [pv, mx, ls, table],
        [
            lg.input("o1", (D, W, 1), F32),
            lg.input("m1", (W, 1), F32),
            lg.input("l1", (W, 1), F32),
        ],
    )
    if owner:
        kvc = lg.tensor((D, T), F32)
        lg.node("gemm", [h32, lg.input("cmp_wkv", (H, D), F32)], [kvc], GEMM_NT)
        if ratio > 1:
            gate = lg.tensor((D, T), F32)
            lg.node("gemm", [h32, lg.input("cmp_wgate", (H, D), F32)], [gate], GEMM_NT)
            lat = lg.tensor((D, G), F32)
            state = lg.input("cmp_state", (2 * D, state_rows), F32)
            lg.node(
                "rg_ds_cpool_f32",
                [kvc, gate, state, lg.input("cmp_groups", (4, G), I32)],
                [lat],
            )
            lg.node(
                "rg_ds_cstate_f32",
                [kvc, gate, lg.input("cmp_rows", (T,), I32), state, lat],
                [lg.input("cmp_sink", (64, 1), I32)],
            )
        else:
            lat = kvc
        latn = lg.norm(lat, "cmp_norm")
        cs_g = lg.input("cs_g", (128, G), F32)
        k = lg.tensor((INDEX_DIM, G), F32)
        lg.node(
            "gemm",
            [lg.cast16(latn), lg.input("ix_wk", (D, INDEX_DIM), BF16)],
            [k],
            GEMM_NT,
        )
        ix32 = lg.fq4(lg.ds_rope(k, cs_g, 1, norm="ix_knorm"), 0)
        lq = lg.fq4(lg.rope(latn, cs_g, 1), 1)
        lg.node(
            "rg_mla_store_i8_f32",
            [arena, lg.input("cmp_store", (8, G), I32), lq, ix32],
            [lg.input("cmp_stored", (64, 1), I32)],
        )
    if source:
        qi = lg.gemm8(qr8, "ix_wq_b", INDEX_HEADS * INDEX_DIM)
        qi = lg.ds_rope(
            lg.view(qi, (INDEX_DIM, INDEX_HEADS * T)), cs, INDEX_HEADS, ts=sq
        )
        W = INDEX_HEADS * T
        lg.node(
            "rg_ix_q8_f32",
            [lg.fq4(qi, 0)],
            [lg.input("ix_q8", (INDEX_DIM, W), F8), lg.input("ix_qs", (1, W), F32)],
        )
        lg.keep(lg.gemm(h16, "ix_wproj", INDEX_HEADS), "ix_w")
    if (
        T > 1
    ):  # the next sub-layer's coefficients, created last
        lg.site(X, "hca")
    return lg.g


def index_score_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    keys: int,
    blocks: int,
    scores: int,
    query_block: int = 0,
    cand_top: int = 0,
    pages: int = 0,
    parts: int = 1,
) -> Graph:
    """An index source's scores: each slot is a query block's index queries against a chunk of ``keys`` index keys
    (``ix_pages`` (keys / 128, S): first rows of the index pages; ``ix_slots`` (8, S) with the compress ratio in
    field 6) -> ``ix_score`` (scores, T), written in place; ``ix_sink`` is its sink. With ``cand_top``, a slot's
    keys are those of 256 of the column's ``cand_top`` candidate blocks (``cand_list``, ``cand_slots``; the
    column's ``cmp_blocks`` (pages, T) and ``cmp_loff``), and their positions go to ``ix_cpos`` (scores, T)."""
    d, T, S, C = dims, tokens, slots, keys
    qb = query_block or T
    # prefill slots: each member reads its tokens' head weights once for 1024 keys, and the slots run in parts
    # (own tables, GEMM and accumulation)
    prefill = qb % 8 == 0 and C % 1024 == 0 and not cand_top
    P = parts if prefill and parts > 1 and S % parts == 0 else 1
    name = (
        f"index_score_T{T}_S{S}_C{C}_Q{qb}_K{scores}"
        + ("_cand" if cand_top else "")
        + (f"_p{P}" if P > 1 else "")
    )
    lg = LayerGraph(device, name, d, T)
    M = INDEX_HEADS * qb
    W = INDEX_HEADS * T
    Sp = S // P
    sfx = [f".p{p}" if P > 1 else "" for p in range(P)]
    q8, qsc = lg.input("ix_q8", (INDEX_DIM, W), F8), lg.input("ix_qs", (W,), F32)
    w, score = (
        lg.input("ix_w", (INDEX_HEADS, T), F32),
        lg.input("ix_score", (scores, T), F32),
    )
    tables = [lg.input("ix_slots" + x, (8, Sp), I32) for x in sfx]
    inputs = []
    for p in range(P):  # every part's gathers first
        table = tables[p]
        qs, sq = lg.tensor((INDEX_DIM, M, Sp), F8), lg.tensor((M, Sp), F32)
        lg.node("rg_ix_slot_q8_f32", [q8, qsc, table], [qs, sq])
        ks2, sk = (
            lg.tensor((2 * INDEX_DIM, C // 2, Sp), F8),
            lg.tensor((C, Sp), F32),
        )  # two keys a row
        if cand_top:
            lg.node(
                "rg_ix_cgather_f32",
                [lg.arena(blocks), lg.input("cand_list", (cand_top, T), I32), lg.input("cand_slots", (8, T), I32),
                 lg.input("cmp_blocks", (pages, T), I32), lg.input("cmp_loff", (64,), I32), table, lg.dep()],
                [ks2, sk, lg.input("ix_cpos", (scores, T), I32)],
                struct.pack("i", PAGE_ROWS),
            )  # fmt: skip
        else:
            lg.node(
                "rg_ds_ix_gather_f32",
                [
                    lg.arena(blocks),
                    lg.input("ix_pages" + sfx[p], (C // PAGE, Sp), I32),
                    table,
                    lg.dep(),
                ],
                [ks2, sk],
            )
        inputs.append((table, qs, sq, lg.view(ks2, (INDEX_DIM, C, Sp)), sk))
    scs = []
    for (
        table,
        qs,
        sq,
        ks,
        sk,
    ) in inputs:  # (prefill: each head's product rounded to bf16, as the reference's)
        sc = lg.tensor((C, M, Sp), BF16 if prefill else F32)
        lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
        scs.append(sc)
    for p, (sc, (table, qs, sq, ks, sk)) in enumerate(zip(scs, inputs, strict=True)):
        lg.node(
            "rg_ds_ix_qacc_f32" if prefill else "rg_ds_ix_slot_acc8_f32",
            [sc, w, table, score, sq, sk],
            [lg.input("ix_sink" + sfx[p], (64, 1), I32)],
            struct.pack("f", INDEX_SCALE),
        )
    return lg.g


def index_select_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    scores: int,
    cand: str = "",
    cand_top: int = 0,
    radix: bool = False,
    sparse: bool = False,
    cand_blocks: int = 0,
    prefilter: int = 0,
) -> Graph:
    """The top-512 compressed entries of each column: ``ix_list`` (512, T) and the sparse slots ``ix_lslots`` (8, T),
    from ``ix_score`` and the visible counts ``ix_vis`` (T). With ``cand`` "source", also the column's candidate
    blocks (the top ``cand_top`` blocks of 8 entries, over ``ix_visb`` (T) padded block counts) as ``cand_bmax``
    (cand_blocks, T) and ``cand_thr`` (64, T), and with ``sparse`` their list ``cand_list`` (cand_top, T) and
    ``cand_slots``; with "user", the top-512 of the entries in those blocks, and with ``sparse`` from the compact
    scores of those entries (``ix_score`` (scores, T), their positions ``ix_cpos``, ``ix_visc`` (T) entries a
    column has). With ``radix``, the 512th key comes from 8 radix passes over all cores (ix_rcnt, ix_rsel)
    instead of ix_thr's bisection on one core per column. With ``prefilter`` (a prefill step whose every column sees
    more than 8 ``prefilter`` entries), the top-512 come from the entries of the column's ``prefilter`` blocks of 8
    with the largest maxima (every entry of the top 512 is in one of them): ix_bmax over the scores (over ``ix_visb``
    (T) padded block counts), those blocks listed (ix_list), their scores packed (ix_bgather) with their positions,
    then the selection over the packed scores (``ix_visc`` (T) of them)."""
    d, T, K = dims, tokens, scores
    name = (
        f"index_select_T{T}_K{K}"
        + (f"_{cand}" if cand else "")
        + ("_radix" if radix else "")
        + ("_sparse" if sparse else "")
        + (f"_pf{prefilter}" if prefilter else "")
    )
    lg = LayerGraph(device, name, d, T)
    score, vis = lg.input("ix_score", (K, T), F32), lg.input("ix_vis", (T,), I32)
    KB = cand_blocks or K // 8
    cpos = None
    if cand == "user" and sparse:
        cpos = lg.input("ix_cpos", (K, T), I32)
        masked = lg.tensor((K, T), F32)
        lg.node("rg_ix_cfix_f32", [score, cpos, vis], [masked])
        score, vis = masked, lg.input("ix_visc", (T,), I32)
    elif cand == "user":
        masked = lg.tensor((K, T), F32)
        lg.node(
            "rg_ix_cmask_f32",
            [score, vis, lg.input("cand_bmax", (KB, T), F32), lg.input("cand_thr", (64, T), I32)],
            [masked],
        )  # fmt: skip
        score = masked
    elif cand == "source":
        bmax, visb = lg.input("cand_bmax", (KB, T), F32), lg.input("ix_visb", (T,), I32)
        cthr = lg.input("cand_thr", (64, T), I32)
        lg.node("rg_ix_bmax_f32", [score, vis], [bmax])
        lg.node("rg_ix_thr_f32", [bmax, visb], [cthr], struct.pack("i", cand_top))
        if sparse:
            cnt, gsum = lg.tensor((KB // 64, T), I32), lg.tensor((KB // 4096, T), I32)
            lg.node("rg_ix_bcnt_f32", [bmax, visb, cthr], [cnt, gsum])
            lg.node(
                "rg_ix_list_f32",
                [bmax, visb, cthr, cnt, gsum],
                [lg.input("cand_list", (cand_top, T), I32), lg.input("cand_slots", (8, T), I32)],
                struct.pack("iii", max(1, 48 // T), cand_top, 0),
            )  # fmt: skip
    if prefilter:
        NBL, KC = prefilter, 8 * prefilter
        if cand == "source":
            bm = bmax
        else:
            visb = lg.input("ix_visb", (T,), I32)
            bm = lg.tensor((KB, T), F32)
            lg.node("rg_ix_bmax_f32", [score, vis], [bm])
        bthr = lg.tensor((64, T), I32)
        lg.node("rg_ix_thr_f32", [bm, visb], [bthr], struct.pack("i", NBL))
        cnt, gsum = lg.tensor((KB // 64, T), I32), lg.tensor((KB // 4096, T), I32)
        lg.node("rg_ix_bcnt_f32", [bm, visb, bthr], [cnt, gsum])
        blist, bslots = lg.tensor((NBL, T), I32), lg.tensor((8, T), I32)
        lg.node(
            "rg_ix_list_f32", [bm, visb, bthr, cnt, gsum], [blist, bslots],
            struct.pack("iii", max(1, 48 // T), NBL, d.heads),
        )  # fmt: skip
        cs, cpos = lg.tensor((KC, T), F32), lg.tensor((KC, T), I32)
        lg.node("rg_ix_bgather_f32", [score, blist], [cs, cpos])
        packed = lg.tensor((KC, T), F32)
        lg.node("rg_ix_cfix_f32", [cs, cpos, vis], [packed])
        score, vis, K = packed, lg.input("ix_visc", (T,), I32), KC
    if radix:
        M = max(1, 24 // T)  # members per column
        thr = lg.input("ix_none", (64, T), I32)  # lo = 0
        for sh in range(28, -1, -4):
            part = lg.tensor((64, M, T), I32)
            params = struct.pack("ii", sh, INDEX_TOP)
            lg.node("rg_ix_rcnt_f32", [score, vis, thr], [part], params)
            nxt = lg.tensor((64, T), I32)
            lg.node("rg_ix_rsel_f32", [part, vis, thr], [nxt], params)
            thr = nxt
    else:  # (the plain lists' threshold is kept for the dense prefill attention's masks)
        thr = (
            lg.input("ix_thr", (64, T), I32)
            if not prefilter and not cand
            else lg.tensor((64, T), I32)
        )
        lg.node("rg_ix_thr_f32", [score, vis], [thr], struct.pack("i", INDEX_TOP))
    cnt, gsum = lg.tensor((K // 64, T), I32), lg.tensor((-(-K // 4096), T), I32)
    lg.node("rg_ix_bcnt_f32", [score, vis, thr], [cnt, gsum])
    lg.node(
        "rg_ix_list_map_f32" if cpos is not None else "rg_ix_list_f32",
        [score, vis, thr, cnt, gsum] + ([cpos] if cpos is not None else []),
        [lg.input("ix_list", (INDEX_TOP, T), I32), lg.input("ix_lslots", (8, T), I32)],
        struct.pack("iii", max(1, 48 // T), INDEX_TOP, d.heads),
    )
    return lg.g


def _gather_listed(lg: LayerGraph, arena_blocks: int, blocks: int) -> Tensor:
    """The listed compressed latents of every column: (head_dim, 512, T) bf16."""
    D, K, T = lg.d.head_dim, INDEX_TOP, lg.T
    ks = lg.tensor((D, K, T), BF16)
    lg.node(
        "rg_ds_gather_list_f32",
        [lg.arena(arena_blocks), lg.input("ix_list", (K, T), I32), lg.input("cmp_blocks", (blocks, T), I32),
         lg.input("cmp_loff", (64,), I32), lg.input("ix_lslots", (8, T), I32), lg.dep()],
        [ks],
    )  # fmt: skip
    return ks


def _gather_dense(lg: LayerGraph, arena_blocks: int, keys: int) -> Tensor:
    """Every visible compressed latent of each query block's slot: (head_dim, keys, slots) bf16, from the owner's
    latent pages ``cmp_lat`` (keys / 128, slots) and the slots ``cslots`` (8, slots)."""
    D, S = lg.d.head_dim, lg.T // QUERY_BLOCK_DENSE
    ks = lg.tensor((D, keys, S), BF16)
    lp = lg.input("cmp_lat", (keys // PAGE, S), I32)
    lg.node(
        "rg_mla_gather_f32",
        [lg.arena(arena_blocks), lp, lp, lg.input("cslots", (8, S), I32), lg.dep()],
        [ks],
        struct.pack("i", 0),
    )
    return ks


def _dense_part(
    lg: LayerGraph, ks: Tensor, scores: int
) -> tuple[Tensor, Tensor, Tensor]:
    """The compressed part over every visible entry of each query block, with each column's list as a mask
    (ds_slot_pk) -> (o2, m2, l2) as the listed part's."""
    d, T = lg.d, lg.T
    D, nh, keys, S = d.head_dim, d.heads, ks.shape[1], ks.shape[2]
    W, M = nh * T, nh * QUERY_BLOCK_DENSE
    table = lg.input("cslots", (8, S), I32)
    qs = lg.view(lg.input("q_all", (D, W), BF16), (D, M, S))  # slot s: columns 128 s ..
    sc = lg.tensor((keys, M, S), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    P, mx, ls = (
        lg.tensor((keys, M, S), BF16),
        lg.tensor((M, S), F32),
        lg.tensor((M, S), F32),
    )
    lg.node(
        "rg_ds_slot_pk_f32",
        [sc, table, lg.input("ix_score", (scores, T), F32), lg.input("ix_thr", (64, T), I32),
         lg.input("ix_lslots", (8, T), I32)],
        [P, mx, ls],
        struct.pack("fi", d.scale, nh),
    )  # fmt: skip
    pv = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    o2, m2, l2 = (
        lg.tensor((D, W, 1), F32),
        lg.tensor((W, 1), F32),
        lg.tensor((W, 1), F32),
    )
    lg.node("rg_ds_slot_ml_f32", [pv, mx, ls, table], [o2, m2, l2])
    return o2, m2, l2


def attention_out_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    arena_blocks: int,
    blocks: int,
    into: LayerGraph | None = None,
    ks: Tensor | None = None,
    dense: int = 0,
    scores: int = 0,
) -> Graph:
    """A compressed layer's last launch: the listed compressed latents (``ix_list``, block rows ``cmp_blocks``
    (blocks, T), offsets ``cmp_loff``), their attention part, merged with the window part (``o1``, ``m1``,
    ``l1``) and the sink, the rotation removed, then wo_a and wo_b -> ``p_out``. With ``dense`` (a prefill step
    whose columns see at most ``dense`` entries), the compressed part attends over every visible entry of each
    query block with the lists as masks (``_dense_part``; ``scores`` rows of ``ix_score``)."""
    d, T = dims, tokens
    lg = into or LayerGraph(
        device,
        f"attn_out_T{T}_B{blocks}_A{arena_blocks}" + (f"_d{dense}" if dense else ""),
        d,
        T,
    )
    D, nh = d.head_dim, d.heads
    W = nh * T
    if dense:
        ks = ks if ks is not None else _gather_dense(lg, arena_blocks, dense)
        o2, m2, l2 = _dense_part(lg, ks, scores)
        _attention_comb(lg, o2, m2, l2)
        return lg.g
    K = INDEX_TOP if ks is None else ks.shape[1]
    lslots = lg.input("ix_lslots", (8, T), I32)
    if ks is None:
        ks = _gather_listed(lg, arena_blocks, blocks)
    qs = lg.view(
        lg.input("q_all", (D, W), BF16), (D, nh, T)
    )  # column c's heads: slot c
    sc = lg.tensor((K, nh, T), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    Mp = -(-nh // 64) * 64
    P, mx, ls = (
        lg.tensor((K, nh, T), BF16),
        lg.tensor((Mp, T), F32),
        lg.tensor((Mp, T), F32),
    )
    lg.node(
        "rg_ds_slot_p1_f32", [sc, lslots], [P, mx, ls], struct.pack("fi", d.scale, nh)
    )
    pv = lg.tensor((D, nh, T), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    o2, m2, l2 = (
        lg.tensor((D, W, 1), F32),
        lg.tensor((W, 1), F32),
        lg.tensor((W, 1), F32),
    )
    # few columns: one member each; else each column's slot is the column (no scan of the slots)
    ml = "rg_ds_slot_ml1_f32" if W <= 32 else "rg_ds_slot_mld_f32"
    lg.node(ml, [pv, mx, ls, lslots], [o2, m2, l2])
    _attention_comb(lg, o2, m2, l2)
    return lg.g


def _attention_comb(lg: LayerGraph, o2: Tensor, m2: Tensor, l2: Tensor) -> None:
    """The compressed part merged with the window part and the sink, the rotation removed, wo_a and wo_b"""
    d, T = lg.d, lg.T
    D, nh, W = d.head_dim, d.heads, d.heads * lg.T
    o = lg.tensor((D, W, 1), F32)
    lg.node(
        "rg_ds_attn_comb_r_f32",  # the merge, then the inverse rotation
        [lg.input("o1", (D, W, 1), F32), lg.input("m1", (W, 1), F32), lg.input("l1", (W, 1), F32), o2, m2, l2,
         lg.input("attn_sink", (nh,), F32), lg.input("cs_inv", (128, T), F32)],
        [o],
        struct.pack("i", nh),
    )  # fmt: skip
    _attention_out(lg, lg.view(o, (D, W)))


def attention_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    keys: int,
    arena_blocks: int,
    blocks: int,
    *,
    ratio: int,
    residual: bool = True,
    query_block: int = 0,
    capture: bool = False,
    dense: int = 0,
    scores: int = 0,
) -> Graph:
    """A compressed layer that neither owns its KV nor sources its index (it reads the index list its source
    left) as one launch: ``attention_pre_graph`` then ``attention_out_graph``, the second part reading the
    first's ``q_all``, ``o1``, ``m1``, ``l1`` (``dense``, ``scores``: see attention_out_graph)."""
    T, qb = tokens, query_block or tokens
    early: list[
        Tensor
    ] = []  # the compressed latents, gathered before the first GEMM
    lg = LayerGraph(
        device,
        f"attn_T{T}_S{slots}_C{keys}_Q{qb}_r{ratio}_B{blocks}_A{arena_blocks}"
        + ("" if residual else "_eg")
        + ("_cap" if capture else "")
        + (f"_d{dense}" if dense else ""),
        dims,
        T,
    )
    gather = (
        (lambda g: early.append(_gather_dense(g, arena_blocks, dense)))
        if dense
        else (lambda g: early.append(_gather_listed(g, arena_blocks, blocks)))
    )
    attention_pre_graph(
        device, dims, T, slots, keys, arena_blocks, ratio=ratio, owner=False, source=False,
        groups=T, residual=residual, query_block=query_block, into=lg, after_collapse=gather, capture=capture,
    )  # fmt: skip
    attention_out_graph(
        device,
        dims,
        T,
        arena_blocks,
        blocks,
        into=lg,
        ks=early[0],
        dense=dense,
        scores=scores,
    )
    return lg.g
