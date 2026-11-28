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

import os
import struct
from dataclasses import dataclass

from resin_models_gaudi.deepseek_v41.cache import CMP_PAGE_ROWS
from resin_models_gaudi.glm5_3_flash.layers import GEMM_NN, GEMM_NT
from resin_models_gaudi.glm5_3_flash.layers import LayerGraph as GlmLayerGraph
from resin_models_gaudi.glm5_3_flash.weights import NS
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, U8, Device

ROPE = 64
WINDOW = 128
PAGE = 128  # tokens per latent page: 128 FP8 rows and one row of their scales
QUERY_BLOCK_DENSE = 128  # prefill columns per slot of the dense compressed attention
DECODE_TOKENS = 16  # steps of up to 16 tokens: a slot merge member per token
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

    COEF_PER_TOKEN = DECODE_TOKENS
    MIX_CHUNKS = DECODE_TOKENS

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
        dt: int = F32,
    ) -> Tensor:
        """FP8 x FP8 -> ``dt`` (f32 by default) with the weight's row scales, times the per-token scale ``s``
        (1, T) of q if given.  ``dt`` is BF16 where the reference's own product is (``fp8_gemm out_dtype=BF16``)
        and the consumer reads a bf16 tile: half the bytes for the same fidelity."""
        y = out if out is not None else self.tensor((M, q.shape[1]), dt)
        w = self.input(name, (q.shape[0], M), F8)
        sa = s if s is not None else self.input("one", (1,), F32)
        self.node(
            "fp8_gemm_f32" if dt == F32 else "fp8_gemm_bf16",
            [q, w, sa, self.input(name + ".s", (M,), F32)],
            [y],
            GEMM_NT,
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


def expert_out_dtype(lg: LayerGraph) -> int:
    """The routed experts' product dtype: BF16, as in the reference (``fp8_gemm`` ``out_dtype=BF16``, added into
    an f32 ``y``), except where the activation kernel cannot read a bf16 tile.

    ``rg_ds_xact_*`` loads 128 rows of the product at a time (a bf16 tensor load is 128 elements; there is no
    64-lane bf16 vector), so the pair of tail rows after ``w0 = w // 128 * 128`` must be exactly 128 rows: that
    is ``t = w - w0`` of 0 or 64. ``w = 576`` (4 ranks) gives 64; ``w = 288`` (8 ranks) gives 32 and keeps f32.
    """
    w = lg.d.expert
    return BF16 if w - w // 128 * 128 in (0, 64) else F32


def _xact_lin(dt: int) -> str:
    """The SwiGLU-and-quantize kernel for an expert product of ``dt`` (``rg_ds_xact_lin_b16_f32`` reads the BF16
    tile 128 rows at a time, ``rg_ds_xact_lin_f32`` the f32 one)."""
    return "rg_ds_xact_lin_b16_f32" if dt == BF16 else "rg_ds_xact_lin_f32"


def _w13(lg: LayerGraph, x: Tensor, ids: Tensor, R: int, dt: int | None = None) -> Tensor:
    """x (H, M, N) FP8 against N decoded w13 slots (``ids``, ``R`` as in mx4_decs) -> (2w, M, N) of ``dt``, each
    slot's 2w rows decoded and multiplied in one part: one ``rg_mx4_decs_f32`` and one ``batch_gemm``, no concat.

    The split form (two per-half dequant+GEMM pairs and a ``cat``) is what the graph compiler turns into four
    bundles and 30 MB of SRAM; one pair is two bundles and 18 MB, with a bit-identical result and a faster
    launch (``test_expert_product_in_parts_equals_whole`` pins the equivalence). The dequant tile itself is
    SRAM-resident either way -- see ``cases/wright/notes/moe-dequant-tile-sram.md``.

    ``dt`` is the product's dtype (``expert_out_dtype``): BF16 halves the product's bytes and is what the
    reference's ``fp8_gemm`` produces, so it costs no fidelity -- the device accumulates in f32 and rounds once
    (``test_slot_product_bf16_is_the_rounded_f32``)."""
    H, w = lg.d.hidden, lg.d.expert
    return _halves(lg, x, ids, R, "x13", 2 * w * H, (H, 2 * w), 1, dt)


def _halves(
    lg: LayerGraph,
    x: Tensor,
    ids: Tensor,
    R: int,
    name: str,
    size: int,
    part: tuple[int, int],
    n: int,
    dt: int | None = None,
) -> Tensor:
    """x (K, M, N) against ``n`` consecutive parts (K, rows) of each decoded slot of ``name`` (``size`` elements a
    slot), each decoded and multiplied apart -> (n rows, M, N) of ``dt`` (F32 by default)."""
    E = lg.d.experts
    K, rows = part
    M, N = x.shape[1], x.shape[2]
    dt = F32 if dt is None else dt
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
        y = lg.tensor((rows, M, N), dt)
        lg.node("batch_gemm", [x, lg.view(dec, (K, rows, N))], [y], GEMM_NT)
        outs.append(y)
    if n == 1:
        return outs[0]
    y = lg.tensor((n * rows, M, N), dt)
    lg.node("concat", outs, [y], struct.pack("I", 0))
    return y


def _w2(lg: LayerGraph, h: Tensor, ids: Tensor, R: int, dt: int | None = None) -> Tensor:
    """h (w, M, N) FP8 against N decoded w2 slots -> (H, M, N) of ``dt``, one dequant and one ``batch_gemm`` (as
    in ``_w13``: the two-half form costs two more bundles for the same tile)."""
    H, w = lg.d.hidden, lg.d.expert
    return _halves(lg, h, ids, R, "x2", H * w, (w, H), 1, dt)


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
    # BF16: the reference's shared expert is a bf16 product too; rg_ds_xsum_b16 reads it 128 rows at a time
    sh = lg.gemm8(a8, "sh_w2", H, dt=BF16)
    if paged:
        _paged_experts(lg, h32, ids, wts, sh, shs)
        lg.site(X, "hcf")
        return lg.g
    dt = expert_out_dtype(lg)
    y13 = _w13(lg, xq, ids, K, dt)
    hq, hsl = lg.tensor((w, 1, S), F8), lg.tensor((1, S), F32)
    lg.node(
        _xact_lin(dt),
        [y13, y13, lg.input("x13.c", (2 * w, E), F32), sx, ids],
        [hq, hsl],
        struct.pack("ifii", K, d.limit, w - w0, 0),
    )
    y2 = _w2(lg, hq, ids, K, BF16)  # always BF16: rg_ds_xsum_b16 reads it 128 rows at a time
    if (
        T > 1
    ):  # the next sub-layer's coefficients, created late
        lg.site(X, "hcf")
    lg.node(
        "rg_ds_xsum_b16_f32",
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
    psh, P, S = paged_shape(T, K, E, min_ps=64)
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
    dt = expert_out_dtype(lg)
    y13 = _w13(lg, lg.view(xq, (H, PS, P)), pe, 0, dt)
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
            _xact_lin(dt),
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
    y2 = lg.tensor((H, PS, P), BF16)  # always BF16: rg_ds_xcomb_b16 reads it 128 rows at a time
    lg.node(
        "batch_gemm", [lg.view(hq, (w, PS, P)), lg.view(d2, (w, H, P))], [y2], GEMM_NT
    )
    lg.node(
        "rg_ds_xcomb_b16_f32",
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
    "attention_out_tile_graph",
    "attention_out_wo_graph",
    "attention_pre_graph",
    "attention_window_graph",
    "ATTN_TILE_Q",
    "ATTN_KEY_BLK",
    "ATTN_BLOCKS",
    "engram_gather_graph",
    "engram_graph",
    "embed_graph",
    "ffn_graph",
    "head_graph",
    "index_block_select_graph",
    "index_blocks_graph",
    "index_mask_graph",
    "index_merge_graph",
    "index_score_graph",
    "index_select_graph",
    "INDEX_TILE",
    "INDEX_RUN",
    "INDEX_BLOCKS",
    "BLOCK_TILE",
]


# ------------------------------------------------------------------ compressed layers (ratio 1 or 2)
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOP = 512
MARKOV_RANK = 256  # DSpark's Markov head width
INDEX_SCALE = 1.0 / 64  # index_head_dim^-0.5 * index_heads^-0.5


def _compress_store(
    lg: LayerGraph, arena, h32, *, ratio: int, groups: int, state_rows: int
) -> None:
    """A KV owner's compressor on the collapsed h32 (H, T): ``groups`` latents, normed, their index keys and the
    latents (RoPE, FP4) stored with ``cmp_store`` (see attention_pre_graph)."""
    D, H, T, G = lg.d.head_dim, lg.d.hidden, lg.T, groups
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
    ix_rope = lg.ds_rope(k, cs_g, 1, norm="ix_knorm")
    lat_rope = lg.rope(latn, cs_g, 1)
    # true fp4 codes + group scales (the store quantizes; no fq4 round-trip re-quantization)
    lg.node(
        "rg_ds_store_f4_f32",
        [arena, lg.input("cmp_store", (8, G), I32), lat_rope, ix_rope],
        [lg.input("cmp_stored", (64, 1), I32)],
    )


def owner_store_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    *,
    ratio: int,
    groups: int,
    state_rows: int = 1,
) -> Graph:
    """Bounded replay's cut: a KV owner's first launch reduced to its compressor. The residual (``X_out``), the
    collapse, and the latents and index keys of every column stored as attention_pre_graph stores them; no
    queries, window keys or attention. The layers after the owner then run only on the kept columns."""
    lg = LayerGraph(
        device, f"owner_store_T{tokens}_r{ratio}_G{groups}_R{state_rows}", dims, tokens
    )
    X = lg.residual()
    _, h32, _, _ = lg.collapse8(X, lg.input("c", (tokens, 24), F32), "attn_norm")
    _compress_store(
        lg, lg.arena(blocks), h32, ratio=ratio, groups=groups, state_rows=state_rows
    )
    return lg.g


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
    store_out=None,
    window_attn: bool = True,
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
    # the window store's marker: when window_attn=True the window gather consumes it; when False (the bounded
    # stream does its own window blocks) it must be a PERSISTENT output, else Synapse DCE removes the store
    # (no consumer) and the window keys are never written to the arena.
    stored = lg.tensor((64, 1), I32) if window_attn else lg.input("win_stored", (64, 1), I32)
    lg.node(
        "rg_mla_store_p2_f32",
        [arena, lg.input("rows", (8, T), I32), kn, lg.input("ix_none", (128, T), BF16)],
        [stored],
    )
    if store_out is not None:
        store_out(stored)
    qr8, sq = lg.q8p(qa, norm="q_norm")
    q = lg.gemm8(qr8, "wq_b", nh * D)
    q16 = lg.ds_rope(
        lg.view(q, (D, W)), cs, nh, ts=sq, out=lg.input("q_all", (D, W), BF16)
    )
    if window_attn:  # the window attention part (o1/m1/l1); the bounded stream does its own window blocks
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
        if qb == 1:  # a slot per column
            ml = _slot_merge(W, T)
        else:  # few columns: one member each; else a scan of the slots
            ml = "rg_ds_slot_ml1_f32" if W <= 32 else "rg_ds_slot_ml_f32"
        lg.node(
            ml,
            [pv, mx, ls, table],
            [
                lg.input("o1", (D, W, 1), F32),
                lg.input("m1", (W, 1), F32),
                lg.input("l1", (W, 1), F32),
            ],
        )
    if owner:
        _compress_store(lg, arena, h32, ratio=ratio, groups=G, state_rows=state_rows)
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
    cpages: int = 0,
) -> Graph:
    """An index source's scores: each slot is a query block's index queries against a chunk of ``keys`` index keys
    (``ix_slots`` (8, S), the compress ratio in field 6, the tile's first index page in field 7) -> ``ix_score``
    (scores, T), written in place; ``ix_sink`` is its sink. The slot's page rows come from the step's own
    compressed-page table (``cmp_blocks`` (cpages, T)) plus its owner's latent offset (``cmp_loff``), so no
    per-slot page table exists. With ``cand_top``, a slot's keys are those of 256 of the column's ``cand_top``
    candidate blocks (``cand_list``, ``cand_slots``), and their positions go to ``ix_cpos`` (scores, T)."""
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
        )  # two keys a row; sk = 2^k, the per-key tensor scale restored by qacc
        codes = lg.tensor((INDEX_DIM // 2, C, Sp), U8)
        scales = lg.tensor((INDEX_DIM // 32, C, Sp), U8)
        ktab = lg.tensor((C, Sp), I32)  # per-key k = maxS-132 (mx4_decs row scale)
        ids = lg.tensor((C, Sp), I32)  # identity row map (c + C*s) for mx4_decs R=0
        if cand_top:
            lg.node(
                "rg_ix_cgather_f4_f32",
                [lg.arena(blocks), lg.input("cand_list", (cand_top, T), I32), lg.input("cand_slots", (8, T), I32),
                 lg.input("cmp_blocks", (pages, T), I32), lg.input("cmp_loff", (64,), I32), table, lg.dep()],
                [codes, scales, sk, ktab, ids, lg.input("ix_cpos", (scores, T), I32)],
                struct.pack("i", CMP_PAGE_ROWS),
            )  # fmt: skip
        else:
            lg.node(
                "rg_ds_ix_gather_f4_f32",
                [
                    lg.arena(blocks),
                    lg.input("cmp_blocks", (cpages, T), I32),
                    table,
                    lg.dep(),
                    lg.input("cmp_loff", (64,), I32),
                ],
                [codes, scales, sk, ktab, ids],
                struct.pack("i", CMP_PAGE_ROWS),
            )
        # fp4 -> fp8 NORMALIZED by the per-key tensor scale k (mx4_decs, per-row k from ktab): the fp8
        # code is code*2^(S-127-k) (< 240), and qacc restores 2^k via sk.  k=0 would saturate large scales.
        codes2 = lg.tensor((INDEX_DIM // 2, C * Sp), U8)
        scales2 = lg.tensor((INDEX_DIM // 32, C * Sp), U8)
        ktab2 = lg.tensor((C * Sp,), I32)
        ids2 = lg.tensor((C * Sp,), I32)
        dec = lg.tensor((INDEX_DIM, C * Sp), F8)
        lg.node("reshape", [codes], [codes2])
        lg.node("reshape", [scales], [scales2])
        lg.node("reshape", [ktab], [ktab2])
        lg.node("reshape", [ids], [ids2])
        lg.node("rg_mx4_decs_f32", [codes2, scales2, ids2, ktab2], [dec], struct.pack("ii", 0, 0))
        lg.node("reshape", [dec], [ks2])
        inputs.append((table, qs, sq, lg.view(ks2, (INDEX_DIM, C, Sp)), sk))
    scs = []
    for p, (
        table,
        qs,
        sq,
        ks,
        sk,
    ) in enumerate(inputs):  # (prefill: each head's product rounded to bf16, as the reference's)
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
            [score, vis, lg.input("cand_bmax", (KB, T), F32), lg.input("cand_thr", (64, T), I32),
             lg.input("ix_gbase", (1,), I32)],
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
    # ix_bcnt writes whole 64-block groups of a column, so the counts cover every group of 4096 rows
    groups = -(-K // 4096)
    cnt, gsum = lg.tensor((64 * groups, T), I32), lg.tensor((groups, T), I32)
    lg.node("rg_ix_bcnt_f32", [score, vis, thr], [cnt, gsum])
    lg.node(
        "rg_ix_list_map_f32" if cpos is not None else "rg_ix_list_f32",
        [score, vis, thr, cnt, gsum] + ([cpos] if cpos is not None else []),
        [lg.input("ix_list", (INDEX_TOP, T), I32), lg.input("ix_lslots", (8, T), I32)],
        struct.pack("iii", max(1, 48 // T), INDEX_TOP, d.heads),
    )
    return lg.g


INDEX_TILE = 2048  # index keys per tile (one launch's chunk of a slot)
INDEX_RUN = INDEX_TOP  # the rows of a tile group's running candidates (the top entries so far)
# The merge can run as ONE kernel (three reductions per listed entry: one to find the next entry and two to
# broadcast the position and the score at the chosen lane to the slot's lane) or as TWO nodes (`ds_ix_lanes` +
# `ds_ix_emit`: one reduction, then the emit picks the source lane with a scalar, which needs the lane to cross a
# node boundary). MEASURED at 93k, T=2048: one kernel 50.1 s / 1864.7 tok/s, two nodes 71.1 s / 1315 tok/s -- the
# split loses, because five scalar memory ops per entry cost far more than the two reductions they replace (a
# scalar load or store is hundreds of cycles and the entry loop is serial, a `RED` is six vector ops). The split
# stays behind this switch as the instrument for the head-reduction work, and off by default.
IX_SPLIT_MERGE = bool(int(os.environ.get("RESIN_IX_SPLIT_MERGE", "0")))
INDEX_BLOCKS = 2048  # candidate blocks of 8 entries (candidate_topk_blocks)
BLOCK_TILE = INDEX_TILE // 8  # blocks of 8 entries in a tile


def index_merge_graph(
    device: Device, dims: Dims, tokens: int, rows: int, run: int, heads: int = 0
) -> Graph:
    """A tile group's merge: the top ``run`` of the running candidates (rows 0 .. run of ``ix_merge``) and the
    group's own scores (rows run .. run + rows, written by ``index_score_graph``) -> the compacted running set
    (``ix_sout``, ``ix_pos_out``) and the column's list (``ix_list``, ``ix_lslots``).

    ``ix_gvis`` (T,) is the group's visible rows (``run`` plus the group's visible keys), ``ix_gcnt`` (T,) the
    column's visible entries (at most ``run``) and ``ix_gbase`` (1,) the group's first key: the merge lists the
    top ``run`` exactly as one pass over every key would (see ds_ix_merge.c), so the tiling leaves the full
    selection unchanged. With ``heads`` 0 the list is a block list (the candidate blocks of 8 entries).

    The merge reads one (run + rows, T) score buffer (``ix_merge``) whose rows 0 .. run - 1 are the running
    candidates, because its visible-rows count is a prefix of the row index. ``ix_sout`` is declared
    (run + rows, T) because it is bound to the *next* group's ``ix_merge`` and must carry that buffer's column
    stride; the caller places the group's own (rows, T) scores at rows run .. of ``ix_merge`` itself (a
    (rows, T) tensor's column stride is rows, not run + rows, so that cannot be a straight upload)."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ix_merge_T{T}_R{run}_{rows}", d, T)
    K = run + rows
    score = lg.input("ix_merge", (K, T), F32)
    seen, base = lg.input("ix_vis", (T,), I32), lg.input("ix_gbase", (1,), I32)
    vis, exact = lg.tensor((T,), I32), lg.tensor((T,), I32)
    lg.node("rg_ix_gvis_f32", [seen, base], [vis, exact], struct.pack("iii", run, rows, run))
    pos = lg.input("ix_pos", (run, T), I32)
    thr = lg.input("ix_thr", (64, T), I32)
    lg.node("rg_ix_thr_f32", [score, vis], [thr], struct.pack("i", run))
    cnt, gsum = lg.tensor((K // 64, T), I32), lg.tensor((-(-K // 4096), T), I32)
    lg.node("rg_ix_bcnt_f32", [score, vis, thr], [cnt, gsum])
    # The merge is TWO nodes: the ordering (ds_ix_lanes, one reduction and one store per entry) and the emit
    # (ds_ix_emit, all scalar). A partial store's source lane is a scalar and a vector cannot become one inside a
    # kernel, so the lane has to cross a node boundary -- and with it crossed, the ordering needs one reduction per
    # entry instead of the three a single kernel needed to broadcast the position and score to the slot's lane.
    _outs = [
        lg.input("ix_lslots", (8, T), I32),
        lg.input("ix_list", (run, T), I32),
        lg.input("ix_sout", (K, T), F32),
        lg.input("ix_pos_out", (run, T), I32),
    ]
    # one member per block of 64 rows, so a member's slots never span two 64-slot vectors (a partial-vector store
    # writes one vector's lanes, so a member that spans a vector boundary would shift its entries)
    _nm = max(1, 48 // T)
    if IX_SPLIT_MERGE:
        _lanes = lg.tensor((run, T), I32)
        lg.node("rg_ds_ix_lanes_f32", [score, vis, thr, cnt, gsum, exact, pos, base],
                [_outs[0], _lanes], struct.pack("iiii", _nm, run, heads, run))
        lg.node("rg_ds_ix_emit_f32", [score, vis, gsum, pos, base, _lanes],
                _outs[1:], struct.pack("iii", _nm, run, run))
    else:
        lg.node("rg_ds_ix_merge_f32", [score, vis, thr, cnt, gsum, exact, pos, base], _outs,
                struct.pack("iiii", _nm, run, heads, run))
    return lg.g


def index_chain_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    keys: int,
    blocks: int,
    run: int,
    tiles: int,
    query_block: int = 0,
    parts: int = 1,
    cpages: int = 0,
    heads: int = 0,
    carry: bool = False,
    kind: str = "plain",
    stop: str = "merge",
) -> Graph:
    """The unrolled tiled index chain: ``tiles`` tiles of gather + dequant + score GEMM + acc + assemble + merge
    in ONE recipe, the running top-``run`` carried device-resident in double-buffered buffers. Each tile's acc
    writes its scores to a graph-local tile; ``ds_ix_assemble`` places the previous running set (rows 0 .. run)
    and the tile (rows run ..) into the tile's merge buffer — a single-producer device replacement for the host
    copy_multi — and the merge folds it into the next running set. No (kmax, T) score array. The staging tensors
    are reused across tiles (a control dependency serializes the tiles), so the workspace is one tile's,
    constant in the context. ``ix_gb{i}`` is tile i's first key and ``ix_s{i}.p{p}`` its part p's slot records
    (ds_ix_gslot builds them on the device).

    ``carry`` marks a continuation recipe of a G-tile bucketed walk: its tile 0 folds the incoming running set
    (``ix_rinit``/``ix_pinit``, the previous recipe's ``ix_rout``/``ix_pout``) instead of starting fresh, so
    ceil(B/G) recipes carry one device-resident top-``run`` across the whole context with O(1) host metadata.
    ``kind`` adds the candidate-source block maxima ("bmax": each tile publishes ``ix_bmax{i}``, its slice of
    cand_bmax) or the consumer candidate mask ("mask": each tile's scores are masked to the published blocks
    ``ix_bmax{i}`` + ``cand_thr`` before the merge). The last tile's running set lands in the persistent
    ``ix_rout``/``ix_pout`` (the next recipe reads them) and its list in ``ix_list``/``ix_lslots``."""
    d, T, S, C = dims, tokens, slots, keys
    qb = query_block or T
    prefill = qb % 8 == 0 and C % 1024 == 0
    P = 1  # one part: the acc's score slab is a single-producer graph-local tensor the assemble reads
    lg = LayerGraph(device, f"ix_chain_T{T}_S{S}_C{C}_Q{qb}_B{tiles}_p{P}_{kind}{'c' if carry else ''}", d, T)
    M = INDEX_HEADS * qb
    W = INDEX_HEADS * T
    Sp = S // P
    rows = C  # one tile of keys per merge
    K = run + rows
    q8, qsc = lg.input("ix_q8", (INDEX_DIM, W), F8), lg.input("ix_qs", (W,), F32)
    w = lg.input("ix_w", (INDEX_HEADS, T), F32)
    seen = lg.input("ix_vis", (T,), I32)
    pages, loff = lg.input("cmp_blocks", (cpages, T), I32), lg.input("cmp_loff", (64,), I32)
    rinit = lg.input("ix_rinit", (run, T), F32)
    pinit = lg.input("ix_pinit", (run, T), I32)
    lslots, ilist = lg.input("ix_lslots", (8, T), I32), lg.input("ix_list", (run, T), I32)
    rout = lg.input("ix_rout", (run, T), F32)
    pout = lg.input("ix_pout", (run, T), I32)
    if kind == "mask":
        clist_m = lg.input("cand_list", (INDEX_BLOCKS, T), I32)
        cslots_m = lg.input("cand_slots", (8, T), I32)
    # kind=bmax also keeps a running top-INDEX_BLOCKS BLOCK list (the candidate blocks) while it streams the
    # tiles' block maxima, so the full (kb, T) cand_bmax array is never published: the block merge folds each
    # tile's rows//8 block maxima into a device-resident top-INDEX_BLOCKS, and the last tile lists cand_list.
    if kind == "bmax":
        visb = lg.input("ix_visb", (T,), I32)
        clist = lg.input("cand_list", (INDEX_BLOCKS, T), I32)
        cslots = lg.input("cand_slots", (8, T), I32)
        cthr_out = lg.input("cand_thr", (64, T), I32)
        brinit = lg.input("ix_brinit", (INDEX_BLOCKS, T), F32)
        bpinit = lg.input("ix_bpinit", (INDEX_BLOCKS, T), I32)
        brout = lg.input("ix_brout", (INDEX_BLOCKS, T), F32)
        bpout = lg.input("ix_bpout", (INDEX_BLOCKS, T), I32)
        brprev, bpprev = brinit, bpinit
        bprev = -1
    prev = -1
    rprev, pprev = rinit, pinit
    for i in range(tiles):
        base = lg.input(f"ix_gb{i}", (1,), I32)
        tile = lg.tensor((rows, T), F32)
        m = lg.tensor((K, T), F32)
        rnext = lg.tensor((run, T), F32)
        pnext = lg.tensor((run, T), I32)
        first_nodes = []
        # The prefill branch gathers, dequantizes and scores the tile ONCE for all of its slots; the decode
        # branch keeps the per-slot path (its slots are different rows' columns, so their pages differ).
        for p in range(P):
            table = lg.input(f"ix_s{i}.p{p}", (8, Sp), I32)
            qs, sq = lg.tensor((INDEX_DIM, M, Sp), F8), lg.tensor((M, Sp), F32)
            codes = lg.tensor((INDEX_DIM // 2, C, Sp), U8)
            scales = lg.tensor((INDEX_DIM // 32, C, Sp), U8)
            sk, ktab, ids = (
                lg.tensor((C, Sp), F32),
                lg.tensor((C, Sp), I32),
                lg.tensor((C, Sp), I32),
            )
            codes2 = lg.tensor((INDEX_DIM // 2, C * Sp), U8)
            scales2 = lg.tensor((INDEX_DIM // 32, C * Sp), U8)
            ktab2, ids2 = lg.tensor((C * Sp,), I32), lg.tensor((C * Sp,), I32)
            dec = lg.tensor((INDEX_DIM, C * Sp), F8)
            ks2 = lg.tensor((2 * INDEX_DIM, C // 2, Sp), F8)
            sc = lg.tensor((C, M, Sp), BF16 if prefill else F32)
            sink = lg.tensor((64, 1), I32)
            if prefill:
                # ONE gather, ONE dequant, ONE GEMM for the whole tile. The per-tile record ix_t{i} carries
                # lanes 3 and 7 (per-tile) and a klen of the whole tile, so nothing is zeroed here and the
                # per-slot mask is ds_ix_qacc_tile's `lim`. ix_q8 is already the global (INDEX_DIM, NH*T) query
                # slab in global column order, so the per-slot query gather is gone too.
                c1 = lg.tensor((INDEX_DIM // 2, C, 1), U8)
                s1 = lg.tensor((INDEX_DIM // 32, C, 1), U8)
                k1, t1, i1 = (
                    lg.tensor((C, 1), F32),
                    lg.tensor((C, 1), I32),
                    lg.tensor((C, 1), I32),
                )
                cf = lg.tensor((INDEX_DIM // 2, C), U8)
                sf = lg.tensor((INDEX_DIM // 32, C), U8)
                kf, tf, idf = lg.tensor((C,), F32), lg.tensor((C,), I32), lg.tensor((C,), I32)
                decf = lg.tensor((INDEX_DIM, C), F8)
                ksf = lg.tensor((2 * INDEX_DIM, C // 2), F8)
                ng = lg.g.node(
                    "rg_ds_ix_gather_tile_f32",
                    [lg.arena(blocks), pages, table, lg.dep(), loff],
                    [c1, s1, k1, t1, i1],
                    struct.pack("i", CMP_PAGE_ROWS),
                )
                first_nodes.append(ng)
                if stop == "gather":  # CHAIN_BREAKDOWN: the gather alone
                    continue
                for flat, shaped in ((cf, c1), (sf, s1), (kf, k1), (tf, t1), (idf, i1)):
                    lg.node("reshape", [shaped], [flat])
                if stop == "dequant":  # + the single dequant
                    continue
                lg.node("rg_mx4_decs_f32", [cf, sf, idf, tf], [decf], struct.pack("ii", 0, 0))
                lg.node("reshape", [decf], [ksf])
                # the same packing view the per-slot path takes: the (2*DIM, C/2) fp8 slab IS the (DIM, C)
                # key operand, two keys interleaved per row
                ksv = lg.view(ksf, (INDEX_DIM, C))
                # ONE GEMM over the whole step, and the accumulator reads it (C, W). The (W, C) transpose was
                # tried and abandoned: a vector load runs along dim 0, so keys in the SECOND axis are not a
                # contiguous run at all (the standalone probe showed plausible magnitudes at the wrong
                # coordinates), while (C, W) already had the 128-key run contiguous along dim 0 -- which means
                # attempt 2's 8.6 % is the GEMM's tiling at N = W, not the accumulator's reads, and the lever
                # is the GEMM, not the layout.
                scf = lg.tensor((C, W), BF16)
                lg.node("batch_gemm", [q8, ksv], [scf], GEMM_NT)
                if stop == "gemm":  # + the GEMM (everything but the accumulator)
                    continue
                lg.node(
                    "rg_ds_ix_qacc_tile_f32",
                    [scf, w, table, qsc, kf],
                    [sink, tile],
                    struct.pack("f", INDEX_SCALE),
                )
                continue
            nq = lg.g.node("rg_ix_slot_q8_f32", [q8, qsc, table], [qs, sq])
            ng = lg.g.node(
                "rg_ds_ix_gather_f4_f32",
                [lg.arena(blocks), pages, table, lg.dep(), loff],
                [codes, scales, sk, ktab, ids],
                struct.pack("i", CMP_PAGE_ROWS),
            )
            first_nodes += [nq, ng]
            lg.node("reshape", [codes], [codes2])
            lg.node("reshape", [scales], [scales2])
            lg.node("reshape", [ktab], [ktab2])
            lg.node("reshape", [ids], [ids2])
            lg.node("rg_mx4_decs_f32", [codes2, scales2, ids2, ktab2], [dec], struct.pack("ii", 0, 0))
            lg.node("reshape", [dec], [ks2])
            ks = lg.view(ks2, (INDEX_DIM, C, Sp))
            lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
            if stop != "gemm":  # RESIN_IX_STOP diagnostic: stop after the GEMM to time the score GEMM alone
                if prefill:
                    lg.node(
                        "rg_ds_ix_qacc_o_f32",
                        [sc, w, table, sq, sk],
                        [sink, tile],
                        struct.pack("f", INDEX_SCALE),
                    )
                else:
                    lg.node(
                        "rg_ds_ix_slot_acc8_o_f32",
                        [sc, w, table, sq, sk],
                        [sink, tile],
                        struct.pack("f", INDEX_SCALE),
                    )
        if stop in ("gemm", "acc"):  # diagnostic: skip the assemble + merge prep + merge
            continue
        if prev >= 0:  # tile i starts only after tile i-1's merge (the running set it reads)
            for n in first_nodes:
                lg.g.dep(prev, n)
        if kind == "bmax" and bprev >= 0:  # also after tile i-1's BLOCK merge (its running block set)
            for n in first_nodes:
                lg.g.dep(bprev, n)
        scored = tile
        tile_bmax = None
        if kind == "bmax":  # the candidate source's tile block maxima feed the running top-INDEX_BLOCKS below
            tile_bmax = lg.tensor((rows // 8, T), F32)
            # natural block order (row == block) for the running block merge's row-based vis/position arithmetic:
            # the kernel reads the tile's (rows, T) scores directly (lane == key), reduces blocks of 8, transposes
            # the permuted 8x8 to natural.
            lg.node("rg_ds_ix_bmax_nat_f32", [tile, seen, base], [tile_bmax])
        elif kind == "mask":  # the consumer masks this tile's scores to the running top-2048 candidate blocks
            masked = lg.tensor((rows, T), F32)
            lg.node("rg_ix_cmask_list_f32", [tile, seen, clist_m, cslots_m, base], [masked])
            scored = masked
        # tile 0 starts fresh unless this is a continuation recipe (carry) reading the incoming running set
        run0 = run if (i > 0 or carry) else 0
        prev = lg.g.node("rg_ds_ix_assemble_f32", [rprev, scored], [m], struct.pack("ii", run0, rows))
        vis, exact = lg.tensor((T,), I32), lg.tensor((T,), I32)
        thr = lg.tensor((64, T), I32)
        cnt, gsum = lg.tensor((K // 64, T), I32), lg.tensor((-(-K // 4096), T), I32)
        if stop == "assemble":  # RESIN_IX_STOP diagnostic: stop after the merge-buffer assemble
            continue
        prev = lg.g.node(
            "rg_ix_gvis_f32", [seen, base], [vis, exact], struct.pack("iii", run, rows, run)
        )
        if stop == "gvis":
            continue
        prev = lg.g.node("rg_ix_thr_f32", [m, vis], [thr], struct.pack("i", run))
        if stop == "thr":
            continue
        prev = lg.g.node("rg_ix_bcnt_f32", [m, vis, thr], [cnt, gsum])
        if stop == "bcnt":
            continue
        # only the last tile's list is the column's final one; earlier merges list into scratch. The last
        # tile's running set lands in the persistent ix_rout/ix_pout so the next G-tile recipe can carry it.
        lo, li = (lslots, ilist) if i == tiles - 1 else (lg.tensor((8, T), I32), lg.tensor((run, T), I32))
        rn, pn = (rout, pout) if i == tiles - 1 else (rnext, pnext)
        if IX_SPLIT_MERGE:
            lanes = lg.tensor((run, T), I32)
            prev = lg.g.node(
                "rg_ds_ix_lanes_f32",
                [m, vis, thr, cnt, gsum, exact, pprev, base],
                [lo, lanes],
                struct.pack("iiii", max(1, 48 // T), run, heads, run),
            )
            lg.g.node(
                "rg_ds_ix_emit_f32", [m, vis, gsum, pprev, base, lanes], [li, rn, pn],
                struct.pack("iii", max(1, 48 // T), run, run),
            )
        else:
            prev = lg.g.node(
                "rg_ds_ix_merge_f32",
                [m, vis, thr, cnt, gsum, exact, pprev, base],
                [lo, li, rn, pn],
                struct.pack("iiii", max(1, 48 // T), run, heads, run),
            )
        rprev, pprev = rn, pn
        if kind == "bmax":
            # running top-INDEX_BLOCKS BLOCK merge: fold this tile's rows//8 block maxima into the device-resident
            # candidate-block set. Block positions are block indices (bbase = tile*rows//8), so cand_list lists
            # blocks exactly as one pass over the full (kb, T) maxima would — no full array is ever published.
            bbase = lg.input(f"ix_gbb{i}", (1,), I32)
            bK = INDEX_BLOCKS + rows // 8
            bm = lg.tensor((bK, T), F32)
            brun0 = INDEX_BLOCKS if carry else min(i * (rows // 8), INDEX_BLOCKS)
            lg.node("rg_ds_ix_assemble_f32", [brprev, tile_bmax], [bm], struct.pack("ii", brun0, rows // 8))
            bvis, bexact = lg.tensor((T,), I32), lg.tensor((T,), I32)
            bthr = cthr_out if i == tiles - 1 else lg.tensor((64, T), I32)
            bcnt = lg.tensor((bK // 64, T), I32)
            bgsum = lg.tensor((-(-bK // 4096), T), I32)
            lg.node("rg_ix_gvis_f32", [visb, bbase], [bvis, bexact], struct.pack("iii", INDEX_BLOCKS, rows // 8, INDEX_BLOCKS))
            lg.node("rg_ix_thr_f32", [bm, bvis], [bthr], struct.pack("i", INDEX_BLOCKS))
            lg.node("rg_ix_bcnt_f32", [bm, bvis, bthr], [bcnt, bgsum])
            blo, bli = (cslots, clist) if i == tiles - 1 else (lg.tensor((8, T), I32), lg.tensor((INDEX_BLOCKS, T), I32))
            brn, bpn = (brout, bpout) if i == tiles - 1 else (lg.tensor((INDEX_BLOCKS, T), F32), lg.tensor((INDEX_BLOCKS, T), I32))
            if IX_SPLIT_MERGE:
                blanes = lg.tensor((INDEX_BLOCKS, T), I32)
                bprev = lg.g.node(
                    "rg_ds_ix_lanes_f32",
                    [bm, bvis, bthr, bcnt, bgsum, bexact, bpprev, bbase],
                    [blo, blanes],
                    struct.pack("iiii", max(1, 48 // T), INDEX_BLOCKS, 0, INDEX_BLOCKS),
                )
                lg.g.node(
                    "rg_ds_ix_emit_f32",
                    [bm, bvis, bgsum, bpprev, bbase, blanes],
                    [bli, brn, bpn],
                    struct.pack("iii", max(1, 48 // T), INDEX_BLOCKS, INDEX_BLOCKS),
                )
            else:
                bprev = lg.g.node(
                    "rg_ds_ix_merge_f32",
                    [bm, bvis, bthr, bcnt, bgsum, bexact, bpprev, bbase],
                    [blo, bli, brn, bpn],
                    struct.pack("iiii", max(1, 48 // T), INDEX_BLOCKS, 0, INDEX_BLOCKS),
                )
            brprev, bpprev = brn, bpn
    return lg.g


def index_sparse_chain_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    keys: int,
    blocks: int,
    run: int,
    tiles: int,
    cand_top: int,
    pages: int,
    heads: int = 0,
) -> Graph:
    """The unrolled SPARSE index chain (the consumer layers' candidate-entry top-512): ``tiles`` tiles of the
    column's ``cand_top`` candidate blocks' keys, gathered by ix_cgather_f4_local (tile-local cpos), scored, and
    folded into a device-resident running top-``run`` by ds_ix_merge_cpos (tile positions read from cpos). ONE
    recipe for a batch of ``tokens`` columns replaces the (kc, T) compact-score staging + per-chunk launches: the
    merge buffer is (run + keys, tokens), constant in the context. ``ix_s{i}`` is tile i's slot records (one slot
    per column, key0 = i*keys), ``ix_gb{i}`` its first candidate-entry index; the candidate list/slots and the
    step's page table are bound to the batch's column slice."""
    d, T, C = dims, tokens, keys
    rows = C
    K = run + rows
    S = T  # one slot per column (query_block 1)
    lg = LayerGraph(device, f"ix_schain_T{T}_C{C}_B{tiles}_ct{cand_top}", d, T)
    M = INDEX_HEADS
    W = INDEX_HEADS * T
    q8, qsc = lg.input("ix_q8", (INDEX_DIM, W), F8), lg.input("ix_qs", (W,), F32)
    w = lg.input("ix_w", (INDEX_HEADS, T), F32)
    seen = lg.input("ix_visc", (T,), I32)
    cand = lg.input("cand_list", (cand_top, T), I32)
    cslots = lg.input("cand_slots", (8, T), I32)
    pagesT, loff = lg.input("cmp_blocks", (pages, T), I32), lg.input("cmp_loff", (64,), I32)
    rinit = lg.input("ix_rinit", (run, T), F32)
    pinit = lg.input("ix_pinit", (run, T), I32)
    lslots, ilist = lg.input("ix_lslots", (8, T), I32), lg.input("ix_list", (run, T), I32)
    prev = -1
    rprev, pprev = rinit, pinit
    for i in range(tiles):
        base = lg.input(f"ix_gb{i}", (1,), I32)
        table = lg.input(f"ix_s{i}", (8, S), I32)
        tile = lg.tensor((rows, T), F32)
        m = lg.tensor((K, T), F32)
        cpos = lg.tensor((rows, T), I32)
        qs, sq = lg.tensor((INDEX_DIM, M, S), F8), lg.tensor((M, S), F32)
        codes = lg.tensor((INDEX_DIM // 2, C, S), U8)
        scales = lg.tensor((INDEX_DIM // 32, C, S), U8)
        sk, ktab, ids = lg.tensor((C, S), F32), lg.tensor((C, S), I32), lg.tensor((C, S), I32)
        codes2 = lg.tensor((INDEX_DIM // 2, C * S), U8)
        scales2 = lg.tensor((INDEX_DIM // 32, C * S), U8)
        ktab2, ids2 = lg.tensor((C * S,), I32), lg.tensor((C * S,), I32)
        dec = lg.tensor((INDEX_DIM, C * S), F8)
        ks2 = lg.tensor((2 * INDEX_DIM, C // 2, S), F8)
        sc = lg.tensor((C, M, S), F32)
        sink = lg.tensor((64, 1), I32)
        nq = lg.g.node("rg_ix_slot_q8_f32", [q8, qsc, table], [qs, sq])
        ng = lg.g.node(
            "rg_ix_cgather_f4_local_f32",
            [lg.arena(blocks), cand, cslots, pagesT, loff, table, lg.dep()],
            [codes, scales, sk, ktab, ids, cpos],
            struct.pack("i", CMP_PAGE_ROWS),
        )
        if prev >= 0:
            lg.g.dep(prev, nq)
            lg.g.dep(prev, ng)
        lg.node("reshape", [codes], [codes2])
        lg.node("reshape", [scales], [scales2])
        lg.node("reshape", [ktab], [ktab2])
        lg.node("reshape", [ids], [ids2])
        lg.node("rg_mx4_decs_f32", [codes2, scales2, ids2, ktab2], [dec], struct.pack("ii", 0, 0))
        lg.node("reshape", [dec], [ks2])
        ks = lg.view(ks2, (INDEX_DIM, C, S))
        lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
        lg.node("rg_ds_ix_slot_acc8_o_f32", [sc, w, table, sq, sk], [sink, tile], struct.pack("f", INDEX_SCALE))
        lg.node("rg_ds_ix_assemble_f32", [rprev, tile], [m], struct.pack("ii", 0 if i == 0 else run, rows))
        vis, exact = lg.tensor((T,), I32), lg.tensor((T,), I32)
        thr = lg.tensor((64, T), I32)
        cnt, gsum = lg.tensor((K // 64, T), I32), lg.tensor((-(-K // 4096), T), I32)
        lg.node("rg_ix_gvis_f32", [seen, base], [vis, exact], struct.pack("iii", run, rows, run))
        lg.node("rg_ix_thr_f32", [m, vis], [thr], struct.pack("i", run))
        lg.node("rg_ix_bcnt_f32", [m, vis, thr], [cnt, gsum])
        lo, li = (lslots, ilist) if i == tiles - 1 else (lg.tensor((8, T), I32), lg.tensor((run, T), I32))
        rnext, pnext = lg.tensor((run, T), F32), lg.tensor((run, T), I32)
        prev = lg.g.node(
            "rg_ds_ix_merge_cpos_f32",
            [m, vis, thr, cnt, gsum, exact, pprev, base, cpos],
            [lo, li, rnext, pnext],
            struct.pack("iiii", max(1, 48 // T), run, heads, run),
        )
        rprev, pprev = rnext, pnext
    return lg.g


def index_gslot_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    lpg: int,
    tp: int,
    nblk: int,
    qb: int,
    ratio: int,
    groups: int,
    keys: int,
    run: int,
    page: int,
) -> Graph:
    """The tiled index chain's *group* tables, built on the device: every merge group's slot records (``ix_rec``
    (8, groups * lpg * slots); lane 7 is the tile's first index page, which ds_ix_gather reads instead of a
    per-slot page table) and the groups' first keys (``ix_gbase`` (groups,)), from the step's O(T) ``blk``.
    One launch per ratio per step whatever the number of groups: the host fill writes no per-group value."""
    d, T = dims, tokens
    S2 = lpg * slots
    lg = LayerGraph(device, f"ix_gslot_T{T}_S{slots}_L{lpg}_P{tp}_r{ratio}", d, T)
    lg.node(
        "rg_ds_ix_gslot_f32",
        [lg.input("blk", (2, T), I32)],
        [lg.input("ix_rec", (8, groups * S2), I32), lg.input("ix_gbase", (64, groups), I32)],
        struct.pack("iiiiiiiii", S2, nblk, qb, INDEX_HEADS, ratio, keys, run, tp, page),
    )
    return lg.g


def index_blocks_graph(device: Device, dims: Dims, tokens: int, rows: int) -> Graph:
    """A tile group's candidate blocks: the maxima of the group's blocks of 8 entries (``ix_bmax``, which the
    caller binds to the group's slice of the published block maxima) with the column's ``ix_pin`` (T,) block (its
    newest, or -1 when another group holds it) set to +inf, and ``ix_tvis`` (T,) the group's visible keys."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ix_blocks_T{T}_{rows}", d, T)
    lg.node(
        "rg_ds_ix_bmax_f32",
        [
            lg.input("ix_tile", (rows, T), F32),
            lg.input("ix_vis", (T,), I32),
            lg.input("ix_gbase", (1,), I32),
        ],
        [lg.input("ix_bmax", (rows // 8, T), F32)],
    )
    return lg.g


def index_mask_graph(device: Device, dims: Dims, tokens: int, rows: int) -> Graph:
    """A tile group's scores masked to the candidate blocks the source layer published (``rg_ix_cmask`` over the
    group's rows, read from the published block maxima and ``cand_thr``); the caller binds ``ix_tile`` and
    ``ix_tile_out`` to the same rows."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ix_mask_T{T}_{rows}", d, T)
    lg.node(
        "rg_ix_cmask_f32",
        [
            lg.input("ix_tile", (rows, T), F32),
            lg.input("ix_vis", (T,), I32),
            lg.input("ix_bmax", (rows // 8, T), F32),
            lg.input("cand_thr", (64, T), I32),
            lg.input("ix_gbase", (1,), I32),
        ],
        [lg.input("ix_tile_out", (rows, T), F32)],
    )
    return lg.g


def index_block_select_graph(
    device: Device, dims: Dims, tokens: int, blocks: int, top: int, listed: bool
) -> Graph:
    """A column's candidate blocks from the published block maxima ``cand_bmax`` (blocks, T): the threshold
    ``cand_thr`` (64, T) that keeps the ``top`` largest, and with ``listed`` their list (``cand_list`` (top, T),
    ``cand_slots`` (8, T))."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ix_bsel_T{T}_K{blocks}" + ("_l" if listed else ""), d, T)
    bmax, visb = lg.input("cand_bmax", (blocks, T), F32), lg.input("ix_visb", (T,), I32)
    thr = lg.input("cand_thr", (64, T), I32)
    lg.node("rg_ix_thr_f32", [bmax, visb], [thr], struct.pack("i", top))
    if listed:
        cnt, gsum = lg.tensor((blocks // 64, T), I32), lg.tensor((-(-blocks // 4096), T), I32)
        lg.node("rg_ix_bcnt_f32", [bmax, visb, thr], [cnt, gsum])
        lg.node(
            "rg_ix_list_f32",
            [bmax, visb, thr, cnt, gsum],
            [lg.input("cand_list", (top, T), I32), lg.input("cand_slots", (8, T), I32)],
            struct.pack("iii", max(1, 48 // T), top, 0),
        )
    return lg.g


def _gather_listed(lg: LayerGraph, arena_blocks: int, blocks: int) -> Tensor:
    """The listed compressed latents of every column: (head_dim, 512, T) bf16.  The gather emits a packed
    fp4 codes+scales staging slab; a separate dequant producer (ds_ldec_f4) decodes it to bf16 so the
    compiler SRAM-slices the dequant tile feeding batch_gemm and overlaps TPC-dequant with MME-GEMM."""
    D, K, T = lg.d.head_dim, INDEX_TOP, lg.T
    codes = lg.tensor((D // 2, K, T), U8)
    scales = lg.tensor((D // 16, K, T), U8)
    lg.node(
        "rg_ds_gather_list_f4_f32",
        [lg.arena(arena_blocks), lg.input("ix_list", (K, T), I32), lg.input("cmp_blocks", (blocks, T), I32),
         lg.input("cmp_loff", (64,), I32), lg.input("ix_lslots", (8, T), I32), lg.dep()],
        [codes, scales],
    )  # fmt: skip
    codes2 = lg.tensor((D // 2, K * T), U8)
    scales2 = lg.tensor((D // 16, K * T), U8)
    ks2 = lg.tensor((D, K * T), BF16)
    lg.node("reshape", [codes], [codes2])
    lg.node("reshape", [scales], [scales2])
    lg.node("rg_ds_ldec_f4_f32", [codes2, scales2], [ks2])
    ks = lg.tensor((D, K, T), BF16)
    lg.node("reshape", [ks2], [ks])
    return ks


def _gather_dense(lg: LayerGraph, arena_blocks: int, keys: int) -> Tensor:
    """Every visible compressed latent of each query block's slot: (head_dim, keys, slots) bf16, from the owner's
    latent pages ``cmp_lat`` (keys / 128, slots) and the slots ``cslots`` (8, slots).  Staging gather + separate
    dequant producer (ds_ldec_f4) for SRAM slicing / TPC-dequant ∥ MME-GEMM overlap, as _gather_listed."""
    D, S = lg.d.head_dim, lg.T // QUERY_BLOCK_DENSE
    codes = lg.tensor((D // 2, keys, S), U8)
    scales = lg.tensor((D // 16, keys, S), U8)
    lp = lg.input("cmp_lat", (keys // PAGE, S), I32)
    lg.node(
        "rg_ds_gather_dense_f4_f32",
        [lg.arena(arena_blocks), lp, lp, lg.input("cslots", (8, S), I32), lg.dep()],
        [codes, scales],
    )
    codes2 = lg.tensor((D // 2, keys * S), U8)
    scales2 = lg.tensor((D // 16, keys * S), U8)
    ks2 = lg.tensor((D, keys * S), BF16)
    lg.node("reshape", [codes], [codes2])
    lg.node("reshape", [scales], [scales2])
    lg.node("rg_ds_ldec_f4_f32", [codes2, scales2], [ks2])
    ks = lg.tensor((D, keys, S), BF16)
    lg.node("reshape", [ks2], [ks])
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


def _listed_fused(
    lg: LayerGraph, arena_blocks: int, ks_cmp: Tensor, qs: Tensor, lslots: Tensor,
    win_slots: int, win_keys: int, o_out: Tensor | None = None, win_dep: Tensor | None = None,
) -> Tensor:
    """The listed compressed attention as ONE single-pass global-max softmax over the window keys and the
    listed compressed keys (the official sparse_attn formula for <= 64 visible keys): gather the window keys
    once (``mla_gather``, S query-block slots), score both key sets, ds_slot_fused casts bf16 P at the keys-only
    global max M and sums FP32 ls, two P.V gemms, then ds_attn_fused adds the parts and the sink in the
    denominator and removes the rotation -> o (D, W, 1). The window gemms run in the slot layout (M = nh*qb,
    S) matching the gathered keys and are viewed to (nh, T): column h + nh*t of (nh, T) is column m + M*s of
    (M, S) for t = qb*s + col, m = h + nh*col, so the flat layouts coincide for any S."""
    d, T = lg.d, lg.T
    D, nh = d.head_dim, d.heads
    W = nh * T
    C, K, S = win_keys, ks_cmp.shape[1], win_slots
    qb = T // S
    M = nh * qb
    wtable = lg.input("slots", (8, S), I32)
    lp = lg.input("lat_pages", (C // PAGE, S), I32)
    rp = lg.input("rope_pages", (C // PAGE, S), I32)
    ksw = lg.tensor((D, C, S), BF16)
    lg.node(
        "rg_mla_gather_f32", [lg.arena(arena_blocks), lp, rp, wtable, win_dep if win_dep is not None else lg.dep()], [ksw],
        struct.pack("i", 0),
    )  # fmt: skip
    qsb = lg.view(qs, (D, M, S))
    scwb = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [qsb, ksw], [scwb], GEMM_NT)
    scw = lg.view(scwb, (C, nh, T))
    scc = lg.tensor((K, nh, T), F32)
    lg.node("batch_gemm", [qs, ks_cmp], [scc], GEMM_NT)
    Mp = -(-nh // 64) * 64
    Pw, Pc = lg.tensor((C, nh, T), BF16), lg.tensor((K, nh, T), BF16)
    mx, lsw, lsc = (lg.tensor((Mp, T), F32), lg.tensor((Mp, T), F32), lg.tensor((Mp, T), F32))
    lg.node(
        "rg_ds_slot_fused_f32", [scw, scc, wtable, lslots], [Pw, Pc, mx, lsw, lsc],
        struct.pack("fii", d.scale, nh, qb),
    )  # fmt: skip
    pv1b = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [lg.view(Pw, (C, M, S)), ksw], [pv1b], GEMM_NN)
    pv1 = lg.view(pv1b, (D, nh, T))
    pv2 = lg.tensor((D, nh, T), F32)
    lg.node("batch_gemm", [Pc, ks_cmp], [pv2], GEMM_NN)
    o = o_out if o_out is not None else lg.tensor((D, W, 1), F32)
    lg.node(
        "rg_ds_attn_fused_f32",
        [lg.view(pv1, (D, W, 1)), lg.view(pv2, (D, W, 1)), mx, lsw, lsc,
         lg.input("attn_sink", (nh,), F32), lg.input("cs_inv", (128, T), F32)],
        [o], struct.pack("i", nh),
    )  # fmt: skip
    return o


def _dense_fused(
    lg: LayerGraph, arena_blocks: int, ks_dense: Tensor, scores: int,
    win_slots: int, win_keys: int, win_dep: Tensor | None = None, o_out: Tensor | None = None,
) -> Tensor:
    """The DENSE prefill compressed attention as ONE single-pass global-max softmax over the window keys and
    every visible compressed entry of each query block (each column's top-512 list as a mask), the official
    sparse_attn formula for <= 64 visible keys. Both key sets are in the slot layout (M = nh*128, S); the
    window keys are gathered once (mla_gather), ds_slot_fused_dense casts bf16 P at the keys-only global max
    and sums FP32 ls, two P.V gemms, ds_attn_fused_dense adds the parts and the sink and removes the rotation
    -> o (D, W, 1)."""
    d, T = lg.d, lg.T
    D, nh = d.head_dim, d.heads
    W = nh * T
    C, K, S = win_keys, ks_dense.shape[1], win_slots
    qb = QUERY_BLOCK_DENSE
    M = nh * qb
    wtable = lg.input("slots", (8, S), I32)
    ctable = lg.input("cslots", (8, S), I32)
    lp = lg.input("lat_pages", (C // PAGE, S), I32)
    rp = lg.input("rope_pages", (C // PAGE, S), I32)
    ksw = lg.tensor((D, C, S), BF16)
    lg.node(
        "rg_mla_gather_f32", [lg.arena(arena_blocks), lp, rp, wtable, win_dep if win_dep is not None else lg.dep()], [ksw],
        struct.pack("i", 0),
    )  # fmt: skip
    qs = lg.view(lg.input("q_all", (D, W), BF16), (D, M, S))
    scw = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [qs, ksw], [scw], GEMM_NT)
    scc = lg.tensor((K, M, S), F32)
    lg.node("batch_gemm", [qs, ks_dense], [scc], GEMM_NT)
    Pw, Pc = lg.tensor((C, M, S), BF16), lg.tensor((K, M, S), BF16)
    mx, lsw, lsc = (lg.tensor((M, S), F32), lg.tensor((M, S), F32), lg.tensor((M, S), F32))
    lg.node(
        "rg_ds_slot_fused_dense_f32",
        [scw, scc, wtable, ctable, lg.input("ix_score", (scores, T), F32),
         lg.input("ix_thr", (64, T), I32), lg.input("ix_lslots", (8, T), I32)],
        [Pw, Pc, mx, lsw, lsc], struct.pack("fi", d.scale, nh),
    )  # fmt: skip
    pv1 = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [Pw, ksw], [pv1], GEMM_NN)
    pv2 = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [Pc, ks_dense], [pv2], GEMM_NN)
    o = o_out if o_out is not None else lg.tensor((D, W, 1), F32)
    lg.node(
        "rg_ds_attn_fused_dense_f32",
        [lg.view(pv1, (D, W, 1)), lg.view(pv2, (D, W, 1)), mx, lsw, lsc,
         lg.input("attn_sink", (nh,), F32), lg.input("cs_inv", (128, T), F32)],
        [o], struct.pack("ii", nh, qb),
    )  # fmt: skip
    return o


# ------------------------------------------------------------------ bounded online attention stream
# The listed compressed attention as a bounded query-tile x key-block online-softmax stream (design §1):
# ONE query-tile of Q columns attends over the unified 640-key set (window 128 + listed 512) streamed as
# 10 blocks of B = 64, reusing one bounded ks_blk operand buffer. acc_o / max / sum are loop-carried
# device-resident across the 10 blocks (the exact NVIDIA sparse_attn recurrence: scale before max, bf16 P
# cast before PV, f32 accumulate, sink in the denominator only). The full-T (D, 512, T) gathered slab and
# the (512, nh, T) score / P / pv slabs never exist: the working set is one tile's (~15 MiB live). The step
# launches T / Q tile recipes per layer (each binding tile_off + its q_all / cs_inv / o_out column slice),
# then one wo_a / wo_b recipe over the full T output (attention_out_wo_graph). PREFILL only; decode keeps
# the small single-launch path (attention_out_graph).
ATTN_TILE_Q = 256  # query-tile columns (the batch_gemm batch dim S = Q, nh a separate matrix dim)
# keys per block. 64 is NVIDIA's block, but the cost of this recipe is the number of blocks, not the keys:
# each block round-trips the loop-carried acc_o (D, nh, Q) f32 accumulator, so 10 blocks pay that 10 times,
# and every block's tiles (gather ~4.2 MB, online ~3 MB) sit far below the ~16 MB where the TPC/DMA envelope
# is measured. 128 halves the block count and doubles the tile: 5 blocks, ~8.4 MB tiles. The kernels take B
# as a runtime parameter (ds_gather_list_blk_f4's index space is [1, B/16, Q]), and the online recurrence's
# f32 noise compounds with the block count, so fewer blocks is also the numerically safer direction.
ATTN_KEY_BLK = 128
ATTN_WIN_BLOCKS = WINDOW // ATTN_KEY_BLK  # window blocks (128 keys, fp8 -> bf16, causal mask)
ATTN_LIST_BLOCKS = INDEX_TOP // ATTN_KEY_BLK  # listed blocks (512 keys, fp4 -> bf16, count mask)
ATTN_BLOCKS = ATTN_WIN_BLOCKS + ATTN_LIST_BLOCKS  # blocks in the unified stream


def _stream_listed_block(
    lg: LayerGraph, arena_blocks: int, blocks: int, b0: int, B: int, Q: int,
    codes: Tensor, scales: Tensor, idx: Tensor, codes2: Tensor, scales2: Tensor,
    ks2: Tensor, ks_blk: Tensor,
) -> int:
    """One listed key-block (b0) of a query-tile into the REUSED staging: gather the fp4 codes+scales of the
    block's B listed latents for the tile's Q columns (ds_gather_list_blk_f4, runtime tile_off) plus the
    block's per-column key indices idx (B, Q) (-1 past the column's count), then dequant to ks_blk (D, B, Q)
    bf16 (ds_ldec_f4, the separate SRAM-slicable producer). Returns the gather node index (for the block
    control dependency)."""
    gn = lg.g.node(
        "rg_ds_gather_list_blk_f4_f32",
        [lg.arena(arena_blocks), lg.input("ix_list", (INDEX_TOP, lg.T), I32),
         lg.input("cmp_blocks", (blocks, lg.T), I32), lg.input("cmp_loff", (64,), I32),
         lg.input("ix_lslots", (8, lg.T), I32), lg.input("tile_off", (1,), I32), lg.dep()],
        [codes, scales, idx],
        struct.pack("iii", b0, B, Q),
    )  # fmt: skip
    lg.node("reshape", [codes], [codes2])
    lg.node("reshape", [scales], [scales2])
    lg.node("rg_ds_ldec_f4_f32", [codes2, scales2], [ks2])
    lg.node("reshape", [ks2], [ks_blk])
    return gn


def _stream_window_block(
    lg: LayerGraph, arena_blocks: int, b0: int, B: int, Q: int, win_pages: int,
    win_dep: Tensor | None, ks_blk: Tensor, idx: Tensor,
) -> int:
    """One window key-block (b0 in 0..1) of a query-tile into the REUSED staging (attn-gather's committed
    ds_gather_win_blk, option A): gather the fp8 window latents of the block's B positions for the tile's Q
    columns, dequant fp8 -> bf16 in the gather (mla_gather mode-0 convention), into ks_blk (D, B, Q) bf16,
    plus idx (B, Q) = each column's window key absolute position (-1 when invalid, the causal mask). Column
    s = tile_off + q (tile_off the STEP-COLUMN offset, uniform with the listed gather); window key k is at
    absolute position pos = pos0 + s - (WINDOW-1) + b0*B + k (pos0 from meta; NVIDIA get_window_topk_idxs:
    window [t-127, t] includes the column's own position); valid iff pos >= 0 && pos < klen && the
    position's page exists. ``win_pages`` (WP,) i32 is the GLOBAL positional page table
    (win_pages[pos>>7] = first latent arena row of this layer's 129-row page, -1 = zeros); ``wslots`` (8, T)
    i32 field 4 = each column's visible key count. Returns the gather node index."""
    gn = lg.g.node(
        "rg_ds_gather_win_blk_f32",
        [lg.arena(arena_blocks), lg.input("win_pages", (win_pages,), I32),
         lg.input("tile_off", (1,), I32), lg.input("wslots", (8, lg.T), I32), lg.meta(),
         win_dep if win_dep is not None else lg.dep()],
        [ks_blk, idx],
        struct.pack("iii", b0, B, Q),
    )  # fmt: skip
    return gn


def attention_out_tile_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    arena_blocks: int,
    blocks: int,
    *,
    win_slots: int,
    win_keys: int,
    win_pages: int = 0,
    tile_q: int = 0,
    dbg_o: str = "",
) -> Graph:
    """ONE query-tile of the bounded listed attention stream: ``tile_q`` columns (default min(ATTN_TILE_Q, T))
    attend over the unified 640-key set as ATTN_BLOCKS unrolled key-blocks of B = 64, the online-softmax
    accumulator acc_o / max / sum loop-carried device-resident across every block (window blocks 0..1 then
    listed blocks 2..9, ONE recurrence — no split-then-merge). The staging tensors (ks_blk / sc / P / pv /
    idx / codes / scales) are reused across blocks; a control dependency serializes block i+1's gather after
    block i's accumulate (Phase 1 single ks_blk buffer; Phase 2 double-buffers for TPC-gather ∥ MME-GEMM).

    The tile's q_all / cs_inv / o_tile are bound to the tile's column slice of the persistent buffers; the
    listed gather reads the FULL ix_list / ix_lslots / cmp_blocks at column tile_off + q (tile_off a runtime
    input). ``o_tile`` (D, nh, Q) f32 is the normalized inverse-roped output (ds_attn_online_fin, f32 to keep
    the o_inv fidelity gate); the caller binds it to o_out's columns [tile_off .. +Q]. ``dbg_o`` names an
    optional persistent f32 mirror of o_tile for the fidelity gate."""
    d, T = dims, tokens
    D, nh = d.head_dim, d.heads
    Q = tile_q or min(ATTN_TILE_Q, T)
    B = ATTN_KEY_BLK
    lg = LayerGraph(device, f"attn_out_tile_T{T}_Q{Q}_B{blocks}_A{arena_blocks}", d, T)
    q_tile = lg.view(lg.input("q_all", (D, nh * Q), BF16), (D, nh, Q))
    # The loop-carried online-softmax state is SSA across the blocks of ONE recipe: Synapse treats
    # input-only tensors as immutable within a recipe (in-place writes only commit across separate
    # launches), so each block PRODUCES the next block's state as a FRESH single-producer output.
    # acc_o[i]/mx[i]/sm[i] are block i's state; init seeds [0], online/acc produce [i+1], fin reads
    # [ATTN_BLOCKS]. The compiler reuses the dead [i] memory across the dep-serialized blocks (bounded
    # working set, ~2 acc_o live), exactly as the per-block staging below.
    acc_o = [lg.tensor((D, nh, Q), F32) for _ in range(ATTN_BLOCKS + 1)]
    mx = [lg.tensor((nh, Q), F32) for _ in range(ATTN_BLOCKS + 1)]
    sm = [lg.tensor((nh, Q), F32) for _ in range(ATTN_BLOCKS + 1)]
    lg.node("rg_ds_attn_online_init_f32", [], [acc_o[0], mx[0], sm[0]])
    # Per-block staging: each tensor has a SINGLE producer (Synapse rejects a tensor produced by multiple
    # nodes — "producer is already registered"). The buffers are allocated per block and the compiler reuses
    # their memory across the control-dep-serialized blocks (as index_chain does), bounding the workspace to
    # one block's staging (~15 MiB), NOT the graph-tensor identity.
    prev = -1
    for blk in range(ATTN_BLOCKS):
        ks_blk = lg.tensor((D, B, Q), BF16)
        idx = lg.tensor((B, Q), I32)
        sc = lg.tensor((B, nh, Q), F32)
        P = lg.tensor((B, nh, Q), BF16)
        sscale = lg.tensor((nh, Q), F32)
        pv = lg.tensor((D, nh, Q), F32)
        if blk < ATTN_WIN_BLOCKS:
            gn = _stream_window_block(
                lg, arena_blocks, blk, B, Q, win_pages, None, ks_blk, idx
            )
        else:
            b0 = blk - ATTN_WIN_BLOCKS  # listed key-block index 0..7
            codes = lg.tensor((D // 2, B, Q), U8)
            scales = lg.tensor((D // 16, B, Q), U8)
            codes2 = lg.tensor((D // 2, B * Q), U8)
            scales2 = lg.tensor((D // 16, B * Q), U8)
            ks2 = lg.tensor((D, B * Q), BF16)
            gn = _stream_listed_block(
                lg, arena_blocks, blocks, b0, B, Q, codes, scales, idx, codes2, scales2, ks2, ks_blk
            )
        # No control dependency on the previous block: every block's staging is its own tensor (ks_blk, idx, sc,
        # P, sscale, pv are all allocated inside this loop), so the gathers and the QK GEMMs of all five blocks
        # can be in flight together and only the online-softmax/acc chain is serial. The dep this replaces forced
        # block i+1's gather to wait for block i's acc, which is what made the recipe ten strictly sequential
        # small GEMMs (M=128, N=nh=8) instead of a pipeline. Workspace grows from one block's staging to five
        # (~75 MB), which is the trade.
        lg.node("batch_gemm", [q_tile, ks_blk], [sc], GEMM_NT)
        lg.node(
            "rg_ds_attn_online_f32", [sc, mx[blk], sm[blk], idx], [P, sscale, mx[blk + 1], sm[blk + 1]],
            struct.pack("fii", d.scale, nh, B),
        )  # fmt: skip
        lg.node("batch_gemm", [P, ks_blk], [pv], GEMM_NN)
        prev = lg.g.node("rg_ds_attn_acc_f32", [acc_o[blk], sscale, pv], [acc_o[blk + 1]])
    o_tile = lg.input("o_tile", (D, nh, Q), F32)
    lg.g.node(
        "rg_ds_attn_online_fin_f32",
        [acc_o[ATTN_BLOCKS], mx[ATTN_BLOCKS], sm[ATTN_BLOCKS], lg.input("attn_sink", (nh,), F32), lg.input("cs_inv", (128, Q), F32)],
        [o_tile],
    )  # fmt: skip
    if dbg_o:
        lg.node("memcpy", [o_tile], [lg.input(dbg_o, (D, nh, Q), F32)])
    return lg.g


def attention_out_wo_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """The bounded stream's output projection over the full T: the tiles' inverse-roped outputs ``o_out``
    (D, nh*T) f32 -> wo_a per group (FP8) -> wo_b -> this rank's partial ``p_out``. Identical to the tail of
    the single-launch attention_out_graph (_attention_out), run as its own recipe after the T/Q tile launches."""
    d, T = dims, tokens
    D, nh = d.head_dim, d.heads
    lg = LayerGraph(device, f"attn_out_wo_T{T}", d, T)
    o = lg.input("o_out", (D, nh * T), F32)
    _attention_out(lg, o)
    return lg.g


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
    win_slots: int = 0,
    win_keys: int = 0,
    dbg_o: str = "",
    win_dep: Tensor | None = None,
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
        if win_keys:
            o_out = lg.input(dbg_o, (D, W, 1), F32) if dbg_o else None
            o = _dense_fused(lg, arena_blocks, ks, scores, win_slots, win_keys, win_dep, o_out)
            _attention_out(lg, lg.view(o, (D, W)))
            return lg.g
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
    if win_keys and not dense:
        o_out = lg.input(dbg_o, (D, W, 1), F32) if dbg_o else None
        o = _listed_fused(lg, arena_blocks, ks, qs, lslots, win_slots, win_keys, o_out, win_dep)
        _attention_out(lg, lg.view(o, (D, W)))
        return lg.g
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
    lg.node(_slot_merge(W, T), [pv, mx, ls, lslots], [o2, m2, l2])
    _attention_comb(lg, o2, m2, l2)
    return lg.g


def _slot_merge(W: int, T: int) -> str:
    """The slot merge build for W columns of T tokens when column c's slot is c // heads (one slot per column, in
    column order): one member per column for few columns, one per 8 columns (a token's heads) for the other decode
    steps, else 64 columns per member. Only ds_slot_ml1 scans the slots."""
    if W <= 32:
        return "rg_ds_slot_ml1_f32"
    return "rg_ds_slot_ml8_f32" if T <= DECODE_TOKENS else "rg_ds_slot_mld_f32"


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
    wstored: list[Tensor] = []  # the window store marker, so the fused gather reads after the store
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
        store_out=wstored.append,
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
        win_slots=slots,
        win_keys=keys,
        win_dep=wstored[0] if wstored else None,
    )
    return lg.g
