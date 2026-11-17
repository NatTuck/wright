"""Synapse graphs of Qwen3.8-Flash-Next (qwen4_exp) on Gaudi: one launch per sub-layer of a step of T columns.

The residual is four hyper-connection streams ``X`` (4H, T) f32. Every sub-layer graph begins where the previous
one ended: ``hc_combine`` adds the previous module's output ``p`` (H, T, NR: one rank's partial, or the ranks'
all-gathered partials, summed here) into the streams with the previous module's inject gates ``lo`` and norms
the result with this module's hyper-connection weight; ``hc_lo`` / the down and up GEMMs / ``hc_mix`` turn the
streams into the module's input ``m`` (H, T); the module runs; its output is this graph's ``p_out`` and its inject
gates ``lo_out``. So the streams ``X`` / ``X_out``, ``lo`` / ``lo_out`` and ``p`` / ``p_out`` ping-pong between
launches and no graph needs the whole layer.

Modules: the delta net (``gdn``: the Qwen3.8 kernels with the sigmoid output gate), attention (``attn_pre``:
projections, Q/K norm + RoPE, the K/V page store, the QSA indexer's raw and block keys; the slot launches of
``common.layers.attn_slots_graph``; ``attn_post``: the output gate and projection), the MoE (``moe``: softmax
router, gated shared expert, the local routed experts through the work-list, per-token or paged kernels), the
PLE at its layer (inside ``attn_pre``: n-gram rows gathered from the table, gate and dilated conv on the
streams), the embedding (``embed``) and the head (``head``: the final mixer and the LM head slice).
All dense GEMMs are bf16 x bf16 -> f32 (the checkpoint's dense weights are BF16); activations for them come out of
the TPC kernels in sigma order and the weights' K axes are permuted to match (``checkpoint``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from resin.models.qwen4_exp.config import Qwen4ExpConfig
from resin_models_gaudi.common import layers as Q
from resin_models_gaudi.common.layers import (
    ARENA_BLOCK_ROWS,
    GDN_CHUNK,
    GEMM_NN,
    GEMM_NT,
    SLOT_KEYS,
)
from resin_models_gaudi.common.sampling import PARAMS, sampler
from resin_models_gaudi.qwen4_exp.checkpoint import HCR, Shard
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, U8, Device

NS = 4  # hyper-connection streams
ROW_BYTES = 512
IX_DIM = 128  # QSA indexer head dim (raw and block keys: one 512-byte row each)
WORK_LIST_PAIRS = 50  # (token, route) pairs the work-list MoE kernels take per launch
PER_TOKEN_MAX = 128  # columns through the per-token MoE kernels before the paged path


@dataclass(frozen=True, slots=True)
class Dims:
    """This rank's model dimensions, as the graphs see them."""

    hidden: int
    vocab: int  # this rank's LM head rows
    vocab_all: int  # the embedding table (replicated)
    table_rows: int  # this rank's n-gram table rows + the zero row (0: no table)
    heads: int  # attention heads held
    kv_groups: int  # KV groups held
    head_dim: int
    nk: int  # delta-net key heads held
    nv: int  # value heads held
    experts: int  # all experts (router width)
    local_experts: int
    expert_lo: int
    top: int
    ffn: int  # routed expert width
    shared: int  # shared expert width
    ix_heads: int
    ple_heads: int
    ple_dim: int
    ple_conv: int
    eps: float
    ranks: int
    rank: int
    dense_ranks: int = 1  # ranks the dense weights are split over (1: replicated)
    gather_tokens: int = 0

    @classmethod
    def from_config(
        cls,
        c: Qwen4ExpConfig,
        shard: Shard,
        *,
        table_rows: int = 0,
        gather_tokens: int = 0,
    ) -> Dims:
        n = shard.nranks
        lo, hi = shard.vocab_range
        e0, e1 = shard.expert_range
        return cls(
            hidden=c.hidden_size,
            vocab=hi - lo,
            vocab_all=c.vocab_size,
            table_rows=table_rows,
            heads=len(shard.attn_heads),
            kv_groups=len(shard.attn_groups),
            head_dim=c.head_dim,
            nk=shard.local_nk,
            nv=shard.local_nv,
            experts=c.num_experts,
            local_experts=e1 - e0,
            expert_lo=e0,
            top=c.num_experts_per_tok,
            ffn=c.moe_intermediate_size,
            shared=c.shared_expert_intermediate_size,
            ix_heads=c.indexer_n_heads,
            ple_heads=c.ple_heads,
            ple_dim=c.ple_head_dim,
            ple_conv=c.ple_conv_kernel_size,
            eps=c.rms_norm_eps,
            ranks=n,
            rank=shard.rank,
            dense_ranks=shard.dense_ranks,
            gather_tokens=gather_tokens,
        )

    @property
    def hpg(self) -> int:
        return self.heads // self.kv_groups

    @property
    def conv_dim(self) -> int:
        return (2 * self.nk + self.nv) * 128

    @property
    def gdn_proj(self) -> int:
        """rows of the delta-net input projection: [q | k | v | z]"""
        return self.conv_dim + 128 * self.nv

    @property
    def attn_proj(self) -> int:
        """rows of the attention projection: [q_h | gate_h] x heads, k x groups, v x groups"""
        return 2 * self.head_dim * self.heads + 2 * self.head_dim * self.kv_groups

    @property
    def ix_proj(self) -> int:
        """the indexer rows after the attention projection: [q (ix_heads x 128) | k (128)]"""
        return (self.ix_heads + 1) * IX_DIM

    @property
    def ssm_rows(self) -> int:
        return 128 * self.nv

    @property
    def conv_rows(self) -> int:
        return self.conv_dim * 3 // 128

    @property
    def ple_rows(self) -> int:
        """arena rows of the PLE conv state (4H x 9 f32)"""
        return NS * self.hidden * 9 // 128

    def gathered(self, tokens: int) -> bool:
        return self.ranks > 1 and tokens <= self.gather_tokens

    def partials(self, tokens: int) -> int:
        """rank slices of an incoming partial of a sharded module: all-gathered small launches sum in the graph"""
        return self.ranks if self.gathered(tokens) else 1

    def parts_after(self, module: str, tokens: int) -> int:
        """rank slices of the partial a graph reads after ``module``: the experts are always split over the
        ranks, the dense modules over ``dense_ranks``"""
        sharded = self.ranks > 1 if module == "moe" else self.dense_ranks > 1
        return self.partials(tokens) if sharded else 1

    def qwen_dims(self) -> Q.Dims:
        """The Qwen3.5 Dims the shared slot attention graph reads (heads, groups, head_dim)."""
        return Q.Dims(
            hidden=self.hidden,
            ffn=0,
            vocab=self.vocab,
            heads=self.heads,
            kv_groups=self.kv_groups,
            head_dim=self.head_dim,
            nk=self.nk,
            nv=self.nv,
            conv_dim=self.conv_dim,
            eps=self.eps,
        )


class LayerGraph:
    """A Graph with the decoder's building blocks."""

    def __init__(
        self, device: Device, name: str, dims: Dims, tokens: int, parts: int = 1
    ) -> None:
        """``parts``: rank slices of the partial the graph's first module reads (``p`` (H, T, parts))"""
        tag = f"_r{dims.ranks}d{dims.dense_ranks}" if dims.ranks > 1 else ""
        self.g = Graph(
            device, f"fnext_{name}{tag}" + (f"_p{parts}" if parts > 1 else "")
        )
        self.d = dims
        self.T = tokens
        self.parts = parts
        self._inputs: dict[str, Tensor] = {}
        self._flow: tuple[Tensor, Tensor, Tensor] | None = (
            None  # the next module's (X, p, lo) within the graph
        )

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

    def cast32(self, x: Tensor) -> Tensor:
        y = self.tensor(x.shape, F32)
        self.node("cast_bf16_to_f32", [x], [y])
        return y

    def meta(self) -> Tensor:
        """[pos0, real tokens, 0, 0] of a prefill chunk; [0, T, 0, 0] of a decode step"""
        return self.input("meta", (4,), I32)

    def one(self) -> Tensor:
        """ones (1, T): the unit per-token scale of the FP8-path kernels fed f32 projections"""
        return self.input("one", (1, self.T), F32)

    def arena32(self, blocks: int) -> Tensor:
        return self.input("arena", (ROW_BYTES // 4, ARENA_BLOCK_ROWS, blocks), F32)

    def arena16(self, blocks: int) -> Tensor:
        return self.input("arena", (ROW_BYTES // 2, ARENA_BLOCK_ROWS, blocks), BF16)

    def gemm(
        self, x: Tensor, name: str, M: int, *, out: Tensor | None = None
    ) -> Tensor:
        """bf16 activation (K, T) x bf16 weight (M, K) -> f32 (M, T)"""
        y = out if out is not None else self.tensor((M, x.shape[1]), F32)
        self.node("gemm", [x, self.input(name, (x.shape[0], M), BF16)], [y], GEMM_NT)
        return y

    # ---- hyper connections
    def streams(self) -> tuple[Tensor, Tensor, Tensor]:
        """the incoming residual streams, the previous module's partial (summed over its rank slices) and
        inject gates: the previous module's within this graph, else the graph's inputs"""
        if self._flow is not None:
            return self._flow
        H, T = self.d.hidden, self.T
        return (
            self.input("X", (NS * H, T), F32),
            self.input("p", (H, T, self.parts), F32),
            self.input("lo", (HCR, T), F32),
        )

    def combine(
        self,
        X: Tensor,
        p: Tensor,
        lo: Tensor,
        norm: str,
        mode: int,
        out: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """mode 1: X + p * inject(lo), mode 2: p copied into every stream, mode 0: X; then the per-stream
        RMSNorm with ``norm`` -> (streams, normed bf16 sigma, 1 / rms per stream)"""
        H, T = self.d.hidden, self.T
        xo = out if out is not None else self.tensor((NS * H, T), F32)
        xn, rt = self.tensor((NS * H, T), BF16), self.tensor((64, NS, T), F32)
        kernel = "rg_hc_combine_t_f32" if T >= 64 else "rg_hc_combine_f32"
        self.node(
            kernel,
            [X, p, lo, self.input(norm, (NS * H,), F32)],
            [xo, xn, rt],
            struct.pack("fi", self.d.eps, mode),
        )
        return xo, xn, rt

    def hc(
        self, X: Tensor, xn: Tensor, rt: Tensor, tag: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """the module's input mix -> (m16 (H, T) bf16 sigma, m32 (H, T) f32, lo (HCR, T) f32: this module's
        down projection, whose rows 320 .. 323 are its inject gates)"""
        H, T = self.d.hidden, self.T
        lo = self.gemm(xn, f"{tag}.dn", HCR)
        lb = self.tensor((HCR, T), BF16)
        self.node("rg_hc_lo_f32", [lo], [lb])
        up = self.gemm(lb, f"{tag}.up", NS * H)
        m16, m32 = self.tensor((H, T), BF16), self.tensor((H, T), F32)
        self.node(
            "rg_hc_mix_f32",
            [X, self.input(f"{tag}.w", (NS * H,), F32), up, rt],
            [m16, m32],
        )
        return m16, m32, lo

    def enter(self, norm: str, *, last: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """a module's start: the streams updated by the previous module and normed for this one -> (X_out, xn,
        rt); the graph's ``X_out`` when the module is the graph's last, else an internal tensor"""
        X, p, lo = self.streams()
        shape = (NS * self.d.hidden, self.T)
        out = self.input("X_out", shape, F32) if last else self.tensor(shape, F32)
        return self.combine(X, p, lo, norm, 1, out=out)

    def p_target(self, *, last: bool = True) -> Tensor:
        """where a module writes its output: the graph's ``p_out`` (H, T), or an internal tensor"""
        shape = (self.d.hidden, self.T)
        return self.input("p_out", shape, F32) if last else self.tensor(shape, F32)

    def leave(self, X_out: Tensor, p: Tensor, lo: Tensor, *, last: bool = True) -> None:
        """a module's end: its inject gates go out as ``lo_out`` when it is the graph's last; otherwise its
        streams, output (one slice) and gates feed the next module of the graph"""
        if last:
            self.node("memcpy", [lo], [self.input("lo_out", (HCR, self.T), F32)])
        else:
            self._flow = (X_out, self.view(p, (self.d.hidden, self.T, 1)), lo)

    # ---- arena state
    def state_load(self, arena: Tensor, bases: str, rows: int) -> Tensor:
        out = self.tensor((128, rows, 1), F32)
        self.node("rg_state_load_f32", [arena, self.input(bases, (1,), I32)], [out])
        return out

    def state_store(self, arena: Tensor, bases: str, vals: Tensor, sink: str) -> None:
        self.node(
            "rg_state_store_f32",
            [arena, self.input(bases, (1,), I32), vals],
            [self.input(sink, (64, 1), I32)],
        )


# ------------------------------------------------------------------ embedding, head
def embed_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """tokens -> the streams: the embedding copied into every stream (``hc_combine`` mode 2 over zero streams)"""
    lg = LayerGraph(device, f"embed_T{tokens}", dims, tokens)
    H, T = dims.hidden, tokens
    e = lg.tensor((H, T), BF16)
    lg.node(
        "gather_fwd_bf16",
        [lg.input("E_gather", (H, dims.vocab_all), BF16), lg.input("tok", (T,), I32)],
        [e],
        struct.pack("i", 1),
    )
    e3 = lg.view(lg.cast32(e), (H, T, 1))
    lg.combine(
        lg.input("zx", (NS * H, T), F32),
        e3,
        lg.input("zl", (HCR, T), F32),
        "hch.w",
        2,
        out=lg.input("X_out", (NS * H, T), F32),
    )
    return lg.g


def head_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    columns: int,
    *,
    sample: bool = True,
    parts: int = 1,
) -> Graph:
    """the last module's output into the streams, the final mixer and the LM head over ``columns`` columns
    (``cols`` (columns,) i32) -> ``logits`` (vocab rows held, columns) f32, and with ``sample`` the device sampler
    over them (``samp_params`` (PARAMS, columns) -> ``tok_out``, ``lp_out``, ``top_idx``, ``top_lp``)"""
    lg = LayerGraph(
        device,
        f"head_T{tokens}_C{columns}{'_s' if sample else ''}",
        dims,
        tokens,
        parts,
    )
    H, T = dims.hidden, tokens
    xo, xn, rt = lg.enter("hch.w")
    m16, _, _ = lg.hc(xo, xn, rt, "hch")
    if columns != T:
        picked = lg.tensor((H, columns), BF16)
        lg.node(
            "gather_fwd_bf16",
            [m16, lg.input("cols", (columns,), I32)],
            [picked],
            struct.pack("i", 1),
        )
        m16 = picked
    logits = lg.gemm(
        m16, "E_head", dims.vocab, out=lg.input("logits", (dims.vocab, columns), F32)
    )
    if sample:
        if dims.ranks > 1:
            raise ValueError(
                "the sampler takes the all-gathered logits of every rank (sample_graph)"
            )
        sampler(lg.g, logits, lg.input("samp_params", (PARAMS, columns), F32))
    return lg.g


def sample_graph(device: Device, dims: Dims, columns: int) -> Graph:
    """The device sampler over the ranks' all-gathered logits slices ``logits_all`` (V_r, columns, ranks) f32
    (rank-major, as the all-gather leaves them) -> ``tok_out``, ``lp_out``, ``top_idx``, ``top_lp``."""
    d = dims
    lg = LayerGraph(device, f"sample_C{columns}", d, columns)
    perm = lg.tensor((d.vocab, d.ranks, columns), F32)
    lg.node(
        "rg_ml4_perm_f32",
        [lg.input("logits_all", (d.vocab, columns, d.ranks), F32)],
        [perm],
    )
    full = lg.view(perm, (d.vocab * d.ranks, columns))
    sampler(lg.g, full, lg.input("samp_params", (PARAMS, columns), F32))
    return lg.g


# ------------------------------------------------------------------ delta net
def gdn_nodes(lg: LayerGraph, blocks: int, *, decode: bool, last: bool = True) -> None:
    """A delta-net module. Decode: T sequences, one token each, the recurrent state read from and written to
    the arena (``gdn_bases`` (4, T) [conv_src, ssm_src, conv_dst, ssm_dst], ``conv_dst`` (T,)). Prefill: one
    sequence's T tokens (a multiple of 64) through the chunked delta rule (``conv_src`` / ``ssm_src`` /
    ``conv_dst`` / ``ssm_dst`` (1,))."""
    d, T = lg.d, lg.T
    xo, xn, rt = lg.enter("hca.w", last=last)
    m16, _, lo = lg.hc(xo, xn, rt, "hca")
    arena = lg.arena32(blocks)
    p = lg.gemm(m16, "w_in", d.gdn_proj)
    conv_w = lg.input("conv_w", (d.conv_dim, 4), F32)
    ssm_a, dt_bias = lg.input("ssm_a", (d.nv,), F32), lg.input("dt_bias", (d.nv,), F32)
    ssm_norm = lg.input("ssm_norm", (128,), F32)
    prm = struct.pack("fi", d.eps, d.nk)
    g16 = lg.tensor(
        (d.nv * 128, T), BF16
    )  # the module output in sigma order (feeds w_out)
    if decode:
        gn = lg.tensor((d.nv * 128, T), F32)
        ba = lg.gemm(m16, "w_ba", 2 * d.nv)
        cst_out = lg.tensor((d.conv_dim, 3, T), F32)
        lg.node(
            "rg_gdn_decode_sg_f32",
            [
                p,
                lg.one(),
                ba,
                conv_w,
                arena,
                lg.input("gdn_bases", (4, T), I32),
                ssm_a,
                dt_bias,
                ssm_norm,
            ],
            [gn, cst_out],
            prm,
        )
        lg.node(
            "rg_state_store_f32",
            [
                arena,
                lg.input("conv_dst", (T,), I32),
                lg.view(cst_out, (128, d.conv_rows, T)),
            ],
            [lg.input("sink_conv", (64, 1), I32)],
        )
        # the decode kernel writes f32 in natural order: to bf16 in sigma order (the prefill kernel's)
        lg.node(
            "gather_fwd_bf16",
            [lg.cast16(gn), lg.input("sigma_idx", (d.nv * 128,), I32)],
            [g16],
            struct.pack("i", 0),
        )
    else:
        cst_in = lg.view(lg.state_load(arena, "conv_src", d.conv_rows), (d.conv_dim, 3))
        sst_in = lg.view(
            lg.state_load(arena, "ssm_src", d.ssm_rows), (128, 128, d.nv, 1)
        )
        cst_out, sst_out = (
            lg.tensor((d.conv_dim, 3), F32),
            lg.tensor((128, 128, d.nv, 1), F32),
        )
        baT = lg.tensor((T, 2 * d.nv), F32)
        lg.node(
            "gemm", [lg.input("w_ba", (d.hidden, 2 * d.nv), BF16), m16], [baT], GEMM_NT
        )
        _gdn_chunked(
            lg,
            p,
            baT,
            conv_w,
            cst_in,
            sst_in,
            ssm_a,
            dt_bias,
            ssm_norm,
            g16,
            cst_out,
            sst_out,
        )
        lg.state_store(
            arena, "conv_dst", lg.view(cst_out, (128, d.conv_rows, 1)), "sink_conv"
        )
        lg.state_store(
            arena, "ssm_dst", lg.view(sst_out, (128, d.ssm_rows, 1)), "sink_ssm"
        )
    out = lg.gemm(g16, "w_out", d.hidden, out=lg.p_target(last=last))
    lg.leave(xo, out, lo, last=last)


def gdn_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    *,
    decode: bool,
    parts: int = 1,
) -> Graph:
    """A delta-net module as one launch (``gdn_nodes``)."""
    lg = LayerGraph(
        device,
        f"gdn_{'dec' if decode else 'pre'}_T{tokens}_NB{blocks}",
        dims,
        tokens,
        parts,
    )
    gdn_nodes(lg, blocks, decode=decode)
    return lg.g


def _gdn_chunked(
    lg, p, baT, conv_w, cst_in, sst_in, ssm_a, dt_bias, ssm_norm, out, cst_out, sst_out
) -> None:
    """The chunked (WY) gated delta rule of one prefill chunk on the bf16-path kernels, sigmoid output gate
    (``qwen3_5.layers.gdn_chunked`` for the FP8 path); ``out`` (128 nv, T) bf16 in sigma order."""
    d, T = lg.d, lg.T
    if T % GDN_CHUNK:
        raise ValueError(
            f"a prefill chunk of {T} tokens is not a multiple of {GDN_CHUNK}"
        )
    CH, NC, nk, nv = GDN_CHUNK, T // GDN_CHUNK, d.nk, d.nv
    blk, sq, gram = (128, CH, nv, NC), (128, 128, nv, NC), (CH, CH, nv, NC)

    def bg(x, w, ta, tb, shape):
        y = lg.tensor(shape, F32)
        lg.node("batch_gemm", [x, w], [y], struct.pack("??", ta, tb))
        return y

    Vn = lg.tensor(blk, F32)
    GB = lg.tensor((CH, 2, nv, NC), F32)
    cst_qk, cst_v = lg.tensor((2 * nk * 128, 3), F32), lg.tensor((nv * 128, 3), F32)
    prm = struct.pack("fi", d.eps, nk)
    kblk = (128, CH, nk, NC)
    Qk, Kk = lg.tensor(kblk, F32), lg.tensor(kblk, F32)
    meta = lg.meta()
    lg.node("rg_gdn_ck_qk_f32", [p, conv_w, cst_in, meta], [Qk, Kk, cst_qk], prm)
    if nv == nk:
        Qn, Kn = Qk, Kk
    else:
        Qn, Kn = lg.tensor(blk, F32), lg.tensor(blk, F32)
        lg.node("concat", [Qk] * (nv // nk), [Qn], struct.pack("I", 2))
        lg.node("concat", [Kk] * (nv // nk), [Kn], struct.pack("I", 2))
    lg.node(
        "rg_gdn_ck_v_f32",
        [p, baT, conv_w, cst_in, ssm_a, dt_bias, meta],
        [GB, Vn, cst_v],
        prm,
    )
    lg.node("concat", [cst_qk, cst_v], [cst_out], struct.pack("I", 0))
    kgram = (CH, CH, nk, NC)
    KK = bg(Kk, Kk, False, True, kgram)
    QK = bg(Qk, Kk, False, True, kgram)
    P, Tb, Tbg, Tkb, Tkbg = (lg.tensor(gram, F32) for _ in range(5))
    lg.node("rg_gdn_ck_solve_f32", [KK, QK, GB], [P, Tb, Tbg, Tkb, Tkbg])
    U = bg(Tb, Vn, False, False, blk)
    W = bg(Tbg, Kn, False, False, blk)
    B = bg(Kn, bg(Tkb, Vn, False, False, blk), True, False, sq)
    KW = bg(Kn, bg(Tkbg, Kn, False, False, blk), True, False, sq)
    one = (128, 128, nv, 1)

    def chunks(x, shape):
        if NC == 1:
            return [x]
        parts = [lg.tensor(shape, F32) for _ in range(NC)]
        lg.node("split", [x], parts, struct.pack("I", 3))
        return parts

    KWc, Bc, GBc = chunks(KW, one), chunks(B, one), chunks(GB, (CH, 2, nv, 1))
    S, states = sst_in, [sst_in]
    for c in range(NC):
        X = bg(KWc[c], S, False, False, one)
        y = lg.tensor(one, F32)
        lg.node("rg_gdn_ck_step_f32", [S, X, Bc[c], GBc[c]], [y])
        S = y
        if c < NC - 1:
            states.append(y)
    lg.node("memcpy", [S], [sst_out])
    if NC > 1:
        Sall = lg.tensor(sq, F32)
        lg.node("concat", states, [Sall], struct.pack("I", 3))
    else:
        Sall = sst_in
    QS = bg(Qn, Sall, False, False, blk)
    D = lg.tensor(blk, F32)
    lg.node("sub_fwd_f32", [U, bg(W, Sall, False, False, blk)], [D])
    PD = bg(P, D, False, False, blk)
    lg.node(
        "rg_gdn_ck_out_sg_f32",
        [QS, PD, GB, p, ssm_norm, meta],
        [out],
        struct.pack("fi", d.eps, nk),
    )


# ------------------------------------------------------------------ attention
def _attn_pre_nodes(
    lg: LayerGraph,
    blocks: int,
    positions: int,
    *,
    q_all: Tensor | None = None,
    last: bool = True,
):
    """The attention module's first half: streams in, the projections (kept in ``attn_p`` for the gate and the
    indexer), Q/K norm + RoPE at each column's position (``pos`` (T,) into ``cs_tab`` (128, positions)), the K/V
    rows stored into their pages (``kv_rows`` (8, T)) -> (the queries (head_dim, W G) bf16 (``q_all``, or an
    internal tensor), the store's sink, the streams out, the inject gates). With ``last`` the gates go out as
    ``lo_out`` here (the module's second half runs as its own launch)."""
    d, T = lg.d, lg.T
    G, hpg = d.kv_groups, d.hpg
    Tv = 128 if 1 < T < 128 else T
    W = hpg * T
    xo, xn, rt = lg.enter("hca.w", last=last)
    m16, _, lo = lg.hc(xo, xn, rt, "hca")
    p = lg.gemm(
        m16,
        "w_qkv",
        d.attn_proj + d.ix_proj,
        out=lg.input("attn_p", (d.attn_proj + d.ix_proj, T), F32),
    )
    cs = lg.tensor((128, T), F32)
    lg.node(
        "gather_fwd_f32",
        [lg.input("cs_tab", (128, positions), F32), lg.input("pos", (T,), I32)],
        [cs],
        struct.pack("i", 1),
    )
    qs = [lg.tensor((d.head_dim, W), BF16) for _ in range(G)]
    kn = [lg.tensor((d.head_dim, T), BF16) for _ in range(G)]
    vn = [lg.tensor((d.head_dim, Tv), BF16) for _ in range(G)]
    kernel = "rg_attn_prep_f32" if G == 2 else f"rg_attn_prep_g{G}_f32"
    lg.node(
        kernel,
        [p, lg.input("qk_norm", (2 * d.head_dim,), F32), cs, lg.meta()],
        qs + kn + vn,
        struct.pack("fi", d.eps, hpg),
    )
    arena = lg.arena16(blocks)
    spare = 4 - G
    sink = lg.input("sink_kv", (64, 1), I32)
    lg.node(
        "rg_kv_store_f32",
        [
            arena,
            lg.input("kv_rows", (8, T), I32),
            *kn,
            *kn[:1] * spare,
            *vn,
            *vn[:1] * spare,
        ],
        [sink],
    )
    q = q_all if q_all is not None else lg.tensor((d.head_dim, W * G), BF16)
    lg.node("concat", qs, [q], struct.pack("I", 1))
    return q, sink, xo, lo


def attn_pre_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    positions: int,
    *,
    parts: int = 1,
) -> Graph:
    """The attention module's first launch of a prefill chunk: ``_attn_pre_nodes`` into ``q_all`` plus the
    accumulators of the slot launches (``attn_slots_graph``), whose result ``attn_post_graph`` takes."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"attn_pre_T{T}_NB{blocks}_P{positions}", d, T, parts)
    G, W = d.kv_groups, d.hpg * T
    _, _, _, lo = _attn_pre_nodes(
        lg, blocks, positions, q_all=lg.input("q_all", (d.head_dim, W * G), BF16)
    )
    lg.node("memcpy", [lo], [lg.input("lo_out", (HCR, T), F32)])
    acc = [
        lg.input("acc_o", (d.head_dim, W, G), F32),
        lg.input("acc_m", (W, G), F32),
        lg.input("acc_l", (W, G), F32),
    ]
    lg.node("rg_attn_acc_init_f32", acc, [lg.input("sink_acc", (64, 1), I32)])
    return lg.g


def attn_decode_nodes(
    lg: LayerGraph, blocks: int, positions: int, slots: int, *, last: bool = True
) -> None:
    """The whole attention module of a decode step: ``_attn_pre_nodes``, then every row's slot (``slots`` (8, S)
    i32 [group, qcol0, qpos0, key0, klen, qtok0], ``k_pages`` / ``v_pages`` (16, S): the keys a row sees, up to
    2048 a slot) gathered from the pages (after the K/V store, by a data edge), scored, softmaxed and reduced per
    slot, then ``_attn_post_nodes``. Slots of group -1 are padding."""
    d, T, S = lg.d, lg.T, slots
    G, hpg = d.kv_groups, d.hpg
    W, M, C = hpg * T, hpg, SLOT_KEYS
    q_all, sink, xo, lo = _attn_pre_nodes(lg, blocks, positions, last=last)
    table = lg.input("slots", (8, S), I32)
    q = lg.tensor((d.head_dim, M, S), BF16)
    lg.node("rg_attn_slot_q_f32", [q_all, table], [q], struct.pack("ii", G, 0))
    arena = lg.arena16(blocks)
    ks = lg.tensor((d.head_dim, C, S), BF16)
    lg.node(
        "rg_kv_gather_d_f32",
        [arena, lg.input("k_pages", (C // 128, S), I32), table, sink],
        [ks],
        struct.pack("i", 0),
    )
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, S), BF16)
    Mp = (M + 63) // 64 * 64
    mx, ls = lg.tensor((Mp, S), F32), lg.tensor((Mp, S), F32)
    lg.node(
        "rg_attn_slot_p_f32",
        [sc, table],
        [P, mx, ls],
        struct.pack("fi", 1.0 / d.head_dim**0.5, hpg),
    )
    vs = lg.tensor((d.head_dim, C, S), BF16)
    lg.node(
        "rg_kv_gather_d_f32",
        [arena, lg.input("v_pages", (C // 128, S), I32), table, sink],
        [vs],
        struct.pack("i", 1),
    )
    pv = lg.tensor((d.head_dim, M, S), F32)
    lg.node("batch_gemm", [P, vs], [pv], GEMM_NN)
    on = lg.tensor((d.head_dim, W, G), F32)
    lg.node("rg_attn_slot_fin_f32", [pv, ls, table], [on])
    out = _attn_post_nodes(lg, blocks, positions, on, decode=True, last=last)
    lg.leave(xo, out, lo, last=last)


def attn_decode_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    positions: int,
    slots: int,
    *,
    parts: int = 1,
) -> Graph:
    """The attention module of a decode step as one launch (``attn_decode_nodes``)."""
    lg = LayerGraph(
        device,
        f"attn_dec_T{tokens}_NB{blocks}_P{positions}_S{slots}",
        dims,
        tokens,
        parts,
    )
    attn_decode_nodes(lg, blocks, positions, slots)
    return lg.g


def ple_embed_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """The first half of a multi-rank PLE launch: every column's rows of this rank's table (``prow``) gathered
    and converted -> ``ple_e`` (256 heads, T) bf16, which the ranks sum (every head's row lives on one rank, the
    others gather the zero row) before ``ple_graph(..., gathered=True)`` projects it."""
    d, T = dims, tokens
    lg = LayerGraph(device, f"ple_embed_T{T}", d, T)
    NH, D = d.ple_heads, d.ple_dim
    idx = lg.input("prow", (NH * T,), I32)
    e8 = lg.tensor((D, NH * T), U8)
    lg.node(
        "gather_fwd_u8",
        [lg.input("ple.tab", (D, d.table_rows), U8), idx],
        [e8],
        struct.pack("i", 1),
    )
    lg.node("rg_ple_cvt_f32", [e8], [lg.input("ple_e", (256 * NH, T), BF16)])
    return lg.g


def ple_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    *,
    decode: bool,
    gathered: bool = False,
    parts: int = 1,
) -> Graph:
    """The PLE layer's n-gram embedding into the streams, before its attention module: the previous module's output
    combined in, every column's rows (``prow`` (heads T,) i32: this rank's table rows, or its zero row) gathered from
    ``ple.tab`` (D, rows + 1) u8 and converted, the key / value projection (the table's scale folded in), the gate and
    the dilated conv with its state in the STATE block (``ple_src`` / ``ple_dst`` (B,): a row per sequence at decode, one for a
    prefill chunk). Without a table the projection input is zeros (``zkv``); with ``gathered`` the converted
    rows come summed over the ranks in ``ple_e`` (``ple_embed_graph``). The streams go out un-normed
    (``X_out``); the attention launch combines them with a zero partial and zero gates."""
    d, T = dims, tokens
    lg = LayerGraph(
        device,
        f"ple_{'dec' if decode else 'pre'}_T{T}_NB{blocks}{'_g' if gathered else ''}",
        d,
        T,
        parts,
    )
    H, NH, D = d.hidden, d.ple_heads, d.ple_dim
    X, p_in, lo_in = lg.streams()
    x, _, _ = lg.combine(X, p_in, lo_in, "hca.w", 1)
    if gathered:
        kv = lg.gemm(lg.input("ple_e", (256 * NH, T), BF16), "ple.kv", (NS + 1) * H)
    elif d.table_rows:
        idx = lg.input("prow", (NH * T,), I32)
        e8 = lg.tensor((D, NH * T), U8)
        lg.node(
            "gather_fwd_u8",
            [lg.input("ple.tab", (D, d.table_rows), U8), idx],
            [e8],
            struct.pack("i", 1),
        )
        e16 = lg.tensor((256 * NH, T), BF16)
        lg.node("rg_ple_cvt_f32", [e8], [e16])
        if d.ranks > 1:
            raise ValueError(
                "a multi-rank PLE launch takes the gathered rows (ple_embed_graph)"
            )
        kv = lg.gemm(e16, "ple.kv", (NS + 1) * H)
    else:
        kv = lg.input("zkv", ((NS + 1) * H, T), F32)
    gv, gvn = lg.tensor((NS * H, T), F32), lg.tensor((NS * H, T), F32)
    norms = [lg.input(f"ple.n{n}", (NS * H,), F32) for n in "kqc"]
    lg.node("rg_ple_gate_f32", [x, kv, *norms], [gv, gvn], struct.pack("f", d.eps))
    arena = lg.arena32(blocks)
    B = T if decode else 1
    st_in = lg.tensor((128, d.ple_rows, B), F32)
    lg.node("rg_state_load_f32", [arena, lg.input("ple_src", (B,), I32)], [st_in])
    # (an output takes the kernel's shape; a view of it feeds the store: a view is a reshape node)
    st_out = lg.tensor((NS * H, 9, B) if decode else (NS * H, 9), F32)
    xo = lg.input("X_out", (NS * H, T), F32)
    conv = lg.input("ple.conv", (NS * H, d.ple_conv), F32)
    if decode:
        lg.node(
            "rg_ple_conv_b_f32",
            [x, gv, gvn, conv, lg.view(st_in, (NS * H, 9, B)), lg.meta()],
            [xo, st_out],
        )
    else:
        lg.node(
            "rg_ple_conv_f32",
            [x, gv, gvn, conv, lg.view(st_in, (NS * H, 9)), lg.meta()],
            [xo, st_out],
        )
    lg.node(
        "rg_state_store_f32",
        [
            arena,
            lg.input("ple_dst", (B,), I32),
            lg.view(st_out, (128, d.ple_rows, B)),
        ],
        [lg.input("sink_ple", (64, 1), I32)],
    )
    return lg.g


def _attn_post_nodes(
    lg: LayerGraph,
    blocks: int,
    positions: int,
    acc: Tensor,
    *,
    decode: bool,
    last: bool = True,
) -> Tensor:
    """The attention module's second half: the output gate on the attention result ``acc`` (head_dim, W, G) and
    the output projection -> ``p_out``; and the QSA indexer's keys of the step into their pages: each column's
    raw key (``ix_rows`` (T,)) and the block keys of the blocks the step touches (``kb_rows`` (4 nbc,) at
    prefill: the chunk's blocks; (4 T,) at decode: each row's current block, recomputed from its earlier keys
    ``kb_tail`` (T,) read from the raw-key page). Rows of -1 skip the store."""
    d, T = lg.d, lg.T
    G, W = d.kv_groups, d.hpg * T
    parts = [lg.tensor((d.head_dim, W, 1), F32) for _ in range(G)]
    lg.node("split", [acc], parts, struct.pack("I", 2))
    on = [lg.view(part, (d.head_dim, W)) for part in parts]
    p = lg.input("attn_p", (d.attn_proj + d.ix_proj, T), F32)
    ga = lg.tensor((d.head_dim * d.heads, T), BF16)
    lg.node(f"rg_attn_gate_np_g{G}_f32", [*on, p], [ga], struct.pack("i", d.hpg))
    out = lg.gemm(ga, "w_o", d.hidden, out=lg.p_target(last=last))
    arena = lg.arena32(blocks)
    ix_norm = lg.input("ix_norm", (2 * IX_DIM,), F32)
    kofs = d.attn_proj + d.ix_heads * IX_DIM
    if decode:
        tail = lg.tensor((128, 4, T), F32)
        lg.node("rg_state_load_f32", [arena, lg.input("kb_tail", (T,), I32)], [tail])
        iks, kbs = lg.tensor((IX_DIM, T), F32), lg.tensor((IX_DIM, 4, T), F32)
        lg.node(
            "rg_qsa_kb_b_f32",
            [
                p,
                tail,
                lg.input("pos", (T,), I32),
                lg.input("cs_all", (128, positions), F32),
                ix_norm,
            ],
            [iks, kbs],
            struct.pack("fi", d.eps, kofs),
        )
        kb_rows = 4 * T
    else:
        nbc = T // 4 + 1
        iks, kbs = lg.tensor((IX_DIM, T), F32), lg.tensor((IX_DIM, 4 * nbc), F32)
        lg.node(
            "rg_qsa_kb_f32",
            [
                p,
                lg.input("zik", (IX_DIM, 4), F32),
                lg.input("cs_all", (128, positions), F32),
                ix_norm,
                lg.meta(),
            ],
            [iks, kbs],
            struct.pack("fi", d.eps, kofs),
        )
        kb_rows = 4 * nbc
    lg.node(
        "rg_page_store_f32",
        [arena, lg.input("ix_rows", (T,), I32), iks],
        [lg.input("sink_ix", (64, 1), I32)],
    )
    lg.node(
        "rg_page_store_f32",
        [arena, lg.input("kb_rows", (kb_rows,), I32), lg.view(kbs, (IX_DIM, kb_rows))],
        [lg.input("sink_kb", (64, 1), I32)],
    )
    return out


def attn_post_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    positions: int,
    *,
    decode: bool,
) -> Graph:
    """The attention module's last launch of a prefill chunk, on the slot launches' accumulated ``acc_o``."""
    d, T = dims, tokens
    lg = LayerGraph(
        device,
        f"attn_post_{'dec' if decode else 'pre'}_T{T}_NB{blocks}_P{positions}",
        d,
        T,
    )
    acc = lg.input("acc_o", (d.head_dim, d.hpg * T, d.kv_groups), F32)
    _attn_post_nodes(lg, blocks, positions, acc, decode=decode)
    return lg.g


# ------------------------------------------------------------------ MoE
def moe_nodes(lg: LayerGraph) -> None:
    """The MoE module, a graph's last: the previous module's output into the streams, the router (softmax over
    the top K of E logits; row E is the shared expert's gate), the shared expert, the local routed experts
    (``w12`` / ``s12`` / ``w3`` / ``s3`` packed by ``checkpoint.pack_experts``) -> ``p_out`` (this rank's partial;
    rank 0 adds the shared expert)."""
    d, T = lg.d, lg.T
    H, K, E, El, F = d.hidden, d.top, d.experts, d.local_experts, d.ffn
    NB, NI, NO = F // 128, H // 128, H // 256
    xo, xn, rt = lg.enter("hcm.w")
    m16, m32, lo = lg.hc(xo, xn, rt, "hcm")
    lg.leave(xo, lg.p_target(), lo)
    lgt = lg.gemm(m16, "router", E + 1)
    ids, wts = lg.tensor((64, T), I32), lg.tensor((64, T), F32)
    lg.node("rg_moe_route_f32", [lgt, lg.meta()], [ids, wts], struct.pack("i", K))
    gu = lg.gemm(m16, "sh_gu", 2 * d.shared)
    sg = lg.tensor((d.shared, T), BF16)
    lg.node("rg_swiglu_f32", [gu], [sg])
    sh = lg.gemm(sg, "sh_down", H)
    W12 = lg.input("w12", (256, 128, NB, NI, El), F8)
    S12 = lg.input("s12", (64, 2, NB, NI, El), F32)
    W3 = lg.input("w3", (256, 128, NO, NB, El), F8)
    S3 = lg.input("s3", (64, 2, NO, NB, El), F32)
    addsh = int(d.rank == 0)
    y = lg.p_target()
    if T * K <= WORK_LIST_PAIRS:
        hp = lg.tensor((256, NB, NI, K, T), F32)
        lg.node(
            "rg_moe_up_w_f32",
            [m32, W12, S12, ids],
            [hp],
            struct.pack("ii", d.expert_lo, K),
        )
        yk = lg.tensor((H, NB * K, T), F32)
        lg.node(
            "rg_moe_down_w_f32",
            [hp, W3, S3, ids, wts],
            [yk],
            struct.pack("ii", d.expert_lo, K),
        )
        lg.node(
            "rg_moe_sum_w_f32",
            [yk, sh, lgt, ids],
            [y],
            struct.pack("iiiii", addsh, E, d.expert_lo, El, K),
        )
    elif T <= PER_TOKEN_MAX:
        hh = lg.tensor((128, NB, K, T), F32)
        lg.node(
            "rg_moe_up_f32",
            [m32, W12, S12, ids],
            [hh],
            struct.pack("ii", d.expert_lo, K),
        )
        lg.node(
            "rg_moe_down_f32",
            [hh, W3, S3, ids, wts, sh, lgt],
            [y],
            struct.pack("iiii", d.expert_lo, K, addsh, E),
        )
    else:
        _paged_experts(lg, m16, m32, ids, wts, sh, lgt, W12, S12, W3, S3, y, addsh)


def moe_graph(device: Device, dims: Dims, tokens: int, *, parts: int = 1) -> Graph:
    """The MoE module as one launch (``moe_nodes``)."""
    lg = LayerGraph(device, f"moe_T{tokens}", dims, tokens, parts)
    moe_nodes(lg)
    return lg.g


def layer_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    positions: int,
    *,
    kind: str,
    decode: bool,
    slots: int = 0,
    parts: int = 1,
) -> Graph:
    """A whole layer as one launch when its module's output needs no collective (the dense weights replicated,
    or one rank): the module (``kind`` "gdn", or "attn" of a decode step with ``slots`` slots) feeding the MoE
    within the graph; ``p_out`` is the MoE's partial."""
    lg = LayerGraph(
        device,
        f"layer_{kind}_{'dec' if decode else 'pre'}_T{tokens}_NB{blocks}_S{slots}",
        dims,
        tokens,
        parts,
    )
    if kind == "gdn":
        gdn_nodes(lg, blocks, decode=decode, last=False)
    elif kind == "attn" and decode:
        attn_decode_nodes(lg, blocks, positions, slots, last=False)
    else:
        raise ValueError(
            f"no merged graph for {kind} at {'decode' if decode else 'prefill'}"
        )
    moe_nodes(lg)
    return lg.g


def paged_shape(
    tokens: int, top: int, local_experts: int, ranks: int
) -> tuple[int, int]:
    """(row shift, pages): pages of 2^shift (token, route) rows, enough for twice the average share"""
    psh = 8
    PS = 1 << psh
    P = min(
        tokens * top // PS + local_experts,
        -(-2 * tokens * top // (PS * ranks)) + local_experts,
    )
    return psh, P


def _paged_experts(lg, m16, m32, ids, wts, sh, lgt, W12, S12, W3, S3, y, addsh) -> None:
    """Prefill experts on the MME: per expert, pages of its (token, route) rows; each page's expert dequantized to
    bf16; batched GEMMs over the pages; SwiGLU; weighted combine. Pages beyond the budget (routing is uneven)
    and the shared expert go through the per-token kernels."""
    d, T = lg.d, lg.T
    H, K, E, El, F = d.hidden, d.top, d.experts, d.local_experts, d.ffn
    NB = F // 128
    psh, P = paged_shape(T, K, El, d.ranks)
    PS = 1 << psh
    meta = lg.meta()
    cntm = lg.tensor((El, 24), I32)
    lg.node("rg_moe_cnt_f32", [ids, meta], [cntm], struct.pack("ii", d.expert_lo, K))
    po, pe = lg.tensor((El,), I32), lg.tensor((P,), I32)
    lg.node("rg_moe_pages_f32", [cntm], [po, pe], struct.pack("i", psh))
    gidx, bm = lg.tensor((PS * P,), I32), lg.tensor((64, T), I32)
    lg.node(
        "rg_moe_asg_f32",
        [ids, meta, cntm, po, pe],
        [gidx, bm],
        struct.pack("iii", d.expert_lo, K, psh),
    )
    wb12, wb3 = lg.tensor((2 * F, H, P), BF16), lg.tensor((H, F, P), BF16)
    lg.node("rg_moe_deq_p_f32", [W12, S12, pe], [wb12], struct.pack("ii", 128, F))
    lg.node("rg_moe_deq_p_f32", [W3, S3, pe], [wb3], struct.pack("ii", 256, 128))
    xpad = lg.tensor((H, T + 1), BF16)
    lg.node(
        "concat", [m16, lg.input("zcol", (H, 1), BF16)], [xpad], struct.pack("I", 1)
    )
    X2 = lg.tensor((H, PS * P), BF16)
    lg.node("gather_fwd_bf16", [xpad, gidx], [X2], struct.pack("i", 1))
    y1 = lg.tensor((2 * F, PS, P), BF16)
    lg.node("batch_gemm", [lg.view(X2, (H, PS, P)), wb12], [y1], GEMM_NN)
    h2 = lg.tensor((F, PS * P), BF16)
    lg.node("rg_moe_act_g_f32", [lg.view(y1, (2 * F, PS * P)), gidx, meta], [h2])
    y2 = lg.tensor((H, PS, P), BF16)
    lg.node("batch_gemm", [lg.view(h2, (F, PS, P)), wb3], [y2], GEMM_NN)
    ido = lg.tensor((64, T), I32)
    lg.node(
        "rg_moe_ovf_f32", [ids, bm, meta], [ido], struct.pack("iii", d.expert_lo, El, K)
    )
    hh = lg.tensor((128, NB, K, T), F32)
    lg.node(
        "rg_moe_up_f32", [m32, W12, S12, ido], [hh], struct.pack("ii", d.expert_lo, K)
    )
    yo = lg.tensor((H, T), F32)
    lg.node(
        "rg_moe_down_f32",
        [hh, W3, S3, ido, wts, sh, lgt],
        [yo],
        struct.pack("iiii", d.expert_lo, K, addsh, E),
    )
    lg.node(
        "rg_moe_combine_p_f32",
        [y2, yo, ids, bm, wts, meta],
        [y],
        struct.pack("iii", d.expert_lo, El, K),
    )


__all__ = [
    "HCR",
    "IX_DIM",
    "NS",
    "PER_TOKEN_MAX",
    "WORK_LIST_PAIRS",
    "Dims",
    "LayerGraph",
    "attn_post_graph",
    "attn_pre_graph",
    "embed_graph",
    "gdn_graph",
    "head_graph",
    "moe_graph",
    "paged_shape",
    "ple_graph",
]
