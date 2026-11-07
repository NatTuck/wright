"""Per-layer Synapse graphs of the FP8 Qwen3.5/3.8 decoder.

Each graph is one layer for T tokens and is reused for every layer of its kind: weights are
persistent inputs bound per launch under generic names (``w_in``, ``w_in.s``, ...). Layers hand
each other a carry, the residual and the next layer's normed input as FP8 bytes, per-token scale
and bf16: inputs ``x h8 hs h16``, outputs ``x_out h8_out hs_out h16_out``.

FP8 projections run as ``fp8_gemm``. Recurrent state lives in the cache arena, read by
``state_load`` and written by ``state_store``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.qwen3_5.sampling import PARAMS, sampler, statistics
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, Device

GEMM_NT = struct.pack("??", False, True)  # y(M, N) = x(K, N) . w(K, M)^T
GEMM_NN = struct.pack("??", False, False)  # y(M, N) = x(K, N) . w(M, K)
GDN_CHUNK = 64
ARENA_BLOCK_ROWS = 1024  # rows of 512 bytes per arena block (the arena's dim 1)


@dataclass(frozen=True, slots=True)
class Dims:
    """Model dimensions the graphs are built from."""

    hidden: int
    ffn: int
    vocab: int
    heads: int
    kv_groups: int
    head_dim: int
    nk: int
    nv: int
    conv_dim: int
    eps: float

    @classmethod
    def from_config(cls, config: Qwen35Config) -> Dims:
        return cls(
            hidden=config.hidden_size,
            ffn=config.intermediate_size,
            vocab=config.vocab_size,
            heads=config.num_attention_heads,
            kv_groups=config.num_key_value_heads,
            head_dim=config.head_dim,
            nk=config.linear_num_key_heads,
            nv=config.linear_num_value_heads,
            conv_dim=config.linear_conv_dim,
            eps=config.rms_norm_eps,
        )

    @property
    def hpg(self) -> int:
        return self.heads // self.kv_groups

    @property
    def gdn_proj(self) -> int:
        """FP8 GDN input projection rows: [q | k | v | z]."""
        return self.conv_dim + self.nv * 128

    @property
    def ssm_rows(self) -> int:
        """512-byte arena rows of one layer's (128, 128, nv) f32 recurrent state."""
        return 128 * self.nv

    @property
    def conv_rows(self) -> int:
        """512-byte arena rows of one layer's (conv_dim, 3) f32 conv state."""
        return self.conv_dim * 3 // 128


class LayerGraph:
    """A Graph with the decoder's building blocks."""

    def __init__(self, device: Device, name: str, dims: Dims, tokens: int) -> None:
        self.g = Graph(device, name)
        self.d = dims
        self.T = tokens
        self._inputs: dict[str, Tensor] = {}

    def input(self, name: str, shape: tuple[int, ...], dtype: int) -> Tensor:
        """A persistent tensor, declared once per name."""
        tensor = self._inputs.get(name)
        if tensor is None:
            tensor = self._inputs[name] = self.g.input(name, shape, dtype)
        return tensor

    def tensor(self, shape: tuple[int, ...], dtype: int) -> Tensor:
        return self.g.tensor(shape, dtype)

    def node(self, guid: str, ins, outs, params: bytes = b"") -> None:
        self.g.node(guid, ins, outs, params)

    # ---- shared inputs
    def meta(self) -> Tensor:
        """[pos0, real tokens, 0, 0] of the chunk."""
        return self.input("meta", (4,), I32)

    def one(self) -> Tensor:
        return self.input("one", (1,), F32)

    def weight(self, name: str, shape: tuple[int, ...], dtype: int) -> Tensor:
        return self.input(name, shape, dtype)

    def fp8_weight(self, name: str, rows: int, cols: int) -> tuple[Tensor, Tensor]:
        return self.weight(name, (cols, rows), F8), self.weight(
            name + ".s", (rows,), F32
        )

    # ---- blocks
    def gemm8(self, act: Tensor, name: str, rows: int) -> Tensor:
        w, ws = self.fp8_weight(name, rows, act.shape[0])
        y = self.tensor((rows, act.shape[1]), F32)
        self.node("fp8_gemm_f32", [act, w, self.one(), ws], [y], GEMM_NT)
        return y

    def gemm(self, act: Tensor, w: Tensor, rows: int, nt: bool = True) -> Tensor:
        y = self.tensor((rows, act.shape[1]), F32)
        self.node("gemm", [act, w], [y], GEMM_NT if nt else GEMM_NN)
        return y

    def add_rms8(
        self,
        x: Tensor,
        y: Tensor | None,
        ys: Tensor | None,
        norm: str,
        outs: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """x + y * ys, RMS-normed with ``norm`` -> (residual, h8, hs, h16)."""
        H, T = self.d.hidden, self.T
        if outs is None:
            outs = (
                self.tensor((H, T), F32),
                self.tensor((H, T), F8),
                self.tensor((1, T), F32),
                self.tensor((H, T), BF16),
            )
        w = self.weight(norm, (H,), F32)
        mode = 0 if y is None else 1
        ins = [x, y if y is not None else x, ys if ys is not None else x, w]
        self.node(
            "rg_add_rms_f8_f32", ins, list(outs), struct.pack("fi", self.d.eps, mode)
        )
        return outs

    def quant8(self, v: Tensor) -> tuple[Tensor, Tensor]:
        q, s = self.tensor(v.shape, F8), self.tensor((1, self.T), F32)
        self.node("rg_quant_fp8_f32", [v], [q, s])
        return q, s

    def carry_in(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        H, T = self.d.hidden, self.T
        return (
            self.input("x", (H, T), F32),
            self.input("h8", (H, T), F8),
            self.input("hs", (1, T), F32),
            self.input("h16", (H, T), BF16),
        )

    def carry_out(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        H, T = self.d.hidden, self.T
        return (
            self.input("x_out", (H, T), F32),
            self.input("h8_out", (H, T), F8),
            self.input("hs_out", (1, T), F32),
            self.input("h16_out", (H, T), BF16),
        )

    def mlp(self, x: Tensor, a: Tensor, a_s: Tensor) -> None:
        """Residual add + post norm, SwiGLU MLP, residual add + the next layer's norm."""
        F = self.d.ffn
        x, h8, hs, _ = self.add_rms8(x, a, a_s, "post_norm")
        gu = self.gemm8(h8, "w_gu", 2 * F)
        sg = self.tensor((F, self.T), F32)
        self.node("rg_swiglu_f8_f32", [gu, hs], [sg])
        sg8, sgs = self.quant8(sg)
        m = self.gemm8(sg8, "w_down", self.d.hidden)
        self.add_rms8(x, m, sgs, "next_norm", self.carry_out())

    # ---- arena state
    def arena(self, blocks: int) -> Tensor:
        return self.input("arena", (128, ARENA_BLOCK_ROWS, blocks), F32)

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

    def reshape(self, t: Tensor, shape: tuple[int, ...]) -> Tensor:
        out = self.tensor(shape, t.dtype)
        self.node("reshape", [t], [out])
        return out


def embed_graph(device: Device, dims: Dims, tokens: int, media: bool = False) -> Graph:
    """Token embedding + the first layer's input norm -> the carry.

    ``media`` adds the splice that replaces the image_pad tokens' embedding with the vision
    tower's rows. It is a separate graph, not an always-present node with count 0: Synapse
    will not accept a tensor as both a node's input and its output (`synNodeCreate` status
    26), so there is no in-place splice, so a count=0 node would copy the whole embedding on
    every text prefill. A text-only deployment never builds this variant at all.
    """
    lg = LayerGraph(device, f"qwen35_embed_T{tokens}" + ("_img" if media else ""), dims, tokens)
    H, T = dims.hidden, tokens
    emb = lg.tensor((H, T), BF16)
    table = lg.weight("E_gather", (H, dims.vocab), BF16)
    lg.node(
        "gather_fwd_bf16",
        [table, lg.input("tok", (T,), I32)],
        [emb],
        struct.pack("i", 1),
    )
    x0 = lg.tensor((H, T), F32)
    lg.node("cast_bf16_to_f32", [emb], [x0])
    if media:
        # the image rows have to be in place before the norm: a host-side write into the
        # carry afterwards is too late, the norm has already run on the un-spliced rows
        spliced = lg.tensor((H, T), F32)
        lg.node(
            "rg_splice_f32",
            [x0, lg.input("img", (H, T), F32), lg.input("img_span", (2,), I32)],
            [spliced],
        )
        x0 = spliced
    lg.add_rms8(x0, None, None, "next_norm", lg.carry_out())
    return lg.g


def head_graph(device: Device, dims: Dims, tokens: int, rows: int = 1) -> Graph:
    """Logits and sampled tokens of ``rows`` columns: the chunk's last real position (``last``)
    for a prefill chunk (rows = 1), or every column of a decode batch (rows = T). Sampling takes
    ``samp_params`` (``sampling.row_params``) and writes ``tok_out``, ``lp_out``, ``top_idx`` and
    ``top_lp``."""
    assert rows in (1, tokens)
    lg = LayerGraph(device, f"qwen35_head_T{tokens}_R{rows}", dims, tokens)
    H, V = dims.hidden, dims.vocab
    h = lg.input("h16", (H, tokens), BF16)
    if rows < tokens:
        hl = lg.tensor((H, rows), BF16)
        last = lg.input("last", (rows,), I32)
        lg.node("gather_fwd_bf16", [h, last], [hl], struct.pack("i", 1))
        h = hl
    logits = lg.input("logits", (V, rows), F32)
    lg.node("gemm", [h, lg.weight("E_head", (H, V), BF16)], [logits], GEMM_NT)
    sampler(lg.g, logits, lg.input("samp_params", (PARAMS, rows), F32))
    return lg.g


SCORE_ROWS = 128  # prompt positions scored per launch
SCORE_PAIRS = 1024  # (position, token) logprobs gathered per launch


def score_graph(device: Device, dims: Dims, max_tokens: int, top: int) -> Graph:
    """Prompt scoring of up to SCORE_ROWS positions of a prefill chunk: ``score_rows`` (R,)
    columns of the chunk's final hidden states ``h16`` (H, max_tokens; a chunk of T tokens fills
    the first T), the raw logprobs of ``score_pairs`` (2, P) [row, token] (``score_lp``), each
    row's statistics (``score_stats``, lane 1 the log-sum-exp) and, when ``top`` > 0, the
    ``top`` largest logits of each row and their ids (``score_top_v``, ``score_top_i``)."""
    lg = LayerGraph(device, f"qwen35_score_P{max_tokens}_K{top}", dims, SCORE_ROWS)
    H, V, R = dims.hidden, dims.vocab, SCORE_ROWS
    h = lg.tensor((H, R), BF16)
    lg.node(
        "gather_fwd_bf16",
        [lg.input("h16", (H, max_tokens), BF16), lg.input("score_rows", (R,), I32)],
        [h],
        struct.pack("i", 1),
    )
    logits = lg.tensor((V, R), F32)
    lg.node("gemm", [h, lg.weight("E_head", (H, V), BF16)], [logits], GEMM_NT)
    stats = lg.input("score_stats", (16, R), F32)
    statistics(lg.g, logits, lg.input("score_params", (PARAMS, R), F32), stats)
    lg.node(
        "rg_score_gather_f32",
        [logits, stats, lg.input("score_pairs", (2, SCORE_PAIRS), I32)],
        [lg.input("score_lp", (SCORE_PAIRS,), F32)],
    )
    if top:
        lg.node(
            "topk",
            [logits],
            [
                lg.input("score_top_v", (top, R), F32),
                lg.input("score_top_i", (top, R), I32),
            ],
            struct.pack("III", top, 0, 0),
        )
    return lg.g


def gdn_layer_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    blocks: int,
    *,
    decode: bool,
    verify: int = 0,
) -> Graph:
    """A gated delta-net layer.

    Decode: T = B sequences, one token each (``gdn_decode`` keeps the recurrent state in the
    arena); ``gdn_bases`` (4, B) [conv_src, ssm_src, conv_dst, ssm_dst] and ``conv_dst`` (B,).
    Verify (``verify`` = Tq tokens per sequence, T = B Tq): ``gdn_verify`` reads each sequence's
    state (``gdn_bases``) without writing it and records the tokens' updates in ``gdn_pend``
    (128, 3, nv, Tq, B) and ``gdn_pconv`` (C, Tq, B) for ``gdn_commit``.
    Prefill: one sequence's T tokens (a multiple of 64) through the chunked delta rule on dense
    state copies; ``conv_src``, ``ssm_src``, ``conv_dst``, ``ssm_dst`` (1,). Sources of -1 read a
    fresh (zero) state; destinations of -1 store nothing."""
    kind = f"ver{verify}" if verify else ("dec" if decode else "pre")
    lg = LayerGraph(device, f"qwen35_gdn_{kind}_T{tokens}_NB{blocks}", dims, tokens)
    d, T = dims, tokens
    x, h8, hs, h16 = lg.carry_in()
    arena = lg.arena(blocks)
    p = lg.gemm8(h8, "w_in", d.gdn_proj)
    conv_w = lg.weight("conv_w", (d.conv_dim, 4), F32)
    ssm_a = lg.weight("ssm_a", (d.nv,), F32)
    dt_bias = lg.weight("dt_bias", (d.nv,), F32)
    ssm_norm = lg.weight("ssm_norm", (128,), F32)
    w_ba = lg.weight("w_ba", (d.hidden, 2 * d.nv), BF16)
    gn = lg.tensor((d.nv * 128, T), F32)
    if verify:
        B = T // verify
        ba = lg.gemm(h16, w_ba, 2 * d.nv)
        ins = [p, hs, ba, conv_w, arena, lg.input("gdn_bases", (4, B), I32)]
        lg.node(
            "rg_gdn_verify_f32",
            [*ins, ssm_a, dt_bias, ssm_norm],
            [
                gn,
                lg.input("gdn_pend", (128, 3, d.nv, verify, B), F32),
                lg.input("gdn_pconv", (d.conv_dim, verify, B), F32),
            ],
            struct.pack("fi", d.eps, d.nk),
        )
    elif decode:
        ba = lg.gemm(h16, w_ba, 2 * d.nv)
        cst_out = lg.tensor((d.conv_dim, 3, T), F32)
        ins = [p, hs, ba, conv_w, arena, lg.input("gdn_bases", (4, T), I32)]
        lg.node(
            "rg_gdn_decode_f32",
            [*ins, ssm_a, dt_bias, ssm_norm],
            [gn, cst_out],
            struct.pack("fi", d.eps, d.nk),
        )
        lg.node(
            "rg_state_store_f32",
            [
                arena,
                lg.input("conv_dst", (T,), I32),
                lg.reshape(cst_out, (128, d.conv_rows, T)),
            ],
            [lg.input("sink_conv", (64, 1), I32)],
        )
    else:
        cst_in = lg.reshape(
            lg.state_load(arena, "conv_src", d.conv_rows), (d.conv_dim, 3)
        )
        ssm_flat = lg.state_load(arena, "ssm_src", d.ssm_rows)
        cst_out = lg.tensor((d.conv_dim, 3), F32)
        sst_in = lg.reshape(ssm_flat, (128, 128, d.nv, 1))
        sst_out = lg.tensor((128, 128, d.nv, 1), F32)
        baT = lg.tensor((T, 2 * d.nv), F32)
        lg.node("gemm", [w_ba, h16], [baT], GEMM_NT)
        gdn_chunked(
            lg,
            p=p,
            ps=hs,
            baT=baT,
            conv_w=conv_w,
            cst_in=cst_in,
            sst_in=sst_in,
            ssm_a=ssm_a,
            dt_bias=dt_bias,
            ssm_norm=ssm_norm,
            out=gn,
            cst_out=cst_out,
            sst_out=sst_out,
        )
        lg.state_store(
            arena, "conv_dst", lg.reshape(cst_out, (128, d.conv_rows, 1)), "sink_conv"
        )
        lg.state_store(
            arena, "ssm_dst", lg.reshape(sst_out, (128, d.ssm_rows, 1)), "sink_ssm"
        )
    gn8, gns = lg.quant8(gn)
    a = lg.gemm8(gn8, "w_out", d.hidden)
    lg.mlp(x, a, gns)
    return lg.g


def commit_graph(
    device: Device, dims: Dims, sequences: int, verify: int, layers: int, blocks: int
) -> Graph:
    """The accepted tokens of a verify step, committed: ``gdn_commit`` over every GDN layer's
    entries (``commit_bases`` (4, B L), ``commit_counts`` (B L,), the layers' ``gdn_pend`` and
    ``gdn_pconv`` runs back to back), and each sequence's accepted column of the target's final
    residual ``x`` (H, B Tq) into its STATE block's hidden slot (``hidden_table`` (2, B))."""
    lg = LayerGraph(
        device, f"qwen35_commit_B{sequences}_Q{verify}_L{layers}_NB{blocks}", dims, 1
    )
    d, BL = dims, sequences * layers
    arena = lg.arena(blocks)
    lg.node(
        "rg_gdn_commit_f32",
        [
            arena,
            lg.input("gdn_pend", (128, 3, d.nv, verify, BL), F32),
            lg.input("gdn_pconv", (d.conv_dim, verify, BL), F32),
            lg.input("commit_bases", (4, BL), I32),
            lg.input("commit_counts", (BL,), I32),
        ],
        [lg.input("sink_commit", (64, 1), I32)],
        struct.pack("i", d.nk),
    )
    hidden_store(
        lg, arena, lg.input("x", (d.hidden, sequences * verify), F32), sequences
    )
    return lg.g


def hidden_store(lg: LayerGraph, arena: Tensor, x: Tensor, rows: int) -> None:
    lg.node(
        "rg_hidden_store_f32",
        [arena, x, lg.input("hidden_table", (2, rows), I32)],
        [lg.input("sink_hidden", (64, 1), I32)],
    )


def hidden_store_graph(device: Device, dims: Dims, tokens: int, blocks: int) -> Graph:
    """One prefill chunk's last real column of the final residual ``x`` (H, tokens: the largest
    chunk; a chunk of T tokens fills the first T) into its STATE block's hidden slot."""
    lg = LayerGraph(device, f"qwen35_hidden_T{tokens}_NB{blocks}", dims, 1)
    hidden_store(lg, lg.arena(blocks), lg.input("x", (dims.hidden, tokens), F32), 1)
    return lg.g


def mtp_in_graph(device: Device, dims: Dims, tokens: int, columns: int) -> Graph:
    """The MTP layer's input for T columns: column c combines the embedding of ``mtp_tok`` [c]
    with hidden column ``mtp_cols`` [c] of [``mtp_prev`` (H, 1) | ``x`` (H, columns)] (residuals,
    normed with ``mtp_xnorm``: the final norm for the target's, mtp.norm for the MTP layer's
    own), as fc([enorm(e), hnorm(norm(x))]); then the MTP decoder layer's input norm -> the
    carry."""
    lg = LayerGraph(device, f"qwen35_mtp_in_T{tokens}_X{columns}", dims, tokens)
    H, T = dims.hidden, tokens
    both = lg.tensor((H, columns + 1), F32)
    lg.node(
        "concat",
        [lg.input("mtp_prev", (H, 1), F32), lg.input("x", (H, columns), F32)],
        [both],
        struct.pack("I", 1),
    )
    xs = lg.tensor((H, T), F32)
    lg.node(
        "gather_fwd_f32",
        [both, lg.input("mtp_cols", (T,), I32)],
        [xs],
        struct.pack("i", 1),
    )
    e = lg.tensor((H, T), BF16)
    table = lg.weight("E_gather", (H, dims.vocab), BF16)
    lg.node(
        "gather_fwd_bf16",
        [table, lg.input("mtp_tok", (T,), I32)],
        [e],
        struct.pack("i", 1),
    )
    ef = lg.tensor((H, T), F32)
    lg.node("cast_bf16_to_f32", [e], [ef])
    cat = lg.tensor((2 * H, T), BF16)
    lg.node(
        "rg_mtp_in_f32",
        [
            xs,
            ef,
            lg.weight("mtp_xnorm", (H,), F32),
            lg.weight("mtp.hnorm", (H,), F32),
            lg.weight("mtp.enorm", (H,), F32),
        ],
        [cat],
        struct.pack("f", dims.eps),
    )
    f = lg.gemm(cat, lg.weight("mtp.fc", (2 * H, H), BF16), H)
    lg.add_rms8(f, None, None, "attn_norm", lg.carry_out())
    return lg.g


def gdn_chunked(
    lg: LayerGraph,
    *,
    p,
    ps,
    baT,
    conv_w,
    cst_in,
    sst_in,
    ssm_a,
    dt_bias,
    ssm_norm,
    out,
    cst_out,
    sst_out,
) -> None:
    """The chunked (WY) gated delta rule of one FP8 prefill chunk: batched
    products over every (head, chunk) pair, then a sequential state step per 64-token chunk.
    Per (head, chunk) blocks live in (cols, rows, nv, NC) tensors."""
    d, T = lg.d, lg.T
    assert T % GDN_CHUNK == 0
    CH, NC, nk, nv = GDN_CHUNK, T // GDN_CHUNK, d.nk, d.nv
    blk, sq, gram = (128, CH, nv, NC), (128, 128, nv, NC), (CH, CH, nv, NC)

    def bg(x: Tensor, w: Tensor, ta: bool, tb: bool, shape) -> Tensor:
        y = lg.tensor(shape, F32)
        lg.node("batch_gemm", [x, w], [y], struct.pack("??", ta, tb))
        return y

    # stage 1: q / k per shared head, gates and v per value head
    Vn = lg.tensor(blk, F32)
    GB = lg.tensor((CH, 2, nv, NC), F32)
    cst_qk, cst_v = lg.tensor((2 * nk * 128, 3), F32), lg.tensor((nv * 128, 3), F32)
    prm = struct.pack("fi", d.eps, nk)
    kblk = (128, CH, nk, NC)
    Qk, Kk = lg.tensor(kblk, F32), lg.tensor(kblk, F32)
    meta = lg.meta()
    lg.node("rg_gdn_ck_qk_f8_f32", [p, ps, conv_w, cst_in, meta], [Qk, Kk, cst_qk], prm)
    if nv == nk:
        Qn, Kn = Qk, Kk
    else:  # value head h reads q/k head h % nk
        Qn, Kn = lg.tensor(blk, F32), lg.tensor(blk, F32)
        lg.node("concat", [Qk] * (nv // nk), [Qn], struct.pack("I", 2))
        lg.node("concat", [Kk] * (nv // nk), [Kn], struct.pack("I", 2))
    lg.node(
        "rg_gdn_ck_v_f8_f32",
        [p, ps, baT, conv_w, cst_in, ssm_a, dt_bias, meta],
        [GB, Vn, cst_v],
        prm,
    )
    lg.node("concat", [cst_qk, cst_v], [cst_out], struct.pack("I", 0))
    # stage 2: Gram matrices, decay and masks, T = (I + A)^-1 with the gate scalings folded in
    kgram = (CH, CH, nk, NC)
    KK = bg(Kk, Kk, False, True, kgram)
    QK = bg(Qk, Kk, False, True, kgram)
    P, Tb, Tbg, Tkb, Tkbg = (lg.tensor(gram, F32) for _ in range(5))
    lg.node("rg_gdn_ck_solve_f32", [KK, QK, GB], [P, Tb, Tbg, Tkb, Tkbg])
    # stage 3: chunk-local solutions and state-transition terms (D = U - W S0)
    U = bg(Tb, Vn, False, False, blk)
    W = bg(Tbg, Kn, False, False, blk)
    B = bg(Kn, bg(Tkb, Vn, False, False, blk), True, False, sq)
    KW = bg(Kn, bg(Tkbg, Kn, False, False, blk), True, False, sq)
    # stage 4: the sequential state chain over chunks
    one = (128, 128, nv, 1)

    def chunks(x: Tensor, shape) -> list[Tensor]:
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
    # stage 5: o = gamma (Q S0) + P (U - W S0), gated RMSNorm
    QS = bg(Qn, Sall, False, False, blk)
    D = lg.tensor(blk, F32)
    lg.node("sub_fwd_f32", [U, bg(W, Sall, False, False, blk)], [D])
    PD = bg(P, D, False, False, blk)
    lg.node(
        "rg_gdn_ck_out_f8_f32",
        [QS, PD, GB, p, ps, ssm_norm, meta],
        [out],
        struct.pack("fi", d.eps, nk),
    )


# ---- full attention
SLOT_KEYS = 2048  # keys per attention slot (16 pages)
QUERY_BLOCK = 512  # prefill tokens per attention slot
# prefill slots of at least this many query tokens use ``_attn_slots_fused`` (0: never)
FUSED_QUERY_BLOCK = 64
# softmax mode: 0 default, 4 f32
SDPA_SOFTMAX = 0


def sdpa_stats_params() -> bytes:
    """Node parameters of ``sdpa_recomp_fwd_bf16``."""
    return struct.pack(
        "<f?3xfI?3x?3xi", 1.0 / 16.0, False, 0.0, 0, True, False, SDPA_SOFTMAX
    )


def attn_query_block(tokens: int) -> int:
    return min(tokens, QUERY_BLOCK)


def attn_proj(dims: Dims) -> int:
    """QKV projection rows: [q_h | gate_h] x 256 per head, then k and v per group."""
    return 2 * dims.head_dim * dims.heads + 2 * dims.head_dim * dims.kv_groups


def attn_pre_graph(
    device: Device, dims: Dims, tokens: int, blocks: int, positions: int
) -> Graph:
    """QKV projection, Q/K norm + RoPE at each column's position (``pos`` (T,) into the
    ``cs_tab`` (128, positions) rope table), the chunk's K/V rows stored into their pages
    (``kv_rows`` (8, T): K groups then V groups, -1 skips), Q for the slot launches (``q_all``,
    (256, hpg T G) bf16), the projection for the gate (``p_out``) and the accumulator reset."""
    lg = LayerGraph(
        device, f"qwen35_attn_pre_T{tokens}_NB{blocks}_P{positions}", dims, tokens
    )
    d, T, G = dims, tokens, dims.kv_groups
    Tv = 128 if 1 < T < 128 else T
    W = d.hpg * T
    _, h8, hs, _ = lg.carry_in()
    p = lg.input("p_out", (attn_proj(d), T), F32)
    w, ws = lg.fp8_weight("w_qkv", attn_proj(d), d.hidden)
    lg.node("fp8_gemm_f32", [h8, w, lg.one(), ws], [p], GEMM_NT)
    cs = lg.tensor((128, T), F32)
    lg.node(
        "gather_fwd_f32",
        [lg.input("cs_tab", (128, positions), F32), lg.input("pos", (T,), I32)],
        [cs],
        struct.pack("i", 1),
    )
    ins = [p, hs, lg.weight("qk_norm", (2 * d.head_dim,), F32), cs, lg.meta()]
    prm = struct.pack("fi", d.eps, d.hpg)
    qs = [lg.tensor((d.head_dim, W), BF16) for _ in range(G)]
    lg.node(f"rg_attn_prep_q_f8_g{G}_f32", ins, qs, prm)
    kn = [lg.tensor((d.head_dim, T), BF16) for _ in range(G)]
    vn = [lg.tensor((d.head_dim, Tv), BF16) for _ in range(G)]
    lg.node(f"rg_attn_prep_kv_f8_g{G}_f32", ins, kn + vn, prm)
    arena = lg.input("arena", (256, ARENA_BLOCK_ROWS, blocks), BF16)
    rows = lg.input("kv_rows", (8, T), I32)
    spare = 4 - G  # kv_store takes 4 groups, and skips the ones whose rows are -1
    lg.node(
        "rg_kv_store_f32",
        [arena, rows, *kn, *kn[:1] * spare, *vn, *vn[:1] * spare],
        [lg.input("sink_kv", (64, 1), I32)],
    )
    lg.node(
        "concat",
        qs,
        [lg.input("q_all", (d.head_dim, W * G), BF16)],
        struct.pack("I", 1),
    )
    acc = [
        lg.input("acc_o", (d.head_dim, W, G), F32),
        lg.input("acc_m", (W, G), F32),
        lg.input("acc_l", (W, G), F32),
    ]
    lg.node("rg_attn_acc_init_f32", acc, [lg.input("sink_acc", (64, 1), I32)])
    return lg.g


def attn_slots_graph(
    device: Device, dims: Dims, tokens: int, slots: int, blocks: int, query_block: int
) -> Graph:
    """One launch of ``slots`` attention slots of ``query_block`` tokens each (a prefill chunk's
    block of up to 512 tokens, or one decode token), merged into the running accumulators of
    the T columns. Inputs per launch: ``slots`` (8, S) i32 [group, qcol0, qpos0, key0, klen]
    (group -1: padding), ``k_pages`` / ``v_pages`` (C / 128, S) i32 first arena rows (-1: none).
    Cache rows of keys at or past klen are never read."""
    if FUSED_QUERY_BLOCK and query_block >= FUSED_QUERY_BLOCK:
        return _attn_slots_fused(device, dims, tokens, slots, blocks, query_block)
    name = f"qwen35_attn_slots_T{tokens}_Q{query_block}_S{slots}_NB{blocks}"
    lg = LayerGraph(device, name, dims, tokens)
    d, T, G, S, C = dims, tokens, dims.kv_groups, slots, SLOT_KEYS
    W, M = d.hpg * T, d.hpg * query_block
    Mp = (M + 63) // 64 * 64
    arena = lg.input("arena", (256, ARENA_BLOCK_ROWS, blocks), BF16)
    table = lg.input("slots", (8, S), I32)
    q = lg.tensor((d.head_dim, M, S), BF16)
    lg.node(
        "rg_attn_slot_q_f32",
        [lg.input("q_all", (d.head_dim, W * G), BF16), table],
        [q],
        struct.pack("ii", G, 0),
    )
    ks = lg.tensor((d.head_dim, C, S), BF16)
    k_pages = lg.input("k_pages", (C // 128, S), I32)
    lg.node("rg_kv_gather_f32", [arena, k_pages, table], [ks], struct.pack("i", 0))
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, S), BF16)
    mx, ls = lg.tensor((Mp, S), F32), lg.tensor((Mp, S), F32)
    lg.node(
        "rg_attn_slot_p_f32",
        [sc, table],
        [P, mx, ls],
        struct.pack("fi", 1.0 / d.head_dim**0.5, d.hpg),
    )
    vs = lg.tensor((d.head_dim, C, S), BF16)
    v_pages = lg.input("v_pages", (C // 128, S), I32)
    lg.node("rg_kv_gather_f32", [arena, v_pages, table], [vs], struct.pack("i", 1))
    pv = lg.tensor((d.head_dim, M, S), F32)
    lg.node("batch_gemm", [P, vs], [pv], GEMM_NN)
    acc = [
        lg.input("acc_o", (d.head_dim, W, G), F32),
        lg.input("acc_m", (W, G), F32),
        lg.input("acc_l", (W, G), F32),
    ]
    lg.node(
        "rg_attn_slot_acc_f32",
        [pv, mx, ls, table, *acc],
        [lg.input("sink_slots", (64, 1), I32)],
    )
    return lg.g


def _attn_slots_fused(
    device: Device, dims: Dims, tokens: int, slots: int, blocks: int, query_block: int
) -> Graph:
    """``attn_slots_graph`` for prefill blocks."""
    name = f"qwen35_attn_slots_sdpa{SDPA_SOFTMAX}_T{tokens}_Q{query_block}_S{slots}_NB{blocks}"
    lg = LayerGraph(device, name, dims, tokens)
    d, T, G, S, C, Tq = dims, tokens, dims.kv_groups, slots, SLOT_KEYS, query_block
    W, M, H = d.hpg * T, d.hpg * query_block, dims.hpg
    arena = lg.input("arena", (256, ARENA_BLOCK_ROWS, blocks), BF16)
    table = lg.input("slots", (8, S), I32)
    q = lg.tensor((d.head_dim, M, S), BF16)
    lg.node(
        "rg_attn_slot_q_f32",
        [lg.input("q_all", (d.head_dim, W * G), BF16), table],
        [q],
        struct.pack("ii", G, Tq),
    )
    ks = lg.tensor((d.head_dim, C, S), BF16)
    k_pages = lg.input("k_pages", (C // 128, S), I32)
    lg.node("rg_kv_gather_f32", [arena, k_pages, table], [ks], struct.pack("i", 0))
    vs = lg.tensor((d.head_dim, C, S), BF16)
    v_pages = lg.input("v_pages", (C // 128, S), I32)
    lg.node("rg_kv_gather_f32", [arena, v_pages, table], [vs], struct.pack("i", 2))
    mask = lg.tensor((C, Tq, 1, S), BF16)
    lg.node("rg_attn_slot_mask_f32", [table], [mask])
    o = lg.tensor((d.head_dim, Tq, H, S), F32)
    mx, linv = lg.tensor((1, Tq, H, S), F32), lg.tensor((1, Tq, H, S), F32)
    lg.node(
        "sdpa_recomp_fwd_bf16",
        [
            lg.reshape(q, (d.head_dim, Tq, H, S)),
            lg.reshape(ks, (d.head_dim, C, 1, S)),
            lg.reshape(vs, (d.head_dim, C, 1, S)),
            mask,
        ],
        [o, mx, linv],
        sdpa_stats_params(),
    )
    acc = [
        lg.input("acc_o", (d.head_dim, W, G), F32),
        lg.input("acc_m", (W, G), F32),
        lg.input("acc_l", (W, G), F32),
    ]
    lg.node(
        "rg_attn_sdpa_acc_f32",
        [o, lg.reshape(mx, (Tq, H, S)), lg.reshape(linv, (Tq, H, S)), table, *acc],
        [lg.input("sink_slots", (64, 1), I32)],
        struct.pack("i", T),
    )
    return lg.g


def attn_post_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """Output gate on the accumulated attention (``acc_o``), output projection and MLP."""
    lg = LayerGraph(device, f"qwen35_attn_post_T{tokens}", dims, tokens)
    d, T, G = dims, tokens, dims.kv_groups
    W = d.hpg * T
    x, _, hs, _ = lg.carry_in()
    p = lg.input("p_out", (attn_proj(d), T), F32)
    acc = lg.input("acc_o", (d.head_dim, W, G), F32)
    parts = [lg.tensor((d.head_dim, W, 1), F32) for _ in range(G)]
    lg.node("split", [acc], parts, struct.pack("I", 2))
    on = [lg.reshape(part, (d.head_dim, W)) for part in parts]
    ga8 = lg.tensor((d.head_dim * d.heads, T), F8)
    gas = lg.tensor((1, T), F32)
    lg.node(
        f"rg_attn_gate_np_f8_g{G}_f32",
        [*on, p, hs],
        [ga8, gas],
        struct.pack("i", d.hpg),
    )
    a = lg.gemm8(ga8, "w_o", d.hidden)
    lg.mlp(x, a, gas)
    return lg.g
