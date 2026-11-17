"""The Synapse graph builder of the Gaudi models (``LayerGraph``), their dimensions (``Dims``)
and the attention slot graphs.

A graph is one layer for T tokens and is reused for every layer of its kind: weights are
persistent inputs bound per launch under generic names (``w_in``, ``w_in.s``, ...). Layers hand
each other a carry, the residual and the next layer's normed input as FP8 bytes, per-token scale
and bf16: inputs ``x h8 hs h16``, outputs ``x_out h8_out hs_out h16_out``.

FP8 projections run as ``fp8_gemm``. Recurrent state lives in the cache arena, read by
``state_load`` and written by ``state_store``. The Qwen3.5/3.8 layer graphs are in
``qwen3_5.layers``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, U8, Device

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

    def shard(self, shard) -> Dims:
        """A tensor-parallel rank's dimensions (``loading.shard.DenseShard``): its attention heads and KV
        groups, GDN key and value heads (128 channels each), and MLP channels."""
        if self.conv_dim != 2 * 128 * self.nk + 128 * self.nv:
            raise ValueError("GDN heads of other than 128 channels do not shard")
        return replace(
            self,
            heads=shard.local_heads,
            kv_groups=shard.local_groups,
            nk=shard.local_nk,
            nv=shard.local_nv,
            ffn=shard.local_ffn,
            conv_dim=2 * 128 * shard.local_nk + 128 * shard.local_nv,
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


# ---- full attention
SLOT_KEYS = 2048  # keys per attention slot (16 pages)
# slot softmax builds and their one-member-per-column builds (bit for bit the same results)
PER_COLUMN_SOFTMAX = {
    "rg_attn_slot_p_f32": "rg_attn_slot_p1_f32",
    "rg_attn_slot_pc_f32": "rg_attn_slot_pc1_f32",
    "rg_attn_slot_pd_f32": "rg_attn_slot_pd1_f32",
}
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


def kv_arena(lg: LayerGraph, blocks: int, kv8: int) -> Tensor:
    """The cache arena as a full-attention layer's K/V kernels take it: bf16 rows, or with FP8 pages
    (``kv8``) bytes (``kv_store8``, ``kv_gather8``)."""
    if kv8:
        return lg.input("arena", (512, ARENA_BLOCK_ROWS, blocks), U8)
    return lg.input("arena", (256, ARENA_BLOCK_ROWS, blocks), BF16)


def kv_gather(
    lg: LayerGraph,
    arena: Tensor,
    pages: Tensor,
    table: Tensor,
    out: Tensor,
    sig: int,
    kv8: int,
    v: bool,
) -> None:
    """Gather one slot launch's K pages (``v`` False) or V pages into ``out`` (``kv_gather``; with FP8 pages,
    ``kv_gather8``, which reads the byte view of the arena and dequantizes the FP8 pages)."""
    if kv8:
        fp8 = (kv8 >> int(v)) & 1
        lg.node(
            "rg_kv_gather8_f32",
            [arena, pages, table],
            [out],
            struct.pack("ii", sig, fp8),
        )
    else:
        lg.node("rg_kv_gather_f32", [arena, pages, table], [out], struct.pack("i", sig))


def attn_slots_graph(
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    blocks: int,
    query_block: int,
    *,
    kernel: str = "rg_attn_slot_p_f32",
    scale: float | None = None,
    per_column: bool = False,
    mask: bool = True,
    kv8: int = 0,
) -> Graph:
    """One launch of ``slots`` attention slots of ``query_block`` tokens each (a prefill chunk's
    block of up to 512 tokens, or one decode token), merged into the running accumulators of
    the T columns. Inputs per launch: ``slots`` (8, S) i32 [group, qcol0, qpos0, key0, klen]
    (group -1: padding), ``k_pages`` / ``v_pages`` (C / 128, S) i32 first arena rows (-1: none).
    Cache rows of keys at or past klen are never read. ``kernel`` is the softmax kernel (a
    ``attn_slot_p`` build) and ``scale`` the score scale (1 / sqrt(head_dim)); ``per_column``
    takes its build with one member per query column (``PER_COLUMN_SOFTMAX``), with the same
    results, for slots of few query columns (decode, verify, drafting blocks).
    ``mask=False`` (fused prefill launches only): every query of a slot sees every key of it, so
    the launch builds no mask (the same results for such slots).
    ``kv8``: FP8 K pages (bit 0), V pages (bit 1) (``kv_gather``)."""
    if FUSED_QUERY_BLOCK and query_block >= FUSED_QUERY_BLOCK:
        if kernel != "rg_attn_slot_p_f32" or scale is not None:
            raise ValueError("the fused slot attention takes the default softmax")
        return _attn_slots_fused(
            device, dims, tokens, slots, blocks, query_block, mask, kv8
        )
    if not mask:
        raise ValueError("only the fused slot attention launches without a mask")
    if (
        per_column
    ):  # the build with one member per query column
        kernel = PER_COLUMN_SOFTMAX[kernel]
    variant = (
        ""
        if kernel == "rg_attn_slot_p_f32"
        else "_" + kernel.removeprefix("rg_attn_slot_").removesuffix("_f32")
    )
    name = f"qwen35_attn_slots{variant}_T{tokens}_Q{query_block}_S{slots}_NB{blocks}"
    if kv8:
        name += f"_kv{kv8}"
    lg = LayerGraph(device, name, dims, tokens)
    d, T, G, S, C = dims, tokens, dims.kv_groups, slots, SLOT_KEYS
    W, M = d.hpg * T, d.hpg * query_block
    Mp = (M + 63) // 64 * 64
    arena = kv_arena(lg, blocks, kv8)
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
    kv_gather(lg, arena, k_pages, table, ks, 0, kv8, False)
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [q, ks], [sc], GEMM_NT)
    P = lg.tensor((C, M, S), BF16)
    mx, ls = lg.tensor((Mp, S), F32), lg.tensor((Mp, S), F32)
    lg.node(
        kernel,
        [sc, table],
        [P, mx, ls],
        struct.pack("fi", 1.0 / d.head_dim**0.5 if scale is None else scale, d.hpg),
    )
    vs = lg.tensor((d.head_dim, C, S), BF16)
    v_pages = lg.input("v_pages", (C // 128, S), I32)
    kv_gather(lg, arena, v_pages, table, vs, 1, kv8, True)
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
    device: Device,
    dims: Dims,
    tokens: int,
    slots: int,
    blocks: int,
    query_block: int,
    mask: bool = True,
    kv8: int = 0,
) -> Graph:
    """``attn_slots_graph`` for prefill blocks."""
    name = f"qwen35_attn_slots_sdpa{SDPA_SOFTMAX}_T{tokens}_Q{query_block}_S{slots}_NB{blocks}"
    if not mask:
        name += "_full"
    if kv8:
        name += f"_kv{kv8}"
    lg = LayerGraph(device, name, dims, tokens)
    d, T, G, S, C, Tq = dims, tokens, dims.kv_groups, slots, SLOT_KEYS, query_block
    W, M, H = d.hpg * T, d.hpg * query_block, dims.hpg
    arena = kv_arena(lg, blocks, kv8)
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
    kv_gather(lg, arena, k_pages, table, ks, 0, kv8, False)
    vs = lg.tensor((d.head_dim, C, S), BF16)
    v_pages = lg.input("v_pages", (C // 128, S), I32)
    kv_gather(lg, arena, v_pages, table, vs, 2, kv8, True)
    inputs = [
        lg.reshape(q, (d.head_dim, Tq, H, S)),
        lg.reshape(ks, (d.head_dim, C, 1, S)),
        lg.reshape(vs, (d.head_dim, C, 1, S)),
    ]
    if mask:
        visible = lg.tensor((C, Tq, 1, S), BF16)
        lg.node("rg_attn_slot_mask_f32", [table], [visible])
        inputs.append(visible)
    o = lg.tensor((d.head_dim, Tq, H, S), F32)
    mx, linv = lg.tensor((1, Tq, H, S), F32), lg.tensor((1, Tq, H, S), F32)
    lg.node("sdpa_recomp_fwd_bf16", inputs, [o, mx, linv], sdpa_stats_params())
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


