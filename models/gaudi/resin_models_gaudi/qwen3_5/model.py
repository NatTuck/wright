"""FP8 forward of Qwen3.5/3.8 on one Gaudi card over the paged cache arena, one recipe launch
per layer: one sequence's prefill chunk, one decode token for each of a batch of sequences, or,
with MTP, a speculative verify step for each of a batch of sequences.

Every recipe follows from the configuration (prefill buckets, decode batch buckets, MTP draft
length, slots per attention launch, rope table length, arena size) and is compiled at
construction. The arena is sized first, from the memory left after weights,
and rounded down to a multiple of ``CAPACITY_QUANTUM`` slabs.

A step runs ``embed``, the layers and ``head``; a full-attention layer is ``attn_pre``, as many
``attn_slots`` launches as the context needs and ``attn_post``. Per-step tables are written once:
attention tables use layer 0's page rows, and each full-attention layer binds the arena at its
own row offset (a layer's pages sit at a fixed offset inside every KV block). GDN state rows are
per-layer tables at fixed addresses.

MTP (``mtp`` = K drafts): a decode step verifies [anchor, d_1 .. d_K] of every sequence in one
target pass (T = K + 1 per sequence; ``gdn_verify`` records the GDN state updates instead of
writing them), samples y_0 .. y_K, and runs the MTP layer at every verify position. The host
accepts d_1 .. d_m while d_t = y_(t-1), emits y_0 .. y_m, commits the first m + 1 tokens' state
(``gdn_commit``) and the next drafts are the MTP's draft at position m chained K - 1 times
through the MTP layer. A prefill runs the MTP layer over its chunk (filling its KV) and drafts
after the sampled token. Output equals non-speculative decoding: every token is the target's
sample at its position.

DFlash 2 (``drafter`` = a ``Qwen3.8-27B-DFlash2`` directory, ``dflash.py``): the same verify step
with K drafts (``mtp`` = 1 .. 7, default 7: the block makes 7, a step may verify fewer), and the
drafts come from the separate block-diffusion drafter instead of the MTP layer: every target pass feeds its tap layers' residuals to the drafter's cache, and after the
accepted tokens are known one block step makes the next 7 drafts. MTP and the drafter are
alternatives. A ``Qwen3.8-27B-DSpine-b16`` directory selects the DSpine drafter in the same place:
its block makes 15 drafts (``mtp`` = 1 .. 15, default 15).
"""

from __future__ import annotations

import logging
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from resin.backend.capacity import plan_cache_capacity
from resin.backend.speculative import SpeculativeStats
from resin_models_gaudi.arena import ARENA_PAD_BLOCKS, SharedArena
from resin_models_gaudi.common.attention import SlotRow, kv_rows, plan_slots
from resin_models_gaudi.common.dflash import (
    Drafter,
    DrafterConfig,
    DraftRow,
    drafter_is_fp8,
    slots_per_launch,
)
from resin_models_gaudi.common.dflash import prepare_weights as prepare_drafter
from resin_models_gaudi.common.dflash import tap_graph as dflash_tap_graph
from resin_models_gaudi.common.layers import (
    ARENA_BLOCK_ROWS,
    FUSED_QUERY_BLOCK,
    QUERY_BLOCK,
    SLOT_KEYS,
    Dims,
    attn_slots_graph,
)
from resin_models_gaudi.common.sampling import (
    PARAMS,
    TOP_MAX,
    Sampling,
    row_params,
)
from resin_models_gaudi.common.scoring import ScoreRequest, Scores
from resin_models_gaudi.common.weights import weight_items
from resin_models_gaudi.loading.convert import rope_table
from resin_models_gaudi.loading.progress import progress, weight_file_bytes
from resin_models_gaudi.loading.shard import DenseShard
from resin_models_gaudi.loading.upload import WeightLoader
from resin_models_gaudi.qwen3_5.cache import (
    ROW_BYTES,
    STATE_TOKENS,
    CacheGeometry,
    kv8_bits,
)
from resin_models_gaudi.qwen3_5.checkpoint import Qwen35Checkpoint
from resin_models_gaudi.qwen3_5.layers import (
    SCORE_PAIRS,
    SCORE_ROWS,
    attn_post_graph,
    attn_pre_graph,
    attn_proj,
    commit_graph,
    embed_graph,
    gdn_layer_graph,
    head_graph,
    hidden_store_graph,
    mlp_graph,
    mtp_in_graph,
    score_graph,
)
from resin_models_gaudi.synapse.graph import Graph, LaunchPlan, Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import I32, Device

PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
DECODE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
# the widest decode bucket of a speculative step (rows): wider decode steps run without drafts. Verify
# modes, the drafter's block steps and the speculative buffers are sized from this, not DECODE_BUCKETS[-1]
SPEC_ROWS = 32
# verify widths (columns per row) a DSpine drafter's steps choose among, besides all 16
DSPINE_WIDTHS = (4, 8)
# P(a step accepts at least j of its drafts | the row's last step), j = 0 .. 15, by class of the
# last step: accepted 0, 1, 2, 3, 4-5, 6-8, 9-14, 15, then at least 3 (all, at width 4) and at
# least 7 (all, at width 8). Measured with the DSpine drafter verifying all 15 drafts (scripts/
# bench.py, greedy code and English, 8K context, 1 and 8 rows, 2026-11-09).
# fmt: off
SURVIVAL = np.array(
    (
        (1.000, 0.695, 0.480, 0.329, 0.222, 0.151, 0.103, 0.065, 0.048, 0.036, 0.032, 0.028, 0.022, 0.017, 0.012, 0.012),
        (1.000, 0.710, 0.487, 0.339, 0.216, 0.144, 0.090, 0.063, 0.041, 0.031, 0.023, 0.018, 0.012, 0.007, 0.006, 0.003),
        (0.999, 0.747, 0.535, 0.346, 0.219, 0.151, 0.093, 0.069, 0.048, 0.035, 0.025, 0.015, 0.015, 0.010, 0.009, 0.008),
        (0.999, 0.735, 0.511, 0.373, 0.262, 0.197, 0.130, 0.105, 0.082, 0.064, 0.049, 0.031, 0.029, 0.024, 0.024, 0.014),
        (0.999, 0.735, 0.535, 0.387, 0.293, 0.226, 0.157, 0.114, 0.090, 0.060, 0.042, 0.023, 0.018, 0.013, 0.013, 0.013),
        (0.999, 0.802, 0.618, 0.490, 0.344, 0.242, 0.173, 0.132, 0.121, 0.109, 0.104, 0.083, 0.070, 0.055, 0.052, 0.050),
        (0.998, 0.852, 0.731, 0.593, 0.520, 0.439, 0.407, 0.346, 0.302, 0.277, 0.253, 0.229, 0.217, 0.168, 0.144, 0.128),
        (0.998, 0.983, 0.938, 0.879, 0.865, 0.820, 0.736, 0.682, 0.667, 0.653, 0.643, 0.628, 0.589, 0.549, 0.534, 0.515),
        (1.000, 0.785, 0.604, 0.472, 0.373, 0.301, 0.235, 0.195, 0.172, 0.150, 0.135, 0.116, 0.107, 0.092, 0.087, 0.080),
        (0.999, 0.876, 0.756, 0.643, 0.571, 0.497, 0.440, 0.388, 0.363, 0.347, 0.334, 0.315, 0.297, 0.257, 0.241, 0.227),
    )
)
# fmt: on
SURVIVAL_CLASS = (0, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7)


def spec_fix_graph(
    device: Device,
    bucket: int,
    width: int,
    layers: int,
    patches: int,
    block_tokens: int,
    table_words: int,
) -> Graph:
    """A verify step's acceptance and the tables after it on the device (``spec_fix``): the
    samples ``tok_out`` against the step's input tokens ``fix_tin`` -> ``fix_acc``, the commit's
    ``commit_bases`` / ``commit_counts`` / ``hidden_table`` and the next block step's tokens,
    positions, anchors and slot records (the drafter's ``btok`` / ``bpos`` / ``banc`` /
    ``dtables``)."""
    g = Graph(device, f"qwen35_spec_fix_B{bucket}_Q{width}_L{layers}_P{patches}")
    BL = bucket * layers
    g.node(
        "rg_spec_fix_f32",
        [
            g.input("tok_out", (width, bucket), I32),
            g.input("fix_tin", (width, bucket), I32),
            g.input("fix_rows", (8, bucket), I32),
            g.input("fix_cb0", (4, BL), I32),
            g.input("fix_cb1", (4, BL), I32),
            g.input("fix_patch", (4, patches), I32),
            g.input("fix_meta", (4,), I32),
        ],
        [
            g.input("fix_acc", (bucket,), I32),
            g.input("commit_bases", (4, BL), I32),
            g.input("commit_counts", (BL,), I32),
            g.input("hidden_table", (2, bucket), I32),
            g.input("btok", (block_tokens,), I32),
            g.input("bpos", (block_tokens,), I32),
            g.input("banc", (bucket,), I32),
            g.input("dtables", (table_words,), I32),
        ],
        struct.pack("i", layers),
    )
    return g


def verify_mode(bucket: int, width: int, full: int) -> str:
    """The verify mode of bucket rows of width columns (v{B} takes every draft)."""
    return f"v{bucket}" if width == full else f"v{bucket}w{width}"


def step_ms(width: int, bucket: int) -> float:
    """A speculative step's time (ms) with the DSpine drafter, bucket rows of width
    columns: a fit of one card's steps at 4K context."""
    per_column = max(0.4, 0.71 - 0.026 * (bucket - 1))
    return 22.3 + 1.44 * (bucket - 1) + per_column * bucket * width


def plain_ms(bucket: int) -> float:
    """A plain decode step's time (ms), bucket rows: a fit of one card's steps at 8K context."""
    return 16.0 + 1.07 * bucket


def mtp_step_ms(width: int, bucket: int) -> float:
    """A speculative step's time (ms) with the MTP layer, bucket rows of width columns: a fit of
    one card's MTP-2 steps at 8K context."""
    return 18.0 + 0.92 * bucket * width


def choose_width(
    widths: Sequence[int], accepted: np.ndarray, width: int, bucket: int = 1
) -> int:
    """The next verify step's columns per row: the width of the most expected tokens per ms
    (step_ms), each row's tokens expected from its accepted drafts this step (of width -
    1 verified) through SURVIVAL."""
    if len(widths) == 1:
        return widths[0]
    classes = []
    for a in np.asarray(accepted).tolist():
        if a >= width - 1 and width < 16:  # all accepted: at least a
            classes.append(8 if width <= 4 else 9)
        else:
            classes.append(SURVIVAL_CLASS[min(int(a), 15)])
    sums = np.cumsum(
        SURVIVAL, axis=1
    )  # sums[c, j]: expected drafts accepted of the first j
    best, choice = -1.0, widths[-1]
    for w in widths:
        tokens = sum(1.0 + sums[c, min(w - 1, 15)] - SURVIVAL[c, 0] for c in classes)
        rate = tokens / step_ms(w, bucket)
        if rate > best:
            best, choice = rate, w
    return choice


SMALL = 256  # bytes of each small device buffer
WORKSPACE_RESERVE = 2 << 30  # bytes held back for recipe workspaces
CAPACITY_QUANTUM = 8
SCORE_TOPS = (0, 64, 1024)  # top-logprob list lengths of the scoring recipes
OFFSET_STRIDE = 32  # Philox offsets per position (resin.backend.sampling)
# A split model on several cards all-gathers its partials in decode and verify steps whose columns times ranks are at
# most GATHER_LIMIT, and each graph sums them in rank order (rsum). Wider steps all-reduce them.
# RESIN_QWEN_TP_GATHER=0: always all-reduce.
GATHER_LIMIT = int(os.environ.get("RESIN_QWEN_TP_GATHER", "64"))
# The native executor puts the first slots of a decode or verify step in launches of WIDE_SLOTS slots, and only the
# rest in the mode's narrow launches: a long context has far fewer attention launches. RESIN_QWEN_WIDE_SLOTS=0:
# narrow launches only.
WIDE_SLOTS = int(os.environ.get("RESIN_QWEN_WIDE_SLOTS", "64"))
# A native prefill chunk launches the slots whose every key each query of the block sees first, without a mask
# (the same results), and the others after them with one.
# RESIN_QWEN_UNMASKED=0: every slot launch has a mask.
UNMASKED = os.environ.get("RESIN_QWEN_UNMASKED", "1") != "0"
DYNAMIC = (
    "slots",
    "k_pages",
    "v_pages",
    "mtp_prev",
    "mtp_tok",
    "fix_tin",
)  # bound per launch


def prefill_chunks(
    n: int, buckets: Sequence[int] = PREFILL_BUCKETS
) -> list[tuple[int, int]]:
    """(real tokens, bucket T) per chunk: the largest bucket while it fits, then the smallest
    bucket holding the rest when that pads by under 25%, else the largest bucket that fits."""
    chunks = []
    while n > 0:
        real = min(n, buckets[-1])
        if real < buckets[-1]:
            bucket = min(b for b in buckets if b >= real)
            if bucket > 1.25 * real:
                real = max([b for b in buckets if b <= real] or [real])
        chunks.append((real, min(b for b in buckets if b >= real)))
        n -= real
    return chunks


def _round(nbytes: int) -> int:
    return max(SMALL, -(-nbytes // SMALL) * SMALL)


@dataclass(frozen=True, slots=True)
class RowCache:
    """One sequence's cache for a step: its first position, the arena rows of its KV blocks (and
    MTP KV blocks) in logical order covering every position the step writes, and its STATE
    blocks: ``state_src`` holds its state after position pos0 - 1 (-1: fresh), ``state_dst``
    receives the state after the step's last token; ``state_rows`` (from logical block
    ``state_first``) resolve any position's block (a verify step commits a variable number)."""

    pos0: int
    kv_blocks: np.ndarray
    state_src: int
    state_dst: int
    mtp_blocks: np.ndarray | None = None
    state_first: int = 0
    state_rows: tuple[int, ...] = ()
    draft_blocks: np.ndarray | None = None  # the DFlash drafter's DRAFT blocks (window)
    draft_first: int = 0  # the logical block of draft_blocks[0]

    def state_at(self, position: int) -> int:
        if not self.state_rows:
            return self.state_dst
        return self.state_rows[position // STATE_TOKENS - self.state_first]

    def draft_row(self, position: int) -> DraftRow:
        """The drafter's view of this row with its block at ``position``."""
        if self.draft_blocks is None:
            raise ValueError("the row has no DRAFT blocks")
        return DraftRow(position, self.draft_blocks, self.draft_first)


@dataclass(frozen=True, slots=True)
class Prefilled:
    """A prefill's next token, its prompt scores and (MTP) the drafts after the token."""

    token: int
    scores: Scores | None = None
    drafts: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Verified:
    """A verify step: each row's emitted tokens (y_0 .. y_m), its accepted draft count m, its
    next drafts, and the step's columns of each row (for their logprobs)."""

    tokens: list[list[int]]
    accepted: np.ndarray
    drafts: np.ndarray
    columns: list[np.ndarray] = field(default_factory=list)
    width: int = 0
    """columns per row of the step (its drafts + 1)"""


class Staging:
    """Pinned host buffers one in-flight step copies its outputs into, and its two events:
    0 after the step's outputs (tokens, chosen-token logprobs and top lists per column; a
    verify step's MTP drafts per column), 1 after a verify step's draft chain (drafts 2 .. K,
    (K, B))."""

    def __init__(self, device: Device, columns: int, drafts: int, rows: int, slot: int):
        self.device, self.slot = device, slot
        self.tokens = device.host_buffer(columns, np.int32)
        self.mtp_tokens = device.host_buffer(columns, np.int32)
        self.lp = device.host_buffer(columns, np.float32)
        self.top_idx = device.host_buffer((columns, TOP_MAX), np.int32)
        self.top_lp = device.host_buffer((columns, TOP_MAX), np.float32)
        self.drafts = device.host_buffer((max(drafts, 1), rows), np.int32)
        # a DFlash block step's [anchor, drafts] per row, ``next_width`` per row
        self.next = device.host_buffer((rows, drafts + 1), np.int32)
        self.next_width = drafts + 1
        self.accepted = device.host_buffer(rows, np.int32)  # spec_fix's counts

    def next_drafts(self, rows: int, drafts: int) -> np.ndarray:
        """(rows, drafts) i64: the staged block step's drafts per row, zeros past its width."""
        w = self.next_width
        out = np.zeros((rows, drafts), np.int64)
        flat = self.next.reshape(-1)[: rows * w].reshape(rows, w)
        n = min(drafts, w - 1)
        out[:, :n] = flat[:, 1 : n + 1]
        return out

    def record(self, event: int = 0) -> None:
        self.device.record(self.slot + 2 * event)

    def wait(self, event: int = 0) -> None:
        self.device.wait(self.slot + 2 * event)


@dataclass(frozen=True, slots=True)
class _Mode:
    """A step shape: ``name`` keys its recipes (p{T} prefill, d{B} decode, v{B} verify)."""

    name: str
    columns: int
    per_row: int
    query_block: int
    slots: int
    wide: int = 0
    """slots of the mode's wide launches (0: none; the native executor's decode and verify steps)"""
    full: bool = False
    """the wide launches take the slots that need no mask (a fused prefill's), not the step's first slots"""


@dataclass(frozen=True, slots=True)
class Launched:
    """A verify pass left running (``Qwen35Gaudi.verify_launch``), for ``verify_finish``."""

    rows: Sequence[RowCache]
    drafts: np.ndarray | Callable[[], np.ndarray]
    sampling: Sampling | None
    stage: Staging | None
    mode: _Mode
    device_accept: bool = False
    """``spec_fix`` accepted on the device and the commit and block step are already queued"""
    next_width: int = 0


@dataclass(frozen=True, slots=True)
class _Binds:
    slots: np.ndarray
    addresses: np.ndarray


class _SpecHost:
    """The model as its drafter's ``DrafterHost``: block steps take the speculative row buckets
    (``spec_buckets``), not every decode bucket."""

    def __init__(self, m) -> None:
        self._m = m

    @property
    def decode_buckets(self) -> Sequence[int]:
        return self._m.spec_buckets

    def __getattr__(self, name: str):
        return getattr(self._m, name)


class Qwen35Gaudi:
    """``prefill(row, ids)`` runs one sequence's tokens; ``decode(rows, tokens)`` one token of
    each sequence; with MTP, ``verify(rows, anchors, drafts)`` one speculative step of each."""

    def __init__(
        self,
        device: Device,
        checkpoint: str,
        *,
        parent_capacity: int | None = None,
        memory_utilization: float = 0.9,
        workspace_reserve: int = WORKSPACE_RESERVE,
        max_positions: int = 32768,
        buckets: Sequence[int] = PREFILL_BUCKETS,
        decode_buckets: Sequence[int] = DECODE_BUCKETS,
        prefill_slots: int = 16,
        decode_slots: int | None = None,
        mtp: int = 0,
        layers: int | None = None,
        drafter: str | None = None,
        defer_cache: bool = False,
        verify_widths: Sequence[int] | None = None,
        tensor_parallel: tuple[int, int] | None = None,
        split: bool | None = None,
        reduce_capacity: Callable[[int], int] | None = None,
        native: bool = False,
        cache_types: tuple[str, str] = ("bf16", "bf16"),
    ) -> None:
        """``tensor_parallel``: (rank, ranks) of a tensor-parallel model, whose part of the model this rank
        holds (``loading.shard``); the caller has made the device's communicator.
        ``split``: split layers (``layers.split_entry``), which tensor parallelism takes and
        ``RESIN_QWEN_TP_SPLIT=1`` forces on one card; only the native executor runs them.
        ``reduce_capacity``: the cache capacity every rank agrees on (the smallest).
        ``native``: the native executor makes the steps, so the modes also get its launches (wide slot launches,
        a prefill chunk's slots without a mask); the Python executor takes neither, nor split layers.
        ``cache_types``: the types of the K and V pages of the full-attention layers and the MTP layer
        (``cache.KV_TYPES``: bf16 or fp8)."""
        self.reduce_capacity = reduce_capacity
        self.device = device
        ckpt = Qwen35Checkpoint(checkpoint)
        self.config = config = ckpt.config
        shard: DenseShard | None = None
        if tensor_parallel is not None and tensor_parallel[1] > 1:
            shard = ckpt.shard(*tensor_parallel)
        self.shard = shard
        self.nranks = 1 if shard is None else shard.nranks
        if split is None:
            split = self.nranks > 1 or os.environ.get("RESIN_QWEN_TP_SPLIT") == "1"
        if self.nranks > 1 and not split:
            raise ValueError("tensor parallelism takes split layers")
        if split and not native:
            raise ValueError("split layers run on the native executor only (not with the oldqwen feature)")
        self.split = split
        self.dims = Dims.from_config(config)
        # the MTP layer stays whole on every rank (``Qwen35Checkpoint.weights``): it keeps the model's dimensions
        self.mtp_dims = self.dims
        if shard is not None:
            self.dims = self.dims.shard(shard)
        # the MTP layer's drafts (0: none), or the DFlash 2 drafter's block
        self.dfl_cfg = None if drafter is None else DrafterConfig.load(drafter, config)
        if self.dfl_cfg is None:
            self.mtp = self.K = mtp
        else:  # the drafter's block makes dfl_cfg.drafts; a step verifies the first K of them
            if not 0 <= mtp <= self.dfl_cfg.drafts:
                raise ValueError(
                    f"the drafter makes up to {self.dfl_cfg.drafts} drafts a step, not {mtp}"
                )
            self.mtp, self.K = 0, mtp or self.dfl_cfg.drafts
        self.drafter: Drafter | None = None
        # the columns per row a verify step may take (the largest: every draft); a DSpine
        # drafter's step takes the one ``choose_width`` picks from the last step's acceptance
        full = self.K + 1
        if verify_widths is None:
            dspine = self.dfl_cfg is not None and self.dfl_cfg.kind == "dspine"
            verify_widths = DSPINE_WIDTHS if dspine else ()
        self.verify_widths = (
            tuple(
                sorted({int(w) for w in verify_widths if 2 <= int(w) < full} | {full})
            )
            if self.K
            else ()
        )
        self.next_width = full
        # a test's verify width for every step after a prompt (None: ``choose_width``'s, from the acceptance)
        self.force_width: int | None = None
        # device acceptance for drafter steps (spec_fix), and the last step's counts that pick
        # the next width
        self.device_accept = True
        self._last_accepted: np.ndarray = np.zeros(0, np.int64)
        self._last_width = full
        self._fix_launches = 0
        # verify acceptance counters (and the benchmark's fixed-acceptance mode)
        self.speculation = SpeculativeStats(self.K)
        self.geometry = CacheGeometry.from_config(
            config,
            mtp=self.mtp > 0,
            draft_layers=0 if self.dfl_cfg is None else self.dfl_cfg.layers,
            draft_layer_rows=0 if self.dfl_cfg is None else self.dfl_cfg.layer_rows,
            draft_window=0 if self.dfl_cfg is None else self.dfl_cfg.window,
            draft_write=max(buckets),
            shard=shard,
            mtp_kv_groups=self.mtp_dims.kv_groups,
            # on one card the MTP layer takes the target's attention recipes and their shorter arena view (a split
            # model's MTP layer has recipes of its own over the whole arena)
            mtp_short_view=self.mtp > 0 and not self.split,
            kv8=kv8_bits(cache_types),
        )
        self.n_layers = layers or config.num_hidden_layers
        self.mtp_layer = config.num_hidden_layers  # the MTP layer's weight index
        self.full = [i for i in range(self.n_layers) if i in set(config.full_layer_ids)]
        self.gdn = [i for i in range(self.n_layers) if i not in set(self.full)]
        self.buckets = tuple(buckets)
        self.decode_buckets = tuple(decode_buckets)
        self.spec_buckets = tuple(b for b in self.decode_buckets if b <= SPEC_ROWS)
        self.max_positions = max_positions
        self.modes: dict[str, _Mode] = {}
        for T in self.buckets:
            qb = min(T, QUERY_BLOCK)
            full = native and UNMASKED and bool(FUSED_QUERY_BLOCK) and qb >= FUSED_QUERY_BLOCK
            self.modes[f"p{T}"] = _Mode(
                f"p{T}", T, T, qb, prefill_slots, prefill_slots if full else 0, full
            )
        for B in self.decode_buckets:
            # by default one 2048-key chunk of every sequence and KV group per launch
            S = decode_slots or max(16, self.dims.kv_groups * B)
            wide = WIDE_SLOTS if native and WIDE_SLOTS > S and WIDE_SLOTS % S == 0 else 0
            self.modes[f"d{B}"] = _Mode(f"d{B}", B, 1, 1, S, wide)
            for Tq in self.verify_widths if B in self.spec_buckets else ():
                name = verify_mode(B, Tq, self.K + 1)
                self.modes[name] = _Mode(name, B * Tq, Tq, Tq, S, wide)
        self.columns = max(m.columns for m in self.modes.values())
        self.recipes = RecipeCache(device)
        weight_layers = set(range(self.n_layers)) | (
            {self.mtp_layer} if self.mtp else set()
        )
        total = weight_file_bytes(ckpt.path, "*.safetensors")
        if drafter is not None:
            total += weight_file_bytes(Path(drafter), "*.safetensors")
        device.begin_load(total)
        # exact allocation sizes: free memory after the load sizes the cache arena
        self.loader = WeightLoader(device, self.recipes, min_alloc=0)
        self.loader.load(
            weight_items(
                ckpt.weights(mtp=self.mtp > 0, layers=weight_layers, shard=shard)
            )
        )
        if self.dfl_cfg is not None:
            logging.getLogger("wright").info(
                f"drafter {self.dfl_cfg.kind} from {drafter} "
                f"({'fp8_inc' if drafter_is_fp8(drafter) else 'bf16'})"
            )
            self.loader.load(
                weight_items(prepare_drafter(drafter, self.dfl_cfg, head=ckpt.head))
            )
        device.end_load()
        self.weights = self.loader.w
        self._buffers()
        self.memory_utilization = memory_utilization
        self.workspace_reserve = workspace_reserve
        if not defer_cache:
            self.attach(parent_capacity=parent_capacity)

    def attach(
        self, *, parent_capacity: int | None = None, arena: SharedArena | None = None
    ) -> None:
        """The cache arena, the recipes and their bindings. Without ``arena``, this model's
        own arena from the card memory that is free (or ``parent_capacity`` slabs); with it,
        the arena that this model shares with the other models of the card."""
        device = self.device
        if arena is not None:
            if self.dfl_cfg is not None:
                raise NotImplementedError("a DFlash drafter in a shared cache arena")
            self.geometry = replace(self.geometry, shared_slab_rows=arena.slab_rows)
            self.parent_capacity = arena.parent_capacity
            self.blocks = arena.blocks
            self.arena_bytes = arena.nbytes
            self.arena = arena.address
        else:
            if parent_capacity is None:
                free, total = device.memory_info()
                planned = plan_cache_capacity(
                    total,
                    free,
                    self.memory_utilization,
                    self.geometry.slab_bytes,
                    overhead_bytes=self.workspace_reserve,
                ).parent_capacity
                parent_capacity = max(
                    1, planned // CAPACITY_QUANTUM * CAPACITY_QUANTUM
                )
                if self.reduce_capacity is not None:
                    parent_capacity = self.reduce_capacity(parent_capacity)
            self.parent_capacity = parent_capacity
            self.blocks = self.geometry.arena_blocks(parent_capacity)
            scratch_rows = self.blocks * ARENA_BLOCK_ROWS
            if self.dfl_cfg is not None:  # the block steps' scratch pages follow the cache
                extra = Drafter.scratch_rows(self.dfl_cfg, self.spec_buckets[-1])
                self.blocks += -(-extra // ARENA_BLOCK_ROWS)
                self.drafter = Drafter(
                    _SpecHost(self), self.dfl_cfg, scratch_rows
                )  # its buffers first
            self.blocks += ARENA_PAD_BLOCKS  # the last attention layer's view ends past the slabs
            nbytes = self.blocks * ARENA_BLOCK_ROWS * ROW_BYTES
            self.arena_bytes = nbytes
            self.arena = device.malloc(nbytes)
            device.memset32(self.arena, 0, nbytes // 4)
        logging.getLogger("wright").info("preparing recipes")
        self._compile()
        workspace = max(r.workspace for r in self.recipe.values())
        if workspace > self.workspace_reserve:
            raise MemoryError(
                f"recipe workspace {workspace} exceeds the reserve {self.workspace_reserve}"
            )
        self._bind_all()

    # ---- setup
    def _buffers(self) -> None:
        d, dev = self.dims, self.device
        H, G, V = d.hidden, d.kv_groups, d.vocab
        T = self.columns
        B = self.decode_buckets[-1]
        W = d.hpg * T
        # the attention's buffers hold a whole MTP layer's (a tensor-parallel rank's layers have fewer KV groups)
        dm = self.mtp_dims if self.mtp else d
        G = max(G, dm.kv_groups)

        def buf(nbytes: int) -> int:
            return dev.malloc(_round(nbytes))

        def carry() -> dict[str, int]:
            return {
                "x": buf(H * T * 4),
                "h8": buf(H * T),
                "hs": buf(T * 4),
                "h16": buf(H * T * 2),
            }

        self.carry = [carry(), carry()]
        if self.split:  # the split layers' partials: the mixer's, then the MLP's (layers.split_entry)
            # RESIN_QWEN_TP_PARTS (a debug aid, default 2): the number of partial buffers, used in turn
            n_parts = int(os.environ.get("RESIN_QWEN_TP_PARTS", "2"))
            self.parts = [buf(H * T * 4) for _ in range(n_parts)]
            # steps of up to GATHER_LIMIT / ranks columns all-gather each partial into its gathered buffer (H, T, ranks)
            gather_t = min(GATHER_LIMIT // self.nranks, T) if self.nranks > 1 else 0
            self.gathered_parts = (
                [buf(self.nranks * H * gather_t * 4) for _ in self.parts]
                if gather_t
                else []
            )
            self.ones_t = buf(T * 4)
            dev.h2d(self.ones_t, np.ones(T, np.float32))
            # RESIN_QWEN_DUMP (a debug aid): decode steps of up to 8 columns copy, for each layer, the mixer's carry
            # out, its partial before and after the all-reduce, then the MLP's (6 slots of H x 8 f32 a layer)
            self.dump = (
                buf(6 * self.n_layers * H * 8 * 4) if os.environ.get("RESIN_QWEN_DUMP") else None
            )
        self.buf = {
            "p_out": buf(max(attn_proj(d), attn_proj(dm)) * T * 4),
            "q_all": buf(256 * W * G * 2),
            "acc_o": buf(256 * W * G * 4),
            "acc_m": buf(max(W, 64) * G * 4),
            "acc_l": buf(max(W, 64) * G * 4),
            "kv_rows": buf(T * 8 * 4),
            "tok": buf(T * 4),
            "pos": buf(T * 4),
            "logits": buf(V * T * 4),
            "tok_out": buf(T * 4),
            "lp_out": buf(T * 4),
            "top_idx": buf(TOP_MAX * T * 4),
            "top_lp": buf(TOP_MAX * T * 4),
            "samp_params": buf(PARAMS * T * 4),
            "score_rows": buf(SCORE_ROWS * 4),
            "score_pairs": buf(2 * SCORE_PAIRS * 4),
            "score_lp": buf(SCORE_PAIRS * 4),
            "score_stats": buf(16 * SCORE_ROWS * 4),
            "score_params": buf(PARAMS * SCORE_ROWS * 4),
            "score_top_v": buf(SCORE_TOPS[-1] * SCORE_ROWS * 4),
            "score_top_i": buf(SCORE_TOPS[-1] * SCORE_ROWS * 4),
            "cs_tab": buf(self.max_positions * 512),
            "hidden_table": buf(2 * B * 4),
        }
        for name in (
            "meta",
            "one",
            "last",
            "sink_conv",
            "sink_ssm",
            "sink_kv",
            "sink_acc",
            "sink_slots",
            "sink_commit",
            "sink_hidden",
        ):
            self.buf[name] = buf(SMALL)
        # per GDN layer: prefill [conv_src, ssm_src, conv_dst, ssm_dst] (1,) each; decode and
        # verify gdn_bases (4, B) and conv_dst (B,)
        self.pre_bases = buf(len(self.gdn) * 4 * SMALL)
        self.dec_stride = _round(4 * B * 4) + _round(B * 4)
        self.dec_bases = buf(len(self.gdn) * self.dec_stride)
        # slot tables of every launch of a step
        S = max(m.slots for m in self.modes.values())
        self.launch_stride = _round(S * 8 * 4) + 2 * _round(S * 16 * 4)
        chunks = -(-self.max_positions // SLOT_KEYS)
        self.max_launches = max(
            -(
                -(m.columns // m.per_row)
                * -(-m.per_row // m.query_block)
                * G
                * chunks
                // m.slots
            )
            for m in self.modes.values()
        )
        self.tables = buf(self.max_launches * self.launch_stride)
        dev.h2d(self.buf["cs_tab"], rope_table(self.max_positions, self.config))
        dev.h2d(self.buf["one"], np.ones(64, np.float32))
        # scoring rows: greedy, with the log-sum-exp
        score = row_params(SCORE_ROWS, d.vocab, logprobs=np.ones(SCORE_ROWS))
        dev.h2d(self.buf["score_params"], score)
        # two in-flight steps' output copies (the Engine keeps one step ahead)
        # two staging sets, each with two events (slot + 2 event): four event slots of this model
        events = dev.event_slots(4)
        self.staging = [
            Staging(dev, T, self.K, B, events + slot) for slot in (0, 1)
        ]
        self.spec_buf: dict[str, int] = {}
        if self.K:
            Tq, L = self.K + 1, len(self.gdn)
            B = self.spec_buckets[-1]  # rows of a speculative step
            self.spec_buf = {
                "commit_bases": buf(4 * B * L * 4),
                "commit_counts": buf(B * L * 4),
                # an enqueued MTP step's continuation tokens [anchor, d_1 .. d_K] per row: the
                # anchor and d_1 from the host, d_2 .. d_K copied from the draft chain
                "next_tok": buf(B * Tq * 4),
            }
            if self.dfl_cfg is not None:  # device acceptance (spec_fix)
                c = self.dfl_cfg
                self.fix_patch = (
                    c.layers
                    * (c.window // SLOT_KEYS + 2)
                    * slots_per_launch(B, c.pairs)
                )
                self.spec_buf.update(
                    {
                        "fix_acc": buf(B * 4),
                        "fix_rows": buf(8 * B * 4),
                        "fix_cb0": buf(4 * B * L * 4),
                        "fix_cb1": buf(4 * B * L * 4),
                        "fix_patch": buf(4 * self.fix_patch * 4),
                        "fix_meta": buf(4 * 4),
                    }
                )
            self.pend_layer = 128 * 3 * d.nv * Tq * 4  # bytes per sequence and layer
            self.pconv_layer = d.conv_dim * Tq * 4
            self.pend = buf(self.pend_layer * B * L)
            self.pconv = buf(self.pconv_layer * B * L)
        if self.mtp:
            self.mtp_carry = [carry(), carry()]
            self.mtp_buf = {
                "mtp_cols": buf(T * 4),
                "mtp_tok": buf(T * 4),
                "mtp_zero": buf(H * 4),
                "mtp_tok_out": buf(T * 4),
                "mtp_lp_out": buf(T * 4),
                "mtp_top_idx": buf(TOP_MAX * T * 4),
                "mtp_top_lp": buf(TOP_MAX * T * 4),
                "mtp_logits": buf(V * T * 4),
                "mtp_params": buf(PARAMS * T * 4),
            }
            dev.memset32(self.mtp_buf["mtp_zero"], 0, H)

    def _compile(self) -> None:
        progress.building_graphs()
        dev, d, P = self.device, self.dims, self.max_positions
        nb = self.blocks
        na = self.geometry.attention_blocks(self.parent_capacity)
        graphs = {}
        for m in self.modes.values():
            T = m.columns
            verify = m.per_row if m.name[0] == "v" else 0
            graphs[("embed", m.name)] = embed_graph(dev, d, T)
            split = self.split
            R = self.gathered(m.name)
            graphs[("gdn", m.name)] = gdn_layer_graph(
                dev,
                d,
                T,
                nb,
                decode=m.name[0] != "p",
                verify=verify,
                split=split,
                ranks=R,
            )
            if split:
                graphs[("gdn0", m.name)] = gdn_layer_graph(
                    dev,
                    d,
                    T,
                    nb,
                    decode=m.name[0] != "p",
                    verify=verify,
                    split=True,
                    first=True,
                )
                graphs[("mlp", m.name)] = mlp_graph(dev, d, T, R)
            kv8 = self.geometry.kv8
            graphs[("pre", m.name)] = attn_pre_graph(
                dev, d, T, na, P, split=split, ranks=R, kv8=kv8
            )
            graphs[("slots", m.name)] = attn_slots_graph(
                dev, d, T, m.slots, na, m.query_block, per_column=True, kv8=kv8
            )
            if m.wide:
                graphs[("wslots", m.name)] = attn_slots_graph(
                    dev,
                    d,
                    T,
                    m.wide,
                    na,
                    m.query_block,
                    per_column=True,
                    mask=not m.full,
                    kv8=kv8,
                )
            graphs[("post", m.name)] = attn_post_graph(dev, d, T, split=split)
            graphs[("head", m.name)] = head_graph(
                dev, d, T, 1 if m.name[0] == "p" else T, split=split, ranks=R
            )
            if self.mtp:
                graphs[("mtp_in", m.name)] = mtp_in_graph(dev, d, T, self.columns)
            if self.mtp and split:  # the whole MTP layer on every rank: its attention and head unsplit
                dm = self.mtp_dims
                graphs[("mtp_pre", m.name)] = attn_pre_graph(
                    dev, dm, T, nb, P, kv8=kv8
                )
                if m.name[0] != "p":  # prefill chunks store its keys only
                    graphs[("mtp_slots", m.name)] = attn_slots_graph(
                        dev,
                        dm,
                        T,
                        m.slots,
                        nb,
                        m.query_block,
                        per_column=True,
                        kv8=kv8,
                    )
                    graphs[("mtp_post", m.name)] = attn_post_graph(dev, dm, T)
                    graphs[("mtp_head", m.name)] = head_graph(dev, dm, T, T)
            if verify:
                B = T // verify
                graphs[("commit", m.name)] = commit_graph(
                    dev, d, B, verify, len(self.gdn), nb
                )
                if self.drafter is not None:
                    graphs[("spec_fix", m.name)] = spec_fix_graph(
                        dev,
                        B,
                        verify,
                        len(self.gdn),
                        self.fix_patch,
                        self.drafter.TD,
                        self.drafter.tables_words,
                    )
        for K in SCORE_TOPS:
            graphs[("score", f"k{K}")] = score_graph(dev, d, self.columns, K)
        if self.mtp:
            graphs[("hidden", "x")] = hidden_store_graph(dev, d, self.columns, nb)
        if self.drafter is not None:
            graphs.update(self.drafter.graphs(nb))
        recipes = self.recipes.compile_many(list(graphs.values()))
        self.recipe: dict[tuple[str, str], Recipe] = dict(zip(graphs, recipes))

    def _bind_all(self) -> None:
        self.binds: dict[tuple[str, str, int, str], _Binds] = {}
        # launch plans of whole passes, by (pass, mode, slot launches per layer, ...)
        self._plans: dict[tuple, LaunchPlan] = {}
        M = self.mtp_layer
        for key, recipe in self.recipe.items():
            if key[0].startswith("dfl_"):
                continue  # the drafter binds its own recipes
            kind, mode = key
            if kind == "gdn":
                targets = [(i, "") for i in self.gdn if not (self.split and i == 0)]
            elif kind == "gdn0":  # a split model's first layer
                targets = [(0, "")]
            elif kind == "mlp":  # a split layer's MLP
                targets = [(i, "") for i in range(self.n_layers)]
            elif kind in ("pre", "slots", "wslots", "post"):
                targets = [(i, "") for i in self.full]
                if self.mtp and not self.split and kind != "wslots":  # (MTP passes: narrow only)
                    targets.append((M, "mtp"))
            elif kind in ("mtp_pre", "mtp_slots", "mtp_post"):
                targets = [(M, "mtp")]
            elif kind == "mtp_head":
                targets = [(-1, "mtp")]
            elif kind == "head":
                targets = [(-1, "")] + ([(-1, "mtp")] if self.mtp and not self.split else [])
            elif kind == "mtp_in":
                targets = [(M, "mtp"), (M, "mtpc")]
            else:
                targets = [(-1, "")]
            for layer, variant in targets:
                self.binds[(kind, mode, layer, variant)] = self._bind_table(
                    recipe, kind, mode, layer, variant
                )

    def _bind_table(
        self, recipe: Recipe, kind: str, mode: str, layer: int, variant: str
    ) -> _Binds:
        """Every persistent tensor of ``recipe`` launched as ``layer`` (-1: embed / head) in
        ``variant`` ("": the target, "mtp": the MTP layer fed from the target, "mtpc": the MTP
        layer fed from its own previous output), except the per-launch DYNAMIC ones."""
        w, g = self.weights, self.geometry
        final = self.carry[self.n_layers % 2]
        names = dict(self.buf)
        names.update(self.spec_buf)
        if self.drafter is not None:
            dr = self.drafter
            names.update(btok=dr.tok, bpos=dr.pos, banc=dr.anchors, dtables=dr.tables)
        if self.mtp:
            names.update(self.mtp_buf)
        if variant:
            src, dst = self.mtp_carry
            if kind == "mtp_in":
                x = self.final_x if variant == "mtp" else self.mtp_carry[1]["x"]
                norm = "out_norm" if variant == "mtp" else "mtp.norm"
                names.update({f"{k}_out": v for k, v in src.items()})
                names.update(x=x, mtp_xnorm=w[norm])
            elif kind in ("head", "mtp_head"):
                names["h16"] = dst["h16"]
                for k in ("tok_out", "lp_out", "top_idx", "top_lp", "logits"):
                    names[k] = self.mtp_buf[f"mtp_{k}"]
                names["samp_params"] = self.mtp_buf["mtp_params"]
            else:  # the MTP decoder layer: its own KV blocks, mtp.norm as the next norm
                names.update(src)
                names.update({f"{k}_out": v for k, v in dst.items()})
                names["arena"] = self.arena
                names["next_norm"] = w["mtp.norm"]
        else:
            src = 0 if layer < 0 else layer % 2
            names.update(self.carry[src])
            names.update({f"{k}_out": v for k, v in self.carry[1 - src].items()})
            names["arena"] = self.arena
            if (
                layer < 0
            ):  # embed writes carry 0; the head and scoring read the final one
                names.update({f"{k}_out": v for k, v in self.carry[0].items()})
                names["h16"] = final["h16"]
                names["x"] = final["x"]
                names["next_norm"] = w["l0.attn_norm"]
                if self.K:
                    names["gdn_pend"], names["gdn_pconv"] = self.pend, self.pconv
            else:
                nxt = (
                    f"l{layer + 1}.attn_norm"
                    if layer + 1 < self.n_layers
                    else "out_norm"
                )
                names["next_norm"] = w[nxt]
            if self.split and kind != "embed":
                names.update(self._split_names(kind, layer, mode))
            if layer in self.full:
                a = self.full.index(layer)
                names["arena"] = self.arena + a * g.kv_layer_rows * ROW_BYTES
            if layer in self.gdn:
                j = self.gdn.index(layer)
                if mode[0] == "p":
                    for k, key in enumerate(
                        ("conv_src", "ssm_src", "conv_dst", "ssm_dst")
                    ):
                        names[key] = self.pre_bases + (4 * j + k) * SMALL
                else:
                    base = self.dec_bases + j * self.dec_stride
                    names["gdn_bases"] = base
                    names["conv_dst"] = base + _round(4 * self.decode_buckets[-1] * 4)
                if (
                    mode[0] == "v"
                ):  # back to back per layer at this width (commit_graph)
                    m = self.modes[mode]
                    B, Tq = m.columns // m.per_row, m.per_row
                    pend = self.pend_layer // (self.K + 1) * Tq
                    pconv = self.pconv_layer // (self.K + 1) * Tq
                    names["gdn_pend"] = self.pend + j * B * pend
                    names["gdn_pconv"] = self.pconv + j * B * pconv
        pairs = []
        for name in recipe.persistent:
            if name in DYNAMIC:
                continue
            if layer >= 0 and f"l{layer}.{name}" in w:
                address = w[f"l{layer}.{name}"]
            elif name in w:
                address = w[name]
            else:
                address = names[name]
            pairs.append((recipe.slot(name), address))
        return _Binds(
            np.array([p[0] for p in pairs], np.int32),
            np.array([p[1] for p in pairs], np.uint64),
        )

    @property
    def final_x(self) -> int:
        """The target's final residual: the last layer's carry out, or a split model's head's (which adds the last
        partial)."""
        if self.split:
            return self.carry[1]["x"]
        return self.carry[self.n_layers % 2]["x"]

    def mtp_kind(self, kind: str) -> str:
        """The recipe kind of the MTP layer's ``pre``, ``slots``, ``post`` or ``head`` launch: a split model's MTP
        layer stays whole (``mtp_pre`` ...), one card's takes the target's recipes."""
        return f"mtp_{kind}" if self.split else kind

    def part(self, layer: int, mlp: bool) -> int:
        """The partial that layer ``layer``'s mixer (or MLP) writes; the next graph reads it (the head: layer -1
        reads the last MLP's)."""
        if layer < 0:
            return self.parts[(2 * self.n_layers) % len(self.parts)]
        return self.parts[(2 * layer + 1 + int(mlp)) % len(self.parts)]

    def gathered(self, mode: str) -> int:
        """The ranks whose partials a step of ``mode`` all-gathers (decode and verify steps whose columns times ranks
        are at most ``GATHER_LIMIT``), else 1: it all-reduces them."""
        m = self.modes.get(mode)  # (scoring and drafter recipes have no step mode)
        if (
            m is not None
            and self.split
            and self.nranks > 1
            and mode[0] != "p"
            and m.columns * self.nranks <= GATHER_LIMIT
        ):
            return self.nranks
        return 1

    def gathered_part(self, part: int) -> int:
        """The buffer that partial ``part`` (a buffer of ``self.parts``) is all-gathered into."""
        return self.gathered_parts[self.parts.index(part)]

    def _split_names(self, kind: str, layer: int, mode: str) -> dict[str, int]:
        """A split layer's carries and partials (``layers.split_entry``): the mixer (``gdn``, ``gdn0``,
        ``pre``; ``post`` ends it) reads carry 0 and the last MLP's partial and writes carry 1 and its partial, the
        MLP reads carry 1 and the mixer's partial and writes carry 0 and its partial, and the head reads carry 0
        and the last MLP's partial (``part``: two partials in turn by default)."""
        c0, c1 = self.carry
        names = {"ones_t": self.ones_t}
        mixer_out = self.part(layer, False) if layer >= 0 else self.part(-1, False)
        mlp_in = self.parts[(2 * layer) % len(self.parts)] if layer >= 0 else self.part(-1, False)
        if kind in ("gdn", "gdn0", "pre", "head"):
            src, dst, pin, pout = c0, c1, mlp_in, mixer_out
        else:  # post, mlp (and slots, which take neither)
            src, dst = c1, c0
            pin, pout = mixer_out, self.part(layer, True)
        if kind == "post":
            pout = mixer_out
        names.update(src)
        names.update({f"{k}_out": v for k, v in dst.items()})
        if self.gathered(mode) > 1:
            pin = self.gathered_part(pin)
        names.update(part_in=pin, part_out=pout)
        return names

    # ---- launching
    def _upload(self, address: int, array: np.ndarray) -> None:
        """Queue a table upload behind the launches already submitted."""
        array = np.ascontiguousarray(array)
        if array.nbytes <= 65536:
            self.device.h2d_nb(address, array)
        else:
            self.device.h2d(address, array)

    # ---- the drafter's host (``dflash.DrafterHost``)
    def weight(self, name: str) -> int:
        return self.weights[name]

    def buffer(self, name: str) -> int:
        return self.buf[name]

    def recipe_of(self, key: tuple) -> Recipe:
        return self.recipe[key]

    def upload(self, address: int, array: np.ndarray) -> None:
        self._upload(address, array)

    def tap_source(self, layer: int) -> int:
        """the output carry of target layer ``layer`` (split layers: the next layer's mixer adds the last
        partial into carry 1, and the tap runs after it)"""
        if self.split:
            return self.carry[1]["x"]
        return self.carry[(layer + 1) % 2]["x"]

    def tap_graph(self, tokens: int) -> Graph:
        return dflash_tap_graph(self.device, self.dims, tokens)

    def _launch(
        self,
        kind: str,
        mode: str,
        layer: int = -1,
        variant: str = "",
        extra: dict[str, int] | None = None,
    ) -> None:
        recipe = self.recipe[(kind, mode)]
        b = self.binds[(kind, mode, layer, variant)]
        recipe.bind_slots(b.slots, b.addresses)
        if extra:
            recipe.bind_all(extra)
        recipe.launch()

    def _slot_tables(self, mode: _Mode, rows: Sequence[SlotRow]) -> list[tuple]:
        """Write every slot launch's tables; returns each launch's (slots, k, v) addresses."""
        S = mode.slots
        launches = plan_slots(rows, mode.query_block, self.dims.hpg, S)
        if len(launches) > self.max_launches:
            raise ValueError(
                f"{len(launches)} attention launches exceed {self.max_launches}"
            )
        stride, k_off = self.launch_stride, _round(S * 8 * 4)
        v_off = k_off + _round(S * 16 * 4)
        table = np.zeros((len(launches), stride // 4), np.int32)
        for i, launch in enumerate(launches):
            table[i, : S * 8] = launch.slots.reshape(-1)
            table[i, k_off // 4 : k_off // 4 + S * 16] = launch.k_pages.reshape(-1)
            table[i, v_off // 4 : v_off // 4 + S * 16] = launch.v_pages.reshape(-1)
        self._upload(self.tables, table)
        recipe = self.recipe[("slots", mode.name)]
        slots = np.array(
            [recipe.slot(n) for n in ("slots", "k_pages", "v_pages")], np.int32
        )
        out = []
        for i in range(len(launches)):
            base = self.tables + i * stride
            addresses = np.array([base, base + k_off, base + v_off], np.uint64)
            out.append((slots, addresses))
        return out

    def _step(
        self,
        kind: str,
        mode: str,
        layer: int = -1,
        variant: str = "",
        extra: dict[str, int] | None = None,
    ) -> tuple[Recipe, np.ndarray, np.ndarray]:
        """One launch of a plan: ``_launch``'s recipe and bindings."""
        recipe = self.recipe[(kind, mode)]
        b = self.binds[(kind, mode, layer, variant)]
        if not extra:
            return recipe, b.slots, b.addresses
        slots = np.array([recipe.slot(name) for name in extra], np.int32)
        addresses = np.array(list(extra.values()), np.uint64)
        return (
            recipe,
            np.concatenate([b.slots, slots]),
            np.concatenate([b.addresses, addresses]),
        )

    def _attention_steps(self, mode: _Mode, layer: int, variant: str, tables) -> list:
        """A full-attention layer's launches: pre, one slot launch per table, post."""
        recipe = self.recipe[("slots", mode.name)]
        b = self.binds[("slots", mode.name, layer, variant)]
        steps = [self._step("pre", mode.name, layer, variant)]
        for slots, addresses in tables:
            steps.append(
                (
                    recipe,
                    np.concatenate([b.slots, slots]),
                    np.concatenate([b.addresses, addresses]),
                )
            )
        steps.append(self._step("post", mode.name, layer, variant))
        return steps

    def _pages(
        self, mode: _Mode, rows: Sequence[RowCache], mtp: bool, positions
    ) -> list:
        """Slot rows (one per sequence) of the target's or the MTP layer's pages; positions[i]
        is row i's first position."""
        g, out = self.geometry, []
        for i, row in enumerate(rows):
            blocks = row.mtp_blocks if mtp else row.kv_blocks
            k, v = g.kv_pages(blocks)
            real = mode.per_row if mode.name[0] != "p" else positions[1]
            out.append(SlotRow(i * mode.per_row, positions[0][i], real, k, v))
        return out

    # ---- the target pass
    def _target(
        self,
        mode: _Mode,
        rows: Sequence[RowCache],
        tokens: np.ndarray,
        real: int,
        params: np.ndarray,
        token_source: int | None = None,
    ) -> None:
        """Launch the target model over ``mode``'s columns: ``tokens`` (columns,), or a
        callable returning them (called once the step's tables are queued), or with
        ``token_source`` the device buffer holding them (a previous step's sampled tokens); a
        prefill's ``real`` tokens or ``real`` rows of a decode / verify batch; ``params``
        (columns or 1, PARAMS) sampler inputs."""
        if self.split:
            raise NotImplementedError("split layers run on the native executor only")
        dev, prefill = self.device, mode.name[0] == "p"
        T = mode.columns
        dev.h2d_nb(self.buf["samp_params"], params)
        pos = np.zeros(T, np.int32)
        if prefill:
            pos[:] = np.minimum(rows[0].pos0 + np.arange(T), self.max_positions - 1)
            starts = ([rows[0].pos0], real)
        else:
            q = mode.per_row
            for i, row in enumerate(rows):
                pos[i * q : (i + 1) * q] = row.pos0 + np.arange(q)
            starts = ([row.pos0 for row in rows], q)
        if (
            int(pos[: real if prefill else len(rows) * mode.per_row].max())
            >= self.max_positions
        ):
            raise ValueError(f"position exceeds {self.max_positions}")
        dev.h2d_nb(self.buf["pos"], pos)
        dev.h2d_nb(
            self.buf["meta"],
            np.array([rows[0].pos0 if prefill else 0, real, 0, 0], np.int32),
        )
        if prefill:
            dev.h2d_nb(self.buf["last"], np.array([real - 1], np.int32))
        self._write_state_tables(rows, prefill)
        slot_rows = self._pages(mode, rows, False, starts)
        self._upload(self.buf["kv_rows"], kv_rows(T, slot_rows, self.geometry.kv8))
        tables = self._slot_tables(mode, slot_rows)
        if token_source is None:
            tokens = tokens() if callable(tokens) else tokens
            dev.h2d_nb(self.buf["tok"], np.asarray(tokens).astype(np.int32))
        # a pass's launches depend only on its mode, slot launches and token source
        key = ("target", mode.name, len(tables), token_source)
        plan = self._plans.get(key)
        if plan is None:
            extra = None if token_source is None else {"tok": token_source}
            steps = [self._step("embed", mode.name, extra=extra)]
            taps = self.drafter.tap_steps(mode.name) if self.drafter is not None else {}
            for i in range(self.n_layers):
                if i in self.gdn:
                    steps.append(self._step("gdn", mode.name, i))
                else:
                    steps += self._attention_steps(mode, i, "", tables)
                if i in taps:  # the drafter's context projection of this layer's output
                    steps.append(taps[i])
            steps.append(self._step("head", mode.name))
            plan = self._plans[key] = LaunchPlan(self.device, steps)
        plan.run()

    def _write_state_tables(self, rows: Sequence[RowCache], prefill: bool) -> None:
        """Every GDN layer's state rows for this step: [conv_src, ssm_src, conv_dst, ssm_dst]
        per row (-1 for a fresh source and for padding)."""
        n = len(self.gdn)
        table4 = self._state_fields(
            [r.state_src for r in rows], [r.state_dst for r in rows]
        )
        if not prefill:
            B = self.decode_buckets[-1]
            table = np.full((n, self.dec_stride // 4), -1, np.int32)
            table[:, : 4 * len(rows)] = table4.reshape(n, -1)
            off = _round(4 * B * 4) // 4
            table[:, off : off + len(rows)] = table4[:, :, 2]
            self._upload(self.dec_bases, table)
        else:
            table = np.zeros((n, 4, SMALL // 4), np.int32)
            table[:, :, 0] = table4[:, 0]
            self._upload(self.pre_bases, table)

    def _state_fields(self, srcs, dsts) -> np.ndarray:
        """(layers, rows, 4) [conv_src, ssm_src, conv_dst, ssm_dst] of STATE block rows."""
        g, n = self.geometry, len(self.gdn)
        layer = np.arange(n, dtype=np.int64)[:, None] * g.state_layer_rows
        fields = []
        for bases in (srcs, dsts):
            base = np.asarray(bases, np.int64)[None, :]
            ssm = np.where(base >= 0, base + layer, -1)
            fields += [np.where(base >= 0, ssm + g.ssm_rows, -1), ssm]
        return np.stack(fields, axis=-1).astype(np.int32)

    # ---- the MTP layer pass
    def _mtp_pass(
        self,
        mode: _Mode,
        variant: str,
        rows: Sequence[RowCache],
        starts: Sequence[int],
        real: int,
        cols: np.ndarray,
        tokens: np.ndarray | int,
        prev: int,
        params: np.ndarray | None,
    ) -> None:
        """The MTP layer over ``mode``'s columns: column c reads token ``tokens`` [c] (or the
        device buffer ``tokens``), hidden column ``cols`` [c] of [prev | x] and sits at its
        row's position ``starts`` [row] + its index in the row; it writes MTP KV for the first
        ``real`` columns of each row (prefill: of the chunk). With ``params``, sample drafts;
        without, only the MTP KV: the input and the attention's pre launch (which stores K and V),
        no slot launches and no post."""
        dev, T = self.device, mode.columns
        dev.h2d_nb(self.mtp_buf["mtp_cols"], np.asarray(cols, np.int32))
        extra = {"mtp_prev": prev}
        if isinstance(tokens, np.ndarray):
            dev.h2d_nb(self.mtp_buf["mtp_tok"], tokens.astype(np.int32))
            extra["mtp_tok"] = self.mtp_buf["mtp_tok"]
        else:
            extra["mtp_tok"] = tokens
        pos = np.zeros(T, np.int32)
        q = mode.per_row
        for i, start in enumerate(starts):
            pos[i * q : i * q + (real if mode.name[0] == "p" else q)] = (
                start + np.arange(real if mode.name[0] == "p" else q)
            )
        dev.h2d_nb(self.buf["pos"], np.minimum(pos, self.max_positions - 1))
        slot_rows = self._pages(mode, rows, True, (list(starts), real))
        self._upload(self.buf["kv_rows"], kv_rows(T, slot_rows, self.geometry.kv8))
        keys_only = params is None
        tables = () if keys_only else self._slot_tables(mode, slot_rows)
        key = ("mtp", mode.name, variant, "keys" if keys_only else len(tables))
        plan = self._plans.get(key)
        if (
            plan is None
        ):  # mtp_prev and mtp_tok (the first launch's last bindings) are patched
            first = self._step(
                "mtp_in", mode.name, self.mtp_layer, variant, dict.fromkeys(extra, 0)
            )
            if keys_only:
                steps = [first, self._step("pre", mode.name, self.mtp_layer, "mtp")]
            else:
                steps = [
                    first,
                    *self._attention_steps(mode, self.mtp_layer, "mtp", tables),
                ]
            plan = self._plans[key] = LaunchPlan(self.device, steps)
        n = int(plan.starts[1])
        plan.patch(n - 2, extra["mtp_prev"])
        plan.patch(n - 1, extra["mtp_tok"])
        plan.run()
        if params is not None:
            dev.h2d_nb(self.mtp_buf["mtp_params"], params)
            self._launch("head", mode.name, -1, "mtp")

    def _prev_hidden(self, block_row: int) -> int:
        """Device address of a STATE block's hidden slot (the zero buffer for none)."""
        row = self.geometry.hidden_row(block_row)
        return self.mtp_buf["mtp_zero"] if row < 0 else self.arena + row * ROW_BYTES

    def _chain(
        self,
        rows: Sequence[RowCache],
        anchor_pos: Sequence[int],
        first: np.ndarray,
        first_cols: np.ndarray,
        sampling: Sampling | None,
        base_steps: Sequence[int],
        stage: Staging | None = None,
        scatter: int | None = None,
    ) -> np.ndarray:
        """Drafts d_1 .. d_K of each row: d_1 = ``first``; link j (MTP at position
        anchor + j - 1, token d_j, hidden: link j - 1's MTP output, the first link's at
        ``first_cols`` of the last MTP pass) samples d_(j+1) for position anchor + j + 1, which
        is ``base_steps`` [row] + j + 1 positions past the row's first sampled position. With
        ``stage``, drafts 2 .. K are copied into ``stage.drafts`` [j] without waiting (and
        returned as zeros); with ``scatter`` (a (rows, K + 1) token buffer) also into its
        column j + 1."""
        K, B = self.mtp, len(rows)
        drafts = np.zeros((B, K), np.int64)
        drafts[:, 0] = first
        if K == 1:
            return drafts
        bucket = min(b for b in self.decode_buckets if b >= B)
        mode = self.modes[f"d{bucket}"]
        tokens = np.zeros(bucket, np.int64)
        tokens[:B] = first
        cols = np.zeros(bucket, np.int64)
        cols[:B] = first_cols
        for j in range(1, K):
            starts = [anchor_pos[i] + j - 1 for i in range(B)]
            params = np.zeros((bucket, PARAMS), np.float32)
            if sampling is not None:
                params[:B] = sampling.at(range(B), [int(s) + j + 1 for s in base_steps])
            self._mtp_pass(
                mode,
                "mtpc",
                rows,
                starts,
                1,
                cols,
                tokens if j == 1 else self.mtp_buf["mtp_tok_out"],
                self.mtp_buf["mtp_zero"],
                params,
            )
            if stage is None:
                drafts[:, j] = self.device.d2h(
                    self.mtp_buf["mtp_tok_out"], (bucket,), np.int32
                )[:B]
            else:
                self.device.d2h_async(
                    stage.drafts[j, :bucket], self.mtp_buf["mtp_tok_out"]
                )
            if scatter is not None:
                b = np.arange(B)
                self.device.d2d_multi(
                    self.mtp_buf["mtp_tok_out"] + 4 * b,
                    np.full(B, 4),
                    scatter + 4 * (b * (K + 1) + j + 1),
                )
            cols = np.arange(bucket) + 1  # the previous link's output columns
        return drafts

    # ---- steps
    def prefill(
        self,
        row: RowCache,
        ids: Sequence[int],
        sampling: Sampling | np.ndarray | None = None,
        scores: ScoreRequest | None = None,
        draft: bool = True,
    ) -> Prefilled:
        """One sequence's tokens at row.pos0, in bucketed chunks: the next token, sampled with
        ``sampling`` (1 row) after the last chunk (None: greedy), the requested prompt scores
        and, with MTP and ``draft`` (the prompt ends here), the drafts after the token."""
        if isinstance(sampling, np.ndarray):
            sampling = Sampling(sampling)
        token, i = -1, 0
        chunks = prefill_chunks(len(ids), self.buckets)
        result = None
        if scores is not None:
            n = len(scores.positions)
            result = Scores([None] * n, [None] * n, [None] * n)
        drafts = None
        for n, (real, T) in enumerate(chunks):
            src = row.state_src if i == 0 else row.state_dst
            chunk = RowCache(
                row.pos0 + i,
                row.kv_blocks,
                src,
                row.state_dst,
                row.mtp_blocks,
                draft_blocks=row.draft_blocks,
                draft_first=row.draft_first,
            )
            last = n == len(chunks) - 1
            mode = self.modes[f"p{T}"]
            tokens = np.zeros(T, np.int64)
            tokens[:real] = ids[i : i + real]
            params = np.zeros((1, PARAMS), np.float32)
            if last and sampling is not None:
                params[:] = sampling.params[:1]
            self._target(mode, [chunk], tokens, real, params)
            token = int(self.device.d2h(self.buf["tok_out"], (1,), np.int32)[0])
            if scores is not None:
                inside = np.flatnonzero(
                    (scores.positions >= i) & (scores.positions < i + real)
                )
                self._score(scores, inside, i, result)
            if self.mtp:
                drafts = self._prefill_mtp(
                    chunk, ids, i, real, tokens, last and draft, sampling
                )
            elif self.drafter is not None:
                drafts = self._prefill_draft(chunk, mode, real, token, last and draft)
            i += real
        return Prefilled(token, result, drafts)

    def _prefill_draft(
        self, chunk: RowCache, mode: _Mode, real: int, token: int, last: bool
    ) -> np.ndarray | None:
        """The drafter's cache entries of a prefill chunk and, after the prompt's last chunk,
        the block after its sampled token (at the position after the prompt)."""
        drafter = self.drafter
        assert drafter is not None
        drafter.inject(mode.name, [(0, chunk.pos0, real, chunk.draft_row(chunk.pos0))])
        if not last:
            return None
        bucket = self.decode_buckets[0]
        self.next_width = self.K + 1  # the prompt's drafts: all of them
        drafter.block([chunk.draft_row(chunk.pos0 + real)], [token], bucket)
        return drafter.read_next(bucket)[0, 1:].astype(np.int64)

    def _prefill_mtp(
        self, chunk, ids, i, real, tokens, last, sampling
    ) -> np.ndarray | None:
        """The MTP layer over a prefill chunk (its KV), then the chunk's hidden into the STATE
        block; after the prompt's last chunk, the first draft following its sampled token (the
        draft chain would write MTP KV past the prompt, where a prefill leases no blocks, so
        drafts 2 .. K start as zeros)."""
        pos0, T = chunk.pos0, len(tokens)
        mode = self.modes[f"p{T}"]
        # column c: position pos0 - 1 + c, token ids[c], hidden c of [prev | x] (prev: the
        # hidden at pos0 - 1); a sequence's first chunk starts one later (no position -1)
        shift = 0 if pos0 > 0 else 1
        n = real - shift
        if n > 0:
            mtp_tokens = np.zeros(T, np.int64)
            mtp_tokens[:n] = ids[i + shift : i + real]
            cols = np.zeros(T, np.int64)
            cols[:n] = np.arange(n) + shift
            self._mtp_pass(
                mode,
                "mtp",
                [chunk],
                [pos0 - 1 + shift],
                n,
                cols,
                mtp_tokens,
                self._prev_hidden(chunk.state_src),
                None,
            )
        # the chunk's last hidden, for the next chunk's first MTP column
        self.device.h2d_nb(
            self.buf["hidden_table"],
            np.array([real - 1, self.geometry.hidden_row(chunk.state_dst)], np.int32),
        )
        self._launch("hidden", "x")
        if not last:
            return None
        # the sampled token at pos0 + real: MTP at pos0 + real - 1 drafts pos0 + real + 1
        one = self.modes[f"d{self.decode_buckets[0]}"]
        params = np.zeros((one.columns, PARAMS), np.float32)
        if sampling is not None:
            params[:1] = sampling.at([0], [1])
        cols = np.zeros(one.columns, np.int64)
        cols[0] = real
        self._mtp_pass(
            one,
            "mtp",
            [chunk],
            [pos0 + real - 1],
            1,
            cols,
            self.buf["tok_out"],
            self.mtp_buf["mtp_zero"],
            params,
        )
        drafts = np.zeros(self.mtp, np.int64)
        drafts[0] = self.device.d2h(self.mtp_buf["mtp_tok_out"], (1,), np.int32)[0]
        return drafts

    def _score(self, request: ScoreRequest, which: np.ndarray, start: int, out: Scores):
        """Score request positions ``which`` (columns position - start of the last chunk), in
        launches of up to SCORE_ROWS positions and SCORE_PAIRS target logprobs."""
        dev = self.device
        groups, group, pairs = [], [], 0
        for k in which.tolist():
            n = len(request.targets[k])
            if group and (len(group) == SCORE_ROWS or pairs + n > SCORE_PAIRS):
                groups.append(group)
                group, pairs = [], 0
            if n > SCORE_PAIRS:
                raise ValueError(
                    f"more than {SCORE_PAIRS} prompt targets at one position"
                )
            group.append(k)
            pairs += n
        if group:
            groups.append(group)
        for group in groups:
            rows = np.zeros(SCORE_ROWS, np.int32)
            rows[: len(group)] = request.positions[group] - start
            table = np.full((SCORE_PAIRS, 2), -1, np.int32)
            spans, p = [], 0
            for r, k in enumerate(group):
                targets = np.asarray(request.targets[k], np.int32)
                table[p : p + len(targets)] = np.stack(
                    [np.full(len(targets), r, np.int32), targets], axis=1
                )
                spans.append((p, p + len(targets)))
                p += len(targets)
            top = int(request.tops[group].max())
            K = min(t for t in SCORE_TOPS if t >= top)
            dev.h2d_nb(self.buf["score_rows"], rows)
            dev.h2d_nb(self.buf["score_pairs"], table)
            self._launch("score", f"k{K}")
            lp = dev.d2h(self.buf["score_lp"], (SCORE_PAIRS,), np.float32)
            if K:
                lse = dev.d2h(self.buf["score_stats"], (SCORE_ROWS, 16), np.float32)[
                    :, 1
                ]
                tv = dev.d2h(self.buf["score_top_v"], (SCORE_ROWS, K), np.float32)
                ti = dev.d2h(self.buf["score_top_i"], (SCORE_ROWS, K), np.int32)
            for r, k in enumerate(group):
                a, b = spans[r]
                out.targets[k] = lp[a:b].copy()
                n = int(request.tops[k])
                out.top_ids[k] = ti[r, :n].copy() if n else np.zeros(0, np.int32)
                out.top_logprobs[k] = (
                    (tv[r, :n] - lse[r]).astype(np.float32)
                    if n
                    else np.zeros(0, np.float32)
                )

    def decode(
        self,
        rows: Sequence[RowCache],
        tokens: Sequence[int] | None,
        sampling: Sampling | np.ndarray | None = None,
        *,
        stage: Staging | None = None,
        scores: bool = False,
    ) -> np.ndarray | None:
        """One token of each row; the next tokens, sampled with ``sampling`` (None: greedy).
        ``tokens`` None reads them from the device: the previous decode step's samples, row
        for row. With ``stage``, launch only: the tokens (and with ``scores`` the logprobs)
        are copied into ``stage`` and its event recorded."""
        if isinstance(sampling, np.ndarray):
            sampling = Sampling(sampling)
        B = len(rows)
        bucket = min(b for b in self.decode_buckets if b >= B)
        mode = self.modes[f"d{bucket}"]
        ids = np.zeros(bucket, np.int64)
        if tokens is not None:
            ids[:B] = tokens
        params = np.zeros((bucket, PARAMS), np.float32)
        if sampling is not None:
            params[:B] = sampling.params[:B]
        source = None if tokens is not None else self.buf["tok_out"]
        self._target(mode, rows, ids, B, params, token_source=source)
        if self.mtp:
            # the MTP layer's KV at each row's position: its sampled next token and the target's
            # hidden; a plain step samples no drafts, so a row that speculates again verifies zeros
            self._mtp_pass(
                mode,
                "mtp",
                rows,
                [row.pos0 for row in rows],
                1,
                np.arange(bucket) + 1,
                self.buf["tok_out"],
                self.mtp_buf["mtp_zero"],
                None,
            )
        if self.drafter is not None:
            self.drafter.inject(
                mode.name,
                [
                    (b, row.pos0, 1, row.draft_row(row.pos0))
                    for b, row in enumerate(rows)
                ],
            )
        if stage is None:
            return self.device.d2h(self.buf["tok_out"], (B,), np.int32)
        dev = self.device
        dev.d2h_async(stage.tokens[:B], self.buf["tok_out"])
        if scores:
            dev.d2h_async(stage.lp[:B], self.buf["lp_out"])
            dev.d2h_async(stage.top_idx[:B], self.buf["top_idx"])
            dev.d2h_async(stage.top_lp[:B], self.buf["top_lp"])
        stage.record()
        return None

    def verify(
        self,
        rows: Sequence[RowCache],
        anchors: Sequence[int],
        drafts: np.ndarray,
        sampling: Sampling | None = None,
    ) -> Verified:
        """One speculative step of each row: verify [anchor, drafts] (anchor at row.pos0),
        accept, commit, and draft the next step."""
        return self.verify_finish(self.verify_launch(rows, anchors, drafts, sampling))

    def verify_launch(
        self,
        rows: Sequence[RowCache],
        anchors: Sequence[int] | Callable[[], np.ndarray],
        drafts: np.ndarray | Callable[[], np.ndarray],
        sampling: Sampling | None = None,
        *,
        stage: Staging | None = None,
        scores: bool = False,
        chained: bool = False,
        width: int | None = None,
    ) -> Launched:
        """Launch the verify pass of [anchor, drafts] for each row (anchor at row.pos0): the
        target over K + 1 columns per row, then the MTP layer at every column. ``anchors`` and
        ``drafts`` may be callables, read once the step's tables are queued. ``chained``: the
        same rows continue the last staged ``verify_finish``, whose continuation tokens are on
        the device; ``drafts`` is then a callable for their host copy, read at the finish.
        With ``stage``, the samples (and with ``scores`` their logprobs) are copied into it
        without waiting. ``width``: columns per row (one of ``verify_widths``; the first
        ``width - 1`` drafts are verified); by default the width the last step chose
        (``next_width``), which a chained step must take: its tokens were drafted for it."""
        if chained:  # the width the drafter wrote the tokens for
            Tq = self.K + 1 if self.drafter is None else self.drafter.next_width
        else:
            Tq = width or self.force_width or self.next_width
        if Tq not in self.verify_widths:
            raise ValueError(f"no verify step of {Tq} columns ({self.verify_widths})")
        B = len(rows)
        bucket = min(b for b in self.decode_buckets if b >= B)
        mode = self.modes[verify_mode(bucket, Tq, self.K + 1)]
        resolved: list[np.ndarray] = []

        def tokens() -> np.ndarray:
            out = np.zeros((bucket, Tq), np.int64)
            out[:B, 0] = anchors() if callable(anchors) else anchors
            got = drafts() if callable(drafts) else np.asarray(drafts)
            resolved.append(np.asarray(got)[:, : Tq - 1])
            out[:B, 1:] = resolved[0]
            return out.reshape(-1)

        params = np.zeros((bucket, Tq, PARAMS), np.float32)
        if sampling is not None:  # column t samples position pos0 + t + 1
            params[:B] = sampling.grid(B, range(Tq))
        if chained:
            self._target(
                mode,
                rows,
                None,
                B,
                params.reshape(-1, PARAMS),
                token_source=self.next_tokens,
            )
            resolved.append(drafts)
        else:
            self._target(mode, rows, tokens, B, params.reshape(-1, PARAMS))
        if self.mtp:
            # the MTP layer at every verify column: token y_t, the target's hidden at the column
            cols = np.arange(bucket * Tq) + 1
            mtp_params = np.zeros((bucket, Tq, PARAMS), np.float32)
            if sampling is not None:  # column t drafts position pos0 + t + 2
                mtp_params[:B] = sampling.grid(B, range(1, Tq + 1))
            self._mtp_pass(
                mode,
                "mtp",
                rows,
                [row.pos0 for row in rows],
                Tq,
                cols,
                self.buf["tok_out"],
                self.mtp_buf["mtp_zero"],
                mtp_params.reshape(-1, PARAMS),
            )
        else:  # every column into the drafter's cache: the next steps rewrite the rejected ones
            assert self.drafter is not None
            self.drafter.inject(
                mode.name,
                [
                    (b * Tq, row.pos0, Tq, row.draft_row(row.pos0))
                    for b, row in enumerate(rows)
                ],
            )
        device_accept = self.drafter is not None and self.device_accept
        w = 0
        if device_accept:  # accept on the device, the commit and block step behind it
            w = self._fix(mode, rows, chained)
        if stage is not None:
            n = bucket * Tq
            d2h = self.device.d2h_async
            d2h(stage.tokens[:n], self.buf["tok_out"])
            if device_accept:
                d2h(stage.accepted[:bucket], self.spec_buf["fix_acc"])
            if self.mtp:
                d2h(stage.mtp_tokens[:n], self.mtp_buf["mtp_tok_out"])
            if scores:
                d2h(stage.lp[:n], self.buf["lp_out"])
                d2h(stage.top_idx[:n], self.buf["top_idx"])
                d2h(stage.top_lp[:n], self.buf["top_lp"])
            stage.record()
        if device_accept:
            self._launch("commit", mode.name)
            assert self.drafter is not None
            self.drafter.run(bucket, self._fix_launches, w)
            self.next_width = w
            if (
                stage is not None
            ):  # read at the continuation's finish (stage.next, event 1)
                stage.next_width = w
                self.device.d2h_async(
                    stage.next.reshape(-1)[: bucket * w], self.drafter.next
                )
                stage.record(1)
        return Launched(rows, resolved[0], sampling, stage, mode, device_accept, w)

    def _fix(self, mode: _Mode, rows: Sequence[RowCache], chained: bool) -> int:
        """Upload ``spec_fix``'s tables for a verify step and launch it: the commit's state
        fields for the accepted count's two possible STATE blocks, and the next block step's
        tables over every anchor position the count allows (``Drafter.prepare_device``).
        Returns the next block step's width (from the rows' last acceptance)."""
        drafter = self.drafter
        assert drafter is not None
        Tq, B = mode.per_row, len(rows)
        bucket = mode.columns // Tq
        L = len(self.gdn)
        last = self._last_accepted if len(self._last_accepted) == B else None
        w = self.force_width or choose_width(
            self.verify_widths,
            last if last is not None else np.full(B, Tq - 1),
            self._last_width if last is not None else Tq,
            bucket,
        )
        first = [row.state_at(row.pos0) for row in rows]
        final = [row.state_at(row.pos0 + Tq - 1) for row in rows]
        srcs = [row.state_src for row in rows]
        cb = []
        for dsts in (first, final):
            fields = np.full((L, bucket, 4), -1, np.int32)
            fields[:, :B] = self._state_fields(srcs, dsts)
            cb.append(fields)
        table = np.zeros((bucket, 8), np.int32)
        for b, row in enumerate(rows):
            switch = next(
                (m for m in range(Tq) if row.state_at(row.pos0 + m) != first[b]), Tq
            )
            table[b, :5] = (
                row.pos0,
                1,
                switch,
                self.geometry.hidden_row(first[b]),
                self.geometry.hidden_row(final[b]),
            )
        self._upload(self.spec_buf["fix_cb0"], cb[0])
        self._upload(self.spec_buf["fix_cb1"], cb[1])
        self._upload(self.spec_buf["fix_rows"], table)
        # padding rows commit nothing
        self._upload(
            self.spec_buf["commit_bases"], np.full((L, bucket, 4), -1, np.int32)
        )
        self._upload(self.spec_buf["commit_counts"], np.zeros((L, bucket), np.int32))
        self._upload(self.buf["hidden_table"], np.full((bucket, 2), -1, np.int32))
        self._fix_launches, patch = drafter.prepare_device(
            [row.draft_row(row.pos0 + Tq) for row in rows],
            [row.pos0 + 1 for row in rows],
            bucket,
            w,
        )
        if len(patch) > self.fix_patch:
            raise ValueError(f"{len(patch)} slot patches exceed {self.fix_patch}")
        full = np.zeros((self.fix_patch, 4), np.int32)
        full[: len(patch)] = patch
        self._upload(self.spec_buf["fix_patch"], full)
        meta = np.array(
            [len(patch), drafter.block_cfg(w).block, drafter.cfg.window, 0], np.int32
        )
        self._upload(self.spec_buf["fix_meta"], meta)
        tin = self.next_tokens if chained else self.buf["tok"]
        self._launch("spec_fix", mode.name, extra={"fix_tin": tin})
        return w

    def verify_finish(self, launched: Launched) -> Verified:
        """Accept each row's drafts against the verify pass's samples, commit the accepted
        tokens and launch the next step's draft chain. With the launch's stage this waits only
        for the verify pass; drafts 2 .. K land in ``stage.drafts`` (event 1), and the next
        step's tokens on the device (for a ``chained`` launch)."""
        rows, drafts, sampling, stage, mode = (
            launched.rows,
            launched.drafts,
            launched.sampling,
            launched.stage,
            launched.mode,
        )
        Tq, B = mode.per_row, len(rows)
        bucket = mode.columns // Tq
        if callable(drafts):
            drafts = drafts()
        drafts = np.asarray(drafts)[:, : Tq - 1]
        if stage is None:
            y = self.device.d2h(self.buf["tok_out"], (bucket, Tq), np.int32)[:B]
        else:
            stage.wait()
            y = stage.tokens[: bucket * Tq].reshape(bucket, Tq)[:B].copy()
        acceptance = self.speculation.accept(drafts.tolist(), y.tolist())
        accepted = np.asarray(acceptance.accepted_counts, np.int64)
        emitted = acceptance.token_rows
        if launched.device_accept:  # committed and drafted on the device already
            if stage is None:
                on_device = self.device.d2h(
                    self.spec_buf["fix_acc"], (bucket,), np.int32
                )
            else:
                on_device = stage.accepted[:bucket]
            if not np.array_equal(on_device[:B], accepted):
                raise RuntimeError(
                    f"device acceptance {on_device[:B].tolist()} != host {accepted.tolist()}"
                )
            self._last_accepted, self._last_width = accepted, Tq
            w = launched.next_width
            next_drafts = np.zeros((B, self.K), np.int64)
            if stage is None:
                assert self.drafter is not None
                next_drafts[:, : w - 1] = self.drafter.read_next(bucket, w)[:B, 1:]
            columns = [b * Tq + np.arange(accepted[b] + 1) for b in range(B)]
            return Verified(emitted, accepted, next_drafts, columns, Tq)
        self._commit(mode, rows, accepted)
        anchor_pos = [rows[b].pos0 + int(accepted[b]) + 1 for b in range(B)]
        if self.mtp:
            if stage is None:
                q = self.device.d2h(
                    self.mtp_buf["mtp_tok_out"], (bucket, Tq), np.int32
                )[:B]
            else:
                q = stage.mtp_tokens[: bucket * Tq].reshape(bucket, Tq)[:B].copy()
            first = q[np.arange(B), accepted]
            first_cols = np.arange(B) * Tq + accepted + 1
            scatter = None
            if stage is not None:
                scatter = self.spec_buf["next_tok"]
                following = np.zeros((bucket, Tq), np.int32)
                following[:B, 0] = [row[-1] for row in emitted]
                following[:B, 1] = first
                self._upload(scatter, following)
            next_drafts = self._chain(
                rows, anchor_pos, first, first_cols, sampling, accepted, stage, scatter
            )
        else:  # the block step after each row's new anchor: the next step's tokens
            drafter = self.drafter
            assert drafter is not None
            # the next step's width, which the block step writes its tokens for
            self.next_width = w = self.force_width or choose_width(
                self.verify_widths, accepted, Tq, bucket
            )
            drafter.block(
                [rows[b].draft_row(anchor_pos[b]) for b in range(B)],
                [row[-1] for row in emitted],
                bucket,
                width=w,
            )
            next_drafts = np.zeros((B, self.K), np.int64)
            if stage is None:
                next_drafts[:, : w - 1] = drafter.read_next(bucket, w)[:B, 1:]
            else:  # read at the continuation's finish (stage.next, event 1)
                stage.next_width = w
                self.device.d2h_async(
                    stage.next.reshape(-1)[: bucket * w], drafter.next
                )
        if stage is not None:
            stage.record(1)
        columns = [b * Tq + np.arange(accepted[b] + 1) for b in range(B)]
        return Verified(emitted, accepted, next_drafts, columns, Tq)

    @property
    def next_tokens(self) -> int:
        """The device buffer holding a chained verify step's tokens [anchor, drafts] per row."""
        if self.drafter is not None:
            return self.drafter.next
        return self.spec_buf["next_tok"]

    @property
    def host_drafts(self) -> int:
        """How many of a staged verify step's next drafts the host knows at its finish (the
        rest land in the staging: MTP's chain in ``drafts``, the block step in ``next``)."""
        return 1 if self.mtp else 0

    def _commit(
        self, mode: _Mode, rows: Sequence[RowCache], accepted: np.ndarray
    ) -> None:
        """Apply each row's first accepted + 1 tokens' GDN updates and store the hidden of its
        last one, into the STATE block of that position."""
        L, B = len(self.gdn), mode.columns // mode.per_row
        dsts = [
            row.state_at(row.pos0 + int(m))
            for row, m in zip(rows, accepted, strict=True)
        ]
        fields = self._state_fields(
            [row.state_src for row in rows], dsts
        )  # (L, rows, 4)
        bases = np.full((L, B, 4), -1, np.int32)
        bases[:, : len(rows)] = fields
        counts = np.zeros((L, B), np.int32)
        counts[:, : len(rows)] = accepted + 1
        self._upload(self.spec_buf["commit_bases"], bases)
        self._upload(self.spec_buf["commit_counts"], counts)
        table = np.full((B, 2), -1, np.int32)
        for b, (m, dst) in enumerate(zip(accepted, dsts, strict=True)):
            table[b] = (b * mode.per_row + int(m), self.geometry.hidden_row(dst))
        self._upload(self.buf["hidden_table"], table)
        self._launch("commit", mode.name)

    def scores(self, rows: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The last step's raw logprobs per column: the chosen token's (rows,), and the top
        list's ids and logprobs (rows, TOP_MAX), filled to each column's ``ntop``."""
        d2h = self.device.d2h
        return (
            d2h(self.buf["lp_out"], (rows,), np.float32),
            d2h(self.buf["top_idx"], (rows, TOP_MAX), np.int32),
            d2h(self.buf["top_lp"], (rows, TOP_MAX), np.float32),
        )

    def logits(self, rows: int = 1) -> np.ndarray:
        """(rows, vocab) logits of the last step (a prefill's last token, or each column)."""
        V = self.dims.vocab
        if rows == 1:
            return self.device.d2h(self.buf["logits"], (V,), np.float32)
        return self.device.d2h(self.buf["logits"], (rows, V), np.float32)

    def close(self) -> None:
        self.loader.close()


class SingleSequence:
    """One sequence over a range of a model's slabs (tests and tools): KV blocks are taken in
    order from the first slab up and STATE blocks from the last slab down as the sequence grows,
    one STATE block per 4096 positions, with the Engine's rule (read the block of pos0 - 1,
    write the block of the step's last position). With MTP, MTP KV blocks come after the KV
    blocks' slabs, and speculative steps carry their drafts."""

    def __init__(self, model: Qwen35Gaudi, slabs: range | None = None) -> None:
        self.model = model
        self.slabs = slabs or range(1, model.parent_capacity + 1)
        # with MTP: its KV blocks in a fixed slab below at most three STATE blocks
        self.mtp_slab = self.slabs[-4] if model.mtp else None
        # with the DFlash drafter: a ring of DRAFT blocks covering its window, in the slabs
        # below the STATE blocks (and the MTP slab)
        self.ring: np.ndarray | None = None
        if model.dfl_cfg is not None:
            g, cfg = model.geometry, model.dfl_cfg
            R = cfg.window // 128 + 2
            per = g.draft_blocks_per_slab
            top = self.slabs[-4]
            slabs = np.arange(R) // per
            self.ring = g.draft_block_row(top - slabs, np.arange(R) % per)
            self.ring_slabs = int(slabs.max()) + 1
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.state_blocks: dict[int, int] = {}
        self.drafts: np.ndarray | None = None

    def _state(self, logical: int) -> int:
        if logical not in self.state_blocks:
            slab = self.slabs[-1 - len(self.state_blocks)]
            self.state_blocks[logical] = self.model.geometry.state_block_row(slab, 0)
        return self.state_blocks[logical]

    def _row(self, n: int, lookahead: int = 0) -> RowCache:
        g, last = self.model.geometry, self.pos + n - 1
        reach = last + lookahead
        blocks = np.arange(reach // 128 + 1)
        src = -1 if self.pos == 0 else self._state((self.pos - 1) // STATE_TOKENS)
        dst = self._state(last // STATE_TOKENS)
        first = (self.pos - 1) // STATE_TOKENS if self.pos else 0
        states = tuple(self._state(k) for k in range(first, reach // STATE_TOKENS + 1))
        per_slab = g.kv_blocks_per_slab
        kv_slabs = self.slabs[0] + blocks // per_slab
        mtp_rows = None
        top = self.slabs[-1] - len(self.state_blocks)
        if self.mtp_slab is not None:
            mtp_rows = g.mtp_block_row(np.full(len(blocks), self.mtp_slab), blocks)
            top = min(top, self.mtp_slab - 1)
            if len(self.state_blocks) > 3:
                raise MemoryError("sequence exceeds its slabs")
        draft_rows, draft_first = None, 0
        if self.ring is not None:
            R = len(self.ring)
            draft_first = max(0, int(blocks[-1]) + 1 - R)
            logical = np.arange(draft_first, int(blocks[-1]) + 1)
            draft_rows = self.ring[logical % R]
            top = min(top, self.slabs[-4] - self.ring_slabs)
            if len(self.state_blocks) > 3:
                raise MemoryError("sequence exceeds its slabs")
        if kv_slabs.max() > top:
            raise MemoryError("sequence exceeds its slabs")
        rows = g.kv_block_row(kv_slabs, blocks % per_slab)
        return RowCache(
            self.pos, rows, src, dst, mtp_rows, first, states, draft_rows, draft_first
        )

    def prefill(self, ids: Sequence[int], sampling=None) -> int:
        done = self.model.prefill(self._row(len(ids)), ids, sampling)
        self.pos += len(ids)
        self.drafts = done.drafts
        return done.token

    def decode(self, token: int) -> int:
        out = int(self.model.decode([self.next_row()], [token])[0])
        self.advance()
        return out

    def verify(self, token: int, sampling: Sampling | None = None) -> list[int]:
        """One speculative step from ``token`` (the last emitted): the emitted tokens."""
        K = self.model.K
        drafts = self.drafts if self.drafts is not None else np.zeros(K, np.int64)
        row = self._row(1, lookahead=2 * K)
        done = self.model.verify([row], [token], drafts[None], sampling)
        self.pos += int(done.accepted[0]) + 1
        self.drafts = done.drafts[0]
        return done.tokens[0]

    def next_row(self) -> RowCache:
        """The cache of this sequence's next decode token (``advance`` after the step)."""
        return self._row(1)

    def advance(self, n: int = 1) -> None:
        self.pos += n

    def logits(self) -> np.ndarray:
        return self.model.logits()
