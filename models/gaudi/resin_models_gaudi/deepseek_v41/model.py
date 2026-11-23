"""DeepSeek-V4.1-Flash forward steps on one Gaudi card: one tensor-parallel rank of ``nranks``.

A step runs T columns: one sequence's prefill chunk, or one decode token for each of up to MAX_T sequences. The
host fills the step's tables and uploads them in one copy, then launches the embedding, every layer, and the
head. A layer is an Engram lookup (layers in ``engram_layer_ids``), the attention launches, and the MoE. Each
sub-layer's partial is summed over the ranks between launches, as are the Engram rows; the logits slices are
all-gathered.

Attention stores the step's keys before it reads them:

- every layer stores each column's key in its WINDOW page and attends over the 128 positions up to the
  column's own, in slots of ``query_block`` columns (decode: one column per slot);
- a KV owner (ratio r) also compresses: each group of r tokens that the step completes becomes one entry of
  the owner's HISTORY pages, with its index key. A ratio-2 group whose first token ran in an earlier step reads
  that token from the pending state, and a step that ends inside a group leaves its token there;
- an index source scores the owner's index keys for every column (launches of up to 2048 keys per slot) and
  lists the column's top 512 entries; a compressed layer attends over its window and the listed entries of its
  owner in one softmax with the attention sink.

Engram hash ids are computed on the host from each row's last tokens (``Row.before``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from resin.models.deepseek_v41 import DeepSeekV41Config
from resin_models_gaudi.common.sampling import PARAMS, TOP_MAX
from resin_models_gaudi.deepseek_v41 import layers as L
from resin_models_gaudi.deepseek_v41.cache import (
    HISTORY_TOKENS,
    INDEX_ROWS,
    PAGE_ROWS,
    PAGE_TOKENS,
    PENDING_SLOTS,
    STATE_FLOATS,
    CacheGeometry,
)
from resin_models_gaudi.deepseek_v41.rope import cs_table, layer_frequencies
from resin_models_gaudi.deepseek_v41.weights import (
    RankSlices,
    engram_hash_tables,
    engram_table,
    prepare_compressed,
    prepare_draft_stage,
    prepare_engram,
    prepare_layer,
    reader,
    table_rows,
)
from resin_models_gaudi.glm5_3_flash.layers import sample_graph
from resin_models_gaudi.glm5_3_flash.model import (
    _Op,
)
from resin_models_gaudi.glm5_3_flash.weights import NS
from resin_models_gaudi.loading.convert import kperm
from resin_models_gaudi.loading.progress import info, progress, weight_file_bytes
from resin_models_gaudi.loading.upload import WeightLoader
from resin_models_gaudi.synapse.graph import Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import Device

DECODE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
PREFILL_BUCKETS = (32, 128, 512, 1024, 2048)
PAGED_T = 128  # prefill launches this wide decode each routed expert once per page of its rows
MAX_T = DECODE_BUCKETS[-1]
# steps of up to GATHER_T columns all-gather the ranks' partials and sum them in the next graph (one collective
# that moves T x nranks columns); wider steps all-reduce them
GATHER_T = 16
TMAX = PREFILL_BUCKETS[-1]
QUERY_BLOCK = 128  # prefill columns per window slot
INDEX_KEYS = 2048  # index keys per index slot
PREFILL_IX_SLOTS = 16
PREFILL_IX_PARTS = 4  # a prefill index launch's slots in parts (index_score_graph)
# a prefill step whose every column sees more than PREFILTER_MIN entries picks each column's top 512 from the
# entries of its PREFILTER_BLOCKS blocks of 8 with the largest maxima (index_select_graph)
PREFILTER_BLOCKS = (
    576  # 512, the newest block (ix_bmax keeps it), in whole groups of 64
)
PREFILTER_MIN = 10240
# a prefill step whose columns see at most DENSE_KEYS compressed entries attends over every visible entry of each
# query block, with the lists as masks (attention_out_graph's dense part), instead of gathering each column's list
DENSE_KEYS = 2048
DSPARK_ROWS = (
    2  # rows per speculative step: (block + 1) columns each fit the decode buckets
)
RADIX_MIN = 20480  # decode steps whose columns see more index keys pick their top 512 by radix passes (ix_thr counts up to 20480 keys)
J = 24  # Engram rows per column: (max n-gram - 1) x heads
# the registers of a decode step's native program: sparse, cand, candidate launches, and per index ratio
# (1, 2) the radix selection and the index launches
_R_SPARSE, _R_CAND, _R_IXC = 0, 1, 2
_R_RADIX = {1: 3, 2: 4}
_R_IX = {1: 5, 2: 6}
_R_COLUMN = 7  # the column whose logits a one-column head gathers
# per index ratio, a prefill step's dense attention and its prefiltered selection
_R_DENSE = {1: 8, 2: 9}
_R_PF = {1: 10, 2: 11}
_REGISTERS = 12
# Bytes kept out of the arena for the step tables and the recipes' workspaces. A recipe compiles with its own
# workspace on the card and keeps it: after the 281 graphs of this model compile, 6.3 GiB of a card's free memory
# is gone (measured: 20.2 GiB free at attach, 13.9 GiB left after the compile), and it stays gone for the process.
# So the reserve has to cover that 6.3 GiB as well as the step's own scratch; at 5 GiB a cold recipe cache sizes
# the arena into memory the compile is holding and the arena allocation fails. 7 GiB leaves ~1.3 GiB of slack.
WORKSPACE_RESERVE = 7 << 30
# Diagnostic: RESIN_DS41_WINDOW_ONLY=1 makes every layer a window-only layer -- no compressed KV, no index
# chain, no compressor -- which splits "the answer is garbage" into "the compressed path" vs "everything
# else" in one boot.  It is a probe, never a serving configuration.
WINDOW_ONLY = bool(os.environ.get("RESIN_DS41_WINDOW_ONLY"))
# the sampler's outputs for up to MAX_T columns: tokens, logprobs, top ids, top logprobs (256-byte aligned)
_SAMPLED = {
    "tok_out": 0,
    "lp_out": 4 * MAX_T,
    "top_idx": 8 * MAX_T,
    "top_lp": 8 * MAX_T + 4 * TOP_MAX * MAX_T,
}
_SAMPLED_BYTES = 8 * MAX_T + 8 * TOP_MAX * MAX_T
_SAMPLE_TABLE = 4 * PARAMS * MAX_T + 256  # the rows' parameters, then ``last``
_SAMPLE_LAST = 4 * PARAMS * MAX_T


def bucket(n: int, prefill: bool) -> int:
    for b in PREFILL_BUCKETS if prefill else DECODE_BUCKETS:
        if n <= b:
            return b
    raise ValueError(
        f"{n} columns exceed the {'prefill' if prefill else 'decode'} buckets"
    )


def _native_rows(rows) -> tuple[np.ndarray, ...]:
    """``Ds41Fill``'s row arguments: per row its column, pos0, real tokens and first window block, then the
    WINDOW rows and ids, the HISTORY rows and the tokens before pos0 of every row, flat with their offsets."""

    def flat(parts):
        lens = [len(p) for p in parts]
        indptr = np.zeros(len(parts) + 1, np.int64)
        np.cumsum(lens, out=indptr[1:])
        values = (
            np.concatenate([np.asarray(p, np.int64) for p in parts])
            if indptr[-1]
            else np.zeros(0, np.int64)
        )
        return indptr, values

    window_indptr, window = flat([r.window for r in rows])
    _, window_ids = flat([r.window_ids for r in rows])
    history_indptr, history = flat([r.history for r in rows])
    before_indptr, before = flat([r.before for r in rows])
    return (
        np.array([r.column for r in rows], np.int64),
        np.array([r.pos0 for r in rows], np.int64),
        np.array([r.real for r in rows], np.int64),
        np.array([r.window_first for r in rows], np.int64),
        window_indptr, window, window_ids, history_indptr, history, before_indptr, before,
    )  # fmt: skip


@dataclass(frozen=True, slots=True)
class Shape:
    """A step's graph shape: T columns, prefill (one row) or decode (one column per row)."""

    T: int
    prefill: bool

    @property
    def qb(self) -> int:
        """columns per window and index slot"""
        return min(self.T, QUERY_BLOCK) if self.prefill else 1

    @property
    def win_slots(self) -> int:
        return self.T // self.qb

    @property
    def win_keys(self) -> int:
        """a slot's window keys: up to qb + 127 positions from a page start"""
        return 384 if self.prefill else 256

    @property
    def blocks(self) -> int:
        """query blocks of the step: one per row when decoding, ``qb`` columns each when prefilling"""
        return self.T // self.qb

    @property
    def ix_slots(self) -> int:
        """index slots per launch (decode: fewer launches for wider steps); at most eight blocks' worth, so a
        launch covers whole tiles of a merge group and the group stays inside ix_thr's vector window"""
        slots = PREFILL_IX_SLOTS if self.prefill else 32 if self.T >= 4 else 16
        return min(slots, 8 * self.blocks)

    @property
    def groups2(self) -> int:
        """ratio-2 groups a step can complete"""
        return self.T // 2 + 1 if self.prefill else self.T


@dataclass(frozen=True, slots=True)
class Row:
    """One sequence of a step: its first column, first position, real tokens, and its cache blocks: WINDOW block
    rows and ids from logical block ``window_first``, HISTORY block rows from block 0; ``before`` holds the
    sequence's token ids at the (Engram max n-gram - 1) positions before pos0 (those that exist) and ``tokens`` the step's."""

    column: int
    pos0: int
    real: int
    window: np.ndarray
    window_ids: np.ndarray
    window_first: int
    history: np.ndarray
    before: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def window_row(self, position):
        return self.window[np.asarray(position) // PAGE_TOKENS - self.window_first]

    def window_id(self, position):
        return self.window_ids[np.asarray(position) // PAGE_TOKENS - self.window_first]


class _Tables:
    """Byte offsets of a step's tables (one shape) in the device table buffer, and views of each in the pinned
    host copies (``v[k]``; ``rows`` and ``lat`` stack every layer's table)."""

    def __init__(self, shape: Shape, model: DeepSeekV41Gaudi) -> None:
        T, geo, i32, f32 = shape.T, model.geo, np.int32, np.float32
        self.shape = shape
        self.off: dict[str, int] = {}
        self.spec: dict[str, tuple] = {}
        self.nbytes = 0

        def add(name: str, dtype, *dims: int) -> None:
            self.off[name], self.spec[name] = self.nbytes, (dtype, dims)
            self.nbytes += -(-4 * int(np.prod(dims)) // 256) * 256

        add("tok", i32, T)
        add("meta", i32, 4)
        for k in ("cs0", "cs0_inv", "cs1", "cs1_inv"):
            add(k, f32, T, 128)
        add("cs_g2", f32, shape.groups2, 128)
        add("cs_g1", f32, T, 128)
        for o in geo.owners:
            G = shape.groups2 if model.cfg.compress_ratios[o] > 1 else T
            add(f"cmp_store.{o}", i32, G, 8)
        for o in geo.pending_owners:
            add(f"cmp_groups.{o}", i32, shape.groups2, 4)
            add(f"cmp_rows.{o}", i32, T)
        add("slots", i32, shape.win_slots, 8)
        layers = model.n_layers + model.stages  # (DSpark stages: their window pages)
        for layer in range(layers):
            add(f"rows.{layer}", i32, T, 8)
        for layer in range(layers):
            add(f"lat.{layer}", i32, shape.win_slots, shape.win_keys // PAGE_TOKENS)
        if (
            model.stages and not shape.prefill
        ):  # a draft step's block keys, their slots, the anchors
            add("dsp_bidx", i32, model.block * T)
            add("dsp_lslots", i32, T, 8)
            add("dsp_anchor", i32, 16)
        if model.dense(
            shape
        ):  # the dense compressed attention's slots and each owner's latent pages
            for r in (1, 2):
                add(f"cslots{r}", i32, shape.win_slots, 8)
            for o in geo.owners:
                add(f"cmp_lat.{o}", i32, shape.win_slots, DENSE_KEYS // PAGE_TOKENS)
        add("ix_visb", i32, T)
        add("ix_visb2", i32, T)
        add("ix_visc", i32, T)
        for j in range(model.ixc_launches(shape)):
            add(f"ixc.{j}", i32, shape.ix_slots, 8)
        # (T, 2): each query block's first position and the entries its last column sees (the graph tensor is
        # (2, T): lane 0 the position, lane 1 the count). Everything the tiled index chain needs per *group* is
        # derived from this and ix_vis on the device (ds_ix_gslot, ix_gvis, ds_ix_bmax, ix_cmask, ds_ix_merge),
        # so the fill's index work is O(T) rather than groups x T.
        add("blk1", i32, T, 2)
        add("blk2", i32, T, 2)
        for r in (1, 2):
            add(f"ix_vis{r}", i32, T)
            add(f"cmp_blocks{r}", i32, T, model.pages)
            # every merge group's slot records (lane 7: the tile's first index page) and the groups' first keys
            add(f"ix{r}.rec", i32, shape.ix_slots * model.ix_launches(shape, r), 8)
            add(f"ix{r}.gb", i32, model.ix_groups(shape, r))
        for layer in model.engram_layers:
            if model.engram == "host":
                add(f"eg.{layer}", f32, T, 256 * J)
            elif model.hashed(shape):  # the older tokens' XORs, as (hi, lo) per lane
                add(f"eg_xh.{layer}", i32, T, 64)
                add(f"eg_xl.{layer}", i32, T, 64)
            else:
                add(f"eg.{layer}", i32, T, J)
        self.layers = layers
        self.v: list[dict[str, np.ndarray]] = []
        # byte ranges _fill zeroes: all but the tables it writes whole
        # tables written whole, so the fill does not zero them: the compressed-page tables, and the index
        # chain's per-group records and group bases (ds_ix_gslot writes every group of both on the device)
        whole = sorted(
            (self.off[n], self.off[n] + 4 * int(np.prod(self.spec[n][1])))
            for n in (
                "cmp_blocks1", "cmp_blocks2",
                "ix1.rec", "ix2.rec", "ix1.gb", "ix2.gb",
            )
        )  # fmt: skip
        self.zero, a = [], 0
        for lo, hi in whole:
            self.zero.append((a, lo))
            a = hi
        self.zero.append((a, self.nbytes))

    def bind(self, host: np.ndarray) -> None:
        """Add views of every table in ``host`` (a host copy, at least ``nbytes``)."""
        v: dict[str, np.ndarray] = {}
        self.v.append(v)
        for name, (dtype, dims) in self.spec.items():
            a = self.off[name]
            v[name] = host[a : a + 4 * int(np.prod(dims))].view(dtype).reshape(dims)
        for kind in ("rows", "lat"):
            first, (_, dims) = self.off[f"{kind}.0"], self.spec[f"{kind}.0"]
            stride = (
                (self.off[f"{kind}.1"] - first) // 4
                if self.layers > 1
                else int(np.prod(dims))
            )
            block = (
                host[first : first + 4 * stride * self.layers]
                .view(np.int32)
                .reshape(self.layers, stride)
            )
            v[kind] = block[:, : int(np.prod(dims))].reshape(self.layers, *dims)


class _IxMergeOp(_Op):
    """A merge launch, preceded by the copies that place the group's own scores at their rows of the merge buffer.

    The score launches write a (rows, T) tile, whose column stride is ``rows``; the merge buffer's is
    ``run + rows`` and its rows 0 .. run - 1 are the running candidates. So the tile goes in per column, at rows
    run .. (rows 0 .. for the first group): T copies of ``rows`` floats, one batched copy on the launch stream,
    ordered before the merge without a submission of its own."""

    __slots__ = ("copy",)

    def __init__(self, recipe, names, copy) -> None:
        super().__init__(recipe, names)
        self.copy = tuple(np.asarray(a, np.uint64) for a in copy)


class _Program:
    """Every launch of one step shape, bound once."""

    def __init__(self) -> None:
        self.embed: _Op | None = None
        # the tiled index chain's group tables, one launch per ratio, run once per step before the layers
        self.ixg: list[_Op] = []
        self.layers: list[dict] = []
        self.head: _Op | None = None
        self.inject: _Op | None = None
        # bounded replay (prefill shapes): the cut owner's store-only launch and the buffers of its inputs
        self.store: _Op | None = None
        self.cut: tuple[int, int, int] | None = None  # (X, c, p)


class DeepSeekV41Gaudi:
    def __init__(
        self,
        device: Device,
        checkpoint: str | Path,
        cfg: DeepSeekV41Config,
        *,
        capacity: int | Callable[[int], int],
        max_seq_len: int,
        rank: int = 0,
        nranks: int = 1,
        layers: int | None = None,
        recipes: RecipeCache | None = None,
        engram: str = "card",
        engram_prewarm: bool = True,
        gather: bool = True,
        weights: dict[str, int] | None = None,
        barrier: Callable[[], None] | None = None,
        log: Callable[[str], None] = info,
        dspark: bool = False,
        replay: int = 0,
        defer_cache: bool = False,
    ) -> None:
        """``defer_cache``: stop after the weights and the buffers; :meth:`attach` then makes the cache arena,
        the step tables and the recipes, which depend on its capacity.

        ``replay``: bounded replay's kept span (0: off). A prefill chunk then runs the layers after the cut
        (the first ratio-1 layer, the KV owner of every layer after it) only on its columns among the prompt's last
        ``replay`` positions; see ``launch_replay``."""
        if engram not in ("card", "host"):
            raise ValueError(f"engram {engram!r}: 'card' or 'host'")
        self.device, self.cfg, self.log = device, cfg, log
        self.rank, self.nranks = rank, nranks
        self.n_layers = cfg.num_hidden_layers if layers is None else layers
        self.engram = engram
        self.engram_prewarm = engram_prewarm
        self.checkpoint = Path(checkpoint)
        self.sl = RankSlices.of(cfg, rank, nranks)
        # DSpark: its stages after the full backbone, drafting blocks of ``block`` tokens
        self.stages = (
            cfg.num_nextn_predict_layers
            if dspark
            and cfg.dspark_block_size
            and self.n_layers == cfg.num_hidden_layers
            else 0
        )
        self.block = cfg.dspark_block_size
        self.geo = CacheGeometry.from_config(
            cfg, layers=self.n_layers, max_write=TMAX, draft_layers=self.stages
        )
        # narrow decode steps all-gather the ranks' partials and sum them in the next graph
        self.d = replace(
            L.Dims.of(cfg, nranks, self.sl.vocab.stop - self.sl.vocab.start),
            gather_tokens=GATHER_T if nranks > 1 and gather else 0,
        )
        self.d_draft = replace(
            self.d,
            experts=cfg.dspark_n_routed_experts,
            top=cfg.dspark_num_experts_per_tok,
        )
        self.max_seq_len = max_seq_len
        self.recipes = recipes or RecipeCache(device)
        self.src = reader(checkpoint)
        self.engram_layers = tuple(e for e in cfg.engram_layer_ids if e < self.n_layers)
        self.replay = replay
        self.cut = self._replay_cut() if replay else None
        self.freqs = {r: layer_frequencies(cfg, r) for r in (0, 1)}
        # index score rows: every compressed entry a ratio-1 owner can hold, in whole 4096-key groups
        self.kmax = -(-(max_seq_len + TMAX) // 4096) * 4096
        # every position's rotation table (cs_table), read by prefill steps
        self._cs_all = {
            r: cs_table(np.arange(self.kmax), self.freqs[r]) for r in (0, 1)
        }
        self.pages = self.kmax // PAGE_TOKENS  # latent pages per column (ratio 1)
        # candidate blocks: block maxima rows (whole 4096-row groups), and the entries of a column's blocks
        self.kb = -(-self.kmax // 8 // 4096) * 4096
        self.kc = (
            cfg.candidate_topk_blocks * 8 if cfg.candidate_source_layer >= 0 else 0
        )
        self._hash = None
        self._allocs: list[int] = []
        self.loader: WeightLoader | None = None  # None: ``weights`` given
        t0 = time.time()
        self.w = weights if weights is not None else self._load(barrier)
        log(f"ds41 rank {rank}/{nranks}: weights in {time.time() - t0:.0f} s")
        self.eg_vocab = 0
        self.eg_host = {}  # host-Engram: layer -> (weight (rows,256) u8, scale (rows,8) u8) mmap views
        self._eg_host_maps = []  # holds the mappings + views alive for the model's lifetime
        if self.engram_layers and engram == "card":
            self._hash_tables()
            for layer in (
                self.engram_layers
            ):  # this rank's rows of wkv: offsets into the whole weight
                key = f"l{layer}.eg_wkv"
                if key in self.w and f"{key}_r" not in self.w:
                    rows = self.eg_rows_r
                    self.w[f"{key}_r"] = self.w[key] + rank * rows * 6144
                    self.w[f"{key}_r.s"] = self.w[f"{key}.s"] + 4 * rank * rows
        if self.engram_layers and engram == "host":
            self._load_host_tables()
        self._buffers()
        self.capacity: int | None = None
        if not defer_cache:
            self.attach(capacity)

    def attach(
        self,
        capacity: int | Callable[[int], int],
        *,
        slab_rows: int | None = None,
        arena: int | None = None,
    ) -> None:
        """The cache arena of ``capacity`` slabs (a callable gets the free card memory less the workspace
        reserve), its pending state, the step tables and the recipes: every part that depends on the capacity.
        ``slab_rows``: slabs of that many rows (another model's slab size), each with as many blocks as fit.
        ``arena``: the address of a cleared arena of ``capacity`` such slabs plus the null slab, which other
        models of the card share (the caller allocates and frees it); else the model allocates its own."""
        if self.capacity is not None:
            raise RuntimeError("the cache arena is attached already")
        if slab_rows is not None:
            self.geo = self.geo.with_slab_rows(slab_rows)
        device, log = self.device, self.log
        free = device.memory_info()[0]
        if callable(capacity):
            capacity = capacity(free - self.workspace_reserve)
        self.capacity = capacity
        self.arena_blocks = self.geo.arena_blocks(capacity)
        # the pending state's size and rows are graph shapes too; only its memory is allocated below
        state = self.geo.state_bytes(capacity)
        self.state_rows = state // (4 * STATE_FLOATS)
        # The arena's block count is a graph shape, so the capacity is fixed here -- but the arena itself is
        # allocated *after* the recipes compile. A recipe compiles with its own workspace on the card (the
        # largest is 6.6 GiB), and with the arena already resident a cold recipe cache has nowhere to put it:
        # ``synDeviceMalloc ... status=26``. Compiling first gives every compile the card's free memory, and
        # the arena is then sized from what is actually left.
        self.tables = {s: _Tables(s, self) for s in self.shapes}
        size = max(t.nbytes for t in self.tables.values())
        self.table = self._malloc(size)
        self._layer_rows = np.array(
            [self.geo.layer_offset(k) for k in range(self.n_layers + self.stages)],
            np.int64,
        )
        # the native executor (deepseek_v41.native) fills every step's tables and runs its programs
        self._native = self._native_fill()
        self._glue = device.glue
        if self._glue is None:
            raise RuntimeError(
                "DeepSeek-V4.1 steps run in the native executor: it needs the runtime library's glue (Device.glue)"
            )
        self._native_programs: dict[Shape, object] = {}
        self._programs: dict[Shape, _Program] = {}
        self._draft_programs: dict[Shape, dict] = {}
        self._draft_natives: dict[Shape, object] = {}  # the draft programs as native programs (_draft_native)
        self._sample_ops: dict[tuple, _Op] = {}
        # device event slots: the sampler's staging copies, and the pinned tables' uploads
        self._stage_event, self._host_event = device.event_slots(2), device.event_slots(2)
        self._compile()
        # the arena and its pending state, from the memory the compiles left
        arena_bytes = self.arena_blocks * 1024 * 512
        left = device.memory_info()[0]
        if arena is None and left < arena_bytes + state:
            raise RuntimeError(
                f"ds41: the cache arena of {capacity} slabs needs {arena_bytes / 2**30:.1f} GiB and its "
                f"state {state / 2**30:.1f} GiB, but {left / 2**30:.1f} GiB is left after the recipes "
                f"compiled ({free / 2**30:.1f} GiB free at attach): lower the cache or warm the recipe cache"
            )
        if arena is None:
            self.arena = self._malloc(arena_bytes)
            device.memset32(self.arena, 0, self.arena_blocks * 1024 * 128)
        else:
            self.arena = arena
        self.state = self._malloc(state)
        device.memset32(self.state, 0, state // 4)
        log(
            f"ds41: cache {capacity} slabs, {self.arena_blocks / 1024 * 0.5:.1f} GiB per card "
            f"({free / 2**30:.1f} GiB free at attach, {left / 2**30:.1f} GiB after the recipes compiled)"
        )

    def _replay_cut(self) -> int:
        """The first ratio-1 layer, when every later layer reads its compressed KV and index keys: the layers
        after it keep nothing of a prompt's earlier positions but their windows."""
        cfg, n = self.cfg, self.n_layers
        ratios = cfg.compress_ratios[:n]
        cut = next((i for i, r in enumerate(ratios) if r == 1), None)
        if cut is None or cut not in cfg.kv_source_layers:
            raise ValueError("bounded replay needs a ratio-1 KV owner")
        if any(ratios[i] != 1 or cfg.kv_owner(i) != cut for i in range(cut, n)):
            raise ValueError(
                f"bounded replay: a layer after {cut} keeps its own compressed KV"
            )
        if any(e >= cut for e in self.engram_layers):
            raise ValueError(f"bounded replay: an Engram layer after {cut}")
        if self.stages and any(t < cut for t in cfg.dspark_target_layer_ids):
            raise ValueError(f"bounded replay: a DSpark target layer before {cut}")
        if not cfg.sliding_window <= self.replay <= TMAX:
            raise ValueError(
                f"bounded replay keeps {self.replay} positions: {cfg.sliding_window} to {TMAX}"
            )
        return cut

    @property
    def workspace_reserve(self) -> int:
        """Bytes kept out of the arena for the step tables and the shared workspace. The indexer's scores are
        tiled (index_merge_graph: one group's running candidates and its own keys), so the only term that grows
        with the context is the published block maxima (kb, TMAX) as f32, which the candidate blocks and the
        layers that read them share."""
        pairs = max((L.INDEX_RUN + self.ix_rows(s)) * s.T for s in self.shapes)
        decode = [s for s in self.shapes if not s.prefill]
        bpairs = max((L.INDEX_BLOCKS + self.ix_tp(s) * L.BLOCK_TILE) * s.T for s in decode)
        return max(
            WORKSPACE_RESERVE,
            4 * TMAX * self.kb + 8 * pairs + 8 * L.INDEX_RUN * TMAX + 4 * bpairs + (1 << 30),
        )

    def _malloc(self, nbytes: int) -> int:
        address = self.device.malloc(nbytes)
        self._allocs.append(address)
        return address

    def release(self) -> None:
        """Free the cache and buffers (not the weights)."""
        for address in self._allocs:
            self.device.free(address)
        self._allocs = []

    @property
    def shapes(self) -> list[Shape]:
        return [Shape(t, False) for t in DECODE_BUCKETS] + [
            Shape(t, True) for t in PREFILL_BUCKETS
        ]

    def ixc_launches(self, shape: Shape) -> int:
        """index launches over candidate entries a decode step of ``shape`` may need (none for prefill)"""
        if shape.prefill or not self.kc:
            return 0
        return -(-shape.T * (self.kc // INDEX_KEYS) // shape.ix_slots)

    def dense(self, s: Shape) -> bool:
        """whether prefill steps of ``s`` have the dense compressed attention (query blocks of 128)"""
        return bool(DENSE_KEYS) and s.prefill and s.qb == L.QUERY_BLOCK_DENSE

    def _visb(self, ratio: int) -> str:
        """the table of padded block counts for ``ratio``: ix_visb is the candidate source's ratio"""
        cs = self.cfg.candidate_source_layer
        cand = cs < 0 or ratio == self.cfg.compress_ratios[cs]
        return "ix_visb" if cand else "ix_visb2"

    def _ix_names(
        self, s: Shape, ratio: int, j: int, owner: int, head: dict, scratch: bool = False
    ) -> dict[str, int]:
        """an index launch's tables: its slots and pages, or each part's rows of them. Its scores always go to the
        tile buffer (a (rows, T) tensor, which is what a score launch writes); the merge op's copy places
        copies each column's rows to their own rows of the group's merge buffer, and a candidate layer's mask
        writes the group's rows there in place."""
        base = (
            self.table
            + self.tables[s].off[f"ix{ratio}.rec"]
            + 4 * 8 * s.ix_slots * j
        )
        pages = self.table + self.tables[s].off[f"cmp_blocks{ratio}"]
        tile = self.ix_tile
        recipe = self.r["ix", s]
        off = {"cmp_blocks": pages, "cmp_loff": self.loff[owner]}
        if "ix_slots" in recipe.persistent:
            return {**head, **off, "ix_slots": base, "ix_pages": pages, "ix_score": tile}
        names = dict(head, **off, ix_score=tile)
        P = sum(n.startswith("ix_slots.p") for n in recipe.persistent)
        Sp = s.ix_slots // P
        for p in range(P):
            names[f"ix_slots.p{p}"] = base + 4 * 8 * Sp * p
            names[f"ix_pages.p{p}"] = pages
            names[f"ix_sink.p{p}"] = self.io["ix_sink"] + 256 * p
        return names

    def ix_tp(self, shape: Shape) -> int:
        """tiles (of ``INDEX_KEYS`` index keys) one merge group covers: a whole number of launches, at most eight,
        and at most 4 M score floats (the group's merge buffer stays small and its threshold inside ix_thr's
        vector window)"""
        q = max(1, shape.ix_slots // max(1, shape.blocks))
        tp = q
        while tp * 2 <= 8 and (tp * 2) * INDEX_KEYS * shape.T <= 4 << 20:
            tp *= 2
        return tp

    def ix_rows(self, shape: Shape) -> int:
        """the score rows of one merge group (its tiles)"""
        return self.ix_tp(shape) * INDEX_KEYS

    def ix_parts(self, shape: Shape) -> int:
        """Parts of a prefill index launch. Each part's slots table is bound at its own address (``_ix_names``),
        and a recipe's section base must be 128-byte aligned: a part's table is (8, ix_slots / P) i32, i.e.
        32 (ix_slots / P) bytes, so P is the largest of PREFILL_IX_PARTS, 2, 1 that leaves a multiple of four
        slots per part (8 slots per launch with 4 parts is 64 bytes and fails the launch: status=30, "Section is
        not aligned")."""
        if not shape.prefill:
            return 1
        for p in (PREFILL_IX_PARTS, 2, 1):
            if shape.ix_slots % p == 0 and (shape.ix_slots // p) % 4 == 0:
                return p
        return 1

    def ix_lpg(self, shape: Shape) -> int:
        """index launches per merge group"""
        per_tile = max(1, -(-shape.blocks // shape.ix_slots))
        return max(1, self.ix_tp(shape) * per_tile // max(1, shape.ix_slots // max(1, shape.blocks)))

    def ix_groups(self, shape: Shape, ratio: int) -> int:
        """merge groups a step of ``shape`` may need for ratio-``ratio`` keys"""
        chunks = -(-self.kmax // ratio // INDEX_KEYS)
        return -(-chunks // self.ix_tp(shape))

    def ix_launches(self, shape: Shape, ratio: int) -> int:
        """index launches a step of ``shape`` may need for ratio-``ratio`` keys"""
        return self.ix_groups(shape, ratio) * self.ix_lpg(shape)

    def _ix_group_names(
        self, s: Shape, ratio: int, g: int, head: dict, kind: str
    ) -> dict:
        """A merge group's tables: kind "m" the group's merge (its visible rows, the column's visible entries,
        the group's first key, the running positions and the buffers it reads and writes), "b" its block maxima
        (at its slice of the published array) with its pin, or "k" its candidate mask."""
        tb, T, run = self.tables[s], s.T, L.INDEX_RUN
        first = g * self.ix_tp(s) * L.BLOCK_TILE
        names = dict(head)
        # the group's first key and the column's visible entries: the kernels derive the group's visible rows,
        # counts and pin from these (the fill writes neither)
        base = {
            "ix_vis": self.table + tb.off[f"ix_vis{ratio}"],
            "ix_gbase": self.table + tb.off[f"ix{ratio}.gb"] + 4 * g,
        }
        if kind == "b":
            names.update(
                base,
                ix_tile=self.ix_tile,
                ix_bmax=self.io["cand_bmax"] + 4 * T * first,
            )
        elif kind == "k":  # the mask reads the score launches' tile and writes the group's rows in place
            names.update(
                base,
                ix_tile=self.ix_tile,
                ix_tile_out=self.ix_tile,
                ix_bmax=self.io["cand_bmax"] + 4 * T * first,
                cand_thr=self.io["cand_thr"],
            )
        else:
            names.update(
                base,
                ix_merge=self.ix_merge[g % 2],
                ix_pos=self.ix_pos[g % 2],
                ix_pos_out=self.ix_pos[1 - g % 2],
                ix_thr=self.io["ix_thr"] if g == 0 else self.io["ix_thr2"],
                ix_lslots=self.io["ix_lslots"],
                ix_list=self.io["ix_list"],
                ix_sout=self.ix_merge[1 - g % 2],
            )
        return names

    def _ix_ops(
        self, s: Shape, ratio: int, layer: int, owner: int, head: dict, tail: dict, t
    ) -> dict:
        """The tiled index chain's launches of one index source layer: the plain arm (each group's score launches
        and its merge), the candidate arm (with the group's block maxima, or its candidate mask, and the tail that
        publishes the blocks), and the sparse reindex path's launches and selection."""
        cfg = self.cfg
        groups, per = self.ix_groups(s, ratio), self.ix_lpg(s)
        cs = cfg.candidate_source_layer
        cand = "" if cs < 0 or layer < cs else "source" if layer == cs else "user"
        plain: list = []
        cnd: list = []
        for g in range(groups):
            acc = [
                self._op(
                    self.r["ix", s],
                    layer,
                    self._ix_names(
                        s, ratio, g * per + l, owner, head, scratch=cand == "user"
                    ),
                )
                for l in range(per)
            ]
            plain.extend(acc)
            cnd.extend(acc)
            if not s.prefill:
                kind = "k" if cand == "user" else "b"
                cnd.append(
                    self._op(
                        self.r[f"ix{kind}", s],
                        layer,
                        self._ix_group_names(s, ratio, g, head, kind),
                    )
                )
            merge = self._op(
                self.r["ixm", s],
                layer,
                self._ix_group_names(s, ratio, g, head, "m"),
                copy=self._ix_tile_copy(s, g),
            )
            plain.append(merge)
            cnd.append(merge)
        ops: dict = {"ixp": plain, "ixc2": cnd}
        if not s.prefill:
            sel = {
                "ix_visb": self.table + self.tables[s].off["ix_visb"],
                "cand_bmax": self.io["cand_bmax"],
                "cand_thr": self.io["cand_thr"],
                "cand_list": self.io["cand_list"],
                "cand_slots": self.io["cand_slots"],
            }
            ops["ixc_tail"] = (
                [self._op(self.r["ixbl", s], layer, sel)] if cand == "source" else []
            )
            if cand == "user" and self.ixc_launches(s):
                ops["ixsp"] = [
                    self._op(
                        self.r["ixc", s],
                        layer,
                        {
                            **head,
                            **tail,
                            "ix_slots": t(f"ixc.{j}"),
                            "ix_score": self.io["ix_score_sparse"],
                            "ix_cpos": self.io["ix_cpos"],
                            # the candidate-entry gather reads the step's page table and the owner's offset
                            # (tail carries cmp_blocks already; cmp_loff is the sparse launch's own)
                            "cmp_loff": self.loff[owner],
                        },
                    )
                    for j in range(self.ixc_launches(s))
                ]
                ops["ixsp_tail"] = [
                    self._op(
                        self.r["sel", s, "user", "sparse"],
                        layer,
                        {
                            "ix_score": self.io["ix_score_sparse"],
                            "ix_cpos": self.io["ix_cpos"],
                            "ix_vis": t(f"ix_vis{ratio}"),
                            "ix_visc": t("ix_visc"),
                        },
                    )
                ]
        return ops

    # ------------------------------------------------------------------ weights
    def _load(self, barrier) -> dict[str, int]:
        cfg, src, sl = self.cfg, self.src, self.sl
        self.loader = loader = WeightLoader(self.device, self.recipes)
        files = weight_file_bytes(self.checkpoint, "*.safetensors")
        self.device.begin_load(files // self.nranks)

        def put(prefix: str, parts: dict[str, np.ndarray]) -> None:
            loader.load((prefix + name, a, "plain") for name, a in parts.items())

        put(
            "",
            {
                "emb": src.read("embed.weight"),
                "head": kperm(src.read("head.weight", sl.vocab)),
                "norm": (src.read("norm.weight").astype(np.uint32) << 16).view(
                    np.float32
                ),
            },
        )
        for layer in range(self.n_layers):
            if barrier is not None:
                barrier()  # the ranks read each layer's shards together
            t0 = time.time()
            progress.note(f"layer {layer + 1} of {self.n_layers}")
            parts = prepare_layer(src, cfg, layer, sl)
            if cfg.compress_ratios[layer]:
                parts.update(prepare_compressed(src, cfg, layer))
            if layer in self.engram_layers:
                parts.update(prepare_engram(src, layer))
            put(f"l{layer}.", parts)
            del parts
            if layer in self.engram_layers and self.engram == "card":
                self._load_table(loader, layer)
            self.log(f"layer {layer} in {time.time() - t0:.0f} s")
        for k in range(self.stages):
            t0 = time.time()
            put(f"l{self.n_layers + k}.", prepare_draft_stage(src, cfg, k, sl))
            self.log(f"DSpark stage {k} in {time.time() - t0:.0f} s")
        self.device.end_load()
        self._draft_names(loader.w)
        return loader.w

    def _draft_names(self, w: dict[str, int]) -> None:
        """The DSpark tensors the inject and Markov graphs read by plain names (their stage's copies)."""
        if not self.stages:
            return
        first, last = self.n_layers, self.n_layers + self.stages - 1
        for name in ("main_proj", "main_proj.s", "main_norm"):
            w[name] = w[f"l{first}.{name}"]
        for name in ("mk_embed", "mk_head"):
            w[name] = w[f"l{last}.{name}"]
        for k in range(self.stages):
            for name in ("wkv", "wkv.s", "kv_norm"):
                w[name.replace("wkv", f"wkv{k}").replace("kv_norm", f"kv_norm{k}")] = w[
                    f"l{self.n_layers + k}.{name}"
                ]

    @property
    def eg_rows_r(self) -> int:
        """rows of the Engram wkv product each rank computes in a sharded decode step"""
        return (L.NS + 1) * self.d.hidden // self.d.ranks

    def eg_sharded(self, s: Shape) -> bool:
        """whether a step of shape ``s`` computes the Engram wkv product in rank slices and all-gathers it"""
        return self.engram == "card" and self.d.ranks > 1 and self.d.gathered(s.T)

    def hashed(self, s: Shape) -> bool:
        """whether a step of shape ``s`` hashes its Engram rows on the device (decode, tables on the cards)"""
        return self.engram == "card" and not s.prefill

    def _engram_hash(self):
        if self._hash is None:
            from tokenizers import Tokenizer

            from resin.models.deepseek_v41.engram import EngramHash

            self._hash = EngramHash.build(
                self.cfg, Tokenizer.from_file(str(self.checkpoint / "tokenizer.json"))
            )
        return self._hash

    def _hash_tables(self) -> None:
        """Each Engram layer's ``eg_hash`` tables (``eg_a0``, ``eg_hp``), uploaded once."""
        h = self._engram_hash()
        self.eg_vocab = len(h.token_map)
        for layer in self.engram_layers:
            if f"l{layer}.eg_a0" in self.w:
                continue
            for name, a in engram_hash_tables(
                h, self.cfg.engram_layer_ids.index(layer)
            ).items():
                self.w[f"l{layer}.{name}"] = address = self.device.malloc(
                    max(256, a.nbytes)
                )
                self.device.h2d(address, np.ascontiguousarray(a))

    def _load_table(self, loader: WeightLoader, layer: int) -> None:
        """This rank's rows of an Engram table, converted in chunks."""
        dev, w = self.device, loader.w
        total = self.src.info(f"layers.{layer}.engram.embed.weight")[1][0]
        rows = table_rows(total, self.rank, self.nranks)
        n = rows.stop - rows.start
        w[f"l{layer}.eg_table"] = table = loader.malloc(256 * n)
        w[f"l{layer}.eg_rs"] = scales = loader.malloc(4 * n)
        w[f"l{layer}.eg_row0"] = rows.start
        w[f"l{layer}.eg_rows"] = n
        for first, data, rs in engram_table(self.src, layer, rows):
            dev.h2d(table + 256 * first, data)
            dev.h2d(scales + 4 * first, rs)

    # ------------------------------------------------------------------ host-Engram tables
    def _load_host_tables(self) -> None:
        """Host-Engram: hold the lookup tables in host RAM as a file-backed MAP_SHARED mmap of the
        checkpoint shards (one physical copy shared by the ranks through the page cache), in FILE
        format (256 OCP E4M3 codes + 8 E8M0 scale bytes per row). Prewarm them resident, check
        residency with mincore and log it (report if low — never silently fall back), reject e8m0
        0xFF scales at load, and keep numpy views of each layer's weight (rows,256) u8 and scale
        (rows,8) u8. The Rust fill gathers and dequantizes these per step; nothing is uploaded here."""
        import ctypes
        import json
        import mmap
        import os
        import struct

        from resin_models_gaudi.deepseek_v41.weights import (
            _reject_nan_engram_scales,
            reader,
        )

        src = reader(self.checkpoint)
        shards = {
            layer: src.weight_map[f"layers.{layer}.engram.embed.weight"]
            for layer in self.engram_layers
        }
        # prewarm each distinct shard once (a sequential read pulls it into the page cache)
        if self.engram_prewarm:
            for shard in sorted(set(shards.values())):
                self._prewarm_shard(self.checkpoint / shard)
        page = os.sysconf("SC_PAGE_SIZE")
        libc = ctypes.CDLL(None, use_errno=True)
        for layer, shard in shards.items():
            path = self.checkpoint / shard
            fd = os.open(path, os.O_RDONLY)
            try:
                header_bytes = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                header = json.loads(os.pread(fd, header_bytes, 8))
                base = 8 + header_bytes
                name = f"layers.{layer}.engram.embed"
                # Python's mmap defaults to MAP_SHARED: the ranks share one physical copy
                mapping = mmap.mmap(fd, path.stat().st_size, prot=mmap.PROT_READ)
            finally:
                os.close(fd)
            raw = np.frombuffer(mapping, np.uint8)
            woff = base + header[f"{name}.weight"]["data_offsets"][0]
            soff = base + header[f"{name}.scale"]["data_offsets"][0]
            rows = header[f"{name}.weight"]["shape"][0]
            weight = raw[woff : woff + rows * 256].reshape(rows, 256)
            scale = raw[soff : soff + rows * 8].reshape(rows, 8)
            _reject_nan_engram_scales(scale)
            # mincore over the whole mapping (raw starts at the page-aligned mmap base)
            pages = (raw.nbytes + page - 1) // page
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(raw.ctypes.data), ctypes.c_size_t(raw.nbytes), vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {shard}")
            resident = sum(1 for v in vec if v & 1) / pages
            self.log(
                f"ds41 host-Engram layer {layer}: {rows} rows from {shard}, {resident:.1%} resident"
            )
            if resident < 0.95:
                self.log(
                    f"WARNING: host-Engram layer {layer} ({shard}) is only {resident:.1%} resident "
                    "after prewarm; the per-step gather will fault pages (no fallback is taken)"
                )
            self.eg_host[layer] = (weight, scale)
            # hold the mapping and its views alive for the model's lifetime (the views export buffers)
            self._eg_host_maps.append((mapping, raw, weight, scale))

    @staticmethod
    def _prewarm_shard(path) -> None:
        """Read a shard once, sequentially, into the page cache so the mmap gather is RAM-resident."""
        import os

        fd = os.open(path, os.O_RDONLY)
        try:
            view = memoryview(bytearray(1 << 30))
            offset, size = 0, path.stat().st_size
            while offset < size:
                read = os.preadv(fd, [view], offset)
                if read <= 0:
                    break
                offset += read
        finally:
            os.close(fd)

    # ------------------------------------------------------------------ buffers and graphs
    def _buffers(self) -> None:
        d, dev, T = self.d, self.device, TMAX
        H, D, nh = d.hidden, d.head_dim, d.heads

        def buf(nbytes: int) -> int:
            return self._malloc(max(256, -(-nbytes // 256) * 256))

        def const(array: np.ndarray) -> int:
            a = np.ascontiguousarray(array)
            address = buf(a.nbytes)
            dev.h2d(address, a)
            return address

        f = np.float32
        self.pzero = const(np.zeros((max(TMAX, GATHER_T * self.nranks), H), f))

        def identity(t: int) -> np.ndarray:
            """(24, t) in numpy: pre one-hot(0), post 0, comb I"""
            c = np.zeros((24, t), f)
            c[0] = 1.0
            c[8 + np.arange(4) * 5] = 1.0
            return c

        self.c_identity = {
            t: const(identity(t)) for t in (*DECODE_BUCKETS, *PREFILL_BUCKETS)
        }
        self.loff = {
            o: const(np.array([self.geo.owner_offset(o), -1] + [0] * 62, np.int32))
            for o in self.geo.owners
        }
        self.x = [buf(4 * 4 * H * T) for _ in range(3)]
        self.cbuf = [buf(4 * 24 * T), buf(4 * 24 * T)]
        self.pbuf = [buf(4 * H * T), buf(4 * H * T)]
        self.pall = [
            buf(4 * H * GATHER_T * self.nranks) for _ in range(2)
        ]  # the partials all-gathered
        W = nh * T
        self.io = {
            "q_all": buf(2 * D * W),
            "o1": buf(4 * D * W),
            "m1": buf(4 * W),
            "l1": buf(4 * W),
            "ix_q8": buf(L.INDEX_DIM * L.INDEX_HEADS * T),
            "ix_qs": buf(4 * L.INDEX_HEADS * T),
            "ix_w": buf(4 * L.INDEX_HEADS * T),
            "ix_thr2": buf(4 * 64 * T),
            "ix_list": buf(4 * L.INDEX_TOP * T),
            "ix_lslots": buf(4 * 8 * T),
            "cand_bmax": buf(4 * self.kb * T),
            "ix_score_sparse": buf(4 * max(1, self.kc) * MAX_T),
            "ix_cpos": buf(4 * max(1, self.kc) * MAX_T),
            "cand_thr": buf(4 * 64 * T),
            "cand_list": buf(4 * max(1, self.cfg.candidate_topk_blocks) * T),
            "cand_slots": buf(4 * 8 * T),
            "ix_cpos": buf(4 * max(1, self.kc) * T),
            "cmp_sink": buf(256),
            "cmp_stored": buf(256),
            "ix_sink": buf(256 * 8),  # (a part's sink each, prefill index launches)
            "ix_thr": buf(4 * 64 * T),
            "eg_emb": buf(4 * 256 * J * T),
            # decode: each rank's rows of the Engram wkv product, and all of them gathered
            "eg_kv": buf(4 * self.eg_rows_r * MAX_T),
            "eg_kv_all": buf(4 * self.eg_rows_r * MAX_T * d.ranks),
        }
        if self.stages:  # DSpark: the target layers' stream means, the draft tokens, the inject stores' sinks
            for k in range(len(self.cfg.dspark_target_layer_ids)):
                self.io[f"dsp_main{k}"] = buf(4 * H * TMAX)
            self.io["dsp_out"] = buf(4 * self.block * MAX_T)
            for k in range(self.stages):
                self.io[f"dsp_sink{k}"] = buf(256)
        # the tiled indexer: a merge group's running candidates and its own keys, alternating between the group
        # that reads and the group that writes, and the running candidates' positions
        pairs = max((L.INDEX_RUN + self.ix_rows(s)) * s.T for s in self.shapes)
        self.ix_merge = [buf(4 * pairs), buf(4 * pairs)]
        self.ix_pos = [buf(4 * L.INDEX_RUN * T) for _ in range(2)]
        self.ix_tile = buf(4 * max(self.ix_rows(s) * s.T for s in self.shapes))
        self.logits = buf(4 * d.vocab * TMAX)
        self.logits_all = (
            buf(4 * d.vocab * TMAX * self.nranks) if self.nranks > 1 else self.logits
        )
        self.sampled_buf = buf(_SAMPLED_BYTES)
        self.sampled = {n: self.sampled_buf + off for n, off in _SAMPLED.items()}
        self.sample_table = buf(_SAMPLE_TABLE)
        if self.cut is not None:
            # bounded replay: the kept columns' streams, partial and coefficients pass through here
            self.replay_scratch = buf(4 * (NS * H + H + 24) * bucket(self.replay, True))

    def _graphs(self) -> dict[tuple, object]:
        dev, d, cfg, nb = self.device, self.d, self.cfg, self.arena_blocks
        graphs: dict[tuple, object] = {}
        R = self.state_rows
        for s in self.shapes:
            T, qb = s.T, s.qb
            graphs["embed", s] = L.embed_graph(dev, d, T, cfg.vocab_size)
            graphs["ffn", s] = L.ffn_graph(dev, d, T, paged=s.prefill and T >= PAGED_T)
            graphs["head", s] = L.head_graph(dev, d, T)
            if self.cut is not None and s.prefill:
                graphs["store", s] = L.owner_store_graph(
                    dev, d, T, nb, ratio=1, groups=T, state_rows=R
                )
            if self.stages:
                taps = len(cfg.dspark_target_layer_ids)
                graphs["inject", s] = L.inject_graph(dev, d, T, self.stages, nb, taps)
                rows = [
                    b
                    for b in range(1, DSPARK_ROWS + 1)
                    if not s.prefill and bucket(b * self.block, False) == T
                ]
                if rows:  # a draft step's shape
                    graphs["dsp_attn", s] = L.draft_attention_graph(
                        dev, d, T, s.win_slots, s.win_keys, nb, self.block
                    )
                    graphs["dsp_ffn", s] = L.ffn_graph(dev, self.d_draft, T)
                    for b in rows:
                        graphs["dsp_markov", s, b] = L.markov_graph(
                            dev, d, T, b, self.block, self.nranks, cfg.vocab_size
                        )
            for layer in range(self.n_layers):
                key = self._attention_key(layer, s)
                if key in graphs:
                    continue
                kind, ratio, owner, source, residual, capture = (
                    key[0],
                    key[2],
                    key[3],
                    key[4],
                    key[5],
                    key[6],
                )
                if capture and kind == "win":
                    raise NotImplementedError(
                        "a DSpark target layer without compression"
                    )
                if kind == "win":
                    graphs[key] = L.attention_window_graph(
                        dev,
                        d,
                        T,
                        s.win_slots,
                        s.win_keys,
                        nb,
                        residual=residual,
                        query_block=qb,
                    )
                elif kind == "attn":
                    graphs[key] = L.attention_graph(
                        dev, d, T, s.win_slots, s.win_keys, nb, self.pages, ratio=ratio, residual=residual,
                        query_block=qb, capture=capture,
                    )  # fmt: skip
                    if self.dense(s):
                        graphs[key + ("dense",)] = L.attention_graph(
                            dev, d, T, s.win_slots, s.win_keys, nb, self.pages, ratio=ratio, residual=residual,
                            query_block=qb, capture=capture, dense=DENSE_KEYS, scores=self.ix_rows(s),
                        )  # fmt: skip
                else:
                    graphs[key] = L.attention_pre_graph(
                        dev, d, T, s.win_slots, s.win_keys, nb, ratio=ratio, owner=owner, source=source,
                        groups=s.groups2 if ratio > 1 else T, state_rows=R, residual=residual, query_block=qb,
                        capture=capture,
                    )  # fmt: skip
            if not WINDOW_ONLY and any(cfg.compress_ratios[i] for i in range(self.n_layers)):
                rows, top = self.ix_rows(s), cfg.candidate_topk_blocks
                graphs["ix", s] = L.index_score_graph(
                    dev, d, T, s.ix_slots, INDEX_KEYS, nb, rows, query_block=qb,
                    parts=self.ix_parts(s), cpages=self.pages,
                )  # fmt: skip
                for r in (1, 2):
                    graphs["ixg", s, r] = L.index_gslot_graph(
                        dev, d, T, s.ix_slots, self.ix_lpg(s), self.ix_tp(s),
                        s.blocks, s.qb, r, self.ix_groups(s, r), INDEX_KEYS,
                        L.INDEX_RUN, PAGE_TOKENS,
                    )  # fmt: skip
                graphs["ixm", s] = L.index_merge_graph(dev, d, T, rows, L.INDEX_RUN)
                if not s.prefill:  # decoding: the candidate blocks the source layer publishes
                    graphs["ixb", s] = L.index_blocks_graph(dev, d, T, rows)
                    graphs["ixk", s] = L.index_mask_graph(dev, d, T, rows)
                    graphs["ixbs", s] = L.index_block_select_graph(
                        dev, d, T, self.kb, top, False
                    )
                    graphs["ixbl", s] = L.index_block_select_graph(
                        dev, d, T, self.kb, top, True
                    )
                    if self.ixc_launches(s):
                        graphs["ixc", s] = L.index_score_graph(
                            dev, d, T, s.ix_slots, INDEX_KEYS, nb, self.kc, query_block=qb,
                            cand_top=top, pages=self.pages,
                        )  # fmt: skip
                        graphs["sel", s, "user", "sparse"] = L.index_select_graph(
                            dev, d, T, self.kc, "user", top, sparse=True
                        )
                graphs["out", s] = L.attention_out_graph(dev, d, T, nb, self.pages)
                if self.dense(s):
                    graphs["out", s, "dense"] = L.attention_out_graph(
                        dev, d, T, nb, self.pages, dense=DENSE_KEYS, scores=self.ix_rows(s)
                    )
            for layer in self.engram_layers:
                if self.eg_sharded(s):
                    graphs["eg", s] = L.engram_graph(dev, d, T, part="kv")
                    graphs["eg_gate", s] = L.engram_graph(dev, d, T, part="gate")
                else:
                    graphs["eg", s] = L.engram_graph(dev, d, T)
                if self.engram == "card":
                    rows, row0 = (
                        self.w[f"l{layer}.eg_rows"],
                        self.w[f"l{layer}.eg_row0"],
                    )
                    graphs["eg_gather", s, rows, row0] = L.engram_gather_graph(
                        dev, d, T, rows, row0, self.eg_vocab if self.hashed(s) else 0
                    )
            for R_ in (
                (1,) if s.prefill else (1, T)
            ):  # a prefill chunk samples its last column
                graphs["sample", s, R_] = sample_graph(dev, d, T, R_, self.nranks)  # type: ignore[arg-type]
        return graphs

    def _ratio(self, layer: int) -> int:
        """A layer's compress ratio; 0 for every layer under the window-only diagnostic."""
        return 0 if WINDOW_ONLY else self.cfg.compress_ratios[layer]

    def _attention_key(self, layer: int, s: Shape) -> tuple:
        cfg = self.cfg
        ratio = self._ratio(layer)
        residual = layer not in self.engram_layers
        capture = (
            bool(self.stages) and layer in cfg.dspark_target_layer_ids
        )  # (a DSpark input)
        if not ratio:
            return ("win", s, 0, False, False, residual, capture)
        owner = layer in cfg.kv_source_layers
        source = layer in cfg.index_source_layers
        if (
            not owner and not source
        ):  # one launch: no index launches between its two parts
            return ("attn", s, ratio, owner, source, residual, capture)
        return ("pre", s, ratio, owner, source, residual, capture)

    def _compile(self) -> None:
        progress.building_graphs()
        graphs = self._graphs()
        # Each compile thread gets its own device context, and a context grows to its recipe's workspace — the
        # largest here is 6.6 GiB (ds41_ffn_paged_T128, 0x1881d0c00) — so two at once already exceed a card's
        # free memory on a cold recipe cache (synDeviceMalloc ... status=26). One at a time fits; the disk cache
        # makes this a first-boot cost only.
        recipes = self.recipes.compile_many(list(graphs.values()), workers=1)
        self.r: dict[tuple, Recipe] = dict(zip(graphs.keys(), recipes, strict=True))
        for g in graphs.values():
            g.destroy()
        self.compiles_at_start = self.device.compile_count
        # constant inputs, one buffer per name: ones for rope_b* and one, zeros for the rest, each as large as
        # its largest use
        sizes: dict[str, int] = {}
        for recipe in self.r.values():
            for n, t in recipe.persistent.items():
                if n.startswith(("rope_b", "rope_s", "rope_lp")) or n in (
                    "one",
                    "dep",
                    "ix_none",
                    "rope_pages",
                ):
                    sizes[n] = max(sizes.get(n, 0), t.nbytes)
        self.consts = {}
        for n, nbytes in sizes.items():
            a = np.ones if n.startswith("rope_b") or n == "one" else np.zeros
            self.consts[n] = self._malloc(max(256, nbytes))
            self.device.h2d(self.consts[n], a(max(256, nbytes) // 4, np.float32))

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start

    # ------------------------------------------------------------------ diagnostics
    def dump_step(self, tag: str = "") -> None:
        """RESIN_DS41_DUMP: after a step, print what the residual stream and the head actually hold.

        The end-to-end symptom is one token repeated, which is what a hidden state dominated by the current
        token's embedding looks like.  These numbers separate that from a broken head/sampler without any
        reference: the two residual buffers' magnitudes, whether they are distinct (i.e. the layers are
        adding something), each column's cosine against the embedding rows, and the logits' shape.
        """
        d, dev = self.d, self.device
        H, nh = d.hidden, d.heads
        n = 4 * H * 4  # one column of a (4H, T) residual: 4 streams, f32
        out = {}
        # per-layer partials/residual of the last step (localize the first diverging layer)
        import os as _os
        _df = None
        if _os.environ.get("RESIN_DS41_DUMP_FILE"):
            _df = open(_os.environ["RESIN_DS41_DUMP_FILE"], "a")
        for s, prog in self._programs.items():
            if s.prefill and getattr(prog, "layers", None):
                print(f"[dump {tag}] per-layer column 0 (T={s.T}):")
                for i, ops in enumerate(prog.layers):
                    pa = dev.d2h(ops["p_attn"], (H,), np.float32)
                    pf = dev.d2h(ops["p_ffn"], (H,), np.float32)
                    xf = dev.d2h(ops["x_ffn"], (4 * H,), np.float32)
                    line = (f"[dump {tag}]   L{i}: |p_attn| {np.linalg.norm(pa):.4f} "
                            f"|p_ffn| {np.linalg.norm(pf):.4f} |x_ffn| {np.linalg.norm(xf):.4f} "
                            f"finite {np.isfinite(xf).all()}")
                    print(line)
                    if _df:
                        _df.write(line + "\n")
                break
        if _df:
            _df.close()
        for k in (0, 1):
            a = dev.d2h(self.x[k], (4 * H,), np.float32)
            out[f"x{k}"] = a
            print(f"[dump {tag}] x[{k}] column 0: norm {np.linalg.norm(a):.4f} "
                  f"absmax {np.abs(a).max():.4f} finite {np.isfinite(a).all()}")
        a, b = out["x0"], out["x1"]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"[dump {tag}] cos(x[0], x[1]) = {cos:.6f}  (1.0 means the layers changed nothing)")
        logits = dev.d2h(self.logits, (d.vocab,), np.float32)
        srt = np.sort(logits)[::-1]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        ent = float(-(p * np.log(p + 1e-30)).sum())
        print(f"[dump {tag}] logits: norm {np.linalg.norm(logits):.3f} top1 {srt[0]:.3f} "
              f"top2 {srt[1]:.3f} margin {srt[0] - srt[1]:.3f} entropy {ent:.3f} "
              f"top1 p {p.max():.4f} top5 ids {np.argsort(logits)[::-1][:5].tolist()}")
        # is the residual (or its mean over the 4 streams) an embedding row?
        emb = np.asarray(dev.d2h(self.w["emb"], (d.vocab, H), np.float32))
        h = a.reshape(4, H).mean(0)
        en = np.linalg.norm(emb, axis=1)
        num = emb @ h
        cosr = num / (en * np.linalg.norm(h) + 1e-30)
        best = int(np.argmax(cosr))
        print(f"[dump {tag}] |hidden| {np.linalg.norm(h):.4f} vs |emb| median {np.median(en):.4f}: "
              f"best cosine to any embedding row {cosr[best]:.6f} (row {best})")
        for k in (0, 1):
            hk = out[f"x{k}"].reshape(4, H).mean(0)
            c = emb @ hk / (en * np.linalg.norm(hk) + 1e-30)
            print(f"[dump {tag}] x[{k}] mean-stream: best embedding cosine {float(c.max()):.6f}")
        # (a) the four streams the head's collapse mixes: sane individually, or is one of them dead?
        for k in (0, 1):
            st = out[f"x{k}"].reshape(4, H)
            norms = [float(np.linalg.norm(st[i])) for i in range(4)]
            cs = [(float(st[i] @ st[j] / (norms[i] * norms[j] + 1e-30)), i, j)
                  for i in range(4) for j in range(i + 1, 4)]
            print(f"[dump {tag}] x[{k}] stream norms {['%.2f' % n for n in norms]} "
                  f"pairwise cos max {max(cs)[0]:.4f} min {min(cs)[0]:.4f}")
        # (b) the final norm's weight, and the residual's RMS (what the head's norm sees)
        nw = dev.d2h(self.w["norm"], (H,), np.float32)
        print(f"[dump {tag}] norm weight: mean {nw.mean():.5f} rms {np.sqrt((nw**2).mean()):.5f} "
              f"min {nw.min():.5f} max {nw.max():.5f}")
        for k in (0, 1):
            print(f"[dump {tag}] x[{k}] rms {np.sqrt((out[f'x{k}']**2).mean()):.4f} "
                  f"rms after norm {np.sqrt(((out[f'x{k}'] * np.tile(nw, 4))**2).mean()):.4f}")
        # (c) the top-1 GLOBAL token id (the head's output is this rank's vocabulary slice)
        start = self.sl.vocab.start
        top = np.argsort(logits)[::-1][:5]
        from tokenizers import Tokenizer as _Tk

        _tk = _Tk.from_file(str(self.checkpoint / "tokenizer.json"))
        print(f"[dump {tag}] top-5 global ids {[int(start + t) for t in top]} (slice start {start}) "
              f"as text {[_tk.decode([int(start + t)]) for t in top]}")
        # (e) the collapse coefficients the head actually uses, against c_identity
        for k in (0, 1):
            cc = dev.d2h(self.cbuf[k], (24,), np.float32)
            print(f"[dump {tag}] c[{k}] lanes 0..11 {np.round(cc[:12], 4).tolist()}")
        ident = np.zeros(24, np.float32)
        ident[0] = 1.0
        ident[8 + np.arange(4) * 5] = 1.0
        print(f"[dump {tag}] c_identity lanes 0..11 {np.round(ident[:12], 4).tolist()}")
        # (d) the head ALONE, on an embedding row: does it produce a sane token from a known input?
        s1 = Shape(1, False)
        prog1 = self._programs.get(s1)
        if prog1 is not None and prog1.head is not None:
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(str(self.checkpoint / "tokenizer.json"))
            xb = dev.malloc(max(256, 16 * H))
            ob = dev.malloc(max(256, 16 * H))  # the head's X_out, which it also writes
            lb = dev.malloc(4 * d.vocab)
            try:
                for word in ("1", " Paris", " the"):
                    ids = tk.encode(word).ids
                    if not ids:
                        continue
                    row = dev.d2h(self.w["emb"] + 4 * H * ids[0], (H,), np.float32)
                    x = np.zeros((4 * H, 1), np.float32)
                    x[:H, 0] = row
                    dev.h2d(xb, x)
                    op = prog1.head
                    # the program's own bindings first (weights, consts), then the inputs we replace
                    op.recipe.bind_slots(op.slots, op.addresses)
                    op.recipe.bind("X", xb)
                    op.recipe.bind("c", self.c_identity[1])
                    op.recipe.bind("p", self.pzero)
                    op.recipe.bind("logits", lb)
                    op.recipe.bind("X_out", ob)
                    op.recipe.launch()
                    dev.sync()
                    lo = dev.d2h(lb, (d.vocab,), np.float32)
                    t5 = np.argsort(lo)[::-1][:5]
                    print(f"[dump {tag}] head alone on emb[{word!r}]: top-5 global "
                          f"{[int(start + t) for t in t5]} "
                          f"as text {[tk.decode([int(start + t)]) for t in t5]}")
            finally:
                dev.free(xb)
                dev.free(ob)
                dev.free(lb)

    # ------------------------------------------------------------------ programs
    def _op(
        self, recipe: Recipe, layer: int | None, names: dict[str, int], copy=None
    ) -> _Op:
        bound: dict[str, int] = {}
        for n in recipe.persistent:
            if n in names:
                bound[n] = names[n]
            elif layer is not None and f"l{layer}.{n}" in self.w:
                bound[n] = self.w[f"l{layer}.{n}"]
            elif n in self.w:
                bound[n] = self.w[n]
            elif n in self.io:
                bound[n] = self.io[n]
            elif n in self.consts:
                bound[n] = self.consts[n]
            else:
                raise ValueError(f"{recipe.name}: unbound {n}")
        return _IxMergeOp(recipe, bound, copy) if copy else _Op(recipe, bound)

    def _ix_tile_copy(self, s: Shape, g: int) -> tuple[list, list, list]:
        """(src, size, dst) of the per-column copies that place a group's own scores (a (rows, T) tile at
        ``ix_tile``) at rows ``run`` .. of its merge buffer (a (run + rows, T) tensor), or at rows 0 .. for the
        first group, which has no running candidates."""
        T, run = s.T, L.INDEX_RUN
        rows = self.ix_rows(s)
        K = run + rows
        row0 = 0 if g == 0 else run
        src = [self.ix_tile + 4 * rows * t for t in range(T)]
        dst = [self.ix_merge[g % 2] + 4 * (K * t + row0) for t in range(T)]
        return src, [4 * rows] * T, dst

    def _program(self, s: Shape) -> _Program:
        if (prog := self._programs.get(s)) is not None:
            return prog
        cfg, tb, base = self.cfg, self.tables[s], self.table
        gathered = self.d.gathered(s.T)

        def t(name: str) -> int:
            return base + tb.off[name]

        def rd(partial: int) -> int:
            """where a graph reads a partial: its all-gathered copy in a gathered step"""
            return (
                self.pall[self.pbuf.index(partial)]
                if gathered and partial in self.pbuf
                else partial
            )

        prog = _Program()
        x, c, p = self.x, self.cbuf, self.pbuf
        head = {
            "tok": t("tok"),
            "meta": t("meta"),
            "arena": self.arena,
            "cmp_state": self.state,
        }
        prog.embed = self._op(
            self.r["embed", s], None, {**head, "X": x[1], "c": c[1], "X_out": x[0]}
        )
        if ("ixg", s, 1) in self.r:
            prog.ixg = [
                self._op(
                    self.r["ixg", s, r],
                    None,
                    {
                        "blk": t(f"blk{r}"),
                        "ix_rec": t(f"ix{r}.rec"),
                        "ix_gbase": t(f"ix{r}.gb"),
                    },
                )
                for r in (1, 2)
            ]
        cur, cin, pin = (
            0,
            self.c_identity[s.T],
            self.pzero,
        )  # the streams, the incoming coefficients and partial
        cnext, pnext = 0, 0
        for layer in range(self.n_layers):
            ratio = self._ratio(layer)
            cs = "cs1" if ratio else "cs0"
            ops: dict = {}
            common = {
                **head,
                "cs": t(cs),
                "cs_inv": t(cs + "_inv"),
                "rows": t(f"rows.{layer}"),
                "slots": t("slots"),
                "lat_pages": t(f"lat.{layer}"),
            }
            if layer in self.engram_layers:
                other = next(k for k in (0, 1) if k != cur)
                eg = {
                    "eg_emb": self.io["eg_emb"]
                    if self.engram == "card"
                    else t(f"eg.{layer}")
                }
                if self.engram == "card":
                    rows, row0 = (
                        self.w[f"l{layer}.eg_rows"],
                        self.w[f"l{layer}.eg_row0"],
                    )
                    idx = (
                        {
                            "tok": t("tok"),
                            "eg_xh": t(f"eg_xh.{layer}"),
                            "eg_xl": t(f"eg_xl.{layer}"),
                        }
                        if self.hashed(s)
                        else {"eg_idx": t(f"eg.{layer}")}
                    )
                    ops["eg_gather"] = self._op(
                        self.r["eg_gather", s, rows, row0], layer, {**idx, **eg}
                    )
                ops["eg"] = self._op(
                    self.r["eg", s],
                    layer,
                    {
                        **eg,
                        "X": x[cur],
                        "c": cin,
                        "p": rd(pin),
                        "X_out": x[other],
                        "X_eg": x[2],
                        "eg_kv": self.io["eg_kv"],
                    },
                )
                if self.eg_sharded(s):
                    ops["eg_gate"] = self._op(
                        self.r["eg_gate", s], layer,
                        {"X": x[other], "X_eg": x[2], "eg_kv_all": self.io["eg_kv_all"]},
                    )  # fmt: skip
                cur = 2
                attn_io = {"X": x[cur], "c": cin}
            else:
                other = next(k for k in (0, 1) if k != cur)
                attn_io = {"X": x[cur], "c": cin, "p": rd(pin), "X_out": x[other]}
                cur = other
            attn_io.update(c_out=c[cnext], p_out=p[pnext])
            key = self._attention_key(layer, s)
            if key[6]:  # a DSpark target layer: its stream mean
                tap = cfg.dspark_target_layer_ids.index(layer)
                common["dsp_main"] = self.io[f"dsp_main{tap}"]
            if not ratio:
                ops["attn"] = self._op(self.r[key], layer, {**common, **attn_io})
            else:
                owner = cfg.kv_owner(layer)
                if layer in cfg.kv_source_layers:
                    common.update(
                        {
                            "cs_g": t("cs_g2" if ratio > 1 else "cs_g1"),
                            "cmp_store": t(f"cmp_store.{layer}"),
                        }
                    )
                    if ratio > 1:
                        common.update(
                            cmp_groups=t(f"cmp_groups.{layer}"),
                            cmp_rows=t(f"cmp_rows.{layer}"),
                        )
                tail = {
                    "cmp_blocks": t(f"cmp_blocks{ratio}"),
                    "cmp_loff": self.loff[owner],
                }
                dense = (
                    {"cslots": t(f"cslots{ratio}"), "cmp_lat": t(f"cmp_lat.{owner}")}
                    if self.dense(s)
                    else {}
                )
                if key[0] == "attn":  # both parts in one launch
                    ops["attn"] = self._op(
                        self.r[key], layer, {**common, **attn_io, **tail}
                    )
                    if (*key, "dense") in self.r:
                        ops["attn_dense"] = self._op(
                            self.r[(*key, "dense")],
                            layer,
                            {
                                **common,
                                **attn_io,
                                **dense,
                                "ix_score": self.ix_tile,
                                "ix_thr": self.io["ix_thr"],
                            },
                        )
                else:
                    ops["attn"] = self._op(self.r[key], layer, {**common, **attn_io})
                    if layer in cfg.index_source_layers:
                        ops.update(
                            self._ix_ops(s, ratio, layer, owner, head, tail, t)
                        )
                    ops["out"] = self._op(
                        self.r["out", s], layer, {**common, **tail, "p_out": p[pnext]}
                    )
                    if ("out", s, "dense") in self.r:
                        ops["out_dense"] = self._op(
                            self.r["out", s, "dense"],
                            layer,
                            {
                                **common,
                                **dense,
                                "p_out": p[pnext],
                                "ix_score": self.ix_tile,
                                "ix_thr": self.io["ix_thr"],
                            },
                        )
            if not WINDOW_ONLY and layer == self.cut and s.prefill:  # (no compressor under the probe)
                prog.store = self._op(self.r["store", s], layer, {**common, **attn_io})
                prog.cut = (attn_io["X"], attn_io["c"], attn_io["p"])
            cin, pin = c[cnext], p[pnext]
            cnext, pnext = 1 - cnext, 1 - pnext
            other = next(k for k in (0, 1) if k != cur)
            ops["ffn"] = self._op(
                self.r["ffn", s], layer,
                {"meta": t("meta"), "X": x[cur], "c": cin, "p": rd(pin), "X_out": x[other], "c_out": c[cnext],
                 "p_out": p[pnext]},
            )  # fmt: skip
            ops["p_attn"], ops["p_ffn"], ops["x_ffn"] = pin, p[pnext], x[other]
            cur = other
            cin, pin = c[cnext], p[pnext]
            cnext, pnext = 1 - cnext, 1 - pnext
            prog.layers.append(ops)
        prog.head = self._op(
            self.r["head", s], None,
            {"X": x[cur], "c": cin, "p": rd(pin), "X_out": x[next(k for k in (0, 1) if k != cur)], "logits": self.logits},
        )  # fmt: skip
        if self.stages:
            prog.inject = self._op(
                self.r["inject", s], None,
                {"cs": t("cs0"), "arena": self.arena,
                 **{f"rows{k}": t(f"rows.{self.n_layers + k}") for k in range(self.stages)}},
            )  # fmt: skip
        self._programs[s] = prog
        return prog

    # ------------------------------------------------------------------ the step

    # ------------------------------------------------------------------ DSpark

    def _draft_native(self, s: Shape, prog: dict):
        """``_draft_launches`` as one native program: the Markov launch of ``registers[0]`` rows a select."""
        from resin._resin import GaudiProgram

        T, ccl = s.T, self.nranks > 1
        native = GaudiProgram(1)
        native.section("draft")

        def launch(op) -> None:
            if isinstance(op, _IxMergeOp):  # the group's own scores, at their rows of the merge buffer
                self.device.d2d_multi(*op.copy)
            native.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())

        def reduce(partial: int) -> None:
            count = self.d.hidden * T
            if self.d.gathered(T):
                native.allgather(partial, self.pall[self.pbuf.index(partial)], count)
            else:
                native.allreduce(partial, count)

        launch(prog["embed"])
        for ops in prog["stages"]:
            launch(ops["attn"])
            if ccl:
                reduce(ops["p_attn"])
            launch(ops["ffn"])
            if ccl:
                reduce(ops["p_ffn"])
        launch(prog["head"])
        if ccl:
            native.allgather(self.logits, self.logits_all, self.d.vocab * T)
        rows = sorted(prog["markov"])  # the Markov launch of each row count, the arms in this order
        table = [rows.index(b) if b in prog["markov"] else -1 for b in range(rows[-1] + 1)]
        native.select([(0, rows[-1] + 1)], table)
        for b in rows:
            native.arm()
            launch(prog["markov"][b])
        native.end()
        native.finish()
        self._draft_natives[s] = native
        return native

    def _draft_program(self, s: Shape) -> dict:
        if (prog := self._draft_programs.get(s)) is not None:
            return prog
        tb, base = self.tables[s], self.table
        gathered = self.d.gathered(s.T)

        def t(name: str) -> int:
            return base + tb.off[name]

        def rd(partial: int) -> int:
            return (
                self.pall[self.pbuf.index(partial)]
                if gathered and partial in self.pbuf
                else partial
            )

        x, c, p = self.x, self.cbuf, self.pbuf
        head = {
            "tok": t("tok"),
            "meta": t("meta"),
            "arena": self.arena,
            "cmp_state": self.state,
        }
        prog: dict = {
            "embed": self._op(self.r["embed", s], None, {**head, "X": x[1], "c": c[1], "X_out": x[0]}),
            "stages": [],
        }  # fmt: skip
        # the first stage starts as layer 0 does: identity coefficients, a zero partial
        cur, cin, pin, cnext, pnext = 0, self.c_identity[s.T], self.pzero, 0, 0
        for k in range(self.stages):
            layer = self.n_layers + k
            io = {
                **head, "cs": t("cs0"), "cs_inv": t("cs0_inv"), "rows": t(f"rows.{layer}"), "slots": t("slots"),
                "lat_pages": t(f"lat.{layer}"), "ix_lslots": t("dsp_lslots"), "dsp_bidx": t("dsp_bidx"),
                "X": x[cur], "c": cin, "p": rd(pin), "X_out": x[1 - cur], "c_out": c[cnext], "p_out": p[pnext],
            }  # fmt: skip
            ops = {"attn": self._op(self.r["dsp_attn", s], layer, io)}
            cur, cin, pin = 1 - cur, c[cnext], p[pnext]
            cnext, pnext = 1 - cnext, 1 - pnext
            ops["ffn"] = self._op(
                self.r["dsp_ffn", s], layer,
                {"meta": t("meta"), "X": x[cur], "c": cin, "p": rd(pin), "X_out": x[1 - cur], "c_out": c[cnext],
                 "p_out": p[pnext]},
            )  # fmt: skip
            ops["p_attn"], ops["p_ffn"] = pin, p[pnext]
            cur, cin, pin = 1 - cur, c[cnext], p[pnext]
            cnext, pnext = 1 - cnext, 1 - pnext
            prog["stages"].append(ops)
        prog["head"] = self._op(
            self.r["head", s], self.n_layers + self.stages - 1,
            {"X": x[cur], "c": cin, "p": rd(pin), "X_out": x[1 - cur], "logits": self.logits},
        )  # fmt: skip
        prog["markov"] = {
            b: self._op(
                self.r["dsp_markov", s, b],
                None,
                {
                    "logits_all": self.logits_all,
                    "dsp_anchor": t("dsp_anchor"),
                    "dsp_out": self.io["dsp_out"],
                },
            )  # fmt: skip
            for b in range(1, DSPARK_ROWS + 1)
            if ("dsp_markov", s, b) in self.r
        }
        self._draft_programs[s] = prog
        return prog

    # ------------------------------------------------------------------ tables

    def _step_program(self, s: Shape, prog: _Program):
        """``_step_launches``' calls of a step of ``s`` as one native program (``resin._resin.GaudiProgram``) with
        a section per part (``_sections``): the embedding, each layer (in ``_run_layer``'s order: the dense or
        plain attention and output launches, and a compressed layer's index launches and selection, chosen by the
        step's counts, ``_step_registers``), the DSpark injection, the head for every column's logits and for
        column ``registers[_R_COLUMN]`` (``_head``), and a prefill step's owner store (bounded replay)."""
        native = self._native_programs.get(s)
        if native is not None:
            return native
        from resin._resin import GaudiProgram

        native = GaudiProgram(_REGISTERS)
        T, ccl = s.T, self.nranks > 1

        def launch(op) -> None:
            if isinstance(op, _IxMergeOp):  # the group's own scores, at their rows of the merge buffer
                native.copy_multi(
                    op.copy[0].tolist(), op.copy[1].tolist(), op.copy[2].tolist()
                )
            native.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())

        def reduce(partial: int) -> None:
            count = self.d.hidden * T
            if self.d.gathered(T):
                native.allgather(partial, self.pall[self.pbuf.index(partial)], count)
            else:
                native.allreduce(partial, count)

        def dense_or(ops: dict, name: str, ratio: int) -> None:
            """``name``'s launch, or its dense form in a step whose counts say dense for ``ratio``"""
            if ratio in _R_DENSE and f"{name}_dense" in ops:
                native.select([(_R_DENSE[ratio], 2)], [0, 1])
                native.arm()
                launch(ops[name])
                native.arm()
                launch(ops[f"{name}_dense"])
                native.end()
            else:
                launch(ops[name])

        def index(ratio: int, ops: dict) -> None:
            """The tiled index chain: the plain arm (each group's score launches and its merge), the candidate
            arm (with each group's block maxima or candidate mask, and the tail that publishes the blocks), and
            the sparse reindex path (its launches over the candidate blocks and their compact selection). The
            register of each arm takes its groups' launches, so a step runs only the groups it needs."""
            plain = ops["ixp"]
            arms = (
                (plain, _R_IX[ratio], []),
                (ops.get("ixc2", plain), _R_PF[ratio], ops.get("ixc_tail", [])),
                (ops.get("ixsp", plain), _R_IXC, ops.get("ixsp_tail", [])),
            )
            # a step that needs an arm the layer does not have is an error, not a silent fallback
            native.select(
                [(_R_SPARSE, 2), (_R_CAND, 2)],
                [
                    0,
                    1 if "ixc2" in ops else -1,
                    2 if "ixsp" in ops else 0,
                    2 if "ixsp" in ops else 1,
                ],
            )
            for arm, register, tail in arms:
                native.arm()
                native.take(register)
                for op in arm:
                    launch(op)
                native.end()
                for op in tail:
                    launch(op)
            native.end()


        native.section("embed")
        launch(prog.embed)
        # the tiled index chain's group tables: one launch per ratio for the whole step, whatever the number of
        # merge groups (ds_ix_gslot derives each group's slot records from the step's positions)
        if prog.ixg:
            native.section("ix groups")
            for op in prog.ixg:
                launch(op)
        for layer, ops in enumerate(prog.layers):
            native.section(f"layer {layer}")
            ratio = self._ratio(layer)
            if "eg" in ops:
                if "eg_gather" in ops:
                    launch(ops["eg_gather"])
                    if ccl:
                        native.allreduce(self.io["eg_emb"], 256 * J * T)
                launch(ops["eg"])
                if "eg_gate" in ops:
                    if ccl:
                        native.allgather(self.io["eg_kv"], self.io["eg_kv_all"], self.eg_rows_r * T)
                    launch(ops["eg_gate"])
            dense_or(ops, "attn", ratio)
            if "ixp" in ops:
                index(ratio, ops)
            if "out" in ops:
                dense_or(ops, "out", ratio)
            if ccl:
                reduce(ops["p_attn"])
            launch(ops["ffn"])
            if ccl:
                reduce(ops["p_ffn"])
        native.section("inject")
        if prog.inject is not None:
            launch(prog.inject)
        # the head (``_head``): every column's logits, or one column's (gathered as a one-column step's)
        native.section("head")
        if prog.head is not None:
            launch(prog.head)
            if ccl:
                native.allgather(self.logits, self.logits_all, self.d.vocab * T)
        native.section("head_column")
        if prog.head is not None:
            launch(prog.head)
            if ccl:
                native.allgather((self.logits, _R_COLUMN, 4 * self.d.vocab), self.logits_all, self.d.vocab)
        native.section("store")
        if prog.store is not None:
            launch(prog.store)
        native.finish()  # the program holds copies of the ops' bindings
        self._native_programs[s] = native
        return native

    def _native_fill(self):
        """The native step fill (``resin._resin.Ds41Fill``, built with the ``gaudi`` feature) of every shape's tables
        (the CPU tests keep a numpy fill of the same tables)."""
        try:
            from resin._resin import Ds41Fill
        except ImportError as error:
            raise RuntimeError(
                "DeepSeek-V4.1 steps run in the native executor: resin must be built with the gaudi feature"
            ) from error
        cfg, geo = self.cfg, self.geo
        # host-Engram: the fill gathers + dequantizes the rows from the shared mmap tables in Rust.
        # One (weight, scale) pair per Engram layer index e (a dummy empty pair for a layer this rank
        # does not run, which no shape's engram_emb descriptor references).
        engram_host = None
        if self.engram == "host" and self.engram_layers:
            engram_host = []
            for layer in cfg.engram_layer_ids:
                if layer in self.eg_host:
                    engram_host.append(self.eg_host[layer])
                else:
                    engram_host.append(
                        (np.zeros((0, 256), np.uint8), np.zeros((0, 8), np.uint8))
                    )
        engram = None
        if self.engram_layers:
            h = self._engram_hash()
            engram = (
                np.asarray(h.token_map, np.int64).tolist(),
                int(h.pad_id),
                np.asarray(h.multipliers, np.int64).ravel().tolist(),
                cfg.engram_max_ngram_size,
                cfg.engram_n_heads,
                len(cfg.engram_layer_ids),
                np.asarray(h.primes, np.int64).ravel().tolist(),
                np.asarray(h.offsets, np.int64).ravel().tolist(),
            )
        cs = cfg.candidate_source_layer
        native = Ds41Fill(
            heads=self.d.heads,
            index_heads=L.INDEX_HEADS,
            sliding_window=cfg.sliding_window,
            layer_rows=self._layer_rows.tolist(),
            owners=[
                (cfg.compress_ratios[o], geo.owner_pages(o), geo.owner_offset(o))
                for o in geo.owners
            ],
            pending=[geo.owners.index(o) for o in geo.pending_owners],
            pages=self.pages,
            kmax=self.kmax,
            kc=self.kc,
            candidate=(cfg.compress_ratios[cs], cfg.candidate_topk_blocks)
            if cs >= 0
            else None,
            engram=engram,
            cs0=np.ascontiguousarray(self._cs_all[0], np.float32),
            cs1=np.ascontiguousarray(self._cs_all[1], np.float32),
            radix_min=RADIX_MIN,
            prefilter=(PREFILTER_BLOCKS, PREFILTER_MIN),
            dense_keys=DENSE_KEYS,
            geometry=(PAGE_TOKENS, HISTORY_TOKENS, PAGE_ROWS, INDEX_ROWS, INDEX_KEYS, PENDING_SLOTS),
            engram_host=engram_host,
        )
        for s, tb in self.tables.items():
            off = tb.off
            layers = range(self.n_layers + self.stages)
            dense = self.dense(s)
            native.add_shape(
                {
                    "t": s.T,
                    "prefill": s.prefill,
                    "qb": s.qb,
                    "win_slots": s.win_slots,
                    "win_pages": s.win_keys // PAGE_TOKENS,
                    "groups2": s.groups2,
                    "ix_visb2": off["ix_visb2"],
                    "cslots": (off["cslots1"], off["cslots2"]) if dense else None,
                    "cmp_lat": [off[f"cmp_lat.{o}"] for o in geo.owners] if dense else [],
                    "draft": (off["dsp_bidx"], off["dsp_lslots"], off["dsp_anchor"])
                    if "dsp_bidx" in off
                    else None,
                    "engram_ids": [
                        (off[f"eg.{layer}"], cfg.engram_layer_ids.index(layer))
                        for layer in self.engram_layers
                    ]
                    if self.engram == "card" and not self.hashed(s)
                    else [],
                    "engram_emb": [
                        (off[f"eg.{layer}"], cfg.engram_layer_ids.index(layer))
                        for layer in self.engram_layers
                    ]
                    if self.engram == "host"
                    else [],
                    "ix_slots": s.ix_slots,
                    "ix_tp": self.ix_tp(s),
                    "ix_lpg": self.ix_lpg(s),
                    "ix_rows": self.ix_rows(s),
                    "blk": (off["blk1"], off["blk2"]),
                    "ix_rec": (off["ix1.rec"], off["ix2.rec"]),
                    "ix_gb": (off["ix1.gb"], off["ix2.gb"]),
                    "nbytes": tb.nbytes,
                    "zero": [(int(a), int(b)) for a, b in tb.zero],
                    "tok": off["tok"],
                    "meta": off["meta"],
                    "cs": (off["cs0"], off["cs1"]),
                    "cs_inv": (off["cs0_inv"], off["cs1_inv"]),
                    "cs_g2": off["cs_g2"],
                    "cs_g1": off["cs_g1"],
                    "cmp_store": [off[f"cmp_store.{o}"] for o in geo.owners],
                    "cmp_groups": [off[f"cmp_groups.{o}"] for o in geo.pending_owners],
                    "cmp_rows": [off[f"cmp_rows.{o}"] for o in geo.pending_owners],
                    "slots": off["slots"],
                    "rows": [off[f"rows.{layer}"] for layer in layers],
                    "lat": [off[f"lat.{layer}"] for layer in layers],
                    "ix_visb": off["ix_visb"],
                    "ix_visc": off["ix_visc"],
                    "ixc": [off[f"ixc.{j}"] for j in range(self.ixc_launches(s))],
                    "ix_vis": (off["ix_vis1"], off["ix_vis2"]),
                    "cmp_blocks": (off["cmp_blocks1"], off["cmp_blocks2"]),
                    "engram": [
                        (
                            off[f"eg_xh.{layer}"],
                            off[f"eg_xl.{layer}"],
                            cfg.engram_layer_ids.index(layer),
                        )
                        for layer in self.engram_layers
                    ]
                    if self.engram_layers and self.hashed(s)
                    else [],
                }
            )
        return native
