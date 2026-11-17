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
WORKSPACE_RESERVE = 3 << 30
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
    def ix_slots(self) -> int:
        """index slots per launch (decode: fewer launches for wider steps)"""
        return PREFILL_IX_SLOTS if self.prefill else 32 if self.T >= 4 else 16

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
        for r in (1, 2):
            add(f"ix_vis{r}", i32, T)
            add(f"cmp_blocks{r}", i32, T, model.pages)
            for j in range(model.ix_launches(shape, r)):
                add(f"ix{r}.{j}", i32, shape.ix_slots, 8)
                for o in geo.owners:
                    if model.cfg.compress_ratios[o] == r:
                        add(
                            f"ix{r}.{j}.{o}",
                            i32,
                            shape.ix_slots,
                            INDEX_KEYS // PAGE_TOKENS,
                        )
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
        whole = sorted(
            (self.off[n], self.off[n] + 4 * int(np.prod(self.spec[n][1])))
            for n in ("cmp_blocks1", "cmp_blocks2")
        )
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


class _Program:
    """Every launch of one step shape, bound once."""

    def __init__(self) -> None:
        self.embed: _Op | None = None
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
        if callable(capacity):
            capacity = capacity(device.memory_info()[0] - self.workspace_reserve)
        self.capacity = capacity
        self.arena_blocks = self.geo.arena_blocks(capacity)
        if arena is None:
            self.arena = self._malloc(self.arena_blocks * 1024 * 512)
            device.memset32(self.arena, 0, self.arena_blocks * 1024 * 128)
        else:
            self.arena = arena
        state = self.geo.state_bytes(capacity)
        self.state = self._malloc(state)
        device.memset32(self.state, 0, state // 4)
        self.state_rows = state // (4 * STATE_FLOATS)
        log(
            f"ds41: cache {capacity} slabs, {self.arena_blocks / 1024 * 0.5:.1f} GiB per card"
        )
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
        """Bytes kept out of the arena for the step tables and the shared workspace. The workspace grows with
        the context: a prefill index selection over candidate blocks (index_select_graph, cand "user") holds
        the masked scores (kmax, TMAX) and the block maxima (kb, TMAX) as f32."""
        return max(WORKSPACE_RESERVE, 4 * TMAX * (self.kmax + self.kb) + (1 << 30))

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
        self, s: Shape, ratio: int, j: int, owner: int, head: dict
    ) -> dict[str, int]:
        """an index launch's tables: its slots and pages, or each part's rows of them"""
        base = self.table + self.tables[s].off[f"ix{ratio}.{j}"]
        pages = self.table + self.tables[s].off[f"ix{ratio}.{j}.{owner}"]
        recipe = self.r["ix", s]
        if "ix_slots" in recipe.persistent:
            return {**head, "ix_slots": base, "ix_pages": pages}
        names = dict(head)
        P = sum(n.startswith("ix_slots.p") for n in recipe.persistent)
        Sp = s.ix_slots // P
        for p in range(P):
            names[f"ix_slots.p{p}"] = base + 4 * 8 * Sp * p
            names[f"ix_pages.p{p}"] = pages + 4 * (INDEX_KEYS // PAGE_TOKENS) * Sp * p
            names[f"ix_sink.p{p}"] = self.io["ix_sink"] + 256 * p
        return names

    def ix_launches(self, shape: Shape, ratio: int) -> int:
        """index launches a step of ``shape`` may need for ratio-``ratio`` keys"""
        chunks = -(-self.kmax // ratio // INDEX_KEYS)
        blocks = shape.T // shape.qb
        return -(-blocks * chunks // shape.ix_slots)

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
            "ix_score": buf(4 * self.kmax * T),
            "ix_list": buf(4 * L.INDEX_TOP * T),
            "ix_lslots": buf(4 * 8 * T),
            "cand_bmax": buf(4 * self.kb * T),
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
                            query_block=qb, capture=capture, dense=DENSE_KEYS, scores=self.kmax,
                        )  # fmt: skip
                else:
                    graphs[key] = L.attention_pre_graph(
                        dev, d, T, s.win_slots, s.win_keys, nb, ratio=ratio, owner=owner, source=source,
                        groups=s.groups2 if ratio > 1 else T, state_rows=R, residual=residual, query_block=qb,
                        capture=capture,
                    )  # fmt: skip
            if any(cfg.compress_ratios[i] for i in range(self.n_layers)):
                graphs["ix", s] = L.index_score_graph(
                    dev, d, T, s.ix_slots, INDEX_KEYS, nb, self.kmax, query_block=qb,
                    parts=PREFILL_IX_PARTS if s.prefill else 1,
                )  # fmt: skip
                cands = (
                    ("", "source", "user") if cfg.candidate_source_layer >= 0 else ("",)
                )
                top = cfg.candidate_topk_blocks
                for cand in cands:
                    for radix in (False,) if s.prefill else (False, True):
                        graphs["sel", s, cand, radix] = L.index_select_graph(
                            dev, d, T, self.kmax, cand, top, radix,
                            sparse=cand == "source" and not s.prefill, cand_blocks=self.kb,
                        )  # fmt: skip
                    if s.prefill and PREFILTER_BLOCKS:
                        graphs["sel", s, cand, "pf"] = L.index_select_graph(
                            dev, d, T, self.kmax, cand, top, cand_blocks=self.kb, prefilter=PREFILTER_BLOCKS
                        )  # fmt: skip
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
                        dev, d, T, nb, self.pages, dense=DENSE_KEYS, scores=self.kmax
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

    def _attention_key(self, layer: int, s: Shape) -> tuple:
        cfg = self.cfg
        ratio = cfg.compress_ratios[layer]
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
        recipes = self.recipes.compile_many(list(graphs.values()))
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

    # ------------------------------------------------------------------ programs
    def _op(self, recipe: Recipe, layer: int | None, names: dict[str, int]) -> _Op:
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
        return _Op(recipe, bound)

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
        cur, cin, pin = (
            0,
            self.c_identity[s.T],
            self.pzero,
        )  # the streams, the incoming coefficients and partial
        cnext, pnext = 0, 0
        for layer in range(self.n_layers):
            ratio = cfg.compress_ratios[layer]
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
                            {**common, **attn_io, **dense},
                        )
                else:
                    ops["attn"] = self._op(self.r[key], layer, {**common, **attn_io})
                    if layer in cfg.index_source_layers:
                        ops["ix"] = [
                            self._op(self.r["ix", s], layer, self._ix_names(s, ratio, j, owner, head))
                            for j in range(self.ix_launches(s, ratio))
                        ]  # fmt: skip
                        cs = cfg.candidate_source_layer
                        cand = (
                            ""
                            if cs < 0 or layer < cs
                            else "source"
                            if layer == cs
                            else "user"
                        )
                        names = {"ix_vis": t(f"ix_vis{ratio}"), "ix_visb": t("ix_visb")}
                        ops["sel"] = {
                            radix: self._op(self.r["sel", s, cand, radix], layer, names)
                            for radix in ((False,) if s.prefill else (False, True))
                        }
                        if cand:  # steps where every column's blocks are all candidates
                            ops["sel"]["all"] = self._op(
                                self.r["sel", s, "", False], layer, names
                            )
                        if ("sel", s, cand, "pf") in self.r:
                            pf = {
                                **names,
                                "ix_visb": t(self._visb(ratio)),
                                "ix_visc": t("ix_visc"),
                            }
                            ops["sel"]["pf"] = self._op(
                                self.r["sel", s, cand, "pf"], layer, pf
                            )
                            if cand:
                                ops["sel"]["pf_all"] = self._op(
                                    self.r["sel", s, "", "pf"], layer, pf
                                )
                        if cand == "user" and self.ixc_launches(s):
                            ops["ixc"] = [
                                self._op(self.r["ixc", s], layer, {**head, **tail, "ix_slots": t(f"ixc.{j}")})
                                for j in range(self.ixc_launches(s))
                            ]  # fmt: skip
                            ops["sel"]["sparse"] = self._op(
                                self.r["sel", s, "user", "sparse"], layer,
                                {"ix_vis": t(f"ix_vis{ratio}"), "ix_visc": t("ix_visc")},
                            )  # fmt: skip
                    ops["out"] = self._op(
                        self.r["out", s], layer, {**common, **tail, "p_out": p[pnext]}
                    )
                    if ("out", s, "dense") in self.r:
                        ops["out_dense"] = self._op(
                            self.r["out", s, "dense"],
                            layer,
                            {**common, **dense, "p_out": p[pnext]},
                        )
            if layer == self.cut and s.prefill:
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
            """``_run_layer``'s index launches and selection: by candidates in a sparse step, else the first
            counts[ratio] index launches and the prefiltered, all, or radix selection"""
            sel, arms = ops["sel"], []

            def arm(register: int, launches: list, select) -> int:
                arms.append((register, launches, select))
                return len(arms) - 1

            sparse = arm(_R_IXC, ops["ixc"], sel["sparse"]) if "ixc" in ops and "sparse" in sel else -1
            every = arm(_R_IX[ratio], ops["ix"], sel["all"]) if "all" in sel else -1
            radix = [arm(_R_IX[ratio], ops["ix"], sel[r]) if r in sel else -1 for r in (False, True)]
            pf = arm(_R_IX[ratio], ops["ix"], sel["pf"]) if "pf" in sel else -1
            pf_all = arm(_R_IX[ratio], ops["ix"], sel["pf_all"]) if "pf_all" in sel else -1

            def chosen(is_sparse: int, cand: int, by_radix: int, prefiltered: int) -> int:
                if is_sparse and "ixc" in ops:
                    return sparse
                if prefiltered and "pf" in sel:
                    return pf_all if not cand and "pf_all" in sel else pf
                if not cand and "all" in sel:
                    return every
                return radix[by_radix]

            # keys (sparse, cand, radix, prefiltered), the first the most significant; -1: a step _run_layer
            # cannot make
            table = [chosen(*key) for key in np.ndindex(2, 2, 2, 2)]
            keys = [(_R_SPARSE, 2), (_R_CAND, 2), (_R_RADIX[ratio], 2), (_R_PF[ratio], 2)]
            native.select(keys, table)
            for register, launches, select in arms:
                native.arm()
                native.take(register)
                for op in launches:
                    launch(op)
                native.end()
                launch(select)
            native.end()

        native.section("embed")
        launch(prog.embed)
        for layer, ops in enumerate(prog.layers):
            native.section(f"layer {layer}")
            ratio = self.cfg.compress_ratios[layer]
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
            if "ix" in ops:
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
        if self.engram == "host" and self.engram_layers:
            raise ValueError("the native fill hashes the Engram rows on the cards (engram='card')")
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
                    if not self.hashed(s)
                    else [],
                    "ix_slots": s.ix_slots,
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
                    "ix": tuple(
                        [off[f"ix{r}.{j}"] for j in range(self.ix_launches(s, r))]
                        for r in (1, 2)
                    ),
                    "ix_owner": tuple(
                        [
                            [
                                off[f"ix{r}.{j}.{o}"]
                                for o in geo.owners
                                if cfg.compress_ratios[o] == r
                            ]
                            for j in range(self.ix_launches(s, r))
                        ]
                        for r in (1, 2)
                    ),
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
