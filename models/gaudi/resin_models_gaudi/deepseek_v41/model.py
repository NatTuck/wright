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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from resin.models.deepseek_v41 import DeepSeekV41Config
from resin_models_gaudi.deepseek_v41 import layers as L
from resin_models_gaudi.deepseek_v41.cache import (
    HISTORY_TOKENS,
    INDEX_ROWS,
    PAGE_ROWS,
    PAGE_TOKENS,
    PAIR_ROWS,
    STATE_FLOATS,
    CacheGeometry,
)
from resin_models_gaudi.deepseek_v41.rope import cs_table, layer_frequencies
from resin_models_gaudi.deepseek_v41.weights import (
    RankSlices,
    engram_hash_tables,
    engram_rows,
    engram_table,
    prepare_compressed,
    prepare_draft_stage,
    prepare_engram,
    prepare_layer,
    reader,
    split64,
    table_rows,
)
from resin_models_gaudi.glm5_3_flash.layers import sample_graph
from resin_models_gaudi.glm5_3_flash.model import (
    _SAMPLE_LAST,
    _SAMPLE_TABLE,
    _SAMPLED,
    _SAMPLED_BYTES,
    Staged,
    _Op,
    _sampled_fields,
)
from resin_models_gaudi.qwen3_5.sampling import PARAMS
from resin_models_gaudi.synapse.graph import Recipe, RecipeCache
from resin_models_gaudi.synapse.progress import info
from resin_models_gaudi.synapse.runtime import Device

DECODE_BUCKETS = (1, 2, 4, 8, 16)
PREFILL_BUCKETS = (32, 128, 512, 1024, 2048)
PAGED_T = 128  # prefill launches this wide decode each routed expert once per page of its rows
MAX_T = DECODE_BUCKETS[-1]
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
RADIX_MIN = 16384  # decode steps whose columns see more index keys pick their top 512 by radix passes
J = 24  # Engram rows per column: (max n-gram - 1) x heads
WORKSPACE_RESERVE = 3 << 30
_HOST_EVENT = 60  # 60, 61: the pinned table copies' uploads
_STAGE_EVENT = 56  # 56, 57: the sampler's staging copies


def bucket(n: int, prefill: bool) -> int:
    for b in PREFILL_BUCKETS if prefill else DECODE_BUCKETS:
        if n <= b:
            return b
    raise ValueError(
        f"{n} columns exceed the {'prefill' if prefill else 'decode'} buckets"
    )


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
    ) -> None:
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
        # decode steps all-gather the ranks' partials and sum them in the next graph
        self.d = replace(
            L.Dims.of(cfg, nranks, self.sl.vocab.stop - self.sl.vocab.start),
            gather_tokens=MAX_T if nranks > 1 and gather else 0,
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
        if callable(capacity):
            capacity = capacity(device.memory_info()[0] - self.workspace_reserve)
        self.capacity = capacity
        self.arena_blocks = self.geo.arena_blocks(capacity)
        self.arena = self._malloc(self.arena_blocks * 1024 * 512)
        device.memset32(self.arena, 0, self.arena_blocks * 1024 * 128)
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
        self.host = [device.host_buffer(size, np.uint8) for _ in range(2)]
        self._host_turn, self._host_pending = 0, [False, False]
        for tb in self.tables.values():
            for h in self.host:
                tb.bind(h)
        self._layer_rows = np.array(
            [self.geo.layer_offset(k) for k in range(self.n_layers + self.stages)],
            np.int64,
        )
        self._programs: dict[Shape, _Program] = {}
        self._draft_programs: dict[Shape, dict] = {}
        self.probe = None  # (layer, ops, T): after each layer's ffn and its collective
        self._sample_ops: dict[tuple, _Op] = {}
        self._staging = [Staged(device, _STAGE_EVENT + k) for k in range(2)]
        self._stage_turn = 0
        self._column = False  # the last step all-gathered one column's logits
        self._compile()

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
        dev, cfg, src, sl = self.device, self.cfg, self.src, self.sl
        w: dict[str, int] = {}

        def put(name: str, array: np.ndarray) -> None:
            a = np.ascontiguousarray(array)
            w[name] = dev.malloc(max(256, a.nbytes))
            dev.h2d(w[name], a)

        from resin_models_gaudi.qwen3_5.checkpoint import kperm

        put("emb", src.read("embed.weight"))
        put("head", kperm(src.read("head.weight", sl.vocab)))
        put("norm", (src.read("norm.weight").astype(np.uint32) << 16).view(np.float32))
        for layer in range(self.n_layers):
            if barrier is not None:
                barrier()  # the ranks read each layer's shards together
            t0 = time.time()
            parts = prepare_layer(src, cfg, layer, sl)
            if cfg.compress_ratios[layer]:
                parts.update(prepare_compressed(src, cfg, layer))
            if layer in self.engram_layers:
                parts.update(prepare_engram(src, layer))
            for name, array in parts.items():
                put(f"l{layer}.{name}", array)
            del parts
            if layer in self.engram_layers and self.engram == "card":
                self._load_table(w, layer)
            self.log(f"layer {layer} in {time.time() - t0:.0f} s")
        for k in range(self.stages):
            t0 = time.time()
            for name, array in prepare_draft_stage(src, cfg, k, sl).items():
                put(f"l{self.n_layers + k}.{name}", array)
            self.log(f"DSpark stage {k} in {time.time() - t0:.0f} s")
        self._draft_names(w)
        return w

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

    def _load_table(self, w: dict[str, int], layer: int) -> None:
        """This rank's rows of an Engram table, converted in chunks."""
        dev = self.device
        total = self.src.info(f"layers.{layer}.engram.embed.weight")[1][0]
        rows = table_rows(total, self.rank, self.nranks)
        n = rows.stop - rows.start
        w[f"l{layer}.eg_table"] = table = dev.malloc(256 * n)
        w[f"l{layer}.eg_rs"] = scales = dev.malloc(4 * n)
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
        self.pzero = const(np.zeros((max(TMAX, MAX_T * self.nranks), H), f))

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
            buf(4 * H * MAX_T * self.nranks) for _ in range(2)
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
        self.sample_table = buf(4096)

    def _graphs(self) -> dict[tuple, object]:
        dev, d, cfg, nb = self.device, self.d, self.cfg, self.arena_blocks
        graphs: dict[tuple, object] = {}
        R = self.state_rows
        for s in self.shapes:
            T, qb = s.T, s.qb
            graphs["embed", s] = L.embed_graph(dev, d, T, cfg.vocab_size)
            graphs["ffn", s] = L.ffn_graph(dev, d, T, paged=s.prefill and T >= PAGED_T)
            graphs["head", s] = L.head_graph(dev, d, T)
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
        t0 = time.time()
        graphs = self._graphs()
        recipes = self.recipes.compile_many(list(graphs.values()))
        self.r: dict[tuple, Recipe] = dict(zip(graphs.keys(), recipes, strict=True))
        for g in graphs.values():
            g.destroy()
        cached = sum(r.from_cache for r in recipes)
        self.log(
            f"ds41: {len(recipes)} recipes ({cached} cached) in {time.time() - t0:.1f} s"
        )
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
    def forward(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        rows: Sequence[Row],
        prefill: bool = False,
    ):
        """T columns (``tokens``, ``positions``) whose sequences are ``rows`` -> logits (n, V) f32"""
        T = self.launch(tokens, positions, rows, prefill)
        return self.logits_host(T)[: len(tokens)]

    def logits_host(self, T: int) -> np.ndarray:
        self.device.sync()
        V = self.d.vocab
        out = self.device.d2h(self.logits_all, (self.nranks, T, V), np.float32)
        return np.ascontiguousarray(out.transpose(1, 0, 2).reshape(T, self.nranks * V))

    def sample(self, tokens, positions, rows, params, last=None, prefill=False):
        """A step, then the device sampler over every column (decode) or column ``last`` (a prefill chunk's last
        real token) -> (tokens, logprobs, top ids, top logprobs)"""
        T = self.launch(
            tokens, positions, rows, prefill, "all" if last is None else last
        )
        return self.sample_launched(Shape(T, prefill), params, last)

    def sample_launched(self, s: Shape, params: np.ndarray, last: int | None = None):
        R = self._sample_launch(s, params, last)
        dev = self.device
        dev.sync()
        out = dev.d2h(self.sampled_buf, (_SAMPLED_BYTES,), np.uint8)
        return _sampled_fields(out, R, len(params))

    def sample_async(self, s: Shape, params: np.ndarray) -> Staged:
        """Sample every column of the decode step just launched and copy the results into a pinned staging
        buffer without waiting; ``Staged.result`` waits for them."""
        self._sample_launch(s, params, None)
        k = self._stage_turn
        self._stage_turn ^= 1
        staged = self._staging[k]
        staged.drain()
        self.device.d2h_async(staged.buffer, self.sampled_buf)
        self.device.record(_STAGE_EVENT + k)
        staged.arm(s.T, len(params))
        return staged

    def _sample_launch(self, s: Shape, params: np.ndarray, last: int | None) -> int:
        if last is not None and self._column:  # the step gathered only column ``last``
            s, last = Shape(1, False), None
        R = 1 if last is not None else s.T
        n = len(params)
        table = np.zeros(_SAMPLE_TABLE // 4, np.float32)
        table[: n * PARAMS] = np.asarray(params, np.float32).ravel()
        table.view(np.int32)[_SAMPLE_LAST // 4] = 0 if last is None else last
        dev = self.device
        dev.h2d_nb(self.sample_table, table)
        op = self._sample_ops.get((s, R))
        if op is None:
            op = self._sample_ops[s, R] = self._op(
                self.r["sample", s, R], None,
                {**self.sampled, "logits_all": self.logits_all, "samp_params": self.sample_table,
                 "last": self.sample_table + _SAMPLE_LAST},
            )  # fmt: skip
        op.run(dev.lib)
        return R

    def launch(
        self,
        tokens: np.ndarray | None,
        positions: np.ndarray,
        rows: Sequence[Row],
        prefill: bool = False,
        logits: str | int | None = "all",
    ) -> int:
        """Enqueue a step through the head and the logits all-gather; returns its bucket. ``tokens`` None: a
        decode step whose columns' tokens are the previous step's samples, read on the device. ``logits``: "all"
        columns; one column (its index), all-gathered alone for sample_launched(..., last=that column); or None
        (no head: a prefill chunk that samples nothing)."""
        dev, d = self.device, self.d
        n = len(positions)
        s = Shape(bucket(n, prefill), prefill)
        if prefill and len(rows) != 1:
            raise ValueError("a prefill step runs one row")
        if tokens is None and not self.hashed(s):
            raise ValueError(
                "device tokens need a decode step with the Engram tables on the cards"
            )
        tb = self.tables[s]
        k = self._host_turn
        self._host_turn ^= 1
        if self._host_pending[k]:
            dev.wait(_HOST_EVENT + k)
        counts = self._fill(
            k,
            tb,
            None if tokens is None else np.asarray(tokens),
            np.asarray(positions),
            rows,
        )
        dev.h2d_async(self.table, self.host[k][: tb.nbytes])
        dev.record(_HOST_EVENT + k)
        self._host_pending[k] = True
        if tokens is None:
            dev.d2d(
                self.table + tb.off["tok"],
                self.sampled_buf + _SAMPLED["tok_out"],
                4 * n,
            )
        lib, prog = dev.lib, self._program(s)
        T = s.T
        ccl = self.nranks > 1
        prog.embed.run(lib)
        for layer, ops in enumerate(prog.layers):
            if "eg" in ops:
                if "eg_gather" in ops:
                    ops["eg_gather"].run(lib)
                    if ccl:
                        dev.allreduce(self.io["eg_emb"], 256 * J * T)
                ops["eg"].run(lib)
                if "eg_gate" in ops:
                    if ccl:
                        dev.allgather(
                            self.io["eg_kv"], self.io["eg_kv_all"], self.eg_rows_r * T
                        )
                    ops["eg_gate"].run(lib)
            ratio = self.cfg.compress_ratios[layer]
            dense = counts.get(("dense", ratio), False)
            (ops["attn_dense"] if dense and "attn_dense" in ops else ops["attn"]).run(
                lib
            )
            if "ix" in ops:
                ratio = self.cfg.compress_ratios[layer]
                sel = ops["sel"]
                if counts["sparse"] and "ixc" in ops:
                    # only the candidate entries
                    for op in ops["ixc"][: counts["ixc"]]:
                        op.run(lib)
                    sel["sparse"].run(lib)
                else:
                    for op in ops["ix"][: counts[ratio]]:
                        op.run(lib)
                    if counts["pf", ratio] and "pf" in sel:
                        sel[
                            "pf_all" if not counts["cand"] and "pf_all" in sel else "pf"
                        ].run(lib)
                    elif not counts["cand"] and "all" in sel:
                        sel["all"].run(lib)
                    else:
                        sel[counts["radix", ratio]].run(lib)
            if "out" in ops:
                (ops["out_dense"] if dense and "out_dense" in ops else ops["out"]).run(
                    lib
                )
            if ccl:
                self._reduce(ops["p_attn"], s)
            ops["ffn"].run(lib)
            if ccl:
                self._reduce(ops["p_ffn"], s)
            if self.probe is not None:
                self.probe(layer, ops, T)
        if (
            prog.inject is not None
        ):  # the DSpark stages' window keys at the step's positions
            prog.inject.run(lib)
        self._column = False
        if logits is None:
            return T
        prog.head.run(lib)
        if ccl and logits == "all":
            dev.allgather(self.logits, self.logits_all, d.vocab * T)
        elif ccl:  # one column, gathered as a one-column step's logits
            dev.allgather(
                self.logits + 4 * d.vocab * int(logits), self.logits_all, d.vocab
            )
            self._column = True
        return T

    # ------------------------------------------------------------------ DSpark
    def draft(self, anchors: np.ndarray, rows: Sequence[Row]) -> np.ndarray:
        """DSpark's drafts (B, block) for up to DSPARK_ROWS rows: row b's block is its anchor ``anchors[b]`` (the
        token at position ``rows[b].pos0``, not yet processed) and block - 1 noise tokens, through the stages
        (each column against the window keys of the target's last positions before the anchor and its block's
        keys), the head with the last stage's norm, and the Markov steps."""
        B, K, dev, d = len(rows), self.block, self.device, self.d
        if not self.stages or not 0 < B <= DSPARK_ROWS:
            raise ValueError(f"drafts need DSpark and 1 to {DSPARK_ROWS} rows")
        s = Shape(bucket(B * K, False), False)
        tb = self.tables[s]
        k = self._host_turn
        self._host_turn ^= 1
        if self._host_pending[k]:
            dev.wait(_HOST_EVENT + k)
        self._fill_draft(k, tb, np.asarray(anchors, np.int64), rows)
        dev.h2d_async(self.table, self.host[k][: tb.nbytes])
        dev.record(_HOST_EVENT + k)
        self._host_pending[k] = True
        lib, prog = dev.lib, self._draft_program(s)
        ccl = self.nranks > 1
        prog["embed"].run(lib)
        for ops in prog["stages"]:
            ops["attn"].run(lib)
            if ccl:
                self._reduce(ops["p_attn"], s)
            ops["ffn"].run(lib)
            if ccl:
                self._reduce(ops["p_ffn"], s)
        prog["head"].run(lib)
        if ccl:
            dev.allgather(self.logits, self.logits_all, d.vocab * s.T)
        prog["markov"][B].run(lib)
        dev.sync()
        return dev.d2h(self.io["dsp_out"], (K, B), np.int32).T.astype(np.int64)

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

    def _fill_draft(
        self, k: int, tb: _Tables, anchors: np.ndarray, rows: Sequence[Row]
    ) -> None:
        """A draft step's tables: row b's block in columns b block .. b block + block - 1 at positions pos0 ..
        (their RoPE), each column's window slot over the stages' pages of the target's last positions before
        pos0 (the block's own keys are not stored), and the block keys' columns and slots."""
        cfg, s, v = self.cfg, tb.shape, tb.v[k]
        T, K, nh, B = s.T, self.block, self.d.heads, len(rows)
        n = B * K
        self.host[k][: tb.nbytes] = 0
        row_of, i_of = np.repeat(np.arange(B), K), np.tile(np.arange(K), B)
        start = np.array([r.pos0 for r in rows], np.int64)
        tok = np.full(n, cfg.dspark_noise_token_id, np.int64)
        tok[i_of == 0] = anchors
        v["tok"][:n] = tok
        v["meta"][:] = (int(start[0]), n, 0, 0)
        pos = np.zeros(T, np.int64)
        pos[:n] = start[row_of] + i_of
        for r in (0, 1):
            cs = self._cs(r, pos)
            v[f"cs{r}"][:] = cs
            v[f"cs{r}_inv"][:, :64] = cs[:, :64]
            v[f"cs{r}_inv"][:, 64:] = -cs[:, 64:]
        q = (
            start[row_of] - 1
        )  # every column's window ends at the position before its anchor
        key0 = np.maximum(q - (cfg.sliding_window - 1), 0) // PAGE_TOKENS * PAGE_TOKENS
        cols, zeros = np.arange(n), np.zeros(n, np.int64)
        slots = v["slots"]
        slots[:, 0] = -1
        slots[:n] = np.stack(
            [zeros, nh * cols, q, key0, start[row_of], zeros, zeros, np.full(n, cfg.sliding_window)], 1
        )  # fmt: skip
        pages = s.win_keys // PAGE_TOKENS
        lat0 = np.full((s.win_slots, pages), -1, np.int64)
        for col in range(n):
            r = rows[row_of[col]]
            idx = key0[col] // PAGE_TOKENS + np.arange(pages) - r.window_first
            ok = (
                (idx >= 0)
                & (idx < len(r.window))
                & (idx + r.window_first <= q[col] // PAGE_TOKENS)
            )
            ok &= q[col] >= 0
            lat0[col] = np.where(ok, np.asarray(r.window)[np.where(ok, idx, 0)], -1)
        v["lat"][:] = -1
        v["rows"][:] = -1  # nothing stored
        for st in range(self.stages):
            layer = self.n_layers + st
            v["lat"][layer] = np.where(lat0 >= 0, lat0 + self._layer_rows[layer], -1)
        v["dsp_bidx"][: n * K] = (row_of[:, None] * K + np.arange(K)).ravel()
        ls = v["dsp_lslots"]
        ls[:, 0] = -1
        ls[:n] = np.stack(
            [zeros, nh * cols, np.full(n, K), zeros, np.full(n, K), cols, zeros, zeros],
            1,
        )
        v["dsp_anchor"][:B] = anchors

    def _reduce(self, partial: int, s: Shape) -> None:
        """Sum a sub-layer's partial over the ranks, or all-gather it for the next graph to sum."""
        count = self.d.hidden * s.T
        if self.d.gathered(s.T):
            self.device.allgather(partial, self.pall[self.pbuf.index(partial)], count)
        else:
            self.device.allreduce(partial, count)

    # ------------------------------------------------------------------ tables
    def _cs(self, r: int, positions: np.ndarray) -> np.ndarray:
        """cs_table(positions, freqs[r]); for many positions, rows of the precomputed table"""
        positions = np.asarray(positions, np.int64)
        if len(positions) <= 64:
            return cs_table(positions, self.freqs[r])
        return self._cs_all[r][positions]

    def _fill(
        self, k: int, tb: _Tables, tokens, positions, rows: Sequence[Row]
    ) -> dict[int, int]:
        """Write a step's tables into host copy ``k``; returns the index launches per ratio."""
        cfg, geo, s, v = self.cfg, self.geo, tb.shape, tb.v[k]
        T, n, nh, R = s.T, len(positions), self.d.heads, len(rows)
        for a, b in tb.zero:
            self.host[k][a:b] = 0
        if tokens is not None:
            v["tok"][:n] = tokens
        v["meta"][:] = (int(positions[0]), n, 0, 0)
        pos = np.zeros(T, np.int64)
        pos[:n] = positions
        for r in (0, 1):
            cs = self._cs(r, pos)
            v[f"cs{r}"][:] = cs
            v[f"cs{r}_inv"][:, :64] = cs[:, :64]
            v[f"cs{r}_inv"][:, 64:] = -cs[:, 64:]
        # each row's blocks, padded: WINDOW rows and ids from logical block wfirst, HISTORY rows from block 0
        wlen = max(len(r.window) for r in rows)
        wrow, wid = np.zeros((R, wlen), np.int64), np.zeros((R, wlen), np.int64)
        hist = np.full(
            (R, max(self.pages, max(len(r.history) for r in rows))), -1, np.int64
        )
        wfirst, pos0 = np.empty(R, np.int64), np.empty(R, np.int64)
        prev_id = np.zeros(R, np.int64)  # the WINDOW block id of position pos0 - 1
        col_row = np.full(T, -1, np.int64)
        slot_row, slot_b0 = [], []
        for k, r in enumerate(rows):
            wrow[k, : len(r.window)], wid[k, : len(r.window)] = r.window, r.window_ids
            hist[k, : len(r.history)] = r.history
            wfirst[k], pos0[k] = r.window_first, r.pos0
            if r.pos0:
                prev_id[k] = r.window_ids[(r.pos0 - 1) // PAGE_TOKENS - r.window_first]
            col_row[r.column : r.column + r.real] = k
            for b0 in range(0, r.real, s.qb):
                slot_row.append(k)
                slot_b0.append(b0)
        cols = np.flatnonzero(col_row >= 0)
        live = col_row >= 0
        cr, cp = col_row[cols], pos[cols]
        block = cp // PAGE_TOKENS - wfirst[cr]
        win = np.zeros(T, np.int64)
        win_id = np.zeros(T, np.int64)
        win[cols], win_id[cols] = wrow[cr, block], wid[cr, block]
        last = np.array([r.pos0 + r.real - 1 for r in rows], np.int64)
        last_col = np.array([r.column + r.real - 1 for r in rows], np.int64)
        # window slots, and every layer's store rows and window pages
        sr, sb = np.array(slot_row, np.int64), np.array(slot_b0, np.int64)
        S, pages = len(sr), s.win_keys // PAGE_TOKENS
        q0 = pos0[sr] + sb
        key0 = np.maximum(q0 - (cfg.sliding_window - 1), 0) // PAGE_TOKENS * PAGE_TOKENS
        col0 = np.array([rows[k].column for k in slot_row], np.int64) + sb
        slots = v["slots"]
        slots[:, 0] = -1
        slots[:S] = np.stack(
            [np.zeros(S, np.int64), nh * col0, q0, key0, last[sr] + 1, np.zeros(S, np.int64),
             np.zeros(S, np.int64), np.full(S, cfg.sliding_window)], 1,
        )  # fmt: skip
        pg = key0[:, None] // PAGE_TOKENS + np.arange(pages)
        ok = pg <= last[sr, None] // PAGE_TOKENS
        lat0 = np.full((s.win_slots, pages), -1, np.int64)
        lat0[:S] = np.where(
            ok, wrow[sr[:, None], np.where(ok, pg - wfirst[sr, None], 0)], -1
        )
        off = self._layer_rows[:, None, None]
        v["lat"][:] = np.where(lat0 >= 0, lat0[None] + off, -1)
        slot_pos = cp % PAGE_TOKENS
        rows0 = np.full((T, 8), -1, np.int64)
        rows0[cols, 0] = win[cols] + slot_pos
        rows0[cols, 3] = win[cols] + PAGE_TOKENS
        rows0[cols, 4] = slot_pos
        shift = np.zeros((T, 8), np.int64)
        shift[cols, 0] = shift[cols, 3] = 1
        np.add(rows0[None], shift[None] * off, out=v["rows"], casting="unsafe")
        # compressor: ratio-2 groups (completed at odd positions) and pending tokens, ratio-1 entries
        odd = cols[cp % 2 == 1]
        gr = col_row[odd]
        inside = pos[odd] - 1 >= pos0[gr]
        G2 = len(odd)
        entries2 = np.full(s.groups2, -1, np.int64)
        entries2[:G2] = pos[odd] // 2
        if geo.pending_owners:
            v["cs_g2"][:] = self._cs(1, np.where(entries2 >= 0, 2 * entries2, 0))
        ends = last % 2 == 0
        npend = len(geo.pending_owners)
        for i, o in enumerate(geo.pending_owners):
            groups = v[f"cmp_groups.{o}"]
            groups[:G2, 0] = np.where(inside, odd - 1, -1)
            groups[:G2, 1] = odd
            groups[:G2, 2] = np.where(inside, 0, prev_id[gr] * npend + i)
            pend = v[f"cmp_rows.{o}"]
            pend[:] = -1
            pend[last_col[ends]] = win_id[last_col[ends]] * npend + i
        if len(geo.pending_owners) < len(geo.owners):
            v["cs_g1"][:] = self._cs(1, np.where(live, pos, 0))
        for o in geo.owners:
            ratio, op, ooff = (
                cfg.compress_ratios[o],
                geo.owner_pages(o),
                geo.owner_offset(o),
            )
            store = v[f"cmp_store.{o}"]
            store[:] = -1
            g, j, er = (
                (np.arange(G2), entries2[:G2], gr) if ratio > 1 else (cols, cp, cr)
            )
            P = j // PAGE_TOKENS
            page = hist[er, P // op] + ooff + PAIR_ROWS * (P % op)
            e = j % PAGE_TOKENS
            store[g, 0] = page + e
            store[g, 2] = (
                4 * (page + PAGE_ROWS) + e
            )  # FP8 index key: unit e of the index page
            store[g, 5] = page + PAGE_ROWS + INDEX_ROWS - 1  # its scale: row 32, f32 e
            store[g, 6] = e
            store[g, 3] = page + PAGE_TOKENS
            store[g, 4] = e
        # indexer: visible entries, compressed pages, index launches
        counts = {"sparse": False, "cand": True}
        lanes = np.arange(INDEX_KEYS // PAGE_TOKENS)
        for ratio in (1, 2):
            v[f"ix_vis{ratio}"][:] = np.where(live, (pos + 1) // ratio, 0)
            counts["radix", ratio] = (
                not s.prefill and int(v[f"ix_vis{ratio}"].max()) > RADIX_MIN
            )
            vis = v[f"ix_vis{ratio}"]
            counts["pf", ratio] = bool(
                s.prefill
                and PREFILTER_BLOCKS
                and np.all(vis[live] > max(PREFILTER_MIN, 8 * PREFILTER_BLOCKS))
            )
            if counts["pf", ratio]:
                # blocks of 8 entries, in whole groups of 64 for ix_thr; the packed entries
                nb = -(-vis // 8)
                v[self._visb(ratio)][:] = np.minimum(-(-nb // 64) * 64, self.kmax // 8)
                v["ix_visc"][:] = np.where(live, 8 * PREFILTER_BLOCKS, 0)
            counts["dense", ratio] = self.dense(s) and bool(
                np.all(vis[live] <= DENSE_KEYS)
            )
            if counts["dense", ratio]:
                # a slot per query block over entries 0 .. klen - 1 (what its last column sees), and the
                # owners' latent pages of those entries
                kl = np.minimum(q0 + s.qb, last[sr] + 1) // ratio
                cslots = v[f"cslots{ratio}"]
                cslots[:] = 0
                cslots[:, 0] = -1
                z = np.zeros(S, np.int64)
                cslots[:S] = np.stack(
                    [z, nh * col0, q0, z, kl, col0, np.full(S, ratio), z], 1
                )
                pg = np.arange(DENSE_KEYS // PAGE_TOKENS)
                ok = pg[None, :] * PAGE_TOKENS < kl[:, None]
                for o in geo.owners:
                    if cfg.compress_ratios[o] != ratio:
                        continue
                    op, ooff = geo.owner_pages(o), geo.owner_offset(o)
                    lat = v[f"cmp_lat.{o}"]
                    lat[:] = -1
                    lat[:S] = np.where(
                        ok,
                        hist[sr[:, None], np.where(ok, pg // op, 0)]
                        + ooff
                        + PAIR_ROWS * (pg % op),
                        -1,
                    )
            if (
                cfg.candidate_source_layer >= 0
                and ratio == cfg.compress_ratios[cfg.candidate_source_layer]
            ):
                # candidate blocks of 8 entries, in whole groups of 64 for ix_thr
                vis = v[f"ix_vis{ratio}"]
                nb = -(-vis // 8)
                v["ix_visb"][:] = np.minimum(-(-nb // 64) * 64, self.kmax // 8)
                # every live column sees more than kc entries: index sources after the candidate source score
                # only their columns' candidate entries
                counts["cand"] = int(vis.max()) > cfg.candidate_topk_blocks * 8
                sparse = self.ixc_launches(s) > 0 and bool(np.all(vis[live] > self.kc))
                counts["sparse"] = sparse
                if sparse:
                    v["ix_visc"][:] = np.where(live, self.kc, 0)
                    lc = np.flatnonzero(live)
                    chunks = self.kc // INDEX_KEYS
                    rec = np.zeros((len(lc) * chunks, 8), np.int64)
                    rec[:, 1] = L.INDEX_HEADS * np.repeat(lc, chunks)
                    # query position kc - 1: every compact entry is visible to the slot
                    rec[:, 2] = self.kc - 1
                    rec[:, 3] = INDEX_KEYS * np.tile(np.arange(chunks), len(lc))
                    rec[:, 4] = self.kc
                    rec[:, 5] = np.repeat(lc, chunks)
                    rec[:, 6] = 1
                    n = self.ixc_launches(s)
                    table = np.zeros((n * s.ix_slots, 8), np.int64)
                    table[:, 0] = -1
                    table[: len(rec)] = rec
                    counts["ixc"] = -(-len(rec) // s.ix_slots)
                    for j in range(n):
                        v[f"ixc.{j}"][:] = table[j * s.ix_slots : (j + 1) * s.ix_slots]
            per = HISTORY_TOKENS // ratio // PAGE_TOKENS  # pages per HISTORY block
            pr = np.where(
                hist[:, :, None] >= 0, hist[:, :, None] + PAIR_ROWS * np.arange(per), -1
            )
            blocks = v[f"cmp_blocks{ratio}"]
            blocks[~live] = -1
            if R == 1:  # every column is the one row's (a broadcast, not a gather)
                blocks[cols] = pr.reshape(R, -1)[0, : self.pages]
            else:
                blocks[cols] = pr.reshape(R, -1)[cr, : self.pages]
            rec = []  # (row, b0, q0, key0, klen)
            for k, r in enumerate(rows):
                seen_all = (r.pos0 + r.real) // ratio
                for b0 in range(0, r.real, s.qb):
                    q = r.pos0 + b0
                    seen = min((q + s.qb) // ratio, seen_all)
                    rec.extend(
                        (k, b0, q, key, seen_all) for key in range(0, seen, INDEX_KEYS)
                    )
            launches = -(-len(rec) // s.ix_slots)
            if launches > self.ix_launches(s, ratio):
                raise ValueError("a step's index launches exceed the table capacity")
            counts[ratio] = launches
            if not rec:
                continue
            rk, rb, rq, rkey, rlen = np.array(rec, np.int64).T
            col = np.array([rows[k].column for k in rk], np.int64) + rb
            table = np.zeros((launches * s.ix_slots, 8), np.int64)
            table[:, 0] = -1
            table[: len(rec)] = np.stack(
                [
                    np.zeros_like(rk),
                    L.INDEX_HEADS * col,
                    rq,
                    rkey,
                    rlen,
                    col,
                    np.full_like(rk, ratio),
                    np.zeros_like(rk),
                ],
                1,
            )
            P = rkey[:, None] // PAGE_TOKENS + lanes
            ok = (
                lanes
                < np.minimum(len(lanes), -(-(rlen - rkey) // PAGE_TOKENS))[:, None]
            )
            owners = [o for o in geo.owners if cfg.compress_ratios[o] == ratio]
            for o in owners:
                op, ooff = geo.owner_pages(o), geo.owner_offset(o)
                pages_o = np.full((launches * s.ix_slots, len(lanes)), -1, np.int64)
                pages_o[: len(rec)] = np.where(
                    ok,
                    hist[rk[:, None], np.where(ok, P // op, 0)]
                    + ooff
                    + PAIR_ROWS * (P % op)
                    + PAGE_ROWS,
                    -1,
                )
                for j in range(launches):
                    v[f"ix{ratio}.{j}.{o}"][:] = pages_o[
                        j * s.ix_slots : (j + 1) * s.ix_slots
                    ]
            for j in range(launches):
                v[f"ix{ratio}.{j}"][:] = table[j * s.ix_slots : (j + 1) * s.ix_slots]
        # Engram
        if self.engram_layers and self.hashed(s):
            xor = self._engram_older(rows, T, tokens)
            for layer in self.engram_layers:
                k = self.cfg.engram_layer_ids.index(layer)
                v[f"eg_xh.{layer}"][:, :J] = xor[:, k, :, 0]
                v[f"eg_xl.{layer}"][:, :J] = xor[:, k, :, 1]
        elif self.engram_layers:
            ids = self._engram_ids(rows, tokens, T)
            for layer in self.engram_layers:
                k = self.cfg.engram_layer_ids.index(layer)
                if self.engram == "card":
                    v[f"eg.{layer}"][:] = ids[:, k]
                else:
                    emb = v[f"eg.{layer}"]
                    for r in rows:
                        c = slice(r.column, r.column + r.real)
                        emb[c] = engram_rows(
                            self.src, layer, ids[c, k].reshape(-1)
                        ).reshape(r.real, -1)
        return counts

    def _engram_ids(
        self, rows: Sequence[Row], tokens: np.ndarray, T: int
    ) -> np.ndarray:
        """(T, Engram layers, J) i32 hashed rows of every column (padding: row 0)"""
        h = self._engram_hash()
        n = self.cfg.engram_max_ngram_size
        out = np.zeros((T, len(self.cfg.engram_layer_ids), J), np.int32)
        windows = np.zeros(
            (T, n), np.int64
        )  # each column's token and those before it, newest first
        blocked = np.zeros((T, n), bool)
        back = np.arange(n)
        for r in rows:
            c = slice(r.column, r.column + r.real)
            seq = np.concatenate(
                [np.asarray(r.before, np.int64), np.asarray(tokens[c], np.int64)]
            )
            at = len(r.before) + np.arange(r.real)[:, None] - back
            windows[c] = seq[np.maximum(at, 0)]
            blocked[c] = (r.pos0 + np.arange(r.real))[:, None] < back
        cols = np.concatenate([np.arange(r.column, r.column + r.real) for r in rows])
        out[cols] = h.hash(
            np.where(blocked[cols], h.pad_id, h.token_map[windows[cols]])
        )
        return out

    def _engram_older(self, rows: Sequence[Row], T: int, tokens=None) -> np.ndarray:
        """(T, Engram layers, J, 2) i32: per decode column and lane 8 (n - 2) + head, the XOR of the products of
        the n - 1 tokens before the column's (``Row.before``, then the row's earlier columns of the step:
        ``tokens``, needed when a row has more than one column), as (hi, lo); ``eg_hash`` adds the column's
        own."""
        h = self._engram_hash()
        n = self.cfg.engram_max_ngram_size
        older = np.full((T, n - 1), h.pad_id, np.int64)
        for r in rows:
            if r.real != 1 and tokens is None:
                raise ValueError(
                    "a step with several columns per row needs its tokens on the host"
                )
            seen = np.asarray(r.before, np.int64)
            for j in range(r.real):
                if j:
                    seen = np.append(seen, tokens[r.column + j - 1])
                before = seen[::-1][: n - 1]  # newest first
                older[r.column + j, : len(before)] = h.token_map[before]
        xor = h.older(older)  # (T, layers, n - 1)
        heads = self.cfg.engram_n_heads
        return split64(np.repeat(xor, heads, axis=2))
