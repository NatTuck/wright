"""Qwen3.8-Flash-Next forward steps on one Gaudi card (one tensor-parallel rank; the experts and the n-gram table
are sharded over the ranks).

A step runs T columns: one sequence's prefill chunk (T a multiple of 64 tokens), or one decode token for each of up
to B sequences (the decode buckets). Per step the host fills one image with every layer's tables (tokens,
positions, sampler parameters, the n-gram rows, delta-net state bases, K/V page rows, slot launches, QSA key rows,
PLE state bases), uploads it in one copy and runs the step's program (``_Program``: every launch with its bindings
and the collective after it, prepared once per step shape) natively: embed, per layer the module (delta net, or
the PLE at its layer then attention pre / slot launches / post) and the MoE, then the head with the sampler. The streams, inject gates and partial sums ping-pong between
buffers (``layers``). The caches live in the Engine's paged arena (``cache``): a row names its KV blocks (128
tokens) and STATE blocks (4096 tokens); a step reads the states left after position pos0 - 1 and writes the states
after its last position.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resin.backend.capacity import plan_cache_capacity
from resin_models_gaudi.common.attention import SlotRow, kv_rows, plan_slots
from resin_models_gaudi.common.layers import (
    ARENA_BLOCK_ROWS,
    GDN_CHUNK,
    QUERY_BLOCK,
    SLOT_KEYS,
    attn_slots_graph,
)
from resin_models_gaudi.common.sampling import PARAMS, TOP_MAX
from resin_models_gaudi.loading.convert import rope_table, sigma_inv, sigma_perm
from resin_models_gaudi.loading.progress import progress, weight_file_bytes
from resin_models_gaudi.loading.upload import WeightLoader
from resin_models_gaudi.qwen4_exp import layers as L
from resin_models_gaudi.qwen4_exp.cache import (
    PAGE_TOKENS,
    ROW_BYTES,
    STATE_TOKENS,
    CacheGeometry,
)
from resin_models_gaudi.qwen4_exp.checkpoint import HCR, PleTable, Qwen4ExpCheckpoint
from resin_models_gaudi.qwen4_exp.layers import NS, Dims
from resin_models_gaudi.synapse.graph import Graph, Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import Device, SynapseError

DECODE_BUCKETS = (1, 2, 4, 8, 16)
PREFILL_BUCKETS = (128, 256, 512, 1024, 2048)
SLOTS = 16  # attention slots per launch
SMALL = (
    256  # the small tables' slot
)
WORKSPACE_RESERVE = 2 << 30  # bytes held back for recipe workspaces
CAPACITY_QUANTUM = 4  # slabs
GATHER_TOKENS = 16  # launches up to this wide all-gather the ranks' partials (decode); wider ones all-reduce
HEAD_ALL_MAX = 256  # widest every-column logits read (tests)
SLOT_STRIDE = 4 * (
    SLOTS * 8 + 2 * SLOTS * 16
)  # a slot launch's table: slots, K pages, V pages


def bucket(n: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")


@dataclass(frozen=True, slots=True)
class RowCache:
    """One sequence's cache as a step sees it: its first position, the arena rows of its KV blocks (logical order,
    covering every position the step touches), its STATE blocks (``state_src``: the block holding the states after
    pos0 - 1, -1 for a fresh sequence; ``state_dst``: the block receiving the states after the step's last token;
    ``state_rows`` from logical block ``state_first`` resolve any position), and the two tokens before pos0 (-1:
    none, EOS stands in) for the n-gram rows."""

    pos0: int
    kv_blocks: np.ndarray
    state_src: int
    state_dst: int
    state_first: int = 0
    state_rows: tuple[int, ...] = ()
    history: tuple[int, int] = (-1, -1)

    def state_at(self, position: int) -> int:
        """the STATE block of ``position`` (-1 before the sequence)"""
        if position < 0:
            return -1
        if not self.state_rows:
            return self.state_dst
        return self.state_rows[position // STATE_TOKENS - self.state_first]

    def advanced(self, count: int, tokens: Sequence[int]) -> RowCache:
        """the row ``count`` positions on, ``tokens`` being the last tokens run (at least the last two)"""
        start = self.pos0 + count
        last = [int(t) for t in tokens[-2:]]
        history = (self.history[1], last[-1]) if len(last) == 1 else (last[0], last[1])
        return RowCache(
            start,
            self.kv_blocks,
            self.state_at(start - 1),
            self.state_at(start),
            self.state_first,
            self.state_rows,
            history,
        )


class _Layout:
    """byte offsets of a step shape's tables in the table image: a table of ``rows`` (one per layer) occupies
    ``rows`` slots of ``stride`` bytes (whole 256-byte units)"""

    def __init__(self) -> None:
        self.size = 0
        self.at: dict[str, int] = {}
        self.stride: dict[str, int] = {}

    def add(self, name: str, nbytes: int, rows: int = 1) -> int:
        offset = self.size
        self.at[name] = offset
        self.stride[name] = stride = -(-nbytes // SMALL) * SMALL
        self.size += rows * stride
        return offset

    def of(self, name: str, row: int = 0) -> int:
        return self.at[name] + row * self.stride[name]


_NO_REGISTERS = np.zeros(
    0, np.uint64
)  # a Flash-Next step decides nothing per step in its program
_GLUE: dict[str, object] = {}  # the runtime library's glue, by its path


class _Program:
    """A step's launches in order, each with its slot bindings and the collective after it, run natively as one
    ``resin._resin.GaudiProgram``. A collective is (kind, src, dst, count): kind 1 all-reduces src in place, kind 2
    all-gathers src to dst (count values per rank); kind bit 2 set is bf16."""

    def __init__(self) -> None:
        self.ops: list[tuple[Recipe, dict[str, int], tuple | None]] = []

    def add(
        self, recipe: Recipe, names: dict[str, int], collective: tuple | None
    ) -> None:
        self.ops.append((recipe, names, collective))

    def freeze(self) -> None:
        from resin._resin import GaudiProgram

        native = GaudiProgram(0)
        native.section("step")
        for recipe, names, collective in self.ops:
            native.launch(
                recipe.handle, [recipe.slot(n) for n in names], list(names.values())
            )
            if collective is None:
                continue
            kind, src, dst, count = collective
            bf16 = bool(kind >> 2 & 1)
            if kind & 3 == 1:
                native.allreduce(src, count, bf16=bf16)
            elif kind & 3 == 2:
                native.allgather(src, dst, count, bf16=bf16)
            else:
                raise ValueError(f"{recipe.name}: collective kind {kind}")
        native.finish()
        self.n, self.native = len(self.ops), native

    def run(self, device: Device) -> None:
        path = str(device.libraries.runtime)
        if (glue := _GLUE.get(path)) is None:
            from resin._resin import GaudiGlue

            glue = _GLUE[path] = GaudiGlue(path)
        try:
            self.native.run(glue, _NO_REGISTERS, (0,))
        except RuntimeError as error:  # a failed call, as the device reports one
            raise SynapseError(f"program: {error}") from error


class Qwen4ExpGaudi:
    def __init__(
        self,
        device: Device,
        path: str | Path,
        *,
        layers: int | None = None,
        table: bool = True,
        max_positions: int = 8192,
        parent_capacity: int | None = None,
        memory_utilization: float = 0.9,
        decode_buckets: Sequence[int] = DECODE_BUCKETS,
        prefill_buckets: Sequence[int] = PREFILL_BUCKETS,
        rank: int = 0,
        nranks: int = 1,
        gather_tokens: int = GATHER_TOKENS,
        replicate_dense: bool = False,
        reduce_free: Callable[[int], int] | None = None,
        recipes: RecipeCache | None = None,
        log: Callable[[str], None] = print,
        defer_cache: bool = False,
    ) -> None:
        """``rank`` of ``nranks`` (the HCCL communicator is up); launches of up to ``gather_tokens`` columns
        all-gather the ranks' partials and sum them in the next graph, wider ones all-reduce in place. The
        experts and the n-gram table are always split over the ranks; with ``replicate_dense`` every rank
        holds the whole delta-net, attention and hyper-connection weights (a layer then runs as one launch with
        one collective, after its experts, but every rank computes the whole dense module), otherwise those are split too
        (tensor parallel: two collectives a layer; the default). ``reduce_free`` maps this card's free memory to the arena budget every rank can afford.
        ``defer_cache``: load the weights and the fixed buffers and stop; ``attach`` then makes the cache arena and
        the recipes, which depend on its capacity."""
        self.device, self.log = device, log
        self.rank, self.nranks = rank, nranks
        self.dense_ranks = 1 if nranks == 1 or replicate_dense else nranks
        self.check_overlaps = False  # host check of every launch's bindings (tests)
        self.profile: list[tuple[int, float, float, float]] | None = (
            None  # (T, tables, launches, wait) seconds
        )
        self.ck = ck = Qwen4ExpCheckpoint(path)
        self.cfg = c = ck.config
        self.n_layer = c.num_hidden_layers if layers is None else layers
        self.layer_ids = list(range(self.n_layer))
        self.full = [i for i in self.layer_ids if c.is_full(i)]
        self.linear = [i for i in self.layer_ids if not c.is_full(i)]
        self.ple_layer = next((i for i in c.ple_layers if i < self.n_layer), None)
        self.has_table = table and self.ple_layer is not None
        self.ngram = ck.ngram() if self.ple_layer is not None else None
        self.max_positions = max_positions
        self.decode_buckets = tuple(decode_buckets)
        self.prefill_buckets = tuple(prefill_buckets)
        shard = ck.shard(rank, nranks, dense_ranks=self.dense_ranks)
        self.table_first = 0
        # ---- weights
        t0 = time.time()
        self.recipes = recipes or RecipeCache(device)
        self.loader = WeightLoader(device, self.recipes, min_alloc=SMALL)
        tables: list[PleTable] = []

        def items():
            for name, payload, kind in ck.weights(
                rank,
                nranks,
                layers=set(self.layer_ids),
                table=self.has_table,
                dense_ranks=self.dense_ranks,
            ):
                if isinstance(payload, PleTable):
                    tables.append(payload)
                yield name, payload, kind

        device.begin_load(weight_file_bytes(ck.path, "*.safetensors") // nranks)
        self.loader.load(items())
        device.end_load()
        self.w = self.loader.w
        nbytes = self.loader.nbytes
        table_rows = tables[0].shape[0] if tables else 0
        if tables:
            self.table_first = tables[0].first_row
        device.sync()
        self.d = d = Dims.from_config(
            c,
            shard,
            table_rows=table_rows,
            gather_tokens=gather_tokens if nranks > 1 else 0,
        )
        self.vocab_all = c.vocab_size
        log(
            f"fnext: {nbytes / 1e9:.1f} GB of weights in {time.time() - t0:.1f} s ({self.n_layer} layers)"
        )
        self.geo = CacheGeometry.from_dims(
            d,
            full_layers=len(self.full),
            linear_layers=len(self.linear),
            ple=self.ple_layer is not None,
        )
        # what the weights leave free: a model of its own sizes its arena from it (``attach``), as it did before
        # its buffers moved ahead of the arena
        self._free_after_weights = device.memory_info()[0]
        # ---- buffers
        Tm = max(self.prefill_buckets)
        H, V = d.hidden, d.vocab
        pool = [device.malloc(1 << 20), 0]

        def buf(n: int) -> int:
            """small tables share 256-byte slots of one allocation; big buffers are their own"""
            if n > 4096:
                return device.malloc(n)
            size = -(-n // SMALL) * SMALL
            if pool[1] + size > (1 << 20):
                pool[0], pool[1] = device.malloc(1 << 20), 0
            address = pool[0] + pool[1]
            pool[1] += size
            return address

        def zeros(n: int) -> int:
            a = buf(n)
            device.memset32(a, 0, n // 4 if n > 4096 else -(-n // SMALL) * (SMALL // 4))
            return a

        self.X = [buf(4 * NS * H * Tm) for _ in range(2)]
        self.P = [buf(4 * H * Tm) for _ in range(2)]
        NR = d.ranks
        self.P_all = (
            [buf(4 * H * gather_tokens * NR) for _ in range(2)] if NR > 1 else []
        )
        self.LO = [buf(4 * HCR * Tm) for _ in range(2)]
        self.zx, self.zy, self.zl = (
            zeros(4 * NS * H * Tm),
            zeros(4 * H * max(Tm, gather_tokens * NR)),
            zeros(4 * HCR * Tm),
        )
        self.zkv, self.zcol, self.zik = (
            zeros(4 * (NS + 1) * H * Tm),
            zeros(2 * H),
            zeros(512 * 4),
        )
        self.one = buf(4 * Tm)
        device.h2d(self.one, np.ones(Tm, np.float32))
        self.cs_tab, self.cs_all = buf(512 * max_positions), buf(512 * max_positions)
        rope = rope_table(max_positions, c)
        device.h2d(self.cs_tab, rope)
        device.h2d(self.cs_all, rope)
        self.tok, self.pos, self.meta, self.cols = (
            buf(4 * Tm),
            buf(4 * Tm),
            buf(SMALL),
            buf(4 * Tm),
        )
        self.prow = buf(4 * d.ple_heads * Tm)
        self.samp_params = buf(4 * PARAMS * Tm)
        self.q_all = buf(2 * d.head_dim * d.hpg * Tm * d.kv_groups)
        self.acc_o = buf(4 * d.head_dim * d.hpg * Tm * d.kv_groups)
        self.acc_m, self.acc_l = (
            buf(4 * d.hpg * Tm * d.kv_groups),
            buf(4 * d.hpg * Tm * d.kv_groups),
        )
        self.attn_p = buf(4 * (d.attn_proj + d.ix_proj) * Tm)
        self.logits = buf(4 * V * Tm)
        self.logits_all = buf(4 * V * HEAD_ALL_MAX * NR) if NR > 1 else 0
        self.ple_e = buf(2 * 256 * d.ple_heads * Tm) if NR > 1 and self.has_table else 0
        self.tok_out, self.lp_out = buf(4 * Tm), buf(4 * Tm)
        self.top_idx, self.top_lp = buf(4 * TOP_MAX * Tm), buf(4 * TOP_MAX * Tm)
        self.sigma_idx = buf(4 * d.ssm_rows)
        device.h2d(self.sigma_idx, sigma_perm(d.ssm_rows).astype(np.int32))
        sinks = (
            "sink_kv",
            "sink_acc",
            "sink_slots",
            "sink_conv",
            "sink_ssm",
            "sink_ix",
            "sink_kb",
            "sink_ple",
        )
        self.sinks = {n: buf(SMALL) for n in sinks}
        # ---- the table image: every table of a step at a fixed offset, one pinned host copy uploaded per step
        image_bytes = self._layouts()
        self.tables_dev = device.malloc(image_bytes)
        self.image = device.host_buffer(image_bytes // 4, np.int32)
        self.image[:] = 0
        self.programs: dict[tuple, _Program] = {}
        self._native = self._native_fill()
        self._native_check = (
            self._native is not None and os.environ.get("RESIN_FNEXT_NATIVE") == "check"
        )
        self.image_check = (
            np.zeros(image_bytes // 4, np.int32)
            if self._native_check
            else np.zeros(0, np.int32)
        )
        self.parent_capacity: int | None = None
        self._memory_utilization, self._reduce_free = memory_utilization, reduce_free
        if not defer_cache:
            self.attach(parent_capacity)

    def attach(
        self,
        parent_capacity: int | None = None,
        *,
        slab_rows: int | None = None,
        arena: int | None = None,
    ) -> None:
        """The cache arena of ``parent_capacity`` slabs (None: sized from the memory that the weights left, less
        the recipes' workspace) and the recipes. ``slab_rows``: slabs of that many rows (another model's slab
        size). ``arena``: the address of a cleared arena of ``parent_capacity`` such slabs plus the null slab,
        which other models of the card share (the caller allocates and frees it)."""
        if self.parent_capacity is not None:
            raise RuntimeError("the cache arena is attached already")
        device = self.device
        if slab_rows is not None:
            self.geo = self.geo.with_slab_rows(slab_rows)
        g = self.geo
        if parent_capacity is None:
            free, total = self._free_after_weights, device.memory_info()[1]
            if self._reduce_free is not None:
                free = self._reduce_free(free)
            planned = plan_cache_capacity(
                total,
                free,
                self._memory_utilization,
                g.slab_bytes,
                overhead_bytes=WORKSPACE_RESERVE,
            ).parent_capacity
            parent_capacity = max(1, planned // CAPACITY_QUANTUM * CAPACITY_QUANTUM)
        self.parent_capacity = parent_capacity
        self.blocks = g.arena_blocks(parent_capacity)
        self.arena_bytes = self.blocks * ARENA_BLOCK_ROWS * ROW_BYTES
        if arena is None:
            self.arena = device.malloc(self.arena_bytes)
            device.memset32(self.arena, 0, self.arena_bytes // 4)
        else:
            self.arena = arena
        self._compile()

    def _compile(self) -> None:
        device, d = self.device, self.d
        NR, max_positions = d.ranks, self.max_positions
        # ---- recipes
        progress.building_graphs()
        graphs: dict[tuple, Graph] = {}
        self.merged = (
            d.dense_ranks == 1
        )  # a layer's module and experts in one launch (no collective between)
        for T in (*self.decode_buckets, *self.prefill_buckets):
            decode = T in self.decode_buckets
            after_moe, after_dense = d.parts_after("moe", T), d.parts_after("dense", T)
            blocks, P = self.blocks, max_positions
            graphs["embed", T] = L.embed_graph(device, d, T)
            if self.linear and self.merged:
                graphs["layer_gdn", T] = L.layer_graph(
                    device, d, T, blocks, P, kind="gdn", decode=decode, parts=after_moe
                )
            elif self.linear:
                graphs["gdn", T] = L.gdn_graph(
                    device, d, T, blocks, decode=decode, parts=after_moe
                )
            if self.full and decode and self.merged:
                graphs["layer_attn", T] = L.layer_graph(
                    device,
                    d,
                    T,
                    blocks,
                    P,
                    kind="attn",
                    decode=True,
                    slots=self._slot_count(T),
                    parts=after_moe,
                )
            elif self.full and decode:
                graphs["attn", T] = L.attn_decode_graph(
                    device, d, T, blocks, P, self._slot_count(T), parts=after_moe
                )
            elif self.full:
                graphs["attn_pre", T] = L.attn_pre_graph(
                    device, d, T, blocks, P, parts=after_moe
                )
                graphs["attn_post", T] = L.attn_post_graph(
                    device, d, T, blocks, P, decode=decode
                )
                qb = min(T, QUERY_BLOCK)
                graphs["slots", T] = attn_slots_graph(
                    device, d.qwen_dims(), T, SLOTS, blocks, qb
                )
            if self.ple_layer is not None:
                gathered = NR > 1 and self.has_table
                graphs["ple", T] = L.ple_graph(
                    device,
                    d,
                    T,
                    blocks,
                    decode=decode,
                    gathered=gathered,
                    parts=after_moe,
                )
                if gathered:
                    graphs["ple_embed", T] = L.ple_embed_graph(device, d, T)
            if not self.merged or (self.full and not decode):
                graphs["moe", T] = L.moe_graph(device, d, T, parts=after_dense)
            columns = T if decode else 1
            graphs["head", T] = L.head_graph(
                device, d, T, columns, sample=NR == 1, parts=after_moe
            )
            if NR > 1 and ("sample", columns) not in graphs:
                graphs["sample", columns] = L.sample_graph(device, d, columns)
            if (
                not decode and T <= HEAD_ALL_MAX
            ):  # every column's logits, no sampler (tests)
                graphs["head_all", T] = L.head_graph(
                    device, d, T, T, sample=False, parts=after_moe
                )
        compiled = self.recipes.compile_many(list(graphs.values()))
        self.r: dict[tuple, Recipe] = dict(zip(graphs.keys(), compiled, strict=True))
        for graph in graphs.values():
            graph.destroy()
        self.compiles_at_start = device.compile_count

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start

    # ------------------------------------------------------------------ tables
    def _slot_count(self, T: int) -> int:
        """slots of a decode step (one launch): every row's key chunks x groups"""
        return T * -(-self.max_positions // SLOT_KEYS) * self.d.kv_groups

    def _slot_launches(self, T: int, decode: bool) -> int:
        """slot launches a prefill step of this shape may need: the query blocks x key chunks x groups"""
        if decode:
            return 1
        chunks = -(-self.max_positions // SLOT_KEYS)
        return -(-(-(-T // QUERY_BLOCK) * chunks * self.d.kv_groups) // SLOTS)

    def _slot_table_bytes(self, T: int, decode: bool) -> int:
        """a layer's slot table: decode, one group of S slots (8, S) (its page tables are tables of their own);
        prefill, a 16-slot table per launch"""
        if decode:
            return 4 * 8 * self._slot_count(T)
        return self._slot_launches(T, decode) * SLOT_STRIDE

    def _layouts(self) -> int:
        """Every step shape's table layout (``layouts``) and the rows of each per-layer table -> the image's bytes"""
        self.layouts = {
            (T, decode): self._layout(T, decode)
            for T, decode in [(T, True) for T in self.decode_buckets]
            + [(T, False) for T in self.prefill_buckets]
        }
        self.layouts_rows = {
            n: len(self.linear)
            for n in ("gdn_bases", "conv_dst", "conv_src", "ssm_src", "ssm_dst")
        }
        per_full = (
            "kv_rows",
            "ix_rows",
            "kb_rows",
            "kb_tail",
            "slots",
            "k_pages",
            "v_pages",
        )
        self.layouts_rows.update({n: len(self.full) for n in per_full})
        return max(lay.size for lay in self.layouts.values())

    def _layout(self, T: int, decode: bool) -> _Layout:
        lay = _Layout()
        lay.add("tok", 4 * T)
        lay.add("pos", 4 * T)
        lay.add("meta", 16)
        lay.add("cols", 4 * T)
        lay.add("samp_params", 4 * PARAMS * T)
        if self.ple_layer is not None:
            lay.add("prow", 4 * self.d.ple_heads * T)
            lay.add("ple_src", 4 * T)
            lay.add("ple_dst", 4 * T)
        NL, NF = (
            len(self.linear),
            len(self.full),
        )  # a row per delta-net / attention layer
        if decode:  # gdn_bases (4, T) and conv_dst (T,)
            lay.add("gdn_bases", 16 * T, NL)
            lay.add("conv_dst", 4 * T, NL)
        else:  # (1,) each
            for n in ("conv_src", "ssm_src", "conv_dst", "ssm_dst"):
                lay.add(n, 4, NL)
        lay.add("kv_rows", 32 * T, NF)
        lay.add("ix_rows", 4 * T, NF)
        lay.add("kb_rows", 16 * T if decode else 16 * (T // 4 + 1), NF)
        if decode:
            lay.add("kb_tail", 4 * T, NF)
        lay.add("slots", self._slot_table_bytes(T, decode), NF)
        if decode:
            PC = SLOT_KEYS // PAGE_TOKENS
            lay.add("k_pages", 4 * PC * self._slot_count(T), NF)
            lay.add("v_pages", 4 * PC * self._slot_count(T), NF)
        return lay

    def _ngram_rows(
        self, columns: Sequence[tuple[RowCache, int, Sequence[int]]], T: int
    ) -> np.ndarray:
        """(T heads,) this rank's n-gram table rows of every column (padding, other ranks' rows -> the zero row)"""
        d = self.d
        zero = max(d.table_rows - 1, 0)
        out = np.full((T, d.ple_heads), zero, np.int32)
        if self.ngram is None or not d.table_rows:
            return out.reshape(-1)
        eos = self.cfg.eos_token_id
        for row, column, ids in columns:
            history = [eos if t < 0 else t for t in row.history]
            r = self.ngram.rows([int(t) for t in ids], advance=False, history=history)
            local = r - self.table_first
            out[column : column + len(ids)] = np.where(
                (local >= 0) & (local < zero), local, zero
            )
        return out.reshape(-1)

    def _kb_rows(self, row: RowCache, layer: int, blocks: np.ndarray) -> np.ndarray:
        """block-key page rows (4 per block) of the QSA blocks ``blocks`` (absolute block numbers)"""
        _, kb = self.geo.ix_pages(row.kv_blocks, layer)
        first = 4 * blocks
        # (a prefill bucket's trailing blocks may lie past the row's leased pages; the caller masks them)
        base = kb[np.minimum(first // PAGE_TOKENS, len(kb) - 1)] + first % PAGE_TOKENS
        return (base[:, None] + np.arange(4)[None, :]).reshape(-1)

    # ------------------------------------------------------------------ programs
    def _weights(self, recipe: Recipe, layer: int | None) -> dict[str, int]:
        out = {}
        prefix = f"l{layer}." if layer is not None else ""
        for name in recipe.persistent:
            if prefix and prefix + name in self.w:
                out[name] = self.w[prefix + name]
            elif name in self.w:
                out[name] = self.w[name]
        return out

    def _program(self, T: int, decode: bool, all_columns: bool) -> _Program:
        """Every launch of a step of this shape with its bindings and collectives prepared once: the tables sit
        at fixed offsets of the image and the buffers ping-pong the same way in every step."""
        key = (T, decode, all_columns)
        if (prog := self.programs.get(key)) is not None:
            return prog
        d, lay = self.d, self.layouts[T, decode]
        H, NR = d.hidden, d.ranks

        def tab(name: str, row: int = 0) -> int:
            return self.tables_dev + lay.of(name, row)

        prog = _Program()

        def op(
            key: tuple, layer: int | None, binds: dict[str, int], collective=None
        ) -> None:
            recipe = self.r[key]
            names = self._weights(recipe, layer)
            for n, a in binds.items():
                if n in recipe.persistent:
                    names[n] = a
            missing = set(recipe.persistent) - set(names)
            if missing:
                raise ValueError(f"{recipe.name}: unbound {sorted(missing)}")
            if self.check_overlaps:
                spans = sorted(
                    (a, a + recipe.persistent[n].nbytes, n) for n, a in names.items()
                )
                for (a0, e0, n0), (a1, _e1, n1) in zip(spans, spans[1:], strict=False):
                    if a1 < e0:
                        raise ValueError(
                            f"{recipe.name}: {n0} [{a0:#x}, {e0:#x}) overlaps {n1} at {a1:#x}"
                        )
            prog.add(recipe, names, collective)

        def reduced(partial: int, module: str):
            """the collective after ``module`` ("dense" / "moe") and the address the next graph reads the
            partials from: nothing when the module's weights are whole on this rank"""
            sharded = NR > 1 if module == "moe" else d.dense_ranks > 1
            if not sharded:
                return None, partial
            if d.gathered(T):
                gathered = self.P_all[0] if partial == self.P[0] else self.P_all[1]
                return (2, partial, gathered, H * T), gathered
            return (1, partial, partial, H * T), partial

        fixed = {
            "zx": self.zx, "zy": self.zy, "zl": self.zl, "zkv": self.zkv, "zcol": self.zcol, "zik": self.zik,
            "one": self.one, "cs_tab": self.cs_tab, "cs_all": self.cs_all, "arena": self.arena,
            "q_all": self.q_all, "acc_o": self.acc_o, "acc_m": self.acc_m, "acc_l": self.acc_l,
            "attn_p": self.attn_p, "logits": self.logits, "logits_all": self.logits_all, "ple_e": self.ple_e,
            "sigma_idx": self.sigma_idx, "tok_out": self.tok_out, "lp_out": self.lp_out,
            "top_idx": self.top_idx, "top_lp": self.top_lp, **self.sinks,
            "tok": tab("tok"), "pos": tab("pos"), "meta": tab("meta"), "cols": tab("cols"),
            "samp_params": tab("samp_params"),
        }  # fmt: skip
        if self.ple_layer is not None:
            fixed.update(
                prow=tab("prow"), ple_src=tab("ple_src"), ple_dst=tab("ple_dst")
            )
        X_in, X_out = self.X[0], self.X[1]
        p_in, lo_in = (
            self.zy,
            self.zl,
        )  # layer 0 starts from the embedding: no partial, no gates
        lo_out, p_out = self.LO[0], self.P[0]
        op(("embed", T), None, {**fixed, "X_out": X_in})

        def streams() -> dict[str, int]:
            return {
                **fixed,
                "X": X_in,
                "X_out": X_out,
                "p": p_in,
                "lo": lo_in,
                "lo_out": lo_out,
                "p_out": p_out,
            }

        def advance(collective) -> None:
            nonlocal X_in, X_out, p_in, lo_in, lo_out, p_out
            X_in, X_out = X_out, X_in
            lo_in = lo_out
            lo_out = self.LO[1] if lo_out == self.LO[0] else self.LO[0]
            p_out = self.P[1] if p_out == self.P[0] else self.P[0]

        for i in self.layer_ids:
            if (
                i == self.ple_layer
            ):  # the n-gram embedding into the streams, then the module from zero gates
                if ("ple_embed", T) in self.r:  # the ranks' table rows, summed
                    op(
                        ("ple_embed", T),
                        i,
                        streams(),
                        (1 | 4, self.ple_e, self.ple_e, 256 * d.ple_heads * T),
                    )
                op(("ple", T), i, streams())
                X_in, X_out = X_out, X_in
                p_in, lo_in = self.zy, self.zl
            if i in self.linear:
                nth = self.linear.index(i)
                names = {
                    n: tab(n, nth)
                    for n in (
                        ("gdn_bases", "conv_dst")
                        if decode
                        else ("conv_src", "ssm_src", "conv_dst", "ssm_dst")
                    )
                }
                module = ("layer_gdn" if self.merged else "gdn", T)
            else:
                a = self.full.index(i)
                if decode:
                    names = {
                        n: tab(n, a)
                        for n in (
                            "kv_rows",
                            "slots",
                            "k_pages",
                            "v_pages",
                            "ix_rows",
                            "kb_rows",
                            "kb_tail",
                        )
                    }
                    module = ("layer_attn" if self.merged else "attn", T)
                else:  # prefill attention: three launches, then the experts as their own
                    op(("attn_pre", T), i, {**streams(), "kv_rows": tab("kv_rows", a)})
                    for j in range(self._slot_launches(T, decode)):
                        base = tab("slots", a) + j * SLOT_STRIDE
                        op(
                            ("slots", T),
                            None,
                            {
                                **fixed,
                                "slots": base,
                                "k_pages": base + 4 * SLOTS * 8,
                                "v_pages": base + 4 * SLOTS * 8 + 4 * SLOTS * 16,
                            },
                        )
                    names = {n: tab(n, a) for n in ("ix_rows", "kb_rows")}
                    module = ("attn_post", T)
            merged = module[0].startswith("layer_")
            collective, nxt = reduced(p_out, "moe" if merged else "dense")
            op(module, i, {**streams(), **names}, collective)
            advance(collective)
            p_in = nxt
            if not merged:
                collective, nxt = reduced(p_out, "moe")
                op(("moe", T), i, streams(), collective)
                advance(collective)
                p_in = nxt
        columns = T if all_columns or decode else 1
        gather = (
            (2, self.logits, self.logits_all, d.vocab * columns) if NR > 1 else None
        )
        op(("head_all" if all_columns else "head", T), None, streams(), gather)
        if NR > 1 and not all_columns:
            op(("sample", columns), None, fixed)
        prog.freeze()
        self.programs[key] = prog
        return prog

    def _decode_attention_tables(self, columns, T: int, rows_of) -> None:
        """A decode step's attention tables for every attention layer at once: each row's K/V rows, raw-key row,
        block-key rows and tail row at its position, and its slots (one per key chunk and group) with their page
        tables, all from the first attention layer's pages (every later layer a layer further)."""
        d, g = self.d, self.geo
        NF, G, PC = len(self.full), d.kv_groups, SLOT_KEYS // PAGE_TOKENS
        B = len(columns)
        pos = np.array([row.pos0 for row, _, _ in columns], np.int64)
        npages = max(len(row.kv_blocks) for row, _, _ in columns)
        blocks = np.full(
            (B, npages), -1, np.int64
        )  # (B, pages) KV block rows, -1 past a row's lease
        for b, (row, _, _) in enumerate(columns):
            blocks[b, : len(row.kv_blocks)] = row.kv_blocks
        page = blocks[
            np.arange(B), pos // PAGE_TOKENS
        ]  # the block of each row's position
        off, soff = pos % PAGE_TOKENS, sigma_inv(pos % PAGE_TOKENS)
        groups = np.arange(G) * PAGE_TOKENS
        # K rows (B, G) and V rows (B, G) of the first attention layer (layer offset 0 within the block)
        k0 = page[:, None] + groups[None, :] + off[:, None]
        v0 = page[:, None] + G * PAGE_TOKENS + groups[None, :] + soff[:, None]
        kv0 = np.full((T, 8), -1, np.int64)
        kv0[:B, :G], kv0[:B, 4 : 4 + G] = k0, v0
        # the raw-key page, then the block-key page, after the K and V pages (cache.CacheGeometry.ix_pages)
        raw0 = page + 2 * G * PAGE_TOKENS
        ix0 = np.full(T, -1, np.int64)
        ix0[:B] = raw0 + off
        first = 4 * (
            pos // 4
        )  # the QSA block's first position: the tail rows and the block-key rows
        page_f = blocks[np.arange(B), first // PAGE_TOKENS]
        tail0 = np.full(T, -1, np.int64)
        tail0[:B] = page_f + 2 * G * PAGE_TOKENS + first % PAGE_TOKENS
        kb0 = np.full((T, 4), -1, np.int64)
        kb0[:B] = (page_f + (2 * G + 1) * PAGE_TOKENS + first % PAGE_TOKENS)[
            :, None
        ] + np.arange(4)[None, :]
        # slots: row b, key chunk c (2048 keys), group grp -> [grp, hpg b, pos, 2048 c, pos + 1, b, 0, 0]
        S, chunks = self._slot_count(T), -(-self.max_positions // SLOT_KEYS)
        st = np.full((S, 8), -1, np.int64)
        kp = np.full((S, PC), -1, np.int64)
        vp = np.full((S, PC), -1, np.int64)
        k = 0
        for c in range(chunks):
            live = pos + 1 > c * SLOT_KEYS  # rows that have keys in this chunk
            for grp in range(G):
                for b in np.nonzero(live)[0]:
                    st[k] = (grp, d.hpg * b, pos[b], c * SLOT_KEYS, pos[b] + 1, b, 0, 0)
                    span = blocks[b, c * PC : (c + 1) * PC]
                    kp[k, : len(span)] = np.where(span >= 0, span + groups[grp], -1)
                    vp[k, : len(span)] = np.where(
                        span >= 0, span + G * PAGE_TOKENS + groups[grp], -1
                    )
                    k += 1
        shifts = (np.arange(NF) * g.kv_layer_rows)[:, None]

        def shifted(base: np.ndarray) -> np.ndarray:
            base = base.reshape(1, -1)
            return np.where(base >= 0, base + shifts, -1)

        rows_of("kv_rows", 8 * T)[:] = shifted(kv0)
        rows_of("ix_rows", T)[:] = shifted(ix0)
        rows_of("kb_rows", 4 * T)[:] = shifted(kb0)
        rows_of("kb_tail", T)[:] = shifted(tail0)
        rows_of("slots", 8 * S)[:] = st.reshape(1, -1)
        rows_of("k_pages", PC * S)[:] = shifted(kp)
        rows_of("v_pages", PC * S)[:] = shifted(vp)

    # ------------------------------------------------------------------ the tables
    def _fill(
        self,
        lay: _Layout,
        img: np.ndarray,
        columns,
        T: int,
        n: int,
        decode: bool,
        all_columns: bool,
        params,
        rows,
        tokens,
    ) -> None:
        """A step's tables into the image ``img`` (numpy; the reference the native fill matches byte for byte)."""
        d, g = self.d, self.geo
        prefill = not decode

        def view(name: str, count: int, dtype=np.int32) -> np.ndarray:
            off = lay.at[name] // 4
            return img[off : off + count].view(dtype)

        def rows_of(name: str, count: int) -> np.ndarray:
            """(layers, count) the first ``count`` ints of every layer's slot of a per-layer table"""
            off, stride = lay.at[name] // 4, lay.stride[name] // 4
            rows = self.layouts_rows[name]
            return img[off : off + rows * stride].reshape(rows, stride)[:, :count]

        # ---- the step's tables, every layer's, into the image
        tok, pos = view("tok", T), view("pos", T)
        tok[:] = 0
        pos[:] = 0
        for row, column, ids in columns:
            tok[column : column + len(ids)] = ids
            pos[column : column + len(ids)] = row.pos0 + np.arange(len(ids))
        view("meta", 4)[:] = (rows[0].pos0 if prefill else 0, n, 0, 0)
        cols = view("cols", T)
        cols[:] = np.arange(T) if all_columns else n - 1
        sp = view("samp_params", PARAMS * T, np.float32).reshape(T, PARAMS)
        sp[:] = 0
        if params is not None:
            sp[: len(params)] = params
        if self.ple_layer is not None:
            view("prow", d.ple_heads * T)[:] = self._ngram_rows(columns, T)
            src, dst = view("ple_src", T), view("ple_dst", T)
            src[:] = -1
            dst[:] = -1
            for row, b, _ in columns:
                src[b], dst[b] = g.ple_row(row.state_src), g.ple_row(row.state_dst)
        # delta-net states: the first layer's bases, every later layer a stride further (one write per table)
        NL, NF = len(self.linear), len(self.full)
        bases0 = np.full((T, 4), -1, np.int64)  # conv_src, ssm_src, conv_dst, ssm_dst
        for row, b, _ in columns:
            bases0[b, 1], bases0[b, 0] = g.state_rows(row.state_src, 0)
            bases0[b, 3], bases0[b, 2] = g.state_rows(row.state_dst, 0)
        strides = (np.arange(NL) * g.state_layer_rows)[:, None]
        bases = np.where(
            bases0.reshape(1, -1) >= 0, bases0.reshape(1, -1) + strides, -1
        )  # (NL, 4T)
        if decode:
            rows_of("gdn_bases", 4 * T)[:] = bases
            rows_of("conv_dst", T)[:] = bases.reshape(NL, T, 4)[:, :, 2]
        else:
            for k, name in enumerate(("conv_src", "ssm_src", "conv_dst", "ssm_dst")):
                rows_of(name, 1)[:] = bases[:, k : k + 1]
        if NF and decode:
            self._decode_attention_tables(columns, T, rows_of)
        elif NF:  # prefill: the first attention layer's rows and slots, every later layer a layer further
            row = rows[0]
            k_pages, v_pages = g.kv_pages(row.kv_blocks, 0)
            slot_rows = [SlotRow(0, row.pos0, n, k_pages, v_pages)]
            raw, _ = g.ix_pages(row.kv_blocks, 0)
            positions = row.pos0 + np.arange(n)
            ix0 = np.full(T, -1, np.int64)
            ix0[:n] = raw[positions // PAGE_TOKENS] + positions % PAGE_TOKENS
            kv0 = kv_rows(T, slot_rows).astype(np.int64).reshape(-1)
            blocks = row.pos0 // 4 + np.arange(T // 4 + 1)
            touched = np.repeat(4 * blocks < row.pos0 + n, 4)
            kb0 = np.where(touched, self._kb_rows(row, 0, blocks), -1)
            launches = plan_slots(slot_rows, min(T, QUERY_BLOCK), d.hpg, SLOTS)
            n_launch = self._slot_launches(T, decode)
            assert len(launches) <= n_launch, (len(launches), n_launch)
            table0 = np.full((n_launch, SLOT_STRIDE // 4), -1, np.int64)
            for j, launch in enumerate(launches):
                table0[j, : SLOTS * 8] = launch.slots.reshape(-1)
                table0[j, SLOTS * 8 : SLOTS * 8 + SLOTS * 16] = launch.k_pages.reshape(
                    -1
                )
                table0[j, SLOTS * 8 + SLOTS * 16 :] = launch.v_pages.reshape(-1)
            shifts = (np.arange(NF) * g.kv_layer_rows)[:, None]

            def shifted(base: np.ndarray) -> np.ndarray:
                base = base.reshape(1, -1)
                return np.where(base >= 0, base + shifts, -1)

            rows_of("kv_rows", 8 * T)[:] = shifted(kv0)
            rows_of("ix_rows", T)[:] = shifted(ix0)
            rows_of("kb_rows", len(kb0))[:] = shifted(kb0)
            slot_ints = SLOTS * 8
            pages = table0[:, slot_ints:]
            slots = rows_of("slots", table0.size).reshape(NF, n_launch, -1)
            slots[:, :, :slot_ints] = table0[None, :, :slot_ints]
            slots[:, :, slot_ints:] = np.where(
                pages[None] >= 0, pages[None] + shifts[:, :, None], -1
            )

    def _native_fill(self):
        """The native decode-step fill (``resin._resin.FnextFill``, built with the ``gaudi`` feature), or None:
        not built, or ``RESIN_FNEXT_NATIVE=0``. It writes the same bytes as ``_fill``; ``RESIN_FNEXT_NATIVE=check``
        also runs the numpy fill each step and raises if a byte differs."""
        if os.environ.get("RESIN_FNEXT_NATIVE", "1") == "0":
            return None
        try:
            from resin._resin import FnextFill
        except ImportError:
            return None
        d, g = self.d, self.geo
        ngram = None
        if self.ngram is not None and d.table_rows:
            ng = self.ngram
            ngram = (
                [int(m) for m in ng.mult],
                [int(z) for z in ng.sizes],
                [int(o) for o in ng.offs],
                ng.hpn,
                int(ng.eos),
                int(self.table_first),
                int(d.table_rows),
            )
        native = FnextFill(
            linear_layers=len(self.linear),
            full_layers=len(self.full),
            kv_groups=d.kv_groups,
            hpg=d.hpg,
            kv_layer_rows=g.kv_layer_rows,
            state_layer_rows=g.state_layer_rows,
            ssm_rows=g.ssm_rows,
            ple_offset=len(self.linear) * g.state_layer_rows
            if self.ple_layer is not None
            else None,
            max_positions=self.max_positions,
            ple_heads=d.ple_heads if self.ple_layer is not None else 0,
            ngram=ngram,
        )
        for (T, decode), lay in self.layouts.items():
            if not decode:
                continue
            at, strided = lay.at, lay.of

            def table(name: str) -> tuple[int, int]:
                return (lay.at[name], lay.stride[name]) if name in lay.at else (0, 0)

            native.add_shape(
                {
                    "t": T,
                    "nbytes": lay.size,
                    "slots": self._slot_count(T),
                    "tok": at["tok"],
                    "pos": at["pos"],
                    "meta": at["meta"],
                    "cols": at["cols"],
                    "samp_params": at["samp_params"],
                    "prow": at.get("prow"),
                    "ple_src": at.get("ple_src"),
                    "ple_dst": at.get("ple_dst"),
                    "gdn_bases": table("gdn_bases"),
                    "conv_dst": table("conv_dst"),
                    "kv_rows": table("kv_rows"),
                    "ix_rows": table("ix_rows"),
                    "kb_rows": table("kb_rows"),
                    "kb_tail": table("kb_tail"),
                    "slot_table": table("slots"),
                    "k_pages": table("k_pages"),
                    "v_pages": table("v_pages"),
                }
            )
            del strided
        return native

    def _fill_native(
        self, lay: _Layout, img: np.ndarray, columns, T: int, params
    ) -> None:
        """``_fill`` of a decode step, natively."""
        n = len(columns)
        blocks = [np.asarray(row.kv_blocks, np.int64) for row, _, _ in columns]
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum([len(b) for b in blocks], out=indptr[1:])
        samp = np.zeros((n, PARAMS), np.float32)
        if params is not None:
            samp[: len(params)] = params[:n]
        assert self._native is not None
        self._native.fill(
            T,
            img.view(np.uint8),
            np.array([row.pos0 for row, _, _ in columns], np.int64),
            np.array([row.state_src for row, _, _ in columns], np.int64),
            np.array([row.state_dst for row, _, _ in columns], np.int64),
            indptr,
            np.concatenate(blocks) if n else np.zeros(0, np.int64),
            np.array([ids[0] for _, _, ids in columns], np.int64),
            np.array([t for row, _, _ in columns for t in row.history], np.int64),
            samp.reshape(-1),
        )

    # ------------------------------------------------------------------ a step
    def _step(
        self,
        rows: Sequence[RowCache],
        tokens: Sequence[Sequence[int]],
        params: np.ndarray | None,
        *,
        all_columns: bool = False,
    ) -> np.ndarray:
        """Prefill: one row and its chunk; decode: a token per row. Returns the sampled tokens (rows,), or with
        ``all_columns`` the logits (T, vocab) of every column (prefill buckets up to 256)."""
        dev = self.device
        prefill = len(rows) == 1 and len(tokens[0]) > 1
        if prefill:
            n = len(tokens[0])
            T = bucket(n, self.prefill_buckets)
            columns = [(rows[0], 0, tokens[0])]
        else:
            n = len(rows)
            T = bucket(n, self.decode_buckets)
            columns = [
                (row, b, ids)
                for b, (row, ids) in enumerate(zip(rows, tokens, strict=True))
            ]
        if any(r.pos0 + len(ids) > self.max_positions for r, _, ids in columns):
            raise ValueError("past the context")
        if any(
            r.pos0 + len(ids) > self.cfg.indexer_budget + 3 for r, _, ids in columns
        ):
            raise NotImplementedError(
                "sparse QSA attention past the indexer budget is not wired yet"
            )
        t0 = time.perf_counter()
        decode = not prefill
        lay = self.layouts[T, decode]
        img = self.image
        if decode and self._native is not None:
            if (
                self._native_check
            ):  # the check copy starts from the image: bytes neither fill writes agree
                self.image_check[:] = img
            self._fill_native(lay, img, columns, T, params)
            if self._native_check:
                self._fill(
                    lay,
                    self.image_check,
                    columns,
                    T,
                    n,
                    decode,
                    all_columns,
                    params,
                    rows,
                    tokens,
                )
                a, b = img[: lay.size // 4], self.image_check[: lay.size // 4]
                if not np.array_equal(a, b):
                    bad = int(np.flatnonzero(a != b)[0]) * 4
                    name = max((o, n) for n, o in lay.at.items() if o <= bad)[1]
                    raise RuntimeError(
                        f"native fill differs from numpy at byte {bad} ({name}), T={T}: native "
                        f"{a[bad // 4 : bad // 4 + 8].tolist()} numpy {b[bad // 4 : bad // 4 + 8].tolist()}, rows "
                        f"{[(r.pos0, len(r.kv_blocks)) for r, _, _ in columns]}"
                    )
        else:
            self._fill(
                lay, img, columns, T, n, decode, all_columns, params, rows, tokens
            )
        dev.h2d_async(self.tables_dev, img[: lay.size // 4])
        t1 = time.perf_counter()
        # ---- the launches and collectives
        self._program(T, decode, all_columns).run(dev)
        t2 = time.perf_counter()
        dev.sync()
        if self.profile is not None:
            self.profile.append((T, t1 - t0, t2 - t1, time.perf_counter() - t2))
        columns_out = T if all_columns or decode else 1
        self._last = (T if all_columns else 1 if prefill else n, columns_out)
        if all_columns:
            return self.last_logits()
        return dev.d2h(self.tok_out, (T,), np.int32)[: 1 if prefill else n]

    def scores(self, rows: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The last step's raw logprobs per row: the chosen token's (rows,), and the top list's ids and logprobs
        (rows, TOP_MAX), filled to each row's ``ntop``."""
        d2h = self.device.d2h
        return (
            d2h(self.lp_out, (rows,), np.float32),
            d2h(self.top_idx, (rows, TOP_MAX), np.int32),
            d2h(self.top_lp, (rows, TOP_MAX), np.float32),
        )

    def close(self) -> None:
        """The device owns the memory; closing it frees everything."""

    def last_logits(self) -> np.ndarray:
        """(rows, vocab) the logits the last step sampled from (prefill: its last column; decode: a row per
        sequence)"""
        rows, columns = self._last
        d, NR = self.d, self.d.ranks
        if NR == 1:
            return self.device.d2h(self.logits, (columns, d.vocab), np.float32)[:rows]
        out = self.device.d2h(self.logits_all, (NR, columns, d.vocab), np.float32)
        return np.ascontiguousarray(
            out.transpose(1, 0, 2).reshape(columns, NR * d.vocab)
        )[:rows]

    def prefill(
        self, row: RowCache, ids: Sequence[int], params: np.ndarray | None = None
    ) -> int:
        """Prefill ``ids`` at ``row`` (fresh, or continuing at ``row.pos0``) in chunks -> the sampled next token."""
        ids = [int(t) for t in ids]
        pos, token = 0, -1
        while pos < len(ids):
            n = min(len(ids) - pos, self.prefill_buckets[-1])
            if n % GDN_CHUNK and len(ids) - pos > n:
                n -= n % GDN_CHUNK
            chunk = ids[pos : pos + n]
            token = int(self._step([row], [chunk], params)[0])
            row = row.advanced(n, chunk)
            pos += n
        return token

    def decode(
        self,
        rows: Sequence[RowCache],
        tokens: Sequence[int],
        params: np.ndarray | None = None,
    ) -> np.ndarray:
        """One token of each row -> the sampled next tokens (rows,)."""
        return self._step(rows, [[int(t)] for t in tokens], params)

    def logits_of(self, row: RowCache, ids: Sequence[int]) -> np.ndarray:
        """(len(ids), vocab) logits of a chunk of 2 .. 256 tokens (tests)."""
        return self._step([row], [[int(t) for t in ids]], None, all_columns=True)[
            : len(ids)
        ]

    # ------------------------------------------------------------------ rows for tests
    def sequence_rows(self, slot: int, positions: int) -> RowCache:
        """A fresh sequence of up to ``positions`` tokens on whole slabs of the arena (tests drive sequences by
        hand; a server's Engine leases blocks). A ``slot`` keeps its slabs: asking again gives the same blocks."""
        g = self.geo
        slots = self.__dict__.setdefault("_seq_slots", {})
        if slot not in slots:
            free = self.__dict__.setdefault(
                "_free_slabs", list(range(self.parent_capacity, 0, -1))
            )
            pages = -(-positions // PAGE_TOKENS)
            states = -(-positions // STATE_TOKENS)
            kv_slabs = -(-pages // g.kv_blocks_per_slab)
            state_slabs = -(-states // g.state_blocks_per_slab)
            if kv_slabs + state_slabs > len(free):
                raise ValueError(
                    f"{positions} positions need {kv_slabs + state_slabs} slabs, {len(free)} are free"
                )
            kv_of = [free.pop() for _ in range(kv_slabs)]
            st_of = [free.pop() for _ in range(state_slabs)]
            per = g.kv_blocks_per_slab
            kv = g.kv_block_row(
                [kv_of[k // per] for k in range(pages)], [k % per for k in range(pages)]
            )
            per = g.state_blocks_per_slab
            rows = tuple(
                g.state_block_row(st_of[k // per], k % per) for k in range(states)
            )
            slots[slot] = (kv, rows)
        kv, rows = slots[slot]
        return RowCache(0, kv, -1, rows[0], 0, rows)


__all__ = [
    "DECODE_BUCKETS",
    "PREFILL_BUCKETS",
    "SLOTS",
    "Qwen4ExpGaudi",
    "RowCache",
    "bucket",
]
