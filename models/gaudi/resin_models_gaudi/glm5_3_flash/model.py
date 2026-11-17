"""GLM-5.3-Flash forward steps on one Gaudi card: one tensor-parallel rank of ``nranks``.

A step runs T columns: one sequence's prefill chunk (one query block, T a multiple of the KDA chunk), or one
decode token for each of up to MAX_T sequences. The host uploads the step's tables in one copy, then launches
embed, per layer the attention launches and the ffn, then the head. Between each sub-layer's last launch and
the next graph the rank's partial is reduced over HCCL; the logits slices are all-gathered.

The streams and the coefficients alternate between two buffers (attention X0 -> X1, ffn X1 -> X0) and so do the
partials (attention writes pa, ffn reads it and writes pb). Layer 0
starts from the embedding with zero partials and identity coefficients, which leave the streams as they are.

MTP (``mtp`` = K drafts): a decode step verifies [anchor, d_1 .. d_K] of every row in one target pass (Tq = K + 1
columns per row, ``kda_verify`` records the recurrent updates instead of writing them), samples y_0 .. y_K, and
runs the MTP layer at every column. The host accepts d_1 .. d_m while d_t = y_(t-1), emits y_0 .. y_m, commits the
first m + 1 tokens' state (``kda_commit``), and the next drafts are the MTP's draft at column m chained K - 1 times.
Greedy speculative output equals plain decoding. A prefill runs the MTP layer's
input and key store over its chunk (position p holds the pair hidden p, token p + 1) and drafts after the sampled
token; each chunk stores its last hidden in its STATE block, which the next chunk's first pair reads.

DFlash 2 (``drafter`` = a ``GLM-5.3-Flash-DFlash2`` directory, ``dflash.py``): the same verify step, with the
drafts from the block-diffusion drafter instead of the MTP layer. Every target pass feeds the mean of its tap
layers' streams to the drafter's cache (``inject``), and after the accepted tokens are known one block step makes
the next ``K`` drafts (``verify_finish``). MTP and the drafter are alternatives.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from resin.backend.speculative import SpeculativeStats
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi.common.sampling import PARAMS, TOP_MAX
from resin_models_gaudi.glm5_3_flash import layers as L
from resin_models_gaudi.glm5_3_flash.cache import (
    ARENA_BLOCK_ROWS,
    INDEX_PAGE_ROWS,
    LATENT_PAGE_ROWS,
    PAGE_TOKENS,
    POOL,
    ROW_BYTES,
    STATE_TOKENS,
    CacheGeometry,
    tail_pools_for,
)
from resin_models_gaudi.glm5_3_flash.dflash import (
    Drafter,
    DrafterConfig,
    DraftRow,
    GlmDrafterHost,
    drafter_items,
    drafter_weight_bytes,
    row_buckets,
)
from resin_models_gaudi.glm5_3_flash.layers import (
    LIST_KEYS,
    PAGE_POOLS,
    ROWS_FIELDS,
    SLOT_KEYS,
    SLOT_POOLS,
    Dims,
)
from resin_models_gaudi.glm5_3_flash.weights import NS, RankSlices, prepare
from resin_models_gaudi.loading.progress import info, progress, weight_file_bytes
from resin_models_gaudi.loading.source import Safetensors
from resin_models_gaudi.loading.upload import WeightLoader
from resin_models_gaudi.synapse.graph import Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import Device

DECODE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
MAX_T = DECODE_BUCKETS[-1]
# a verify step's columns (MTP or the DFlash 2 drafter): up to MTP_T // (K + 1) rows of anchor + K drafts. With MTP,
# wider decode steps run plain (Glm53Gaudi.mtp_keys)
MTP_T = 16
# with drafts, steps of up to MTP_GATHER_T columns all-gather the ranks' partials; wider steps all-reduce them
MTP_GATHER_T = 16
PREFILL_T = 4096
# multiples of the 64-token KDA chunk
PREFILL_BUCKETS = (128, 1024, PREFILL_T)
ALL_BUCKETS = (*DECODE_BUCKETS, *PREFILL_BUCKETS)
TMAX = PREFILL_T
SLOTS = {1: 16, 128: 1, 1024: 1, PREFILL_T: 1}  # slots per launch by query block
GATHER_T = 4
HEAD_COLS = 16  # a prefill chunk's head columns
WORKSPACE_RESERVE = 3 << 30  # bytes kept out of the arena
TAIL_OFFSET = LATENT_PAGE_ROWS + INDEX_PAGE_ROWS


def bucket(n: int, prefill: bool) -> int:
    for b in PREFILL_BUCKETS if prefill else DECODE_BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"{n} columns exceed the {'prefill' if prefill else 'decode'} buckets")


def _wide(T: int) -> int:
    """a decode column is a query block of its own: a step wider than one launch's SLOTS[1] slots needs that many
    times the key and index launches of one column's keys"""
    return max(1, -(-T // SLOTS[1])) if T <= MAX_T else 1


def _head_columns(T: int) -> int:
    """the head's logits columns of a step: every column of a decode step, HEAD_COLS of a prefill chunk"""
    return T if T <= MAX_T else HEAD_COLS


def prefill_chunks(n: int) -> range:
    """chunk starts of an n-token prefill, ``step`` tokens per chunk: 1024 up to three such launches, then the widest"""
    return range(0, n, 1024 if n <= 3 * 1024 else PREFILL_T)


@dataclass(frozen=True, slots=True)
class Row:
    """One sequence of a step: its first column, first position, real tokens, the first arena rows of its KV
    blocks (block k holds positions 128 k .. 128 k + 127), and its STATE blocks: ``state_src`` holds its state
    after position pos0 - 1 (-1: fresh), ``state_dst`` receives the state after the step's last token."""

    column: int
    pos0: int
    real: int
    blocks: np.ndarray
    state_src: int
    state_dst: int
    mtp_blocks: np.ndarray | None = None
    state_first: int = 0
    state_rows: tuple[int, ...] = ()
    draft_blocks: np.ndarray | None = None  # the DFlash drafter's DRAFT blocks (window)
    draft_first: int = 0  # the logical block of draft_blocks[0]

    def state_at(self, position: int) -> int:
        """the STATE block of any position (a verify step commits a variable number of tokens)"""
        if not self.state_rows:
            return self.state_dst
        return self.state_rows[position // STATE_TOKENS - self.state_first]

    def draft_row(self, position: int) -> DraftRow:
        """The drafter's view of this row with its block at ``position``."""
        if self.draft_blocks is None:
            raise ValueError("the row has no DRAFT blocks")
        return DraftRow(position, self.draft_blocks, self.draft_first)


class _Arena:
    """The cache arena: ``blocks`` 1024-row blocks of its own, or at ``address``, an arena that other models of
    the card share (cleared, allocated and freed by its owner)."""

    def __init__(self, device: Device, blocks: int, address: int | None = None) -> None:
        self.blocks = blocks
        self.nbytes = blocks * ARENA_BLOCK_ROWS * ROW_BYTES
        if address is None:
            self.address = device.malloc(self.nbytes)
            device.memset32(self.address, 0, self.nbytes // 4)
        else:
            self.address = address


class Glm53Gaudi:
    def __init__(
        self,
        device: Device,
        checkpoint: str | Path,
        cfg: Glm53Config,
        *,
        capacity: int | Callable[[int], int],
        max_seq_len: int,
        rank: int = 0,
        nranks: int = 1,
        layers: int | None = None,
        recipes: RecipeCache | None = None,
        log: Callable[[str], None] = info,
        gather_tokens: int = GATHER_T,
        mtp: int = 0,
        drafter: str | Path | None = None,
        defer_cache: bool = False,
    ) -> None:
        """``defer_cache``: load the weights and the fixed buffers and stop; ``attach`` then makes the cache arena
        and the recipes, which depend on its capacity."""
        self.device, self.cfg, self.log = device, cfg, log
        # the MTP layer's drafts (0: none), or the DFlash 2 drafter's block
        self.dfl_cfg = None if drafter is None else DrafterConfig.load(drafter, cfg)
        if self.dfl_cfg is None:
            self.mtp = self.K = mtp
        else:  # the drafter's block makes dfl_cfg.drafts; a step verifies the first K of them
            if not 0 <= mtp <= self.dfl_cfg.drafts:
                raise ValueError(
                    f"the drafter makes up to {self.dfl_cfg.drafts} drafts a step, not {mtp}"
                )
            self.mtp, self.K = 0, mtp or self.dfl_cfg.drafts
        self.Tq = self.K + 1  # columns per row of a verify step
        self.drafter: Drafter | None = None
        self.mtp_layer = cfg.num_hidden_layers
        self.speculation = SpeculativeStats(self.K)
        if self.K:
            gather_tokens = MTP_GATHER_T
        self.rank, self.nranks = rank, nranks
        self.n_layers = cfg.num_hidden_layers if layers is None else layers
        if self.dfl_cfg is not None and max(self.dfl_cfg.taps) + 1 >= self.n_layers:
            raise ValueError(
                f"the drafter taps layer {max(self.dfl_cfg.taps)}, which {self.n_layers} layers do not complete"
            )
        self.sl = RankSlices.of(cfg, rank, nranks)
        heads = self.sl.kda_heads.stop - self.sl.kda_heads.start
        dfl = self.dfl_cfg
        self.geo = CacheGeometry.from_config(
            cfg,
            heads,
            layers=self.n_layers,
            mtp=self.mtp > 0,
            tail_pools=tail_pools_for(self.K),
            draft_layers=0 if dfl is None else dfl.layers,
            draft_layer_rows=0 if dfl is None else dfl.layer_rows,
            draft_window=0 if dfl is None else dfl.window,
            draft_write=PREFILL_T,
        )
        self.d = replace(
            Dims.of(cfg, self.sl, self.geo.conv_rows), ranks=nranks, gather_tokens=gather_tokens
        )
        self.max_seq_len = max_seq_len
        self.recipes = recipes or RecipeCache(device)
        t0 = time.time()
        src = Safetensors(checkpoint)
        loader = WeightLoader(device, self.recipes)
        total = weight_file_bytes(checkpoint, "*.safetensors") // nranks
        if drafter is not None:
            assert self.dfl_cfg is not None
            total += drafter_weight_bytes(self.dfl_cfg, drafter)
        device.begin_load(total)
        loader.load(prepare(src, cfg, rank, nranks, range(self.n_layers), mtp=self.mtp > 0))
        if drafter is not None:
            assert self.dfl_cfg is not None
            loader.load(drafter_items(drafter, self.dfl_cfg, src))
        device.end_load()
        src.close()
        self.loader = loader
        self.w = loader.w
        log(
            f"glm rank {rank}/{nranks}: {loader.nbytes / 2**30:.1f} GiB of weights "
            f"in {time.time() - t0:.0f} s"
        )
        positions = max_seq_len + TMAX
        self._positions = positions
        # DSA pool score scratch rows: whole indexer slots, and ix_bcnt's 4096-pool groups
        unit = max(SLOT_POOLS, 4096)
        self.index_pools = -(-(positions // POOL + 1) // unit) * unit
        self.pages = -(-positions // PAGE_TOKENS)  # block-table rows per column (sparse)
        self.dsa = self.geo.dsa_layers
        self.kda = self.geo.kda_layers
        self._dsa_off = np.array([self.geo.layer_offset(i) for i in self.dsa], np.int64)
        self._mtp_off = np.zeros(1, np.int64)
        self._mtp_programs: dict[tuple, object] = {}
        self._mtables = self._mtp_tables() if self.mtp else {}
        self._programs: dict[tuple, _Program] = {}
        # device event slots (the native executor's): the sampler's staging copies, the pinned tables' uploads,
        # and the MTP pass's
        self._stage_event, self._host_event = device.event_slots(2), device.event_slots(2)
        self._mtp_event, self._mhost_event = device.event_slots(2), device.event_slots(2)
        self._sample_ops: dict[tuple, _Op] = {}
        self._buffers()
        self._native = self._native_fill()
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
        reserve) and the recipes. ``slab_rows``: slabs of that many rows (another model's slab size).
        ``arena``: the address of a cleared arena of ``capacity`` such slabs plus the null slab, which other
        models of the card share (the caller allocates and frees it)."""
        if self.capacity is not None:
            raise RuntimeError("the cache arena is attached already")
        device, log, positions = self.device, self.log, self._positions
        if slab_rows is not None:
            self.geo = self.geo.with_slab_rows(slab_rows)
        if callable(capacity):  # sized from the memory left after weights and buffers
            capacity = capacity(device.memory_info()[0] - WORKSPACE_RESERVE)
        self.capacity = capacity
        blocks = self.geo.arena_blocks(capacity)
        if self.dfl_cfg is not None and arena is not None:
            raise ValueError("GLM-5.3-Flash with a DFlash 2 drafter cannot share its cards' KV cache yet")
        if self.dfl_cfg is not None:  # the block steps' scratch pages follow the cache
            rows = row_buckets(DECODE_BUCKETS, self.K, MTP_T)
            host = GlmDrafterHost(
                self,
                self.dfl_cfg,
                buckets=ALL_BUCKETS,
                row_buckets=rows,
                columns=TMAX,
                max_positions=positions,
            )
            scratch_rows = blocks * ARENA_BLOCK_ROWS
            blocks += -(-Drafter.scratch_rows(self.dfl_cfg, rows[-1]) // ARENA_BLOCK_ROWS)
            self.drafter = Drafter(host, self.dfl_cfg, scratch_rows)
        self.arena = _Arena(device, blocks, arena)
        log(
            f"glm: cache {capacity} slabs, "
            f"{self.arena.nbytes / 2**30:.1f} GiB per card "
            f"({self.geo.slab_rows // self.geo.kv_block_rows} KV blocks or one STATE block per slab)"
        )
        log("preparing recipes")
        self._compile()

    # ------------------------------------------------------------------ setup
    def _buffers(self) -> None:
        d, dev = self.d, self.device
        H, T = d.hidden, TMAX

        def buf(nbytes: int) -> int:
            return dev.malloc(max(256, -(-nbytes // 256) * 256))

        def const(array: np.ndarray) -> int:
            a = np.ascontiguousarray(array)
            address = buf(a.nbytes)
            dev.h2d(address, a)
            return address

        f = np.float32
        self.consts = {
            "dep": const(np.zeros(64, f)),
            "one": const(np.ones(64, f)),
        }

        def identity(t: int) -> np.ndarray:
            """(24, t) in numpy: T is the tensor's fastest dim. [pre 4 | post 4 | comb 4 x 4]: post 0, comb I"""
            c = np.zeros((24, t), f)
            c[8 + np.arange(4) * 5] = 1.0
            return c

        self.c_identity = {t: const(identity(t)) for t in ALL_BUCKETS}
        self.pzero = const(np.zeros((max(self.nranks, 1), T, H), f))
        self.x = [buf(4 * NS * H * T), buf(4 * NS * H * T)]
        self.c = [buf(4 * 24 * T), buf(4 * 24 * T)]
        self.pa, self.pb = buf(4 * H * T), buf(4 * H * T)
        # the ranks' partials all-gathered (Dims.gathered launches): (H, T, ranks) each
        G = 4 * H * d.gather_tokens * self.nranks
        self.pa_all, self.pb_all = buf(G), buf(G)
        W = d.mla_heads * T
        K = self.index_pools
        self.io = {
            "q_all": buf(2 * d.latent * W),
            "acc_o": buf(4 * d.latent * W),
            "acc_m": buf(4 * W),
            "acc_l": buf(4 * W),
            "ix_q": buf(2 * d.index_dim * d.index_heads * T),
            "ix_w": buf(4 * d.index_heads * T),
            "sink_kv": buf(256),
            "sink_pool": buf(256),
            "sink_acc": buf(256),
            "sink_slots": buf(256),
            "sink_ix": buf(256),
            "sink_conv": buf(256),
            "sink_ssm": buf(256),
            "ix_score": buf(4 * K * T),
            "ix_mask": buf(4 * 4 * K * T),
            "ix_keys": buf(4 * LIST_KEYS * MAX_T),
            "ix_slots": buf(4 * 8 * MAX_T),
            "hf": buf(4 * H * T),
        }
        self.logits = buf(4 * d.vocab * MAX_T)
        self.logits_all = (
            buf(4 * d.vocab * MAX_T * self.nranks) if self.nranks > 1 else self.logits
        )
        self.head_cols = buf(4 * HEAD_COLS)  # prompt logprobs' columns (head_logits)
        self.sampled_buf = buf(_SAMPLED_BYTES)
        self.sampled = {n: self.sampled_buf + off for n, off in _SAMPLED.items()}
        self.sample_table = buf(4096)
        if self.K:  # a verify step's recorded KDA updates and its commit
            B = MTP_T // self.Tq
            nh = d.kda_heads
            self.pend_bytes = lambda b: 4 * 128 * 3 * nh * self.Tq * b  # noqa: E731
            self.pconv_bytes = lambda b: 4 * 3 * d.C * self.Tq * b  # noqa: E731
            L = len(self.kda)
            self.m = {
                "kda_pend": buf(self.pend_bytes(B) * max(L, 1)),
                "kda_pconv": buf(self.pconv_bytes(B) * max(L, 1)),
                "commit_bases": buf(4 * 4 * B * max(L, 1)),
                "commit_counts": buf(4 * B * max(L, 1)),
                "hidden_table": buf(256),
                "sink_commit": buf(256),
                "sink_hidden": buf(256),
            }
        if self.mtp:
            self.m.update(
                {
                    "mtp_cols": buf(4 * TMAX),
                    "mtp_tok": buf(4 * TMAX),
                    "mtp_zero": const(np.zeros(H, f)),
                    "x": [buf(4 * H * TMAX), buf(4 * H * TMAX)],
                    "hf": [buf(4 * H * MAX_T), buf(4 * H * MAX_T)],
                    "sampled": buf(_SAMPLED_BYTES),
                    "sample_table": buf(4096),
                }
            )
            self.msampled = {n: self.m["sampled"] + off for n, off in _SAMPLED.items()}
            self.mtable = buf(max(tb.nbytes for tb in self._mtp_tables().values()))
        self.loff = []  # sparse attention's per-layer page offsets [latent, -1]
        for off in self._dsa_off:
            a = np.zeros(64, np.int32)
            a[:2] = int(off), -1
            self.loff.append(const(a))
        self.tables = self._step_tables()
        nbytes = max(tb.nbytes for tb in self.tables.values())
        self.table = buf(nbytes)

    def _graphs(self) -> dict[tuple, object]:
        dev, d, c, nb = self.device, self.d, self.cfg, self.arena.blocks
        K, graphs = self.index_pools, {}
        kda = bool(self.kda)
        moe = any(i >= c.first_k_dense_replace for i in range(self.n_layers))
        for T in ALL_BUCKETS:
            prefill = T > MAX_T
            graphs["embed", T] = L.embed_graph(dev, d, T, c.vocab_size)
            if kda:
                graphs["kda", T] = (
                    L.kda_prefill_graph(dev, d, T, nb)
                    if prefill
                    else L.kda_decode_graph(dev, d, T, nb)
                )
            for index in (False, True):
                graphs["mla_pre", T, index] = L.mla_pre_graph(dev, d, T, nb, index)
            graphs["mla_post", T] = L.mla_post_graph(dev, d, T)
            graphs["ffn_dense", T] = L.ffn_graph(dev, d, T, False, False)
            if moe:
                graphs["ffn_moe", T] = L.ffn_graph(dev, d, T, True, prefill)
            if prefill:
                graphs["head", T] = L.head_columns_graph(dev, d, T, HEAD_COLS)
                graphs["index_sel", T] = L.index_mask_graph(dev, d, T, K)
            else:
                graphs["head", T] = L.head_graph(dev, d, T)
                graphs["index_sel", T] = L.index_list_graph(dev, d, T, K)
                graphs["sparse", T] = L.attn_sparse_graph(dev, d, T, nb, self.pages)
                for R in (1, T):
                    graphs["sample", T, R] = L.sample_graph(dev, d, T, R, self.nranks)
            qb = T if prefill else 1
            S = SLOTS[qb]
            graphs["index_slots", T] = L.index_slots_graph(dev, d, T, qb, S, K, nb)
            for keep in (0, 4 * K) if prefill else (0,):
                graphs["slots", T, keep] = L.attn_slots_graph(dev, d, T, qb, S, nb, keep)
            if self.K and not prefill:
                self._verify_graphs(graphs, T)
            if self.mtp:
                self._mtp_graphs(graphs, T, prefill)
        if self.drafter is not None:
            graphs.update(self.drafter.graphs(nb))
        return graphs

    def _verify_graphs(self, graphs: dict, T: int) -> None:
        """A verify step of T columns, Tq per row: the KDA layers record, the commit replays."""
        dev, d, nb = self.device, self.d, self.arena.blocks
        Tq = self.Tq
        if T >= Tq:
            B = T // Tq
            if self.kda:
                graphs["kda_verify", T] = L.kda_verify_graph(dev, d, T, Tq, nb)
            graphs["commit", B] = L.commit_graph(dev, d, B, Tq, max(len(self.kda), 1), nb)

    def _mtp_graphs(self, graphs: dict, T: int, prefill: bool) -> None:
        dev, d, c, nb = self.device, self.d, self.cfg, self.arena.blocks
        graphs["mtp_in", T, T] = L.mtp_in_graph(dev, d, T, T, c.vocab_size)
        graphs["mtp_pre", T, False] = L.mla_pre_graph(dev, d, T, nb, False, mtp=True)
        if prefill:
            graphs["final_hidden", T] = L.final_hidden_graph(dev, d, T)
            graphs["hidden", T] = L.hidden_store_graph(dev, d, T, nb)
            # the draft after a prompt reads its chunk's hidden columns
            graphs["mtp_in", 1, T] = L.mtp_in_graph(dev, d, 1, T, c.vocab_size)
            return
        graphs["mtp_pre", T, True] = L.mla_pre_graph(dev, d, T, nb, True, mtp=True)
        graphs["mtp_ffn", T] = L.ffn_graph(dev, d, T, True, False, mtp=True)
        graphs["mtp_head", T] = L.mtp_head_graph(dev, d, T)

    def _compile(self) -> None:
        progress.building_graphs()
        graphs = self._graphs()
        recipes = self.recipes.compile_many(list(graphs.values()))
        self.r: dict[tuple, Recipe] = dict(zip(graphs.keys(), recipes, strict=True))
        for g in graphs.values():
            g.destroy()
        self.compiles_at_start = self.device.compile_count

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start

    # ------------------------------------------------------------------ programs
    def _weights_of(self, layer: int, recipe: Recipe) -> dict[str, int]:
        p = f"l{layer}."
        out = {}
        for n in recipe.persistent:
            if p + n in self.w:
                out[n] = self.w[p + n]
            elif n in self.w:
                out[n] = self.w[n]
            elif n in self.consts:
                out[n] = self.consts[n]
        return out

    def _op(self, recipe: Recipe, names: dict[str, int]) -> _Op:
        live = {n: a for n, a in names.items() if n in recipe.persistent}
        missing = set(recipe.persistent) - set(live)
        if missing:
            raise ValueError(f"{recipe.name}: unbound {sorted(missing)}")
        return _Op(recipe, live)

    def _p(self, partial: int, T: int) -> int:
        if not self.d.gathered(T):
            return partial
        return self.pa_all if partial == self.pa else self.pb_all

    def _program(self, T: int, index: bool, sparse: bool, verify: int = 0) -> _Program:
        """Every launch of a step of shape (T, indexer on, sparse attention) with its bindings prepared once:
        tables live at fixed offsets and the buffers ping-pong the same way in every layer. A layer is
        (attention ops, index slot ops, index end op, slot ops, post op, ffn op)."""
        key = (T, index, sparse, verify)
        if (prog := self._programs.get(key)) is not None:
            return prog
        tb, base, arena = self.tables[T], self.table, self.arena.address
        x0, x1 = self.x
        c0, c1 = self.c
        head = {
            "tok": base + tb.TOK,
            "meta": base + tb.META,
            "vis": base + tb.VIS,
            "klen": base + tb.KLEN,
        }
        common = {**self.consts, **self.io, **head, "arena": arena}
        prog = _Program()
        r = self.r["embed", T]
        emb = {"tok": head["tok"], "emb": self.w["emb"], "X": x1, "c": c1, "X_out": x0}
        prog.embed = self._op(r, emb)
        prog.embed_device = self._op(r, {**emb, "tok": self.sampled["tok_out"]})
        ix_sel = self.r["index_sel", T]
        ix_slots = self.r["index_slots", T]
        sum_nth, kda_nth = 0, 0
        for i in range(self.n_layers):
            first = i == 0
            res = {
                "X": x0,
                "c": self.c_identity[T] if first else c1,
                "p": self.pzero if first else self._p(self.pb, T),
                "X_out": x1,
                "c_out": c0,
                "p_out": self.pa,
            }
            if self.cfg.layer_types[i] == "linear_attention":
                r = self.r["kda_verify", T] if verify else self.r["kda", T]
                a = base + tb.kda(kda_nth)
                names = {**common, **self._weights_of(i, r), **res}
                if verify:
                    B = T // verify
                    names.update(
                        kda_bases=a,
                        kda_pend=self.m["kda_pend"] + kda_nth * self.pend_bytes(B),
                        kda_pconv=self.m["kda_pconv"] + kda_nth * self.pconv_bytes(B),
                    )
                elif T <= MAX_T:
                    names.update(kda_bases=a, conv_dst=a + tb.BASES_BYTES)
                else:
                    names.update(conv_src=a, ssm_src=a + 256, ssm_dst=a + 512, conv_dst=a + 768)
                kda_nth += 1
                attn, ix, ix_end, slots, post = self._op(r, names), [], None, [], None
            else:
                nth = sum_nth
                sum_nth += 1
                r = self.r["mla_pre", T, index]
                names = {
                    **common,
                    **self._weights_of(i, r),
                    **res,
                    "kv_rows": base + tb.store(nth),
                    "pool_rows": base + tb.pool(nth),
                }
                attn = self._op(r, names)
                ix, ix_end = [], None
                if index:
                    for j in range(tb.index_launches):
                        ao = base + tb.index(j, nth)
                        ix.append(
                            self._op(
                                ix_slots, {**common, "slots": ao, "ix_pages": ao + tb.SLOTS_BYTES}
                            )
                        )
                    ix_end = self._op(ix_sel, common)
                slots = []
                if sparse:
                    slots.append(
                        self._op(
                            self.r["sparse", T],
                            {**common, "kv_blocks": base + tb.BLOCKS, "loff": self.loff[nth]},
                        )
                    )
                else:
                    keep = 4 * self.index_pools if index and T > MAX_T else 0
                    rs = self.r["slots", T, keep]
                    for j in range(tb.key_launches):
                        ao = base + tb.slot(j, nth)
                        slots.append(
                            self._op(
                                rs, {**common, "slots": ao, "lat_pages": ao + tb.SLOTS_BYTES}
                            )
                        )
                r = self.r["mla_post", T]
                post = self._op(r, {**common, **self._weights_of(i, r), "p_out": self.pa})
            kind = "moe" if i >= self.cfg.first_k_dense_replace else "dense"
            r = self.r[f"ffn_{kind}", T]
            ffn = self._op(
                r,
                {
                    **common,
                    **self._weights_of(i, r),
                    "X": x1,
                    "c": c0,
                    "p": self._p(self.pa, T),
                    "X_out": x0,
                    "c_out": c1,
                    "p_out": self.pb,
                },
            )
            prog.layers.append((attn, ix, ix_end, slots, post, ffn))
        r = self.r["head", T]
        bind = {
            **self._weights_of(-1, r),
            **self.consts,
            "X": x0,
            "c": c1,
            "p": self._p(self.pb, T),
            "X_out": x1,
            "hf": self.io["hf"],
            "logits": self.logits,
            "cols": base + tb.HEAD,
        }
        prog.head = self._op(r, bind)
        if T > MAX_T:
            prog.head_scored = self._op(r, {**bind, "cols": self.head_cols})
        if self.drafter is not None:
            # the drafter's context projection of layer t's output: the streams the next layer's
            # attention launch completes (x1), after that launch
            assert self.dfl_cfg is not None
            r = self.r["dfl_tap", f"T{T}"]
            for k, t in enumerate(self.dfl_cfg.taps):
                prog.taps[t + 1] = self._op(
                    r, {"x": x1, "fc": self.w[f"dfl.fc{k}"], "feat": self.drafter.feat[k]}
                )
        self._programs[key] = prog
        return prog

    # ------------------------------------------------------------------ the step
    def _sample_op(self, T: int, R: int, mtp: bool = False) -> _Op:
        """The sampler over R columns of a step of T columns (R = 1: column ``last`` of its table), its table at
        ``sample_table`` (the MTP pass's: ``m["sample_table"]``)"""
        op = self._sample_ops.get((T, R, mtp))
        if op is None:
            addr = self.m["sample_table"] if mtp else self.sample_table
            op = self._sample_ops[T, R, mtp] = self._op(
                self.r["sample", T, R],
                {
                    **self.consts,
                    **(self.msampled if mtp else self.sampled),
                    "logits_all": self.logits_all,
                    "samp_params": addr,
                    "last": addr + _SAMPLE_LAST,
                },
            )
        return op

    # ------------------------------------------------------------------ MTP
    def _step_tables(self) -> dict[int, _Tables]:
        """Every step bucket's tables: an index launch per 2048 pools, a key launch per 2048 keys"""
        launches = -(-self.index_pools // SLOT_POOLS)
        key_launches = -(-(self.max_seq_len + TMAX) // SLOT_KEYS)
        return {
            t: _Tables(
                t, len(self.dsa), len(self.kda), key_launches * _wide(t), launches * _wide(t), self.pages
            )
            for t in ALL_BUCKETS
        }

    def _mtp_tables(self) -> dict[int, _Tables]:
        key_launches = -(-(self.max_seq_len + TMAX) // SLOT_KEYS)
        launches = -(-self.index_pools // SLOT_POOLS)
        return {
            t: _Tables(t, 1, 0, key_launches * _wide(t), launches * _wide(t), self.pages)
            for t in ALL_BUCKETS
        }

    def _mtp_program(self, T: int, columns: int, index: bool, sparse: bool) -> _MtpProgram:
        """The MTP pass's launches for T columns reading ``columns`` hidden columns"""
        key = (T, columns, index, sparse)
        if (prog := self._mtp_programs.get(key)) is not None:
            return prog
        tb = self._mtables[T]
        base, m = self.mtable, self.m
        M = self.mtp_layer
        head = {"meta": base + tb.META, "vis": base + tb.VIS, "klen": base + tb.KLEN}
        common = {**self.consts, **self.io, **head, "arena": self.arena.address}
        x0, x1 = m["x"]
        prog = _MtpProgram()
        r = self.r["mtp_in", T, columns]
        prog.mtp_in = self._op(
            r,
            {
                **self._weights_of(M, r),
                "mtp_prev": m["mtp_zero"],
                "hf": self.io["hf"],
                "mtp_cols": m["mtp_cols"],
                "mtp_tok": m["mtp_tok"],
                "x_out": x0,
            },
        )
        prog.slot_prev = prog.mtp_in.index("mtp_prev")
        prog.slot_hf = prog.mtp_in.index("hf")
        prog.slot_tok = prog.mtp_in.index("mtp_tok")
        bucketed = T in self.c_identity
        if bucketed:
            r = self.r["mtp_pre", T, index]
            prog.pre = self._op(
                r,
                {
                    **common,
                    **self._weights_of(M, r),
                    "x": x0,
                    "c": self.c_identity[T],
                    "kv_rows": base + tb.store(0),
                    "pool_rows": base + tb.pool(0),
                },
            )
        if T > MAX_T:  # a prefill chunk: only the keys
            self._mtp_programs[key] = prog
            return prog
        if index:
            ix_slots = self.r["index_slots", T]
            for j in range(tb.index_launches):
                ao = base + tb.index(j, 0)
                prog.ix.append(
                    self._op(ix_slots, {**common, "slots": ao, "ix_pages": ao + tb.SLOTS_BYTES})
                )
            prog.ix_end = self._op(self.r["index_sel", T], common)
        if sparse:
            prog.slots.append(
                self._op(
                    self.r["sparse", T],
                    {**common, "kv_blocks": base + tb.BLOCKS, "loff": self._mtp_loff()},
                )
            )
        else:
            rs = self.r["slots", T, 0]
            for j in range(tb.key_launches):
                ao = base + tb.slot(j, 0)
                prog.slots.append(
                    self._op(rs, {**common, "slots": ao, "lat_pages": ao + tb.SLOTS_BYTES})
                )
        r = self.r["mla_post", T]
        prog.post = self._op(r, {**common, **self._weights_of(M, r), "p_out": self.pa})
        r = self.r["mtp_ffn", T]
        prog.ffn = self._op(
            r,
            {
                **common,
                **self._weights_of(M, r),
                "x": x0,
                "p": self._p(self.pa, T),
                "x_out": x1,
                "c": self.c_identity[T],
                "p_out": self.pb,
            },
        )
        r = self.r["mtp_head", T]
        prog.head = self._op(
            r,
            {
                **self._weights_of(-1, r),
                **self._weights_of(M, r),
                **self.consts,
                "x": x1,
                "p": self._p(self.pb, T),
                "x_out": x0,
                "c": self.c_identity[T],
                "hf_out": m["hf"][0],
                "logits": self.logits,
            },
        )
        prog.slot_hf_out = prog.head.index("hf_out")
        self._mtp_programs[key] = prog
        return prog

    def _mtp_loff(self) -> int:
        if not hasattr(self, "_mloff"):
            a = np.zeros(64, np.int32)
            a[:2] = 0, -1
            self._mloff = self.device.malloc(256)
            self.device.h2d(self._mloff, a)
        return self._mloff

    def _mtp_native(self, prog: _MtpProgram, T: int):
        """``_mtp_launches``' calls for ``prog`` as one native program: section 0 the input and pre launches, section
        1 (when the program has the layer) the rest; the input's hidden and token bindings and the head's hidden
        output read registers (``_M_*``)."""
        from resin._resin import GaudiProgram

        d, gather, ccl = self.d, self.d.gathered(T), self.nranks > 1
        native = GaudiProgram(_M_REGISTERS)

        def launch(op: _Op, patched: dict[int, int] | None = None) -> None:
            """``patched``: binding index -> the register its address comes from"""
            patched = patched or {}
            fixed = [i for i in range(len(op.slots)) if i not in patched]
            native.launch(
                op.recipe.handle,
                [int(op.slots[i]) for i in fixed],
                [int(op.addresses[i]) for i in fixed],
                dynamic=[(int(op.slots[i]), (0, r, 1)) for i, r in patched.items()],
            )

        def take(register: int, ops: list[_Op]) -> None:
            if ops:
                native.take(register)
                for op in ops:
                    launch(op)
                native.end()

        def reduce(partial: int, gathered: int) -> None:
            if ccl and gather:
                native.allgather(partial, gathered, d.hidden * T)
            elif ccl:
                native.allreduce(partial, d.hidden * T)

        native.section("keys")
        launch(prog.mtp_in, {prog.slot_prev: _M_PREV, prog.slot_hf: _M_HF, prog.slot_tok: _M_TOK})
        launch(prog.pre)
        if prog.head is not None:  # a prefill chunk's program has only the keys
            native.section("layer")
            take(_M_INDEX, prog.ix)
            if prog.ix_end is not None:
                launch(prog.ix_end)
            take(_M_SLOTS, prog.slots)
            launch(prog.post)
            reduce(self.pa, self.pa_all)
            launch(prog.ffn)
            reduce(self.pb, self.pb_all)
            launch(prog.head, {prog.slot_hf_out: _M_HF_OUT})
            if ccl:
                native.allgather(self.logits, self.logits_all, d.vocab * T)
        native.finish()
        prog.native = native
        return native

    @property
    def speculative_rows(self) -> int:
        """Rows of one speculative step: as many as fit MTP_T columns of Tq, and with the drafter the largest of its
        block-step buckets (a power of two: 8 with 1 draft, 4 with 2 or 3, 2 with 4 to 7). A batch of more rows runs
        in groups of this size."""
        if self.drafter is not None:
            return max(self.drafter.m.decode_buckets)
        return MTP_T // self.Tq

    def _commit_op(self, Bc: int) -> _Op:
        """A verify step's commit of Bc rows: its tables at ``commit_bases``, ``commit_counts``, ``hidden_table``"""
        m, key = self.m, (Bc, "commit")
        if (op := self._mtp_programs.get(key)) is None:
            op = self._mtp_programs[key] = self._op(
                self.r["commit", Bc],
                {
                    "arena": self.arena.address,
                    "kda_pend": m["kda_pend"],
                    "kda_pconv": m["kda_pconv"],
                    "commit_bases": m["commit_bases"],
                    "commit_counts": m["commit_counts"],
                    "sink_commit": m["sink_commit"],
                    "hf": self.io["hf"],
                    "hidden_table": m["hidden_table"],
                    "sink_hidden": m["sink_hidden"],
                },
            )
        return op

    def _final_op(self, T: int) -> _Op:
        """A prefill chunk's final hidden (every column's, into ``io["hf"]``): the MTP layer's input"""
        if (op := self._mtp_programs.get((T, "final"))) is None:
            x0, x1 = self.x
            c1 = self.c[1]
            op = self._mtp_programs[T, "final"] = self._op(
                self.r["final_hidden", T],
                {**self.consts, "norm": self.w["norm"], "X": x0, "c": c1, "p": self.pb, "X_out": x1,
                 "hf": self.io["hf"]},
            )  # fmt: skip
        return op

    def _hidden_op(self, T: int) -> _Op:
        """A prefill chunk's hidden store: column ``hidden_table`` [0] of ``io["hf"]`` into the STATE block row
        ``hidden_table`` [1]"""
        m = self.m
        if (op := self._mtp_programs.get((T, "hidden"))) is None:
            op = self._mtp_programs[T, "hidden"] = self._op(
                self.r["hidden", T],
                {"arena": self.arena.address, "hf": self.io["hf"], "hidden_table": m["hidden_table"],
                 "sink_hidden": m["sink_hidden"]},
            )  # fmt: skip
        return op

    def _native_program(self, prog: _Program, T: int):
        """``_launches``' calls for ``prog`` as one native program (``resin._resin.GaudiProgram``): the token
        source a select, the index slot and slot launches takes of the step's counts (registers ``_R_*``)."""
        from resin._resin import GaudiProgram

        d, gather, ccl = self.d, self.d.gathered(T), self.nranks > 1
        native = GaudiProgram(_REGISTERS)
        native.section("step")

        def launch(op: _Op) -> None:
            native.launch(op.recipe.handle, op.slots.tolist(), op.addresses.tolist())

        def take(register: int, ops: list[_Op]) -> None:
            if ops:  # a layer without these launches ignores the count
                native.take(register)
                for op in ops:
                    launch(op)
                native.end()

        def reduce(partial: int, gathered: int) -> None:
            if not ccl:
                return
            if gather:
                native.allgather(partial, gathered, d.hidden * T)
            else:
                native.allreduce(partial, d.hidden * T)

        native.select([(_R_DEVICE_TOKENS, 2)], [0, 1])
        native.arm()
        launch(prog.embed)
        native.arm()
        launch(prog.embed_device)
        native.end()
        for i, (attn, ix, ix_end, slots, post, ffn) in enumerate(prog.layers):
            launch(attn)
            if (tap := prog.taps.get(i)) is not None:
                launch(tap)
            take(_R_INDEX, ix)
            if ix_end is not None:
                launch(ix_end)
            take(_R_SLOTS, slots)
            if post is not None:
                launch(post)
            reduce(self.pa, self.pa_all)
            launch(ffn)
            reduce(self.pb, self.pb_all)
        launch(prog.head)
        if ccl:
            native.allgather(self.logits, self.logits_all, d.vocab * _head_columns(T))
        native.finish()  # the program holds copies of the ops' bindings
        prog.native = native
        return native

    def _native_fill(self):
        """The native step fill (``resin._resin.GlmFill``, built with the ``gaudi`` feature) of every table shape,
        or None without it"""
        try:
            from resin._resin import GlmFill
        except ImportError:
            return None
        g, d = self.geo, self.d
        native = GlmFill(
            mla_heads=d.mla_heads, index_heads=d.index_heads, top_pools=d.top_pools, tail_pools=g.tail_pools,
            pages=self.pages, state_layer_rows=g.state_layer_rows, ssm_rows=g.ssm_rows, max_t=MAX_T,
            geometry=(PAGE_TOKENS, POOL, PAGE_POOLS, LATENT_PAGE_ROWS, INDEX_PAGE_ROWS, SLOT_POOLS),
        )  # fmt: skip
        for mtp, tables, offsets in ((False, self.tables, self._dsa_off), (True, self._mtables, self._mtp_off)):
            for tb in tables.values():
                native.add_shape(tb.layout([int(o) for o in offsets]), mtp)
        return native

class _Op:
    """One launch: a recipe and its bindings as prepared slot / address arrays."""

    __slots__ = ("recipe", "slots", "addresses", "args")

    def __init__(self, recipe: Recipe, names: dict[str, int]) -> None:
        self.recipe = recipe
        self.slots = np.array([recipe.slot(n) for n in names], np.int32)
        self.addresses = np.array(list(names.values()), np.uint64)
        self.args = (
            recipe.handle,
            self.slots.ctypes.data,
            self.addresses.ctypes.data,
            len(self.slots),
        )

    def index(self, name: str) -> int:
        """the position of a binding in ``addresses``, for patching it per launch"""
        return int(np.flatnonzero(self.slots == self.recipe.slot(name))[0])

    def run(self, lib) -> None:
        if lib.rg_bind_slots(*self.args) < 0 or lib.rg_launch(self.args[0]) < 0:
            self.recipe.device.check(-1, f"launch {self.recipe.name}")


class _Program:
    def __init__(self) -> None:
        self.embed: _Op | None = None
        self.embed_device: _Op | None = None
        self.head: _Op | None = None
        self.head_scored: _Op | None = None
        self.layers: list[tuple] = []
        self.taps: dict[int, _Op] = {}  # the drafter's tap after layer i's attention launch
        self.native = None  # the step as one native program (Glm53Gaudi._native_program)


class _MtpProgram:
    def __init__(self) -> None:
        self.mtp_in: _Op | None = None
        self.pre: _Op | None = None
        self.ix: list[_Op] = []
        self.ix_end: _Op | None = None
        self.slots: list[_Op] = []
        self.post: _Op | None = None
        self.ffn: _Op | None = None
        self.head: _Op | None = None
        self.slot_prev = self.slot_hf = self.slot_tok = self.slot_hf_out = 0
        self.native = None  # the pass as one native program (Glm53Gaudi._mtp_native)


@dataclass(frozen=True, slots=True)
class _IndexLaunch:
    slots: np.ndarray
    pages: np.ndarray


@dataclass(frozen=True, slots=True)
class Filled:
    """What a step's fill decided: the indexer runs, sparse attention, the slot and index slot launches of each
    attention layer, and the host copy's ranges to upload (offset, bytes; 0 bytes: none)."""

    index: bool
    sparse: bool
    slots: int
    index_launches: int
    uploads: tuple[tuple[int, int], ...]


class _Tables:
    """Byte offsets of a step's tables for buckets of T columns in the device table buffer and its pinned host
    copies: a header, per-layer rows, then slot records ordered by launch first, so a step uploads only the
    launches it uses. KDA records: decode, (4, T) bases then (T,) conv_dst; prefill, 4 values 256 bytes apart."""

    SLOTS_BYTES = 512  # (S <= 16, 8) i32
    PAGES_BYTES = 1024  # (S <= 16, C / 128 = 16) i32
    SLOT_RECORD = SLOTS_BYTES + PAGES_BYTES
    INDEX_RECORD = SLOTS_BYTES + 4 * PAGES_BYTES  # (S <= 16, 64) i32

    def __init__(
        self, T: int, dsa: int, kda: int, key_launches: int, index_launches: int, pages: int
    ) -> None:
        r = lambda n: -(-n // 128) * 128  # noqa: E731
        self.T = T
        self.TOK = 0
        self.META = r(4 * T)
        self.HEAD = self.META + 128
        self.VIS = self.HEAD + 128
        self.KLEN = self.VIS + r(4 * T)
        self.HEADER = self.KLEN + r(4 * T)
        self.ROWS_BYTES = r(4 * ROWS_FIELDS * T)
        self.STORE = self.HEADER
        self.POOL = self.STORE + dsa * self.ROWS_BYTES
        self.KDA_BASE = self.POOL + dsa * self.ROWS_BYTES
        self.BASES_BYTES = r(4 * 4 * T) if T <= MAX_T else 1024
        self.KDA_RECORD = self.BASES_BYTES + (r(4 * T) if T <= MAX_T else 0)
        self.KDA = self.KDA_BASE
        self.dsa, self.key_launches, self.index_launches = dsa, key_launches, index_launches
        self.kda_layers = kda
        self.BLOCKS = r(self.KDA_BASE + kda * self.KDA_RECORD)
        self.pages = pages
        self.slot_base = -(-(self.BLOCKS + 4 * pages * T) // 4096) * 4096
        self.index_base = self.slot_base + key_launches * dsa * self.SLOT_RECORD
        self.nbytes = self.index_base + index_launches * dsa * self.INDEX_RECORD

    def store(self, nth: int) -> int:
        return self.STORE + nth * self.ROWS_BYTES

    def layout(self, offsets: list[int]) -> dict:
        """The offsets for the native fill (``GlmFill.add_shape``), with the DSA layers' page offsets"""
        return {**{k: v for k, v in vars(self).items() if isinstance(v, int)}, "offsets": offsets}

    def pool(self, nth: int) -> int:
        return self.POOL + nth * self.ROWS_BYTES

    def kda(self, nth: int) -> int:
        return self.KDA_BASE + nth * self.KDA_RECORD

    def slot(self, launch: int, nth: int) -> int:
        return self.slot_base + (launch * self.dsa + nth) * self.SLOT_RECORD

    def index(self, launch: int, nth: int) -> int:
        return self.index_base + (launch * self.dsa + nth) * self.INDEX_RECORD


_SAMPLED_COL = -(-4 * MAX_T // 256) * 256  # one f32 or i32 per column, 256-byte aligned
_SAMPLED = {
    "tok_out": 0,
    "lp_out": _SAMPLED_COL,
    "top_idx": 2 * _SAMPLED_COL,
    "top_lp": 2 * _SAMPLED_COL + 4 * TOP_MAX * MAX_T,
}
_SAMPLED_BYTES = 2 * _SAMPLED_COL + 8 * TOP_MAX * MAX_T
_SAMPLE_TABLE = 4 * PARAMS * MAX_T + 256
_SAMPLE_LAST = 4 * PARAMS * MAX_T
# the registers of a step's native program: the token source (1: the previous step's samples on the device),
# and per attention layer the index slot launches and the slot launches it makes
_R_DEVICE_TOKENS, _R_INDEX, _R_SLOTS = 0, 1, 2
_REGISTERS = 3
# the registers of an MTP pass's native program: the input's previous-hidden and hidden buffers and token ids,
# the head's hidden output, and the index slot launches and slot launches of the layer. They follow the step's and
# the native executor's I/O registers (``GlmExecutor``), so that one register set runs every program of a step
_M_BASE = 16
_M_PREV, _M_HF, _M_TOK, _M_HF_OUT, _M_INDEX, _M_SLOTS = range(_M_BASE, _M_BASE + 6)
_M_REGISTERS = _M_BASE + 6
