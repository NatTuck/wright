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
"""

from __future__ import annotations

import os

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from resin.backend.speculative import SpeculativeStats
from resin.models.glm5_3_flash.config import Glm53Config
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
    tail_rows,
)
from resin_models_gaudi.glm5_3_flash.device import WeightLoader
from resin_models_gaudi.glm5_3_flash.layers import (
    LIST_KEYS,
    PAGE_POOLS,
    ROWS_FIELDS,
    SLOT_KEYS,
    SLOT_POOLS,
    Dims,
)
from resin_models_gaudi.glm5_3_flash.weights import NS, RankSlices, Safetensors, prepare
from resin_models_gaudi.qwen3_5.attention import SlotRow, plan_slots
from resin_models_gaudi.qwen3_5.model import Sampling
from resin_models_gaudi.qwen3_5.sampling import PARAMS, TOP_MAX
_LAUNCH_MODE = os.environ.get("RESIN_GAUDI_PROFILE_LAUNCH_MODE", "full")
# Bind once and launch per step: _Op.args is built at construction and the addresses are mutated
# only in the GLM MTP path and LaunchPlan.patch, so a DeepSeek decode step re-binds 109 identical
# slot/address arrays every token (~36 us each, ~3.9 ms/step). Default OFF until the token gate
# proves it bit-identical: RESIN_GAUDI_LAZY_BIND=1 to try it.
# NEVER enable it for the GLM MTP path: six sites mutate _Op addresses (glm5_3_flash/model.py
# 805-835, graph.py:331) and lazy bind would leave them stale - silent corruption, not a crash.
_LAZY_BIND = os.environ.get("RESIN_GAUDI_LAZY_BIND") == "1"
_SYNC_EVERY = int(os.environ.get("RESIN_GAUDI_PROFILE_SYNC_EVERY", "0") or 0)
_SYNC_COUNT = [0]
_SYNC_PENDING = [False]
# the sum of the diagnostic syncs is the serialised device time for the window
_SYNC_TOTAL = [0.0]
_SYNC_TIMES: list[float] = []

from resin_models_gaudi.synapse.graph import (
    LAUNCH_BY_RECIPE,
    LAST_CALLS,
    LAUNCH_STATS,
    _PROFILE_STEP,
)
from resin_models_gaudi.synapse.graph import Recipe, RecipeCache
from resin_models_gaudi.synapse.progress import info, weight_file_bytes
from resin_models_gaudi.synapse.runtime import Device

DECODE_BUCKETS = (1, 2, 4, 8, 16)
MAX_T = DECODE_BUCKETS[-1]
PREFILL_T = 4096
# multiples of the 64-token KDA chunk
PREFILL_BUCKETS = (128, 1024, PREFILL_T)
ALL_BUCKETS = (*DECODE_BUCKETS, *PREFILL_BUCKETS)
TMAX = PREFILL_T
SLOTS = {1: 16, 128: 1, 1024: 1, PREFILL_T: 1}  # slots per launch by query block
GATHER_T = 4
HEAD_COLS = MAX_T  # a prefill chunk's head columns
WORKSPACE_RESERVE = 3 << 30  # bytes kept out of the arena
TAIL_OFFSET = LATENT_PAGE_ROWS + INDEX_PAGE_ROWS


def bucket(n: int, prefill: bool) -> int:
    for b in PREFILL_BUCKETS if prefill else DECODE_BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"{n} columns exceed the {'prefill' if prefill else 'decode'} buckets")


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

    def state_at(self, position: int) -> int:
        """the STATE block of any position (a verify step commits a variable number of tokens)"""
        if not self.state_rows:
            return self.state_dst
        return self.state_rows[position // STATE_TOKENS - self.state_first]


@dataclass(slots=True)
class Launched:
    """A verify pass left running (``Glm53Gaudi.verify_launch``), for ``verify_finish``."""

    rows: Sequence[Row]
    drafts: np.ndarray
    sampling: Sampling | None
    staged: Staged
    staged_drafts: Staged
    T: int


@dataclass(frozen=True, slots=True)
class Verified:
    """A verify step: each row's emitted tokens (y_0 .. y_m), its accepted draft count m, its next drafts, and the
    step's columns of each row (for their logprobs)."""

    tokens: list[list[int]]
    accepted: np.ndarray
    drafts: np.ndarray
    columns: list[np.ndarray]


class _Arena:
    def __init__(self, device: Device, blocks: int) -> None:
        self.blocks = blocks
        self.nbytes = blocks * ARENA_BLOCK_ROWS * ROW_BYTES
        self.address = device.malloc(self.nbytes)
        device.memset32(self.address, 0, self.nbytes // 4)


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
    ) -> None:
        self.device, self.cfg, self.log = device, cfg, log
        self.mtp = mtp
        self.mtp_layer = cfg.num_hidden_layers
        self.speculation = SpeculativeStats(mtp)
        if mtp:
            gather_tokens = MAX_T
        self.rank, self.nranks = rank, nranks
        self.n_layers = cfg.num_hidden_layers if layers is None else layers
        self.sl = RankSlices.of(cfg, rank, nranks)
        heads = self.sl.kda_heads.stop - self.sl.kda_heads.start
        self.geo = CacheGeometry.from_config(cfg, heads, layers=self.n_layers, mtp=mtp > 0)
        self.d = replace(
            Dims.of(cfg, self.sl, self.geo.conv_rows), ranks=nranks, gather_tokens=gather_tokens
        )
        self.max_seq_len = max_seq_len
        self.recipes = recipes or RecipeCache(device)
        t0 = time.time()
        src = Safetensors(checkpoint)
        loader = WeightLoader(device, self.recipes)
        device.begin_load(weight_file_bytes(checkpoint, "*.safetensors") // nranks, log=log)
        loader.load(prepare(src, cfg, rank, nranks, range(self.n_layers), mtp=mtp > 0))
        device.end_load()
        src.close()
        self.w = loader.w
        log(
            f"glm rank {rank}/{nranks}: {loader.nbytes / 2**30:.1f} GiB of weights "
            f"in {time.time() - t0:.0f} s"
        )
        positions = max_seq_len + TMAX
        # DSA pool score scratch rows: whole indexer slots, and ix_bcnt's 4096-pool groups
        unit = max(SLOT_POOLS, 4096)
        self.index_pools = -(-(positions // POOL + 1) // unit) * unit
        self.pages = -(-positions // PAGE_TOKENS)  # block-table rows per column (sparse)
        self.dsa = self.geo.dsa_layers
        self.kda = self.geo.kda_layers
        self._dsa_off = np.array([self.geo.layer_offset(i) for i in self.dsa], np.int64)
        self._mtp_off = np.zeros(1, np.int64)
        self._mtp_programs: dict[tuple, object] = {}
        self._mstage_turn = 0
        self._mtables = self._mtp_tables() if mtp else {}
        self._programs: dict[tuple, _Program] = {}
        self._staging = [Staged(device, _STAGE_EVENT + k) for k in range(2)]
        self._stage_turn = 0
        self._sample_ops: dict[tuple, _Op] = {}
        self._last_program: _Program | None = None
        self._prefill_last = -1
        self.probe = None  # (layer, "attn" | "mlp", T, partial address): tests, after each sub-layer
        self._buffers()
        if callable(capacity):  # sized from the memory left after weights and buffers
            capacity = capacity(device.memory_info()[0] - WORKSPACE_RESERVE)
        self.capacity = capacity
        self.arena = _Arena(device, self.geo.arena_blocks(capacity))
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
        if self.mtp:
            B = MAX_T // (self.mtp + 1)
            nh = d.kda_heads
            self.pend_bytes = lambda b: 4 * 128 * 3 * nh * (self.mtp + 1) * b  # noqa: E731
            self.pconv_bytes = lambda b: 4 * 3 * d.C * (self.mtp + 1) * b  # noqa: E731
            L = len(self.kda)
            self.m = {
                "mtp_cols": buf(4 * TMAX),
                "mtp_tok": buf(4 * TMAX),
                "mtp_zero": const(np.zeros(H, f)),
                "x": [buf(4 * H * TMAX), buf(4 * H * TMAX)],
                "hf": [buf(4 * H * MAX_T), buf(4 * H * MAX_T)],
                "kda_pend": buf(self.pend_bytes(B) * max(L, 1)),
                "kda_pconv": buf(self.pconv_bytes(B) * max(L, 1)),
                "commit_bases": buf(4 * 4 * B * max(L, 1)),
                "commit_counts": buf(4 * B * max(L, 1)),
                "hidden_table": buf(256),
                "sink_commit": buf(256),
                "sink_hidden": buf(256),
                "sampled": buf(_SAMPLED_BYTES),
                "sample_table": buf(4096),
            }
            self.msampled = {n: self.m["sampled"] + off for n, off in _SAMPLED.items()}
            self._mstage = [Staged(dev, _MTP_EVENT + k) for k in range(2)]
            self.mtable = buf(max(tb.nbytes for tb in self._mtp_tables().values()))
            self.mhost = [
                dev.host_buffer(max(tb.nbytes for tb in self._mtp_tables().values()), np.uint8)
                for _ in range(2)
            ]
            self._mturn = 0
            self._mpending = [False, False]
        self.loff = []  # sparse attention's per-layer page offsets [latent, -1]
        for off in self._dsa_off:
            a = np.zeros(64, np.int32)
            a[:2] = int(off), -1
            self.loff.append(const(a))
        launches = -(-K // SLOT_POOLS)  # index launches; a key launch per 2048 keys
        key_launches = -(-(self.max_seq_len + TMAX) // SLOT_KEYS)
        self.tables = {
            t: _Tables(t, len(self.dsa), len(self.kda), key_launches, launches, self.pages)
            for t in ALL_BUCKETS
        }
        nbytes = max(tb.nbytes for tb in self.tables.values())
        self.table = buf(nbytes)
        self.host_tables = [dev.host_buffer(nbytes, np.uint8) for _ in range(2)]
        self._host_turn = 0
        self._host_pending = [False, False]

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
            if self.mtp:
                self._mtp_graphs(graphs, T, prefill)
        return graphs

    def _mtp_graphs(self, graphs: dict, T: int, prefill: bool) -> None:
        dev, d, c, nb = self.device, self.d, self.cfg, self.arena.blocks
        Tq = self.mtp + 1
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
        if T >= Tq:
            B = T // Tq
            if self.kda:
                graphs["kda_verify", T] = L.kda_verify_graph(dev, d, T, Tq, nb)
            graphs["commit", B] = L.commit_graph(dev, d, B, Tq, max(len(self.kda), 1), nb)

    def _compile(self) -> None:
        t0 = time.time()
        graphs = self._graphs()
        recipes = self.recipes.compile_many(list(graphs.values()))
        self.r: dict[tuple, Recipe] = dict(zip(graphs.keys(), recipes, strict=True))
        for g in graphs.values():
            g.destroy()
        cached = sum(r.from_cache for r in recipes)
        self.log(f"glm: {len(recipes)} recipes ({cached} cached) in {time.time() - t0:.1f} s")
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
        self._programs[key] = prog
        return prog

    # ------------------------------------------------------------------ the step
    def forward(
        self, tokens: np.ndarray, positions: np.ndarray, rows: Sequence[Row], prefill: bool = False
    ) -> np.ndarray:
        """T columns (``tokens`` and ``positions``) whose sequences are ``rows`` -> logits (T, V) f32"""
        T = self._launch(tokens, positions, rows, prefill)
        if T > MAX_T:
            return self.head_logits(np.arange(len(tokens)))
        return self.logits_host(T)[: len(tokens)]

    def logits_host(self, T: int) -> np.ndarray:
        """the last step's full logits (T, V) f32; a prefill chunk's (HEAD_COLS, V): its last column"""
        self.device.sync()
        V, T = self.d.vocab, min(T, HEAD_COLS)
        out = self.device.d2h(self.logits_all, (self.nranks, T, V), np.float32)
        return np.ascontiguousarray(out.transpose(1, 0, 2).reshape(T, self.nranks * V))

    def head_logits(self, columns: np.ndarray) -> np.ndarray:
        """The last launched prefill chunk's logits at ``columns`` (n, V), HEAD_COLS columns per launch.
        Overwrites the chunk's sampled logits: sample before calling this."""
        prog, dev = self._last_program, self.device
        assert prog is not None and prog.head_scored is not None, "no prefill chunk"
        out = []
        for g in range(0, len(columns), HEAD_COLS):
            cols = np.asarray(columns[g : g + HEAD_COLS], np.int32)
            dev.sync()
            dev.h2d(self.head_cols, np.resize(cols, HEAD_COLS))
            prog.head_scored.run(dev.lib)
            if self.nranks > 1:
                dev.allgather(self.logits, self.logits_all, self.d.vocab * HEAD_COLS)
            out.append(self.logits_host(HEAD_COLS)[: len(cols)])
        return (
            np.concatenate(out)
            if out
            else np.zeros((0, self.d.vocab * self.nranks), np.float32)
        )

    def sample(self, tokens, positions, rows, params, last=None, prefill=False):
        """A step, then the device sampler over every column (decode) or column ``last`` (a prefill chunk's
        last real token) -> (tokens, logprobs, top ids, top logprobs) of the sampled rows"""
        T = self._launch(tokens, positions, rows, prefill)
        return self.sample_launched(T, params, last)

    def sample_launched(self, T: int, params: np.ndarray, last: int | None = None):
        if T > MAX_T:  # the head ran at the chunk's last column only
            assert last == self._prefill_last, (last, self._prefill_last)
            T, last = HEAD_COLS, 0
        self._sample_launch(T, params, last)
        self.device.sync()
        out = self.device.d2h(self.sampled_buf, (_SAMPLED_BYTES,), np.uint8)
        return _sampled_fields(out, 1 if last is not None else T, len(params))

    def sample_async(self, T: int, params: np.ndarray) -> Staged:
        """Sample every column of the step just launched and copy the results into a pinned staging buffer
        without waiting; ``Staged.result`` waits for them."""
        self._sample_launch(T, params, None)
        k = self._stage_turn
        self._stage_turn ^= 1
        staged = self._staging[k]
        staged.drain()
        self.device.d2h_async(staged.buffer, self.sampled_buf)
        self.device.record(_STAGE_EVENT + k)
        staged.arm(T, len(params))
        return staged

    def _sample_launch(self, T: int, params: np.ndarray, last: int | None, mtp: bool = False) -> None:
        R = 1 if last is not None else T
        n = len(params)
        table = np.zeros(_SAMPLE_TABLE // 4, np.float32)
        table[: n * PARAMS] = np.asarray(params, np.float32).ravel()
        table.view(np.int32)[_SAMPLE_LAST // 4] = 0 if last is None else last
        dev = self.device
        addr = self.m["sample_table"] if mtp else self.sample_table
        dev.h2d_nb(addr, table)
        op = self._sample_ops.get((T, R, mtp))
        if op is None:
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
        op.run(dev.lib)

    # ------------------------------------------------------------------ MTP
    def _mtp_tables(self) -> dict[int, _Tables]:
        key_launches = -(-(self.max_seq_len + TMAX) // SLOT_KEYS)
        launches = -(-self.index_pools // SLOT_POOLS)
        return {t: _Tables(t, 1, 0, key_launches, launches, self.pages) for t in ALL_BUCKETS}

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

    def _mtp_pass(
        self,
        T: int,
        columns: int,
        rows: Sequence[Row],
        cols: np.ndarray,
        tokens: np.ndarray | int,
        prev: int,
        hf: int,
        params: np.ndarray | None,
        hf_out: int | None = None,
    ) -> None:
        """The MTP layer: column c reads token ``tokens`` [c] (or the device buffer ``tokens``) and hidden column
        ``cols`` [c] of [``prev`` | ``hf``]. ``rows``' columns and positions are its own, ``blocks`` its KV. With
        ``params``, the whole layer and the head, sampling the drafts into ``msampled``, its hidden into ``hf_out``
        (default: the first buffer of ``m["hf"]``); without, only its keys."""
        dev, d, m = self.device, self.d, self.m
        tb = self._mtables[T]
        k = self._mturn
        self._mturn ^= 1
        if self._mpending[k]:
            dev.wait(_MHOST_EVENT + k)
        h = self.mhost[k]
        h[: tb.HEADER].view(np.int32)[: tb.HEADER // 4] = 0
        real = sum(r.real for r in rows)
        h[tb.META : tb.META + 16].view(np.int32)[:] = (rows[0].pos0, real, 0, 0)
        launches, ix_launches, index, sparse = self._attention_tables(
            h, tb, rows, T, T if T > MAX_T else 1, SLOTS[T if T > MAX_T else 1], self._mtp_off, True
        )
        self._upload(self.mtable, h, tb, launches, ix_launches, sparse)
        dev.record(_MHOST_EVENT + k)
        self._mpending[k] = True
        dev.h2d_nb(m["mtp_cols"], np.resize(np.asarray(cols, np.int32), T))
        prog = self._mtp_program(T, columns, index and params is not None, sparse and params is not None)
        mi = prog.mtp_in
        mi.addresses[prog.slot_prev] = prev
        mi.addresses[prog.slot_hf] = hf
        if isinstance(tokens, np.ndarray):
            dev.h2d_nb(m["mtp_tok"], np.resize(tokens.astype(np.int32), T))
            mi.addresses[prog.slot_tok] = m["mtp_tok"]
        else:
            mi.addresses[prog.slot_tok] = tokens
        lib = dev.lib
        mi.run(lib)
        prog.pre.run(lib)
        if params is None:
            return
        gather = d.gathered(T)
        ar = None
        if self.nranks > 1:
            ar = lib.rg_ccl_allgather if gather else lib.rg_ccl_allreduce
        dst_a, dst_b = (self.pa_all, self.pb_all) if gather else (self.pa, self.pb)
        count = d.hidden * T
        for op in prog.ix[: len(ix_launches)]:
            op.run(lib)
        if prog.ix_end is not None:
            prog.ix_end.run(lib)
        for op in prog.slots[: 1 if sparse else len(launches)]:
            op.run(lib)
        prog.post.run(lib)
        if ar is not None and ar(self.pa, dst_a, count, 0):
            dev.check(-1, "residual collective")
        prog.ffn.run(lib)
        if ar is not None and ar(self.pb, dst_b, count, 0):
            dev.check(-1, "residual collective")
        prog.head.addresses[prog.slot_hf_out] = m["hf"][0] if hf_out is None else hf_out
        prog.head.run(lib)
        if self.nranks > 1:
            dev.allgather(self.logits, self.logits_all, d.vocab * T)
        self._sample_launch(T, params, None, mtp=True)

    def _sample_mtp_stage(self, T: int, n: int) -> Staged:
        """The last MTP pass's samples, copied without waiting"""
        k = self._mstage_turn
        self._mstage_turn ^= 1
        staged = self._mstage[k]
        staged.drain()
        self.device.d2h_async(staged.buffer, self.m["sampled"])
        self.device.record(_MTP_EVENT + k)
        staged.arm(T, n)
        return staged

    def prefill_mtp(self, row: Row, ids: np.ndarray, T: int, last: bool, sampling) -> np.ndarray | None:
        """The MTP keys of a prefill chunk (after ``_launch``), its last hidden into the STATE block, and, after the
        prompt's last chunk (``last``), the first draft following its sampled token. The draft chain would write MTP
        keys past the prompt, where a prefill leases no blocks, so drafts 2 .. K start as zeros."""
        dev, m, g = self.device, self.m, self.geo
        pos0, real = row.pos0, row.real
        x0, x1 = self.x
        c0, c1 = self.c
        if (T, "final") not in self._mtp_programs:
            r = self.r["final_hidden", T]
            self._mtp_programs[T, "final"] = self._op(
                r,
                {
                    **self.consts,
                    "norm": self.w["norm"],
                    "X": x0,
                    "c": c1,
                    "p": self.pb,
                    "X_out": x1,
                    "hf": self.io["hf"],
                },
            )
        self._mtp_programs[T, "final"].run(dev.lib)
        # column c: position pos0 - 1 + c, token ids[c], hidden c of [prev | hf] (prev: the hidden at pos0 - 1);
        # a sequence's first chunk starts one later (no position -1)
        shift = 0 if pos0 > 0 else 1
        n = real - shift
        prev = m["mtp_zero"]
        if row.state_src >= 0:
            prev = self.arena.address + g.hidden_row(row.state_src) * ROW_BYTES
        if n > 0:
            tok = np.zeros(T, np.int64)
            tok[:n] = ids[shift:real]
            cols = np.zeros(T, np.int64)
            cols[:n] = np.arange(n) + shift
            key_row = Row(0, pos0 - 1 + shift, n, row.mtp_blocks, -1, -1)
            self._mtp_pass(T, T, [key_row], cols, tok, prev, self.io["hf"], None)
        dev.h2d_nb(m["hidden_table"], np.array([real - 1, g.hidden_row(row.state_dst)], np.int32))
        if (T, "hidden") not in self._mtp_programs:
            self._mtp_programs[T, "hidden"] = self._op(
                self.r["hidden", T],
                {
                    "arena": self.arena.address,
                    "hf": self.io["hf"],
                    "hidden_table": m["hidden_table"],
                    "sink_hidden": m["sink_hidden"],
                },
            )
        self._mtp_programs[T, "hidden"].run(dev.lib)
        if not last:
            return None
        # the sampled token at pos0 + real: MTP at pos0 + real - 1 drafts pos0 + real + 1
        params = np.zeros((1, PARAMS), np.float32)
        if sampling is not None:
            params[:1] = sampling.at([0], [1])
        draft_row = Row(0, pos0 + real - 1, 1, row.mtp_blocks, -1, -1)
        self._mtp_pass(
            1, T, [draft_row], np.array([real]), self.sampled["tok_out"], m["mtp_zero"], self.io["hf"], params
        )
        dev.sync()
        out = dev.d2h(self.m["sampled"], (_SAMPLED_BYTES,), np.uint8)
        drafts = np.zeros(self.mtp, np.int64)
        drafts[0] = _sampled_fields(out, 1, 1)[0][0]
        return drafts

    def verify_launch(self, rows: Sequence[Row], anchors, drafts, sampling) -> Launched:
        """One speculative step of each row left running: the target over [anchor, d_1 .. d_K] (column b Tq + t)
        and the MTP layer at every column (token y_t, the target's hidden at the column)."""
        Tq, B = self.mtp + 1, len(rows)
        T = bucket(B * Tq, False)
        tokens = np.concatenate([np.asarray(anchors, np.int64)[:, None], np.asarray(drafts, np.int64)], 1)
        positions = np.concatenate([r.pos0 + np.arange(Tq) for r in rows])
        target = [
            replace(r, column=b * Tq, real=Tq, state_dst=-1) for b, r in enumerate(rows)
        ]
        self._launch(tokens.ravel(), positions, target, verify=Tq)
        params = np.zeros((T, PARAMS), np.float32)
        if sampling is not None:  # column t samples position pos0 + t + 1
            params[: B * Tq] = sampling.grid(B, range(Tq)).reshape(B * Tq, -1)
        staged = self.sample_async(T, params)
        # the target's hidden, every column: the head wrote it
        m_rows = [Row(b * Tq, r.pos0, Tq, r.mtp_blocks, -1, -1) for b, r in enumerate(rows)]
        mp = np.zeros((T, PARAMS), np.float32)
        if sampling is not None:  # column t drafts position pos0 + t + 2
            mp[: B * Tq] = sampling.grid(B, range(1, Tq + 1)).reshape(B * Tq, -1)
        cols = np.arange(T) + 1
        self._mtp_pass(
            T, T, m_rows, cols, self.sampled["tok_out"], self.m["mtp_zero"], self.io["hf"], mp
        )
        staged_drafts = self._sample_mtp_stage(T, B * Tq)
        return Launched(rows, np.asarray(drafts, np.int64), sampling, staged, staged_drafts, T)

    def verify_finish(self, launched: Launched) -> Verified:
        """Accept each row's drafts against the verify pass's samples and commit the accepted tokens. No collective
        runs here: the Engine's driver harvests a step before its followers do, so what a harvest runs must be
        local to the rank. ``Verified.drafts`` holds d_1; ``chain`` completes them, at a point every rank reaches
        in the same order."""
        rows, drafts, T = launched.rows, launched.drafts, launched.T
        Tq, B = self.mtp + 1, len(rows)
        y = launched.staged.result()[0][: B * Tq].reshape(B, Tq)
        q = launched.staged_drafts.result()[0].reshape(B, Tq)
        acceptance = self.speculation.accept(drafts.tolist(), y.tolist())
        accepted = np.asarray(acceptance.accepted_counts, np.int64)
        self._commit(T, rows, accepted)
        next_drafts = np.zeros((B, self.mtp), np.int64)
        next_drafts[:, 0] = q[np.arange(B), accepted]
        columns = [b * Tq + np.arange(accepted[b] + 1) for b in range(B)]
        return Verified(acceptance.token_rows, accepted, next_drafts, columns)

    def _commit(self, T: int, rows: Sequence[Row], accepted: np.ndarray) -> None:
        """Apply each row's first accepted + 1 tokens' KDA updates and store the hidden of its last one, into the
        STATE block of that position."""
        Tq, L = self.mtp + 1, len(self.kda)
        Bc = T // Tq
        g, dev, m = self.geo, self.device, self.m
        bases = np.full((max(L, 1), Bc, 4), -1, np.int32)
        counts = np.zeros((max(L, 1), Bc), np.int32)
        table = np.full((Bc, 2), -1, np.int32)
        for b, (r, a) in enumerate(zip(rows, accepted, strict=True)):
            dst = r.state_at(r.pos0 + int(a))
            table[b] = b * Tq + int(a), g.hidden_row(dst)
            counts[:, b] = int(a) + 1
            for nth in range(L):
                ss, cs = g.state_rows(r.state_src, nth)
                sd, cd = g.state_rows(dst, nth)
                bases[nth, b] = cs, ss, cd, sd
        dev.h2d_nb(m["commit_bases"], bases)
        dev.h2d_nb(m["commit_counts"], counts)
        dev.h2d_nb(m["hidden_table"], table)
        key = (Bc, "commit")
        op = self._mtp_programs.get(key)
        if op is None:
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
        op.run(dev.lib)

    def chain(self, launched: Launched, done: Verified) -> np.ndarray:
        """Each row's drafts d_1 .. d_K: link j (MTP at position anchor + j - 1, token d_j, hidden: link j - 1's output,
        the first link's the accepted column of the last MTP pass) samples d_(j+1). Runs collectives: call it only where
        every rank does, before the next launch overwrites the verify pass's MTP hiddens."""
        rows, accepted, sampling = launched.rows, done.accepted, launched.sampling
        K, B, Tq = self.mtp, len(rows), self.mtp + 1
        drafts = done.drafts.copy()
        first = drafts[:, 0]
        if K == 1:
            return drafts
        dev, m = self.device, self.m
        bt = bucket(B, False)
        # the verify pass's MTP hiddens, by accepted column: the chain's first hiddens
        src = m["hf"][0] + 4 * self.d.hidden * (np.arange(B) * Tq + accepted)
        dev.d2d_multi(
            src.astype(np.uint64),
            np.full(B, 4 * self.d.hidden, np.uint64),
            (m["hf"][1] + 4 * self.d.hidden * np.arange(B)).astype(np.uint64),
        )
        anchors = [r.pos0 + int(a) + 1 for r, a in zip(rows, accepted, strict=True)]
        hf = m["hf"][1]
        tokens = np.zeros(bt, np.int64)
        tokens[:B] = first
        cols = np.arange(bt) + 1
        for j in range(1, K):
            link = [Row(b, anchors[b] + j - 1, 1, r.mtp_blocks, -1, -1) for b, r in enumerate(rows)]
            params = np.zeros((bt, PARAMS), np.float32)
            if sampling is not None:
                params[:B] = sampling.at(range(B), [int(a) + j + 1 for a in accepted])
            out_hf = m["hf"][0] if hf == m["hf"][1] else m["hf"][1]
            self._mtp_pass(bt, bt, link, cols, tokens, m["mtp_zero"], hf, params, out_hf)
            dev.sync()
            out = dev.d2h(self.m["sampled"], (_SAMPLED_BYTES,), np.uint8)
            tok = _sampled_fields(out, bt, B)[0]
            drafts[:, j] = tok
            tokens[:B] = tok
            hf = out_hf
        return drafts

    def _index_launches(self, rows: Sequence[Row], query_block: int, S: int) -> list[_IndexLaunch]:
        """Index slots of a step: per row and query block, every 2048-pool chunk of the pools it may see,
        packed S per launch. Keys are pools; qpos0 past every pool, since visibility is applied per column
        after the scores (ix_thr, gx_mask, ix_list use ``vis``)."""
        per_chunk = SLOT_POOLS // PAGE_POOLS
        out: list[tuple[list[int], np.ndarray]] = []
        NI = self.d.index_heads
        for r in rows:
            total = (r.pos0 + r.real) // POOL
            for t0 in range(0, r.real, query_block):
                seen = min((r.pos0 + t0 + query_block) // POOL, total)
                for key0 in range(0, seen, SLOT_POOLS):
                    first = key0 // PAGE_POOLS
                    live = min(per_chunk, -(-(seen - key0) // PAGE_POOLS))
                    pages = np.full(per_chunk, -1, np.int64)
                    pages[:live] = r.blocks[first : first + live]
                    rec = [0, NI * (r.column + t0), 1 << 30, key0, seen, r.column + t0, 0, 0]
                    out.append((rec, pages))
        launches = []
        for a in range(0, len(out), S):
            part = out[a : a + S]
            slots = np.zeros((S, 8), np.int32)
            slots[:, 0] = -1
            pages = np.full((S, per_chunk), -1, np.int64)
            for s, (rec, pg) in enumerate(part):
                slots[s] = rec
                pages[s] = pg
            launches.append(_IndexLaunch(slots, pages))
        return launches

    def _launch(
        self,
        tokens: np.ndarray | None,
        positions: np.ndarray,
        rows: Sequence[Row],
        prefill: bool = False,
        verify: int = 0,
    ) -> int:
        """Enqueue a step (through the head and the logits all-gather); returns its bucket. ``tokens`` None:
        a decode continuation, whose tokens are the previous step's samples on the device, row for row.
        ``verify``: Tq tokens per row, in columns b Tq + t: KDA state is recorded, not written."""
        dev, d = self.device, self.d
        n = len(positions)
        T = bucket(n, prefill)
        tb = self.tables[T]
        qb = T if prefill else 1
        S = SLOTS[qb]
        np.zeros(T, np.int32)
        np.zeros(T, np.int32)
        k = self._host_turn
        self._host_turn ^= 1
        if self._host_pending[k]:
            dev.wait(_HOST_EVENT + k)
        h = self.host_tables[k]
        h[: tb.HEADER].view(np.int32)[: tb.HEADER // 4] = 0
        if tokens is not None:
            h[tb.TOK : tb.TOK + 4 * n].view(np.int32)[:] = tokens
        # real columns: the router sends padding nowhere and the chunked KDA passes it through
        h[tb.META : tb.META + 16].view(np.int32)[:] = (int(positions[0]), n, 0, 0)
        if prefill:
            self._prefill_last = n - 1
            h[tb.HEAD : tb.HEAD + 4 * HEAD_COLS].view(np.int32)[:] = n - 1
        launches, ix_launches, index, sparse = self._attention_tables(
            h, tb, rows, T, qb, S, self._dsa_off, keep_tails=verify > 0
        )
        len(self.dsa)
        K = len(self.kda)
        if K:
            self._kda_tables(h, tb, rows, T, K, prefill, verify)
        self._upload(self.table, h, tb, launches, ix_launches, sparse)
        dev.record(_HOST_EVENT + k)
        self._host_pending[k] = True
        # ---- launches
        lib, prog = dev.lib, self._program(T, index, sparse, verify)
        gather = d.gathered(T)
        ar = None
        if self.nranks > 1:
            ar = lib.rg_ccl_allgather if gather else lib.rg_ccl_allreduce
        dst_a, dst_b = (self.pa_all, self.pb_all) if gather else (self.pa, self.pb)
        count = d.hidden * T
        (prog.embed if tokens is not None else prog.embed_device).run(lib)
        nsl, nix = 1 if sparse else len(launches), len(ix_launches)
        probe = self.probe
        for i, (attn, ix, ix_end, slots, post, ffn) in enumerate(prog.layers):
            attn.run(lib)
            for op in ix[:nix]:
                op.run(lib)
            if ix_end is not None:
                ix_end.run(lib)
            for op in slots[:nsl]:
                op.run(lib)
            if post is not None:
                post.run(lib)
            if ar is not None and ar(self.pa, dst_a, count, 0):
                dev.check(-1, "residual collective")
            if probe is not None:
                probe(i, "attn", T, dst_a)
            ffn.run(lib)
            if ar is not None and ar(self.pb, dst_b, count, 0):
                dev.check(-1, "residual collective")
            if probe is not None:
                probe(i, "mlp", T, dst_b)
        prog.head.run(lib)
        if self.nranks > 1:
            dev.allgather(self.logits, self.logits_all, d.vocab * min(T, HEAD_COLS))
        self._last_program = prog
        return T

    def _attention_tables(self, h, tb, rows, T, qb, S, offsets, keep_tails=False):
        """Fill a step's attention tables for layers at page ``offsets``: visibility, key and pool stores, slot and
        index records, the sparse block table. -> (key launches, index launches, indexer on, sparse)"""
        d = self.d
        vis = np.zeros(T, np.int32)
        klen = np.zeros(T, np.int32)
        store, pool = _store_rows(T, rows, keep_tails)
        for r in rows:
            p = r.pos0 + np.arange(r.real)
            vis[r.column : r.column + r.real] = (p + 1) // POOL
            klen[r.column : r.column + r.real] = p + 1
        index = bool(vis.max() > d.top_pools)
        sparse = index and qb == 1
        slot_rows = [
            SlotRow(r.column, r.pos0, r.real, r.blocks[None, :], r.blocks[None, :]) for r in rows
        ]
        launches = [] if sparse else plan_slots(slot_rows, qb, d.mla_heads, S)
        ix_launches = self._index_launches(rows, qb, S) if index else []
        if len(launches) > tb.key_launches or len(ix_launches) > tb.index_launches:
            raise ValueError("a step's slot launches exceed the table capacity")
        h[tb.VIS : tb.VIS + 4 * T].view(np.int32)[:] = vis
        h[tb.KLEN : tb.KLEN + 4 * T].view(np.int32)[:] = klen
        M = len(offsets)
        if M:
            off = offsets[:, None, None]
            so = h[tb.STORE : tb.STORE + M * tb.ROWS_BYTES].view(np.int32)
            so = so.reshape(M, tb.ROWS_BYTES // (4 * ROWS_FIELDS), ROWS_FIELDS)
            so[:] = -1
            live = store[:, 0] >= 0
            so[:, :T, 0] = np.where(live, store[:, 0][None] + off[:, :, 0], -1)
            so[:, :T, 3] = np.where(live, store[:, 3][None] + off[:, :, 0], -1)
            so[:, :T, 4] = np.where(live, store[:, 4][None], -1)
            po = h[tb.POOL : tb.POOL + M * tb.ROWS_BYTES].view(np.int32)
            po = po.reshape(M, tb.ROWS_BYTES // (4 * ROWS_FIELDS), ROWS_FIELDS)
            po[:] = -1
            live = pool[:, 0] >= 0
            po[:, :T, 0] = np.where(live, pool[:, 0][None] + 2 * off[:, :, 0], -1)
            for s in (1, 2, 3):
                # a column >= 0, or -(arena row + 1): tails shift with the layer
                po[:, :T, s] = np.where(
                    pool[:, s] < 0, pool[:, s][None] - off[:, :, 0], pool[:, s][None]
                )
            tail = pool[:, 4] >= 0
            po[:, :T, 4] = np.where(tail, pool[:, 4][None] + off[:, :, 0], -1)
        for j, launch in enumerate(launches):
            rec = h[tb.slot(j, 0) : tb.slot(j, 0) + M * tb.SLOT_RECORD].view(np.int32)
            rec = rec.reshape(M, -1)
            pages = launch.k_pages[None]
            rec[:, : 8 * S] = launch.slots.ravel()
            lat = np.where(pages >= 0, pages + offsets[:, None, None], -1)
            o = tb.SLOTS_BYTES // 4
            rec[:, o : o + pages[0].size] = lat.reshape(M, -1)
        for j, launch in enumerate(ix_launches):
            rec = h[tb.index(j, 0) : tb.index(j, 0) + M * tb.INDEX_RECORD].view(np.int32)
            rec = rec.reshape(M, -1)
            pages = launch.pages[None]
            rec[:, : 8 * S] = launch.slots.ravel()
            ixp = np.where(pages >= 0, pages + offsets[:, None, None] + LATENT_PAGE_ROWS, -1)
            o = tb.SLOTS_BYTES // 4
            rec[:, o : o + pages[0].size] = ixp.reshape(M, -1)
        if sparse:
            bt = h[tb.BLOCKS : tb.BLOCKS + 4 * self.pages * T].view(np.int32)
            bt = bt.reshape(T, self.pages)
            bt[:] = -1
            for r in rows:
                bt[r.column : r.column + r.real, : len(r.blocks)] = r.blocks
        return launches, ix_launches, index, sparse

    def _upload(self, base, h, tb, launches, ix_launches, sparse) -> None:
        dev = self.device
        dev.h2d_async(base, h[: tb.BLOCKS])
        if sparse:
            dev.h2d_async(base + tb.BLOCKS, h[tb.BLOCKS : tb.BLOCKS + 4 * self.pages * tb.T])
        if launches:
            dev.h2d_async(base + tb.slot(0, 0), h[tb.slot(0, 0) : tb.slot(len(launches), 0)])
        if ix_launches:
            dev.h2d_async(base + tb.index(0, 0), h[tb.index(0, 0) : tb.index(len(ix_launches), 0)])

    def _kda_tables(self, h, tb, rows, T, K, prefill, verify=0) -> None:
        g = self.geo
        if prefill:
            r = rows[0]
            a = np.full((K, tb.KDA_RECORD // 4), -1, np.int32)
            # conv_src, ssm_src, ssm_dst, conv_dst 256 bytes apart
            for nth in range(K):
                ss, cs = g.state_rows(r.state_src, nth)
                sd, cd = g.state_rows(r.state_dst, nth)
                a[nth, 0], a[nth, 64], a[nth, 128], a[nth, 192] = cs, ss, sd, cd
        else:
            a = np.full((K, tb.KDA_RECORD // 4), -1, np.int32)
            bases = a[:, : tb.BASES_BYTES // 4].reshape(K, -1, 4)
            cdst = a[:, tb.BASES_BYTES // 4 :]
            for r in rows:
                c = r.column // verify if verify else r.column
                for nth in range(K):
                    ss, cs = g.state_rows(r.state_src, nth)
                    sd, cd = g.state_rows(r.state_dst, nth)
                    bases[nth, c] = cs, ss, -1, sd
                    cdst[nth, c] = cd
        h[tb.KDA : tb.KDA + a.nbytes].view(np.int32)[:] = a.ravel()


def _store_rows(T: int, rows: Sequence[Row], keep_tails: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per column, relative to the layer's pages: ``store`` (T, 5) [latent row, -, -, scale row, scale index] for
    ``mla_store``, ``pool`` (T, 5) [pool unit, src0, src1, src2, tail row] for ``gx_pool``. Rows are block rows
    plus the fixed page offsets; the caller adds the layer offset (a tail source -(row + 1) subtracts it)."""
    store = np.full((T, 5), -1, np.int64)
    pool = np.full((T, 5), -1, np.int64)
    # sources default to column 0: a column completing no pool never reads them
    pool[:, 1:4] = 0
    for r in rows:
        pos = r.pos0 + np.arange(r.real)
        cols = r.column + np.arange(r.real)
        b = np.asarray(r.blocks, np.int64)[pos // PAGE_TOKENS]
        slot = pos % PAGE_TOKENS
        store[cols, 0] = b + slot
        store[cols, 3] = b + PAGE_TOKENS
        store[cols, 4] = slot
        last = r.pos0 + r.real - 1
        done = pos % POOL == POOL - 1
        P = pos // POOL
        pool[cols[done], 0] = 2 * (b[done] + LATENT_PAGE_ROWS) + P[done] % PAGE_POOLS
        for s in range(3):
            q = pos[done] - 3 + s
            inside = q >= r.pos0
            pool[cols[done], 1 + s] = np.where(
                inside,
                cols[done] - 3 + s,
                -(b[done] + TAIL_OFFSET + tail_rows(q) + 1),
            )
        # the pool of an unfinished token: its last token is past this launch, so a later launch reads it. A verify
        # step keeps every tail (``keep_tails``): a rejected draft restarts the next step inside any of its pools
        open_ = pos % POOL != POOL - 1
        if not keep_tails:
            open_ &= P * POOL + POOL - 1 > last
        pool[cols[open_], 4] = b[open_] + TAIL_OFFSET + tail_rows(pos[open_])
    return store, pool


def _sampled_fields(out: np.ndarray, R: int, n: int):
    """(tokens, logprobs, top ids, top logprobs) of the first n of R sampled columns"""

    def field(name, dtype, shape):
        off = _SAMPLED[name]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return out[off : off + size].view(dtype).reshape(shape)[:n].copy()

    return (
        field("tok_out", np.int32, (R,)),
        field("lp_out", np.float32, (R,)),
        field("top_idx", np.int32, (R, TOP_MAX)),
        field("top_lp", np.float32, (R, TOP_MAX)),
    )


class Staged:
    """Pinned host copy of one enqueued step's sampler outputs, and its device event."""

    def __init__(self, device: Device, event: int) -> None:
        self.device, self.event = device, event
        self.buffer = device.host_buffer(_SAMPLED_BYTES, np.uint8)
        self._shape: tuple[int, int] | None = None
        self._result = None

    def arm(self, R: int, n: int) -> None:
        self._shape, self._result = (R, n), None

    def result(self):
        if self._result is None:
            assert self._shape is not None, "nothing was staged"
            self.device.wait(self.event)
            self._result = _sampled_fields(self.buffer, *self._shape)
        return self._result

    def drain(self) -> None:
        if self._shape is not None and self._result is None:
            self.result()



def _do_bind(lib, args):
    """rg_bind_slots with a counter: the step profile reports binds per step, so a flag that is set
    but never consulted on the running path is visible before any token comparison."""
    LAUNCH_STATS["binds"] = LAUNCH_STATS.get("binds", 0) + 1
    return lib.rg_bind_slots(*args)


def _do_launch(lib, handle):
    LAUNCH_STATS["launches"] = LAUNCH_STATS.get("launches", 0) + 1
    return lib.rg_launch(handle)



def diagnostics() -> str:
    """The resolved diagnostics, read from the constants the code branches on (not os.environ), so a
    boot log cannot disagree with the behaviour."""
    from resin_models_gaudi.synapse.graph import _PROFILE_STEP

    return (
        f"diagnostics: lazy_bind={_LAZY_BIND} launch_mode={_LAUNCH_MODE} "
        f"sync_every={_SYNC_EVERY} step_profile={_PROFILE_STEP}"
    )


class _Op:
    """One launch: a recipe and its bindings as prepared slot / address arrays."""

    __slots__ = ("recipe", "slots", "addresses", "args", "_bound")

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
        if _PROFILE_STEP:
            # the launch path every Gaudi model uses: one bind plus one launch per op.
            # RESIN_GAUDI_PROFILE_LAUNCH_MODE separates the cost: "full" (normal), "bind" (bind only,
            # no launch), "none" (neither). "bind" and "none" produce garbage tokens and exist only to
            # time the host floor - never leave them on for a correctness run.
            started = time.perf_counter()
            if _LAUNCH_MODE == "none":
                ok = True
            elif _LAUNCH_MODE == "bind":
                ok = _do_bind(lib, self.args) >= 0
            elif _LAZY_BIND:
                # the profiling branch has to honour lazy bind too, or a gate run measures the
                # unmodified path while reporting the flag as on (found the hard way)
                if not getattr(self, "_bound", False):
                    ok = _do_bind(lib, self.args) >= 0
                    self._bound = ok
                else:
                    ok = True
                if ok and _do_launch(lib, self.args[0]) < 0:
                    ok = False
            elif _SYNC_EVERY:
                # the clean host/device separator: never let the queue build, so the per-call time
                # collapses to the pure host call cost if blocking was what we were measuring
                ok = _do_bind(lib, self.args) >= 0 and _do_launch(lib, self.args[0]) >= 0
                _SYNC_COUNT[0] += 1
                # the sync is deliberately OUTSIDE the timed interval: inside it, one call in N
                # carried a full device wait and the median rose instead of collapsing (15j)
                _SYNC_PENDING[0] = _SYNC_COUNT[0] % _SYNC_EVERY == 0
            else:
                ok = _do_bind(lib, self.args) >= 0 and _do_launch(lib, self.args[0]) >= 0
            elapsed = time.perf_counter() - started
            if _SYNC_PENDING[0]:
                _SYNC_PENDING[0] = False
                _sync_started = time.perf_counter()
                self.recipe.device.sync()
                _SYNC_TOTAL[0] += time.perf_counter() - _sync_started
                _SYNC_TIMES.append(time.perf_counter() - _sync_started)
            if not ok:
                self.recipe.device.check(-1, f"launch {self.recipe.name}")
            LAUNCH_STATS["calls"] += 1
            LAUNCH_STATS["seconds"] += elapsed
            LAST_CALLS.append(elapsed)
            entry = LAUNCH_BY_RECIPE.get(self.recipe.name)
            if entry is None:
                # workspace is a benign per-recipe query: cached once, never launched
                entry = LAUNCH_BY_RECIPE[self.recipe.name] = [0, 0.0, self.recipe.workspace]
            entry[0] += 1
            entry[1] += elapsed
            return
        if _LAZY_BIND:
            if not getattr(self, "_bound", False):
                if _do_bind(lib, self.args) < 0:
                    self.recipe.device.check(-1, f"bind {self.recipe.name}")
                self._bound = True
            if _do_launch(lib, self.args[0]) < 0:
                self.recipe.device.check(-1, f"launch {self.recipe.name}")
            return
        if _do_bind(lib, self.args) < 0 or _do_launch(lib, self.args[0]) < 0:
            self.recipe.device.check(-1, f"launch {self.recipe.name}")


class _Program:
    def __init__(self) -> None:
        self.embed: _Op | None = None
        self.embed_device: _Op | None = None
        self.head: _Op | None = None
        self.head_scored: _Op | None = None
        self.layers: list[tuple] = []


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


@dataclass(frozen=True, slots=True)
class _IndexLaunch:
    slots: np.ndarray
    pages: np.ndarray


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
        self.BLOCKS = r(self.KDA_BASE + kda * self.KDA_RECORD)
        self.pages = pages
        self.slot_base = -(-(self.BLOCKS + 4 * pages * T) // 4096) * 4096
        self.index_base = self.slot_base + key_launches * dsa * self.SLOT_RECORD
        self.nbytes = self.index_base + index_launches * dsa * self.INDEX_RECORD

    def store(self, nth: int) -> int:
        return self.STORE + nth * self.ROWS_BYTES

    def pool(self, nth: int) -> int:
        return self.POOL + nth * self.ROWS_BYTES

    def kda(self, nth: int) -> int:
        return self.KDA_BASE + nth * self.KDA_RECORD

    def slot(self, launch: int, nth: int) -> int:
        return self.slot_base + (launch * self.dsa + nth) * self.SLOT_RECORD

    def index(self, launch: int, nth: int) -> int:
        return self.index_base + (launch * self.dsa + nth) * self.INDEX_RECORD


_SAMPLED = {
    "tok_out": 0,
    "lp_out": 256,
    "top_idx": 512,
    "top_lp": 512 + 4 * TOP_MAX * MAX_T,
}
_SAMPLED_BYTES = 512 + 8 * TOP_MAX * MAX_T
_SAMPLE_TABLE = 4 * PARAMS * MAX_T + 256
_SAMPLE_LAST = 4 * PARAMS * MAX_T
_HOST_EVENT = 60
_STAGE_EVENT = 56
_MTP_EVENT = 52  # 52, 53: the MTP drafts' staging copies; 54, 55: its pinned tables
_MHOST_EVENT = 54
