"""Mistral-Large-4 forward steps on one Gaudi card: one tensor-parallel rank of ``nranks``.

A step runs T <= 16 columns: one sequence's prefill chunk (one query block) or one decode token
for each of up to 16 sequences. The host uploads the step's tables in one copy (tokens, rope
rows and log(pos + 1) per column, MoE meta, per-layer store rows and slot tables), then
launches embed, per layer ``attn_pre``, the attention slot launches, ``attn_post`` and
``ffn``, then the head. Between ``attn_post`` / ``ffn`` and the next residual graph the partial
sum is all-reduced across ranks; the logits slices are all-gathered.

The residual alternates between two buffers (``x`` -> ``x_out``) and so do the partials
(``attn_post`` writes ``pa``, ``ffn`` reads it and writes ``pb``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from resin.models.mistral_large4.config import MistralLarge4Config
from resin_models_gaudi.gguf.reader import GgufModel
from resin_models_gaudi.mistral_large4 import dspark
from resin_models_gaudi.mistral_large4 import layers as L
from resin_models_gaudi.mistral_large4.cache import (
    ARENA_BLOCK_ROWS,
    PAGE_TOKENS,
    ROW_BYTES,
    STORE_FIELDS,
    CacheGeometry,
)
from resin_models_gaudi.mistral_large4.layers import SLOT_KEYS, AttnKind, Dims
from resin_models_gaudi.mistral_large4.weights import (
    RankSlices,
    lerope_alpha,
    prepare,
    rope_table,
)
from resin_models_gaudi.qwen3_5.attention import SlotRow, plan_slots
from resin_models_gaudi.qwen3_5.sampling import PARAMS, TOP_MAX
from resin_models_gaudi.synapse.graph import Graph, Recipe, RecipeCache
from resin_models_gaudi.synapse.progress import info
from resin_models_gaudi.synapse.runtime import Device

BUCKETS = (1, 2, 4, 8, 16)  # decode batches, short prefill chunks
MAX_T = BUCKETS[-1]
# the widest prefill launch
PREFILL_COLUMNS = 2048
# buckets from this width use the ``mme`` graphs
DENSE_MME_T = 4
GATHER_T = 4
# a prefill chunk's head runs at HEAD_COLS of its columns (the sampled column, or a group of
# positions prompt logprobs score)
HEAD_COLS = MAX_T
WORKSPACE_RESERVE = 1 << 29  # bytes kept out of the arena


def prefill_buckets(columns: int) -> tuple[int, ...]:
    """The prefill launch widths of a server whose widest launch is ``columns``. A shorter chunk (a
    short prompt, a long prompt's tail) runs at its own width."""
    if columns < 1024 or columns & (columns - 1):
        raise ValueError(f"prefill chunks are a power of two >= 1024, not {columns}")
    return (128, 1024) if columns == 1024 else (128, 1024, columns)


def prefill_chunks(n: int, columns: int = PREFILL_COLUMNS) -> range:
    """chunk starts of an n-token prefill, ``step`` tokens per chunk: MAX_T per GEMV launch for a
    prompt of up to 3 MAX_T; ``columns`` per launch when more than half of them are real, else
    1024"""
    if n <= 3 * MAX_T:
        return range(0, n, MAX_T)
    return range(0, n, columns if n > columns // 2 else 1024)


def slots_per_launch(query_block: int) -> int:
    """a prefill launch's one query block takes one slot; per-column steps take 16"""
    return 16 if query_block == 1 else 1


@dataclass(frozen=True, slots=True)
class Row:
    """One sequence of a step: its first column, first position, real tokens and the first
    arena rows of its KV blocks (block k holds positions 128 k .. 128 k + 127)."""

    column: int
    pos0: int
    real: int
    blocks: np.ndarray


class _Arena:
    def __init__(self, device: Device, blocks: int) -> None:
        self.blocks = blocks
        self.nbytes = blocks * ARENA_BLOCK_ROWS * ROW_BYTES
        self.address = device.malloc(self.nbytes)
        device.memset32(self.address, 0, self.nbytes // 4)


class Ml4Gaudi:
    def __init__(
        self,
        device: Device,
        gguf: GgufModel,
        cfg: MistralLarge4Config,
        *,
        capacity: int | Callable[[int], int],
        max_seq_len: int,
        rank: int = 0,
        nranks: int = 1,
        recipes: RecipeCache | None = None,
        log: Callable[[str], None] = info,
        gather_tokens: int = GATHER_T,
        draft: Path | str | None = None,
        prefill_columns: int = PREFILL_COLUMNS,
    ) -> None:
        self.device, self.cfg, self.log = device, cfg, log
        self.rank, self.nranks = rank, nranks
        self.prefill_columns = prefill_columns
        self.prefill_buckets = prefill_buckets(prefill_columns)
        self.buckets = (*BUCKETS, *self.prefill_buckets)
        positions = max_seq_len + prefill_columns
        self.index_keys = (
            -(-positions // SLOT_KEYS) * SLOT_KEYS
        )  # DSA score scratch rows
        if 4 * self.index_keys * prefill_columns >= 1 << 31:
            raise ValueError(
                f"{prefill_columns} prefill columns over {self.index_keys} keys: "
                "the DSA mask would be too large"
            )
        self.sl = RankSlices.of(cfg, rank, nranks)
        self.d = replace(
            Dims.of(cfg, self.sl),
            ranks=nranks,
            gather_tokens=gather_tokens,
            exact=draft is not None,
        )
        # the DSpark drafter (``draft``: the release's checkpoint directory, params.json and
        # the drafter's safetensors shard), tensor parallel like the target
        self.dd = None
        if draft is not None:
            dspark.drafter_config(draft)
            self.dd = dspark.drafter_dims(cfg, rank, nranks, gather_tokens)
        self.geo = CacheGeometry.from_config(
            cfg, dspark.DRAFT_LAYERS if draft is not None else 0
        )
        self.max_seq_len = max_seq_len
        self.recipes = recipes or RecipeCache(device)
        t0 = time.time()
        self.w: dict[str, int] = {}
        self.kind: dict[str, str] = {}
        nbytes = 0
        device.begin_load(gguf.total_bytes // nranks, log=log)
        for name, array, kind in prepare(gguf, cfg, rank, nranks):
            a = np.ascontiguousarray(array)
            address = device.malloc(max(a.nbytes, 256))
            device.h2d(address, a)
            self.w[name], self.kind[name] = address, kind
            nbytes += a.nbytes
        if draft is not None:
            t1, dbytes = time.time(), 0
            for name, array, kind in dspark.prepare_drafter(draft, cfg, rank, nranks):
                a = np.ascontiguousarray(array)
                address = device.malloc(max(a.nbytes, 256))
                device.h2d(address, a)
                self.w[name], self.kind[name] = address, kind
                dbytes += a.nbytes
            nbytes += dbytes
            log(
                f"ml4: DSpark drafter {dbytes / 2**30:.2f} GiB in {time.time() - t1:.0f} s"
            )
        device.end_load()
        log(
            f"ml4 rank {rank}/{nranks}: {nbytes / 2**30:.1f} GiB of weights in {time.time() - t0:.0f} s"
        )
        self.pages = (
            self.index_keys // PAGE_TOKENS
        )  # block-table rows per column (sparse)
        self.rope = rope_table(cfg, lerope_alpha(gguf), positions)
        self._programs: dict[tuple, _Program] = {}
        self._staging = [Staged(device, _STAGE_EVENT + k) for k in range(2)]
        self._stage_turn = 0
        self._sample_ops: dict[tuple, _Op] = {}
        self.sparse = True  # sparse decode attention when the indexer runs
        self._capture_slots: dict[int, int] = {}
        self.hidden = 0
        self.merged = True  # decode attention as one launch per layer
        self._last_program: _Program | None = None
        self._prefill_last = -1
        self._buffers()
        self._layer_tables()
        self.drafter: dspark.Drafter | None = None
        if draft is not None:
            self.capture(dspark.DRAFT_TAPS)
            self.drafter = dspark.Drafter(self)
        if callable(capacity):  # sized from the memory left after weights, buffers and workspace
            self._hold_workspace()
            capacity = capacity(device.memory_info()[0] - WORKSPACE_RESERVE)
        self.capacity = capacity
        self.arena = _Arena(device, self.geo.arena_blocks(capacity))
        log(
            f"ml4: cache {capacity} blocks ({capacity * PAGE_TOKENS} tokens), "
            f"{capacity * self.geo.block_bytes / 2**30:.1f} GiB per card"
        )
        log("preparing recipes")
        self._compile()

    # ------------------------------------------------------------------ setup
    def bucket(self, n: int) -> int:
        for b in self.buckets:
            if n <= b:
                return b
        raise ValueError(f"{n} columns exceed {self.prefill_columns}")

    def qtype(self, name: str) -> str:
        kind = self.kind[name]
        assert kind.startswith("planar:"), (name, kind)
        return kind[7:]

    def attn_kind(self, layer: int, index_queries: bool = False) -> AttnKind:
        c = self.cfg
        owns_index = c.index_owners[layer]
        return AttnKind(
            c.rope_layers[layer],
            c.owns_kv(layer),
            owns_index,
            index_queries and owns_index,
        )

    def _buffers(self) -> None:
        d, dev = self.d, self.device
        H, T = d.hidden, self.prefill_columns

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
            "zero1": const(np.zeros(64, f)),
            "one1": const(np.ones(64, f)),
            "negpad": const(np.full((T, L.ROUTER_LANES - d.experts), -3e38, f)),
            "zeroH": const(np.zeros(H * T, f)),
        }
        for n in {b for t in self.buckets for b in (t, d.top * t)}:
            self.consts[f"ones{n}"] = const(np.ones(n, f))
        for t in self.prefill_buckets:
            Tq = d.index_tokens(t)
            self.consts[f"ix_cols{t}"] = const(
                np.arange(self.rank * Tq, (self.rank + 1) * Tq, dtype=np.int32)
            )
        self.pzero = const(np.zeros(H * T, f))
        self.x = [buf(4 * H * T), buf(4 * H * T)]
        self.pa, self.pb = buf(4 * H * T), buf(4 * H * T)
        # the ranks' partials all-gathered (Dims.gathered launches): (H, T, ranks) each
        G = 4 * H * d.gather_tokens * self.nranks
        self.pa_all, self.pb_all = buf(G), buf(G)
        W, Tq = d.heads * T, max(MAX_T, d.index_tokens(T))
        self.io = {
            "q_all": buf(2 * 576 * W),
            "acc_o": buf(4 * d.latent * W),
            "acc_m": buf(4 * W),
            "acc_l": buf(4 * W),
            "ix_q": buf(2 * d.index_dim * d.index_heads * T),
            "ix_w": buf(4 * d.index_heads * T),
            "sink_kv": buf(256),
            "sink_acc": buf(256),
            "sink_slots": buf(256),
            "sink_ix": buf(256),
            # a rank's columns: a decode step's, or its share of a prefill chunk's
            "ix_score": buf(4 * self.index_keys * Tq),
            "ix_thr": buf(4 * 64 * Tq),
            # a prefill chunk's selection: this rank's columns' bits, every rank's, their mask
            "ix_bits_own": buf(self.index_keys // 8 * d.index_tokens(T)),
            "ix_bits": buf(self.index_keys // 8 * T),
            "ix_mask": buf(4 * self.index_keys * T),
            # sparse decode attention (MAX_T columns): the DSA key lists and slot records
            "ix_list": buf(4 * self.cfg.index_topk * MAX_T),
            "ix_slots": buf(4 * 8 * MAX_T),
        }
        self.logits = buf(4 * d.vocab * MAX_T)
        self.logits_all = (
            buf(4 * d.vocab * MAX_T * self.nranks) if self.nranks > 1 else self.logits
        )
        self.head_cols = buf(4 * HEAD_COLS)  # prompt logprobs' columns (head_logits)
        # sampler outputs in one buffer, read back with one copy
        self.sampled_buf = buf(_SAMPLED_BYTES)
        self.sampled = {n: self.sampled_buf + off for n, off in _SAMPLED.items()}
        self.consts["inv_cap"] = const(
            np.full(64, 1.0 / self.cfg.final_logit_softcap, f)
        )
        self.consts["cap"] = const(np.full(64, self.cfg.final_logit_softcap, f))
        self.sample_table = buf(4096)
        # per-step tables (_Tables) and their pinned host copies
        launches = -(-self.index_keys // SLOT_KEYS)
        self.tables = {
            T: _Tables(T, self.cfg.num_hidden_layers, launches, self.pages)
            for T in self.buckets
        }
        nbytes = max(tb.nbytes for tb in self.tables.values())
        self.table = buf(nbytes)
        self.host_tables = [dev.host_buffer(nbytes, np.uint8) for _ in range(2)]
        self._host_turn = 0
        self._host_pending = [False, False]

    def _graphs(self) -> dict[tuple, Graph]:
        dev, d, c, nb = self.device, self.d, self.cfg, self.arena.blocks
        graphs: dict[tuple, Graph] = {}
        layer_qt = lambda n: self.qtype(n)  # noqa: E731
        head_qt = layer_qt("head")
        for T in self.buckets:
            prefill = T > MAX_T  # one query block of the chunk, sampled at its end
            # dense weights on the MME; with ``exact``, at every width
            mme = d.exact or T >= DENSE_MME_T
            graphs["embed", T] = L.embed_graph(dev, d, T, c.vocab_size)
            if prefill:
                graphs["head", T] = L.head_columns_graph(
                    dev, d, T, HEAD_COLS, head_qt, HEAD_COLS >= DENSE_MME_T
                )
            else:
                graphs["head", T] = L.head_graph(dev, d, T, head_qt, mme)
                for R in (1, T):
                    graphs["sample", T, R] = L.sample_graph(dev, d, T, R, self.nranks)
            for i in range(c.num_hidden_layers):
                for iq in (False, True):
                    kind = self.attn_kind(i, iq)
                    key = ("attn_pre", T, kind)
                    if key not in graphs:
                        graphs[key] = L.attn_pre_graph(
                            dev, d, T, kind, self._attn_qtypes(i), nb, mme
                        )
                key = ("attn_post", T, layer_qt(f"l{i}.wo"))
                if key not in graphs:
                    graphs[key] = L.attn_post_graph(dev, d, T, key[2], mme)
                moe, qts = self._ffn_qtypes(i)
                key = ("ffn", T, moe, tuple(sorted(qts.items())))
                if key not in graphs:
                    graphs[key] = L.ffn_graph(dev, d, T, moe, qts, mme, prefill)
            K = self.index_keys
            Tq = d.index_tokens(T)
            graphs["index_select", T] = (
                L.index_mask_graph(dev, d, Tq, K, c.index_topk)
                if prefill
                else L.index_select_graph(dev, d, T, K, c.index_topk)
            )
            if prefill:
                graphs["index_unmask", T] = L.index_unmask_graph(dev, d, T, K)
            if not prefill:  # sparse decode attention
                graphs["index_list", T] = L.index_list_graph(dev, d, T, K, c.index_topk)
                for D in (576, 512):
                    graphs["sparse", T, D] = L.attn_sparse_graph(
                        dev, d, T, D, nb, self.pages
                    )
                self._merged_graphs(graphs, T, mme)
            for qb in (T,) if prefill else (1,):
                S = slots_per_launch(qb)
                graphs["index_slots", T, qb] = L.index_slots_graph(
                    dev, d, Tq, Tq if prefill else qb, S, K, nb
                )
                for D in (576, 512):
                    for keep in (0, K):
                        graphs["slots", T, qb, D, keep] = L.attn_slots_graph(
                            dev, d, T, qb, S, D, nb, keep
                        )
        if self.drafter is not None:
            graphs.update(
                dspark.graphs(
                    dev,
                    d,
                    self.dd,
                    nb,
                    head_qt,
                    self.nranks,
                    self.drafter.buckets,
                    self.buckets,
                )
            )
        return graphs

    def _merged_graphs(self, graphs: dict, T: int, mme: bool) -> None:
        """Decode launches merged per layer (attention + ffn): the attention block of every layer
        kind on dense steps (every column sees at most SLOT_KEYS keys) and on sparse steps, and
        the indexer owners' attention tail on sparse steps"""
        dev, d, c, nb = self.device, self.d, self.cfg, self.arena.blocks
        for i in range(c.num_hidden_layers):
            wo = self.qtype(f"l{i}.wo")
            for sparse in (False, True):
                kind = self.attn_kind(i, sparse)
                if sparse and kind.index_queries:
                    key = ("tail", T, kind.key_dim, wo)
                    if key not in graphs:
                        graphs[key] = L.attn_tail_graph(
                            dev, d, T, kind.key_dim, wo, nb, self.pages,
                            self.index_keys, c.index_topk, mme,
                        )  # fmt: skip
                    continue
                pages = self.pages if sparse else 0
                key = ("block", T, kind, wo, pages)
                if key not in graphs:
                    graphs[key] = L.attn_block_graph(
                        dev,
                        d,
                        T,
                        kind,
                        self._attn_qtypes(i),
                        wo,
                        nb,
                        slots_per_launch(1),
                        pages,
                        mme,
                    )

    def _attn_qtypes(self, i: int) -> dict[str, str]:
        p = f"l{i}."
        return {
            n: self.qtype(p + n)
            for n in ("wq", "wkv", "ix_wq", "ix_wk")
            if p + n in self.kind
        }

    def _ffn_qtypes(self, i: int) -> tuple[bool, dict[str, str]]:
        p = f"l{i}."
        moe = self.cfg.is_moe(i)
        names = ("xu", "xd", "shu", "shd") if moe else ("du", "dd")
        return moe, {n: self.qtype(p + n) for n in names}

    def _hold_workspace(self) -> None:
        """Load the prefill launches' routed experts."""
        graphs: dict[tuple, Graph] = {}
        for T in self.prefill_buckets:
            for i in range(self.cfg.num_hidden_layers):
                moe, qts = self._ffn_qtypes(i)
                key = (T, tuple(sorted(qts.items())))
                if moe and key not in graphs:
                    graphs[key] = L.ffn_graph(self.device, self.d, T, moe, qts, True, True)
        self.recipes.compile_many(list(graphs.values()))
        for g in graphs.values():
            g.destroy()

    def _compile(self) -> None:
        t0 = time.time()
        graphs = self._graphs()
        recipes = self.recipes.compile_many(list(graphs.values()))
        self.r: dict[tuple, Recipe] = dict(zip(graphs.keys(), recipes, strict=True))
        for g in graphs.values():
            g.destroy()
        cached = sum(r.from_cache for r in recipes)
        self.log(
            f"ml4: {len(recipes)} recipes ({cached} cached) in {time.time() - t0:.1f} s, "
            f"{self.device.memory_info()[0] / 2**30:.2f} GiB free"
        )
        self.compiles_at_start = self.device.compile_count

    # ------------------------------------------------------------------ programs
    def capture(self, layers: Sequence[int]) -> None:
        """Copy the residual output of ``layers`` (every column) into ``self.hidden`` each step,
        (len(layers), T, H) f32 in the order given, read with ``captured``. For the speculative
        drafter's conditioning (DSpark reads the target's hidden states at a few layers)."""
        if self.hidden:
            self.device.free(self.hidden)
        self._capture_slots = {layer: k for k, layer in enumerate(layers)}
        self.hidden = self.device.malloc(
            4 * len(layers) * self.prefill_columns * self.d.hidden
        )

    def captured(self, T: int, n: int) -> np.ndarray:
        """the last step's captured residuals of its first n columns, (layers, n, H) f32"""
        self.device.sync()
        L = len(self._capture_slots)
        out = self.device.d2h(self.hidden, (L, T, self.d.hidden), np.float32)
        return out[:, :n]

    def _p(self, partial: int, T: int) -> int:
        """what a graph of T columns reads as the ranks' summed partial ``partial`` (pa / pb):
        the all-reduced buffer itself, or its all-gathered copy (``Dims.gathered``)"""
        if not self.d.gathered(T):
            return partial
        return self.pa_all if partial == self.pa else self.pb_all

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

    def _program(
        self, T: int, qb: int, index: bool, sparse: bool = False, merged: bool = False
    ) -> _Program:
        """Every launch of a step of shape (T, query block, indexer on, sparse attention,
        merged launches) with its bindings prepared once: tables live at fixed offsets, the
        residual's ping-pong is the same in every layer (attn_pre x0 -> x1, ffn x1 -> x0) and
        partials alternate pa / pb. Sparse steps (decode with the indexer on) build key lists
        in each owner's index launches and run one attention launch per layer. Merged decode
        steps run a layer's attention as one launch (``attn_block``), or, for indexer owners
        on sparse steps, as attn_pre, the index slots and ``attn_tail``.
        A layer is (pre, index slot ops, index end op, slot ops, post, ffn); ops not run are
        empty or None."""
        key = (T, qb, index, sparse, merged)
        if (prog := self._programs.get(key)) is not None:
            return prog
        c, tb, base = self.cfg, self.tables[T], self.table
        x0, x1 = self.x
        arena = self.arena.address
        head = {
            "tok": base + tb.TOK,
            "cs": base + tb.CS,
            "lp": base + tb.LP,
            "meta": base + tb.META,
            "vis": base + tb.VIS,
        }
        common = {**self.consts, **self.io, **head, "arena": arena}
        own = {}
        if T > MAX_T:  # this rank's columns of a prefill chunk's indexer
            Tq = self.d.index_tokens(T)
            common["ix_cols"] = self.consts[f"ix_cols{T}"]
            own = {
                "ix_bits": self.io["ix_bits_own" if Tq < T else "ix_bits"],
                "vis": base + tb.INDEX_VIS,
            }
        prog = _Program()
        if T > MAX_T:
            prog.index_unmask = self._op(self.r["index_unmask", T], common)
        r = self.r["embed", T]
        emb = {**head, "emb_q": self.w["emb_q"], "emb_s": self.w["emb_s"], "x_out": x0}
        prog.embed = self._op(r, emb)
        # a decode continuation: the previous step's sampled tokens, row for row
        prog.embed_device = self._op(r, {**emb, "tok": self.sampled["tok_out"]})
        ix_sel = self.r["index_list" if sparse else "index_select", T]
        ix_slots = self.r["index_slots", T, qb]
        for i in range(c.num_hidden_layers):
            kind = self.attn_kind(i, index)
            wo = self.qtype(f"l{i}.wo")
            res = {  # the residual in, and the layer's tables
                "x": x0,
                "p": self.pzero if i == 0 else self._p(self.pb, T),
                "x_out": x1,
                "kv_rows": base + tb.store(i),
            }
            dense0 = base + tb.slot(0, i)  # the step's one dense slot launch
            dense = {
                "slots": dense0,
                "lat_pages": dense0 + tb.SLOTS_BYTES,
                "rope_pages": dense0 + tb.SLOTS_BYTES + tb.PAGES_BYTES,
            }
            sp = {"kv_blocks": base + tb.BLOCKS, "loff": self.loff[i]}
            ix = []
            if index and c.index_owners[i]:
                for j in range(tb.launches):
                    a = base + tb.index(j, i)
                    ix.append(
                        self._op(
                            ix_slots,
                            {**common, **own, "slots": a, "ix_pages": a + tb.SLOTS_BYTES},
                        )
                    )
            moe, qts = self._ffn_qtypes(i)
            r = self.r["ffn", T, moe, tuple(sorted(qts.items()))]
            ffn = self._op(
                r,
                {
                    **common,
                    **self._weights_of(i, r),
                    "x": x1,
                    "p": self._p(self.pa, T),
                    "x_out": x0,
                    "p_out": self.pb,
                },
            )
            if merged:
                pages = self.pages if sparse else 0
                if sparse and kind.index_queries:
                    r = self.r["attn_pre", T, kind]
                    pre = self._op(r, {**common, **self._weights_of(i, r), **res})
                    r = self.r["tail", T, kind.key_dim, wo]
                    post = self._op(
                        r, {**common, **self._weights_of(i, r), **sp, "p_out": self.pa}
                    )
                else:
                    r = self.r["block", T, kind, wo, pages]
                    bind = {**common, **self._weights_of(i, r), **res, **dense, **sp}
                    pre, post = self._op(r, {**bind, "p_out": self.pa}), None
                prog.layers.append((pre, ix, None, [], post, ffn))
                continue
            r = self.r["attn_pre", T, kind]
            pre = self._op(r, {**common, **self._weights_of(i, r), **res})
            ix_end = self._op(ix_sel, {**common, **own}) if ix else None
            slots = []
            if sparse:
                r = self.r["sparse", T, kind.key_dim]
                bind = {**common, "kv_blocks": base + tb.BLOCKS, "loff": self.loff[i]}
                slots.append(self._op(r, bind))
            r = self.r["slots", T, qb, kind.key_dim, self.index_keys if index else 0]
            for j in range(0 if sparse else tb.launches):
                a = base + tb.slot(j, i)
                slots.append(
                    self._op(
                        r,
                        {
                            **common,
                            "slots": a,
                            "lat_pages": a + tb.SLOTS_BYTES,
                            "rope_pages": a + tb.SLOTS_BYTES + tb.PAGES_BYTES,
                        },
                    )
                )
            r = self.r["attn_post", T, wo]
            post = self._op(r, {**common, **self._weights_of(i, r), "p_out": self.pa})
            prog.layers.append((pre, ix, ix_end, slots, post, ffn))
        r = self.r["head", T]
        bind = {
            **self._weights_of(-1, r),
            **self.consts,
            "x": x0,
            "p": self._p(self.pb, T),
            "x_out": x1,
            "logits": self.logits,
            "cols": base + tb.HEAD,
        }
        prog.head = self._op(r, bind)
        if T > MAX_T:  # prompt logprobs: the same head at uploaded columns
            prog.head_scored = self._op(r, {**bind, "cols": self.head_cols})
        self._programs[key] = prog
        return prog

    # ------------------------------------------------------------------ the step
    def forward(
        self, tokens: np.ndarray, positions: np.ndarray, rows: Sequence[Row]
    ) -> np.ndarray:
        """T columns (``tokens`` and ``positions``, padding columns any token at position 0)
        whose sequences are ``rows`` -> logits (T, V) f32 (raw: before the softcap)"""
        T = self._launch(tokens, positions, rows)
        if T > MAX_T:
            return self.head_logits(np.arange(len(tokens)))
        return self.logits_host(T)[: len(tokens)]

    def logits_host(self, T: int) -> np.ndarray:
        """the last step's full logits (T, V) f32 (raw); a prefill chunk's (HEAD_COLS, V):
        its last column, repeated"""
        self.device.sync()
        V, T = self.d.vocab, min(T, HEAD_COLS)
        out = self.device.d2h(self.logits_all, (self.nranks, T, V), np.float32)
        return np.ascontiguousarray(out.transpose(1, 0, 2).reshape(T, self.nranks * V))

    def head_logits(self, columns: np.ndarray) -> np.ndarray:
        """The last launched prefill chunk's raw logits at ``columns`` (n, V) f32, HEAD_COLS
        columns per launch (prompt logprobs). Overwrites the chunk's sampled logits: sample
        before calling this."""
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

    def sample(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        rows: Sequence[Row],
        params: np.ndarray,
        last: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A step, then the device sampler over the softcapped logits of every column (decode)
        or of column ``last`` (a prefill chunk's last real token), with ``params``
        (rows, PARAMS) (``sampling.row_params``) -> (tokens, logprobs, top ids, top logprobs)
        of the sampled rows"""
        T = self._launch(tokens, positions, rows)
        return self.sample_launched(T, params, last)

    def sample_launched(self, T: int, params: np.ndarray, last: int | None = None):
        """``sample`` for the step just launched (whose bucket is T)"""
        if T > MAX_T:  # the head ran at the chunk's last column only
            assert last == self._prefill_last, (last, self._prefill_last)
            T, last = HEAD_COLS, 0
        self._sample_launch(T, params, last)
        self.device.sync()
        out = self.device.d2h(self.sampled_buf, (_SAMPLED_BYTES,), np.uint8)
        return _sampled_fields(out, 1 if last is not None else T, len(params))

    def sample_async(self, T: int, params: np.ndarray) -> Staged:
        """Sample every column of the step just launched and copy the results into a pinned
        staging buffer without waiting; ``Staged.result`` waits for them."""
        self._sample_launch(T, params, None)
        k = self._stage_turn
        self._stage_turn ^= 1
        staged = self._staging[k]
        staged.drain()  # (its previous step's results, if never read)
        self.device.d2h_async(staged.buffer, self.sampled_buf)
        self.device.record(_STAGE_EVENT + k)
        staged.arm(T, len(params))
        return staged

    def _sample_launch(self, T: int, params: np.ndarray, last: int | None) -> None:
        R = 1 if last is not None else T
        n = len(params)
        table = np.zeros(_SAMPLE_TABLE // 4, np.float32)
        table[: n * PARAMS] = np.asarray(params, np.float32).ravel()
        table.view(np.int32)[_SAMPLE_LAST // 4] = 0 if last is None else last
        dev = self.device
        dev.h2d_nb(self.sample_table, table)
        op = self._sample_ops.get((T, R))
        if op is None:
            op = self._sample_ops[T, R] = self._op(
                self.r["sample", T, R],
                {
                    **self.consts,
                    **self.sampled,
                    "logits_all": self.logits_all,
                    "samp_params": self.sample_table,
                    "last": self.sample_table + _SAMPLE_LAST,
                },
            )
        op.run(dev.lib)

    def _layer_tables(self) -> None:
        """per-layer page offsets, for vectorized table filling"""
        g, L = self.geo, self.cfg.num_hidden_layers
        self._lat_off = np.array([g.latent_offset(i) for i in range(L)], np.int64)
        self._rope_off = np.array([g.rope_offset(i) for i in range(L)], np.int64)
        self._idx_off = np.array([g.index_offset(i) for i in range(L)], np.int64)
        self._owns = np.array([self.cfg.owns_kv(i) for i in range(L)], bool)
        # sparse attention's per-layer page offsets [latent, rotary (-1: NoPE)]
        self.loff = []
        for i in range(L):
            a = np.zeros(64, np.int32)
            a[:2] = self._lat_off[i], self._rope_off[i]
            self.loff.append(self.device.malloc(256))
            self.device.h2d(self.loff[-1], a)

    def _launch(
        self, tokens: np.ndarray | None, positions: np.ndarray, rows: Sequence[Row]
    ) -> int:
        """Enqueue a step (through the head and the logits all-gather); returns its bucket.
        ``tokens`` None: a decode continuation, whose tokens are the previous step's samples
        on the device, row for row."""
        dev, d, c = self.device, self.d, self.cfg
        n = len(positions)
        T = self.bucket(n)
        tb = self.tables[T]
        pos = np.zeros(T, np.int64)
        pos[:n] = positions
        col_blocks = np.full(T, -1, np.int64)  # each column's KV block row
        vis = np.zeros(T, np.int32)  # keys each column sees
        for r in rows:
            p = r.pos0 + np.arange(r.real)
            col_blocks[r.column : r.column + r.real] = r.blocks[p // PAGE_TOKENS]
            vis[r.column : r.column + r.real] = p + 1
        index = bool(vis.max() > c.index_topk)
        # a prefill launch attends as one query block; every launch of at most MAX_T columns
        # (decode, a short chunk, a speculative verify step) attends per column
        qb = T if T > MAX_T else 1
        S = slots_per_launch(qb)
        # the indexer on and one query column per slot: attention over each column's listed keys
        sparse = index and qb == 1 and self.sparse
        slot_rows = [
            SlotRow(r.column, r.pos0, r.real, r.blocks[None, :], r.blocks[None, :])
            for r in rows
        ]
        launches = [] if sparse else plan_slots(slot_rows, qb, d.heads, S)
        ix_launches = []
        Tq = d.index_tokens(T)
        ix_rows = self._index_rows(slot_rows, Tq) if Tq < T else slot_rows
        if index and ix_rows:  # (none: this rank's columns are all padding)
            ix_launches = plan_slots(ix_rows, Tq if Tq < T else qb, d.index_heads, S)
        if len(launches) > tb.launches or len(ix_launches) > tb.launches:
            raise ValueError("a step's slot launches exceed the table capacity")
        # ---- host tables
        k = self._host_turn
        self._host_turn ^= 1
        if self._host_pending[k]:
            dev.wait(_HOST_EVENT + k)
        h = self.host_tables[k]
        h[: tb.HEADER].view(np.int32)[: tb.HEADER // 4] = 0
        if tokens is not None:
            h[tb.TOK : tb.TOK + 4 * n].view(np.int32)[:] = tokens
        h[tb.CS : tb.CS + 512 * T].view(np.float32)[:] = self.rope[pos].ravel()
        h[tb.LP : tb.LP + 4 * T].view(np.float32)[:] = np.log(
            pos.astype(np.float64) + 1
        )
        # real columns
        real = n if T > MAX_T else T
        h[tb.META : tb.META + 16].view(np.int32)[:] = (int(pos[0]), real, 0, 0)
        if T > MAX_T:  # the head's column: the chunk's last real token
            self._prefill_last = n - 1
            h[tb.HEAD : tb.HEAD + 4 * HEAD_COLS].view(np.int32)[:] = n - 1
        h[tb.VIS : tb.VIS + 4 * T].view(np.int32)[:] = vis
        if T > MAX_T:
            first = self.rank * Tq
            h[tb.INDEX_VIS : tb.INDEX_VIS + 4 * Tq].view(np.int32)[:] = vis[
                first : first + Tq
            ]
        L = c.num_hidden_layers
        store = (
            h[tb.STORE : tb.STORE + L * tb.STORE_BYTES]
            .view(np.int32)
            .reshape(L, tb.STORE_BYTES // (4 * STORE_FIELDS), STORE_FIELDS)
        )
        store[:] = -1
        b = col_blocks[None, :]
        r = (pos % PAGE_TOKENS)[None, :]
        live = (col_blocks >= 0)[None, :]
        kv = live & self._owns[:, None]
        lat = b + self._lat_off[:, None]  # (see CacheGeometry.store_rows)
        store[:, :T, 0] = np.where(kv, lat + r, -1)
        store[:, :T, 3] = np.where(kv, lat + PAGE_TOKENS, -1)
        store[:, :T, 4] = np.where(kv, r, -1)
        store[:, :T, 1] = np.where(
            kv & (self._rope_off >= 0)[:, None],
            4 * (b + self._rope_off[:, None]) + r,
            -1,
        )
        store[:, :T, 2] = np.where(
            live & (self._idx_off >= 0)[:, None],
            2 * (b + self._idx_off[:, None]) + r,
            -1,
        )
        for j, launch in enumerate(launches):
            rec = (
                h[tb.slot(j, 0) : tb.slot(j, 0) + L * tb.SLOT_RECORD]
                .view(np.int32)
                .reshape(L, -1)
            )
            pages = launch.k_pages[None]
            rec[:, : 8 * S] = launch.slots.ravel()
            lat = np.where(pages >= 0, pages + self._lat_off[:, None, None], -1)
            rope = np.where(
                (pages >= 0) & (self._rope_off >= 0)[:, None, None],
                pages + self._rope_off[:, None, None],
                lat,
            )
            o = tb.SLOTS_BYTES // 4
            rec[:, o : o + pages[0].size] = lat.reshape(L, -1)
            o += tb.PAGES_BYTES // 4
            rec[:, o : o + pages[0].size] = rope.reshape(L, -1)
        for j, launch in enumerate(ix_launches):
            rec = (
                h[tb.index(j, 0) : tb.index(j, 0) + L * tb.INDEX_RECORD]
                .view(np.int32)
                .reshape(L, -1)
            )
            pages = launch.k_pages[None]
            rec[:, : 8 * S] = launch.slots.ravel()
            ixp = np.where(
                (pages >= 0) & (self._idx_off >= 0)[:, None, None],
                pages + self._idx_off[:, None, None],
                -1,
            )
            o = tb.SLOTS_BYTES // 4
            rec[:, o : o + pages[0].size] = ixp.reshape(L, -1)
        if sparse:
            bt = h[tb.BLOCKS : tb.BLOCKS + 4 * self.pages * T].view(np.int32)
            bt = bt.reshape(T, self.pages)
            bt[:] = -1
            for r in rows:
                bt[r.column : r.column + r.real, : len(r.blocks)] = r.blocks
        base = self.table
        dev.h2d_async(base, h[: tb.STORE + L * tb.STORE_BYTES])
        if sparse:
            dev.h2d_async(
                base + tb.BLOCKS, h[tb.BLOCKS : tb.BLOCKS + 4 * self.pages * T]
            )
        used = tb.slot(len(launches), 0)
        dev.h2d_async(base + tb.slot(0, 0), h[tb.slot(0, 0) : used])
        if ix_launches:
            dev.h2d_async(
                base + tb.index(0, 0), h[tb.index(0, 0) : tb.index(len(ix_launches), 0)]
            )
        dev.record(_HOST_EVENT + k)
        self._host_pending[k] = True
        # ---- launches
        # one launch for a layer's attention (dense: the step needs one slot launch)
        merged = (
            self.merged and qb == 1 and (sparse or (not index and len(launches) == 1))
        )
        lib, prog = dev.lib, self._program(T, qb, index, sparse, merged)
        # the residual partials: all-reduced in place, or all-gathered
        gather = self.d.gathered(T)
        ar = None
        if self.nranks > 1:
            ar = lib.rg_ccl_allgather if gather else lib.rg_ccl_allreduce
        dst_a, dst_b = (self.pa_all, self.pb_all) if gather else (self.pa, self.pb)
        count = d.hidden * T
        (prog.embed if tokens is not None else prog.embed_device).run(lib)
        nsl, nix = 1 if sparse else len(launches), len(ix_launches)
        cap, nb = self._capture_slots, 4 * d.hidden * T
        for i, (pre, ix, ix_end, slots, post, ffn) in enumerate(prog.layers):
            pre.run(lib)
            if (
                i - 1 in cap
            ):  # layer i - 1's output: x1 once layer i's first launch summed it
                dev.d2d(self.hidden + cap[i - 1] * nb, self.x[1], nb)
            for op in ix[:nix]:
                op.run(lib)
            if ix_end is not None:
                ix_end.run(lib)
                if Tq < T:  # rank-major: every column's bits, in column order
                    dev.allgather(
                        self.io["ix_bits_own"],
                        self.io["ix_bits"],
                        self.index_keys // 32 * Tq,
                    )
                if prog.index_unmask is not None:
                    prog.index_unmask.run(lib)
            for op in slots[:nsl]:
                op.run(lib)
            if post is not None:
                post.run(lib)
            if ar is not None and ar(self.pa, dst_a, count, 0):
                dev.check(-1, "residual collective")
            ffn.run(lib)
            if ar is not None and ar(self.pb, dst_b, count, 0):
                dev.check(-1, "residual collective")
        prog.head.run(lib)
        last = c.num_hidden_layers - 1
        if last in cap:
            dev.d2d(self.hidden + cap[last] * nb, self.x[1], nb)
        if self.nranks > 1:
            dev.allgather(self.logits, self.logits_all, d.vocab * min(T, HEAD_COLS))
        self._last_program = prog
        return T

    def _index_rows(self, rows: Sequence[SlotRow], Tq: int) -> list[SlotRow]:
        """``rows`` cut to this rank's Tq columns of a prefill chunk, renumbered from column 0"""
        first, out = self.rank * Tq, []
        for r in rows:
            a, b = max(first, r.column), min(first + Tq, r.column + r.real)
            if a < b:
                out.append(
                    replace(r, column=a - first, pos0=r.pos0 + a - r.column, real=b - a)
                )
        return out

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start


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
        """(tokens, logprobs, top ids, top logprobs), waiting for the copy once"""
        if self._result is None:
            assert self._shape is not None, "nothing was staged"
            self.device.wait(self.event)
            self._result = _sampled_fields(self.buffer, *self._shape)
        return self._result

    def drain(self) -> None:
        if self._shape is not None and self._result is None:
            self.result()


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

    def run(self, lib) -> None:
        if lib.rg_bind_slots(*self.args) < 0 or lib.rg_launch(self.args[0]) < 0:
            self.recipe.device.check(-1, f"launch {self.recipe.name}")


class _Program:
    def __init__(self) -> None:
        self.embed: _Op | None = None
        self.embed_device: _Op | None = None
        self.head: _Op | None = None
        self.head_scored: _Op | None = None  # prefill: the head at head_cols
        self.index_unmask: _Op | None = None  # prefill: the gathered DSA bits as a mask
        self.layers: list[tuple] = []


class _Tables:
    """Byte offsets of a step's tables (for buckets of T columns) in the device table buffer and
    its pinned host copies: a header, the per-layer store rows, then slot records ordered by
    launch index first, so a step uploads only the launches it uses."""

    SLOTS_BYTES = 512  # (S <= 16, 8) i32
    PAGES_BYTES = 1024  # (S <= 16, C / 128 = 16) i32
    SLOT_RECORD = SLOTS_BYTES + 2 * PAGES_BYTES
    INDEX_RECORD = SLOTS_BYTES + PAGES_BYTES

    def __init__(self, T: int, layers: int, launches: int, pages: int) -> None:
        r = lambda n: -(-n // 128) * 128  # noqa: E731
        self.T = T
        self.TOK = 0
        self.CS = r(4 * T)
        self.LP = self.CS + 512 * T
        self.META = self.LP + r(4 * T)
        self.HEAD = self.META + 128  # (HEAD_COLS,) i32: a prefill chunk's head columns
        self.VIS = self.HEAD + 128
        # (T,) i32 at most: this rank's columns of VIS, a prefill chunk's indexer
        self.INDEX_VIS = self.VIS + r(4 * T)
        self.HEADER = self.INDEX_VIS + r(4 * T)
        self.STORE = self.HEADER
        self.STORE_BYTES = r(4 * STORE_FIELDS * T)  # (T, STORE_FIELDS) i32
        self.layers, self.launches = layers, launches
        # sparse decode: each column's block rows (pages, T) i32, uploaded only on sparse steps
        self.BLOCKS = r(self.STORE + layers * self.STORE_BYTES)
        self.pages = pages
        self.slot_base = -(-(self.BLOCKS + 4 * pages * T) // 4096) * 4096
        self.index_base = self.slot_base + launches * layers * self.SLOT_RECORD
        self.nbytes = self.index_base + launches * layers * self.INDEX_RECORD

    def store(self, layer: int) -> int:
        return self.STORE + layer * self.STORE_BYTES

    def slot(self, launch: int, layer: int) -> int:
        return self.slot_base + (launch * self.layers + layer) * self.SLOT_RECORD

    def index(self, launch: int, layer: int) -> int:
        return self.index_base + (launch * self.layers + layer) * self.INDEX_RECORD


_SAMPLED = {
    "tok_out": 0,
    "lp_out": 256,
    "top_idx": 512,
    "top_lp": 512 + 4 * TOP_MAX * MAX_T,
}
_SAMPLED_BYTES = 512 + 8 * TOP_MAX * MAX_T
_SAMPLE_TABLE = 4 * PARAMS * MAX_T + 256
_SAMPLE_LAST = 4 * PARAMS * MAX_T
_HOST_EVENT = 60  # device event slots 60, 61: the pinned table copies
_STAGE_EVENT = 56  # 56, 57: the sampler output staging copies


__all__ = ["Ml4Gaudi", "Row", "BUCKETS", "MAX_T", "SLOT_KEYS"]
