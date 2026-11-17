"""Qwen3.8-Flash-Next on Gaudi cards as a Resin managed runner: one process per card, the experts, the n-gram
table, the delta-net and attention heads and the LM head sharded over the ranks (``checkpoint.Shard``); the
ranks' partial module outputs are all-gathered (decode) or all-reduced (prefill) between launches and every
rank samples the same tokens from the all-gathered logits.

The Engine owns the paged cache: every plan names each row's KV blocks (128 tokens) and STATE blocks (4096 tokens)
by (slab, slot) in one arena. A row reads its recurrent, conv and PLE states from the STATE block of its position
``context - 1`` and writes them to the block of its last position, so a step that crosses a 4096-token boundary
leaves the earlier block as the prefix snapshot. Prefill rows run one at a time; decode rows run as one batch.

The PLE's n-gram rows need the two tokens before each column: the runner remembers the last two tokens it fed each
sequence (a sequence restored from the prefix cache starts from EOS, the model's own padding history).

Sampling and logprobs run on the device (``common.sampling``) with Resin's semantics; up to 64 top logprobs per
sampled token. Prompt logprobs are not scored yet. Contexts stay within the QSA indexer's budget (2048 tokens of
dense attention) until the sparse path is wired.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from resin import BatchPhase, ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.backend.topology import ParallelismCapabilities
from resin_models_gaudi import ranks
from resin_models_gaudi.common.sampling import TOP_MAX, plan_params
from resin_models_gaudi.common.scoring import score_columns
from resin_models_gaudi.loading.progress import info
from resin_models_gaudi.qwen4_exp.cache import KV_GROUP, STATE_GROUP, STATE_TOKENS
from resin_models_gaudi.qwen4_exp.model import (
    DECODE_BUCKETS,
    PREFILL_BUCKETS,
    WORKSPACE_RESERVE,
    Qwen4ExpGaudi,
    RowCache,
)

PREFILL_CHUNK_SIZE = PREFILL_BUCKETS[-1]


class Qwen4ExpGaudiRunner:
    """``ManagedRunner`` over ``Qwen4ExpGaudi``; tensor parallel over ``groups``' attention replica."""

    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        groups=None,
        card: int | None = None,
        max_batch_size: int = 16,
        max_seq_len: int = 2048,
        memory_utilization: float = 0.9,
        parent_capacity: int | None = None,
        layers: int | None = None,
        table: bool = True,
        replicate_dense: bool = False,
        defer_cache: bool = False,
    ) -> None:
        """``defer_cache``: load the weights and stop; :meth:`attach` then makes the cache arena and what
        depends on its capacity. Every rank attaches at the same time (the capacity is agreed over the ranks)."""
        if max_batch_size > DECODE_BUCKETS[-1]:
            raise ValueError(f"max_batch_size exceeds {DECODE_BUCKETS[-1]}")
        self.groups = groups
        self.device, rank, nranks = ranks.open_rank(groups, card, "Qwen3.8-Flash-Next")
        self.rank, self.nranks = rank, nranks
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        def reduce_free(free: int) -> int:
            """the smallest free memory over the ranks: the arena every rank can hold"""
            return ranks.min_over_ranks(free, groups)

        self.model = Qwen4ExpGaudi(
            self.device,
            checkpoint,
            layers=layers,
            table=table,
            max_positions=max_seq_len,
            parent_capacity=parent_capacity,
            memory_utilization=memory_utilization,
            rank=rank,
            nranks=nranks,
            replicate_dense=replicate_dense,
            reduce_free=reduce_free,
            log=info if rank == 0 else (lambda _: None),
            decode_buckets=[b for b in DECODE_BUCKETS if b < max_batch_size]
            + [min(b for b in DECODE_BUCKETS if b >= max_batch_size)],
            defer_cache=True,
        )
        self._parent_capacity = parent_capacity
        budget = self.model.cfg.indexer_budget + 3
        if max_seq_len > budget:
            raise NotImplementedError(
                f"contexts past {budget} tokens need the sparse QSA path"
            )
        if groups is not None and nranks > 1:
            ranks.barrier(groups.replica.handle)
        self.vocab_size = self.model.vocab_all
        self.geometry = self.model.geo
        self._history: dict[
            int, tuple[int, int]
        ] = {}  # sequence id -> the last two tokens fed
        if not defer_cache:
            self.attach()

    @property
    def own_slab_rows(self) -> int:
        """This model's slab, for planning an arena that other models share."""
        return self.geometry.own_slab_rows

    @property
    def slab_unit(self) -> int:
        """A shared slab is a multiple of this many rows."""
        return self.geometry.slab_unit

    @property
    def workspace_reserve(self) -> int:
        return WORKSPACE_RESERVE

    def shared_capacity(self, free: int, slab_rows: int) -> int:
        """The slabs of ``slab_rows`` rows that ``free`` bytes hold (for an arena that other models share), less
        ``parent_capacity`` when that is smaller."""
        slab_bytes = self.geometry.with_slab_rows(slab_rows).slab_bytes
        return ranks.requested_capacity(max(0, free // slab_bytes - 1), self._parent_capacity)

    def attach(
        self,
        capacity: int | None = None,
        *,
        slab_rows: int | None = None,
        arena: int | None = None,
    ) -> None:
        """Make the cache arena: ``capacity`` slabs, or what the model sizes from its free memory (the smallest
        over the ranks; ``parent_capacity`` when given); then the recipes, the cache layout and the runner
        config. ``slab_rows``: slabs of that many rows, which another model's slabs share. ``arena``: the address
        of a shared arena of ``capacity`` slabs (see ``Qwen4ExpGaudi.attach``)."""
        groups = self.groups
        self.model.attach(
            self._parent_capacity if capacity is None else capacity,
            slab_rows=slab_rows,
            arena=arena,
        )
        self.geometry = self.model.geo
        if groups is not None and self.nranks > 1:
            ranks.barrier(groups.replica.handle)
        self.cache_layout = self.geometry.layout(self.model.parent_capacity)
        self._binding = CachePoolBinding(
            self.cache_layout, resource="Qwen3.8-Flash-Next Gaudi cache"
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=0,
            draft_tokens=0,
            cache_layout=self.cache_layout,
        )
        self.compiles_at_start = self._compiles_seen = self.device.compile_count

    @property
    def kv_pool_bytes(self) -> int:
        return self.model.arena_bytes

    def configure_metrics(self, *, enabled: bool = True) -> None:
        """Nothing to switch on."""

    def metrics_snapshot(self) -> RunnerMetrics:
        return RunnerMetrics()

    def bind_resources(self, resources) -> None:
        self._binding.bind_resources(resources)

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start

    def _check_compiles(self) -> None:
        count = self.device.compile_count
        if count != self._compiles_seen:
            self._compiles_seen = count
            warnings.warn(
                f"{count - self.compiles_at_start} Synapse graph compilations after startup",
                RuntimeWarning,
                stacklevel=3,
            )

    def _rows(self, cache: PagedCachePlan) -> list[RowCache]:
        g = self.geometry
        kv, state = cache.groups[KV_GROUP], cache.groups[STATE_GROUP]
        batch = cache.batch
        rows = []
        for r in range(batch.num_sequences):
            context = int(batch.context_lengths[r])
            last = context + int(batch.query_lengths[r]) - 1
            first_state = int(state.block_table_start_indices[r])
            states = tuple(
                g.state_block_row(slab, slot) for slab, slot in state.row_addresses(r)
            )
            rows.append(
                RowCache(
                    context,
                    g.kv_block_row(kv.block_slabs(r), kv.block_slots(r)),
                    -1
                    if context == 0
                    else states[(context - 1) // STATE_TOKENS - first_state],
                    states[last // STATE_TOKENS - first_state],
                    first_state,
                    states,
                    self._history.get(int(batch.sequence_ids[r]), (-1, -1)),
                )
            )
        return rows

    def _remember(self, sequence: int, fed: Sequence[int]) -> None:
        last = [int(t) for t in fed[-2:]]
        before = self._history.get(sequence, (-1, -1))
        self._history[sequence] = (
            (before[1], last[-1]) if len(last) == 1 else (last[0], last[1])
        )

    def forward(self, plan) -> ForwardOutput:
        self._check_compiles()
        if any(media is not None for media in plan.media):
            raise NotImplementedError(
                "this runner has no vision tower; text prompts only"
            )
        if plan.needs_prompt_logprobs:
            raise NotImplementedError(
                "prompt logprobs are not scored by this runner yet"
            )
        cache = PagedCachePlan.from_plan(plan, self.cache_layout)
        batch = cache.batch
        if batch.num_sequences > self.max_batch_size:
            raise ValueError(f"{batch.num_sequences} rows exceed {self.max_batch_size}")
        if int(np.max(batch.post_lengths)) > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        params = plan_params(plan, self.vocab_size)
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        rows = self._rows(cache)
        scores = plan.needs_logprobs
        lp = top_ids = top_lp = None
        if batch.phase == BatchPhase.PREFILL:
            tokens, parts = [], []
            for r, row in enumerate(rows):
                ids = batch.token_ids[batch.token_slice(r)].tolist()
                tokens.append(self.model.prefill(row, ids, params[r : r + 1]))
                self._remember(int(batch.sequence_ids[r]), ids)
                if scores:
                    parts.append(self.model.scores(1))
            if scores:
                lp, top_ids, top_lp = (
                    np.concatenate(p) for p in zip(*parts, strict=True)
                )
        else:
            fed = batch.token_ids.tolist()
            tokens = self.model.decode(rows, fed, params).tolist()
            for r in range(len(rows)):
                self._remember(int(batch.sequence_ids[r]), fed[r : r + 1])
            if scores:
                lp, top_ids, top_lp = self.model.scores(len(rows))
        counts = np.ones(len(tokens), np.int64)
        columns = (
            score_columns(params, counts, lp, top_ids, top_lp)
            if scores
            else {}
        )
        return ForwardOutput(
            token_ids=np.asarray(tokens, np.uint32),
            token_indptr=np.arange(len(tokens) + 1, dtype=np.uint32),
            **columns,
        )

    def release(self, sequence_ids: list[int]) -> None:
        for sequence in sequence_ids:
            self._history.pop(int(sequence), None)

    def close(self) -> None:
        self.model.close()
        self.device.close()


__all__ = ["PREFILL_CHUNK_SIZE", "Qwen4ExpGaudiRunner"]
