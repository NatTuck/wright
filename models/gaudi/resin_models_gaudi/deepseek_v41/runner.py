"""DeepSeek-V4.1-Flash on Gaudi2 as a Resin managed runner: one tensor-parallel rank per process.

The Engine owns the paged cache: every plan names each row's WINDOW blocks (the 128-token pages a query can still
see) and HISTORY blocks (compressed entries). Every rank receives the same plan, runs the same launches and
collectives, and samples the same tokens from the all-gathered logits on its own device.

Prefill rows run one at a time in chunks of up to TMAX (2048) tokens; decode rows run as one batch of up to MAX_T, and
larger batches as consecutive MAX_T-row steps. ``enqueue`` submits a decode step of up to MAX_T rows without waiting
(the Engine's overlap): a continuation reads its tokens from the previous step's samples on the device. Engram hashes
each position with the three tokens before it; the runner keeps each sequence's last three tokens (a continuation's
own token is hashed on the device) and the last three tokens of each WINDOW block, for a sequence that the Engine's
prefix cache starts from cached blocks (``EngramContext``).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from resin import BatchPhase, ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.models.deepseek_v41 import DeepSeekV41Config
from resin_models_gaudi.deepseek_v41.cache import (
    HISTORY_GROUP,
    WINDOW_GROUP,
    CacheGeometry,
    EngramContext,
)
from resin_models_gaudi.deepseek_v41.model import (
    DSPARK_ROWS,
    MAX_T,
    TMAX,
    DeepSeekV41Gaudi,
    Row,
    Shape,
)
from resin_models_gaudi.glm5_3_flash.runner import _Scorer
from resin_models_gaudi.mistral_large4 import parallel
from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner, score_requests
from resin_models_gaudi.qwen3_5.sampling import TOP_MAX, plan_params
from resin_models_gaudi.synapse.progress import info
from resin_models_gaudi.synapse.graph import LAUNCH_STATS, reset_launch_stats
from resin_models_gaudi.synapse.runtime import open_device

PREFILL_CHUNK_SIZE = TMAX
MAX_BATCH = 64
profile_step = os.environ.get("RESIN_GAUDI_PROFILE_STEP") == "1"


class DeepSeekV41GaudiRunner:
    """``ManagedRunner`` over ``DeepSeekV41Gaudi``; tensor parallel over ``groups``' attention replica."""

    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        checkpoint: str,
        *,
        groups=None,
        card: int | None = None,
        max_batch_size: int = MAX_T,
        max_seq_len: int = 16384,
        parent_capacity: int | None = None,
        num_layers: int | None = None,
        engram: str = "card",
        engram_gather: str = "auto",
        engram_prewarm: bool = False,
        draft_tokens: int = 0,
        log=info,
    ) -> None:
        if max_batch_size > MAX_BATCH:
            raise ValueError(f"max_batch_size exceeds {MAX_BATCH}")
        self.groups = groups
        rank, nranks = 0, 1
        if groups is not None:
            topo = groups.topology
            if topo.cp > 1 or topo.ep > 1 or topo.attn_dp > 1:
                raise ValueError(
                    "DeepSeek-V4.1-Flash on Gaudi supports tensor parallelism only"
                )
            rank, nranks = groups.replica.local_rank, groups.replica.size
        self.rank, self.nranks = rank, nranks
        self.device = open_device(card)
        if groups is not None and nranks > 1:
            parallel.hccl_rendezvous(
                self.device, groups.rank, nranks, groups.replica.handle
            )
        cfg = DeepSeekV41Config.from_hugging_face(
            json.loads((Path(checkpoint) / "config.json").read_text())
        )
        self.config = cfg
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        # DSpark: decode steps of up to DSPARK_ROWS rows verify each row's anchor and draft_tokens drafts
        if draft_tokens and not 0 < draft_tokens <= cfg.dspark_block_size:
            raise ValueError(
                f"draft_tokens: 1 to {cfg.dspark_block_size} (the DSpark block)"
            )
        speculative = draft_tokens > 0
        self.speculative_tokens = draft_tokens
        self.speculate = (
            speculative  # (off: plain decoding with the same graphs, for parity)
        )
        self.lookahead = (
            draft_tokens  # a verify step writes positions p .. p + draft_tokens
        )
        self._speculation = SpeculativeStats(draft_tokens)
        geometry = CacheGeometry.from_config(
            cfg,
            layers=num_layers,
            max_write=TMAX,
            draft_layers=cfg.num_nextn_predict_layers if speculative else 0,
        )

        def capacity(free: int) -> int:
            """the arena every rank can hold: the smallest free memory over the ranks"""
            if groups is not None and nranks > 1:
                import torch
                import torch.distributed as dist

                box = torch.tensor([free], dtype=torch.int64)
                dist.all_reduce(box, op=dist.ReduceOp.MIN, group=groups.replica.handle)
                free = int(box.item())
            fit = geometry.capacity_for(free)
            if parent_capacity is not None:
                if parent_capacity > fit:
                    raise ValueError(f"parent_capacity {parent_capacity} exceeds {fit}")
                return parent_capacity
            return fit

        barrier = None
        if groups is not None and nranks > 1:
            barrier = lambda: parallel.barrier(groups.replica.handle)  # noqa: E731
        self.model = DeepSeekV41Gaudi(
            self.device, checkpoint, cfg, capacity=capacity, max_seq_len=max_seq_len, rank=rank, nranks=nranks,
            layers=num_layers, engram=engram, engram_gather=engram_gather, engram_prewarm=engram_prewarm,
            barrier=barrier, log=log if rank == 0 else (lambda *_: None),
            dspark=speculative,
        )  # fmt: skip
        if barrier is not None:
            barrier()
        self.vocab_size = cfg.vocab_size
        self.geometry = geometry
        self.cache_layout = geometry.layout(self.model.capacity)
        self._binding = CachePoolBinding(
            self.cache_layout, resource="DeepSeek-V4.1-Flash Gaudi cache"
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.lookahead,
            cache_layout=self.cache_layout,
        )
        self.compiles_at_start = self._compiles_seen = self.device.compile_count
        self._context = EngramContext(
            cfg.engram_max_ngram_size - 1, geometry.window_blocks(self.model.capacity)
        )
        self._inflight: list[
            _Completion
        ] = []  # enqueued steps whose samples are not yet recorded in _context
        self.continuations = 0

    @property
    def kv_pool_bytes(self) -> int:
        return self.model.arena_blocks * 1024 * 512

    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot(include_draft_tokens=True)

    def configure_metrics(self, *, enabled: bool = True) -> None:
        """Speculation counters always run, so there is nothing to switch on."""

    def metrics_snapshot(self) -> RunnerMetrics:
        return RunnerMetrics(
            spec_draft_tokens=self._speculation.drafts,
            spec_accepted_tokens=self._speculation.accepted,
            spec_verify_rows=self._speculation.blocks,
        )

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

    def _row(
        self,
        cache: PagedCachePlan,
        r: int,
        column: int,
        pos0: int,
        real: int,
        sequence: int,
    ) -> Row:
        g = self.geometry
        win, hist = cache.groups[WINDOW_GROUP], cache.groups[HISTORY_GROUP]
        slabs, slots = win.block_slabs(r), win.block_slots(r)
        ids, first = (
            g.window_block_id(slabs, slots),
            int(win.block_table_start_indices[r]),
        )
        return Row(
            column, pos0, real, g.window_block_row(slabs, slots), ids, first,
            g.history_block_row(hist.block_slabs(r), hist.block_slots(r)),
            self._context.before(sequence, pos0, first, ids),
        )  # fmt: skip

    def forward(self, plan) -> ForwardOutput:
        t_enter = time.perf_counter() if profile_step else 0.0
        self._harvest()
        t_harvest = time.perf_counter() if profile_step else 0.0
        self._check_compiles()
        if any(media is not None for media in plan.media):
            raise NotImplementedError(
                "this runner has no vision tower; text prompts only"
            )
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead
        )
        t_cache = time.perf_counter() if profile_step else 0.0
        batch = cache.batch
        if batch.num_sequences > self.max_batch_size:
            raise ValueError(f"{batch.num_sequences} rows exceed {self.max_batch_size}")
        if int(batch.post_lengths.max()) + self.lookahead > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        prompt = plan.needs_prompt_logprobs
        if prompt and batch.phase != BatchPhase.PREFILL:
            raise ValueError("prompt logprobs are scored by prefill plans")
        params = plan_params(plan, self.vocab_size)
        t_params = time.perf_counter() if profile_step else 0.0
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        model = self.model
        outs, prompt_scores = [], []
        if self._speculating(plan, batch.num_sequences):
            starts = np.asarray(batch.context_lengths, np.int64)
            anchors = np.asarray(batch.token_ids, np.int64)
            return self._speculate(plan, cache, starts, anchors).result()
        if batch.phase == BatchPhase.PREFILL:
            requests = (
                score_requests(plan, self.vocab_size)
                if prompt
                else [None] * batch.num_sequences
            )
            for r in range(batch.num_sequences):
                sequence = int(batch.sequence_ids[r])
                ids = np.asarray(batch.token_ids[batch.token_slice(r)], np.int64)
                pos0 = int(batch.context_lengths[r])
                scores = _Scorer(requests[r])
                for s in range(0, len(ids), PREFILL_CHUNK_SIZE):
                    chunk = ids[s : s + PREFILL_CHUNK_SIZE]
                    p = pos0 + s
                    row = self._row(cache, r, 0, p, len(chunk), sequence)
                    cols = scores.columns(s, len(chunk))
                    last = s + len(chunk) >= len(ids)
                    # the logits this chunk needs: scored columns, the sampled column, or none
                    need = "all" if len(cols) else len(chunk) - 1 if last else None
                    T = model.launch(
                        chunk,
                        p + np.arange(len(chunk)),
                        [row],
                        prefill=True,
                        logits=need,
                    )
                    self._context.seen(sequence, p, chunk)
                    if len(cols):
                        scores.add(s + cols, model.logits_host(T)[cols])
                    if last:
                        outs.append(
                            model.sample_launched(
                                Shape(T, True), params[r : r + 1], len(chunk) - 1
                            )
                        )
                prompt_scores.append(scores.result())
        else:
            n = batch.num_sequences
            positions = np.asarray(batch.context_lengths, np.int64)
            tokens = np.asarray(batch.token_ids, np.int64)
            for g in range(0, n, MAX_T):
                part = range(g, min(n, g + MAX_T))
                rows = [
                    self._row(
                        cache,
                        r,
                        r - g,
                        int(positions[r]),
                        1,
                        int(batch.sequence_ids[r]),
                    )
                    for r in part
                ]
                sl = slice(g, g + len(rows))
                if profile_step:
                    reset_launch_stats()
                    t_step = time.perf_counter()
                    t_rows = t_step
                T = model.launch(tokens[sl], positions[sl], rows)
                for r in part:
                    self._context.seen(
                        int(batch.sequence_ids[r]), int(positions[r]), tokens[r : r + 1]
                    )
                t_sample = time.perf_counter() if profile_step else 0.0
                outs.append(model.sample_launched(Shape(T, False), params[sl]))
                if profile_step:
                    end = time.perf_counter()
                    enqueue = 1000 * LAUNCH_STATS["seconds"]
                    launch_wall = 1000 * (t_sample - t_step)
                    sample_wall = 1000 * (end - t_sample)
                    step_wall = 1000 * (end - t_step)
                    self._last_step_end = end
                    info(
                        f"step profile T={T} rows={len(rows)}: launches={LAUNCH_STATS['calls']} "
                        f"enqueue={enqueue:.1f} ms, launch_wall={launch_wall:.1f} ms, "
                        f"sync+d2h={sample_wall:.1f} ms, step_wall={step_wall:.1f} ms, "
                        f"residual={step_wall - enqueue - sample_wall:.1f} ms; "
                        f"pre={1000 * (t_rows - t_enter):.1f} ms "
                        f"(harvest {1000 * (t_harvest - t_enter):.1f}, "
                        f"cache_plan {1000 * (t_cache - t_harvest):.1f}, "
                        f"params {1000 * (t_params - t_cache):.1f}, "
                        f"rows {1000 * (t_rows - t_params):.1f})"
                    )
        tok, lp, top_ids, top_lp = (np.concatenate(x) for x in zip(*outs, strict=True))
        counts = np.ones(len(tok), np.int64)
        columns = (
            Qwen3_5GaudiRunner._score_columns(params, counts, lp, top_ids, top_lp)
            if plan.needs_logprobs
            else {}
        )
        if prompt:
            columns.update(Qwen3_5GaudiRunner._prompt_columns(plan, prompt_scores))
        if profile_step and getattr(self, "_last_step_end", None) is not None:
            info(
                f"step post: {1000 * (time.perf_counter() - self._last_step_end):.1f} ms"
            )
        return ForwardOutput(
            token_ids=np.asarray(tok, np.uint32),
            token_indptr=np.arange(len(tok) + 1, dtype=np.uint32),
            **columns,
        )

    def _harvest(self, keep: int = 0) -> None:
        """Record the samples of all but the newest ``keep`` enqueued steps in ``_context``."""
        while len(self._inflight) > keep:
            self._inflight.pop(0).result()

    def enqueue(self, plan, *, feedback: _Feedback | None = None):
        """Submit a decode step and return without waiting for it (the Engine's overlap): a continuation
        (``feedback``) reads its tokens from the previous step's samples on the device, at the previous
        positions + 1. Other plans run synchronously."""
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead
        )
        batch = cache.batch
        B = batch.num_sequences
        if self._speculating(plan, B) and (
            feedback is None or isinstance(feedback, _Verification)
        ):
            self._check_compiles()
            self._harvest()
            if feedback is None:
                starts = np.asarray(batch.context_lengths, np.int64)
                anchors = np.asarray(batch.token_ids, np.int64)
            else:
                feedback.finish()
                starts, anchors = feedback.positions, feedback.anchors
                self.continuations += 1
            return self._speculate(plan, cache, starts, anchors)
        if (
            plan.phase != BatchPhase.DECODE
            or plan.needs_prompt_logprobs
            or B > MAX_T
            or any(media is not None for media in plan.media)
            or not self.model.hashed(Shape(MAX_T, False))
        ):
            # The last clause is load-bearing in host mode, and it is not a bug to "fix": with the
            # Engram tables on the host, `hashed` is False for every decode shape, so every decode
            # plan takes this synchronous path. That is what makes host-mode decode correct - a
            # fresh plan carries the previous step's sampled tokens in `batch.token_ids`, so the
            # Engram hash and `Row.before` see real tokens. Drop the clause to keep a continuation
            # and the Engine hands over a *conservative bound*, one position stale
            # (crates/resin-core/src/engine.rs: "the longest possible previous output"), which
            # model.py's `tokens is None` ValueError then rejects; drop that check too and the
            # stale token is hashed and recorded in EngramContext, poisoning every later
            # Row.before for the sequence - wrong text, no error. Host mode has no overlap to
            # recover anyway: the next step's rows need this step's sampled token on the host.
            self._harvest()
            return self.forward(plan)
        self._check_compiles()
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        # the tokens before each row's position are known once every step but the one feeding this is recorded
        self._harvest(keep=1 if feedback is not None else 0)
        if feedback is None:
            positions = np.asarray(batch.context_lengths, np.int64)
            tokens = np.asarray(batch.token_ids, np.int64)
        else:
            positions = np.asarray(feedback.positions, np.int64)
            tokens = None
            self.continuations += 1
        if int(positions.max()) + 1 > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        sequences = [int(x) for x in batch.sequence_ids]
        rows = [
            self._row(cache, r, r, int(positions[r]), 1, sequences[r]) for r in range(B)
        ]
        params = plan_params(plan, self.vocab_size)
        T = self.model.launch(tokens, positions, rows)
        if tokens is not None:
            for r in range(B):
                self._context.seen(sequences[r], int(positions[r]), tokens[r : r + 1])
        staged = self.model.sample_async(Shape(T, False), params)
        scores = plan.needs_logprobs

        def build() -> ForwardOutput:
            tok, lp, top_ids, top_lp = staged.result()
            columns = (
                Qwen3_5GaudiRunner._score_columns(
                    params, np.ones(B, np.int64), lp, top_ids, top_lp
                )
                if scores
                else {}
            )
            return ForwardOutput(
                token_ids=np.asarray(tok, np.uint32),
                token_indptr=np.arange(B + 1, dtype=np.uint32),
                **columns,
            )

        completion = _Completion(
            build, _Feedback(positions + 1), self, sequences, positions
        )
        self._inflight.append(completion)
        return completion

    def _speculating(self, plan, rows: int) -> bool:
        """whether a plan runs as a speculative step: a decode step of up to DSPARK_ROWS rows, no logprob
        scoring of its prompt"""
        return (
            self.speculate
            and plan.phase == BatchPhase.DECODE
            and not plan.needs_prompt_logprobs
            and rows <= DSPARK_ROWS
        )

    def _speculate(
        self, plan, cache: PagedCachePlan, starts: np.ndarray, anchors: np.ndarray
    ):
        """One speculative decode step: DSpark drafts from each row's anchor (at position ``starts``, not yet
        processed), then one target step over [anchor, drafts] at starts .. starts + K, every column sampled;
        the drafter's window keys of those positions are stored by the target step (rejected ones are rewritten
        before any draft reads them)."""
        batch, model = cache.batch, self.model
        B, K = batch.num_sequences, self.speculative_tokens
        sequences = [int(x) for x in batch.sequence_ids]
        rows = [
            self._row(cache, r, 0, int(starts[r]), 1, sequences[r]) for r in range(B)
        ]
        drafts = model.draft(anchors, rows)[:, :K]
        tokens = np.concatenate([anchors[:, None], drafts], 1).ravel()
        positions = (starts[:, None] + np.arange(K + 1)).ravel()
        rows = [
            self._row(cache, r, r * (K + 1), int(starts[r]), K + 1, sequences[r])
            for r in range(B)
        ]
        T = model.launch(tokens, positions, rows)
        params = plan_params(plan, self.vocab_size)
        staged = model.sample_async(Shape(T, False), np.repeat(params, K + 1, axis=0))
        return _Verification(
            self, plan, sequences, starts, anchors, drafts, params, staged
        )

    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        """``enqueue`` on every rank of the replica (tensor parallel: the same plan)"""
        if plan is None:
            return None
        return self.enqueue(plan, feedback=feedback)

    def forward_distributed(
        self, plan, context, *, prepared=None
    ) -> ForwardOutput | None:
        """Tensor parallel only: every rank of the replica runs the same plan."""
        return None if plan is None else self.forward(plan)

    def release(self, sequence_ids: list[int]) -> None:
        self._context.release(sequence_ids)

    def close(self) -> None:
        if self.nranks > 1:
            self.device.lib.rg_ccl_destroy()
        self.device.close()


class _Feedback:
    """Exact inputs of a decode continuation (the Engine's continuation plan only bounds them): each row's next
    position; its tokens stay on the device."""

    def __init__(self, positions: np.ndarray) -> None:
        self.positions = positions


class _Verification:
    """An enqueued speculative step and its continuation's feedback: ``finish`` waits for its samples, accepts
    each row's longest draft prefix that the target's samples confirm, records the processed tokens it commits
    (anchor and accepted drafts), and leaves ``positions`` / ``anchors``: each row's next anchor."""

    def __init__(
        self, runner, plan, sequences, starts, anchors, drafts, params, staged
    ) -> None:
        self._runner, self._plan, self._sequences = runner, plan, sequences
        self._starts, self._anchors, self._drafts = starts, anchors, drafts
        self._params, self._staged = params, staged
        self._output: ForwardOutput | None = None

    @property
    def feedback(self) -> _Verification:
        return self

    def finish(self) -> None:
        if self._output is not None:
            return
        runner, plan = self._runner, self._plan
        B, K = len(self._drafts), runner.speculative_tokens
        tok, lp, top_ids, top_lp = self._staged.result()
        acceptance = runner._speculation.accept(
            self._drafts.tolist(), np.asarray(tok).reshape(B, K + 1).tolist()
        )
        counts = np.asarray(acceptance.accepted_counts, np.int64) + 1
        rows = acceptance.token_rows
        for r, sequence in enumerate(self._sequences):
            committed = np.asarray([self._anchors[r], *rows[r][:-1]], np.int64)
            runner._context.seen(sequence, int(self._starts[r]), committed)
        self.positions = self._starts + counts
        self.anchors = np.array([row[-1] for row in rows], np.int64)
        columns = {}
        if plan.needs_logprobs:
            picked = np.concatenate(
                [r * (K + 1) + np.arange(counts[r]) for r in range(B)]
            )
            columns = Qwen3_5GaudiRunner._score_columns(
                self._params,
                counts,
                lp[picked],
                top_ids[picked],
                top_lp[picked],
            )
        self._output = ForwardOutput(
            token_ids=np.asarray([t for row in rows for t in row], np.uint32),
            token_indptr=np.concatenate(([0], np.cumsum(counts))).astype(np.uint32),
            **columns,
        )

    def result(self) -> ForwardOutput:
        self.finish()
        assert self._output is not None
        return self._output


class _Completion:
    """One enqueued decode step: ``result`` waits for its staged samples once and records each row's sampled
    token (the token at its position + 1) as the sequence's newest."""

    def __init__(
        self, build, feedback: _Feedback, runner, sequences, positions
    ) -> None:
        self.feedback = feedback
        self._build, self._result = build, None
        self._runner, self._sequences, self._positions = runner, sequences, positions

    def result(self) -> ForwardOutput:
        if self._result is None:
            self._result = self._build()
            runner = self._runner
            if self in runner._inflight:
                runner._inflight.remove(self)
            sampled = np.asarray(self._result.token_ids, np.int64)
            for r, sequence in enumerate(self._sequences):
                if sequence in runner._context.last:  # not released meanwhile
                    runner._context.seen(
                        sequence, int(self._positions[r]) + 1, sampled[r : r + 1]
                    )
        return self._result
