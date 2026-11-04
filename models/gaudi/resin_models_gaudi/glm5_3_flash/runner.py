"""GLM-5.3-Flash on Gaudi2 as a Resin managed runner: one tensor-parallel rank per process.

The Engine owns the paged cache: every plan names each row's KV blocks (128 tokens) and STATE blocks (4096
tokens). Every rank receives the same plan (``DistributedWorker`` broadcasts it within the attention replica),
runs the same launches and collectives, and samples the same tokens from the all-gathered logits on its own
device, so no token broadcast is needed; the driver's output is the one the Engine reads.

Prefill rows run one at a time in launches of up to 4096 tokens; decode rows run as one batch of up to MAX_T, and
larger batches as consecutive MAX_T-row steps. Sampling and logprobs run on the device with Resin's semantics
(``qwen3_5.sampling``).

With ``mtp`` = K, a decode step verifies each row's anchor and carried drafts (K + 1 columns per row, so groups of
MAX_T // (K + 1) rows) and emits the accepted prefix plus the target's next token. Every emitted token is the target's
sample at its position, so speculative output equals plain decoding, greedy or seeded; carried drafts only affect speed
(a row whose drafts do not continue from its anchor verifies zeros).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from resin import BatchPhase, ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi.glm5_3_flash.cache import (
    KV_GROUP,
    MTP_GROUP,
    STATE_GROUP,
    STATE_TOKENS,
    CacheGeometry,
)
from resin_models_gaudi.glm5_3_flash.model import (
    MAX_T,
    Glm53Gaudi,
    Row,
    prefill_chunks,
)
from resin_models_gaudi.glm5_3_flash.weights import RankSlices
from resin_models_gaudi.mistral_large4 import parallel
from resin_models_gaudi.qwen3_5.model import Sampling, ScoreRequest, Scores
from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner, score_requests
from resin_models_gaudi.qwen3_5.sampling import TOP_MAX, plan_params, plan_seeds
from resin_models_gaudi.synapse.progress import info
from resin_models_gaudi.synapse.runtime import open_device

PREFILL_CHUNK_SIZE = 4096
MAX_BATCH = 64


class Glm53FlashGaudiRunner:
    """``ManagedRunner`` over ``Glm53Gaudi``; tensor parallel over ``groups``' attention replica."""

    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        checkpoint: str,
        *,
        groups=None,
        card: int | None = None,
        max_batch_size: int = MAX_T,
        max_seq_len: int = 65536,
        parent_capacity: int | None = None,
        num_layers: int | None = None,
        mtp: int = 0,
        log=info,
    ) -> None:
        if max_batch_size > MAX_BATCH:
            raise ValueError(f"max_batch_size exceeds {MAX_BATCH}")
        self.groups = groups
        rank, nranks = 0, 1
        if groups is not None:
            topo = groups.topology
            if topo.cp > 1 or topo.ep > 1 or topo.attn_dp > 1:
                raise ValueError("GLM-5.3-Flash on Gaudi supports tensor parallelism only")
            rank, nranks = groups.replica.local_rank, groups.replica.size
        self.rank, self.nranks = rank, nranks
        self.device = open_device(card)
        if groups is not None and nranks > 1:
            parallel.hccl_rendezvous(self.device, groups.rank, nranks, groups.replica.handle)
        raw = json.loads((Path(checkpoint) / "config.json").read_text())
        cfg = Glm53Config.from_dict(raw)
        self.config = cfg
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        if mtp > 3:
            raise ValueError(f"glm5_3_flash on Gaudi takes at most 3 draft tokens, not {mtp}")
        self.mtp = mtp
        self.speculative_tokens = mtp
        # a verify step writes positions p .. p + K; its draft chain's MTP keys reach p + 2K - 1
        self.lookahead = 2 * mtp - 1 if mtp else 0
        # sequence id -> (the next anchor's position, the anchor token, its drafts)
        self._drafts: dict[int, tuple[int, int, np.ndarray]] = {}
        self._open: _Verification | None = None  # the last enqueued MTP step
        sl = RankSlices.of(cfg, rank, nranks)
        geometry = CacheGeometry.from_config(
            cfg, sl.kda_heads.stop - sl.kda_heads.start, layers=num_layers, mtp=mtp > 0
        )

        def capacity(free: int) -> int:
            """the arena every rank can hold: the smallest free memory over the ranks"""
            if groups is not None and nranks > 1:
                import torch
                import torch.distributed as dist

                box = torch.tensor([free], dtype=torch.int64)
                dist.all_reduce(box, op=dist.ReduceOp.MIN, group=groups.replica.handle)
                free = int(box.item())
            fit = geometry.capacity_for(free, geometry.slab_bytes)
            if parent_capacity is not None:
                if parent_capacity > fit:
                    raise ValueError(f"parent_capacity {parent_capacity} exceeds {fit}")
                return parent_capacity
            return fit

        self.model = Glm53Gaudi(
            self.device,
            checkpoint,
            cfg,
            capacity=capacity,
            max_seq_len=max_seq_len,
            rank=rank,
            nranks=nranks,
            layers=num_layers,
            mtp=mtp,
            log=log if rank == 0 else (lambda *_: None),
        )
        self._speculation = self.model.speculation if mtp else SpeculativeStats(0)
        if groups is not None and nranks > 1:
            parallel.barrier(groups.replica.handle)
        self.vocab_size = cfg.vocab_size
        self.geometry = geometry
        self.cache_layout = geometry.layout(self.model.capacity)
        self._binding = CachePoolBinding(self.cache_layout, resource="GLM-5.3-Flash Gaudi cache")
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.lookahead,
            cache_layout=self.cache_layout,
        )
        self.compiles_at_start = self._compiles_seen = self.device.compile_count
        self.continuations = 0

    @property
    def kv_pool_bytes(self) -> int:
        """The KV pool on one card."""
        return self.model.arena.nbytes

    @property
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

    def _blocks(self, cache: PagedCachePlan, r: int) -> np.ndarray:
        kv = cache.groups[KV_GROUP]
        return self.geometry.kv_block_row(kv.block_slabs(r), kv.block_slots(r))

    def _state(self, cache: PagedCachePlan, r: int):
        """position -> the first row of the STATE block holding it"""
        state = cache.groups[STATE_GROUP]
        first = int(state.block_table_start_indices[r])
        rows = [
            self.geometry.state_block_row(slab, slot) for slab, slot in state.row_addresses(r)
        ]
        return lambda position: rows[position // STATE_TOKENS - first]

    def _row(self, cache: PagedCachePlan, r: int, column: int, pos0: int, real: int) -> Row:
        state_group = cache.groups[STATE_GROUP]
        state = self._state(cache, r)
        g = self.geometry
        mtp_blocks = None
        if self.mtp:
            m = cache.groups[MTP_GROUP]
            mtp_blocks = g.mtp_block_row(m.block_slabs(r), m.block_slots(r))
        return Row(
            column,
            pos0,
            real,
            self._blocks(cache, r),
            -1 if pos0 == 0 else state(pos0 - 1),
            state(pos0 + real - 1),
            mtp_blocks,
            int(state_group.block_table_start_indices[r]),
            tuple(g.state_block_row(slab, slot) for slab, slot in state_group.row_addresses(r)),
        )

    def forward(self, plan) -> ForwardOutput:
        self._check_compiles()
        if any(media is not None for media in plan.media):
            raise NotImplementedError("this runner has no vision tower; text prompts only")
        cache = PagedCachePlan.from_plan(plan, self.cache_layout, decode_lookahead=self.lookahead)
        batch = cache.batch
        if batch.num_sequences > self.max_batch_size:
            raise ValueError(f"{batch.num_sequences} rows exceed {self.max_batch_size}")
        if int(batch.post_lengths.max()) + self.lookahead > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        prompt = plan.needs_prompt_logprobs
        if prompt and batch.phase != BatchPhase.PREFILL:
            raise ValueError("prompt logprobs are scored by prefill plans")
        params = plan_params(plan, self.vocab_size)
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        model = self.model
        outs, prompt_scores = [], []
        decode = batch.phase != BatchPhase.PREFILL
        if decode and self.mtp:
            return self._speculate_sync(plan, cache, params)
        seeds, offsets = plan_seeds(plan) if self.mtp else (None, None)
        if batch.phase == BatchPhase.PREFILL:
            requests = (
                score_requests(plan, self.vocab_size) if prompt else [None] * batch.num_sequences
            )
            for r in range(batch.num_sequences):
                ids = batch.token_ids[batch.token_slice(r)]
                pos0 = int(batch.context_lengths[r])
                scores = _Scorer(requests[r])
                chunks = prefill_chunks(len(ids))
                for s in chunks:
                    chunk = ids[s : s + chunks.step]
                    p = pos0 + s
                    positions = p + np.arange(len(chunk))
                    row = self._row(cache, r, 0, p, len(chunk))
                    T = model._launch(chunk, positions, [row], prefill=True)
                    if s + len(chunk) >= len(ids):
                        outs.append(model.sample_launched(T, params[r : r + 1], len(chunk) - 1))
                    # after sampling: a prefill chunk's head reruns at the scored columns
                    if len(cols := scores.columns(s, len(chunk))):
                        scores.add(s + cols, model.head_logits(cols))
                    if self.mtp:
                        sampling = Sampling(params[r : r + 1], seeds[r : r + 1], offsets[r : r + 1])
                        ends = s + len(chunk) >= len(ids)
                        drafts = model.prefill_mtp(
                            row, chunk, T, ends and bool(plan.sampling_mask[r]), sampling
                        )
                        if drafts is not None:
                            first = int(outs[-1][0][0])
                            self._drafts[int(batch.sequence_ids[r])] = (p + len(chunk), first, drafts)
                prompt_scores.append(scores.result())
        else:
            n = batch.num_sequences
            positions = np.asarray(batch.context_lengths, np.int64)
            for g in range(0, n, MAX_T):
                rows = [
                    self._row(cache, r, r - g, int(positions[r]), 1)
                    for r in range(g, min(n, g + MAX_T))
                ]
                sl = slice(g, g + len(rows))
                T = model._launch(batch.token_ids[sl], positions[sl], rows)
                outs.append(model.sample_launched(T, params[sl]))
        tok, lp, top_ids, top_lp = (np.concatenate(x) for x in zip(*outs, strict=True))
        counts = np.ones(len(tok), np.int64)
        columns = (
            Qwen3_5GaudiRunner._score_columns(params, counts, lp, top_ids, top_lp)
            if plan.needs_logprobs
            else {}
        )
        if prompt:
            columns.update(Qwen3_5GaudiRunner._prompt_columns(plan, prompt_scores))
        return ForwardOutput(
            token_ids=np.asarray(tok, np.uint32),
            token_indptr=np.arange(len(tok) + 1, dtype=np.uint32),
            **columns,
        )

    def _carried(self, plan, rows, anchors) -> np.ndarray:
        """Each row's drafts carried from its last step (zeros when they do not continue from this anchor,
        which only costs acceptance)."""
        drafts = np.zeros((len(rows), self.mtp), np.int64)
        for r, (sequence, row) in enumerate(zip(plan.sequence_ids, rows, strict=True)):
            carried = self._drafts.get(int(sequence))
            if carried and carried[0] == row.pos0 and carried[1] == anchors[r]:
                drafts[r] = carried[2]
        return drafts

    def _speculate_sync(self, plan, cache, params) -> ForwardOutput:
        """One MTP step of a decode plan, MAX_T // (K + 1) rows at a time"""
        batch = cache.batch
        B, group = batch.num_sequences, MAX_T // (self.mtp + 1)
        positions = np.asarray(batch.context_lengths, np.int64)
        seeds, _ = plan_seeds(plan)
        anchors = np.asarray(batch.token_ids, np.int64)
        tokens = []
        scores = []
        for g in range(0, B, group):
            rows = [self._row(cache, r, r - g, int(positions[r]), 1) for r in range(g, min(B, g + group))]
            sl = slice(g, g + len(rows))
            part = Sampling(params[sl], seeds[sl], (positions[sl] + 1) * 32)
            drafts = self._carried(_Slice(plan, sl), rows, anchors[sl])
            launched = self.model.verify_launch(rows, anchors[sl], drafts, part)
            done = self.model.verify_finish(launched)
            next_drafts = self.model.chain(launched, done)
            for r, row in enumerate(rows):
                sequence = int(batch.sequence_ids[g + r])
                position = row.pos0 + int(done.accepted[r]) + 1
                self._drafts[sequence] = (position, done.tokens[r][-1], next_drafts[r])
            tokens += done.tokens
            if plan.needs_logprobs:
                lp, top_ids, top_lp = launched.staged.result()[1:]
                cols = np.concatenate(done.columns)
                scores.append((lp[cols], top_ids[cols], top_lp[cols]))
        counts = np.array([len(t) for t in tokens], np.int64)
        columns = {}
        if plan.needs_logprobs:
            lp, top_ids, top_lp = (np.concatenate(x) for x in zip(*scores, strict=True))
            columns = Qwen3_5GaudiRunner._score_columns(params, counts, lp, top_ids, top_lp)
        return ForwardOutput(
            token_ids=np.asarray([t for row in tokens for t in row], np.uint32),
            token_indptr=np.concatenate(([0], np.cumsum(counts))).astype(np.uint32),
            **columns,
        )

    def _settle(self, feedback) -> None:
        """Finish an enqueued MTP step that ``feedback`` does not continue, so its commit lands first"""
        step = self._open
        if step is not None and step is not feedback:
            step.finish()
        self._open = None

    def enqueue(self, plan, *, feedback: _Feedback | _Verification | None = None):
        """Submit a decode step and return without waiting for it (the Engine's overlap): a continuation
        (``feedback``) reads its tokens from the previous step's samples on the device, at the previous
        positions + 1, with uniforms drawn at those exact positions. Other plans run synchronously."""
        self._settle(feedback)
        cache = PagedCachePlan.from_plan(plan, self.cache_layout, decode_lookahead=self.lookahead)
        batch = cache.batch
        B = batch.num_sequences
        if (
            plan.phase != BatchPhase.DECODE
            or plan.needs_prompt_logprobs
            or B > MAX_T // (self.mtp + 1)
            or any(media is not None for media in plan.media)
        ):
            if isinstance(feedback, _Verification):
                feedback.finish()
            return self.forward(plan)
        self._check_compiles()
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        if feedback is None:
            positions = np.asarray(batch.context_lengths, np.int64)
            tokens = batch.token_ids
        else:
            if isinstance(feedback, _Verification):
                feedback.finish()
            positions = np.asarray(feedback.positions, np.int64)
            tokens = None
            self.continuations += 1
        if int(positions.max()) + 1 + self.lookahead > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        seeds, _ = plan_seeds(plan)
        sampling = Sampling(plan_params(plan, self.vocab_size), seeds, (positions + 1) * 32)
        params = sampling.at(range(B), [0] * B)
        rows = [self._row(cache, r, r, int(positions[r]), 1) for r in range(B)]
        if self.mtp:
            if feedback is None:
                anchors = np.asarray(batch.token_ids, np.int64)
                drafts = self._carried(plan, rows, anchors)
            else:  # every rank is here, in the same order: the chain's collectives line up
                anchors, drafts = feedback.anchors, self.model.chain(*feedback.chain_inputs)
            sampling = Sampling(params, seeds, (positions + 1) * 32)
            launched = self.model.verify_launch(rows, anchors, drafts, sampling)
            for sequence in plan.sequence_ids:  # this step carries the drafts now
                self._drafts.pop(int(sequence), None)
            self._open = _Verification(self, plan, positions, params, launched)
            return self._open
        T = self.model._launch(tokens, positions, rows)
        staged = self.model.sample_async(T, params)
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

        return _Completion(build, _Feedback(positions + 1))

    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        """``enqueue`` on every rank of the replica (tensor parallel: the same plan)"""
        if plan is None:
            return None
        return self.enqueue(plan, feedback=feedback)

    def forward_distributed(self, plan, context, *, prepared=None) -> ForwardOutput | None:
        """Tensor parallel only: every rank of the replica runs the same plan."""
        return None if plan is None else self.forward(plan)

    def release(self, sequence_ids: list[int]) -> None:
        pass

    def close(self) -> None:
        if self.nranks > 1:
            self.device.lib.rg_ccl_destroy()
        self.device.close()


class _Slice:
    """A group of a plan's rows: their sequence ids"""

    def __init__(self, plan, sl: slice) -> None:
        self.sequence_ids = plan.sequence_ids[sl]


class _Verification:
    """An enqueued MTP step, and its continuation's feedback. Its verify pass is left running; ``finish`` (from the
    continuation's ``enqueue``, or from ``result``: the Engine's driver harvests before its followers do) accepts the
    drafts and commits the accepted tokens, with no collective, and records each row's next anchor and first draft for
    a later step of the row, on every rank alike. A continuation completes the drafts with ``model.chain``; a row
    resuming later verifies its first draft and zeros, which only costs acceptance."""

    def __init__(self, runner, plan, positions, params, launched) -> None:
        self._runner, self._plan, self._params = runner, plan, params
        self._start, self._launched = positions, launched
        self._output: ForwardOutput | None = None

    @property
    def feedback(self) -> _Verification:
        return self

    def finish(self) -> None:
        if self._output is not None:
            return
        runner, plan = self._runner, self._plan
        done = runner.model.verify_finish(self._launched)
        self.positions = self._start + done.accepted + 1
        self.anchors = np.array([t[-1] for t in done.tokens], np.int64)
        self.chain_inputs = (self._launched, done)
        for r, sequence in enumerate(plan.sequence_ids):
            runner._drafts[int(sequence)] = (
                int(self.positions[r]),
                int(self.anchors[r]),
                done.drafts[r],
            )
        counts = np.array([len(t) for t in done.tokens], np.int64)
        columns = {}
        if plan.needs_logprobs:
            lp, top_ids, top_lp = self._launched.staged.result()[1:]
            cols = np.concatenate(done.columns)
            columns = Qwen3_5GaudiRunner._score_columns(
                self._params, counts, lp[cols], top_ids[cols], top_lp[cols]
            )
        self._output = ForwardOutput(
            token_ids=np.asarray([t for row in done.tokens for t in row], np.uint32),
            token_indptr=np.concatenate(([0], np.cumsum(counts))).astype(np.uint32),
            **columns,
        )

    def result(self) -> ForwardOutput:
        self.finish()
        return self._output


class _Feedback:
    """Exact inputs of a decode continuation (the Engine's continuation plan only bounds them): each
    row's next position; its tokens stay on the device."""

    def __init__(self, positions: np.ndarray) -> None:
        self.positions = positions


class _Completion:
    """One enqueued decode step: ``result`` waits for its staged samples once."""

    def __init__(self, build, feedback: _Feedback) -> None:
        self.feedback = feedback
        self._build, self._result = build, None

    def result(self) -> ForwardOutput:
        if self._result is None:
            self._result = self._build()
        return self._result


class _Scorer:
    """Prompt logprobs of one prefill row from its chunks' logits (host side, at the scored positions)."""

    def __init__(self, request: ScoreRequest | None) -> None:
        self.request = request
        self.done: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def columns(self, start: int, length: int) -> np.ndarray:
        if self.request is None:
            return np.zeros(0, np.int64)
        p = np.asarray(self.request.positions, np.int64)
        return np.unique(p[(p >= start) & (p < start + length)]) - start

    def add(self, positions: np.ndarray, logits: np.ndarray) -> None:
        req = self.request
        assert req is not None
        row = {int(p): i for i, p in enumerate(positions)}
        for k, position in enumerate(req.positions):
            if (i := row.get(int(position))) is None:
                continue
            z = logits[i].astype(np.float64)
            m = z.max()
            logp = z - m - np.log(np.exp(z - m).sum())
            top = int(req.tops[k])
            ids = np.argsort(-logp, kind="stable")[:top] if top else np.zeros(0, np.int64)
            self.done[k] = (logp[np.asarray(req.targets[k], np.int64)], ids, logp[ids])

    def result(self) -> Scores | None:
        req = self.request
        if req is None:
            return None
        parts = [self.done[k] for k in range(len(req.positions))]
        return Scores(
            [p[0].astype(np.float32) for p in parts],
            [p[1] for p in parts],
            [p[2].astype(np.float32) for p in parts],
        )
