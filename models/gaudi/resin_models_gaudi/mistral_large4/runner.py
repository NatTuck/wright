"""Mistral-Large-4 on Gaudi2 as a Resin managed runner: one tensor-parallel rank per process.

The Engine owns the paged cache: every plan names each row's KV blocks (128 tokens, one per
slab). Every rank receives the same plan (``DistributedWorker`` broadcasts it within the
attention replica), runs the same launches and collectives, and samples the same tokens from
the all-gathered logits on its own device, so no token broadcast is needed; the driver's
output is the one the Engine reads.

Prefill rows run one at a time; decode rows run as one batch of up to 16. Sampling and logprobs run on the device with Resin's
semantics (``qwen3_5.sampling``), over the softcapped logits.

With a draft checkpoint (``dspark``), a decode step of up to ``dspark.MAX_ROWS`` rows drafts 3
tokens per row from each row's anchor, verifies [anchor, drafts] in one target step and emits the
accepted prefix plus the target's next token (``SpeculativeStats.accept``). Every emitted token is
the target's sample at its position, computed to the bit as plain decoding computes it
(``Dims.exact``), so speculative output equals plain decoding, greedy or seeded. Rejected positions
need no rollback: the next verify step rewrites them before any column can see them. Larger
batches decode without speculation. With the Engine's overlap, the next step's ``enqueue`` waits for
this step's samples (the next block's positions follow its acceptance), drafts and launches, so the
Engine's work overlaps the device's.
"""

from __future__ import annotations

import warnings

import numpy as np

from resin import BatchPhase, ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.models.mistral_large4.config import MistralLarge4Config
from resin_models_gaudi.gguf.reader import GgufModel
from resin_models_gaudi.mistral_large4 import dspark, parallel
from resin_models_gaudi.mistral_large4.cache import KV_GROUP, PAGE_TOKENS, CacheGeometry
from resin_models_gaudi.mistral_large4.model import (
    MAX_T,
    PREFILL_COLUMNS,
    Ml4Gaudi,
    Row,
    prefill_chunks,
)
from resin_models_gaudi.qwen3_5.model import Sampling, ScoreRequest, Scores
from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner, score_requests
from resin_models_gaudi.qwen3_5.sampling import TOP_MAX, plan_params, plan_seeds
from resin_models_gaudi.synapse.progress import info
from resin_models_gaudi.synapse.runtime import open_device

PREFILL_CHUNK_SIZE = PREFILL_COLUMNS
MAX_BATCH = 64  # decode rows; batches above MAX_T run as consecutive MAX_T-row steps


class MistralLarge4GaudiRunner:
    """``ManagedRunner`` over ``Ml4Gaudi``; tensor parallel over ``groups``' attention replica."""

    # heads and every expert's width are sliced across the ranks
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
        draft_checkpoint: str | None = None,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
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
                    "Mistral-Large-4 on Gaudi supports tensor parallelism only"
                )
            rank, nranks = groups.replica.local_rank, groups.replica.size
        self.rank, self.nranks = rank, nranks
        self.device = open_device(card)
        if groups is not None and nranks > 1:
            parallel.hccl_rendezvous(
                self.device, groups.rank, nranks, groups.replica.handle
            )
        gguf = GgufModel(checkpoint)
        cfg = MistralLarge4Config.from_gguf(gguf.metadata)
        if num_layers:
            cfg = cfg.truncated(num_layers)
        self.config = cfg
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        speculative = draft_checkpoint is not None
        geometry = CacheGeometry.from_config(
            cfg, dspark.DRAFT_LAYERS if speculative else 0
        )
        self.speculative_tokens = dspark.DEPTH if speculative else 0
        self.speculate = (
            speculative  # (off: plain decoding with the same graphs, for parity)
        )
        # a verify step writes positions p .. p + K (the draft block's keys are within them)
        self.lookahead = self.speculative_tokens
        self._speculation = SpeculativeStats(self.speculative_tokens)

        def capacity(free: int) -> int:
            """the arena every rank can hold: the smallest free memory over the ranks"""
            if groups is not None and nranks > 1:
                import torch
                import torch.distributed as dist

                box = torch.tensor([free], dtype=torch.int64)
                dist.all_reduce(box, op=dist.ReduceOp.MIN, group=groups.replica.handle)
                free = int(box.item())
            fit = geometry.capacity_for(free, geometry.block_bytes)
            if parent_capacity is not None:
                if parent_capacity > fit:
                    raise ValueError(f"parent_capacity {parent_capacity} exceeds {fit}")
                return parent_capacity
            return fit

        self.model = Ml4Gaudi(
            self.device,
            gguf,
            cfg,
            capacity=capacity,
            max_seq_len=max_seq_len,
            rank=rank,
            nranks=nranks,
            log=log if rank == 0 else (lambda *_: None),
            draft=draft_checkpoint,
            prefill_columns=prefill_chunk_size,
        )
        gguf.close()
        if groups is not None and nranks > 1:
            parallel.barrier(groups.replica.handle)
        self.vocab_size = cfg.vocab_size
        self.geometry = geometry
        self.cache_layout = geometry.layout(self.model.capacity)
        self._binding = CachePoolBinding(
            self.cache_layout, resource="Mistral-Large-4 Gaudi cache"
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=prefill_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.lookahead,
            cache_layout=self.cache_layout,
        )
        self.compiles_at_start = self._compiles_seen = self.device.compile_count
        self.continuations = 0  # decode steps enqueued from device feedback

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
        return (
            self.geometry.block_row(kv.block_slabs(r))
            + np.asarray(kv.block_slots(r), np.int64) * self.geometry.block_rows
        )

    def forward(self, plan) -> ForwardOutput:
        self._check_compiles()
        if any(media is not None for media in plan.media):
            raise NotImplementedError(
                "this runner has no vision tower; text prompts only"
            )
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead
        )
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
        drafter = model.drafter
        prompt_scores = []
        if (
            self.speculate
            and batch.phase != BatchPhase.PREFILL
            and batch.num_sequences <= dspark.MAX_ROWS
        ):
            starts = np.asarray(batch.context_lengths, np.int64)
            anchors = np.asarray(batch.token_ids, np.int64)
            return self._speculate(plan, cache, starts, anchors).result()
        if batch.phase == BatchPhase.PREFILL:
            outs = []
            requests = (
                self._score_requests(plan) if prompt else [None] * batch.num_sequences
            )
            for r in range(batch.num_sequences):
                ids = batch.token_ids[batch.token_slice(r)]
                pos0 = int(batch.context_lengths[r])
                blocks = self._blocks(cache, r)
                scores = _Scorer(requests[r], self.config.final_logit_softcap)
                chunks = prefill_chunks(len(ids), model.prefill_columns)
                for s in chunks:
                    chunk = ids[s : s + chunks.step]
                    p = pos0 + s
                    step = (
                        chunk,
                        p + np.arange(len(chunk)),
                        [Row(0, p, len(chunk), blocks)],
                    )
                    T = model._launch(*step)
                    if drafter is not None:
                        drafter.inject(T, step[1], blocks[step[1] // PAGE_TOKENS])
                    if s + len(chunk) >= len(ids):
                        outs.append(
                            model.sample_launched(T, params[r : r + 1], len(chunk) - 1)
                        )
                    # after sampling: a prefill chunk's head reruns at the scored columns
                    if len(cols := scores.columns(s, len(chunk))):
                        scores.add(
                            s + cols,
                            model.head_logits(cols)
                            if T > MAX_T
                            else model.logits_host(T)[cols],
                        )
                prompt_scores.append(scores.result())
            tok, lp, top_ids, top_lp = (
                np.concatenate(x) for x in zip(*outs, strict=True)
            )
        else:
            n = batch.num_sequences
            positions = np.asarray(batch.context_lengths, np.int64)
            outs = []
            for g in range(0, n, MAX_T):  # MAX_T rows per step
                rows = [
                    Row(r - g, int(positions[r]), 1, self._blocks(cache, r))
                    for r in range(g, min(n, g + MAX_T))
                ]
                sl = slice(g, g + len(rows))
                T = model._launch(batch.token_ids[sl], positions[sl], rows)
                if drafter is not None:
                    drafter.inject(
                        T,
                        positions[sl],
                        np.array([r.blocks[r.pos0 // PAGE_TOKENS] for r in rows]),
                    )
                outs.append(model.sample_launched(T, params[sl]))
            tok, lp, top_ids, top_lp = (
                np.concatenate(x) for x in zip(*outs, strict=True)
            )
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

    def _speculate(
        self, plan, cache: PagedCachePlan, starts: np.ndarray, anchors: np.ndarray
    ) -> _Verification:
        """One speculative decode step at exact anchors: draft from each row's anchor, verify
        [anchor, drafts] in one target launch (every column sampled with its position's uniform)
        and feed the columns' hidden states to the drafter's cache, without waiting for them."""
        batch, model = cache.batch, self.model
        B, K = batch.num_sequences, self.speculative_tokens
        blocks = [self._blocks(cache, r) for r in range(B)]
        drafts = model.drafter.draft(
            [(int(anchors[r]), int(starts[r]), blocks[r]) for r in range(B)]
        )
        tokens = np.concatenate([anchors[:, None], drafts], 1).ravel()
        positions = (starts[:, None] + np.arange(K + 1)).ravel()
        rows = [Row(r * (K + 1), int(starts[r]), K + 1, blocks[r]) for r in range(B)]
        T = model._launch(tokens, positions, rows)
        model.drafter.inject(
            T,
            positions,
            np.concatenate(
                [
                    blocks[r][(starts[r] + np.arange(K + 1)) // PAGE_TOKENS]
                    for r in range(B)
                ]
            ),
        )
        seeds, _ = plan_seeds(plan)
        sampling = Sampling(
            plan_params(plan, self.vocab_size), seeds, (starts + 1) * 32
        )
        grid = sampling.grid(B, range(K + 1)).reshape(B * (K + 1), -1)
        staged = model.sample_async(T, grid)
        params = sampling.at(range(B), [0] * B)
        return _Verification(self, plan, starts, drafts, params, staged)

    def _score_requests(self, plan) -> list[ScoreRequest | None]:
        return score_requests(plan, self.vocab_size)

    def enqueue(self, plan, *, feedback: _Feedback | None = None):
        """Submit a decode step and return without waiting for it (the Engine's overlap): a
        continuation (``feedback``) reads its tokens from the previous step's samples on the
        device, at the previous positions + 1, and its uniforms are drawn at those exact
        positions. A speculative step's continuation first waits for that step's samples: its
        anchors and positions follow the acceptance. Other plans (prefill, prompt logprobs,
        batches above MAX_T rows) run synchronously."""
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead
        )
        batch = cache.batch
        B = batch.num_sequences
        if (
            plan.phase != BatchPhase.DECODE
            or plan.needs_prompt_logprobs
            or B > MAX_T
            or any(media is not None for media in plan.media)
        ):
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
        if self.speculate and B <= dspark.MAX_ROWS:
            anchors = (
                feedback.anchors
                if feedback is not None
                else np.asarray(batch.token_ids, np.int64)
            )
            return self._speculate(plan, cache, positions, anchors)
        seeds, _ = plan_seeds(plan)
        sampling = Sampling(
            plan_params(plan, self.vocab_size), seeds, (positions + 1) * 32
        )
        params = sampling.at(range(B), [0] * B)  # the uniforms of the exact positions
        rows = [Row(r, int(positions[r]), 1, self._blocks(cache, r)) for r in range(B)]
        T = self.model._launch(tokens, positions, rows)
        if self.model.drafter is not None:
            self.model.drafter.inject(
                T,
                positions,
                np.array([r.blocks[r.pos0 // PAGE_TOKENS] for r in rows]),
            )
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

    def forward_distributed(
        self, plan, context, *, prepared=None
    ) -> ForwardOutput | None:
        """Tensor parallel only: every rank of the replica runs the same plan."""
        return None if plan is None else self.forward(plan)

    def release(self, sequence_ids: list[int]) -> None:
        pass

    def close(self) -> None:
        if self.nranks > 1:
            self.device.lib.rg_ccl_destroy()
        self.device.close()


class _Feedback:
    """Exact inputs of a decode continuation (the Engine's continuation plan only bounds
    them): each row's next position; its tokens stay on the device."""

    def __init__(self, positions: np.ndarray) -> None:
        self.positions = positions


class _Verification:
    """An enqueued speculative step, and its continuation's feedback: ``finish`` waits for its
    samples and accepts the drafts; then ``positions`` and ``anchors`` are each row's next anchor
    position and token."""

    def __init__(self, runner, plan, starts, drafts, params, staged) -> None:
        self._runner, self._plan, self._starts = runner, plan, starts
        self._drafts, self._params, self._staged = drafts, params, staged
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
            self._drafts.tolist(), tok.reshape(B, K + 1).tolist()
        )
        counts = np.asarray(acceptance.accepted_counts, np.int64) + 1
        self.positions = self._starts + counts
        self.anchors = np.array([row[-1] for row in acceptance.token_rows], np.int64)
        columns = {}
        if plan.needs_logprobs:
            picked = np.concatenate(
                [r * (K + 1) + np.arange(counts[r]) for r in range(B)]
            )
            columns = Qwen3_5GaudiRunner._score_columns(
                self._params, counts, lp[picked], top_ids[picked], top_lp[picked]
            )
        self._output = ForwardOutput(
            token_ids=np.asarray(
                [t for row in acceptance.token_rows for t in row], np.uint32
            ),
            token_indptr=np.concatenate(([0], np.cumsum(counts))).astype(np.uint32),
            **columns,
        )

    def result(self) -> ForwardOutput:
        self.finish()
        assert self._output is not None
        return self._output


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
    """Prompt logprobs of one prefill row from its chunks' logits (host side: softcap, then
    log-softmax, at the scored positions only)."""

    def __init__(self, request: ScoreRequest | None, cap: float) -> None:
        self.request, self.cap = request, cap
        self.done: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def columns(self, start: int, length: int) -> np.ndarray:
        """the chunk columns [start, start + length) holds that the request scores"""
        if self.request is None:
            return np.zeros(0, np.int64)
        p = np.asarray(self.request.positions, np.int64)
        return np.unique(p[(p >= start) & (p < start + length)]) - start

    def add(self, positions: np.ndarray, logits: np.ndarray) -> None:
        """the raw logits (n, V) at prompt ``positions`` (n,)"""
        req = self.request
        assert req is not None
        row = {int(p): i for i, p in enumerate(positions)}
        for k, position in enumerate(req.positions):
            if (i := row.get(int(position))) is None:
                continue
            z = self.cap * np.tanh(logits[i].astype(np.float64) / self.cap)
            m = z.max()
            logp = z - m - np.log(np.exp(z - m).sum())
            top = int(req.tops[k])
            ids = (
                np.argsort(-logp, kind="stable")[:top] if top else np.zeros(0, np.int64)
            )
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
