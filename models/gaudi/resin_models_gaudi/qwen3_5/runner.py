"""Qwen3.5/3.8 FP8 on one Gaudi card as a Resin managed runner.

The Engine owns the paged cache: every plan names each row's KV blocks (128 tokens) and its
STATE blocks (4096 tokens) by (slab, slot) in one arena. A row reads its recurrent state from
the STATE block of its position ``context - 1`` and writes it to the block of its last
position, so a step that crosses a 4096-token boundary leaves the earlier block as the prefix
snapshot. Prefill rows run one at a time; decode rows run as one batch.

Sampling and logprobs run on the device (``sampling.py``) with Resin's semantics; up to 64 top
logprobs per sampled token. Prompt logprobs (the next prompt token's, up to 1000 top entries
and any requested targets) are scored per prefill chunk (``Qwen35Gaudi.prefill``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from resin import BatchPhase, ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin_models_gaudi.qwen3_5.cache import (
    KV_GROUP,
    MTP_GROUP,
    STATE_GROUP,
    STATE_TOKENS,
)
from resin_models_gaudi.qwen3_5.model import (
    DECODE_BUCKETS,
    Qwen35Gaudi,
    RowCache,
    Sampling,
    ScoreRequest,
)
from resin_models_gaudi.qwen3_5.sampling import TOP_MAX, plan_params, plan_seeds
from resin_models_gaudi.synapse.runtime import open_device

PREFILL_CHUNK_SIZE = 4096
# tokens per Engine prefill step that run as whole launches of the compiled prefill widths
PREFILL_CHUNK_SIZES = (512, 1024, 2048, PREFILL_CHUNK_SIZE)


def score_requests(plan, vocab: int) -> list[ScoreRequest | None]:
    """Per row, the plan's prompt positions to score (indices into the plan's tokens)."""
    indices = np.asarray(plan.prompt_score_indices, np.int64)
    targets = np.asarray(plan.prompt_score_token_ids, np.int64)
    tops = np.minimum(np.asarray(plan.prompt_score_top_counts, np.int64), vocab)
    extra_ptr = np.asarray(plan.prompt_target_indptr, np.int64)
    extra = np.asarray(plan.prompt_target_token_ids, np.int64)
    indptr = np.asarray(plan.token_indptr, np.int64)
    requests = []
    for r in range(plan.num_sequences):
        inside = np.flatnonzero((indices >= indptr[r]) & (indices < indptr[r + 1]))
        if not len(inside):
            requests.append(None)
            continue
        requests.append(
            ScoreRequest(
                indices[inside] - indptr[r],
                [
                    np.concatenate(
                        ([targets[k]], extra[extra_ptr[k] : extra_ptr[k + 1]])
                    )
                    if len(extra_ptr) > 1
                    else np.array([targets[k]])
                    for k in inside
                ],
                tops[inside],
            )
        )
    return requests


class _Feedback:
    """Exact inputs of a target-only decode continuation (the Engine's continuation plan
    only bounds them): each row's next position; the next tokens stay on the device (the
    previous step's samples)."""

    def __init__(self, positions) -> None:
        self.positions = positions


class _Verification:
    """An enqueued MTP step, and its continuation's feedback. Its verify pass is left
    running; ``finish`` (from the continuation's ``enqueue``, or from ``result`` when no step
    continues this one) accepts the drafts, commits the accepted tokens and launches the next
    draft chain, whose drafts 2 .. K land in the step's staging. Then ``positions`` and
    ``anchors`` are each row's next anchor position and token."""

    def __init__(self, runner, plan, positions, params, launched) -> None:
        self._runner, self._plan, self._params = runner, plan, params
        self._start, self._launched = positions, launched
        self._output: ForwardOutput | None = None
        self.continued = False  # a continuation took the drafts

    @property
    def feedback(self) -> _Verification:
        return self

    def finish(self) -> None:
        if self._output is not None:
            return
        runner, stage = self._runner, self._launched.stage
        done = runner.model.verify_finish(self._launched)
        self.positions = self._start + done.accepted + 1
        self.anchors = np.array([t[-1] for t in done.tokens], np.int64)
        self._first = done.drafts[:, 0]
        self._output = runner._verified_output(
            self._plan, done, self._params, (stage.lp, stage.top_idx, stage.top_lp)
        )

    def drafts(self, K: int) -> np.ndarray:
        """Each row's next drafts (waits for the draft chain)."""
        drafts = np.zeros((len(self.positions), K), np.int64)
        drafts[:, 0] = self._first
        if K > 1:
            stage = self._launched.stage
            stage.wait(1)
            drafts[:, 1:] = stage.drafts[1:K, : len(self.positions)].T
        return drafts

    def result(self) -> ForwardOutput:
        self.finish()
        if (
            not self.continued
        ):  # no step continues this one: carry its drafts by sequence
            self.continued = True
            drafts = self.drafts(self._runner.mtp)
            for r, sequence in enumerate(self._plan.sequence_ids):
                self._runner._drafts[int(sequence)] = (
                    int(self.positions[r]),
                    int(self.anchors[r]),
                    drafts[r],
                )
        return self._output


class _Completion:
    """One enqueued decode step: ``result`` waits for its staging event and builds the
    output once."""

    def __init__(self, stage, build, feedback: _Feedback) -> None:
        self.feedback = feedback
        self._stage, self._build, self._result = stage, build, None

    def result(self) -> ForwardOutput:
        if self._result is None:
            self._stage.wait()
            self._result = self._build()
        return self._result


def image_span(ids, count: int) -> int:
    """Where the image's ``count`` placeholder tokens start in the prompt.

    Kept for callers that prepare a prompt from whole-rendered media (the dev harnesses);
    serving does not use it - see `Qwen3_5GaudiRunner._vision` for why scanning token ids
    cannot work across prefill chunks.
    """
    runs, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            runs.append((start, i - start))
            start = i
    hits = [first for first, length in runs if length == count]
    if len(hits) != 1:
        found = ", ".join(str(first) for first in hits) or "none"
        raise ValueError(
            f"cannot place the image: the prompt has {len(hits)} maximal runs of {count} "
            f"identical tokens (expected exactly 1), starting at {found}"
        )
    return hits[0]


class Qwen3_5GaudiRunner:
    """``ManagedRunner`` over ``Qwen35Gaudi``."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        card: int | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 32768,
        memory_utilization: float = 0.9,
        parent_capacity: int | None = None,
        mtp: int = 0,
        media: bool = False,
    ) -> None:
        if max_batch_size > DECODE_BUCKETS[-1]:
            raise ValueError(f"max_batch_size exceeds {DECODE_BUCKETS[-1]}")
        self.device = open_device(card)
        self.checkpoint = checkpoint
        self._tower = None  # built on the first image, so a text server never loads it
        self._prepared = {}  # media digest -> ImagePrompt, so chunks reuse one tower run
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.model = Qwen35Gaudi(
            self.device,
            checkpoint,
            parent_capacity=parent_capacity,
            memory_utilization=memory_utilization,
            max_positions=max_seq_len,
            decode_buckets=[b for b in DECODE_BUCKETS if b < max_batch_size]
            + [min(b for b in DECODE_BUCKETS if b >= max_batch_size)],
            mtp=mtp,
            media=media,
        )
        self.mtp = mtp
        self.speculative_tokens = mtp
        self.vocab_size = self.model.dims.vocab
        self._speculation = self.model.speculation
        # a verify step writes positions p .. p + K; its draft chain's MTP KV reaches p + 2K - 1
        self.lookahead = 2 * mtp - 1 if mtp else 0
        # sequence id -> (the next anchor's position, the anchor token, its drafts)
        self._drafts: dict[int, tuple[int, int, np.ndarray]] = {}
        # each sequence's mrope delta: after an image prompt the model rotates at the token
        # position plus this, because the image's tokens occupy fewer positions than they
        # have tokens. Text prompts leave it at zero.
        self._rope_delta: dict[int, int] = {}
        # sequences whose history includes an image: they decode without speculation, because
        # an image prompt does not run the MTP pass (see `Qwen35Gaudi.prefill`)
        self._no_mtp: set[int] = set()
        self._stage = 0  # the next enqueued step's staging set
        self._open: _Verification | None = None  # the last enqueued MTP step
        self.geometry = self.model.geometry
        self.cache_layout = self.geometry.layout(self.model.parent_capacity)
        self._binding = CachePoolBinding(
            self.cache_layout, resource="Qwen3.5 Gaudi cache"
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.lookahead,
            cache_layout=self.cache_layout,
        )
        self.compiles_at_start = self._compiles_seen = self.device.compile_count

    @property
    def kv_pool_bytes(self) -> int:
        """The KV pool on the card."""
        return self.model.arena_bytes

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

    def _rows(self, cache: PagedCachePlan, positions=None) -> list[RowCache]:
        """Each row's cache; ``positions`` (a continuation's exact first positions) replace
        the plan's context lengths."""
        g = self.geometry
        kv, state = cache.groups[KV_GROUP], cache.groups[STATE_GROUP]
        mtp = cache.groups[MTP_GROUP] if self.mtp else None
        batch = cache.batch
        rows = []
        for r in range(batch.num_sequences):
            context = int(
                batch.context_lengths[r] if positions is None else positions[r]
            )
            last = context + int(batch.query_lengths[r]) - 1
            first_state = int(state.block_table_start_indices[r])
            states = tuple(
                g.state_block_row(slab, slot) for slab, slot in state.row_addresses(r)
            )

            def state_row(position: int) -> int:
                return states[position // STATE_TOKENS - first_state]

            rows.append(
                RowCache(
                    context,
                    g.kv_block_row(kv.block_slabs(r), kv.block_slots(r)),
                    -1 if context == 0 else state_row(context - 1),
                    state_row(last),
                    None
                    if mtp is None
                    else g.mtp_block_row(mtp.block_slabs(r), mtp.block_slots(r)),
                    first_state,
                    states,
                    self._rope_delta.get(int(batch.sequence_ids[r]), 0),
                )
            )
        return rows

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start

    def _check_compiles(self) -> None:
        """Warn when a Synapse graph compiled after startup (every recipe is compiled or
        loaded during construction; serving must not compile)."""
        count = self.device.compile_count
        if count != self._compiles_seen:
            self._compiles_seen = count
            warnings.warn(
                f"{count - self.compiles_at_start} Synapse graph compilations after startup",
                RuntimeWarning,
                stacklevel=3,
            )

    def _vision(self, media):
        """The image rows, their span and this prompt's rope rows, or None for text.

        The tower is built on the first image and kept: loading and uploading the checkpoint
        is 5-15 s, which must not be paid per request. `RESIN_GAUDI_VISION` decides whether
        it runs on the card or on the CPU oracle.

        The span comes from the media payload's ``image_positions`` (see
        `prepare_image_prompt`); the token ids are not scanned, because this runner only
        ever sees one prefill chunk's tokens and the image's run of placeholders can be
        split across chunks (or start inside a prefix-cache restore). The prepared prompt
        is cached on the payload's ``digest``: a chunked prefill hands every chunk the same
        media, and without the cache the tower would rerun per chunk.
        """
        if media is None:
            return None
        from resin_models_gaudi.qwen3_5.media import (
            prepare_image_prompt,
            tower_from_checkpoint,
        )

        if self._tower is None:
            import json
            from pathlib import Path

            raw = json.loads((Path(self.checkpoint) / "config.json").read_text())
            self._tower = tower_from_checkpoint(self.checkpoint, raw, self.device)
        key = media.digest
        prompt = self._prepared.get(key)
        if prompt is None:
            prompt = prepare_image_prompt(media, self._tower, self.model.config)
            self._prepared = {key: prompt}
        return (
            prompt.features,
            prompt.first,
            prompt.image_tokens,
            prompt.rope,
            prompt.delta,
        )

    def _plain(self, plan) -> bool:
        """Whether this batch must decode without speculation: any row with an image in its
        history, since an image prompt does not run the MTP pass."""
        return any(int(sequence) in self._no_mtp for sequence in plan.sequence_ids)

    def _settle(self, feedback=None) -> None:
        """Finish an enqueued MTP step that ``feedback`` does not continue, so its commit
        precedes any other work."""
        open_, self._open = self._open, None
        if open_ is not None and open_ is not feedback:
            open_.result()

    def forward(self, plan) -> ForwardOutput:
        self._check_compiles()
        self._settle()
        decode = plan.phase == BatchPhase.DECODE
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead if decode else 0
        )
        batch = cache.batch
        if batch.num_sequences > self.max_batch_size:
            raise ValueError(f"{batch.num_sequences} rows exceed {self.max_batch_size}")
        if int(batch.post_lengths.max()) > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        prompt = plan.needs_prompt_logprobs
        if prompt and batch.phase != BatchPhase.PREFILL:
            raise ValueError("prompt logprobs are scored by prefill plans")
        params = plan_params(plan, self.model.dims.vocab)
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        if batch.phase == BatchPhase.PREFILL:
            # a prompt starting at position 0 replaces the sequence's history, and an image
            # prompt sets its own delta below; either way nothing older may survive into the
            # rows this plan builds. `_no_mtp` goes too: a sequence re-prefilled from
            # scratch is a new conversation and speculates again
            for r in range(batch.num_sequences):
                if int(batch.context_lengths[r]) == 0 or plan.media[r] is not None:
                    self._rope_delta.pop(int(batch.sequence_ids[r]), None)
                    self._no_mtp.discard(int(batch.sequence_ids[r]))
        rows = self._rows(cache)
        seeds, offsets = plan_seeds(plan)
        scores = plan.needs_logprobs
        lp = top_ids = top_lp = None
        prompt_scores = []
        if decode and self.mtp and not self._plain(plan):
            return self._speculate(plan, rows, params, Sampling(params, seeds, offsets))
        if batch.phase == BatchPhase.PREFILL:
            tokens, parts = [], []
            requests = self._score_requests(plan) if prompt else [None] * len(rows)
            for r, row in enumerate(rows):
                ids = batch.token_ids[batch.token_slice(r)].tolist()
                sampling = Sampling(
                    params[r : r + 1], seeds[r : r + 1], offsets[r : r + 1]
                )
                vision = self._vision(plan.media[r])
                done = self.model.prefill(
                    row,
                    ids,
                    sampling,
                    requests[r],
                    draft=bool(plan.sampling_mask[r]),
                    media=vision,
                )
                if vision is not None:
                    self._rope_delta[int(batch.sequence_ids[r])] = int(vision[4])
                    self._no_mtp.add(int(batch.sequence_ids[r]))
                tokens.append(done.token)
                prompt_scores.append(done.scores)
                if done.drafts is not None:
                    sequence = int(batch.sequence_ids[r])
                    position = row.pos0 + len(ids)
                    self._drafts[sequence] = (position, done.token, done.drafts)
                if scores:
                    parts.append(self.model.scores(1))
            if scores:
                lp, top_ids, top_lp = (
                    np.concatenate(p) for p in zip(*parts, strict=True)
                )
        else:
            tokens = self.model.decode(rows, batch.token_ids.tolist(), params).tolist()
            if scores:
                lp, top_ids, top_lp = self.model.scores(len(rows))
        counts = np.ones(len(tokens), np.int64)
        columns = (
            self._score_columns(params, counts, lp, top_ids, top_lp) if scores else {}
        )
        if prompt:
            columns.update(self._prompt_columns(plan, prompt_scores))
        return ForwardOutput(
            token_ids=np.asarray(tokens, np.uint32),
            token_indptr=np.arange(len(tokens) + 1, dtype=np.uint32),
            **columns,
        )

    def enqueue(self, plan, *, feedback: _Feedback | _Verification | None = None):
        """Submit a decode step and return without waiting for it: the device continues from
        ``feedback`` (the previous step's exact positions, tokens and drafts) while the Engine
        commits the previous step. Other plans run synchronously."""
        if plan.phase != BatchPhase.DECODE or plan.needs_prompt_logprobs:
            return self.forward(plan)
        self._check_compiles()
        self._settle(feedback)
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead
        )
        batch = cache.batch
        B = batch.num_sequences
        if B > self.max_batch_size:
            raise ValueError(f"{B} rows exceed {self.max_batch_size}")
        wanted = [p.top_logprobs for p in plan.sampling_params if p.logprobs]
        if wanted and max(wanted) > TOP_MAX:
            raise NotImplementedError(f"top_logprobs above {TOP_MAX}")
        if feedback is None:
            positions = batch.context_lengths.astype(np.int64)
        else:
            if isinstance(feedback, _Verification):
                feedback.finish()
                feedback.continued = True
            positions = np.asarray(feedback.positions, np.int64)
        if int(positions.max()) + 1 + self.lookahead > self.max_seq_len:
            raise ValueError(f"a row exceeds {self.max_seq_len} positions")
        rows = self._rows(cache, positions)
        seeds, _ = plan_seeds(plan)
        sampling = Sampling(
            plan_params(plan, self.model.dims.vocab), seeds, (positions + 1) * 32
        )
        params = sampling.at(range(B), [0] * B)  # the uniforms of the exact positions
        sampling = Sampling(params, seeds, (positions + 1) * 32)
        stage = self.model.staging[self._stage]
        self._stage ^= 1
        scores = plan.needs_logprobs
        if not self.mtp or self._plain(plan):
            tokens = None if feedback is not None else batch.token_ids.tolist()
            self.model.decode(rows, tokens, sampling, stage=stage, scores=scores)

            def build() -> ForwardOutput:
                counts = np.ones(B, np.int64)
                columns = {}
                if scores:
                    columns = self._score_columns(
                        params,
                        counts,
                        stage.lp[:B],
                        stage.top_idx[:B],
                        stage.top_lp[:B],
                    )
                return ForwardOutput(
                    token_ids=np.array(stage.tokens[:B], np.uint32),
                    token_indptr=np.arange(B + 1, dtype=np.uint32),
                    **columns,
                )

            return _Completion(stage, build, _Feedback(positions + 1))
        if feedback is None:
            anchors = np.asarray(batch.token_ids, np.int64)
            drafts = self._carried(plan, rows, anchors)
        else:  # the previous step left these tokens on the device; drafts read at finish
            anchors = feedback.anchors

            def drafts() -> np.ndarray:
                return feedback.drafts(self.mtp)

        launched = self.model.verify_launch(
            rows,
            anchors,
            drafts,
            sampling,
            stage=stage,
            scores=scores,
            chained=feedback is not None,
        )
        for sequence in plan.sequence_ids:  # this step carries the drafts now
            self._drafts.pop(int(sequence), None)
        self._open = _Verification(self, plan, positions, params, launched)
        return self._open

    def _carried(self, plan, rows, anchors) -> np.ndarray:
        """Each row's drafts carried from its last step (zeros when they do not continue from
        this anchor, which only costs acceptance)."""
        drafts = np.zeros((len(rows), self.mtp), np.int64)
        for r, (sequence, row) in enumerate(zip(plan.sequence_ids, rows, strict=True)):
            carried = self._drafts.get(int(sequence))
            if carried and carried[0] == row.pos0 and carried[1] == anchors[r]:
                drafts[r] = carried[2]
        return drafts

    def _verified_output(self, plan, done, params, scores=None) -> ForwardOutput:
        """A verify step's output; ``scores``: its per-column (lp, top ids, top lp), else
        read from the device."""
        counts = np.array([len(t) for t in done.tokens], np.int64)
        columns = {}
        if plan.needs_logprobs:
            if scores is None:
                bucket = min(b for b in self.model.decode_buckets if b >= len(counts))
                scores = self.model.scores(bucket * (self.mtp + 1))
            lp, top_ids, top_lp = scores
            picked = np.concatenate(done.columns)
            columns = self._score_columns(
                params, counts, lp[picked], top_ids[picked], top_lp[picked]
            )
        return ForwardOutput(
            token_ids=np.asarray([t for row in done.tokens for t in row], np.uint32),
            token_indptr=np.concatenate(([0], np.cumsum(counts))).astype(np.uint32),
            **columns,
        )

    def _speculate(self, plan, rows, params, sampling) -> ForwardOutput:
        """One MTP step of a decode plan: verify each row's anchor and carried drafts (zeros
        when the row's drafts do not continue from this anchor, which only costs acceptance)."""
        anchors = np.asarray(plan.token_ids, np.int64)
        drafts = self._carried(plan, rows, anchors)
        done = self.model.verify(rows, anchors, drafts, sampling)
        output = self._verified_output(plan, done, params)
        for r, (sequence, row) in enumerate(zip(plan.sequence_ids, rows, strict=True)):
            position = row.pos0 + int(done.accepted[r]) + 1
            self._drafts[int(sequence)] = (position, done.tokens[r][-1], done.drafts[r])
        return output

    def _score_requests(self, plan) -> list[ScoreRequest | None]:
        return score_requests(plan, self.model.dims.vocab)

    @staticmethod
    def _prompt_columns(plan, prompt_scores) -> dict[str, np.ndarray]:
        """``resin.backend.sampling.PromptScores.output``'s columns, in the plan's order."""
        targets, top_ids, top_lp = [], [], []
        for scored in prompt_scores:
            if scored is not None:
                targets += scored.targets
                top_ids += scored.top_ids
                top_lp += scored.top_logprobs

        def offsets(counts):
            return np.concatenate(([0], np.cumsum(counts))).astype(np.uint32)

        columns = {
            "prompt_logprobs": np.array([t[0] for t in targets], np.float32),
            "prompt_logprob_indptr": np.asarray(plan.prompt_score_indptr, np.uint32),
        }
        counts = np.array([len(t) for t in top_ids], np.int64)
        if counts.max(initial=0):
            columns["prompt_top_indptr"] = offsets(counts)
            columns["prompt_top_token_ids"] = np.concatenate(top_ids).astype(np.uint32)
            columns["prompt_top_logprobs"] = np.concatenate(top_lp).astype(np.float32)
        extra = [t[1:] for t in targets]
        if sum(len(t) for t in extra):
            columns["prompt_target_indptr"] = offsets([len(t) for t in extra])
            columns["prompt_target_token_ids"] = np.asarray(
                plan.prompt_target_token_ids, np.uint32
            )
            columns["prompt_target_logprobs"] = np.concatenate(extra).astype(np.float32)
        return columns

    @staticmethod
    def _score_columns(params, counts, lp, top_ids, top_lp) -> dict[str, np.ndarray]:
        """``resin.backend.sampling.format_output``'s logprob columns: every output token of a
        scored row (the params' logprob flag) gets its score, then its row's ntop top entries.
        ``counts``: output tokens per row; lp, top_ids, top_lp: per output token."""
        scored_rows = params[:, 6] > 0.5
        scored = np.repeat(scored_rows, counts)
        tops = np.repeat(np.where(scored_rows, params[:, 5], 0), counts).astype(
            np.uint32
        )
        columns = {
            "logprobs": np.ascontiguousarray(lp[scored], np.float32),
            "logprob_indptr": np.concatenate(
                ([0], np.cumsum(counts * scored_rows))
            ).astype(np.uint32),
        }
        if tops.max(initial=0):
            keep = (np.arange(TOP_MAX)[None, :] < tops[:, None]) & scored[:, None]
            columns["top_indptr"] = np.concatenate(
                ([0], np.cumsum(tops[scored]))
            ).astype(np.uint32)
            columns["top_token_ids"] = np.ascontiguousarray(top_ids[keep], np.uint32)
            columns["top_logprobs"] = np.ascontiguousarray(top_lp[keep], np.float32)
        return columns

    def release(self, sequence_ids: list[int]) -> None:
        """Drop the sequences' carried drafts and rope deltas."""
        for sequence in sequence_ids:
            self._drafts.pop(int(sequence), None)
            self._rope_delta.pop(int(sequence), None)
            self._no_mtp.discard(int(sequence))

    def close(self) -> None:
        self._settle()
        self.model.close()
        self.device.close()
