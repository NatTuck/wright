"""Qwen3.5/3.8 FP8 on Gaudi cards as a Resin managed runner: one card, or a tensor-parallel replica of several.

Speculative decoding: ``mtp`` = K drafts from the checkpoint's MTP layer, or ``drafter`` = a
DFlash 2 drafter directory (``mtp`` = the drafts a step verifies, 1 .. 7, 0 for all 7) or a DSpine
drafter directory (1 .. 15, 0 for all 15). Either way a decode step
verifies [anchor, drafts] of every row in one pass and emits the target's samples.

The Engine owns the paged cache: every plan names each row's KV blocks (128 tokens) and its
STATE blocks (4096 tokens) by (slab, slot) in one arena. A row reads its recurrent state from
the STATE block of its position ``context - 1`` and writes it to the block of its last
position, so a step that crosses a 4096-token boundary leaves the earlier block as the prefix
snapshot. Prefill rows run one at a time; decode rows run as one batch.

Two executors make the steps. By default the native executor (``resin._resin.QwenExecutor``, its step programs
from ``qwen3_5.native``) makes every step: the Engine calls its ``forward``, ``enqueue`` and ``release``. With the
``oldqwen`` feature (``wright -f oldqwen``, ``resin_models_gaudi.features``) the Python executor below does
(``Qwen35Gaudi``'s prefill, decode and verify, on one card). A tensor-parallel replica takes the native executor.

Sampling and logprobs run on the device (``common.sampling``) with Resin's semantics; up to 64 top
logprobs per sampled token. Prompt logprobs (the next prompt token's, up to 1000 top entries
and any requested targets) are scored per prefill chunk (``Qwen35Gaudi.prefill``).
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from resin import BatchPhase, ForwardOutput, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.backend.topology import ParallelismCapabilities
from resin_models_gaudi import features, ranks
from resin_models_gaudi.common.sampling import (
    TOP_MAX,
    Sampling,
    plan_params,
    plan_seeds,
)
from resin_models_gaudi.common.scoring import (
    ScoreRequest,
    prompt_columns,
    score_columns,
    score_requests,
)
from resin_models_gaudi.qwen3_5.cache import (
    DRAFT_GROUP,
    KV_GROUP,
    MTP_GROUP,
    STATE_GROUP,
    STATE_TOKENS,
)
from resin_models_gaudi.qwen3_5.model import (
    DECODE_BUCKETS,
    SPEC_ROWS,
    Qwen35Gaudi,
    RowCache,
    mtp_step_ms,
    plain_ms,
    step_ms,
)
from resin_models_gaudi.qwen3_5.native import build_executor
from resin_models_gaudi.synapse.runtime import open_device

PREFILL_CHUNK_SIZE = 4096
# tokens per Engine prefill step that run as whole launches of the compiled prefill widths
PREFILL_CHUNK_SIZES = (512, 1024, 2048, PREFILL_CHUNK_SIZE)
# speculative decode continuations cross KV blocks (leased in place); a chain ends every this
# many tokens of a row so that the prefix cache can publish (it publishes between chains): a
# multiple of every cache block, so the STATE block
CONTINUATION_PUBLISH_TOKENS = STATE_TOKENS


def lookahead_tokens(draft_tokens: int, drafter: bool) -> int:
    """Cache positions a speculative step writes past its anchor: a verify step writes positions
    p .. p + K (the DFlash drafter's cache entries with them, its block in scratch pages); an MTP
    step's draft chain reaches p + 2K - 1."""
    if not draft_tokens:
        return 0
    return draft_tokens if drafter else 2 * draft_tokens - 1


NEEDS_NATIVE = (
    "a Qwen3.5/3.8 model on more than one card runs on the native executor: remove the oldqwen feature "
    "(wright -f oldqwen)"
)


class DraftPolicy:
    """Whether a decode step of B rows verifies drafts. The policy compares two rates for the
    step's bucket: the tokens per ms of its verify steps (their committed tokens over their time,
    at whatever verify width the drafter chose for each) and of its plain steps (B rows a step).
    A rate is the ratio of two averages over the bucket's measured steps (``record``), its tokens
    and its times: a verify step that commits one token counts at its weight, so a run of such
    steps moves the rate no more than it moves the throughput. The averages take the first steps
    equally and then weight the latest step by BETA (about the last 20 steps). A rate not measured
    yet comes from the fits (plain_ms, mtp_step_ms, step_ms, at the drafter's best width) and
    about 60% accepted drafts. A bucket that verifies drafts keeps them until plain steps are
    MARGIN times faster. After a run of plain decisions in a bucket, PROBE_STEPS steps verify
    drafts again (a probe) and measure the verify rate afresh: the first probe comes after
    PROBE_FIRST plain decisions, and each probe that keeps plain decoding doubles the run, up to
    PROBE_MAX. Steps of more than SPEC_ROWS rows decode plainly."""

    BETA = 0.05  # weight of the latest step in a rate's averages, after the first 1 / BETA steps
    SPIKE = 3.0  # a step this many times longer than its kind's average is not counted (a host stall)
    MARGIN = 1.1  # plain steps must be this many times faster to stop verifying drafts
    PROBE_FIRST, PROBE_MAX = 32, 512  # plain decisions before a probe
    # verify steps of a probe: all but the first and the last count (a switch precedes the first, a
    # plain step follows the last)
    PROBE_STEPS = 8

    MODES = ("auto", "always", "never")

    def __init__(self, K: int, drafter: bool, mode: str = "auto") -> None:
        if mode not in self.MODES:
            raise ValueError(f"draft policy {mode!r}: one of {', '.join(self.MODES)}")
        self.K, self.drafter = K, drafter
        # "always": every step of up to SPEC_ROWS rows verifies drafts (as before this policy);
        # "never": every step decodes plainly
        self.mode = mode
        # each kind of step and bucket: [steps counted, average tokens, average ms]
        self.stats: dict[tuple, list[float]] = {}
        self._drafting: dict[int, bool] = {}  # each bucket's last decision
        self._plain_run: dict[int, int] = {}  # plain decisions in a row, by bucket
        self._probe_after: dict[int, int] = {}
        self._probe = 0  # verify steps left in a probe

    def rate(self, key: tuple, widths: Sequence[int] = ()) -> float:
        """The tokens per ms of ``key`` (("plain", bucket) or ("verify", bucket)): measured, else
        the fit (a verify step's at the best of ``widths``)."""
        if (stats := self.stats.get(key)) is not None:
            return stats[1] / stats[2]
        b = key[1]
        if key[0] == "plain":
            return b / plain_ms(b)
        return max(self.fit(b, w) for w in widths)

    def fit(self, bucket: int, width: int) -> float:
        """A verify step's tokens per ms by the step fits, at about 60% accepted drafts."""
        ms = step_ms(width, bucket) if self.drafter else mtp_step_ms(width, bucket)
        return bucket * min(1.0 + 0.6 * self.K, 3.0, float(width)) / ms

    def speculate(self, rows: int, widths: Sequence[int]) -> bool:
        if not self.K or rows > SPEC_ROWS or self.mode == "never":
            return False
        if self.mode == "always":
            return True
        if self._probe:
            self._probe -= 1
            return True
        b = min(x for x in DECODE_BUCKETS if x >= rows)
        spec = self.rate(("verify", b), widths)
        plain = self.rate(("plain", b))
        drafting = self._drafting.get(b, True)
        decide = spec * self.MARGIN >= plain if drafting else spec > plain
        self._drafting[b] = decide
        if decide:
            if not drafting:  # drafts pay again: probe soon if they stop paying
                self._probe_after[b] = self.PROBE_FIRST
            self._plain_run[b] = 0
            return True
        run = self._plain_run.get(b, 0) + 1
        after = self._probe_after.get(b, self.PROBE_FIRST)
        if run >= after:
            self._plain_run[b] = 0
            self._probe_after[b] = min(2 * after, self.PROBE_MAX)
            self._probe = self.PROBE_STEPS - 1
            self.stats.pop(("verify", b), None)  # the probe measures drafts afresh
            return True
        self._plain_run[b] = run
        return False

    def record(self, key: tuple, tokens: float, ms: float) -> None:
        """A measured step of ``key`` (("plain", bucket) or ("verify", bucket)): the tokens it
        committed and its time. A drafter's verify steps of every width count together, so the
        rate is what its own width choices give (DSpine changes width from step to step)."""
        if ms <= 0:
            return
        stats = self.stats.setdefault(key, [0, 0.0, 0.0])
        if stats[0] and ms > self.SPIKE * stats[2]:
            return
        stats[0] += 1
        w = max(self.BETA, 1.0 / stats[0])
        stats[1] += w * (tokens - stats[1])
        stats[2] += w * (ms - stats[2])


class _NativePolicy:
    """The native executor's draft policy as ``DraftPolicy``'s ``mode`` (tools and tests set it)."""

    def __init__(self, executor) -> None:
        self._executor = executor

    @property
    def mode(self) -> str:
        return self._executor.policy_mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._executor.policy_mode = value

    def speculate(self, rows: int, widths: Sequence[int]) -> bool:
        raise RuntimeError("the native executor decides its steps")


class _Feedback:
    """Exact inputs of a target-only decode continuation (the Engine's continuation plan
    only bounds them): each row's next position; the next tokens stay on the device (the
    previous step's samples)."""

    def __init__(self, positions) -> None:
        self.positions = positions
        self.tokens = lambda: np.zeros(
            0, np.int64
        )  # the next tokens on the host (waits)


class _Verification:
    """An enqueued MTP step, and its continuation's feedback. Its verify pass is left
    running; ``finish`` (from the continuation's ``enqueue``, or from ``result`` when no step
    continues this one) accepts the drafts, commits the accepted tokens and launches the next
    draft chain, whose drafts 2 .. K land in the step's staging. Then ``positions`` and
    ``anchors`` are each row's next anchor position and token."""

    def __init__(self, runner, plan, positions, params, launched) -> None:
        self._runner, self._plan, self._params = runner, plan, params
        self._start, self._launched = positions, launched
        self.committed = 0  # tokens the step committed (after finish)
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
        self.committed = int(np.sum(np.asarray(done.accepted) + 1))
        self.positions = self._start + done.accepted + 1
        self.anchors = np.array([t[-1] for t in done.tokens], np.int64)
        self._first = done.drafts[:, 0]
        self._output = runner._verified_output(
            self._plan, done, self._params, (stage.lp, stage.top_idx, stage.top_lp)
        )

    def drafts(self, K: int) -> np.ndarray:
        """Each row's next drafts (waits for the draft chain or the block step)."""
        B = len(self.positions)
        drafts = np.zeros((B, K), np.int64)
        stage, known = self._launched.stage, self._runner.model.host_drafts
        if known:
            drafts[:, :known] = self._first[:, None]
        if K > known:
            stage.wait(1)
            if known:
                drafts[:, known:] = stage.drafts[known:K, :B].T
            else:
                drafts[:] = stage.next_drafts(B, K)
        return drafts

    def result(self) -> ForwardOutput:
        self.finish()
        if (
            not self.continued
        ):  # no step continues this one: carry its drafts by sequence
            self.continued = True
            drafts = self.drafts(self._runner.K)
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


class Qwen3_5GaudiRunner:
    """``ManagedRunner`` over ``Qwen35Gaudi``; tensor parallel over ``groups``' attention replica."""

    # a dense model: expert tensor parallelism (the Engine's topology splits experts over the same ranks) has nothing
    # to split
    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

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
        drafter: str | Path | None = None,
        defer_cache: bool = False,
        tensor_parallel: tuple[int, int] | None = None,
        groups=None,
        native_ranks=None,
        native: bool | None = None,
        cache_types: tuple[str, str] = ("bf16", "bf16"),
    ) -> None:
        """``defer_cache``: load the weights only; ``attach`` gives the cache arena later (a
        card that several models share). ``groups``: the Engine's parallel groups; their attention
        replica is a tensor-parallel model, one rank a process (``ranks.open_rank`` makes the
        communicator). ``tensor_parallel``: (rank, ranks) without groups, on a device whose
        communicator the caller has made. ``native_ranks``: this process's ``resin._resin.GaudiRanks``, a
        tensor-parallel replica without Gloo: the group makes the communicator and agrees the cache capacity, and
        after the start rank 0's native executor sends each runner call to the other ranks, which ``serve``.
        ``native``: the native executor makes the steps (None: yes, unless the ``oldqwen`` feature is on).
        ``cache_types``: the types of the K and V pages (``cache.KV_TYPES``: bf16 or fp8)."""
        if max_batch_size > DECODE_BUCKETS[-1]:
            raise ValueError(f"max_batch_size exceeds {DECODE_BUCKETS[-1]}")
        self.native = not features.enabled("oldqwen") if native is None else native
        if not self.native and (
            native_ranks is not None or (tensor_parallel is not None and tensor_parallel[1] > 1)
        ):
            raise ValueError(NEEDS_NATIVE)
        self.groups = groups
        self.native_ranks = native_ranks
        if groups is not None:
            self.device, rank, nranks = ranks.open_rank(groups, card, "Qwen3.8")
            tensor_parallel = (rank, nranks)
        else:
            self.device = open_device(card)
        if native_ranks is not None:
            if groups is not None or self.device.glue is None:
                raise ValueError(
                    "native ranks need the native executor and no parallel groups"
                )
            native_ranks.ccl_init(self.device.glue)
            tensor_parallel = (native_ranks.rank, native_ranks.size)
        self.nranks = 1 if tensor_parallel is None else tensor_parallel[1]
        if self.nranks > 1 and not self.native:  # (parallel groups of several ranks)
            raise ValueError(NEEDS_NATIVE)
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
            drafter=None if drafter is None else str(drafter),
            defer_cache=defer_cache,
            tensor_parallel=tensor_parallel,
            reduce_capacity=self._reduce_capacity(),
            native=self.native,
            cache_types=cache_types,
        )
        self._barrier()
        self.mtp = self.model.mtp  # the MTP layer's drafts
        self.K = self.model.K  # drafts a step (the MTP layer's or the DFlash drafter's)
        self.speculative_tokens = self.K
        self.vocab_size = self.model.dims.vocab
        self._speculation = self.model.speculation
        self.lookahead = self.lookahead_tokens(self.K, drafter is not None)
        # RESIN_QWEN_DRAFT_POLICY (DraftPolicy.mode): auto (default: drafts where they measure faster than
        # plain steps), always (every step of up to SPEC_ROWS rows verifies drafts) or never
        mode = os.environ.get("RESIN_QWEN_DRAFT_POLICY", "auto")
        if self.nranks > 1 and native_ranks is None and mode == "auto":
            # each rank would decide from its own step times; the ranks must take the same steps
            warnings.warn(
                "tensor-parallel Qwen3.8 takes the draft policy 'always' (auto needs one decision for every rank)",
                stacklevel=2,
            )
            mode = "always"
        self.policy = DraftPolicy(self.K, drafter is not None, mode)
        # the last enqueued decode step's policy key and enqueue time: the time to the next
        # step's enqueue, when that one continues it, is the step's device time
        # the last enqueued decode step: (policy key, enqueue time, rows, its _Verification or
        # None, whether it switched kind or started a chain); the time to the next enqueue, when
        # that one continues it with a step of the same kind, is its device time
        self._timed: tuple | None = None
        # sequence id -> (the next anchor's position, the anchor token, its drafts)
        self._drafts: dict[int, tuple[int, int, np.ndarray]] = {}
        self._stage = 0  # the next enqueued step's staging set
        self._open: _Verification | None = None  # the last enqueued MTP step
        if not defer_cache:
            self._configure()

    @property
    def own_slab_rows(self) -> int:
        """This model's slab, for planning an arena that other models share."""
        return self.model.geometry.own_slab_rows

    @property
    def scratch_rows(self) -> int:
        """Rows this model needs in a shared arena after the slabs."""
        return 0

    def attach(self, arena) -> None:
        """The shared cache arena (``resin_models_gaudi.arena.SharedArena``) of a deferred runner."""
        self.model.attach(arena=arena)
        self._configure()
        self._barrier()

    def _reduce_capacity(self):
        if self.native_ranks is not None:
            return self.native_ranks.min
        if self.groups is not None:
            return lambda capacity: ranks.min_over_ranks(capacity, self.groups)
        return None

    def _barrier(self) -> None:
        if self.native_ranks is not None:
            self.native_ranks.barrier()
        elif self.groups is not None:
            ranks.barrier(self.groups.replica.handle)

    def serve(self) -> int:
        """A follower rank (``native_ranks`` above rank 0): make the steps that rank 0 sends, until it closes the
        group. Returns the runner calls made."""
        return self.executor.serve()

    def _configure(self) -> None:
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
            draft_tokens=self.K,
            cache_layout=self.cache_layout,
            # a speculative step leases lookahead past its end, so without this every row's
            # approach to a KV block end costs a synchronous step (the STATE group still drains)
            continuation_extension=CONTINUATION_PUBLISH_TOKENS if self.K else None,
            # steps of more rows decode plainly: one token a row, no lookahead (_decode_lookahead)
            max_speculative_rows=SPEC_ROWS if self.K else None,
        )
        self.compiles_at_start = self._compiles_seen = self.device.compile_count
        # without the oldqwen feature the native executor (qwen3_5.native) makes every step: the Engine calls it
        self.executor = None
        if self.native:
            self.executor = build_executor(
                self.model,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                lookahead=self.lookahead,
                policy=self.policy.mode,
            )
            self.forward = self.executor.forward
            self.enqueue = self.executor.enqueue
            self.release = self.executor.release
            self.metrics_snapshot = self.executor.metrics_snapshot  # the Engine's metrics, no Python
            if self.K:
                self.policy = _NativePolicy(self.executor)
            if self.native_ranks is not None:
                self.executor.attach_ranks(self.native_ranks)

    @staticmethod
    def lookahead_tokens(draft_tokens: int, drafter: bool) -> int:
        return lookahead_tokens(draft_tokens, drafter)

    @property
    def force_width(self) -> int | None:
        """A test's verify width for every drafter step after a prompt (None: the width the drafter chooses)."""
        return self.model.force_width if self.executor is None else self.executor.force_width

    @force_width.setter
    def force_width(self, width: int | None) -> None:
        if self.executor is None:
            self.model.force_width = width
        else:
            self.executor.force_width = width

    def carried(self, sequence: int):
        """The drafts carried to ``sequence``'s next speculative step: (its anchor's position, the anchor token,
        the drafts), or None."""
        if self.executor is not None:
            return self.executor.carried(sequence)
        return self._drafts.get(int(sequence))

    def _decode_lookahead(self, rows: int) -> int:
        """positions the Engine leases past each decode row's token: none for steps of more
        than SPEC_ROWS rows (RunnerConfig.max_speculative_rows), which decode plainly"""
        return self.lookahead if rows <= SPEC_ROWS else 0

    def _speculating(self, rows: int) -> bool:
        return self.policy.speculate(rows, self.model.verify_widths)

    @property
    def kv_pool_bytes(self) -> int:
        """The KV pool on the card."""
        return self.model.arena_bytes

    def _native_speculation(self) -> None:
        """The native executor's acceptance counters into the shared SpeculativeStats."""
        if getattr(self, "executor", None) is not None:
            s = self._speculation
            s.blocks, s.drafts, s.accepted, s.emitted = self.executor.speculation

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        self._native_speculation()
        return self._speculation.snapshot(include_draft_tokens=True)

    def configure_metrics(self, *, enabled: bool = True) -> None:
        """Speculation counters always run, so there is nothing to switch on."""

    def metrics_snapshot(self) -> RunnerMetrics:
        self._native_speculation()
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
        draft = cache.groups[DRAFT_GROUP] if self.model.drafter is not None else None
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
                    None
                    if draft is None
                    else g.draft_block_row(draft.block_slabs(r), draft.block_slots(r)),
                    0 if draft is None else int(draft.block_table_start_indices[r]),
                )
            )
        return rows

    @property
    def compiles_after_start(self) -> int:
        return self.device.compile_count - self.compiles_at_start

    def mark_started(self) -> None:
        """Startup ends now: later graph compilations count as compilations after startup (the
        other models of a shared card compile theirs after this one's)."""
        self.compiles_at_start = self._compiles_seen = self.device.compile_count

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

    def _settle(self, feedback=None) -> None:
        """Finish an enqueued MTP step that ``feedback`` does not continue, so its commit
        precedes any other work."""
        open_, self._open = self._open, None
        if open_ is not None and open_ is not feedback:
            open_.result()

    def forward(self, plan) -> ForwardOutput:
        self._check_compiles()
        self._settle()
        if any(media is not None for media in plan.media):
            raise NotImplementedError(
                "this runner has no vision tower; text prompts only"
            )
        decode = plan.phase == BatchPhase.DECODE
        cache = PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            decode_lookahead=self._decode_lookahead(plan.num_sequences)
            if decode
            else 0,
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
        rows = self._rows(cache)
        seeds, offsets = plan_seeds(plan)
        scores = plan.needs_logprobs
        lp = top_ids = top_lp = None
        prompt_scores = []
        if decode and self._speculating(len(rows)):
            return self._speculate(plan, rows, params, Sampling(params, seeds, offsets))
        if batch.phase == BatchPhase.PREFILL:
            tokens, parts = [], []
            requests = self._score_requests(plan) if prompt else [None] * len(rows)
            for r, row in enumerate(rows):
                ids = batch.token_ids[batch.token_slice(r)].tolist()
                sampling = Sampling(
                    params[r : r + 1], seeds[r : r + 1], offsets[r : r + 1]
                )
                done = self.model.prefill(
                    row, ids, sampling, requests[r], draft=bool(plan.sampling_mask[r])
                )
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
            score_columns(params, counts, lp, top_ids, top_lp) if scores else {}
        )
        if prompt:
            columns.update(prompt_columns(plan, prompt_scores))
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
        lookahead = self._decode_lookahead(plan.num_sequences)
        cache = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=lookahead
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
        if int(positions.max()) + 1 + lookahead > self.max_seq_len:
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
        # the time from the last step's enqueue to this one is the last step's device time when
        # this step continues it with a step of the same kind (a switch waits on the host)
        now = time.perf_counter()
        bucket = min(b for b in self.model.decode_buckets if b >= B)
        speculating = self._speculating(B)
        # a drafter's verify steps of every width count as one kind (DraftPolicy.record)
        key: tuple = ("verify", bucket) if speculating else ("plain", bucket)
        last = self._timed
        if feedback is not None and last is not None and last[0] == key and not last[4]:
            tokens = last[2] if last[3] is None else last[3].committed
            self.policy.record(key, tokens, (now - last[1]) * 1e3)
        switched = feedback is None or last is None or last[0][0] != key[0]
        if not speculating:
            if feedback is None:
                tokens = batch.token_ids.tolist()
            elif isinstance(feedback, _Verification):  # its samples are not one a row
                tokens = feedback.anchors.tolist()
            else:
                tokens = None
            self.model.decode(rows, tokens, sampling, stage=stage, scores=scores)
            self._timed = (key, now, B, None, switched)

            def build() -> ForwardOutput:
                counts = np.ones(B, np.int64)
                columns = {}
                if scores:
                    columns = score_columns(
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

            completion = _Completion(stage, build, _Feedback(positions + 1))
            completion.feedback.tokens = lambda: np.asarray(
                completion.result().token_ids, np.int64
            )
            return completion
        if feedback is None:
            anchors = np.asarray(batch.token_ids, np.int64)
            drafts = self._carried(plan, rows, anchors)
        elif isinstance(
            feedback, _Feedback
        ):  # after a plain step: its samples, no drafts
            anchors = feedback.tokens()
            drafts = np.zeros((B, self.K), np.int64)
        else:  # the previous step left these tokens on the device; drafts read at finish
            anchors = feedback.anchors

            def drafts() -> np.ndarray:
                return feedback.drafts(self.K)

        launched = self.model.verify_launch(
            rows,
            anchors,
            drafts,
            sampling,
            stage=stage,
            scores=scores,
            chained=isinstance(feedback, _Verification),
        )
        for sequence in plan.sequence_ids:  # this step carries the drafts now
            self._drafts.pop(int(sequence), None)
        self._open = _Verification(self, plan, positions, params, launched)
        self._timed = (key, now, B, self._open, switched)
        return self._open

    def _carried(self, plan, rows, anchors) -> np.ndarray:
        """Each row's drafts carried from its last step (zeros when they do not continue from
        this anchor, which only costs acceptance)."""
        drafts = np.zeros((len(rows), self.K), np.int64)
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
                scores = self.model.scores(bucket * (done.width or self.K + 1))
            lp, top_ids, top_lp = scores
            picked = np.concatenate(done.columns)
            columns = score_columns(
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

    def release(self, sequence_ids: list[int]) -> None:
        """Drop the sequences' carried drafts."""
        for sequence in sequence_ids:
            self._drafts.pop(int(sequence), None)

    def close(self, *, device: bool = True) -> None:
        """Release the model; ``device=False`` keeps the card open for the other models on it."""
        if self.native_ranks is not None and self.native_ranks.rank == 0:
            try:
                self.executor.close()  # the followers stop serving
            except (OSError, RuntimeError) as error:  # they stopped already
                warnings.warn(f"closing the rank group: {error}", stacklevel=2)
        self._settle()
        self.model.close()
        if device:
            self.device.close()
