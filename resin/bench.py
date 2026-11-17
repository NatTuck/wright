"""Resident-engine benchmarks with explicit synthetic routing/verification modes.

Install BenchmarkRunner before constructing the Engine and capturing graphs.
Each point owns an uncached request cohort; loading, capture, and admission
are untimed. Pipelined decode measures continuous completion intervals, with
metrics reduced only after the window so in-flight GPU work stays timed.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from resin import BatchPhase, BatchPlan, SamplingParams
from resin.backend.speculative import SpeculativeStats


@dataclass(frozen=True, kw_only=True)
class BenchCase:
    phase: Literal["prefill", "decode"]
    context_length: int
    batch_size: int
    shared_prefix_tokens: int = 0

    def __post_init__(self):
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if any(
            type(n) is not int or n < 1 for n in (self.context_length, self.batch_size)
        ):
            raise ValueError("context_length and batch_size must be positive integers")
        if (
            type(self.shared_prefix_tokens) is not int
            or not 0 <= self.shared_prefix_tokens < self.context_length
        ):
            raise ValueError(
                "shared_prefix_tokens must be an integer in [0, context_length)"
            )


def decode_output_budget(
    *,
    warmup_steps: int,
    measure_steps: int,
    draft_tokens: int,
    overlap: bool,
    trace_passes: int = 0,
    cache_write_lookahead_tokens: int | None = None,
) -> int:
    """Reserve output tokens without making a trace drain the decode pipeline.

    The Engine retains a continuation only with more than two maximum write
    steps remaining. Traces need that reservation *after* their last completion,
    in addition to the worst-case accepted output count. Before constructing a
    runner, the default lookahead conservatively covers the extra speculative
    cache row used by GLM and Qwen; callers with a runner use its actual config.
    Untraced requests keep their existing budget.
    """
    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 0),
        ("measure_steps", measure_steps, 1),
        ("draft_tokens", draft_tokens, 0),
        ("trace_passes", trace_passes, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if cache_write_lookahead_tokens is None:
        cache_write_lookahead_tokens = draft_tokens + 1 if draft_tokens else 0
    if (
        type(cache_write_lookahead_tokens) is not int
        or cache_write_lookahead_tokens < 0
    ):
        raise ValueError("cache_write_lookahead_tokens must be an integer >= 0")
    if not trace_passes:
        return 1 + (warmup_steps + measure_steps + 1 + int(overlap)) * (
            draft_tokens + 1
        )
    generated = 1 + (warmup_steps + measure_steps + trace_passes) * (draft_tokens + 1)
    reservation = 2 * (cache_write_lookahead_tokens + 1) + 1 if overlap else 0
    return generated + reservation


class _ObservedCompletion:
    """Attribute a pipelined sample to its harvested plan, not the next launch."""

    def __init__(self, runner, plan, completion):
        self._runner, self._plan, self._completion = runner, plan, completion

    @property
    def feedback(self):
        return self._completion.feedback

    def result(self):
        output = self._completion.result()
        self._runner._observe(self._plan)
        self._runner._completed_positions = getattr(self._completion, "positions", None)
        return output


class BenchmarkRunner:
    """Delegate the runner contract while observing actual scheduled input rows.

    Synthetic options affect the real router or speculative acceptance boundary;
    they must be configured before graph capture and never imply model quality.
    """

    def __init__(
        self, runner, *, accepted_drafts: int | None = None, balanced_experts=False
    ):
        self.runner = runner
        self.draft_tokens = int(getattr(runner, "speculative_tokens", 0))
        if accepted_drafts is not None:
            if (
                type(accepted_drafts) is not int
                or not 0 <= accepted_drafts <= self.draft_tokens
            ):
                raise ValueError(
                    "accepted_drafts must be between zero and draft_tokens"
                )
            stats = getattr(runner, "_speculation", None)
            if self.draft_tokens and not isinstance(stats, SpeculativeStats):
                raise ValueError("fixed acceptance requires a speculative model runner")
            if isinstance(stats, SpeculativeStats):
                stats.benchmark_accepted_drafts = accepted_drafts
        self.accepted_drafts = accepted_drafts
        if type(balanced_experts) is not bool:
            raise ValueError("balanced_experts must be a bool")
        self.balanced_experts = balanced_experts
        if balanced_experts:
            model = getattr(runner, "_model", None)
            routers = (
                []
                if model is None
                else [
                    m
                    for m in model.modules()
                    if hasattr(m, "benchmark_balanced_experts")
                ]
            )
            if not routers:
                raise ValueError(
                    "balanced experts are unsupported by this model runner"
                )
            groups = getattr(runner, "parallel_groups", None)
            for router in routers:
                router.benchmark_balanced_experts = True
                router.benchmark_replica = (
                    0 if groups is None else groups.attn_dp.local_rank
                )
                if groups is not None:
                    router.benchmark_expert_parallel_size = groups.ep.size
        self._plan: BatchPlan | None = None
        self._completed_positions = None

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def _observe(self, plan: BatchPlan | None):
        self._plan = plan
        self._completed_positions = None

    @property
    def completed_context_lengths(self):
        if self._plan is None:
            raise RuntimeError("no benchmark result has been harvested")
        return (
            self._completed_positions
            if self._completed_positions is not None
            else self._plan.context_lengths
        )

    @property
    def observation(self) -> dict[str, Any]:
        plan = self._plan
        return (
            {}
            if plan is None
            else {
                "phase": "prefill" if plan.phase == BatchPhase.PREFILL else "decode",
                "input_tokens": len(plan.token_ids),
                "sequences": plan.num_sequences,
            }
        )

    def graph_counts(self):
        counts = {}
        stats = getattr(self.runner, "cuda_graph_counts", None)
        if stats is None:
            stats = getattr(self.runner, "cuda_graph_stats", {})
        for mode, stats in stats.items():
            for name in ("replay_count", "eager_fallback_count"):
                counts[f"{mode}.{name}"] = stats[name]
            for reason, value in stats["fallback_counts"].items():
                counts[f"{mode}.fallback.{reason}"] = value
        return counts

    def forward(self, plan):
        self._observe(plan)
        return self.runner.forward(plan)

    def forward_distributed(self, plan, context, *, prepared=None):
        self._observe(plan)
        return (
            self.runner.forward_distributed(plan, context)
            if prepared is None
            else self.runner.forward_distributed(plan, context, prepared=prepared)
        )

    def enqueue(self, plan, *, feedback=None):
        enqueue = getattr(self.runner, "enqueue", None)
        if enqueue is None:
            return self.forward(plan)
        output = enqueue(plan, feedback=feedback)
        if callable(getattr(output, "result", None)):
            return _ObservedCompletion(self, plan, output)
        self._observe(plan)
        return output

    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        enqueue = getattr(self.runner, "enqueue_distributed", None)
        if enqueue is None:
            return self.forward_distributed(plan, context, prepared=prepared)
        kwargs = {"feedback": feedback}
        if prepared is not None:
            kwargs["prepared"] = prepared
        output = enqueue(plan, context, **kwargs)
        if callable(getattr(output, "result", None)):
            return _ObservedCompletion(self, plan, output)
        self._observe(plan)
        return output

    def speculation(self):
        # speculative_stats first: a native executor keeps its own counters and syncs them there
        stats = getattr(self.runner, "speculative_stats", None)
        if stats is not None:
            return (stats["drafts"], stats["accepted"])
        stats = getattr(self.runner, "_speculation", None)
        return (0, 0) if stats is None else (stats.drafts, stats.accepted)


def distributed_reducer(groups):
    """Reduce unique attention-replica driver counters after each timed step."""

    drivers = groups.drivers
    if drivers is None:
        raise ValueError("only attention-replica drivers run benchmark sweeps")

    def reduce_step(local):
        values = [local]
        if drivers.size > 1:
            values = [None] * drivers.size
            groups.distributed.all_gather_object(values, local, group=drivers.handle)
        return {
            "seconds": max(v["seconds"] for v in values),
            "replica_sequences": [v["sequences"] for v in values],
            **{
                key: sum(v[key] for v in values)
                for key in (
                    "input_tokens",
                    "committed_tokens",
                    "drafts",
                    "accepted",
                    "sequences",
                )
            },
            "graph_counts": {
                key: sum(v.get("graph_counts", {}).get(key, 0) for v in values)
                for key in {key for v in values for key in v.get("graph_counts", {})}
            },
            "overlap_counts": {
                key: sum(v.get("overlap_counts", {}).get(key, 0) for v in values)
                for key in ("enqueues", "continuations")
            },
            "context_start_min": min(v["context_start_min"] for v in values),
            "context_start_max": max(v["context_start_max"] for v in values),
            "context_end_min": min(v["context_end_min"] for v in values),
            "context_end_max": max(v["context_end_max"] for v in values),
            "device_phase_ms": {
                phase: max(v.get("device_phase_ms", {}).get(phase, 0) for v in values)
                for phase in {
                    key for v in values for key in v.get("device_phase_ms", {})
                }
            },
        }

    return reduce_step


@dataclass(frozen=True, slots=True)
class TextWorkload:
    """Real text with real sampling: a distinct passage per row, and a new passage when a row
    stops. Rows keep their cohort, so a decode window is not cut short by the first stop, and
    the prefill of a new passage stays out of the decode timing."""

    name: str
    prompts: Callable[[int, int], list[int]]
    """(passage index, context length) -> token IDs of at most that length."""
    stop_token_ids: tuple[int, ...]
    output_tokens: int
    """The longest generation per passage, which also sets the context reserve."""
    steps: int
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    @property
    def sampling(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
        }

    def params(self, row: int, max_output_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_output_tokens=max_output_tokens,
            stop_token_ids=list(self.stop_token_ids),
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            seed=None if self.seed is None else self.seed + row,
        )


def host_snapshot() -> dict | None:
    """The host's 1-minute load average and CPU time counters (Linux ``/proc``), or None."""
    try:
        with open("/proc/stat") as stream:
            # user nice system idle iowait irq softirq steal (guest time is in user already)
            values = [int(value) for value in stream.readline().split()[1:9]]
        load = os.getloadavg()[0]
    except (OSError, ValueError):
        return None
    if len(values) < 5:
        return None
    return {
        "load1": load,
        "total": sum(values),
        "idle": values[3] + values[4],
        "time": time.time(),
    }


def stall_fields(seconds: Sequence[float]) -> dict:
    """The steps of a window that took more than 10 times the median step and more than
    0.1 s (host stalls, not slower kinds of steps), and the time above the median that they
    added. The rate keeps them: these fields only find the windows that a stall changed."""
    if not seconds:
        return {"stall_steps": 0, "stall_seconds": 0.0}
    median = statistics.median(seconds)
    stalls = [s - median for s in seconds if s > 10 * median and s > 0.1]
    return {"stall_steps": len(stalls), "stall_seconds": round(sum(stalls), 4)}


def host_fields(before: dict | None, after: dict | None) -> dict:
    """A row's host load, so that rows measured on a busy host can be left out: the
    1-minute load average at the row's start and end, the busy cores between them (CPU
    time other than idle and I/O wait, over all cores), and the start and end times (Unix
    seconds)."""
    if before is None or after is None:
        return {"host": None}
    total = after["total"] - before["total"]
    busy = (
        (1 - (after["idle"] - before["idle"]) / total) * (os.cpu_count() or 1)
        if total > 0
        else None
    )
    return {
        "host": {
            "load1_start": before["load1"],
            "load1_end": after["load1"],
            "busy_cores": None if busy is None else round(busy, 2),
            "cpus": os.cpu_count(),
            "start": round(before["time"], 3),
            "end": round(after["time"], 3),
        }
    }


class _Stepper:
    """One timed Engine step and its counters (``run_sweep`` and ``run_grid``).

    Without overlap a step synchronizes the device before and after. With overlap the
    interval of a step starts at the previous completion when ``defer_reduce`` is set, so
    a continuous window of completions is timed with no gaps, and the caller reduces the
    samples after the window. ``previous_completion = None`` starts a new window.
    """

    def __init__(self, engine, runner: BenchmarkRunner, *, synchronize, reduce_step):
        self.engine, self.runner = engine, runner
        self.local = getattr(engine, "local", engine)
        self.overlap = bool(getattr(self.local, "overlap", False)) and callable(
            getattr(runner.runner, "enqueue", None)
        )
        self.synchronize = synchronize
        self.reduce = reduce_step or (lambda value: value)
        self.previous_completion = None
        self.sample_request = None
        self.sample_tokens: list[int] = []

    def watch(self, lease) -> None:
        """Keep the tokens that ``lease``'s request gets from now on (``sample_tokens``)."""
        self.sample_request, self.sample_tokens = lease.id, []

    def __call__(self, *, defer_reduce=False) -> dict:
        runner, local, synchronize = self.runner, self.local, self.synchronize
        before_drafts, before_accepted = runner.speculation()
        before_graphs = runner.graph_counts()
        before_overlap = getattr(local, "overlap_stats", (0, 0))
        runner._plan = None
        if not self.overlap:
            synchronize()
        started = (
            self.previous_completion
            if defer_reduce and self.previous_completion is not None
            else time.perf_counter()
        )
        # Filling an empty pipeline produces no host result. Include any such
        # submission in the next completion interval, including boundary bubbles.
        while True:
            events = self.engine.step()
            if runner._plan is not None:
                break
            if not getattr(local, "overlap_pending", False):
                raise RuntimeError("Engine must use the supplied BenchmarkRunner")
        if not self.overlap:
            synchronize()
        finished = time.perf_counter()
        elapsed = finished - started
        self.previous_completion = finished
        plan = runner._plan
        if plan is None:
            raise RuntimeError("Engine must use the supplied BenchmarkRunner")
        if self.sample_request is not None:
            self.sample_tokens.extend(
                event.token_id
                for event in events
                if event.request_id == self.sample_request
            )
        drafts, accepted = runner.speculation()
        graph_counts = runner.graph_counts()
        after_overlap = getattr(local, "overlap_stats", (0, 0))
        starts = getattr(
            runner, "completed_context_lengths", plan.context_lengths
        ).tolist()
        if plan.phase == BatchPhase.PREFILL:
            offsets = plan.token_indptr.tolist()
            ends = [start + b - a for start, a, b in zip(starts, offsets, offsets[1:])]
        else:
            counts = Counter(event.request_id for event in events)
            ends = [
                start + counts[int(sequence)]
                for start, sequence in zip(starts, plan.sequence_ids)
            ]
        measured = {
            "seconds": elapsed,
            "input_tokens": runner.observation["input_tokens"],
            "sequences": runner.observation["sequences"],
            "replica_sequences": [runner.observation["sequences"]],
            "committed_tokens": len(events),
            "drafts": drafts - before_drafts,
            "accepted": accepted - before_accepted,
            "context_start_min": min(starts),
            "context_start_max": max(starts),
            "context_end_min": min(ends),
            "context_end_max": max(ends),
            "device_phase_ms": (
                getattr(runner, "last_dspark_phase_ms", None) or {}
                if plan.phase == BatchPhase.DECODE
                else {}
            ),
            "graph_counts": {
                key: value - before_graphs.get(key, 0)
                for key, value in graph_counts.items()
            },
            "overlap_counts": dict(
                zip(
                    ("enqueues", "continuations"),
                    (
                        after - before
                        for before, after in zip(before_overlap, after_overlap)
                    ),
                    strict=True,
                )
            ),
        }
        return measured if defer_reduce else self.reduce(measured)


def _text_window(
    stepper: _Stepper,
    leases: list,
    resubmit: Callable[[int], Any],
    *,
    steps: int,
    warmup_steps: int,
) -> tuple[list[dict], int, int]:
    """Decode a text cohort for ``warmup_steps`` and then ``steps`` measured decode steps.

    A row whose request stops gets ``resubmit(row)``'s request; its prefill runs between
    the decode steps and starts a new completion window, so it stays out of the timing.
    Returns the reduced samples, the warmup steps and the replaced requests.
    """
    local, runner, overlap = stepper.local, stepper.runner, stepper.overlap
    samples: list[dict] = []
    warmed = replaced = 0
    while local.next_phase() == BatchPhase.PREFILL:
        stepper()
    stepper.previous_completion = None
    while len(samples) < steps:
        for row, lease in enumerate(leases):
            if lease.done:
                leases[row] = resubmit(row)
                replaced += 1
        if local.next_phase() == BatchPhase.PREFILL:
            while local.next_phase() == BatchPhase.PREFILL:
                stepper()
            stepper.previous_completion = None
        result = stepper(defer_reduce=overlap)
        if runner.observation["phase"] != "decode":
            stepper.previous_completion = None
        elif warmed < warmup_steps:
            warmed += 1
        else:
            samples.append(result)
    if overlap:
        samples = [stepper.reduce(sample) for sample in samples]
    return samples, warmed, replaced


def run_sweep(
    engine,
    cases: Iterable[BenchCase],
    *,
    runner: BenchmarkRunner,
    prompt_tokens: Sequence[int],
    workloads: Sequence[TextWorkload] = (),
    warmup_steps: int = 5,
    measure_steps: int = 20,
    gpu_count: int = 1,
    attn_dp: int = 1,
    synchronize: Callable[[], None] = lambda: None,
    reduce_step: Callable[[dict], dict] | None = None,
    metadata: dict | None = None,
    on_result: Callable[[dict], None] | None = None,
    trace_step: Callable[[BenchCase, Callable[[], dict]], None] | None = None,
    trace_decode_passes: int | None = None,
) -> list[dict]:
    """Measure prefill cohorts or a steady decode window on an existing Engine.

    batch_size counts requests per attention-DP replica. Prefill finishes whole
    cohorts, so its actual step count can exceed the requested minimum when
    scheduling chunks. Decode records exactly the requested number of steps;
    its context grows naturally during warmup and measurement. All DP drivers
    call together with distributed_reducer; records contain global token counts
    and the slowest driver's completion interval. Overlapped decode defers these
    reductions until the continuous measurement window ends. The caller retains
    engine ownership.
    on_result receives each completed record after request cleanup, outside timing.
    Prefill records ``prefill_seconds`` for each complete measured cohort, so
    chunking does not turn a context's prefill latency into a single chunk's time.
    trace_step receives an untimed workload callback: one fresh complete cohort
    for prefill, or one live completion for decode. It may invoke that callback
    twice, or three times for overlapped decode to allow for a completion
    submitted before tracing. A trace hook with a ``decode_passes`` attribute
    declares its exact decode count; other callbacks use these defaults or
    declare trace_decode_passes. More calls than reserved fail before executing
    extra work. All distributed
    drivers must use the same passes. Trace output headroom preserves the normal
    continuation guard; cache-page boundaries can still interrupt overlap unless
    the runner sets ``continuation_extension``.
    """

    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 0),
        ("measure_steps", measure_steps, 1),
        ("gpu_count", gpu_count, 1),
        ("attn_dp", attn_dp, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if gpu_count % attn_dp:
        raise ValueError("gpu_count must be divisible by attn_dp")
    if attn_dp > 1 and reduce_step is None:
        raise ValueError("attention-DP benchmarks require a step reducer")
    if not len(prompt_tokens):
        raise ValueError("prompt_tokens must not be empty")
    local = getattr(engine, "local", engine)
    overlap = bool(getattr(local, "overlap", False)) and callable(
        getattr(runner.runner, "enqueue", None)
    )
    declared_passes = getattr(trace_step, "decode_passes", None)
    if trace_step is None:
        if trace_decode_passes is not None:
            raise ValueError("trace_decode_passes requires trace_step")
        trace_passes = 0
    else:
        if (
            trace_decode_passes is not None
            and declared_passes is not None
            and trace_decode_passes != declared_passes
        ):
            raise ValueError("trace_decode_passes disagrees with the trace callback")
        trace_passes = (
            trace_decode_passes
            if trace_decode_passes is not None
            else declared_passes
            if declared_passes is not None
            else 2 + int(overlap)
        )
        if type(trace_passes) is not int or trace_passes < 1:
            raise ValueError("trace_decode_passes must be a positive integer")
    if local.has_work():
        raise ValueError("benchmark engine must not have active requests")
    stepper = _Stepper(engine, runner, synchronize=synchronize, reduce_step=reduce_step)
    reduce, step = stepper.reduce, stepper
    records = []

    pairs = [(case, workload) for case in cases for workload in (workloads or (None,))]
    for case, workload in pairs:
        if case.batch_size > local.max_running_requests:
            raise ValueError("batch_size exceeds the engine request capacity")
        if case.context_length >= local.context_length:
            raise ValueError("context_length leaves no space for generated tokens")
        tokens = (
            list(prompt_tokens)
            * ((case.context_length + len(prompt_tokens) - 1) // len(prompt_tokens))
        )[: case.context_length]
        leases = []
        samples = []
        prefill_seconds = []
        warmed = 0
        result = None
        nonce = 0
        cached_tokens = []
        replaced = 0
        shared = case.shared_prefix_tokens

        def submit_passage(max_output_tokens):
            nonlocal nonce
            assert workload is not None
            row = nonce
            nonce += 1
            lease = engine.submit(
                workload.prompts(row, case.context_length),
                sampling_params=workload.params(row, max_output_tokens),
            )
            cached_tokens.append(lease.cached_tokens)
            return lease

        def admit(max_output_tokens):
            nonlocal nonce
            if not shared:
                local.clear_prefix_cache()
            leases.clear()
            if workload is not None:
                nonce = 0
                leases.extend(
                    submit_passage(max_output_tokens) for _ in range(case.batch_size)
                )
                return
            for _ in range(case.batch_size):
                request_tokens = tokens
                if shared:
                    if nonce >= runner.vocab_size:
                        raise ValueError(
                            "shared-prefix sweep exhausted distinct private tokens"
                        )
                    request_tokens = tokens.copy()
                    request_tokens[shared] = nonce
                    nonce += 1
                lease = engine.submit(
                    request_tokens,
                    sampling_params=SamplingParams(
                        max_output_tokens=max_output_tokens,
                        stop_token_ids=[],
                        temperature=0.0,
                    ),
                )
                leases.append(lease)
                cached_tokens.append(lease.cached_tokens)
                if lease.cached_tokens != shared:
                    raise ValueError(
                        f"shared prefix reuse differs from requested checkpoint: "
                        f"requested={shared}, cached={lease.cached_tokens}"
                    )

        host_before = host_snapshot()
        try:
            local.clear_prefix_cache()
            if shared:
                if local.prefix_cache_stats is None:
                    raise ValueError(
                        "shared-prefix benchmarks require Engine(prefix_cache=True)"
                    )
                leases.append(
                    engine.submit(
                        tokens[:shared],
                        sampling_params=SamplingParams(
                            max_output_tokens=1, stop_token_ids=[], temperature=0.0
                        ),
                    )
                )
                while not leases[0].done:
                    step()  # Seed the one shared prefix outside warmup/measurement.
            if case.phase == "prefill":
                for target, measured in ((warmup_steps, False), (measure_steps, True)):
                    count = 0
                    while count < target:
                        admit(1)
                        cohort_seconds = 0.0
                        while not all(lease.done for lease in leases):
                            result = step()
                            if runner.observation["phase"] != "prefill":
                                raise RuntimeError("prefill benchmark entered decode")
                            count += 1
                            if measured:
                                samples.append(result)
                                cohort_seconds += result["seconds"]
                        if measured:
                            prefill_seconds.append(cohort_seconds)
                        if not measured:
                            warmed = count
            elif workload is not None:
                if case.context_length + workload.output_tokens > local.context_length:
                    raise ValueError("engine context cannot fit the text output")
                admit(workload.output_tokens)
                stepper.watch(leases[0])
                samples, warmed, replaced = _text_window(
                    stepper,
                    leases,
                    lambda row: submit_passage(workload.output_tokens),
                    steps=workload.steps,
                    warmup_steps=warmup_steps,
                )
            else:
                maximum = decode_output_budget(
                    warmup_steps=warmup_steps,
                    measure_steps=measure_steps,
                    draft_tokens=runner.draft_tokens,
                    overlap=overlap,
                    trace_passes=trace_passes,
                    cache_write_lookahead_tokens=getattr(
                        getattr(runner, "runner_config", None),
                        "cache_write_lookahead_tokens",
                        0,
                    ),
                )
                if case.context_length + maximum > local.context_length:
                    raise ValueError("engine context cannot fit the full decode window")
                admit(maximum)
                while local.next_phase() == BatchPhase.PREFILL:
                    step()
                for index in range(warmup_steps + measure_steps):
                    measuring = index >= warmup_steps
                    if overlap and measuring and warmup_steps == 0 and index == 0:
                        stepper.previous_completion = None
                    result = step(defer_reduce=overlap)
                    if runner.observation["phase"] != "decode":
                        raise RuntimeError("decode benchmark did not execute decode")
                    expected = case.batch_size * (1 if overlap else attn_dp)
                    if result["sequences"] != expected:
                        raise RuntimeError(
                            "decode batch was not fully scheduled: "
                            f"scheduled_global={result['sequences']}, "
                            f"expected_global={case.batch_size * attn_dp}, "
                            f"scheduled_local={runner.observation['sequences']}, "
                            f"expected_per_replica={case.batch_size}, "
                            f"context_length={case.context_length}, phase={case.phase}"
                        )
                    if any(lease.done for lease in leases):
                        raise RuntimeError("decode cohort retired during measurement")
                    if measuring:
                        samples.append(result)
                if overlap:
                    # A per-step reduction or CUDA synchronization lets future
                    # GPU work run outside the timer. Reduce only after the
                    # continuous completion window has finished.
                    samples = [reduce(sample) for sample in samples]
                warmed = warmup_steps
            seconds = [sample["seconds"] for sample in samples]
            inputs = (
                sum(s["input_tokens"] for s in samples)
                if case.phase == "prefill"
                else 0
            )
            drafts = sum(s["drafts"] for s in samples)
            records.append(
                {
                    "schema_version": 1,
                    "phase": case.phase,
                    "context_length": case.context_length,
                    "shared_prefix_tokens": shared,
                    "private_prompt_tokens": case.context_length - shared,
                    "cached_tokens_min": min(cached_tokens),
                    "cached_tokens_max": max(cached_tokens),
                    "batch_size": case.batch_size,
                    "global_batch_size": case.batch_size * attn_dp,
                    "measured_global_batch_size_min": min(
                        s["sequences"] for s in samples
                    ),
                    "measured_global_batch_size_max": max(
                        s["sequences"] for s in samples
                    ),
                    "gpu_count": gpu_count,
                    "attn_dp": attn_dp,
                    "prefill_tokens": inputs / len(samples),
                    "input_tokens": inputs,
                    "draft_tokens": runner.draft_tokens,
                    "accepted_drafts": runner.accepted_drafts,
                    "measured_accept_rate": sum(s["accepted"] for s in samples) / drafts
                    if drafts
                    else None,
                    "balanced_experts": runner.balanced_experts,
                    "balanced_experts_scope": "physical_routing_rows"
                    if runner.balanced_experts
                    else None,
                    "warmup_steps": warmed,
                    "measure_steps": len(samples),
                    "step_seconds": seconds,
                    **({"prefill_seconds": prefill_seconds} if prefill_seconds else {}),
                    **(
                        {
                            "device_phase_ms": [
                                sample["device_phase_ms"] for sample in samples
                            ]
                        }
                        if any(sample.get("device_phase_ms") for sample in samples)
                        else {}
                    ),
                    "window_seconds": sum(seconds),
                    **stall_fields(seconds),
                    "committed_tokens": sum(s["committed_tokens"] for s in samples),
                    "measured_context_start_min": (
                        min(s["context_start_min"] for s in samples)
                        if workload is not None
                        else samples[0]["context_start_min"]
                    ),
                    "measured_context_start_max": samples[0]["context_start_max"],
                    "measured_context_end_min": samples[-1]["context_end_min"],
                    "measured_context_end_max": (
                        max(s["context_end_max"] for s in samples)
                        if workload is not None
                        else samples[-1]["context_end_max"]
                    ),
                    **(
                        {
                            "workload": workload.name,
                            "sampling": workload.sampling,
                            "replaced_requests": replaced,
                            "sample_tokens": stepper.sample_tokens[:256],
                        }
                        if workload is not None
                        else {}
                    ),
                    **(
                        {"compiles_after_start": runner.runner.compiles_after_start}
                        if hasattr(runner.runner, "compiles_after_start")
                        else {}
                    ),
                    "graph_counts": {
                        key: sum(s["graph_counts"].get(key, 0) for s in samples)
                        for key in {key for s in samples for key in s["graph_counts"]}
                    },
                    "overlap_counts": {
                        key: sum(
                            s.get("overlap_counts", {}).get(key, 0) for s in samples
                        )
                        for key in ("enqueues", "continuations")
                    },
                    **host_fields(host_before, host_snapshot()),
                    "metadata": dict(metadata or {}),
                }
            )
            if trace_step is not None:
                synchronize()
                stepper.previous_completion = None
                traced_completions = 0

                def trace_pass():
                    nonlocal traced_completions
                    if case.phase == "decode":
                        if traced_completions >= trace_passes:
                            raise ValueError(
                                "trace exceeded reserved decode completions"
                            )
                        traced_completions += 1
                        return step(defer_reduce=True)
                    # A complete cohort exercises all chunks of this context.
                    # Admission clears private prefix state before each pass.
                    admit(1)
                    traced = []
                    while not all(lease.done for lease in leases):
                        traced.append(step(defer_reduce=True))
                        if runner.observation["phase"] != "prefill":
                            raise RuntimeError("prefill trace entered decode")
                    return {
                        "seconds": sum(item["seconds"] for item in traced),
                        "input_tokens": sum(item["input_tokens"] for item in traced),
                        "sequences": case.batch_size,
                        "steps": len(traced),
                    }

                trace_step(case, trace_pass)
        except Exception as error:
            diagnostic = dict(
                **asdict(case),
                error=f"{type(error).__name__}: {error}",
                attn_dp=attn_dp,
                actual_global=(result or {}).get("sequences"),
                expected_global=case.batch_size * attn_dp,
                observation=runner.observation,
                per_replica=(result or {}).get("replica_sequences"),
                cache={
                    name: getattr(local, name, None)
                    for name in (
                        "ready_request_count",
                        "active_request_count",
                        "new_token_ratio",
                        "cache_parent_capacity",
                        "cache_free_parent_count",
                        "cache_occupied_block_count",
                    )
                },
            )
            print("[bench-error] " + json.dumps(diagnostic), flush=True)
            raise
        finally:
            if not local.poisoned:
                for lease in leases:
                    if not lease.done:
                        engine.cancel(lease)
                local.clear_prefix_cache()
        if on_result is not None:
            on_result(records[-1])
    return records


# ---------------------------------------------------------------------------- the cross grid


@dataclass(frozen=True, kw_only=True)
class CrossGrid:
    """The cross grid of decode cells: rows (each row context at each row stream count) and
    columns (each column stream count at each column context). One set of sequences grows
    from context to context; each cell decodes a subset of the sequences at its context."""

    row_contexts: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536)
    row_streams: tuple[int, ...] = (1, 8)
    column_contexts: tuple[int, ...] = (4096, 65536)
    column_streams: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

    def __post_init__(self):
        for name in (
            "row_contexts",
            "row_streams",
            "column_contexts",
            "column_streams",
        ):
            values = getattr(self, name)
            if any(type(value) is not int or value < 1 for value in values):
                raise ValueError(f"{name} must be positive integers")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        if not self.row_contexts and not self.column_contexts:
            raise ValueError("the grid needs at least one context")
        if self.row_contexts and not self.row_streams:
            raise ValueError("row contexts need row streams")
        if self.column_contexts and not self.column_streams:
            raise ValueError("column contexts need column streams")

    def cells(self) -> dict[int, tuple[int, ...]]:
        """Each context's stream counts: contexts ascending, stream counts descending."""
        streams: dict[int, set[int]] = {}
        for context in self.row_contexts:
            streams.setdefault(context, set()).update(self.row_streams)
        for context in self.column_contexts:
            streams.setdefault(context, set()).update(self.column_streams)
        return {
            context: tuple(sorted(streams[context], reverse=True))
            for context in sorted(streams)
        }

    @property
    def growth(self) -> int:
        """The sequences that grow from context to context: the widest row (one without rows)."""
        return max(self.row_streams) if self.row_contexts else 1

    @property
    def streams(self) -> int:
        return max(n for counts in self.cells().values() for n in counts)


PREFILL_PROBES = 3
"""Fresh prompts that a grid prefill row times when it cannot use the growth chunks."""


def grid_rows(grid: CrossGrid) -> int:
    """The prompt rows that ``run_grid`` can use: the widest cell's, the fresh prefill rows of
    each context, and one more for the warm-up prefill."""
    return grid.streams + PREFILL_PROBES * len(grid.cells()) + 1


@dataclass(frozen=True)
class GridText:
    """A text workload's prompts on the grid. Row ``r``'s prompt at context ``c`` is the first
    ``c`` tokens of the row's prompt stream (its rendered prompt, cut in its passage) and then
    ``tail``, the template's closing tokens (the assistant turn). So the prompts of one row at
    two contexts share the first ``c`` tokens, and the prefix cache keeps them."""

    workload: TextWorkload
    stream: Callable[[int, int], Sequence[int]]
    """(row, length) -> the first ``length`` tokens of the row's prompt stream; rows differ."""
    tail: tuple[int, ...]

    def prompt(self, row: int, context: int) -> np.ndarray:
        tokens = np.asarray(self.stream(row, context), dtype=np.uint32)
        if len(tokens) != context:
            raise ValueError(f"row {row} has no stream of {context} tokens")
        return np.concatenate([tokens, np.asarray(self.tail, dtype=np.uint32)])


@dataclass(frozen=True)
class DraftConfig:
    """A draft configuration that a grid cell is measured with."""

    name: str
    """For example ``none``, ``mtp2``, ``dflash7``, ``dspine15``, ``dspark3``."""
    draft_tokens: int
    rows: int | None = None
    """The most streams whose decode steps draft (None: the runner decides). A wider cell
    decodes plainly, so a plain configuration's row stands for this one there."""

    def drafts(self, streams: int) -> bool:
        return self.draft_tokens > 0 and (self.rows is None or streams <= self.rows)


@dataclass(frozen=True)
class Chunking:
    """How the Engine cuts a prompt into prefill chunks (``planning.rs``), for one request alone
    with the prefix cache on."""

    chunk: int
    budget: int | None = None
    """The tokens a prefill step takes (``max_batch_tokens``)."""
    alignment: int = 1
    """The cache checkpoints: a multiple of every cache group's block tokens."""
    replay: int = 0
    """The final prompt tokens that a prefix hit always computes again."""

    @staticmethod
    def of(runner, chunk: int, budget: int | None = None) -> Chunking:
        layout = getattr(runner, "cache_layout", None)
        if layout is None:
            return Chunking(chunk, budget)
        return Chunking(
            chunk,
            budget,
            math.lcm(*(group.block_tokens for group in layout.groups)),
            layout.replay_suffix_tokens,
        )

    def fresh(self, length: int) -> list[tuple[int, int]]:
        """The chunks of a prompt of ``length`` tokens that the prefix cache does not hold."""
        chunks, start = [], 0
        while start < length:
            count = min(length - start, self.chunk, self.budget or self.chunk)
            if self.alignment > 1:
                proposed = start + count
                restorable = max(0, length - max(self.replay, 1))
                end = min(proposed, restorable) // self.alignment * self.alignment
                if start < end < proposed:
                    count = end - start
            chunks.append((start, start + count))
            start += count
        return chunks


def set_drafting(engine, runner, on: bool, rows: int | None) -> None:
    """Switch a runner's drafts at run time, between steps (``DeepSeekV41GaudiRunner.drafting``):
    the Engine leases write lookahead only for decode steps of up to ``rows`` rows while the
    drafts are on, and for none while they are off. ``rows`` is the runner's
    ``runner_config.max_speculative_rows``."""
    local = getattr(engine, "local", engine)
    before = local.max_speculative_rows
    local.max_speculative_rows = rows if on else 0
    try:
        getattr(runner, "runner", runner).drafting = on
    except BaseException:
        local.max_speculative_rows = before
        raise


def run_grid(
    engine,
    grid: CrossGrid,
    texts: Sequence[GridText],
    *,
    runner: BenchmarkRunner,
    chunking: Chunking,
    configs: Sequence[DraftConfig],
    switch: Callable[[DraftConfig], None] | None = None,
    fits: Callable[[int, int], bool] = lambda context, streams: True,
    reuse: bool = True,
    phases: Sequence[str] = ("prefill", "decode"),
    warmup_steps: int = 5,
    gpu_count: int = 1,
    synchronize: Callable[[], None] = lambda: None,
    reduce_step: Callable[[dict], dict] | None = None,
    metadata: dict | None = None,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Measure the cross grid on an Engine with the prefix cache on, for each text.

    With ``reuse``, the first ``grid.growth`` rows grow: at each context they are prefilled
    one at a time, and each prefill computes only the tokens that the prefix cache does not
    hold from the row's previous context. Each cell then decodes rows ``0 .. n - 1``: their
    prompts again, which the prefix cache holds up to the closing tokens. The other rows stay
    in the prefix cache, idle. Rows above the growth set are prefilled when a cell first needs
    them. Without ``reuse``, every cell and every prefill row starts from an empty prefix
    cache (for checks of the method).

    A decode cell measures as ``run_sweep``'s text workloads do: ``warmup_steps``, then the
    workload's steps; decode = committed tokens / window seconds. A row whose request stops
    decodes its prompt again with the next seed.

    The prefill rows are one prompt each. A growth row's chunks of its prompt (its chunks
    over its contexts, less the chunks that a later prompt computes again) are the chunks of
    a fresh prefill of the prompt when they tile it as ``chunking.fresh`` does. Then each
    chunk takes its median time over the growth rows, and the row is their sum. Otherwise
    ``PREFILL_PROBES`` prompts that the cache does not hold are prefilled to measure it.

    Each cell runs once for each configuration that drafts at its stream count (``switch``
    sets it), and once for the first one that does not, which then also stands for the
    others. ``fits(context, streams)``: cells that do not fit the cache are skipped. One
    attention replica only.
    """
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be an integer >= 0")
    if not configs:
        raise ValueError("the grid needs at least one draft configuration")
    if len(configs) > 1 and switch is None:
        raise ValueError("several draft configurations need a switch")
    if not set(phases) <= {"prefill", "decode"} or not phases:
        raise ValueError("phases must be prefill, decode, or both")
    stepper = _Stepper(engine, runner, synchronize=synchronize, reduce_step=reduce_step)
    local = stepper.local
    if local.has_work():
        raise ValueError("benchmark engine must not have active requests")
    if local.prefix_cache_stats is None:
        raise ValueError("the grid needs Engine(prefix_cache=True)")
    cells = {}
    for context, streams in grid.cells().items():
        fitting = [
            n for n in streams if n <= local.max_running_requests and fits(context, n)
        ]
        skipped = [n for n in streams if n not in fitting]
        if skipped:
            print(
                f"[bench-grid] skipped (cache or requests): context {context} streams {skipped}",
                flush=True,
            )
        if fitting:
            cells[context] = fitting
    records: list[dict] = []
    current: list[DraftConfig | None] = [None]
    plain = next((c for c in configs if c.draft_tokens == 0), None)

    def use(config: DraftConfig) -> None:
        if current[0] is not config and switch is not None and len(configs) > 1:
            switch(config)
        current[0] = config

    def emit(record: dict) -> None:
        records.append(record)
        if on_result is not None:
            on_result(record)

    def prefill(text: GridText, row: int, context: int):
        """One prompt alone to its end: (cached tokens, [(start, end, seconds)] per step)."""
        lease = engine.submit(
            text.prompt(row, context),
            sampling_params=SamplingParams(
                max_output_tokens=1, stop_token_ids=[], temperature=0.0
            ),
        )
        steps = []
        try:
            while not lease.done:
                result = stepper()
                if runner.observation["phase"] != "prefill":
                    raise RuntimeError("a grid prefill entered decode")
                steps.append(
                    (
                        result["context_start_min"],
                        result["context_end_max"],
                        result["seconds"],
                    )
                )
        finally:
            if not lease.done and not local.poisoned:
                engine.cancel(lease)
        return lease.cached_tokens, steps

    common = {
        "schema_version": 1,
        "grid": "cross",
        "reuse": reuse,
        "shared_prefix_tokens": 0,
        "gpu_count": gpu_count,
        "attn_dp": 1,
        "accepted_drafts": None,
        "balanced_experts": False,
        "balanced_experts_scope": None,
    }

    def prefill_record(text, context, samples, method, host_before) -> dict:
        """One request's prefill: ``samples`` are the chunk lists of prompts of this context
        (the same chunks); each chunk takes its median time over the samples."""
        steps = samples[0]
        seconds = [
            statistics.median(sample[i][2] for sample in samples)
            for i in range(len(steps))
        ]
        length = context + len(text.tail)
        return {
            **common,
            "phase": "prefill",
            "context_length": context,
            "prompt_tokens": length,
            "private_prompt_tokens": length,
            "cached_tokens_min": 0,
            "cached_tokens_max": 0,
            "batch_size": 1,
            "global_batch_size": 1,
            "measured_global_batch_size_min": 1,
            "measured_global_batch_size_max": 1,
            "prefill_tokens": length / len(steps),
            "input_tokens": length,
            "draft_tokens": configs[0].draft_tokens,
            "draft_config": configs[0].name,
            "configs": [c.name for c in configs],
            "measured_accept_rate": None,
            "prefill_method": method,
            "prefill_chunks": [[start, end] for start, end, _ in steps],
            "prefill_samples": [sum(c[2] for c in sample) for sample in samples],
            "warmup_steps": 0,
            "measure_steps": len(steps),
            "step_seconds": seconds,
            "prefill_seconds": [sum(seconds)],
            "window_seconds": sum(seconds),
            "committed_tokens": 1,
            "measured_context_start_min": 0,
            "measured_context_start_max": 0,
            "measured_context_end_min": length,
            "measured_context_end_max": length,
            "workload": text.workload.name,
            "sampling": text.workload.sampling,
            "replaced_requests": 0,
            **(
                {"compiles_after_start": runner.runner.compiles_after_start}
                if hasattr(runner.runner, "compiles_after_start")
                else {}
            ),
            "graph_counts": {},
            "overlap_counts": {"enqueues": 0, "continuations": 0},
            **host_fields(host_before, host_snapshot()),
            "metadata": dict(metadata or {}),
        }

    def measure(text, context, streams, config, *, stands_for, resident, grown):
        workload = text.workload
        if context + len(text.tail) + workload.output_tokens > local.context_length:
            raise ValueError("engine context cannot fit the text output")
        use(config)
        host_before = host_snapshot()
        nonce = 0

        def submit(row: int):
            nonlocal nonce
            lease = engine.submit(
                text.prompt(row, context),
                sampling_params=workload.params(nonce, workload.output_tokens),
            )
            nonce += 1
            return lease

        leases: list = []
        try:
            for row in range(streams):
                leases.append(submit(row))
            cached = [lease.cached_tokens for lease in leases]
            stepper.watch(leases[0])
            samples, warmed, replaced = _text_window(
                stepper, leases, submit, steps=workload.steps, warmup_steps=warmup_steps
            )
        except Exception as error:
            print(
                "[bench-error] "
                + json.dumps(
                    dict(
                        phase="decode",
                        context_length=context,
                        batch_size=streams,
                        draft_config=config.name,
                        workload=workload.name,
                        error=f"{type(error).__name__}: {error}",
                        observation=runner.observation,
                    )
                ),
                flush=True,
            )
            raise
        finally:
            if not local.poisoned:
                for lease in leases:
                    if not lease.done:
                        engine.cancel(lease)
        seconds = [sample["seconds"] for sample in samples]
        drafts = sum(s["drafts"] for s in samples)
        names = [config.name, *(c.name for c in stands_for)]
        return {
            **common,
            "phase": "decode",
            "context_length": context,
            "prompt_tokens": context + len(text.tail),
            "private_prompt_tokens": context + len(text.tail),
            "cached_tokens_min": min(cached),
            "cached_tokens_max": max(cached),
            "batch_size": streams,
            "global_batch_size": streams,
            "measured_global_batch_size_min": min(s["sequences"] for s in samples),
            "measured_global_batch_size_max": max(s["sequences"] for s in samples),
            "prefill_tokens": 0.0,
            "input_tokens": 0,
            "draft_tokens": config.draft_tokens,
            "draft_config": config.name,
            "configs": names,
            "drafting": config.drafts(streams),
            "measured_accept_rate": sum(s["accepted"] for s in samples) / drafts
            if drafts
            else None,
            "grown": grown == streams,
            "grown_rows": grown,
            "resident": resident,
            "subset": streams < resident,
            "warmup_steps": warmed,
            "measure_steps": len(samples),
            "step_seconds": seconds,
            **(
                {"device_phase_ms": [sample["device_phase_ms"] for sample in samples]}
                if any(sample.get("device_phase_ms") for sample in samples)
                else {}
            ),
            "window_seconds": sum(seconds),
            **stall_fields(seconds),
            "committed_tokens": sum(s["committed_tokens"] for s in samples),
            "measured_context_start_min": min(s["context_start_min"] for s in samples),
            "measured_context_start_max": samples[0]["context_start_max"],
            "measured_context_end_min": samples[-1]["context_end_min"],
            "measured_context_end_max": max(s["context_end_max"] for s in samples),
            "workload": workload.name,
            "sampling": workload.sampling,
            "replaced_requests": replaced,
            "sample_tokens": stepper.sample_tokens[:256],
            **(
                {"compiles_after_start": runner.runner.compiles_after_start}
                if hasattr(runner.runner, "compiles_after_start")
                else {}
            ),
            "graph_counts": {
                key: sum(s["graph_counts"].get(key, 0) for s in samples)
                for key in {key for s in samples for key in s["graph_counts"]}
            },
            "overlap_counts": {
                key: sum(s.get("overlap_counts", {}).get(key, 0) for s in samples)
                for key in ("enqueues", "continuations")
            },
            **host_fields(host_before, host_snapshot()),
            "metadata": dict(metadata or {}),
        }

    def plan_configs(streams: int):
        """(configuration, the configurations its row stands for) for each run of a cell"""
        drafting = [c for c in configs if c.drafts(streams)]
        rest = [c for c in configs if c not in drafting]
        if not rest:
            return [(c, []) for c in drafting]
        first = plain if plain in rest else rest[0]
        return [(first, [c for c in rest if c is not first])] + [
            (c, []) for c in drafting
        ]

    if texts and cells:
        # untimed: the first prefill after the load
        prefill(texts[0], grid_rows(grid) - 1, next(iter(cells)))
    for text in texts:
        local.clear_prefix_cache()
        probe_rows = iter(
            range(grid.streams, grid_rows(grid) - 1)
        )  # fresh prefill rows
        level: dict[int, int] = {}  # row -> the context it was last prefilled to
        grown: dict[
            int, bool
        ] = {}  # row -> its KV at that context extended a shorter one
        # each growth row's chunks of its current prompt: (start, end, seconds)
        chunks: dict[int, list[tuple[int, int, float]]] = {}

        def grow(row: int, context: int) -> None:
            if level.get(row) == context:
                return
            restored, steps = prefill(text, row, context)
            level[row], grown[row] = context, restored > 0
            kept = [c for c in chunks.get(row, []) if c[1] <= restored]
            chunks[row] = kept + steps

        for context, streams in cells.items():
            length = context + len(text.tail)
            host_before = host_snapshot()
            if reuse:
                # the growth set first (row 0 also without decode), one row at a time
                for row in range(max(1, min(grid.growth, max(streams)))):
                    grow(row, context)
            if "prefill" in phases:
                fresh = chunking.fresh(length)
                if reuse and [(s, e) for s, e, _ in chunks[0]] == fresh:
                    # each chunk's median over the growth rows that computed it
                    same = [
                        chunks[row]
                        for row in sorted(chunks)
                        if level.get(row) == context
                        and [(s, e) for s, e, _ in chunks[row]] == fresh
                    ]
                    emit(prefill_record(text, context, same, "growth", host_before))
                else:
                    probes = []
                    for _ in range(PREFILL_PROBES):
                        if not reuse:
                            local.clear_prefix_cache()
                        restored, steps = prefill(text, next(probe_rows), context)
                        if restored:
                            raise RuntimeError(
                                "a fresh prefill row found cached tokens"
                            )
                        if [(s, e) for s, e, _ in steps] != fresh:
                            print(
                                f"[bench-grid] fresh chunks {[(s, e) for s, e, _ in steps]} "
                                f"differ from the expected {fresh}",
                                flush=True,
                            )
                        probes.append(steps)
                    emit(prefill_record(text, context, probes, "fresh", host_before))
            if "decode" not in phases:
                continue
            resident = max(streams)
            for n in streams:
                if reuse:
                    for row in range(n):
                        grow(row, context)
                for config, stands_for in plan_configs(n):
                    if not reuse:
                        local.clear_prefix_cache()
                    emit(
                        measure(
                            text,
                            context,
                            n,
                            config,
                            stands_for=stands_for,
                            resident=resident if reuse else n,
                            grown=sum(grown.get(r, False) for r in range(n))
                            if reuse
                            else 0,
                        )
                    )
    return records
