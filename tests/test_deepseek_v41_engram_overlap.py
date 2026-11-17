"""DeepSeek-V4.1's Engram context under the Engine's overlap, on CPU: the tokens before each position when the
Engine takes a plain decode step's result and then starts the row's next step afresh, with host tokens.

A plain decode step at position p records its sample at p + 1 when the Engine takes its result: that is the input
of the step that continues it on the device, whose ``before(p + 1)`` runs first. When the Engine breaks the chain
(a request arrives or finishes, a prefix is published, or a row nears its output budget), the next step starts at
p + 1 with that sample as its host token, after the sample is recorded: its ``before(p + 1)`` must not return the
sample, and its ``seen(p + 1, [sample])`` must replace it, not add it again. The WINDOW block tails recorded then
must be right too, for a later prefix hit. Each test runs the native context (``resin._resin.Ds41Context``) and
the numpy one (``ds41_reference``)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ds41_reference import EngramContext
from output_helpers import forward_output

from resin import BatchPhase, Engine, RunnerConfig, SamplingParams, SchedulerConfig
from resin.backend.plans import PagedCachePlan
from resin_models_gaudi.deepseek_v41.cache import (
    PAGE_TOKENS,
    WINDOW_GROUP,
    CacheGeometry,
)

N = 3
FAMILY = 10_000
CHUNK = 256
PUBLISH = 2048  # the runner's continuation_extension (CONTINUATION_PUBLISH_TOKENS)
# two KV owners of ratio 1 make a HISTORY block 768 rows, so three WINDOW blocks share a slab
GEOMETRY = CacheGeometry(2, (0, 1), (1, 1), 128, CHUNK)
CONTEXTS = [EngramContext]
try:
    from resin._resin import Ds41Context

    CONTEXTS.append(Ds41Context)
except ImportError:  # resin built without the gaudi feature
    pass


@pytest.fixture(params=CONTEXTS, ids=lambda c: c.__name__)
def context_class(request):
    return request.param


def tokens(family: int, start: int, end: int) -> list[int]:
    """The tokens of a family at positions start .. end - 1: every family is one long sequence."""
    return [family * FAMILY + p for p in range(start, end)]


class Feedback:
    def __init__(self, positions: list[int]) -> None:
        self.positions = positions


class Completion:
    """An enqueued plain decode step: its result records each row's sample at its position + 1, as the
    executor's completion does."""

    def __init__(self, runner: OverlapRunner, rows: list[tuple[int, int]]) -> None:
        self.runner, self.rows = runner, rows
        self.out = None
        self.feedback = Feedback([p + 1 for _, p in rows])

    def result(self):
        if self.out is None:
            runner = self.runner
            runner.inflight.remove(self)
            for sequence, p in self.rows:
                if runner.context.has(sequence):  # not released meanwhile
                    family = runner.family[sequence]
                    runner.context.seen(sequence, p + 1, tokens(family, p + 1, p + 2))
                    runner.sampled[sequence] = p + 1
            self.out = forward_output(
                [runner.family[s] * FAMILY + p + 1 for s, p in self.rows]
            )
        return self.out


class OverlapRunner:
    """Uses the Engram context as the native executor does, and checks each ``before`` against the tokens of the
    row's family:

    - a prefill chunk at p: ``before(p)``, then ``seen(p, chunk)``;
    - a decode step with host tokens (a fresh enqueue, or the Engine's synchronous forward): ``before(p)``, then
      ``seen(p, [token])``;
    - a decode step that continues the previous one (its tokens on the device): ``before(p)`` only, once the
      samples of every step but the previous one are recorded;
    - a decode step's samples, when the Engine takes its result: ``seen(p + 1, [sample])``."""

    def __init__(self, context_class) -> None:
        self.cache_layout = GEOMETRY.layout(256)
        self.runner_config = RunnerConfig(
            prefill_chunk_size=CHUNK,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
            continuation_extension=PUBLISH,
        )
        self.context = context_class(N, GEOMETRY.window_blocks(256))
        self.inflight: list[Completion] = []
        self.family: dict[int, int] = {}
        self.sampled: dict[int, int] = {}
        """sequence -> the position of its newest sample that the context recorded"""
        self.breaks: list[int] = []
        """the positions of decode steps with host tokens whose token the context had recorded as a sample"""
        self.wrong: list[tuple[int, int, list[int], list[int]]] = []
        """(sequence, position, before, want) of each wrong ``before`` of a sequence with a record"""
        self.wrong_restored: list[tuple[int, int, list[int], list[int]]] = []
        """the same for a sequence that starts from cached blocks (its first ``before``)"""
        self.steps: Counter[str] = Counter()

    def bind_resources(self, binding) -> None:
        pass

    def _before(self, cache, r: int, pos: int) -> None:
        sequence = int(cache.batch.sequence_ids[r])
        win = cache.groups[WINDOW_GROUP]
        ids = GEOMETRY.window_block_id(win.block_slabs(r), win.block_slots(r))
        restored = pos > 0 and not self.context.has(sequence)
        before = self.context.before(
            sequence, pos, int(win.block_table_start_indices[r]), ids
        )
        want = tokens(self.family[sequence], max(0, pos - N), pos)
        if list(before) != want:
            wrong = self.wrong_restored if restored else self.wrong
            wrong.append((sequence, pos, [int(t) for t in before], want))

    def _host_rows(self, cache) -> list[int]:
        """``before`` and ``seen`` of each row of a plan whose tokens are on the host; returns each row's end."""
        batch = cache.batch
        ends = []
        for r in range(batch.num_sequences):
            sequence, pos0 = int(batch.sequence_ids[r]), int(batch.context_lengths[r])
            ids = [int(t) for t in batch.token_ids[batch.token_slice(r)]]
            family = ids[0] // FAMILY
            assert ids == tokens(family, pos0, pos0 + len(ids))
            self.family[sequence] = family
            if self.sampled.get(sequence) == pos0:
                self.breaks.append(pos0)
            self._before(cache, r, pos0)
            self.context.seen(sequence, pos0, ids)
            ends.append(family * FAMILY + pos0 + len(ids))
        return ends

    def harvest(self, keep: int = 0) -> None:
        while len(self.inflight) > keep:
            self.inflight[0].result()

    def forward(self, plan):
        self.harvest()
        cache = PagedCachePlan.from_plan(plan, self.cache_layout)
        self.steps[
            "forward " + ("prefill" if plan.phase == BatchPhase.PREFILL else "decode")
        ] += 1
        return forward_output(self._host_rows(cache))

    def enqueue(self, plan, *, feedback: Feedback | None = None):
        if plan.phase != BatchPhase.DECODE:
            return self.forward(plan)
        cache = PagedCachePlan.from_plan(plan, self.cache_layout)
        batch = cache.batch
        # the tokens before each row's position are known once every step but the one feeding this is recorded
        self.harvest(1 if feedback is not None else 0)
        if feedback is None:
            self.steps["fresh"] += 1
            self._host_rows(cache)
            positions = [int(c) for c in batch.context_lengths]
        else:
            self.steps["continuation"] += 1
            positions = list(feedback.positions)
            for r, pos in enumerate(positions):
                self._before(cache, r, pos)
        rows = [(int(s), p) for s, p in zip(batch.sequence_ids, positions)]
        completion = Completion(self, rows)
        self.inflight.append(completion)
        return completion

    def release(self, sequence_ids) -> None:
        self.harvest()
        self.context.release(sequence_ids)


def engine(runner: OverlapRunner, **options) -> Engine:
    return Engine(
        runner,
        context_length=4096,
        max_running_requests=8,
        enable_overlap=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=CHUNK),
        **options,
    )


def submit(e: Engine, family: int, length: int, out: int):
    return e.submit(
        tokens(family, 0, length), sampling_params=SamplingParams(max_output_tokens=out)
    )


def drain(e: Engine) -> None:
    while e.has_work():
        e.step()


def test_arrivals_and_finishes_break_the_chain(context_class):
    """A request arrives while another decodes, and each finishes while the other decodes; near its output
    budget a row decodes synchronously."""
    runner = OverlapRunner(context_class)
    with engine(runner) as e:
        first = submit(e, 1, 100, 60)
        for _ in range(20):
            e.step()
        second = submit(e, 2, 300, 30)
        drain(e)
        assert first.take_result() == tokens(1, 100, 160)
        assert second.take_result() == tokens(2, 300, 330)
    assert runner.steps["continuation"] > 0 and runner.steps["forward decode"] > 0
    assert len(runner.breaks) >= 3
    assert runner.wrong == []


def test_a_published_prefix_breaks_the_chain(context_class):
    """With the prefix cache, a chain ends before a step that crosses a multiple of ``PUBLISH``."""
    runner = OverlapRunner(context_class)
    with engine(runner, prefix_cache=True) as e:
        handle = submit(e, 1, PUBLISH - 100, 200)
        drain(e)
        assert handle.take_result() == tokens(1, PUBLISH - 100, PUBLISH + 100)
    assert PUBLISH in runner.breaks
    assert runner.wrong == []


@pytest.mark.parametrize("last", [0, 1], ids=["at the last position", "one before"])
def test_block_tails_recorded_after_a_break(context_class, last):
    """A chain break at the last position of the WINDOW block before ``PUBLISH``, or at the one before it; the
    chain then ends before ``PUBLISH`` and the Engine publishes the decode's prefix. A later prefix hit at
    ``PUBLISH`` reads that block's recorded tail in its first ``before``."""
    runner = OverlapRunner(context_class)
    target = PUBLISH - 1 - last  # the break's position
    with engine(runner, prefix_cache=True) as e:
        handle = submit(e, 1, PUBLISH - 100, 200)
        # a request that arrives once the decode has recorded its sample at target - 1 breaks the chain at target
        while max(runner.sampled.values(), default=0) < target - 1:
            e.step()
        other = submit(e, 2, 10, 1)
        drain(e)
        assert handle.take_result() == tokens(1, PUBLISH - 100, PUBLISH + 100)
        assert other.take_result() == tokens(2, 10, 11)
        assert target in runner.breaks, runner.breaks
        hit = submit(e, 1, PUBLISH + 50, 1)
        drain(e)
        assert hit.take_result() == tokens(1, PUBLISH + 50, PUBLISH + 51)
        assert hit.cached_tokens == PUBLISH
    assert runner.wrong_restored == []
    assert runner.wrong == []


@pytest.mark.parametrize("fresh", [PAGE_TOKENS - 2, PAGE_TOKENS - 1])
def test_a_record_that_holds_pos0(context_class, fresh):
    """The context's calls directly: a fresh step at every position from ``fresh`` after the previous step's
    sample is recorded, tokens that replace recorded ones, a speculative step's commit from an anchor the record
    holds, and a prefix hit on the block that the fresh steps completed."""
    context = context_class(N, GEOMETRY.window_blocks(4))
    ids = np.array([2, 0])
    context.before(1, 0, 0, ids)
    context.seen(1, 0, tokens(5, 0, fresh - 1))
    # a plain decode step at fresh - 1, its sample recorded at fresh
    context.before(1, fresh - 1, 0, ids)
    context.seen(1, fresh - 1, tokens(5, fresh - 1, fresh))
    context.seen(1, fresh, tokens(5, fresh, fresh + 1))
    for p in range(fresh, PAGE_TOKENS + 2):
        # a fresh step at p, then its sample at p + 1
        assert context.before(1, p, 0, ids).tolist() == tokens(5, p - N, p)
        context.seen(1, p, tokens(5, p, p + 1))
        context.seen(1, p + 1, tokens(5, p + 1, p + 2))
    # sequence 2 starts after block 0, whose last positions the fresh steps recorded
    assert context.before(2, PAGE_TOKENS, 0, np.array([2, 3])).tolist() == tokens(
        5, PAGE_TOKENS - N, PAGE_TOKENS
    )
    # tokens at a recorded position replace those at it and after
    p = PAGE_TOKENS + 2  # the record ends at p + 1
    context.seen(1, p, [7])
    assert context.before(1, p + 1, 0, ids).tolist() == tokens(5, p - 2, p) + [7]
    context.seen(1, p - 1, [8, 9, 10])
    assert context.before(1, p + 1, 0, ids).tolist() == tokens(5, p - 2, p - 1) + [8, 9]
    assert context.before(1, p + 2, 0, ids).tolist() == [8, 9, 10]
    # a speculative step from an anchor whose position the record holds commits [anchor, accepted drafts]
    p += 2
    context.seen(1, p, [11])
    assert context.before(1, p, 0, ids).tolist() == [8, 9, 10]
    context.seen(1, p, [11, 12, 13])
    assert context.before(1, p + 3, 0, ids).tolist() == [11, 12, 13]
