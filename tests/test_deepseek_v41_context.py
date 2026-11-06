"""DeepSeek-V4.1's Engram context on CPU: the tokens before the first position of every row, with the Engine's
prefix cache (hits after prefill and after decode, the WINDOW ring after a hit, eviction and block reuse)."""

from __future__ import annotations

import numpy as np
import pytest
from output_helpers import forward_output

from resin import Engine, RunnerConfig, SamplingParams, SchedulerConfig
from resin.backend.plans import PagedCachePlan
from resin_models_gaudi.deepseek_v41.cache import (
    PAGE_TOKENS,
    WINDOW_GROUP,
    CacheGeometry,
    EngramContext,
)

N = 3
FAMILY = 10_000
CHUNK = 256
# two KV owners of ratio 1 make a HISTORY block 768 rows, so three WINDOW blocks share a slab
GEOMETRY = CacheGeometry(2, (0, 1), (1, 1), 128, CHUNK)


def tokens(family: int, start: int, end: int) -> list[int]:
    """The tokens of a family at positions start .. end - 1: every family is one long sequence."""
    return [family * FAMILY + p for p in range(start, end)]


class ContextRunner:
    """Uses ``EngramContext`` as the Gaudi runner does (``before`` for each row, then ``seen`` for its tokens) and
    checks each ``before`` against the tokens of the row's family."""

    def __init__(self, capacity: int) -> None:
        self.cache_layout = GEOMETRY.layout(capacity)
        self.runner_config = RunnerConfig(
            prefill_chunk_size=CHUNK,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        self.context = EngramContext(N, GEOMETRY.window_blocks(capacity))
        self.rows = self.restored = 0

    def bind_resources(self, binding) -> None:
        pass

    def forward(self, plan):
        cache = PagedCachePlan.from_plan(plan, self.cache_layout)
        batch, win = cache.batch, cache.groups[WINDOW_GROUP]
        out = []
        for r in range(batch.num_sequences):
            sequence, pos0 = int(batch.sequence_ids[r]), int(batch.context_lengths[r])
            ids = [int(t) for t in batch.token_ids[batch.token_slice(r)]]
            family = ids[0] // FAMILY
            assert ids == tokens(family, pos0, pos0 + len(ids))
            known = sequence in self.context.last
            ids_win = GEOMETRY.window_block_id(win.block_slabs(r), win.block_slots(r))
            before = self.context.before(
                sequence, pos0, int(win.block_table_start_indices[r]), ids_win
            )
            assert before.tolist() == tokens(family, max(0, pos0 - N), pos0)
            self.rows += 1
            self.restored += pos0 > 0 and not known
            self.context.seen(sequence, pos0, ids)
            out.append(family * FAMILY + pos0 + len(ids))
        return forward_output(out)

    def release(self, sequence_ids) -> None:
        self.context.release(sequence_ids)


def run(engine: Engine, family: int, length: int, out: int = 1):
    handle = engine.submit(
        tokens(family, 0, length), sampling_params=SamplingParams(max_output_tokens=out)
    )
    while engine.has_work():
        engine.step()
    assert handle.take_result() == tokens(family, length, length + out)
    return handle


def engine(runner: ContextRunner, **options) -> Engine:
    return Engine(
        runner,
        context_length=4096,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=CHUNK),
        **options,
    )


def test_hits_after_prefill_and_after_a_hit():
    runner = ContextRunner(64)
    with engine(runner) as e:
        run(e, 1, 700)  # publishes 256 and 512
        assert (
            run(e, 1, 900).cached_tokens == 512
        )  # then publishes 768 from its own blocks
        assert run(e, 1, 1000).cached_tokens == 768
    assert runner.restored == 2


def test_hit_on_blocks_that_decode_wrote():
    """A second turn: its prompt is the first turn's prompt and output, and decode published the cached prefix."""
    runner = ContextRunner(64)
    with engine(runner) as e:
        run(e, 2, 200, out=120)  # decode reaches 256 and publishes it
        assert run(e, 2, 400).cached_tokens == 256
    assert runner.restored == 1


def test_records_of_reused_blocks_are_their_new_tokens():
    """A small cache: older families' prefixes are evicted and their blocks hold other families' tokens. A stale
    record would give a hit the tokens of another family."""
    runner = ContextRunner(24)
    hits = 0
    with engine(runner) as e:
        for family in range(1, 13):
            run(e, family, 600)
            hits += run(e, family, 800).cached_tokens > 0
        stats = e.prefix_cache_stats
    assert hits == 12 and runner.restored == 12
    assert stats is not None and stats[3] > 0  # evictions


def test_concurrent_rows_and_hits():
    runner = ContextRunner(48)
    with engine(runner, max_running_requests=4) as e:
        for family in (1, 2):
            run(e, family, 520)
        handles = [
            e.submit(
                tokens(f, 0, n), sampling_params=SamplingParams(max_output_tokens=40)
            )
            for f, n in ((1, 700), (2, 610), (3, 300), (4, 900))
        ]
        while e.has_work():
            e.step()
        for handle, (f, n) in zip(handles, ((1, 700), (2, 610), (3, 300), (4, 900))):
            assert handle.take_result() == tokens(f, n, n + 40)
        assert [h.cached_tokens for h in handles[:2]] == [512, 512]
    assert runner.restored >= 2


def test_a_hit_needs_a_recorded_block_before_it():
    context = EngramContext(N, GEOMETRY.window_blocks(4))
    ids = np.arange(4)
    with pytest.raises(RuntimeError, match="no recorded tokens"):
        context.before(7, PAGE_TOKENS, 0, ids)
    with pytest.raises(RuntimeError, match="not after a block"):
        context.before(7, PAGE_TOKENS + 2, 0, ids)
    # sequence 1 writes blocks 0 and 1 (ids 2 and 3); sequence 7 then starts after block 1
    ids = np.array([2, 3])
    context.before(1, 0, 0, ids)
    context.seen(1, 0, tokens(5, 0, 2 * PAGE_TOKENS - 1))
    context.seen(
        1, 2 * PAGE_TOKENS - 1, tokens(5, 2 * PAGE_TOKENS - 1, 2 * PAGE_TOKENS)
    )
    context.release([1])
    assert 1 not in context.last
    before = context.before(7, 2 * PAGE_TOKENS, 0, ids)
    assert before.tolist() == tokens(5, 2 * PAGE_TOKENS - N, 2 * PAGE_TOKENS)
    assert context.before(7, 0, 0, ids).tolist() == []
    with pytest.raises(ValueError):
        EngramContext(0, 4)
