"""Two Engines, one for each model, that share one CachePool through the Python API."""

from __future__ import annotations

import pytest
from output_helpers import forward_output

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    CachePool,
    Engine,
    ForwardOutput,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
)


class SlabRunner:
    """Each step returns the last input token + 1; slabs hold one 2-token block."""

    def __init__(self, capacity: int) -> None:
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(
                parent_capacity=capacity, groups=[CacheGroupLayout(2)]
            ),
            continuation_prefill=True,
        )
        self.binding: ResourceBinding | None = None
        self.steps = 0

    def bind_resources(self, binding: ResourceBinding) -> None:
        self.binding = binding

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self.steps += 1
        return forward_output(
            [
                plan.token_ids[end - 1] + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )


def _engine(pool: CachePool, **options) -> tuple[Engine, SlabRunner]:
    runner = SlabRunner(pool.parent_capacity)
    engine = Engine(
        runner, context_length=64, enable_overlap=False, cache_pool=pool, **options
    )
    return engine, runner


def _run(engine: Engine) -> None:
    while engine.has_work():
        engine.step()


def test_engines_bind_the_same_pool_and_report_their_own_slabs() -> None:
    pool = CachePool(8)
    (a, runner_a), (b, runner_b) = (
        _engine(pool, prefix_cache=True),
        _engine(pool, prefix_cache=True),
    )
    assert pool.engines == 2
    assert runner_a.binding is not None and runner_b.binding is not None
    assert runner_a.binding.cache_pool_id == runner_b.binding.cache_pool_id
    assert runner_a.binding.cache_pool_id == pool.pool_id
    one = a.submit([1, 2, 3, 4], sampling_params=SamplingParams(max_output_tokens=1))
    _run(a)
    two = b.submit(
        [1, 2, 3, 4, 5, 6], sampling_params=SamplingParams(max_output_tokens=1)
    )
    _run(b)
    assert one.take_result() == [5] and two.take_result() == [7]
    assert pool.bound_slabs() == [2, 3]  # each model's cached prefix
    assert (
        a.cache_free_parent_count == b.cache_free_parent_count == pool.free_slabs == 3
    )


def test_a_model_waits_while_the_other_holds_the_slabs() -> None:
    pool = CachePool(4)
    (a, _), (b, runner_b) = _engine(pool), _engine(pool)
    long = a.submit(
        [1, 2, 3, 4, 5], sampling_params=SamplingParams(max_output_tokens=3)
    )
    a.step()
    waiting = b.submit([9, 8, 7], sampling_params=SamplingParams(max_output_tokens=1))
    while a.has_work():
        assert b.step() == []  # no slab for B while A's request holds them
        a.step()
    assert runner_b.steps == 0
    _run(b)
    assert len(long.take_result()) == 3
    assert waiting.take_result() == [8]


def test_a_layout_must_declare_the_pool_capacity_and_a_floor_needs_a_pool() -> None:
    pool = CachePool(6)
    with pytest.raises(Exception, match="declares 5 slabs"):
        Engine(SlabRunner(5), cache_pool=pool)
    with pytest.raises(ValueError, match="cache_floor requires cache_pool"):
        Engine(SlabRunner(6), cache_floor=2)
    with pytest.raises(ValueError):
        CachePool(0)
    assert "parent_capacity=6" in repr(pool)
