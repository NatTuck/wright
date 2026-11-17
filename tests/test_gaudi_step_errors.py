"""A Synapse error from a Gaudi runner's step stops the Engine (post accept): the device state it leaves is
unknown. An error raised before the step reaches the card (a ValueError check) stays retryable."""

from __future__ import annotations

import pytest

from resin import (
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
)
from resin_models_gaudi.synapse.runtime import SynapseError


class FailingRunner:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.released: list[int] = []
        self.cache_layout = CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2)])
        self.runner_config = RunnerConfig(cache_layout=self.cache_layout)

    def bind_resources(self, binding: ResourceBinding) -> None:
        pass

    def forward(self, plan):
        raise self.error

    def release(self, request_id: int) -> None:
        self.released.append(request_id)


def test_a_synapse_error_in_a_step_stops_the_engine():
    runner = FailingRunner(SynapseError("launch x: device fault"))
    engine = Engine(runner)
    engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=1))
    with pytest.raises(SynapseError, match="device fault"):
        engine.step()
    assert engine.poisoned
    assert runner.released == []  # its resources are not handed back to the allocator


def test_an_error_before_the_card_is_retryable():
    runner = FailingRunner(ValueError("position exceeds 4096"))
    engine = Engine(runner)
    engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=1))
    with pytest.raises(ValueError, match="position exceeds"):
        engine.step()
    assert not engine.poisoned
