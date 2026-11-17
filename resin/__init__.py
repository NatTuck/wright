"""Resin's Rust control plane.

Importing this module intentionally does not import an accelerator framework.
Choose a separately installed model runner only when constructing an engine.
"""

import resin.numa as numa
import resin.process as process
import resin.recipe as recipe
import resin.server as server
from resin._resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CachePool,
    CacheRetention,
    Engine,
    EngineMetrics,
    ForwardOutput,
    Parallelism,
    RequestLease,
    ResourceBinding,
    RunnerConfig,
    RunnerMetrics,
    SamplingParams,
    SchedulerConfig,
    SlabTable,
    TokenEvent,
    TokenLogprobs,
    __version__,
    set_log_level,
    set_log_sink,
)
from resin.protocols import (
    ManagedRunner,
    MediaInput,
    MediaTokenSpan,
    Runner,
)

__all__ = [
    "BatchPhase",
    "BatchPlan",
    "CacheGroupFamily",
    "CacheGroupLayout",
    "CacheGroupPlan",
    "CacheLayout",
    "CachePool",
    "CacheRetention",
    "Engine",
    "EngineMetrics",
    "ForwardOutput",
    "ManagedRunner",
    "Parallelism",
    "MediaInput",
    "MediaTokenSpan",
    "ResourceBinding",
    "RequestLease",
    "RunnerConfig",
    "RunnerMetrics",
    "SamplingParams",
    "SchedulerConfig",
    "SlabTable",
    "TokenEvent",
    "TokenLogprobs",
    "Runner",
    "__version__",
    "numa",
    "set_log_level",
    "set_log_sink",
    "process",
    "recipe",
    "server",
]
