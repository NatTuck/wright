"""Device contexts without a card: a thread's work issues on its current context, ``use`` restores the one
before it, recipe compile threads take the caller's context, and each new context gets its own queue slot."""

from __future__ import annotations

import shutil
import threading
from typing import Any

import pytest

from resin_models_gaudi.synapse import build
from resin_models_gaudi.synapse.dry import DryRecipeCache, dry_device
from resin_models_gaudi.synapse.graph import Graph
from resin_models_gaudi.synapse.runtime import F32, SynapseError


def test_work_issues_on_the_current_context_and_use_restores_the_one_before():
    device = dry_device()
    lib: Any = device.lib
    first, second = device.context(), device.context()
    assert {first.id, second.id} == {1, 2}
    assert device.current_context() == device.default_context
    device.sync()
    with first.use():
        device.sync()
        with second.use():
            device.memset32(0x10, 0, 1)
        device.stream_wait(3)
        with pytest.raises(RuntimeError), second.use():
            raise RuntimeError("the body failed")
        assert device.current_context() == first
    device.sync()
    assert lib.issued_on == [0, 1, 2, 1, 0]
    assert lib.calls[3] == ("stream_wait", 3)


def test_contexts_are_per_thread():
    device = dry_device()
    other = device.context()
    seen = []

    def run() -> None:
        seen.append(device.current_context().id)
        with other.use():
            device.sync()

    with other.use():
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        device.sync()
    assert seen == [0]  # a new thread starts on the default context
    assert device.lib.issued_on == [other.id, other.id]


def test_compile_threads_take_the_callers_context():
    device = dry_device()
    context = device.context()
    cache = DryRecipeCache(device)

    def graph(index: int) -> Graph:
        built = Graph(device, f"g{index}")
        built.input("x", (4, 1), F32)
        return built

    with context.use():
        cache.compile_many([graph(i) for i in range(6)], workers=3)
    cache.compile_many([graph(6)])
    assert device.lib.compiled_on == [context.id] * 6 + [0]


def test_each_new_context_takes_a_free_queue_slot():
    device = dry_device()  # four slots; the default context has the first
    slots = [device.context().affinity for _ in range(3)]
    assert slots == [0x2, 0x4, 0x8]
    with pytest.raises(SynapseError, match="context"):
        device.context()
    assert device.context(0x2).affinity == 0x2  # a slot asked for by its mask may be shared


@pytest.mark.skipif(shutil.which(build.CXX) is None, reason="needs the C++ compiler")
def test_recipes_are_keyed_by_graph_building_not_the_rest_of_the_runtime(tmp_path, monkeypatch):
    """A change of runtime.cpp alone keeps every recipe key, so the recipe cache does not compile again; a change
    of graph.cpp does not. Both rebuild the runtime library."""
    native = tmp_path / "native"
    shutil.copytree(build.NATIVE, native)
    monkeypatch.setattr(build, "NATIVE", native)

    def identity() -> tuple[str, str]:
        library, _, graph = build._identity({"k": "1"})
        return library, graph

    library, graph = identity()
    runtime = native / "runtime.cpp"
    runtime.write_text(runtime.read_text() + "\n// a runtime change\n")
    after_runtime = identity()
    assert after_runtime[0] != library and after_runtime[1] == graph
    source = native / "graph.cpp"
    source.write_text(source.read_text() + "\n// a graph-building change\n")
    after_graph = identity()
    assert after_graph[0] != after_runtime[0] and after_graph[1] != graph


def test_each_context_counts_its_own_compiles():
    """A model's "no compilations after startup" check sees its own compiles, not those of another model of
    the process that loads after it."""
    device = dry_device()
    other = device.context()
    device.count_compile()
    with other.use():
        device.count_compile()
        device.count_compile()
        assert device.compile_count == 2
    assert device.compile_count == 1
