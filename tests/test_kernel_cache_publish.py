"""Several processes that build the same artifact at the same time (every rank of a tensor-parallel model compiles
the same recipes on a cold cache) leave a usable artifact: the next reader finds an artifact whose bytes match its
record, so it loads it and does not build it again. The builds differ in their bytes."""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path

from resin.kernel_cache import ArtifactCache

ABI = "test-abi"
PROCESSES = 8
ROUNDS = 40


def publisher(root: str, index: int, barrier, results) -> None:
    """Build each round's artifact with this process's own bytes, all processes at once, as RecipeCache.compile
    does: build into local scratch, then publish into the cache."""
    cache = ArtifactCache(root, ABI)
    built = 0
    for round_ in range(ROUNDS):
        artifact = f"graph-{round_}.recipe"
        barrier.wait()
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / artifact

            def build() -> str:
                # a different size per process, so a record of one process never fits another's bytes
                source.write_bytes(
                    f"process {index} round {round_}\n".encode() * (64 + index)
                )
                return "built"

            def load(path: Path) -> str:
                assert path.read_bytes().startswith(b"process ")
                return "loaded"

            if (
                cache.load_or_build_from(
                    artifact,
                    source=source,
                    identity=f"graph:{round_}",
                    load=load,
                    build=build,
                )
                == "built"
            ):
                built += 1
    results.put((index, built))


def test_processes_building_one_artifact_at_once_leave_a_usable_artifact(
    tmp_path: Path,
):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(PROCESSES)
    results = context.Queue()
    workers = [
        context.Process(target=publisher, args=(str(tmp_path), i, barrier, results))
        for i in range(PROCESSES)
    ]
    for worker in workers:
        worker.start()
    built = dict(results.get(timeout=300) for _ in workers)
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0

    cache = ArtifactCache(tmp_path, ABI)
    unusable = [
        round_
        for round_ in range(ROUNDS)
        if not cache.validate(f"graph-{round_}.recipe", identity=f"graph:{round_}")
    ]
    assert unusable == [], (
        f"{len(unusable)} of {ROUNDS} artifacts cannot be loaded: rounds {unusable}"
    )
    # one process builds each artifact; the others wait for it and load it
    assert sum(built.values()) == ROUNDS, built


def test_threads_building_one_artifact_at_once_build_it_once(tmp_path: Path):
    """compile_many compiles in a thread pool: threads of one process also take turns"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cache = ArtifactCache(tmp_path, ABI)
    barrier = threading.Barrier(PROCESSES)
    builds: list[int] = []

    def compile_once(index: int) -> str:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "graph.recipe"

            def build() -> str:
                builds.append(index)
                source.write_bytes(b"thread %d\n" % index * (64 + index))
                return "built"

            barrier.wait()
            return cache.load_or_build_from(
                "graph.recipe",
                source=source,
                identity="graph",
                load=lambda path: "loaded",
                build=build,
            )

    with ThreadPoolExecutor(PROCESSES) as pool:
        outcomes = list(pool.map(compile_once, range(PROCESSES)))
    assert sorted(outcomes) == ["built"] + ["loaded"] * (PROCESSES - 1)
    assert len(builds) == 1
    assert cache.validate("graph.recipe", identity="graph")
    assert (cache.hits, cache.misses) == (PROCESSES - 1, 1)


def test_a_failed_build_lets_the_next_caller_build(tmp_path: Path):
    cache = ArtifactCache(tmp_path, ABI)
    source = tmp_path / "scratch.recipe"

    def fail() -> str:
        raise RuntimeError("compiler error")

    def build() -> str:
        source.write_bytes(b"recipe")
        return "built"

    try:
        cache.load_or_build_from(
            "graph.recipe", source=source, identity="graph", load=str, build=fail
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("the failed build did not raise")
    assert not cache.validate("graph.recipe", identity="graph")
    assert (
        cache.load_or_build_from(
            "graph.recipe", source=source, identity="graph", load=str, build=build
        )
        == "built"
    )
    assert cache.validate("graph.recipe", identity="graph")
