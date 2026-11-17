"""Startup progress: the shared board, saved weight totals, and both displays."""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest

from resin_models_gaudi.loading import progress as progress_module
from resin_models_gaudi.loading.progress import Board, Progress, Renderer
from resin_models_gaudi.synapse import build
from resin_models_gaudi.synapse.dry import DryRecipeCache, dry_device
from resin_models_gaudi.synapse.graph import Graph, _compile_summary
from resin_models_gaudi.synapse.runtime import F32


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fresh(monkeypatch):
    """A new process-wide ``progress`` for the test."""
    fresh = Progress()
    monkeypatch.setattr(progress_module, "progress", fresh)
    for module in (
        "resin_models_gaudi.synapse.runtime",
        "resin_models_gaudi.synapse.graph",
        "resin_models_gaudi.synapse.build",
    ):
        monkeypatch.setattr(f"{module}.progress", fresh)
    return fresh


def test_ranks_write_their_records_of_one_board_file(tmp_path):
    path = tmp_path / "board"
    parent = Board(2, 1, path)
    for rank in range(2):
        reporter = Progress()
        reporter.attach(Board(2, 1, path), rank)  # each rank maps the file itself
        reporter.phase("weights", total=100)
        reporter.advance(30 + rank)
        reporter.note(f"layer {rank}")
    rows = [parent.read(0, rank) for rank in range(2)]
    assert [int(r["done"]) for r in rows] == [30, 31]
    assert [r["note"].decode() for r in rows] == ["layer 0", "layer 1"]


def test_a_second_load_uses_the_saved_exact_total(tmp_path):
    first = Progress()
    first.remember(tmp_path, model="m", checkpoint="/c", ranks=8)
    first.attach(Board(8), rank=3)
    first.phase("weights", total=1000)
    assert not first.board.read(0, 3)["exact"]
    first.advance(1234)
    first.finish_weights()

    second = Progress()
    second.remember(tmp_path, model="m", checkpoint="/c", ranks=8)
    second.attach(Board(8), rank=3)
    second.phase("weights", total=1000)
    record = second.board.read(0, 3)
    assert int(record["total"]) == 1234 and record["exact"]

    other = Progress()  # other options: another key
    other.remember(tmp_path, model="m", checkpoint="/c", ranks=4)
    other.attach(Board(4), rank=3)
    other.phase("weights", total=1000)
    assert int(other.board.read(0, 3)["total"]) == 1000


def test_the_log_display_reports_phase_changes_and_every_30_seconds():
    board, clock, lines = Board(), Clock(), []
    reporter = Progress()
    reporter.attach(board)
    renderer = Renderer(
        board, ["Qwen/Qwen3.8-27B-FP8"], live=False, clock=clock, log=lines.append
    )
    reporter.phase("kernels")
    renderer.draw()
    clock.now += 2
    reporter.phase("weights", total=10 * 2**30)
    renderer.draw()
    clock.now += 10
    reporter.advance(5 * 2**30)
    renderer.draw()  # no change of phase, and less than 30 s: nothing
    clock.now += 25
    renderer.draw()
    clock.now += 3
    reporter.phase("recipes", total=4)
    reporter.advance(4)
    reporter.phase("warm-up")
    renderer.draw()  # recipes started and ended between two looks
    reporter.phase("ready")
    clock.now += 1
    renderer.draw()
    assert lines[0].startswith("kernels")
    assert lines[1] == "kernels done in 2 s"
    assert lines[2].startswith("weights:")
    assert "5.0/10.0 GiB per card" in lines[3] and "estimated" in lines[3]
    assert " 50%" in lines[3]
    assert lines[4].startswith("weights done in 38 s")
    assert lines[5] == "recipes done in <1 s"
    assert lines[6].startswith("warm-up")
    assert lines[-2] == "warm-up done in 1 s"
    assert lines[-1] == (
        "ready after 41 s: kernels 2 s, weights 38 s (5.0 GiB per card, 135 MiB/s), "
        "recipes <1 s, warm-up 1 s"
    )


def test_an_estimated_total_stops_at_99_percent_and_an_exact_one_reaches_100():
    board = Board()
    reporter = Progress()
    reporter.attach(board)
    renderer = Renderer(board, ["m"], live=True, stream=io.StringIO(), clock=Clock())
    reporter.phase("weights", total=100)
    reporter.advance(150)  # more than the estimate
    assert renderer._fraction([board.read(0, 0)])[0] == 0.99
    reporter.phase("recipes", total=7)
    reporter.advance(7)
    assert renderer._fraction([board.read(0, 0)])[0] == 1.0


def test_the_live_block_shows_every_rank_and_the_slowest():
    board, clock = Board(4), Clock()
    stream = io.StringIO()
    renderer = Renderer(
        board, ["zai-org/GLM-5.3-Flash"], live=True, stream=stream, clock=clock
    )
    for rank in range(4):
        reporter = Progress()
        reporter.attach(board, rank)
        reporter.phase("kernels")
    renderer.draw()
    clock.now += 3
    for rank in range(4):
        reporter = Progress()
        reporter.attach(board, rank)
        board.write(0, rank, phase=2, total=4 * 2**30, exact=1, done=(rank + 1) * 2**29)
    renderer.draw()
    clock.now += 10
    block = renderer.lines()
    text = "\n".join(block)
    assert block[0] == "zai-org/GLM-5.3-Flash on 4 cards"
    assert "done  kernels" in text
    assert "slowest rank 0 at 12%" in text
    assert "MiB/s" in text and "left" in text and "estimated" not in text
    assert "0 -----  1 #----  2 #----  3 ##---" in text
    # redrawing moves up over the previous block and clears it
    renderer.draw()
    assert f"\x1b[{len(block)}F\x1b[J" in stream.getvalue()


def test_log_lines_appear_above_the_live_block(monkeypatch):
    stream = io.StringIO()
    logger = logging.getLogger("wright")
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    logger.setLevel(logging.INFO)
    board = Board()
    renderer = Renderer(board, ["m"], live=True, stream=stream, clock=Clock())
    renderer._install_log_hooks()
    renderer.draw()
    drawn = renderer._drawn
    logger.info("a log line")
    renderer._remove_log_hooks()
    out = stream.getvalue()
    cleared = out.index(f"\x1b[{drawn}F\x1b[J")
    line = out.index("a log line")
    assert cleared < line  # the block was removed before the line
    assert out.rstrip().endswith(
        renderer.lines()[-1].rstrip()
    )  # and drawn again after it


def test_the_device_and_the_recipe_cache_report_progress(fresh):
    device = dry_device()
    recipes = DryRecipeCache(device)
    device.begin_load(1000)
    device.h2d(device.malloc(64), np.zeros(16, np.float32))
    assert fresh.current() == "weights" and fresh.done() == 64
    device.end_load()
    device.h2d(
        device.malloc(64), np.zeros(16, np.float32)
    )  # after the load: not counted
    assert fresh.done() == 64

    def graph(name: str) -> Graph:
        g = Graph(device, name)
        g.input("x", (4,), F32)
        return g

    recipes.compile_many([graph("a"), graph("b")])
    recipes.compile_many([graph("c")])  # a second batch adds to the phase
    record = fresh.board.read(0, 0)
    assert fresh.current() == "recipes"
    assert int(record["done"]) == 3 and int(record["total"]) == 3
    assert int(record["to_compile"]) == 3 and int(record["compiled"]) == 3
    assert record["note"].decode() == "binding recipes"


def test_the_recipes_phase_counts_graphs_compiles_and_the_cache(fresh):
    device = dry_device()
    recipes = DryRecipeCache(device)
    lines, clock = [], Clock()
    renderer = Renderer(fresh.board, ["m"], live=False, clock=clock, log=lines.append)
    fresh.building_graphs()
    graphs = []
    for name in ("a", "b", "c"):
        graphs.append(Graph(device, name))
        graphs[-1].input("x", (4,), F32)
    renderer.draw()
    assert lines[-1] == "recipes: building graphs: 3"
    recipes.compile_many(graphs)
    record = fresh.board.read(0, 0)
    assert int(record["graphs"]) == 3 and int(record["to_compile"]) == 3
    assert int(record["compiled"]) == 3 and int(record["done"]) == 3
    assert record["exact"]  # a phase that graph building opened has an exact total: it reaches 100%
    assert renderer._fraction([record])[0] == 1.0
    clock.now += 2
    fresh.phase("warm-up")
    renderer.draw()
    fresh.phase("ready")
    renderer.draw()
    assert "recipes 2 s (3 compiled, 0 from the cache)" in lines[-1]


def test_the_recipes_line_has_the_compiles_the_cache_and_the_time_left():
    board, clock = Board(2), Clock()
    renderer = Renderer(board, ["m"], live=False, clock=clock, log=lambda _: None)
    for rank in range(2):
        reporter = Progress()
        reporter.attach(board, rank)
        reporter.phase("recipes", total=100)
    renderer.draw()
    clock.now += 20
    # rank 0: 10 of 40 compiles done (30 left at 2 s each); rank 1: 20 of 40, with 5 by another rank
    board.write(0, 0, to_compile=40, compiled=10, cached=60, done=70)
    board.write(0, 1, to_compile=40, compiled=15, shared=5, cached=60, done=80)
    rows = [board.read(0, rank) for rank in range(2)]
    line = renderer._status(0, rows, 3)
    assert "75/100 graphs per card" in line
    assert "15/40 compiled (0.8/s)" in line
    assert "60 from the cache" in line
    assert "about 60 s left" in line  # the slowest rank: 30 compiles at 20 s per 10
    # the hits and misses do not hide behind notes that differ by rank
    board.write(0, 0, note=b"x")
    board.write(0, 1, note=b"y")
    assert "60 from the cache" in renderer._status(
        0, [board.read(0, r) for r in range(2)], 3
    )


def test_no_time_left_before_every_rank_finished_a_compile():
    board, clock = Board(), Clock()
    renderer = Renderer(board, ["m"], live=False, clock=clock, log=lambda _: None)
    reporter = Progress()
    reporter.attach(board)
    reporter.phase("recipes", total=10)
    renderer.draw()
    clock.now += 5
    board.write(0, 0, to_compile=4, cached=6, done=6)
    assert "left" not in renderer._status(0, [board.read(0, 0)], 3)


def test_systemd_status_follows_the_load_at_most_once_a_second():
    board, clock, sent = Board(), Clock(), []
    reporter = Progress()
    reporter.attach(board)
    renderer = Renderer(
        board, ["m"], live=False, clock=clock, log=lambda _: None, status=sent.append
    )
    reporter.phase("kernels", total=4)
    reporter.advance(1)
    renderer.draw()
    renderer.draw()  # the same second: nothing
    clock.now += 1
    reporter.phase("ready")
    renderer.draw()
    assert sent == ["kernels: 25% 1/4 kernels", "ready"]


def test_the_kernel_build_counts_each_kernel_and_the_libraries(
    fresh, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        build,
        "kernel_sources",
        lambda: {f"k{i}": tmp_path / f"k{i}.c" for i in range(5)},
    )
    monkeypatch.setattr(build, "_run", lambda *args: None)
    monkeypatch.setattr(build, "_embed", lambda *args: None)
    build._build(tmp_path)
    record = fresh.board.read(0, 0)
    assert fresh.current() == "kernels"
    assert int(record["total"]) == 6 and int(record["done"]) == 6
    assert record["note"].decode() == ""


def test_the_compile_summary_names_the_slowest_compiles():
    device = dry_device()

    def recipe(name: str, from_cache: bool, seconds: float):
        from resin_models_gaudi.synapse.graph import Recipe

        return Recipe(device, 0, name, "", from_cache, {}, compile_seconds=seconds)

    recipes = [
        recipe("attn", False, 6.1),
        recipe("moe", False, 2.0),
        recipe("head", True, 0),
        recipe("norm", True, 0),
        recipe("mlp", False, 4.2),
        recipe("rope", False, 1.0),
    ]
    missing = [True, True, True, False, True, True]
    line = _compile_summary(recipes, missing, 12.0, threads=128)
    assert line == (
        "recipes: 6 graphs in 12.0 s: 4 compiled (13 s of compile time on 4 threads), "
        "1 compiled by another process, 1 from the cache; slowest: attn 6.1 s, mlp 4.2 s, moe 2.0 s"
    )
