"""Cache-arena rows stored and read back by a graph.

Rows are chosen by runtime row tables. A row of -1 reads zeros, and a store leaves neighbouring
rows alone.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)

ROW, PAGE_ROWS = 512, 1024


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


@pytest.fixture(scope="module")
def arena(device):
    """A 6 GiB arena."""
    pages = (6 << 30) // (ROW * PAGE_ROWS)
    address = device.malloc(pages * PAGE_ROWS * ROW)
    yield address, pages
    device.free(address)


def test_store_and_gather_through_the_arena_input(device, arena, tmp_path):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import I32

    address, pages = arena
    rows_total = pages * PAGE_ROWS
    device.memset32(address, 0x11111111, rows_total * ROW // 4)
    n = 48
    rows = (
        np.random.default_rng(0).choice(rows_total, n, replace=False).astype(np.int32)
    )
    rows[:3] = [0, rows_total - 1, (5 << 30) // ROW + 3]
    rows[7] = -1
    old = np.arange(n * 128, dtype=np.int32).reshape(n, 128) + 7
    new = np.arange(n * 128, dtype=np.int32).reshape(n, 128) * 3 + 100_000
    for i, row in enumerate(rows):
        if row >= 0:
            device.h2d(address + int(row) * ROW, old[i])

    graph = Graph(device, "test_arena_store")
    pool = graph.input("arena", (128, PAGE_ROWS, pages), I32)
    write_rows = graph.input("write_rows", (n,), I32)
    read_rows = graph.input("read_rows", (n,), I32)
    vals = graph.input("vals", (128, n), I32)
    before = graph.input("before", (128, n), I32)
    after = graph.input("after", (128, n), I32)
    sink = graph.input("sink", (64, 1), I32)
    graph.node("rg_page_rows_f32", [pool, write_rows], [before])
    graph.node("rg_page_store_dep_f32", [pool, write_rows, vals, before], [sink])
    graph.node("rg_page_rows_dep_f32", [pool, read_rows, sink], [after])
    recipe = RecipeCache(device, tmp_path).compile(graph)
    buffers = {
        name: device.malloc(max(256, tensor.nbytes))
        for name, tensor in recipe.persistent.items()
        if name != "arena"
    }
    device.h2d(buffers["write_rows"], rows)
    device.h2d(buffers["read_rows"], rows)
    device.h2d(buffers["vals"], new)
    recipe.bind_all({"arena": address, **buffers})
    recipe.launch()

    valid = rows >= 0
    got_before = device.d2h(buffers["before"], (n, 128), np.int32)
    got_after = device.d2h(buffers["after"], (n, 128), np.int32)
    np.testing.assert_array_equal(got_before[valid], old[valid])
    np.testing.assert_array_equal(got_after[valid], new[valid])
    assert not got_before[~valid].any() and not got_after[~valid].any()
    neighbours = sorted(
        {int(r) + d for r in rows[valid] for d in (-1, 1)} - {int(r) for r in rows}
    )
    for row in (r for r in neighbours if 0 <= r < rows_total):
        assert (device.d2h(address + row * ROW, 128, np.int32) == 0x11111111).all()
    for buffer in buffers.values():
        device.free(buffer)
