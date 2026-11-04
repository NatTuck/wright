"""Recurrent state rows moving between the cache arena and dense graph tensors."""

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

ROW, BLOCK_ROWS = 512, 1024
R, B = 2048, 4  # rows per sequence (crosses arena blocks), sequences


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


def test_load_update_store(device, tmp_path):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, I32

    blocks = 64
    rows_total = blocks * BLOCK_ROWS
    arena = device.malloc(rows_total * ROW)
    rng = np.random.default_rng(0)
    start = rng.standard_normal((rows_total, 128), np.float32)
    device.h2d(arena, start)
    # sequence 0 updates in place across a block boundary, 1 moves to new rows, 2 is fresh
    # (loads zeros, then stores them), 3 loads but skips its store
    src = np.array([BLOCK_ROWS - 300, 5 * BLOCK_ROWS, -1, 9 * BLOCK_ROWS + 7], np.int32)
    dst = np.array(
        [BLOCK_ROWS - 300, 20 * BLOCK_ROWS + 3, 30 * BLOCK_ROWS, -1], np.int32
    )

    graph = Graph(device, "test_state")
    pool = graph.input("arena", (128, BLOCK_ROWS, blocks), F32)
    src_t, dst_t = graph.input("src", (B,), I32), graph.input("dst", (B,), I32)
    state = graph.tensor((128, R, B), F32)
    graph.node("rg_state_load_f32", [pool, src_t], [state])
    twos = graph.input("twos", (128, R, B), F32)
    doubled = graph.tensor((128, R, B), F32)
    graph.node("mult_fwd_f32", [state, twos], [doubled])
    loaded = graph.input("loaded", (128, R, B), F32)
    graph.node("memcpy", [state], [loaded])
    sink = graph.input("sink", (64, 1), I32)
    graph.node("rg_state_store_f32", [pool, dst_t, doubled], [sink])
    recipe = RecipeCache(device, tmp_path).compile(graph)

    buffers = {
        name: device.malloc(max(256, t.nbytes))
        for name, t in recipe.persistent.items()
        if name != "arena"
    }
    device.h2d(buffers["src"], src)
    device.h2d(buffers["dst"], dst)
    device.h2d(buffers["twos"], np.full((B, R, 128), 2.0, np.float32))
    recipe.bind_all({"arena": arena, **buffers})
    recipe.launch()
    got = device.d2h(buffers["loaded"], (B, R, 128), np.float32)
    after = device.d2h(arena, (rows_total, 128), np.float32)

    expected = start.copy()
    for b in range(B):
        old = (
            start[src[b] : src[b] + R]
            if src[b] >= 0
            else np.zeros((R, 128), np.float32)
        )
        np.testing.assert_array_equal(got[b], old)
        if dst[b] >= 0:
            expected[dst[b] : dst[b] + R] = 2 * old
    np.testing.assert_array_equal(after, expected)
    for buffer in buffers.values():
        device.free(buffer)
    device.free(arena)
