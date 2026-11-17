"""Slot attention launched again and again on the same inputs gives the same accumulators.

The case is an MTP-2 verify step of 8 rows (Qwen3.8 shapes: 4 KV groups of 6 heads, 3 columns a row from position
2048, so 2 key chunks a row and 64 slots). Row 7's 18 columns (126 .. 143) put 2 columns in attn_slot_acc's
member block 1, and the row's two slots merge into those columns one after the other.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from test_slot_attention import BLOCK_ROWS, BLOCKS, HPG, PAGE, ROW, D, G, write_history

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)

ROWS, WIDTH, POS0, REPEATS = 8, 3, 2048, 40


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


@pytest.mark.parametrize("slots", [32, 64])
def test_repeated_launches_give_the_same_accumulators(device, slots):
    """``REPEATS`` runs of the step's slot launches (``slots`` a launch: 2 launches or 1) on the same inputs give
    acc_o, acc_m and acc_l equal bit for bit to the first run's."""
    from resin_models_gaudi.common.attention import SlotRow, plan_slots
    from resin_models_gaudi.common.layers import Dims, attn_slots_graph
    from resin_models_gaudi.loading.convert import bf16_rne
    from resin_models_gaudi.synapse.graph import RecipeCache

    rng = np.random.default_rng(7)
    arena = device.malloc(BLOCKS * BLOCK_ROWS * ROW)
    device.memset32(arena, 0, BLOCKS * BLOCK_ROWS * ROW // 4)
    free = rng.permutation(BLOCKS * BLOCK_ROWS // PAGE).tolist()
    rows = []
    for r in range(ROWS):
        k_pages, v_pages, _, _ = write_history(device, arena, POS0 + WIDTH, rng, free)
        rows.append(SlotRow(WIDTH * r, POS0, WIDTH, k_pages, v_pages))
    tokens = ROWS * WIDTH
    W = HPG * tokens
    q = bf16_rne(2 * rng.standard_normal((G, W, D), np.float32))
    planned = plan_slots(rows, WIDTH, HPG, slots)
    dims = Dims(0, 0, 0, G * HPG, G, D, 0, 0, 0, 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        recipe = RecipeCache(device, tmp).compile(
            attn_slots_graph(
                device, dims, tokens, slots, BLOCKS, WIDTH, per_column=True
            )
        )
    start = {
        "q_all": q,
        "acc_o": np.zeros((G, W, D), np.float32),
        "acc_m": np.full((G, W), -3.0e38, np.float32),
        "acc_l": np.zeros((G, W), np.float32),
    }
    tables = ("slots", "k_pages", "v_pages")
    sizes = {
        **{name: a.nbytes for name, a in start.items()},
        **{name: getattr(planned[0], name).nbytes for name in tables},
        "sink_slots": 256,
    }
    address = {name: device.malloc(max(256, n)) for name, n in sizes.items()}
    recipe.bind_all({"arena": arena, **address})
    runs = []
    for _ in range(REPEATS):
        for name, a in start.items():
            device.h2d(address[name], a)
        for launch in planned:
            for name in tables:
                device.h2d(address[name], getattr(launch, name))
            recipe.launch()
        device.sync()
        runs.append(
            (
                device.d2h(address["acc_o"], (G, W, D), np.float32),
                device.d2h(address["acc_m"], (G, W), np.float32),
                device.d2h(address["acc_l"], (G, W), np.float32),
            )
        )
    for a in address.values():
        device.free(a)
    device.free(arena)
    assert np.isfinite(runs[0][0]).all()
    parted = []
    for run, out in enumerate(runs[1:], 1):
        for name, x, y in zip(("acc_o", "acc_m", "acc_l"), runs[0], out, strict=True):
            if not np.array_equal(x, y):
                columns = sorted({int(c) for c in np.argwhere(x != y)[:, 1]})
                parted.append((run, name, columns))
    assert not parted, (len(parted), parted[:4])
