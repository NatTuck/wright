"""Slot attention (``attn_slots_graph`` launches planned by ``plan_slots``) against numpy causal
attention over a paged K/V history.

Cases cover decode over a long history, a chunk at an unaligned start, padded slots, and a decode
batch of sequences at different positions with their own pages. Pages sit at shuffled arena rows
and the rows past each sequence's length are NaN.
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

ROW, BLOCK_ROWS, PAGE = 512, 1024, 128
BLOCKS, G, HPG, D = 160, 4, 6, 256
NAN_BF16 = 0x7FC0


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


def write_history(device, arena: int, klen: int, rng, free: list[int]):
    """Random K/V of ``klen`` keys per group in pages taken from ``free``: (k_pages, v_pages,
    k, v) with k, v (G, klen, 256) f32 (bf16 values)."""
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne, bf16_to_f32, sigma_inv

    pages = -(-klen // PAGE)
    slots = np.array([free.pop() for _ in range(2 * G * pages)]) * PAGE
    k_pages = slots[: G * pages].reshape(G, pages)
    v_pages = slots[G * pages :].reshape(G, pages)
    k = bf16_rne(rng.standard_normal((G, pages * PAGE, D), np.float32))
    v = bf16_rne(rng.standard_normal((G, pages * PAGE, D), np.float32))
    k[:, klen:] = NAN_BF16
    v[:, klen:] = NAN_BF16
    rows = sigma_inv(np.arange(PAGE))
    for g in range(G):
        for j in range(pages):
            device.h2d(
                arena + int(k_pages[g, j]) * ROW, k[g, j * PAGE : (j + 1) * PAGE]
            )
            vpage = np.empty((PAGE, D), np.uint16)
            vpage[rows] = v[g, j * PAGE : (j + 1) * PAGE]
            device.h2d(arena + int(v_pages[g, j]) * ROW, vpage)
    return k_pages, v_pages, bf16_to_f32(k[:, :klen]), bf16_to_f32(v[:, :klen])


# median / max per-row relative error limits
STRICT, DEFAULT = (3e-3, 2e-2), (1.5e-2, 5e-2)


@pytest.mark.parametrize(
    ("tokens", "slots", "query_block", "rows", "launches", "softmax"),
    [
        (1, 4, 1, [(5000, 1)], 3, None),  # decode
        (
            128,
            4,
            128,
            [(2000, 100)],
            2,
            4,
        ),  # an unaligned chunk
        (128, 4, 128, [(2000, 100)], 2, 0),
        (1024, 8, 512, [(1500, 1000)], 2, 4),  # padded slots
        (1024, 8, 512, [(1500, 1000)], 2, 0),
        (4096, 16, 512, [(12288, 4096)], 15, 0),  # a long history
        (4, 16, 1, [(5000, 1), (37, 1), (2047, 1), (0, 1)], 2, None),  # a decode batch
    ],
)
def test_slot_attention(
    device, tmp_path, monkeypatch, tokens, slots, query_block, rows, launches, softmax
):
    from resin_models_gaudi.qwen3_5 import layers
    from resin_models_gaudi.qwen3_5.attention import SlotRow, plan_slots
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne, bf16_to_f32
    from resin_models_gaudi.qwen3_5.layers import Dims, attn_slots_graph
    from resin_models_gaudi.synapse.graph import RecipeCache

    if softmax is not None:
        monkeypatch.setattr(layers, "SDPA_SOFTMAX", softmax)

    rng = np.random.default_rng(tokens)
    W = HPG * tokens
    arena = device.malloc(BLOCKS * BLOCK_ROWS * ROW)
    free = rng.permutation(BLOCKS * BLOCK_ROWS // PAGE).tolist()
    histories, slot_rows, column = [], [], 0
    for pos0, real in rows:
        k_pages, v_pages, k, v = write_history(device, arena, pos0 + real, rng, free)
        histories.append((column, pos0, real, k, v))
        slot_rows.append(SlotRow(column, pos0, real, k_pages, v_pages))
        column += real
    q_bits = bf16_rne(2 * rng.standard_normal((G, W, D), np.float32))
    planned = plan_slots(slot_rows, query_block, HPG, slots)
    assert len(planned) == launches

    dims = Dims(0, 0, 0, G * HPG, G, D, 0, 0, 0, 0.0)
    recipe = RecipeCache(device, tmp_path).compile(
        attn_slots_graph(device, dims, tokens, slots, BLOCKS, query_block)
    )
    buffers = {
        "q_all": q_bits,
        "acc_o": np.zeros((G, W, D), np.float32),
        "acc_m": np.full((G, W), -3.0e38, np.float32),
        "acc_l": np.zeros((G, W), np.float32),
        "slots": planned[0].slots,
        "k_pages": planned[0].k_pages,
        "v_pages": planned[0].v_pages,
        "sink_slots": np.zeros(64, np.int32),
    }
    address = {name: device.malloc(max(256, a.nbytes)) for name, a in buffers.items()}
    for name, a in buffers.items():
        device.h2d(address[name], a)
    recipe.bind_all({"arena": arena, **address})
    for launch in planned:
        device.h2d(address["slots"], launch.slots)
        device.h2d(address["k_pages"], launch.k_pages)
        device.h2d(address["v_pages"], launch.v_pages)
        recipe.launch()
    out = device.d2h(address["acc_o"], (G, W, D), np.float32)
    for a in [arena, *address.values()]:
        device.free(a)

    assert np.isfinite(out).all()
    q = bf16_to_f32(q_bits).reshape(G, tokens, HPG, D)
    got = out.reshape(G, tokens, HPG, D)
    rel = []
    for column, pos0, real, k, v in histories:
        for g in range(G):
            for t in range(0, real, max(1, real // 256)):
                visible = pos0 + t + 1
                s = q[g, column + t].astype(np.float64) @ k[g, :visible].T / np.sqrt(D)
                p = np.exp(s - s.max(axis=1, keepdims=True))
                expected = (p / p.sum(axis=1, keepdims=True)) @ v[g, :visible]
                rel.append(
                    np.linalg.norm(got[g, column + t] - expected, axis=-1)
                    / np.linalg.norm(expected, axis=-1)
                )
    rel = np.concatenate(rel)
    median, worst = DEFAULT if softmax == 0 else STRICT
    assert np.median(rel) < median and rel.max() < worst, (np.median(rel), rel.max())
