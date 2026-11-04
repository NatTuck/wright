"""Paged KV gathers feeding batch_gemm.

The gather reads runtime-selected pages of the cache arena into dense per-slot slabs, with zeros
for keys past each slot's klen (those rows hold NaN here). Scores use a transposed slab, P·V a
plain one.
"""

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)

ROW, PAGE_ROWS, BLOCK_ROWS = 512, 128, 1024
C, S, M = 2048, 64, 18
GEMM_NN = struct.pack("??", False, False)
GEMM_NT = struct.pack("??", False, True)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne

    return bf16_rne(x)


def as_f32(bits: np.ndarray) -> np.ndarray:
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_to_f32

    return bf16_to_f32(bits)


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
    blocks = (6 << 30) // (ROW * BLOCK_ROWS)
    address = device.malloc(blocks * BLOCK_ROWS * ROW)
    yield address, blocks
    device.free(address)


def fill_pages(device, address: int, blocks: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Random pages at distinct arena slots: (first rows (S, C/128), page data (S, C/128, 128, 256))."""
    tiles = C // 128
    slots = blocks * BLOCK_ROWS // PAGE_ROWS
    forced = np.array([slots - 1, (5 << 30) // (ROW * PAGE_ROWS) + 1])
    rest = rng.choice(
        np.setdiff1d(np.arange(slots), forced), S * tiles - 2, replace=False
    )
    chosen = np.concatenate([forced, rest])
    first = (chosen * PAGE_ROWS).astype(np.int32).reshape(S, tiles)
    first[3, 5] = -1
    data = bf16_bits(rng.standard_normal((S, tiles, PAGE_ROWS, 256), np.float32))
    for s in range(S):
        for j in range(tiles):
            if first[s, j] >= 0:
                device.h2d(address + int(first[s, j]) * ROW, data[s, j])
    return first, data


def live_keys(
    device, address: int, first: np.ndarray, data: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Slot tables with a full, a partial-page and a short klen; stale rows NaN-filled.
    Returns (slots (S, 8), live (S, C/128, 128) mask of page rows holding keys < klen)."""
    from resin_models_gaudi.qwen3_5.checkpoint import sigma_perm

    klen = np.full(S, C, np.int32)
    klen[[1, 7, 9]] = [0, 1000, 37]
    slots = np.zeros((S, 8), np.int32)
    slots[:, 4] = klen
    key = np.arange(128) if kind == "k" else sigma_perm(128)
    tiles = C // 128
    keys = np.arange(tiles)[:, None] * 128 + key[None, :]
    live = keys[None] < klen[:, None, None]
    for s in range(S):
        for j in range(tiles):
            if first[s, j] >= 0 and not live[s, j].all():
                page = data[s, j].copy()
                page[~live[s, j]] = 0x7FC0
                device.h2d(address + int(first[s, j]) * ROW, page)
    return slots, live


def compile_gather(device, tmp_path, kind: str, blocks: int):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    graph = Graph(device, f"test_kv_gather_{kind}")
    pool = graph.input("arena", (256, BLOCK_ROWS, blocks), BF16)
    pages = graph.input("pages", (C // 128, S), I32)
    table = graph.input("slots", (8, S), I32)
    slab = graph.tensor((256, C, S), BF16, "slab")
    graph.node(
        "rg_kv_gather_f32", [pool, pages, table], [slab], struct.pack("i", kind == "v")
    )
    if kind == "k":
        a = graph.input("a", (256, M, S), BF16)
        out = graph.input("out", (C, M, S), F32)
        graph.node("batch_gemm", [a, slab], [out], GEMM_NT)
    else:
        a = graph.input("a", (C, M, S), BF16)
        out = graph.input("out", (256, M, S), F32)
        graph.node("batch_gemm", [a, slab], [out], GEMM_NN)
    recipe = RecipeCache(device, tmp_path).compile(graph)
    assert recipe.workspace < slab.nbytes, "the recipe's workspace holds the slab"
    return recipe


def run(device, recipe, address, first, slots, a_bits, out_shape) -> np.ndarray:
    buffers = {
        "pages": device.malloc(first.nbytes),
        "slots": device.malloc(slots.nbytes),
        "a": device.malloc(a_bits.nbytes),
        "out": device.malloc(int(np.prod(out_shape)) * 4),
    }
    device.h2d(buffers["pages"], first)
    device.h2d(buffers["slots"], slots)
    device.h2d(buffers["a"], a_bits)
    recipe.bind_all({"arena": address, **buffers})
    recipe.launch()
    out = device.d2h(buffers["out"], out_shape, np.float32)
    for buffer in buffers.values():
        device.free(buffer)
    return out


def test_k_gather_scores(device, arena, tmp_path):
    address, blocks = arena
    rng = np.random.default_rng(1)
    first, data = fill_pages(device, address, blocks, rng)
    slots, live = live_keys(device, address, first, data, "k")
    q = bf16_bits(rng.standard_normal((S, M, 256), np.float32))
    recipe = compile_gather(device, tmp_path, "k", blocks)
    scores = run(device, recipe, address, first, slots, q, (S, M, C))

    # a page is (key, dimension)
    k = as_f32(data)
    k[first < 0] = 0
    k[~live] = 0
    k = k.reshape(S, C, 256)
    expected = np.einsum("smd,scd->smc", as_f32(q), k, dtype=np.float64)
    np.testing.assert_allclose(scores, expected, rtol=1e-3, atol=1e-3)


def test_v_gather_output(device, arena, tmp_path):
    address, blocks = arena
    rng = np.random.default_rng(2)
    first, data = fill_pages(device, address, blocks, rng)
    slots, live = live_keys(device, address, first, data, "v")
    p = bf16_bits(rng.random((S, M, C), np.float32))
    recipe = compile_gather(device, tmp_path, "v", blocks)
    out = run(device, recipe, address, first, slots, p, (S, M, 256))

    v = as_f32(data)
    v[first < 0] = 0
    v[~live] = 0
    v = v.reshape(S, C, 256)
    expected = np.einsum("smc,scd->smd", as_f32(p), v, dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)
