"""FP8 KV pages (``kv_store8``, ``kv_gather8``) against numpy.

The store writes each key as FP8 E4M3 with its f32 scale (max |x| / 240 rounded up to a bf16 value) in its page;
the gather reads the pages back as dequantized bf16, with zeros for keys past each slot's klen (those slots hold
NaN bytes here). K pages, V pages or both are FP8; the others stay bf16 pages, which both kernels read and write
through the byte view of the arena.
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

ROW, BLOCK_ROWS, BLOCKS = 512, 1024, 64
FP8_ROWS, BF16_ROWS = 65, 128
C, S = 2048, 24
N, NT = (
    512,
    640,
)  # the store's real columns (positions 896 .. 1407: four pages of each group), all columns


def bf16_bits(x: np.ndarray) -> np.ndarray:
    from resin_models_gaudi.loading.convert import bf16_rne

    return bf16_rne(x)


def as_f32(bits: np.ndarray) -> np.ndarray:
    from resin_models_gaudi.loading.convert import bf16_to_f32

    return bf16_to_f32(bits)


def key_scale(x: np.ndarray) -> np.ndarray:
    """The f32 scale of each key (rows of 256 values): max |x| / 240, rounded up to a bf16 value."""
    s = np.maximum(np.abs(x).max(-1), np.float32(1e-30)).astype(
        np.float32
    ) * np.float32(1 / 240)
    u = s.astype(np.float32).view(np.uint32)
    return ((u + np.uint32(0xFFFF)) & np.uint32(0xFFFF0000)).view(np.float32)


def dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """bf16 bits of FP8 E4M3 ``codes`` times their key's ``scale`` (as kv_gather8 rounds it)."""
    from resin_models_gaudi.loading.convert import E4M3

    return bf16_bits(E4M3[codes] * scale[..., None])


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


@pytest.fixture
def arena(device):
    address = device.malloc(BLOCKS * BLOCK_ROWS * ROW)
    device.h2d(
        address, np.full(BLOCKS * BLOCK_ROWS * ROW, 0x7F, np.uint8)
    )  # NaN bytes everywhere
    yield address
    device.free(address)


def page_rows(rng, count: int, fp8: bool) -> np.ndarray:
    """``count`` distinct pages' first rows, not aligned to anything, in the arena."""
    size = FP8_ROWS if fp8 else BF16_ROWS
    starts = (
        rng.choice((BLOCKS * BLOCK_ROWS - size) // size, count, replace=False) * size
    )
    return (starts + rng.integers(0, size, count) % 3).astype(np.int64)


def disjoint_pages(rng, k8: bool, v8: bool) -> tuple[np.ndarray, np.ndarray]:
    """(G, 4) K and (G, 4) V page rows that do not overlap."""
    while True:
        k, v = (
            page_rows(rng, 16, k8).reshape(4, 4),
            page_rows(rng, 16, v8).reshape(4, 4),
        )
        spans = sorted(
            [(int(r), int(r) + (FP8_ROWS if k8 else BF16_ROWS)) for r in k.ravel()]
            + [(int(r), int(r) + (FP8_ROWS if v8 else BF16_ROWS)) for r in v.ravel()]
        )
        if all(a[1] <= b[0] for a, b in zip(spans, spans[1:])):
            return k, v


@pytest.mark.parametrize("fp8", [1, 2, 3], ids=["k8", "v8", "kv8"])
def test_store_then_gather(device, arena, tmp_path, fp8):
    from resin_models_gaudi.loading.convert import e4m3_encode, sigma_inv
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import BF16, I32, U8

    rng = np.random.default_rng(fp8)
    k8, v8 = bool(fp8 & 1), bool(fp8 & 2)
    pos = 896 + np.arange(N)
    kp, vp = disjoint_pages(rng, k8, v8)  # pages of positions 896 .. 1407
    page, slot = pos // 128 - 7, pos % 128
    vslot = sigma_inv(slot)
    rows = np.full((8, NT), -1, np.int64)  # columns past N: padding
    for g in range(4):
        rows[g, :N] = 2 * kp[g, page] + slot if k8 else kp[g, page] + slot
        rows[4 + g, :N] = 2 * vp[g, page] + vslot if v8 else vp[g, page] + vslot
    scale = rng.uniform(0.01, 30.0, (8, NT, 1)).astype(np.float32)
    bits = bf16_bits((rng.standard_normal((8, NT, 256)) * scale).astype(np.float32))
    bits[2, 5] = 0  # a key of zeros
    x = as_f32(bits)
    staged = bits.copy()
    staged[4:, sigma_inv(np.arange(NT))] = bits[
        4:
    ]  # V staging rows in sigma_inv order (attn_prep)

    graph = Graph(device, f"test_kv_store8_{fp8}")
    pool = graph.input("arena", (ROW, BLOCK_ROWS, BLOCKS), U8)
    table = graph.input("rows", (8, NT), I32)
    positions = graph.input("pos", (NT,), I32)
    kv = [graph.input(f"x{j}", (256, NT), BF16) for j in range(8)]
    graph.node(
        "rg_kv_store8_f32",
        [pool, table, positions, *kv],
        [graph.input("sink", (64, 1), I32)],
        struct.pack("i", fp8),
    )
    recipe = RecipeCache(device, tmp_path).compile(graph)
    buffers = {
        "rows": rows.T.astype(np.int32),  # device (8, NT): numpy (NT, 8)
        "pos": np.concatenate([pos, np.zeros(NT - N, np.int64)]).astype(np.int32),
        "sink": np.zeros((1, 64), np.int32),
        **{f"x{j}": staged[j] for j in range(8)},
    }
    addresses = {name: device.malloc(a.nbytes) for name, a in buffers.items()}
    for name, a in buffers.items():
        device.h2d(addresses[name], np.ascontiguousarray(a))
    recipe.bind_all({"arena": arena, **addresses})
    recipe.launch()
    device.sync()
    stored = device.d2h(arena, (BLOCKS * BLOCK_ROWS, ROW), np.uint8)

    for j in range(8):
        pages, f8, slots = (kp, k8, slot) if j < 4 else (vp, v8, vslot)
        first = pages[j % 4, page]
        if f8:
            s = key_scale(x[j, :N])
            codes = e4m3_encode(np.clip(x[j, :N] / s[:, None], -240, 240))
            data = stored[first + slots // 2].reshape(N, 2, 256)[
                np.arange(N), slots % 2
            ]
            got_s = (
                stored[first + 64].view(np.float32).reshape(N, 128)[np.arange(N), slots]
            )
            np.testing.assert_array_equal(got_s, s)
            exact = data == codes
            assert exact.mean() > 0.999, (
                f"group {j}: {exact.mean():.5f} of the codes exact"
            )
            near = np.abs(data.astype(np.int16) - codes.astype(np.int16))
            assert near.max() <= 1, f"group {j}: a code {near.max()} steps off"
        else:
            got = stored[first + slots].view(np.uint16)
            np.testing.assert_array_equal(got, bits[j, :N])

    # the gather of the stored pages: slot s reads group g's four pages for keys 896 .. 1407 (+ two absent pages)
    for kind, f8, pages in (("k", k8, kp), ("v", v8, vp)):
        for sig in (1, 2) if kind == "v" else (0,):
            _check_gather(device, arena, tmp_path, stored, pages, f8, kind, sig)


def _check_gather(device, arena, tmp_path, stored, pages, f8, kind, sig):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import BF16, I32, U8

    tiles = C // 128
    table = np.full((S, tiles), -1, np.int64)
    slots = np.zeros((S, 8), np.int32)
    for s in range(S):
        g = s % 4
        table[s, 7:11] = pages[g]  # keys 896 .. 1407 of 0 .. 2047
        slots[s, 3] = 0
    klen = np.full(S, 1408, np.int32)  # the keys the store wrote
    klen[[1, 5, 9]] = [1000, 1100, 0]
    klen[[2, 6]] = 1290  # a page cut inside (keys 1280 ..)
    slots[:, 4] = klen
    table[3, 9] = -1  # an absent page

    graph = Graph(device, f"test_kv_gather8_{kind}{sig}_{int(f8)}")
    pool = graph.input("arena", (ROW, BLOCK_ROWS, BLOCKS), U8)
    p = graph.input("pages", (tiles, S), I32)
    t = graph.input("slots", (8, S), I32)
    out = graph.input("out", (256, C, S), BF16)
    graph.node(
        "rg_kv_gather8_f32",
        [pool, p, t],
        [out],
        struct.pack("ii", sig if kind == "v" else 0, int(f8)),
    )
    recipe = RecipeCache(device, tmp_path).compile(graph)
    addresses = {
        "pages": device.malloc(table.size * 4),
        "slots": device.malloc(slots.nbytes),
        "out": device.malloc(256 * C * S * 2),
    }
    device.h2d(addresses["pages"], table.astype(np.int32))
    device.h2d(addresses["slots"], slots)
    recipe.bind_all({"arena": arena, **addresses})
    recipe.launch()
    device.sync()
    got = device.d2h(addresses["out"], (S, C, 256), np.uint16)
    for a in addresses.values():
        device.free(a)

    from resin_models_gaudi.loading.convert import sigma_inv

    want = np.zeros((S, C, 256), np.uint16)
    rows = np.arange(128)
    for s in range(S):
        for j in range(tiles):
            first = int(table[s, j])
            if first < 0:
                continue
            # slab row r of the page: its slot and its key
            if kind == "k":
                slot, key = rows, rows
            elif sig == 1:  # V in stored order: slot r holds key sigma(r)
                slot = rows
                key = np.where(rows & 1, 64 + rows // 2, rows // 2)
            else:
                slot, key = sigma_inv(rows), rows
            key = key + j * 128
            if f8:
                data = stored[first + slot // 2].reshape(128, 2, 256)[rows, slot % 2]
                sc = stored[first + 64].view(np.float32)[slot]
                value = dequantize(data, sc)
            else:
                value = stored[first + slot].view(np.uint16)
            ok = key < klen[s]
            want[s, j * 128 + rows[ok]] = value[ok]
    # the gather multiplies in bf16: a product can round one bf16 step from numpy's f32 product rounded to bf16
    # -0 and +0 are the same key value (an FP8 -0 code dequantizes to either)
    got = np.where(got == 0x8000, np.uint16(0), got)
    want = np.where(want == 0x8000, np.uint16(0), want)
    mism = np.argwhere(got != want)
    steps = np.abs(got.astype(np.int32) - want.astype(np.int32))
    same_sign = (got >> 15) == (want >> 15)
    assert (same_sign | (got == want)).all() and steps.max() <= 1, (
        f"{kind} sig {sig} fp8 {f8}: {len(mism)} values differ, first {mism[:3].tolist()}"
    )
    assert len(mism) <= got.size * 1e-5, (kind, sig, f8, len(mism))
    if len(mism):
        print(
            kind,
            sig,
            f8,
            "one bf16 step off:",
            [(as_f32(got[tuple(i)]), as_f32(want[tuple(i)])) for i in mism[:2]],
        )
