"""Native build, recipe cache and kernel smoke tests on a Gaudi2 card."""

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


def bf16_round(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened
    opened.close()


def test_native_build_is_content_keyed(tmp_path):
    from resin_models_gaudi.synapse.build import build_native

    first = build_native(tmp_path)
    second = build_native(tmp_path)
    assert first == second
    assert first.runtime.exists() and first.kernels.exists()
    assert set(first.kernel_digests) >= {"swiglu", "argmax1", "argmax2"}
    # Variants that include another kernel's source depend on it.
    assert first.kernel_digests["swiglu"] != first.kernel_digests["swiglu_f8"]


def test_swiglu_matches_numpy(device, tmp_path):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import BF16, F32

    F, T = 512, 3
    rng = np.random.default_rng(0)
    gu = rng.standard_normal((T, 2 * F), dtype=np.float32)
    graph = Graph(device, "test_swiglu")
    x = graph.input("gu", (2 * F, T), F32)
    y = graph.input("out", (F, T), BF16)
    graph.node("rg_swiglu_f32", [x], [y])
    recipe = RecipeCache(device, tmp_path).compile(graph)
    src, dst = device.malloc(gu.nbytes), device.malloc(F * T * 2)
    device.h2d(src, gu)
    recipe.bind_all({"gu": src, "out": dst})
    recipe.launch()
    out = bf16_to_f32(device.d2h(dst, (T, F), np.uint16))

    g, u = gu[:, :F], gu[:, F:]
    act = g / (1 + np.exp(-g)) * u
    expected = np.empty_like(act)
    for block in range(F // 128):
        i = block * 128
        expected[:, i : i + 128 : 2] = act[:, i : i + 64]
        expected[:, i + 1 : i + 128 : 2] = act[:, i + 64 : i + 128]
    np.testing.assert_allclose(out, bf16_round(expected), rtol=2e-2, atol=2e-2)
    device.free(src)
    device.free(dst)


def test_argmax_and_recipe_reuse(device, tmp_path):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, I32

    V = 248_320

    def build() -> Graph:
        graph = Graph(device, "test_argmax")
        logits = graph.input("logits", (V, 1), F32)
        part = graph.tensor((64, 24), F32)
        token = graph.input("token", (1, 1), I32)
        graph.node("rg_argmax1_f32", [logits], [part])
        graph.node("rg_argmax2_f32", [part], [token])
        return graph

    cache = RecipeCache(device, tmp_path)
    before = device.compile_count
    first = cache.compile(build())
    assert not first.from_cache and device.compile_count == before + 1
    second = cache.compile(build())
    assert second.from_cache and second.identity == first.identity
    assert device.compile_count == before + 1

    rng = np.random.default_rng(1)
    logits = rng.standard_normal(V, dtype=np.float32)
    logits[[7, 200_001]] = 9.0  # ties resolve to the first index
    src, dst = device.malloc(logits.nbytes), device.malloc(4)
    device.h2d(src, logits)
    for recipe in (first, second):
        device.memset32(dst, 0xFFFFFFFF, 1)
        recipe.bind_all({"logits": src, "token": dst})
        recipe.launch()
        assert int(device.d2h(dst, 1, np.int32)[0]) == 7
    device.free(src)
    device.free(dst)


def test_parallel_compiles_match_serial(device, tmp_path):
    import struct

    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import BF16, F32

    K, N, tokens = 512, 1024, (8, 16, 64, 128)

    def build(T: int) -> Graph:
        graph = Graph(device, f"test_parallel_{T}")
        x = graph.input("x", (K, T), BF16)
        w = graph.input("w", (N, K), BF16)
        h = graph.tensor((N, T), BF16)
        graph.node("gemm", [x, w], [h], struct.pack("??", False, False))
        y = graph.input("y", (N, T), F32)
        graph.node("cast_bf16_to_f32", [h], [y])
        return graph

    serial = [
        RecipeCache(device, tmp_path / "serial").compile(build(T)) for T in tokens
    ]
    before = device.compile_count
    parallel = RecipeCache(device, tmp_path / "parallel").compile_many(
        [build(T) for T in tokens], workers=len(tokens)
    )
    assert device.compile_count == before + len(tokens)
    assert not any(r.from_cache for r in parallel)

    rng = np.random.default_rng(3)
    w = bf16_round(rng.standard_normal((K, N), np.float32) / 16)
    w_dev = device.malloc(w.size * 2)
    device.h2d(w_dev, (w.view(np.uint32) >> 16).astype(np.uint16))
    for T, a, b in zip(tokens, serial, parallel):
        x = bf16_round(rng.standard_normal((T, K), np.float32))
        x_dev, y_dev = device.malloc(x.size * 2), device.malloc(N * T * 4)
        device.h2d(x_dev, (x.view(np.uint32) >> 16).astype(np.uint16))
        outs = []
        for recipe in (a, b):
            recipe.bind_all({"x": x_dev, "w": w_dev, "y": y_dev})
            recipe.launch()
            outs.append(device.d2h(y_dev, (T, N), np.float32))
        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_allclose(outs[0], bf16_round(x @ w), rtol=2e-2, atol=2e-2)
        device.free(x_dev)
        device.free(y_dev)
    device.free(w_dev)
