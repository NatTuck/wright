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


@pytest.fixture(scope="module")
def other_context(device):
    """A context on a queue slot of its own, made once for the module."""
    return device.context()


def test_a_context_has_its_own_queue_slot(device, other_context):
    assert other_context.id != 0
    assert other_context.affinity and device.default_context.affinity
    assert other_context.affinity & device.default_context.affinity == 0
    with other_context.use():
        assert device.current_context() == other_context
    assert device.current_context() == device.default_context


def _gemm(device, name: str):
    import struct

    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    K, N, T = 512, 1024, 64
    graph = Graph(device, name)
    x = graph.input("x", (K, T), BF16)
    w = graph.input("w", (N, K), BF16)
    h = graph.tensor((N, T), BF16)
    graph.node("gemm", [x, w], [h], struct.pack("??", False, False))
    y = graph.input("y", (N, T), F32)
    graph.node("cast_bf16_to_f32", [h], [y])
    return graph, (K, N, T)


def test_two_threads_issue_on_their_own_contexts(device, other_context, tmp_path):
    """Each thread uploads (through its context's staging buffers), launches and reads back at once, many
    times; every result is its own."""
    import threading

    from resin_models_gaudi.synapse.graph import RecipeCache

    graph, (K, N, T) = _gemm(device, "test_contexts_gemm")
    with other_context.use():
        recipe_other = RecipeCache(device, tmp_path).compile(graph)
    graph, _ = _gemm(device, "test_contexts_gemm")
    recipe_default = RecipeCache(device, tmp_path).compile(graph)
    assert recipe_default.from_cache  # a recipe does not depend on its context
    rng = np.random.default_rng(5)
    w = bf16_round(rng.standard_normal((K, N), np.float32) / 16)
    w_bits = (w.view(np.uint32) >> 16).astype(np.uint16)
    w_dev = device.malloc(w.size * 2)
    device.h2d(w_dev, w_bits)
    big = rng.standard_normal(48 << 20 >> 2, np.float32)  # 48 MiB: both halves of the staging buffer
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def run(context, recipe, seed: int) -> None:
        try:
            with context.use():
                local = np.random.default_rng(seed)
                x_dev, y_dev = device.malloc(K * T * 2), device.malloc(N * T * 4)
                big_dev = device.malloc(big.nbytes)
                recipe.bind_all({"x": x_dev, "w": w_dev, "y": y_dev})
                barrier.wait()
                for round in range(40):
                    x = bf16_round(local.standard_normal((T, K), np.float32))
                    device.h2d_nb(x_dev, (x.view(np.uint32) >> 16).astype(np.uint16))
                    recipe.launch()
                    out = device.d2h(y_dev, (T, N), np.float32)
                    np.testing.assert_allclose(out, bf16_round(x @ w), rtol=2e-2, atol=2e-2)
                    if round % 10 == 0:
                        device.h2d(big_dev, big * seed)
                        assert np.array_equal(device.d2h(big_dev, big.shape, np.float32), big * seed)
                for address in (x_dev, y_dev, big_dev):
                    device.free(address)
        except BaseException as error:  # noqa: BLE001 - reported by the main thread
            failures.append(error)
            barrier.abort()

    threads = [
        threading.Thread(target=run, args=(device.default_context, recipe_default, 1)),
        threading.Thread(target=run, args=(other_context, recipe_other, 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    device.free(w_dev)
    assert not failures, failures


def test_a_context_waits_on_the_device_for_another_contexts_event(device, other_context, tmp_path):
    """The default context writes a buffer after a long run of launches; the other context copies it out after a
    stream wait on the default context's event, with no host wait in between."""
    from resin_models_gaudi.synapse.graph import RecipeCache

    graph, (K, N, T) = _gemm(device, "test_contexts_gemm")
    recipe = RecipeCache(device, tmp_path).compile(graph)
    x_dev, w_dev, y_dev = (device.malloc(n) for n in (K * T * 2, N * K * 2, N * T * 4))
    device.memset32(x_dev, 0, K * T // 2)
    device.memset32(w_dev, 0, N * K // 2)
    recipe.bind_all({"x": x_dev, "w": w_dev, "y": y_dev})
    words = 1 << 20
    source, target, copy = (device.malloc(words * 4) for _ in range(3))
    device.memset32(source, 7, words)
    device.memset32(target, 1, words)
    with other_context.use():
        device.memset32(copy, 0, words)
    device.sync_all()
    slot = device.event_slots(1)
    for _ in range(400):  # a few ms of device work before the write
        recipe.launch()
    device.d2d(target, source, words * 4)
    device.record(slot)
    with other_context.use():
        device.stream_wait(slot)
        device.d2d(copy, target, words * 4)
        out = device.d2h(copy, words, np.uint32)  # waits for the other context only
    assert (out == 7).all()
    device.sync()
    for address in (x_dev, w_dev, y_dev, source, target, copy):
        device.free(address)
