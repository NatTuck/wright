"""Speculative GDN: ``gdn_verify`` (Tq tokens per sequence from the arena state, which it does not
write, recording each token's update) and ``gdn_commit`` (replaying the accepted tokens into the
arena), against NumPy and against ``gdn_decode`` stepping the same tokens.

Four sequences take three tokens each: in place, moved, fresh and skipped. The commit applies
0 .. 3 tokens per sequence.
"""

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from test_gdn_decode import BLOCK_ROWS, CONV_ROWS, NK, NV, SSM_ROWS, C, D, reference

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)

BLOCKS, B, TQ = 48, 4, 3
COUNTS = np.array([3, 1, 2, 0], np.int32)


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


def inputs(rng):
    cols = B * TQ
    return dict(
        p=(rng.standard_normal((cols, C + NV * D)) * 0.5).astype(np.float32),
        ps=(1 + 0.5 * rng.random(cols)).astype(np.float32),
        ba=rng.standard_normal((cols, 2 * NV)).astype(np.float32),
        cw=(rng.standard_normal((C, 4)) * 0.5).astype(np.float32),
        A=-np.exp(rng.standard_normal(NV)).astype(np.float32),
        dt=rng.standard_normal(NV).astype(np.float32),
        nw=(1 + 0.1 * rng.standard_normal(D)).astype(np.float32),
    )


# [conv_src, ssm_src, conv_dst, ssm_dst]: in place, moved, fresh, skipped (count 0)
SSM = [0, 6144, 12288, 18432, 24576]
CONV = [37 * BLOCK_ROWS + 784, 38912, 39200, 39500, 39936]
BASES = np.array(
    [
        [CONV[0], SSM[0], CONV[0], SSM[0]],
        [CONV[1], SSM[1], CONV[2], SSM[2]],
        [-1, -1, CONV[3], SSM[3]],
        [CONV[4], SSM[4], CONV[4], SSM[4]],
    ],
    np.int32,
)


def numpy_steps(x, arena0, b, n):
    """The state, conv state and outputs of sequence b after its first n tokens."""
    conv_src, ssm_src = BASES[b, :2]
    S = (
        arena0[ssm_src : ssm_src + SSM_ROWS].reshape(NV, D, D)
        if ssm_src >= 0
        else np.zeros((NV, D, D), np.float32)
    )
    cst = (
        arena0[conv_src : conv_src + CONV_ROWS].reshape(3, C)
        if conv_src >= 0
        else np.zeros((3, C), np.float32)
    )
    outs = []
    for t in range(n):
        col = b * TQ + t
        o, S, cst = reference(
            x["p"][col],
            x["ps"][col],
            x["ba"][col],
            x["cw"],
            cst,
            S,
            x["A"],
            x["dt"],
            x["nw"],
        )
        outs.append(o.reshape(-1))
    return S, cst, outs


def build(device, tmp_path):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, I32

    cache = RecipeCache(device, tmp_path)
    graph = Graph(device, "test_gdn_verify")
    arena = graph.input("arena", (D, BLOCK_ROWS, BLOCKS), F32)
    graph.node(
        "rg_gdn_verify_f32",
        [
            graph.input("p", (C + NV * D, B * TQ), F32),
            graph.input("ps", (1, B * TQ), F32),
            graph.input("ba", (2 * NV, B * TQ), F32),
            graph.input("cw", (C, 4), F32),
            arena,
            graph.input("bases", (4, B), I32),
            graph.input("A", (NV,), F32),
            graph.input("dt", (NV,), F32),
            graph.input("nw", (D,), F32),
        ],
        [
            graph.input("out", (NV * D, B * TQ), F32),
            graph.input("pend", (D, 3, NV, TQ, B), F32),
            graph.input("pconv", (C, TQ, B), F32),
        ],
        struct.pack("fi", 1e-6, NK),
    )
    verify = cache.compile(graph)
    graph = Graph(device, "test_gdn_commit")
    graph.node(
        "rg_gdn_commit_f32",
        [
            graph.input("arena", (D, BLOCK_ROWS, BLOCKS), F32),
            graph.input("pend", (D, 3, NV, TQ, B), F32),
            graph.input("pconv", (C, TQ, B), F32),
            graph.input("bases", (4, B), I32),
            graph.input("counts", (B,), I32),
        ],
        [graph.input("sink", (64, 1), I32)],
        struct.pack("i", NK),
    )
    commit = cache.compile(graph)
    return verify, commit


def test_verify_then_commit(device, tmp_path):
    rng = np.random.default_rng(9)
    x = inputs(rng)
    rows = BLOCKS * BLOCK_ROWS
    arena0 = rng.standard_normal((rows, D)).astype(np.float32) * 0.05
    verify, commit = build(device, tmp_path)
    host = {
        "arena": arena0,
        "p": x["p"],
        "ps": x["ps"],
        "ba": x["ba"],
        "cw": np.ascontiguousarray(x["cw"].T),
        "bases": BASES,
        "A": x["A"],
        "dt": x["dt"],
        "nw": x["nw"],
        "out": np.zeros((B * TQ, NV * D), np.float32),
        "pend": np.zeros((B, TQ, NV, 3, D), np.float32),
        "pconv": np.zeros((B, TQ, C), np.float32),
        "counts": COUNTS,
        "sink": np.zeros(64, np.int32),
    }
    address = {name: device.malloc(max(256, a.nbytes)) for name, a in host.items()}
    for name, a in host.items():
        device.h2d(address[name], a)
    verify.bind_all({k: v for k, v in address.items() if k in verify.persistent})
    verify.launch()
    out = device.d2h(address["out"], (B * TQ, NV * D), np.float32)
    untouched = device.d2h(address["arena"], (rows, D), np.float32)
    commit.bind_all({k: v for k, v in address.items() if k in commit.persistent})
    commit.launch()
    arena = device.d2h(address["arena"], (rows, D), np.float32)
    for a in address.values():
        device.free(a)

    np.testing.assert_array_equal(untouched, arena0)
    expected = arena0.copy()
    for b in range(B):
        _, _, outs = numpy_steps(x, arena0, b, TQ)
        for t in range(TQ):
            got = out[b * TQ + t]
            rel = np.abs(got - outs[t]).max() / np.abs(outs[t]).max()
            assert rel < 1e-4, (b, t, rel)
        if COUNTS[b]:
            S, cst, _ = numpy_steps(x, arena0, b, int(COUNTS[b]))
            conv_dst, ssm_dst = BASES[b, 2], BASES[b, 3]
            expected[ssm_dst : ssm_dst + SSM_ROWS] = S.reshape(SSM_ROWS, D)
            expected[conv_dst : conv_dst + CONV_ROWS] = cst.reshape(CONV_ROWS, D)
    np.testing.assert_allclose(arena, expected, rtol=1e-4, atol=1e-5)


def test_commit_equals_decode_steps(device, tmp_path):
    """Replaying k accepted tokens gives gdn_decode's state after stepping them, bit for bit."""
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(10)
    x = inputs(rng)
    rows = BLOCKS * BLOCK_ROWS
    arena0 = rng.standard_normal((rows, D)).astype(np.float32) * 0.05
    verify, commit = build(device, tmp_path)
    graph = Graph(device, "test_gdn_decode_step")
    arena_t = graph.input("arena", (D, BLOCK_ROWS, BLOCKS), F32)
    cst = graph.tensor((C, 3, B), F32)
    graph.node(
        "rg_gdn_decode_f32",
        [
            graph.input("p1", (C + NV * D, B), F32),
            graph.input("ps1", (1, B), F32),
            graph.input("ba1", (2 * NV, B), F32),
            graph.input("cw", (C, 4), F32),
            arena_t,
            graph.input("bases1", (4, B), I32),
            graph.input("A", (NV,), F32),
            graph.input("dt", (NV,), F32),
            graph.input("nw", (D,), F32),
        ],
        [graph.input("out1", (NV * D, B), F32), cst],
        struct.pack("fi", 1e-6, NK),
    )
    cst_rows = graph.tensor((D, CONV_ROWS, B), F32)
    graph.node("reshape", [cst], [cst_rows])
    graph.node(
        "rg_state_store_f32",
        [arena_t, graph.input("conv_dst", (B,), I32), cst_rows],
        [graph.input("sink", (64, 1), I32)],
    )
    step = RecipeCache(device, tmp_path).compile(graph)

    full = np.full(B, TQ, np.int32)
    in_place = BASES.copy()
    in_place[:, 2:] = in_place[:, :2]
    in_place[2] = [CONV[3], SSM[3], CONV[3], SSM[3]]  # every entry updates in place
    host = {
        "arena": arena0,
        "p": x["p"],
        "ps": x["ps"],
        "ba": x["ba"],
        "cw": np.ascontiguousarray(x["cw"].T),
        "bases": in_place,
        "A": x["A"],
        "dt": x["dt"],
        "nw": x["nw"],
        "out": np.zeros((B * TQ, NV * D), np.float32),
        "pend": np.zeros((B, TQ, NV, 3, D), np.float32),
        "pconv": np.zeros((B, TQ, C), np.float32),
        "counts": full,
        "sink": np.zeros(64, np.int32),
        "p1": np.zeros((B, C + NV * D), np.float32),
        "ps1": np.zeros(B, np.float32),
        "ba1": np.zeros((B, 2 * NV), np.float32),
        "bases1": in_place,
        "conv_dst": np.ascontiguousarray(in_place[:, 2]),
        "out1": np.zeros((B, NV * D), np.float32),
    }
    address = {name: device.malloc(max(256, a.nbytes)) for name, a in host.items()}
    for name, a in host.items():
        device.h2d(address[name], a)
    for recipe in (verify, commit):
        recipe.bind_all({k: v for k, v in address.items() if k in recipe.persistent})
    verify.launch()
    commit.launch()
    committed = device.d2h(address["arena"], (rows, D), np.float32)
    device.h2d(address["arena"], arena0)
    step.bind_all({k: v for k, v in address.items() if k in step.persistent})
    for t in range(TQ):
        cols = np.arange(B) * TQ + t
        device.h2d(address["p1"], x["p"][cols])
        device.h2d(address["ps1"], x["ps"][cols])
        device.h2d(address["ba1"], x["ba"][cols])
        step.launch()
    stepped = device.d2h(address["arena"], (rows, D), np.float32)
    for a in address.values():
        device.free(a)
    np.testing.assert_array_equal(committed, stepped)
