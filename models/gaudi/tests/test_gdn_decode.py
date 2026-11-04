"""Batched GDN decode (``gdn_decode``) with recurrent state in the cache arena, against numpy.

Five sequences take one token each: one updates its state in place, one moves it to new rows,
one starts fresh, one is padding (nothing stored) and one reads the moved sequence's source
state (a shared prefix). The conv state goes through ``state_store``.
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

ROW, BLOCK_ROWS, D = 512, 1024, 128
NK, NV = 16, 48
C = (2 * NK + NV) * D
SSM_ROWS, CONV_ROWS = NV * D, C * 3 // D
BLOCKS, B = 48, 5


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


def silu(x):
    return x / (1 + np.exp(-x))


def reference(p, ps, ba, cw, cst, S0, A, dt, nw, eps=1e-6):
    """One token of one sequence: out (NV, D), state, conv state."""
    x = p * ps
    xin = np.concatenate([cst, x[None, :C]], 0).astype(np.float64)
    conv = silu(sum(xin[k] * cw[:, k] for k in range(4)))
    q = conv[: NK * D].reshape(NK, D)
    k = conv[NK * D : 2 * NK * D].reshape(NK, D)
    v = conv[2 * NK * D :].reshape(NV, D)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-6) / np.sqrt(D)
    k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
    hk = np.arange(NV) % NK
    q, k = q[hk], k[hk]
    beta = 1 / (1 + np.exp(-ba[:NV]))
    eg = np.exp(A * np.log1p(np.exp(ba[NV:] + dt)))
    S = S0.astype(np.float64) * eg[:, None, None]
    d = (v - np.einsum("hij,hi->hj", S, k)) * beta[:, None]
    S += k[:, :, None] * d[:, None, :]
    o = np.einsum("hij,hi->hj", S, q)
    z = x[C : C + NV * D].reshape(NV, D)
    out = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps) * nw * silu(z)
    return out, S, xin[1:4]


def test_gdn_decode(device, tmp_path):
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(7)
    p = (rng.standard_normal((B, C + NV * D)) * 0.5).astype(np.float32)
    ps = (1 + 0.5 * rng.random(B)).astype(np.float32)
    ba = rng.standard_normal((B, 2 * NV)).astype(np.float32)
    cw = (rng.standard_normal((C, 4)) * 0.5).astype(np.float32)
    A = -np.exp(rng.standard_normal(NV)).astype(np.float32)
    dt = rng.standard_normal(NV).astype(np.float32)
    nw = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)

    rows = BLOCKS * BLOCK_ROWS
    arena0 = rng.standard_normal((rows, D)).astype(np.float32) * 0.05
    # [conv_src, ssm_src, conv_dst, ssm_dst] per sequence: in place, moved, fresh, padding,
    # shared source with the moved one
    ssm = [0, 6144, 12288, 18432, 24576, 30720]
    conv = [37 * BLOCK_ROWS + 784, 38912, 39200, 39500, 39936, 40200]
    bases = np.array(
        [
            [conv[0], ssm[0], conv[0], ssm[0]],
            [conv[1], ssm[1], conv[2], ssm[2]],
            [-1, -1, conv[3], ssm[3]],
            [conv[4], ssm[4], -1, -1],
            [conv[1], ssm[1], conv[5], ssm[5]],
        ],
        np.int32,
    )
    assert (bases[:, [1, 3]] % D == 0)[bases[:, [1, 3]] >= 0].all()
    assert (
        (bases[:, [0, 2]] % BLOCK_ROWS + CONV_ROWS <= BLOCK_ROWS)
        | (bases[:, [0, 2]] < 0)
    ).all()

    graph = Graph(device, "test_gdn_decode")
    inputs = [
        graph.input("p", (C + NV * D, B), F32),
        graph.input("ps", (1, B), F32),
        graph.input("ba", (2 * NV, B), F32),
        graph.input("cw", (C, 4), F32),
        graph.input("arena", (D, BLOCK_ROWS, BLOCKS), F32),
        graph.input("bases", (4, B), I32),
        graph.input("A", (NV,), F32),
        graph.input("dt", (NV,), F32),
        graph.input("nw", (D,), F32),
    ]
    out = graph.input("out", (NV * D, B), F32)
    cst = graph.tensor((C, 3, B), F32)
    graph.node("rg_gdn_decode_f32", inputs, [out, cst], struct.pack("fi", 1e-6, NK))
    cst_rows = graph.tensor((D, CONV_ROWS, B), F32)
    graph.node("reshape", [cst], [cst_rows])
    graph.node(
        "rg_state_store_f32",
        [inputs[4], graph.input("conv_dst", (B,), I32), cst_rows],
        [graph.input("sink", (64, 1), I32)],
    )
    recipe = RecipeCache(device, tmp_path).compile(graph)

    host = {
        "p": p,
        "ps": ps,
        "ba": ba,
        "cw": np.ascontiguousarray(cw.T),
        "arena": arena0,
        "bases": bases,
        "A": A,
        "dt": dt,
        "nw": nw,
        "conv_dst": np.ascontiguousarray(bases[:, 2]),
        "out": np.zeros((B, NV * D), np.float32),
        "sink": np.zeros(64, np.int32),
    }
    address = {name: device.malloc(max(256, a.nbytes)) for name, a in host.items()}
    for name, a in host.items():
        device.h2d(address[name], a)
    recipe.bind_all(address)
    recipe.launch()
    got_out = device.d2h(address["out"], (B, NV * D), np.float32)
    arena = device.d2h(address["arena"], (rows, D), np.float32)
    for a in address.values():
        device.free(a)

    expected = arena0.copy()
    for b in range(B):
        conv_src, ssm_src, conv_dst, ssm_dst = bases[b]
        S0 = (
            arena0[ssm_src : ssm_src + SSM_ROWS].reshape(NV, D, D)
            if ssm_src >= 0
            else np.zeros((NV, D, D), np.float32)
        )
        c0 = (
            arena0[conv_src : conv_src + CONV_ROWS].reshape(3, C)
            if conv_src >= 0
            else np.zeros((3, C), np.float32)
        )
        o, S, c1 = reference(p[b], ps[b], ba[b], cw, c0, S0, A, dt, nw)
        rel = np.abs(got_out[b] - o.reshape(-1)).max() / np.abs(o).max()
        assert rel < 1e-4, (b, rel)
        if ssm_dst >= 0:
            expected[ssm_dst : ssm_dst + SSM_ROWS] = S.reshape(SSM_ROWS, D)
        if conv_dst >= 0:
            expected[conv_dst : conv_dst + CONV_ROWS] = c1.reshape(CONV_ROWS, D)
    np.testing.assert_allclose(arena, expected, rtol=1e-4, atol=1e-5)
