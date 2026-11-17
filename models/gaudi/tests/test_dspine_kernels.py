"""The DSpine drafter's kernels on one card against numpy: the code bridge's read, codes and
messages, the serial refinement's greedy walk and the causal block's slot softmax."""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
from test_dflash_kernels import NINF, Run, bf16, pytestmark, sigma_pack  # noqa: F401

H = 1024
EPS = 1e-6
BLOCK = 16
R = 512  # the bridge rank dsp_select implements


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
def recipes(device, tmp_path_factory):
    from resin_models_gaudi.synapse.graph import RecipeCache

    return RecipeCache(device, tmp_path_factory.mktemp("recipes"))


def sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_dsp_read(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    T = 2 * BLOCK
    rng = np.random.default_rng(11)
    x = rng.standard_normal((T, H)).astype(np.float32) * 3
    y = rng.standard_normal((T, H)).astype(np.float32)
    ys = rng.uniform(0.1, 2.0, (T, 1)).astype(np.float32)
    g = Graph(device, "t_dsp_read")
    g.node(
        "rg_dsp_read_f32",
        [
            g.input("x", (H, T), F32),
            g.input("y", (H, T), F32),
            g.input("ys", (1, T), F32),
        ],
        [
            g.input("raw", (H, T), F32),
            g.input("nb", (H, T), BF16),
            g.input("ws", (1, T), F32),
        ],
        struct.pack("fi", EPS, BLOCK),
    )
    r = Run(device, recipes, g)({"x": x, "y": y, "ys": ys}, ["raw", "nb", "ws"])
    raw = x + y * ys
    np.testing.assert_allclose(r["raw"], raw, rtol=1e-6, atol=1e-5)
    rms = np.sqrt((raw * raw).mean(-1))
    want_ws = np.where(np.arange(T) % BLOCK == 0, 0.0, rms)
    np.testing.assert_allclose(r["ws"][:, 0], want_ws, rtol=1e-5)
    np.testing.assert_allclose(
        bf16(r["nb"]), sigma_pack(raw / (rms[:, None] + EPS)), rtol=1e-2, atol=1e-3
    )


def test_dsp_codes(device, recipes):
    from resin_models_gaudi.loading.convert import bf16_rne
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    B, C, V = 3, 256, 4096
    T = B * BLOCK
    rng = np.random.default_rng(12)
    proj = rng.standard_normal((T, C)).astype(np.float32) * 5
    codebook = bf16_rne(rng.standard_normal((V, C)).astype(np.float32) * 0.05)
    anchors = rng.integers(0, V, B).astype(np.int32)
    g = Graph(device, "t_dsp_codes")
    g.node(
        "rg_dsp_codes_f32",
        [
            g.input("proj", (C, T), F32),
            g.input("codebook", (C, V), BF16),
            g.input("anchors", (B,), I32),
        ],
        [g.input("codes", (C, T), BF16)],
        struct.pack("fi", EPS, BLOCK),
    )
    r = Run(device, recipes, g)(
        {"proj": proj, "codebook": codebook, "anchors": anchors}, ["codes"]
    )
    want = np.sqrt(C) * proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + EPS)
    want[::BLOCK] = np.sqrt(C) * bf16(codebook[anchors])
    np.testing.assert_allclose(bf16(r["codes"]), sigma_pack(want), rtol=1e-2, atol=1e-3)


def test_dsp_msg(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    T = 2 * BLOCK
    rng = np.random.default_rng(13)
    pk = rng.standard_normal((T, 3 * R)).astype(np.float32) * 2
    bias = rng.standard_normal(R).astype(np.float32)
    g = Graph(device, "t_dsp_msg")
    g.node(
        "rg_dsp_msg_f32",
        [g.input("pk", (3 * R, T), F32), g.input("bias", (R,), F32)],
        [g.input("msg", (R, T), BF16)],
        struct.pack("i", BLOCK),
    )
    r = Run(device, recipes, g)({"pk": pk, "bias": bias}, ["msg"])
    q, s, v = pk[:, :R], pk[:, R : 2 * R], pk[:, 2 * R :]
    want = np.zeros((T, R), np.float32)
    want[1:] = sig(q[1:] + s[:-1] + bias) * v[:-1]
    want[::BLOCK] = 0.0
    np.testing.assert_allclose(bf16(r["msg"]), sigma_pack(want), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("Tq", [BLOCK, 4])
def test_dsp_select(device, recipes, Tq):
    """The greedy walk over the block's candidates with the redone last write, writing the
    anchor and the first Tq - 1 choices of each row."""
    from resin_models_gaudi.loading.convert import bf16_rne
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    B, K, V = 3, 16, 4096
    T = BLOCK * B
    rng = np.random.default_rng(14)
    top_v = rng.standard_normal((T, K)).astype(np.float32) * 2
    top_i = np.stack([rng.choice(V, K, replace=False) for _ in range(T)]).astype(
        np.int32
    )
    x = rng.standard_normal((T, H)).astype(np.float32) * 2
    ws = rng.uniform(0.5, 2.0, (T, 1)).astype(np.float32)
    pk = rng.standard_normal((T, 3 * R)).astype(np.float32)
    bias = rng.standard_normal(R).astype(np.float32) * 0.5
    anchors = rng.integers(0, V, B).astype(np.int32)
    tg = bf16_rne(rng.standard_normal((V, R)).astype(np.float32) * 0.2)
    tsv = bf16_rne(rng.standard_normal((V, 2 * R)).astype(np.float32))
    g = Graph(device, f"t_dsp_select_{Tq}")
    g.node(
        "rg_dsp_select_f32",
        [
            g.input("top_v", (K, T), F32),
            g.input("top_i", (K, T), I32),
            g.input("x", (H, T), F32),
            g.input("ws", (1, T), F32),
            g.input("pk", (3 * R, T), F32),
            g.input("bias", (R,), F32),
            g.input("anchors", (B,), I32),
            g.input("tg", (R, V), BF16),
            g.input("tsv", (2 * R, V), BF16),
        ],
        [g.input("next", (Tq, B), I32)],
        struct.pack("if", BLOCK, EPS),
    )
    r = Run(device, recipes, g)(
        {
            "top_v": top_v,
            "top_i": top_i,
            "x": x,
            "ws": ws,
            "pk": pk,
            "bias": bias,
            "anchors": anchors,
            "tg": tg,
            "tsv": tsv,
        },
        ["next"],
    )
    tgf, tsvf = bf16(tg), bf16(tsv)
    for b in range(B):
        prev = int(anchors[b])
        assert r["next"][b, 0] == prev
        for k in range(1, Tq):
            col = BLOCK * b + k
            ratio = np.sqrt((x[col] ** 2).mean() + EPS) / ws[col, 0]
            q = pk[col, :R] + bias
            mp = sig(q + pk[col - 1, R : 2 * R]) * pk[col - 1, 2 * R :]
            m = sig(q + tsvf[prev, :R]) * tsvf[prev, R:]
            scores = ratio * top_v[col] + tgf[top_i[col]] @ (m - mp)
            order = np.argsort(-scores)
            best = int(top_i[col, order[0]])
            if (
                scores[order[0]] - scores[order[1]] < 1e-3
            ):  # a near tie: follow the device
                best = int(r["next"][b, k])
                assert best in (top_i[col, order[0]], top_i[col, order[1]])
            assert r["next"][b, k] == best, (b, k)
            prev = best


def test_attn_slot_pc(device, recipes):
    """The causal block: column m (token m // hpg) sees the block's keys up to its own
    position; context chunks as attn_slot_pd's (window start, window)."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    C, HPG, S = 2048, 8, 4
    M = HPG * BLOCK
    scale = 1.0 / 128**0.5
    rng = np.random.default_rng(15)
    sc = rng.standard_normal((S, M, C)).astype(np.float32) * 4
    # [pair, qcol0, qpos0, key0, klen, 0, kmin, window]
    slots = np.array(
        [
            [0, 0, 10000, 8192, 10000, 0, 1809, 8192],  # a context chunk
            [1, 0, 10000, 0, 10000, 0, 1809, 8192],  # before the window
            [2, 0, 10000, 10000, 10000 + BLOCK, 0, 10000, 8192],  # the block, causal
            [-1, 0, 0, 0, 0, 0, 0, 0],  # padding
        ],
        np.int32,
    )
    g = Graph(device, "t_attn_slot_pc")
    g.node(
        "rg_attn_slot_pc_f32",
        [g.input("sc", (C, M, S), F32), g.input("slots", (8, S), I32)],
        [
            g.input("P", (C, M, S), BF16),
            g.input("mx", (M, S), F32),
            g.input("l", (M, S), F32),
        ],
        struct.pack("fi", scale, HPG),
    )
    r = Run(device, recipes, g)({"sc": sc, "slots": slots}, ["P", "mx", "l"])
    for s in range(3):
        _, _, qpos0, key0, klen, _, kmin, win = slots[s]
        for m in range(M):
            t = m // HPG
            lim = min(max(min(klen, qpos0 + t + 1) - key0, 0), C)
            lo = max(kmin - key0, qpos0 + t - win + 1 - key0, 0)
            visible = np.zeros(C, bool)
            visible[lo:lim] = True
            mx = (sc[s, m][visible] * scale).max() if visible.any() else NINF * scale
            p = np.where(visible, np.exp(sc[s, m] * scale - mx), 0.0)
            assert r["mx"][s, m] == pytest.approx(mx, rel=1e-5), (s, m)
            assert r["l"][s, m] == pytest.approx(p.sum(), rel=2e-3, abs=1e-6)
            got = bf16(r["P"][s, m])
            np.testing.assert_allclose(
                got, sigma_pack(p.astype(np.float32)), rtol=2e-2, atol=2e-3
            )


@pytest.mark.parametrize("V", [248320, 4096 + 300])
def test_two_stage_top_k(device, recipes, V):
    """dfl_tk1 + dfl_tk2: each column's 16 largest logits and their ids, as a set, against
    numpy; a column of ties and a column whose maxima share chunks."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    T, K = 6, 16
    rng = np.random.default_rng(16)
    x = rng.standard_normal((T, V)).astype(np.float32)
    x[1, :] = 1.0  # all equal
    x[2, 2 * 1024 + 64 * np.arange(16) + 7] += 20  # 16 maxima in one chunk
    x[3, -40:] += 30  # maxima in the last, partial group
    g = Graph(device, f"t_dfl_top_k_{V}")
    xs = g.input("x", (V, T), F32)
    keys = g.tensor((K, T), I32)
    g.node("rg_dfl_tk1_f32", [xs], [keys])
    g.node(
        "rg_dfl_tk2_f32",
        [xs, keys],
        [g.input("top_v", (K, T), F32), g.input("top_i", (K, T), I32)],
    )
    r = Run(device, recipes, g)({"x": x}, ["top_v", "top_i"])
    for t in range(T):
        ids = r["top_i"][t]
        assert len(set(ids.tolist())) == K and (0 <= ids).all() and (ids < V).all()
        np.testing.assert_array_equal(r["top_v"][t], x[t, ids])
        want = np.sort(x[t])[::-1][:K]
        np.testing.assert_array_equal(np.sort(r["top_v"][t])[::-1], want)


@pytest.mark.parametrize(
    "old, new, M, hpg",
    [
        ("rg_attn_slot_p_f32", "rg_attn_slot_p1_f32", 96, 6),
        ("rg_attn_slot_p_f32", "rg_attn_slot_p1_f32", 6, 6),
        ("rg_attn_slot_pc_f32", "rg_attn_slot_pc1_f32", 128, 8),
        ("rg_attn_slot_pd_f32", "rg_attn_slot_pd1_f32", 64, 8),
    ],
)
def test_per_column_softmax_builds(device, recipes, old, new, M, hpg):
    """The one-member-per-column softmax builds give their originals' results bit for bit
    (context chunks, a window start inside a chunk, the block, padding)."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    C, S = 2048, 6
    rng = np.random.default_rng(18)
    sc = rng.standard_normal((S, M, C)).astype(np.float32) * 4
    q = M // hpg
    slots = np.array(
        [
            [0, 0, 10000, 8192, 10000, 0, 1809, 8192],
            [1, 0, 10000, 0, 10000, 0, 1809, 8192],
            [2, 0, 10000, 10000, 10000 + q, 0, 10000, 8192],
            [3, 0, 3000, 2048, 3000 + q, 0, 0, 0],
            [0, 0, 100, 0, 100 + q, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0],
        ],
        np.int32,
    )
    got = {}
    Mp = (M + 63) // 64 * 64
    for kernel in (old, new):
        g = Graph(device, f"t_{kernel}_{M}")
        g.node(
            kernel,
            [g.input("sc", (C, M, S), F32), g.input("slots", (8, S), I32)],
            [
                g.input("P", (C, M, S), BF16),
                g.input("mx", (Mp, S), F32),
                g.input("l", (Mp, S), F32),
            ],
            struct.pack("fi", 1.0 / 16.0, hpg),
        )
        got[kernel] = Run(device, recipes, g)(
            {"sc": sc, "slots": slots}, ["P", "mx", "l"]
        )
    for name in ("P", "mx", "l"):
        a, b = got[old][name], got[new][name]
        if name != "P":
            a, b = a[:, :M], b[:, :M]
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), name
