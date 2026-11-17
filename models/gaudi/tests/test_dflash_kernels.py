"""The DFlash 2 drafter's kernels on one card against numpy: the natural-order norm, the grouped
dynamic convolution (both sides), the paired QKV prologue, the attention epilogue, the candidate
selector and the block's slot softmax (window start, window, bidirectional block)."""

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

H = 1024  # a hidden size the kernels take (a multiple of 1024)
EPS = 1e-6
BLOCK = 8
PAIRS = 4
NINF = -3.0e38


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


class Run:
    """Compile a graph, bind numpy inputs and output buffers by name, launch once."""

    def __init__(self, device, recipes, graph) -> None:
        self.device, self.graph = device, graph
        self.recipe = recipes.compile(graph)

    def __call__(self, inputs: dict[str, np.ndarray], outputs) -> dict[str, np.ndarray]:
        from resin_models_gaudi.synapse.runtime import NUMPY_DTYPE

        d, allocated = self.device, []
        try:
            for name, array in inputs.items():
                a = np.ascontiguousarray(array)
                address = d.malloc(max(a.nbytes, 256))
                allocated.append(address)
                d.h2d(address, a)
                self.recipe.bind(name, address)
            outs = {}
            for name in outputs:
                t = self.recipe.persistent[name]
                address = d.malloc(max(t.nbytes, 256))
                allocated.append(address)
                d.memset32(address, 0, max(t.nbytes, 256) // 4)
                self.recipe.bind(name, address)
                outs[name] = (address, t)
            self.recipe.launch()
            d.sync()
            return {
                name: d.d2h(a, tuple(reversed(t.shape)), NUMPY_DTYPE[t.dtype])
                for name, (a, t) in outs.items()
            }
        finally:
            for address in allocated:
                d.free(address)


# ------------------------------------------------------------------ numpy helpers
def bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def sigma_pack(x: np.ndarray) -> np.ndarray:
    """natural f32 (..., n) -> the stored order of ``pack_sigma`` per 128 block"""
    from resin_models_gaudi.loading.convert import sigma_perm

    return x[..., sigma_perm(x.shape[-1])]


def tau_unpack(q: np.ndarray) -> np.ndarray:
    """stored FP8 bytes (..., K) in tau order -> natural order"""
    from resin_models_gaudi.loading.convert import tau_perm

    out = np.empty_like(q)
    out[..., tau_perm(q.shape[-1])] = q
    return out


def fp8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """per-token FP8 (T, K) tau order with scale (T, 1) -> f32 natural"""
    from resin_models_gaudi.loading.convert import E4M3

    return E4M3[tau_unpack(q)] * scale


def close_fp8(got: np.ndarray, want: np.ndarray, scale: np.ndarray) -> None:
    """within one e4m3 rounding step of the per-token scale"""
    assert np.all(np.abs(got - want) <= np.abs(want) / 16 + scale / 2 + 1e-6)


def rms(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    a, b = x[..., :64], x[..., 64:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)


# ------------------------------------------------------------------ norm
def test_dfl_norm(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    T = 5
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, H)).astype(np.float32)
    y = rng.standard_normal((T, H)).astype(np.float32)
    ys = rng.uniform(0.5, 2, (T, 1)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, H).astype(np.float32)
    for mode in (0, 1):
        g = Graph(device, f"t_dfl_norm_{mode}")
        ins = [
            g.input("x", (H, T), F32),
            g.input("y", (H, T), F32),
            g.input("ys", (1, T), F32),
            g.input("w", (H,), F32),
        ]
        outs = [
            g.input("xo", (H, T), F32),
            g.input("hn", (H, T), F32),
            g.input("h16", (H, T), BF16),
        ]
        g.node("rg_dfl_norm_f32", ins, outs, struct.pack("fi", EPS, mode))
        r = Run(device, recipes, g)(
            {"x": x, "y": y, "ys": ys, "w": w}, ["xo", "hn", "h16"]
        )
        xo = x + y * ys if mode else x
        hn = rms(xo, w)
        np.testing.assert_allclose(r["xo"], xo, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(r["hn"], hn, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(bf16(r["h16"]), sigma_pack(hn), rtol=1e-2, atol=1e-2)


# ------------------------------------------------------------------ convolution
def conv_model(x, dyn, ds, base, side, block, xs=None):
    T, Hh = x.shape
    GR = Hh // 16
    inp = x * xs if xs is not None else x
    prev = np.zeros_like(inp)
    prev[1:] = inp[:-1]
    prev[np.arange(T) % block == 0] = 0
    out = np.zeros_like(inp)
    for tap, src in ((0, inp), (1, prev)):
        k = 2 * side + tap
        coef = base[k][None, :] + np.repeat(
            dyn[:, k * GR : (k + 1) * GR] * ds, 16, axis=1
        )
        out += coef * src
    return out


def test_dfl_conv_sides(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32

    T, GR = 19, H // 16
    rng = np.random.default_rng(1)
    x = rng.standard_normal((T, H)).astype(np.float32)
    xs = rng.uniform(0.01, 0.1, (T, 1)).astype(np.float32)
    dyn = rng.standard_normal((T, 4 * GR)).astype(np.float32)
    ds = rng.uniform(0.01, 0.1, (T, 1)).astype(np.float32)
    base = rng.standard_normal((4, H)).astype(np.float32)
    for side in (0, 1):
        # side 0: f32 in, FP8 out
        g = Graph(device, f"t_dfl_conv_q8_{side}")
        ins = [
            g.input("x", (H, T), F32),
            g.input("dyn", (4 * GR, T), F32),
            g.input("ds", (1, T), F32),
            g.input("base", (H, 4), F32),
        ]
        outs = [g.input("q", (H, T), F8), g.input("qs", (1, T), F32)]
        g.node("rg_dfl_conv_q8_f32", ins, outs, struct.pack("ii", side, BLOCK))
        r = Run(device, recipes, g)(
            {"x": x, "dyn": dyn, "ds": ds, "base": base}, ["q", "qs"]
        )
        want = conv_model(x, dyn, ds, base, side, BLOCK)
        np.testing.assert_allclose(
            r["qs"], np.abs(want).max(1, keepdims=True) / 240, rtol=1e-5
        )
        close_fp8(fp8(r["q"], r["qs"]), want, r["qs"])
        # side 1: a scaled FP8-GEMM output in, f32 out
        g = Graph(device, f"t_dfl_conv_s_{side}")
        ins = [
            g.input("x", (H, T), F32),
            g.input("xs", (1, T), F32),
            g.input("dyn", (4 * GR, T), F32),
            g.input("ds", (1, T), F32),
            g.input("base", (H, 4), F32),
        ]
        g.node(
            "rg_dfl_conv_s_f32",
            ins,
            [g.input("out", (H, T), F32)],
            struct.pack("ii", side, BLOCK),
        )
        r = Run(device, recipes, g)(
            {"x": x, "xs": xs, "dyn": dyn, "ds": ds, "base": base}, ["out"]
        )
        np.testing.assert_allclose(
            r["out"],
            conv_model(x, dyn, ds, base, side, BLOCK, xs),
            rtol=1e-4,
            atol=1e-5,
        )


# ------------------------------------------------------------------ QKV prologue
@pytest.mark.parametrize("with_q", [True, False])
@pytest.mark.parametrize("T", [8, 64])
def test_dfl_prep(device, recipes, with_q, T):
    """Including a 64-column step: the kernel's index space is one member per column, so its
    query columns must not be written by members beyond the real columns."""
    from resin_models_gaudi.loading.convert import sigma_inv
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    NH, NG = 32, 8
    Tv = 128
    R = (NH + 2 * NG) * 128 if with_q else 2 * NG * 128
    W = 8 * T
    rng = np.random.default_rng(2)
    p = rng.standard_normal((T, R)).astype(np.float32)
    ps = rng.uniform(0.01, 0.1, (T, 1)).astype(np.float32)
    qkn = rng.uniform(0.5, 1.5, 256).astype(np.float32)
    ang = rng.uniform(0, 6.28, (T, 64))
    cs = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    g = Graph(device, f"t_dfl_prep_{int(with_q)}_T{T}")
    ins = [
        g.input("p", (R, T), F32),
        g.input("ps", (1, T), F32),
        g.input("qkn", (256,), F32),
        g.input("cs", (128, T), F32),
    ]
    outs = []
    if with_q:
        outs.append(g.input("q_all", (256, W * PAIRS), BF16))
    outs += [g.input(f"kn{i}", (256, T), BF16) for i in range(PAIRS)]
    outs += [g.input(f"vn{i}", (256, Tv), BF16) for i in range(PAIRS)]
    g.node(
        "rg_dfl_prep_qkv_f32" if with_q else "rg_dfl_prep_kv_f32",
        ins,
        outs,
        struct.pack("f", EPS),
    )
    r = Run(device, recipes, g)(
        {"p": p, "ps": ps, "qkn": qkn, "cs": cs}, [o.name for o in outs]
    )
    scaled = p * ps
    cos, sin = cs[:, :64], cs[:, 64:]
    kbase = NH * 128 if with_q else 0
    if with_q:
        q_all = bf16(r["q_all"])  # (4 W, 256)
        for hd in range(NH):
            x = rope(rms(scaled[:, 128 * hd : 128 * hd + 128], qkn[:128]), cos, sin)
            pg, ph = hd // 8, hd % 8
            e = ph // 4
            for t in range(T):
                col = q_all[pg * W + ph + 8 * t]
                np.testing.assert_allclose(
                    col[128 * e : 128 * e + 128], sigma_pack(x[t]), rtol=2e-2, atol=2e-2
                )
                assert not col[128 * (1 - e) : 128 * (1 - e) + 128].any()
    for gi in range(NG):
        pg, e = gi // 2, gi % 2
        k = rope(
            rms(scaled[:, kbase + 128 * gi : kbase + 128 * gi + 128], qkn[128:]),
            cos,
            sin,
        )
        kn = bf16(r[f"kn{pg}"])  # (T, 256)
        np.testing.assert_allclose(
            kn[:, 128 * e : 128 * e + 128], sigma_pack(k), rtol=2e-2, atol=2e-2
        )
        v = scaled[:, kbase + NG * 128 + 128 * gi : kbase + NG * 128 + 128 * gi + 128]
        vn = bf16(r[f"vn{pg}"])  # (Tv, 256)
        for t in range(T):
            np.testing.assert_allclose(
                vn[sigma_inv(t), 128 * e : 128 * e + 128],
                sigma_pack(v[t]),
                rtol=2e-2,
                atol=2e-2,
            )


# ------------------------------------------------------------------ epilogue
def test_dfl_oq(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32

    T, NH = 6, 32
    W = 8 * T
    rng = np.random.default_rng(3)
    acc = rng.standard_normal((PAIRS, W, 256)).astype(np.float32)
    g = Graph(device, "t_dfl_oq")
    g.node(
        "rg_dfl_oq_f32",
        [g.input("acc_o", (256, W, PAIRS), F32)],
        [g.input("o8", (128 * NH, T), F8), g.input("os", (1, T), F32)],
    )
    r = Run(device, recipes, g)({"acc_o": acc}, ["o8", "os"])
    want = np.zeros((T, 128 * NH), np.float32)
    for hd in range(NH):
        pg, ph = hd // 8, hd % 8
        e = ph // 4
        for t in range(T):
            want[t, 128 * hd : 128 * hd + 128] = acc[
                pg, ph + 8 * t, 128 * e : 128 * e + 128
            ]
    np.testing.assert_allclose(
        r["os"], np.abs(want).max(1, keepdims=True) / 240, rtol=1e-5
    )
    close_fp8(fp8(r["o8"], r["os"]), want, r["os"])


# ------------------------------------------------------------------ selector
@pytest.mark.parametrize("Tq", [BLOCK, 4])
def test_dfl_select(device, recipes, Tq):
    """The greedy walk over the block's candidates, writing the anchor and the first Tq - 1
    choices of each row."""
    from resin_models_gaudi.loading.convert import bf16_rne
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    B, K, V, R = 3, 16, 4096, 256
    T = BLOCK * B
    rng = np.random.default_rng(4)
    top_v = rng.standard_normal((T, K)).astype(np.float32) * 3
    top_i = np.stack([rng.choice(V, K, replace=False) for _ in range(T)]).astype(
        np.int32
    )
    hp = rng.standard_normal((T, R)).astype(np.float32)
    anchors = rng.integers(0, V, B).astype(np.int32)
    pred = bf16_rne(rng.standard_normal((V, R)).astype(np.float32) * 0.3)
    succ = bf16_rne(rng.standard_normal((V, R)).astype(np.float32) * 0.3)
    g = Graph(device, f"t_dfl_select_{Tq}")
    g.node(
        "rg_dfl_select_f32",
        [
            g.input("top_v", (K, T), F32),
            g.input("top_i", (K, T), I32),
            g.input("hp", (R, T), F32),
            g.input("anchors", (B,), I32),
            g.input("pred", (R, V), BF16),
            g.input("succ", (R, V), BF16),
        ],
        [g.input("next", (Tq, B), I32)],
        struct.pack("i", BLOCK),
    )
    r = Run(device, recipes, g)(
        {
            "top_v": top_v,
            "top_i": top_i,
            "hp": hp,
            "anchors": anchors,
            "pred": pred,
            "succ": succ,
        },
        ["next"],
    )
    pf, sf = bf16(pred), bf16(succ)
    for b in range(B):
        prev = int(anchors[b])
        assert r["next"][b, 0] == prev
        for k in range(1, Tq):
            col = BLOCK * b + k
            scores = top_v[col] + (sf[top_i[col]] * (pf[prev] * hp[col])[None, :]).sum(
                -1
            )
            order = np.argsort(-scores)
            best = int(top_i[col, order[0]])
            if (
                scores[order[0]] - scores[order[1]] < 1e-3
            ):  # a near tie: follow the device
                best = int(r["next"][b, k])
                assert best in (top_i[col, order[0]], top_i[col, order[1]])
            assert r["next"][b, k] == best, (b, k)
            prev = best


# ------------------------------------------------------------------ slot softmax
def test_attn_slot_pd(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    C, M, HPG, S = 2048, 64, 8, 4
    scale = 1.0 / 128**0.5
    rng = np.random.default_rng(5)
    sc = rng.standard_normal((S, M, C)).astype(np.float32) * 4
    # [pair, qcol0, qpos0, key0, klen, 0, kmin, window]
    slots = np.array(
        [
            [
                0,
                0,
                2100,
                2048,
                2100,
                0,
                53,
                2048,
            ],  # a context chunk, cut by the window at 53..
            [
                1,
                0,
                2100,
                0,
                2100,
                0,
                53,
                2048,
            ],  # the first chunk: window start inside it
            [
                2,
                0,
                2100,
                2100,
                2108,
                0,
                2100,
                2048,
            ],  # the block: all 8 keys for every column
            [-1, 0, 0, 0, 0, 0, 0, 0],  # padding
        ],
        np.int32,
    )
    g = Graph(device, "t_attn_slot_pd")
    g.node(
        "rg_attn_slot_pd_f32",
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
            lim = min(max(klen - key0, 0), C)
            lo = max(kmin - key0, qpos0 + t - win + 1 - key0, 0)
            visible = np.zeros(C, bool)
            visible[lo:lim] = True
            mx = (sc[s, m][visible] * scale).max() if visible.any() else NINF * scale
            p = np.where(visible, np.exp(sc[s, m] * scale - mx), 0.0)
            assert r["mx"][s, m] == pytest.approx(mx, rel=1e-5)
            assert r["l"][s, m] == pytest.approx(p.sum(), rel=2e-3, abs=1e-6)
            got = bf16(r["P"][s, m])
            np.testing.assert_allclose(
                got, sigma_pack(p.astype(np.float32)), rtol=2e-2, atol=2e-3
            )
