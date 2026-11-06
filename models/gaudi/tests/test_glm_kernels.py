"""GLM-5.3-Flash kernels on one card against numpy: the FP8 load path and routed-expert slots, batched KDA decode
over the arena, the mHC chain, the paged DSA pools."""

from __future__ import annotations

import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

GEMM_NN = struct.pack("??", False, False)


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

    def __call__(self, inputs: dict[str, np.ndarray], outputs: list[str], keep=(), stale=None):
        """``stale``: initial contents of the output buffers"""
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
                if stale and name in stale:
                    d.h2d(address, np.ascontiguousarray(stale[name]))
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


def e4m3(b: np.ndarray) -> np.ndarray:
    from resin_models_gaudi.qwen3_5.checkpoint import E4M3

    return E4M3[b]


def fp8_inc_bytes(n: int, rng) -> np.ndarray:
    """e4m3 bytes without NaN"""
    b = rng.integers(0, 256, n, dtype=np.uint8)
    return np.where((b & 0x7F) == 0x7F, b - 1, b).astype(np.uint8)


# ------------------------------------------------------------------ the FP8 load path
def test_fp8_inc_matches_host_table(device, recipes):
    from resin_models_gaudi.qwen3_5.checkpoint import HALVE
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import U8

    g = Graph(device, "t_fp8_inc")
    g.node("rg_fp8_inc_f32", [g.input("w", (256, 2), U8)], [g.input("o", (256, 2), U8)])
    w = np.arange(512, dtype=np.int64).astype(np.uint8).reshape(2, 256)
    out = Run(device, recipes, g)({"w": w}, ["o"])["o"]
    assert np.array_equal(out, HALVE[w])


def test_x8_requant_row_scales(device, recipes):
    """Requantized FP8 with per-row scales equals the input value within FP8's rounding"""
    from resin_models_gaudi.qwen3_5.checkpoint import HALVE
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32

    rng = np.random.default_rng(1)
    E, K, M = 3, 256, 512
    raw = fp8_inc_bytes(E * K * M, rng).reshape(E, K, M)
    inc = HALVE[raw]
    bs = rng.uniform(1e-3, 1e-1, (E, K // 128, M // 128)).astype(np.float32)
    g = Graph(device, "t_x8_requant")
    g.node(
        "rg_x8_requant_f32",
        [g.input("w", (M, K, E), F8), g.input("bs", (M // 128, K // 128, E), F32)],
        [g.input("o", (M, K, E), F8), g.input("c", (M, E), F32)],
    )
    r = Run(device, recipes, g)({"w": inc, "bs": bs}, ["o", "c"])
    want = e4m3(inc) * np.repeat(np.repeat(bs, 128, 1), 128, 2)
    got = e4m3(r["o"]) * r["c"][:, None, :]
    assert np.abs(want).max(1) == pytest.approx(r["c"] * 240, rel=1e-5)
    # within one e4m3 rounding step
    tol = np.maximum(np.abs(want) * 0.07, r["c"][:, None, :] * 2.0**-9 * 1.01)
    assert (np.abs(got - want) <= tol).all()


def test_x8_copy_slots(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, I32

    rng = np.random.default_rng(2)
    E, K, M, R, T = 5, 320, 512, 2, 3
    W = rng.integers(0, 240, (E, K, M), dtype=np.uint8)
    ids = np.full((T, 64), -1, np.int32)
    ids[:, :R] = [[0, 4], [2, 2], [-1, 3]]
    g = Graph(device, "t_x8_copy")
    g.node(
        "rg_x8_copy_f32",
        [g.input("W", (M, K, E), F8), g.input("ids", (64, T), I32)],
        [g.input("o", (M, K, R * T), F8)],
        struct.pack("i", R),
    )
    out = Run(device, recipes, g)({"W": W, "ids": ids}, ["o"])["o"]
    for s in range(R * T):
        e = ids[s // R, s % R]
        assert np.array_equal(out[s], W[e] if e >= 0 else np.zeros((K, M), np.uint8)), s


def test_prefill_chunks():
    from resin_models_gaudi.glm5_3_flash import model as M

    assert M.prefill_chunks(35).step == 1024 and len(M.prefill_chunks(35)) == 1
    assert list(M.prefill_chunks(1100)) == [0, 1024]
    assert list(M.prefill_chunks(2048)) == [0, 1024]
    assert list(M.prefill_chunks(3072)) == [0, 1024, 2048]
    assert list(M.prefill_chunks(3073)) == [0]
    assert list(M.prefill_chunks(9000)) == [0, 4096, 8192]


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_rejected_draft_keeps_the_pool_it_restarts_inside(K):
    """Rows written for a rejected draft are not among the tail rows its restart reads."""
    from resin_models_gaudi.glm5_3_flash import model as M

    blocks = np.array([0, 2000], np.int64)
    for anchor in range(8, 16):  # every alignment, inside one block
        launch = M.Row(0, anchor, K + 1, blocks, -1, 0)
        _, written = M._store_rows(K + 1, [launch], keep_tails=True)
        rejected = {int(t) for t in written[1:, 4] if t >= 0}
        restart = M.Row(0, anchor + 1, K + 1, blocks, -1, 0)
        _, read = M._store_rows(K + 1, [restart], keep_tails=True)
        for sources in read[:, 1:4]:
            for s in sources:
                if s < 0:  # a stored tail
                    assert -int(s) - 1 not in rejected or -int(s) - 1 == written[0, 4], (K, anchor)


def test_store_rows_keep_scales_apart_at_large_rows():
    """A column's store row at a large arena row: every field fits i32, and the scale's row and index are distinct."""
    from resin_models_gaudi.glm5_3_flash import model as M

    b = (1 << 24) + 5 * M.PAGE_TOKENS
    row = M.Row(0, 5, 1, np.array([b], np.int64), -1, 0)
    store, _ = M._store_rows(1, [row])
    assert store[0, 0] == b + 5
    assert store[0, 3] == b + M.PAGE_TOKENS
    assert store[0, 4] == 5
    assert store.max() < 2**31


def test_mla_store_scale_at_large_rows(device, recipes):
    """A key's scale lands in its page's scale row at a large arena row."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    PR = 1024  # arena rows per block
    row = (1 << 24) + 5 * 129
    NB = (row + 2 * PR) // PR
    g = Graph(device, "t_glm_store_large_rows")
    arena = g.input("arena", (512, PR, NB), U8)
    g.node(
        "rg_mla_store_f32",
        [arena, g.input("rows", (8, 1), I32), g.input("kn", (512, 1), F32), g.input("ix", (128, 1), BF16)],
        [g.input("sink", (64, 1), I32)],
    )
    run = Run(device, recipes, g)
    address = device.malloc(512 * PR * NB)
    try:
        device.memset32(address + 512 * row, 0, 512 * 129 // 4)
        kn = np.full((1, 512), 24.0, np.float32)
        kn[0, 3] = 240.0
        rows = np.full((1, 8), -1, np.int32)
        rows[0, 0], rows[0, 3], rows[0, 4] = row + 5, row + 128, 5
        run.recipe.bind("arena", address)
        run({"rows": rows, "kn": kn, "ix": np.zeros((1, 128), np.uint16)}, ["sink"])
        scales = device.d2h(address + 512 * (row + 128), (128,), np.float32)
        assert scales[5] == pytest.approx(1.0, rel=1e-2)
        assert np.count_nonzero(scales) == 1
    finally:
        device.free(address)


def test_padding_columns_route_nowhere(device, recipes):
    """Prefill pages: a padding column reads -1 for any initial contents of bm."""
    from resin_models_gaudi.glm5_3_flash.layers import paged_shape
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import I32

    T, K, E, n_real = 256, 8, 16, 100
    rng = np.random.default_rng(5)
    psh, P, S = paged_shape(T, K, E)
    g = Graph(device, "moe_asg_padding")
    ids = g.input("ids", (64, T), I32)
    meta = g.input("meta", (4,), I32)
    cntm = g.tensor((E, 24), I32)
    g.node("rg_moe_cnt_f32", [ids, meta], [cntm], struct.pack("ii", 0, K))
    po, pe = g.tensor((E,), I32), g.tensor((P,), I32)
    g.node("rg_moe_pages_f32", [cntm], [po, pe], struct.pack("i", psh))
    g.node(
        "rg_moe_asg_f32",
        [ids, meta, cntm, po, pe],
        [g.input("gidx", (S,), I32), g.input("bm", (64, T), I32)],
        struct.pack("iii", 0, K, psh),
    )
    routes = np.full((T, 64), -1, np.int32)
    for t in range(n_real):
        routes[t, :K] = rng.choice(E, K, replace=False)
    out = Run(device, recipes, g)(
        {"ids": routes, "meta": np.array([0, n_real, 0, 0], np.int32)},
        ["gidx", "bm"],
        stale={"bm": np.full((T, 64), 7, np.int32)},
    )
    bm, gidx = out["bm"], out["gidx"]
    assert (bm[n_real:] == -1).all()
    assert (bm[:n_real, K:] == -1).all()
    rows = bm[:n_real, :K]
    assert (rows >= 0).all() and len(np.unique(rows)) == rows.size
    assert (gidx[rows] == np.arange(n_real)[:, None]).all()


def test_routed_experts_through_slots(device, recipes):
    """x13 / x2 in the host layout and the clamped SwiGLU, against f64"""
    from resin_models_gaudi.glm5_3_flash.weights import pair_rows
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32

    rng = np.random.default_rng(3)
    H, w, E, R, T, lim = 512, 256, 4, 2, 2, 10.0
    S = R * T

    def block(M, K):
        b = rng.integers(0, 0x76, (M, K), dtype=np.uint8)
        b |= rng.integers(0, 2, (M, K), dtype=np.uint8) << 7
        s = rng.uniform(1e-3, 3e-3, (M // 128, K // 128)).astype(np.float32)
        return b, s

    gates, ups, downs = [], [], []
    x13, s13, x2, s2 = [], [], [], []
    pH, pw = pair_rows(H), pair_rows(w)
    for _ in range(E):
        (g_, gs), (u_, us), (d_, ds) = block(w, H), block(w, H), block(H, w)
        val = lambda b, s: e4m3(b).astype(np.float64) * np.repeat(np.repeat(s, 128, 0), 128, 1)  # noqa: E731
        gates.append(val(g_, gs)), ups.append(val(u_, us)), downs.append(val(d_, ds))
        x13.append(np.concatenate([g_, u_]).T[pH])
        s13.append(np.concatenate([gs, us]).T)
        x2.append(d_.T[pw])
        s2.append(ds.T)
    x13, s13, x2, s2 = map(np.stack, (x13, s13, x2, s2))

    def requant(w_, s_):
        E_, K_, M_ = w_.shape
        g = Graph(device, f"t_rq_{M_}_{K_}")
        g.node(
            "rg_x8_requant_f32",
            [g.input("w", (M_, K_, E_), F8), g.input("bs", (M_ // 128, K_ // 128, E_), F32)],
            [g.input("o", (M_, K_, E_), F8), g.input("c", (M_, E_), F32)],
        )
        out = Run(device, recipes, g)({"w": w_, "bs": s_}, ["o", "c"])
        return out["o"], out["c"]

    W13, C13 = requant(x13, s13)
    W2, C2 = requant(x2, s2)
    x = rng.standard_normal((T, H)).astype(np.float32)
    ids = np.full((T, 64), -1, np.int32)
    wts = np.zeros((T, 64), np.float32)
    ids[:, :R] = [[0, 3], [2, 1]]
    wts[:, :R] = [[0.7, 1.8], [1.1, 0.4]]
    g = Graph(device, "t_slot_moe")
    gi = lambda n, sh, dt: g.input(n, sh, dt)  # noqa: E731
    h = gi("x", (H, T), F32)
    tid = gi("ids", (64, T), I32)
    xq, sx = g.tensor((H, 1, S), F8), g.tensor((1, T), F32)
    g.node("rg_ds_xq_f32", [h], [xq, sx], struct.pack("i", R))
    wd13, wd2 = g.tensor((2 * w, H, S), F8), g.tensor((H, w, S), F8)
    g.node("rg_x8_copy_f32", [gi("W13", (2 * w, H, E), F8), tid], [wd13], struct.pack("i", R))
    g.node("rg_x8_copy_f32", [gi("W2", (H, w, E), F8), tid], [wd2], struct.pack("i", R))
    y13 = g.tensor((2 * w, 1, S), F32)
    g.node("batch_gemm", [xq, wd13], [y13], GEMM_NN)
    hq, hs = g.tensor((w, 1, S), F8), g.tensor((1, S), F32)
    g.node(
        "rg_ds_xact_f32",
        [y13, y13, gi("C13", (2 * w, E), F32), sx, tid],
        [hq, hs],
        struct.pack("ifii", R, lim, 0, 0),
    )
    y2 = g.tensor((H, 1, S), F32)
    g.node("batch_gemm", [hq, wd2], [y2], GEMM_NN)
    g.node(
        "rg_ds_xsum_f32",
        [y2, gi("C2", (H, E), F32), hs, tid, gi("wts", (64, T), F32), gi("sh", (H, T), F32), gi("shs", (1, T), F32)],
        [gi("y", (H, T), F32)],
        struct.pack("i", R),
    )
    inputs = {
        "x": x, "ids": ids, "W13": W13, "W2": W2, "C13": C13, "C2": C2, "wts": wts,
        "sh": np.zeros((T, H), np.float32), "shs": np.ones((T, 1), np.float32),
    }  # fmt: skip
    got = Run(device, recipes, g)(inputs, ["y"])["y"]
    want = np.zeros((T, H))
    for t in range(T):
        for k in range(R):
            e = ids[t, k]
            gt = np.minimum(x[t] @ gates[e].T, lim)
            up = np.clip(x[t] @ ups[e].T, -lim, lim)
            want[t] += wts[t, k] * ((gt / (1 + np.exp(-gt)) * up) @ downs[e].T)
    # the FP8 model in numpy
    from resin_models_gaudi.qwen3_5.checkpoint import e4m3_encode

    def q8(v):
        scale = max(np.abs(v).max() / 240, 1e-30)
        return e4m3(e4m3_encode(v / scale)).astype(np.float64) * scale

    rows = lambda W, C, e, order: (e4m3(W[e])[np.argsort(order)].T * C[e][:, None]).astype(np.float64)  # noqa: E731
    fp8 = np.zeros((T, H))
    for t in range(T):
        xq_ = q8(x[t])
        for k in range(R):
            e = ids[t, k]
            m13 = rows(W13, C13, e, pH)
            gu = m13 @ xq_
            gt, up = np.minimum(gu[:w], lim), np.clip(gu[w:], -lim, lim)
            fp8[t] += wts[t, k] * (rows(W2, C2, e, pw) @ q8(gt / (1 + np.exp(-gt)) * up))
    assert np.corrcoef(got.ravel(), fp8.ravel())[0, 1] > 0.99995
    assert np.corrcoef(got.ravel(), want.ravel())[0, 1] > 0.99


# ------------------------------------------------------------------ KDA
@pytest.mark.parametrize("nh", [2, 8])
def test_kda_decode_matches_recurrence(device, recipes, nh):
    """Three tokens of two sequences over the arena, state written back and read next launch. nh = 8: a
    conv state of 72 rows padded to 80."""
    from glm_np import kda_recurrent, silu

    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    rng = np.random.default_rng(4 + nh)
    C, B, steps, lb, eps = nh * 128, 2, 3, -5.0, 1e-5
    CR = -(-9 * nh // 16) * 16
    NB = 8
    g = Graph(device, f"t_kda_decode_{nh}")
    ar = g.input("arena", (128, 1024, NB), F32)
    out, cst = g.input("out", (C, B), BF16), g.tensor((128, CR, B), F32)
    g.node(
        "rg_kda_decode_f32",
        [
            g.input("qkv", (3 * C, B), F32), g.input("fg", (C, B), F32), g.input("small", (256 + nh, B), F32),
            g.input("conv_w", (3 * C, 4), F32), ar, g.input("bases", (4, B), I32),
            g.input("par", (C + nh,), F32), g.input("gate", (C, B), F32), g.input("w", (128,), F32),
        ],  # fmt: skip
        [out, cst],
        struct.pack("ff", lb, eps),
    )
    g.node(
        "rg_state_store_f32",
        [ar, g.input("conv_dst", (B,), I32), cst],
        [g.input("sink", (64, 1), I32)],
    )
    recipe = recipes.compile(g)
    arena = device.malloc(NB * 1024 * 512)
    device.memset32(arena, 0x7FC00000, NB * 1024 * 128)  # stale contents must never matter
    addr = {}

    def put(name, a):
        a = np.ascontiguousarray(a)
        if name not in addr:
            addr[name] = device.malloc(max(a.nbytes, 256))
            recipe.bind(name, addr[name])
        device.h2d(addr[name], a)

    recipe.bind("arena", arena)
    for name in ("out", "sink"):
        addr[name] = device.malloc(max(recipe.persistent[name].nbytes, 256))
        recipe.bind(name, addr[name])
    conv_w = rng.standard_normal((4, 3 * C)).astype(np.float32) * 0.5
    dt = rng.standard_normal(C).astype(np.float32)
    A = rng.standard_normal(nh).astype(np.float32) * 0.1
    nw = rng.uniform(0.5, 1.5, 128).astype(np.float32)
    put("conv_w", conv_w), put("par", np.concatenate([dt, A])), put("w", nw)
    # state rows of the sequences
    bases_s, bases_c = [2048, 4096], [1024 - 8, 6144 - 8]
    qkv = rng.standard_normal((steps, B, 3 * C)).astype(np.float32)
    fg = rng.standard_normal((steps, B, C)).astype(np.float32)
    gate = rng.standard_normal((steps, B, C)).astype(np.float32)
    small = rng.standard_normal((steps, B, 256 + nh)).astype(np.float32)
    outs = []
    for t in range(steps):
        src_c = [-1 if t == 0 else c for c in bases_c]
        src_s = [-1 if t == 0 else s for s in bases_s]
        put("bases", np.array([[src_c[b], src_s[b], -1, bases_s[b]] for b in range(B)], np.int32))
        put("conv_dst", np.array(bases_c, np.int32))
        put("qkv", qkv[t]), put("fg", fg[t]), put("gate", gate[t]), put("small", small[t])
        recipe.launch()
        device.sync()
        outs.append(device.d2h(addr["out"], (B, C), np.uint16).copy())
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_to_f32

    for b in range(B):
        x = qkv[:, b].astype(np.float64)
        xp = np.concatenate([np.zeros((3, 3 * C)), x])
        conv = silu(sum(xp[j : j + steps] * conv_w[j].astype(np.float64) for j in range(4)))
        q, k, v = (conv[:, n * C : (n + 1) * C].reshape(steps, nh, 128) for n in range(3))
        f = fg[:, b].astype(np.float64) + dt
        e = lb / (1 + np.exp(-(np.exp(A.astype(np.float64))[None, :, None] * f.reshape(steps, nh, 128))))
        beta = 1 / (1 + np.exp(-small[:, b, 256:].astype(np.float64)))
        o, _ = kda_recurrent(q, k, v, e, beta)
        want = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps) * nw
        want = want / (1 + np.exp(-gate[:, b].reshape(steps, nh, 128).astype(np.float64)))
        for t in range(steps):
            got = bf16_to_f32(outs[t][b]).astype(np.float64)
            for h in range(nh):
                block = got[h * 128 : (h + 1) * 128]
                nat = np.concatenate([block[0::2], block[1::2]])
                assert np.abs(nat - want[t, h]).max() < 3e-2 * max(np.abs(want[t, h]).max(), 1e-3), (b, t, h)
    for a in [*addr.values(), arena]:
        device.free(a)


# ------------------------------------------------------------------ mHC
@pytest.mark.parametrize("T", [1, 3, 16])
def test_mhc_chain_matches_numpy(device, recipes, T):
    from glm_np import mhc, mhc_update

    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    rng = np.random.default_rng(5)
    H, eps, hc_eps, iters = 512, 1e-5, 1e-6, 20
    X = rng.standard_normal((T, 4, H)).astype(np.float32)
    fn = (rng.standard_normal((24, 4 * H)) * 0.05).astype(np.float32)
    pad = np.zeros((32, 4 * H), np.float32)
    pad[:24] = fn
    base = rng.standard_normal(24).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, 3).astype(np.float32)
    w = rng.uniform(0.5, 1.5, H).astype(np.float32)
    y = rng.standard_normal((T, H)).astype(np.float32)
    g = Graph(device, "t_mhc")
    xin = g.input("X", (4 * H, T), F32)
    mix, rt = g.tensor((T, 32), F32), g.tensor((64, T), F32)
    g.node("rg_mhc_mix1_f32" if T <= 2 else "rg_mhc_mix_f32", [g.input("fn", (4 * H, 32), F32), xin], [mix, rt], struct.pack("f", eps))
    coef = g.input("coef", (T, 24), F32)
    g.node(
        "rg_mhc_coef_f32",
        [mix, rt, g.input("base", (24,), F32), g.input("scale", (3,), F32)],
        [coef],
        struct.pack("if", iters, hc_eps),
    )
    h16, h32 = g.tensor((H, T), BF16), g.input("h", (H, T), F32)
    g.node("rg_mhc_pre_f32", [xin, coef, g.input("w", (H,), F32)], [h16, h32], struct.pack("fi", eps, 0))
    yv = g.input("y", (H, T, 1), F32)
    g.node("rg_mhc_res_f32", [xin, yv, coef], [g.input("Xo", (4 * H, T), F32)])
    flat = X.reshape(T, 4 * H)
    out = Run(device, recipes, g)(
        {"X": flat, "fn": pad, "base": base, "scale": scale, "w": w, "y": y[None]},
        ["coef", "h", "Xo"],
    )
    post, comb, coll = mhc(X, fn, base, scale, iters, hc_eps, eps)
    c = out["coef"]  # (24, T)
    assert np.abs(c[4:8].T - post).max() < 2e-3
    assert np.abs(c[8:24].T.reshape(T, 4, 4) - comb).max() < 2e-3
    want_h = coll / np.sqrt((coll * coll).mean(-1, keepdims=True) + eps) * w
    assert np.abs(out["h"] - want_h).max() < 5e-3 * np.abs(want_h).max()
    want_x = mhc_update(X, y, post, comb).reshape(T, 4 * H)
    assert np.abs(out["Xo"] - want_x).max() < 5e-3 * np.abs(want_x).max()


# ------------------------------------------------------------------ DSA pools
def _unbf16(u16):
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_to_f32

    return bf16_to_f32(u16)


def test_pool_keys_cross_launches_and_gather(device, recipes):
    """A pool whose first tokens came in an earlier launch reads them from the tail rows; the gather reads
    32 pools per page."""
    from resin_models_gaudi.glm5_3_flash.model import Row, _store_rows
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    rng = np.random.default_rng(6)
    NB, T, C, S = 8, 6, 2048, 1
    g = Graph(device, "t_gx_pool")
    ar = g.input("arena", (512, 1024, NB), U8)
    g.node(
        "rg_gx_pool_f32",
        [ar, g.input("rows", (8, T), I32), g.input("kg", (256, T), F32), g.input("ape", (128, 4), F32)],
        [g.input("sink", (64, 1), I32)],
    )
    pool = recipes.compile(g)
    arena = device.malloc(NB * 1024 * 512)
    device.memset32(arena, 0x7FC00000, NB * 1024 * 128)
    ape = rng.standard_normal((4, 128)).astype(np.float32)
    aa, ka, ra, sa = device.malloc(ape.nbytes), device.malloc(256 * T * 4), device.malloc(8 * T * 4), device.malloc(256)
    device.h2d(aa, ape)
    pool.bind_all({"arena": arena, "ape": aa, "kg": ka, "rows": ra, "sink": sa})
    tok = rng.standard_normal((12, 256)).astype(np.float32)
    blocks = np.array([1024 * 2, 1024 * 3], np.int64)  # layer pages at a KV block's first row
    for pos0 in (0, 6):  # positions 0..5 then 6..11: pools 0 (inside), 1 (crosses), 2 (inside)
        _, rows = _store_rows(T, [Row(0, pos0, T, blocks, -1, 0)])
        table = np.zeros((T, 8), np.int32)
        table[:, :5] = rows
        device.h2d(ra, table)
        device.h2d(ka, tok[pos0 : pos0 + T])
        pool.launch()
        device.sync()
    g2 = Graph(device, "t_gx_gather")
    g2.node(
        "rg_gx_gather_f32",
        [
            g2.input("arena", (512, 1024, NB), U8), g2.input("pages", (C // 32, S), I32),
            g2.input("slots", (8, S), I32), g2.input("after", (64,), F32),
        ],  # fmt: skip
        [g2.input("o", (128, C, S), BF16)],
    )
    pages = np.full((S, C // 32), -1, np.int32)
    pages[0, 0] = blocks[0] + 129
    slots = np.array([[0, 0, 1 << 30, 0, 3, 0, 0, 0]], np.int32)
    gather = recipes.compile(g2)
    addr = [device.malloc(max(a.nbytes, 256)) for a in (pages, slots)]
    device.h2d(addr[0], pages)
    device.h2d(addr[1], slots)
    oa, af = device.malloc(128 * C * 2), device.malloc(256)
    gather.bind_all({"arena": arena, "pages": addr[0], "slots": addr[1], "after": af, "o": oa})
    gather.launch()
    device.sync()
    got = _unbf16(device.d2h(oa, (S, C, 128), np.uint16).copy())[0]
    for P in range(3):
        k, gt = tok[4 * P : 4 * P + 4, :128].astype(np.float64), tok[4 * P : 4 * P + 4, 128:].astype(np.float64)
        lg = gt + ape
        pr = np.exp(lg - lg.max(0, keepdims=True))
        want = (pr / pr.sum(0, keepdims=True) * k).sum(0)
        assert np.abs(got[P] - want).max() < 0.02, P
    assert np.abs(got[3:]).max() == 0  # past klen: zeros, not the NaN-filled rows
    for a in [arena, aa, ka, ra, sa, oa, af, *addr]:
        device.free(a)


def test_mask_keeps_top_pools_and_tail(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(7)
    Kp, T, top = 4096, 3, 512
    score = rng.standard_normal((T, Kp)).astype(np.float32)
    klen = np.array([2402, 13, 0], np.int32)
    vis = klen // 4
    meta = np.array([2000, 402, 0, 0], np.int32)  # the chunk ends at 2402
    written = 2560
    g = Graph(device, "t_gx_mask")
    s, v = g.input("score", (Kp, T), F32), g.input("vis", (T,), I32)
    thr = g.tensor((64, T), I32)
    g.node("rg_ix_thr_f32", [s, v], [thr], struct.pack("i", top))
    g.node(
        "rg_gx_mask_f32",
        [s, v, g.input("klen", (T,), I32), thr, g.input("meta", (4,), I32)],
        [g.input("mask", (4 * Kp, T), F32)],
    )
    stale = np.full((T, 4 * Kp), np.nan, np.float32)
    out = Run(device, recipes, g)(
        {"score": score, "vis": vis, "klen": klen, "meta": meta}, ["mask"], stale={"mask": stale}
    )["mask"]
    # rows up to ``written`` are set in every column, including a padding column (klen 0)
    assert not np.isnan(out[:, :written]).any()
    assert np.isnan(out[:, written:]).all()  # rows past ``written`` keep their contents
    assert (out[2, :written] < -1e38).all()  # a padding column keeps nothing
    for c in range(T):
        kept = set()
        n = int(vis[c])
        if n:
            for P in np.argsort(-score[c, :n])[: min(top, n)]:
                kept.update(range(4 * int(P), 4 * int(P) + 4))
        kept.update(range(4 * n, int(klen[c])))
        got = set(np.nonzero(out[c] == 0)[0].tolist())
        assert got == kept, c
    assert (out[0, 4 * int(vis[0]) : int(klen[0])] == 0).all()  # the tail, own key included


def test_expand_keeps_own_key_past_top_pools(device, recipes):
    """512 pools of 4 keys and the tail: 2048 + up to 3 keys."""
    from resin_models_gaudi.glm5_3_flash.layers import LIST_KEYS
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import I32

    rng = np.random.default_rng(8)
    KS, T, H, C = 512, 3, 8, LIST_KEYS
    plist = np.zeros((T, KS), np.int32)
    picks = rng.choice(900, KS, replace=False)
    plist[0] = picks
    plist[1, :5] = [0, 1, 2, 3, 7]
    pslots = np.zeros((T, 8), np.int32)
    pslots[:, 4] = [KS, 5, 0]
    vis = np.array([900, 8, 0], np.int32)
    klen = np.array([3603, 33, 0], np.int32)
    g = Graph(device, "t_gx_expand")
    g.node(
        "rg_gx_expand_f32",
        [g.input("pools", (KS, T), I32), g.input("pslots", (8, T), I32), g.input("vis", (T,), I32), g.input("klen", (T,), I32)],
        [g.input("keys", (C, T), I32), g.input("slots", (8, T), I32)],
        struct.pack("ii", H, C),
    )
    out = Run(device, recipes, g)({"pools": plist, "pslots": pslots, "vis": vis, "klen": klen}, ["keys", "slots"])
    keys, slots = out["keys"], out["slots"]
    assert slots[0, 4] == 4 * KS + 3 and slots[1, 4] == 5 * 4 + 1 and slots[2, 0] == -1
    assert slots[1].tolist() == [0, H, C, 0, 21, 1, 0, 0]
    want = (picks[:, None] * 4 + np.arange(4)).ravel()
    assert keys[0, : 4 * KS].tolist() == want.tolist()
    assert keys[0, 4 * KS : 4 * KS + 3].tolist() == [3600, 3601, 3602]  # own key last
    assert keys[1, :21].tolist() == [*range(16), 28, 29, 30, 31, 32]


def test_mhc_coef1_matches_mhc_coef(device, recipes):
    """One token: mhc_mixp (the projection in K chunks, four rows per member) + mhc_coef1 (the comb matrix in
    lanes) against mhc_mix1 + mhc_coef."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(9)
    K, C, iters, hc_eps, eps = 4 * 640, 5, 20, 1e-6, 1e-5
    for trial in range(4):
        inputs = {
            "fn": (rng.standard_normal((32, K)) * 0.05).astype(np.float32),
            "X": (rng.standard_normal((1, K)) * rng.uniform(0.5, 4)).astype(np.float32),
            "base": rng.standard_normal(24).astype(np.float32),
            "scale": rng.uniform(0.1, 2.0, 3).astype(np.float32),
        }
        g = Graph(device, "t_coef_ref")
        mix, rt = g.tensor((1, 32), F32), g.tensor((64, 1), F32)
        g.node("rg_mhc_mix1_f32", [g.input("fn", (K, 32), F32), g.input("X", (K, 1), F32)], [mix, rt],
               struct.pack("f", eps))
        g.node("rg_mhc_coef_f32", [mix, rt, g.input("base", (24,), F32), g.input("scale", (3,), F32)],
               [g.input("coef", (1, 24), F32)], struct.pack("if", iters, hc_eps))
        want = Run(device, recipes, g)(inputs, ["coef"])["coef"].reshape(-1)
        g.destroy()
        g = Graph(device, "t_coef1")
        mixp, ssp = g.tensor((32, C), F32), g.tensor((C,), F32)
        g.node("rg_mhc_mixp_f32", [g.input("fn", (K, 32), F32), g.input("X", (K, 1), F32)], [mixp, ssp])
        g.node("rg_mhc_coef1_f32", [mixp, ssp, g.input("base", (24,), F32), g.input("scale", (3,), F32)],
               [g.input("coef", (24,), F32)], struct.pack("iffi", iters, hc_eps, eps, K))
        got = Run(device, recipes, g)(inputs, ["coef"])["coef"].reshape(-1)
        g.destroy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
