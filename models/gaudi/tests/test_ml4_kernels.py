"""Mistral-Large-4 kernels on one card against numpy: GEMVs over planar GGUF blocks (dense and
routed experts), the small kernels, and MLA attention over the paged arena against dense f32
attention."""

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

GEMM_NT = struct.pack("??", False, True)
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
        self.buffers: dict[str, int] = {}

    def __call__(self, inputs: dict[str, np.ndarray], outputs: dict[str, tuple]):
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


def random_blocks(rng, qtype, rows, inputs):
    from resin_models_gaudi.gguf.quants import LQ_TYPES

    _, elements, size = LQ_TYPES[qtype]
    nb = inputs // elements
    b = rng.integers(0, 256, (rows, nb, size), dtype=np.uint8)
    nf = 1 if qtype == "Q8_0" else 2
    s = rng.uniform(-0.02, 0.02, (rows, nb, nf)).astype(np.float16)
    b[:, :, : 2 * nf] = s.view(np.uint8).reshape(rows, nb, 2 * nf)
    return b.reshape(rows, nb * size)


# ------------------------------------------------------------------ GEMVs
@pytest.mark.parametrize("qtype", ["Q4_K", "Q5_K", "Q8_0"])
@pytest.mark.parametrize("ncol", [1, 4, 16])
def test_dense_gemv(device, recipes, qtype, ncol):
    from resin_models_gaudi.gguf.planar import gguf_planar
    from resin_models_gaudi.gguf.quants import dequantize
    from resin_models_gaudi.mistral_large4 import lq
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, U8

    rng = np.random.default_rng(10 + ncol)
    M, K = 640, 2048
    B = random_blocks(rng, qtype, M, K)
    P = gguf_planar(B, qtype).reshape(-1)
    x = rng.standard_normal((ncol, K)).astype(np.float32)
    g = Graph(device, f"t_gemv_{qtype}_{ncol}")
    y = g.input("y", (ncol * M,), F32)
    lq.gemv(
        g,
        qtype,
        g.input("w", (P.size, 1), U8),
        g.input("x", (K * ncol,), F32),
        g.input("dep", (64,), F32),
        K,
        M,
        ncol,
        out=y,
    )
    out = Run(device, recipes, g)(
        {"w": P, "x": x, "dep": np.zeros(64, np.float32)}, {"y": None}
    )["y"].reshape(ncol, M)
    ref = x.astype(np.float64) @ dequantize(B, qtype).astype(np.float64).T
    err = np.abs(out - ref).max() / np.abs(ref).max()
    assert np.corrcoef(out.ravel(), ref.ravel())[0, 1] > 0.9999 and err < 1e-2, err


@pytest.mark.parametrize("bcast", [True, False])
def test_expert_gemv(device, recipes, bcast):
    """top-4 routed slots of T tokens: gate/up (one x column per token, bcast) and down (one
    x column per slot) over stacked expert matrices"""
    from resin_models_gaudi.gguf.planar import gguf_planar
    from resin_models_gaudi.gguf.quants import dequantize
    from resin_models_gaudi.mistral_large4 import lq
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32, U8

    rng = np.random.default_rng(20 + bcast)
    E, M, K, T, nu = 8, 256, 1024, 5, 4
    Bs = [random_blocks(rng, "Q4_K", M, K) for _ in range(E)]
    W = np.stack([gguf_planar(b, "Q4_K").reshape(-1) for b in Bs])
    ids = np.zeros((T, 64), np.int32)
    ids[:, :nu] = np.stack([rng.choice(E, nu, replace=False) for _ in range(T)])
    ncol = T * nu
    N = T if bcast else ncol
    x = rng.standard_normal((N, K)).astype(np.float32)
    g = Graph(device, f"t_xgemv_{int(bcast)}")
    y = g.input("y", (ncol * M,), F32)
    lq.gemv(
        g,
        "Q4_K",
        g.input("w", (W.shape[1], E), U8),
        g.input("x", (K * N,), F32),
        g.input("dep", (64,), F32),
        K,
        M,
        ncol,
        ids=g.input("ids", (64 * T,), I32),
        nu=nu,
        bcast=bcast,
        out=y,
    )
    out = Run(device, recipes, g)(
        {"w": W, "x": x, "ids": ids, "dep": np.zeros(64, np.float32)}, {"y": None}
    )["y"].reshape(ncol, M)
    dq = [dequantize(b, "Q4_K").astype(np.float64) for b in Bs]
    for t in range(T):
        for s in range(nu):
            c = t * nu + s
            xc = x[t] if bcast else x[c]
            ref = dq[ids[t, s]] @ xc
            np.testing.assert_allclose(
                out[c], ref, rtol=0, atol=2e-2 * np.abs(ref).max()
            )


# ------------------------------------------------------------------ MLA over the paged arena
PR = 1024  # arena rows per block (dim 1)


def _pages(rng, n, size, base, count):
    """n distinct page starts (multiples of size) in [base, base + count * size)"""
    return (base + size * rng.choice(count, n, replace=False)).astype(np.int32)


def _e4m3(b: np.ndarray) -> np.ndarray:
    """FP8 E4M3 bytes to f64"""
    b = b.astype(np.int64)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(b & 0x80, -v, v)


def _store_keys(d, recipes, arena_addr, NB, rows, kn, ix):
    """mla_store of kn (T, D) f32 and ix (T, 128) bf16 bits at rows (T, 8)"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    T, D = kn.shape
    g = Graph(d, f"t_mla_store_{D}_{T}")
    g.node(
        "rg_mla_store_f32",
        [
            g.input("arena", (512, PR, NB), U8),
            g.input("rows", (8, T), I32),
            g.input("kn", (D, T), F32),
            g.input("ix", (128, T), BF16),
        ],
        [g.input("sink", (64, 1), I32)],
    )
    store = Run(d, recipes, g)
    store.recipe.bind("arena", arena_addr)
    store({"rows": rows, "kn": kn, "ix": ix}, {"sink": None})


class _Layout:
    """random page starts for one layer: latent (129 rows), rope (32), index (64)"""

    def __init__(self, rng, npages):
        self.lat = _pages(rng, npages, 129, 0, 120)  # rows 0 .. 15479
        self.rope = _pages(rng, npages, 32, 16384, 256)  # rows 16384 .. 24575
        self.idx = _pages(rng, npages, 64, 24576, 128)  # rows 24576 .. 32767

    def rows(self, positions, rope=True, index=False):
        rows = np.full((len(positions), 8), -1, np.int32)
        page, r = np.divmod(np.asarray(positions), 128)
        rows[:, 0] = self.lat[page] + r
        rows[:, 3] = self.lat[page] + 128
        rows[:, 4] = r
        if rope:
            rows[:, 1] = 4 * self.rope[page] + r
        if index:
            rows[:, 2] = 2 * self.idx[page] + r
        return rows

    def latent(self, arena, p):
        """the stored latent of position p, dequantized (f64)"""
        page, r = divmod(p, 128)
        raw = arena[self.lat[page] + r]  # natural order
        scale = arena[self.lat[page] + 128, 4 * r : 4 * r + 4].view(np.float32)[0]
        assert scale.view(np.uint32) & 0xFFFF == 0  # a bf16 value
        return _e4m3(raw) * scale

    def rope_key(self, arena, p):
        page, r = divmod(p, 128)
        u = 4 * self.rope[page] + r
        return arena[u >> 2, (u & 3) * 128 : (u & 3) * 128 + 128].view(np.uint16)

    def index_key(self, arena, p):
        page, r = divmod(p, 128)
        u = 2 * self.idx[page] + r
        return arena[u >> 1, (u & 1) * 256 : (u & 1) * 256 + 256].view(np.uint16)


NB_TEST = 32  # arena blocks (32 MB)


@pytest.mark.parametrize("D", [576, 512])
def test_mla_store_gather(device, recipes, D):
    """mla_store's FP8 latent (with scale), bf16 rope and index units, read back on the host
    and through mla_gather's key and index-key modes"""
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne, bf16_to_f32
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    rng = np.random.default_rng(D)
    L, C, LAT = 300, 2048, 512  # three pages, the last partly filled
    lay = _Layout(rng, -(-L // 128))
    kn = rng.standard_normal((L, D)).astype(np.float32) * rng.uniform(0.01, 30, (L, 1))
    kn = kn.astype(np.float32)
    ix = bf16_rne(rng.standard_normal((L, 128)).astype(np.float32))
    d = device
    arena = np.full((NB_TEST * PR, 512), 0xFF, np.uint8)  # NaN in every format
    arena_addr = d.malloc(arena.nbytes)
    d.h2d(arena_addr, arena)
    _store_keys(
        d, recipes, arena_addr, NB_TEST, lay.rows(range(L), D > LAT, True), kn, ix
    )
    back = d.d2h(arena_addr, arena.shape, np.uint8)

    lat = np.stack([lay.latent(back, p) for p in range(L)])
    amax = np.abs(kn[:, :LAT]).max(1, keepdims=True)
    np.testing.assert_allclose(amax[:, 0], np.abs(lat).max(1), rtol=2**-4)
    assert (
        np.abs(lat - kn[:, :LAT])
        <= amax / 240 * 2.0**-3 + np.abs(kn[:, :LAT]) * 2.0**-4
    ).all()
    for p in range(L):
        np.testing.assert_array_equal(lay.index_key(back, p), ix[p])
        if D > LAT:
            rk = lay.rope_key(back, p)
            np.testing.assert_array_equal(rk[:64], bf16_rne(kn[p, LAT:]))
            assert (rk[64:] == 0xFFFF).all()  # the next token's unit is untouched

    g = Graph(d, f"t_mla_gather_{D}")
    table = g.input("slots", (8, 1), I32)
    ar = g.input("arena", (512, PR, NB_TEST), U8)
    lp, rp, ip = (
        g.input(n, (C // 128, 1), I32) for n in ("lat_pages", "rope_pages", "idx_pages")
    )
    outs = {"k": (D, C, 1), "i": (128, C, 1)}
    dep = g.input("dep", (64,), F32)
    for mode, (name, shape) in zip((0, 2), outs.items(), strict=True):
        pages = ip if mode == 2 else lp
        g.node(
            "rg_mla_gather_f32",
            [ar, pages, rp, table, dep],
            [g.input(name, shape, BF16)],
            struct.pack("i", mode),
        )
    gather = Run(d, recipes, g)
    gather.recipe.bind("arena", arena_addr)
    pad = np.full(C // 128 - len(lay.lat), -1, np.int32)
    slots = np.zeros((1, 8), np.int32)
    slots[0, 4] = L
    out = gather(
        {
            "slots": slots,
            "lat_pages": np.concatenate([lay.lat, pad])[None],
            "rope_pages": np.concatenate([lay.rope, pad])[None],
            "idx_pages": np.concatenate([lay.idx, pad])[None],
            "dep": np.zeros(64, np.float32),
        },
        dict.fromkeys(outs),
    )
    d.free(arena_addr)
    k, i = (out[n].view(np.uint16)[0] for n in outs)
    want = bf16_to_f32(bf16_rne(lat.astype(np.float32)))
    np.testing.assert_array_equal(bf16_to_f32(k[:L, :LAT]), want)
    if D > LAT:
        np.testing.assert_array_equal(k[:L, LAT:], bf16_rne(kn[:, LAT:]))
    np.testing.assert_array_equal(i[:L], ix)
    assert not k[L:].any() and not i[L:].any()  # past klen and unmapped pages: zeros


@pytest.mark.parametrize("D", [576, 512])
@pytest.mark.parametrize("T,pos0", [(16, 2490), (1, 4100), (16, 0)])
def test_mla_slot_attention(device, recipes, D, T, pos0):
    from resin_models_gaudi.qwen3_5.attention import SlotRow, plan_slots
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne, bf16_to_f32
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(D + T + pos0)
    H, LAT, S = 8, 512, 4
    L = pos0 + T
    lay = _Layout(rng, -(-L // 128))
    keys = rng.standard_normal((L, D)).astype(np.float32)
    keys[:, LAT:] *= 2.0
    d = device
    arena = np.full(
        (NB_TEST * PR, 512), 0xFF, np.uint8
    )  # NaN: stale rows must never be read
    arena_addr = d.malloc(arena.nbytes)
    d.h2d(arena_addr, arena)
    ix = np.zeros((L, 128), np.uint16)
    if pos0:  # the cached positions, then the chunk's own keys as a second store
        _store_keys(
            d,
            recipes,
            arena_addr,
            NB_TEST,
            lay.rows(range(pos0), D > LAT),
            keys[:pos0],
            ix[:pos0],
        )
    _store_keys(
        d,
        recipes,
        arena_addr,
        NB_TEST,
        lay.rows(range(pos0, L), D > LAT),
        keys[pos0:],
        ix[pos0:],
    )
    back = d.d2h(arena_addr, arena.shape, np.uint8)
    kf = np.stack([lay.latent(back, p) for p in range(L)])
    if D > LAT:
        kf = np.concatenate(
            [kf, bf16_to_f32(np.stack([lay.rope_key(back, p)[:64] for p in range(L)]))],
            1,
        )
    kf = bf16_to_f32(bf16_rne(kf.astype(np.float32))).astype(
        np.float64
    )  # gather's bf16 slab

    # slots: one row (sequence) at column 0, the whole chunk as one query block
    row = SlotRow(0, pos0, T, lay.lat[None, :], lay.rope[None, :])
    launches = plan_slots([row], T, H, S)
    W = H * T
    q = rng.standard_normal((W, D)).astype(np.float32)  # column = h + H t
    qbits = bf16_rne(q)
    scale = 128**-0.5
    acc = {"acc_o": (LAT, W, 1), "acc_m": (W, 1), "acc_l": (W, 1)}
    acc_addr = {n: d.malloc(4 * int(np.prod(s))) for n, s in acc.items()}
    g = Graph(d, f"t_mla_init_{W}")
    g.node(
        "rg_mla_acc_init_f32",
        [g.input(n, s, F32) for n, s in acc.items()],
        [g.input("sink", (64, 1), I32)],
    )
    init = Run(d, recipes, g)
    init.recipe.bind_all(acc_addr)
    init({}, {"sink": None})

    # the slot launch (layers.attn_slots)
    from resin_models_gaudi.mistral_large4.layers import LayerGraph, attn_slots

    lg = LayerGraph(d, f"t_mla_slots_{D}_{T}", _dims(), T)
    attn_slots(lg, T, S, D, NB_TEST)
    g = lg.g
    slots = Run(d, recipes, g)
    slots.recipe.bind("arena", arena_addr)
    slots.recipe.bind_all(acc_addr)
    for launch in launches:
        slots(
            {
                "slots": launch.slots,
                "lat_pages": launch.k_pages,
                "rope_pages": launch.v_pages,
                "q_all": qbits,
                "dep": np.zeros(64, np.float32),
            },
            {"sink_slots": None},
        )
    out = d.d2h(acc_addr["acc_o"], (W, LAT), np.float32)
    for a in acc_addr.values():
        d.free(a)
    d.free(arena_addr)

    qf = bf16_to_f32(qbits).astype(np.float64)
    ref = np.empty((W, LAT))
    for c in range(W):
        t = c // H
        s = qf[c] @ kf[: pos0 + t + 1].T * scale
        p = np.exp(s - s.max())
        ref[c] = (p / p.sum()) @ kf[: pos0 + t + 1, :LAT]
    err = np.abs(out - ref).max() / np.abs(ref).max()
    assert err < 2e-2, err


def test_mla_store_at_large_rows(device, recipes):
    """A key's scale lands in its page's scale row at any row of the arena."""
    row = (1 << 24) + 5 * 129
    NB = (row + 2 * PR) // PR
    arena = device.malloc(512 * PR * NB)
    device.memset32(arena + 512 * row, 0, 4 * 512 * 129 // 4)
    try:
        kn = np.full((1, 512), 24.0, np.float32)
        kn[0, 3] = 240.0
        rows = np.full((1, 8), -1, np.int32)
        rows[0, 0], rows[0, 3], rows[0, 4] = row + 5, row + 128, 5
        ix = np.zeros((1, 128), np.uint16)
        _store_keys(device, recipes, arena, NB, rows, kn, ix)
        scales = device.d2h(arena + 512 * (row + 128), (128,), np.float32)
        assert scales[5] == pytest.approx(1.0, rel=1e-2)
        assert np.count_nonzero(scales) == 1
    finally:
        device.free(arena)


def _mkey(v: np.ndarray) -> np.ndarray:
    """ix_thr's monotone u32 key of f32 scores"""
    b = v.astype(np.float32).view(np.uint32)
    return np.where(v < 0, ~b, b | np.uint32(0x80000000))


@pytest.mark.parametrize("NM", [1, 5])
def test_ix_list(device, recipes, NM):
    """ix_thr -> ix_bcnt -> ix_list against numpy: each column's KS-th largest key, then its
    list (keys above the threshold, then ties, ascending, at most KS) and slot record; columns
    with heavy ties, short columns (every key listed) and a padding column (vis 0)"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(NM)
    K, T, KS, H = 8192, 16, 2048, 8
    vis = np.array(
        [
            0,
            1,
            100,
            2048,
            2049,
            5000,
            8192,
            3000,
            7777,
            64,
            4096,
            2050,
            6000,
            8191,
            2500,
            4097,
        ],
        np.int32,
    )
    score = rng.standard_normal((T, K)).astype(np.float32)
    score[[5, 6, 12]] = np.round(score[[5, 6, 12]], 1)  # many exact ties
    score[7] = 1.0  # every key ties
    score[:, :] = np.where(
        np.arange(K) < vis[:, None], score, np.nan
    )  # never read past vis
    g = Graph(device, f"t_ix_list_{NM}")
    sc, vi = g.input("score", (K, T), F32), g.input("vis", (T,), I32)
    thr = g.input("thr", (64, T), I32)
    g.node("rg_ix_thr_f32", [sc, vi], [thr], struct.pack("i", KS))
    cnt, gsum = g.tensor((K // 64, T), I32), g.tensor((K // 4096, T), I32)
    g.node("rg_ix_bcnt_f32", [sc, vi, thr], [cnt, gsum])
    g.node(
        "rg_ix_list_f32",
        [sc, vi, thr, cnt, gsum],
        [g.input("list", (KS, T), I32), g.input("slots", (8, T), I32)],
        struct.pack("iii", NM, KS, H),
    )
    out = Run(device, recipes, g)(
        {"score": score, "vis": vis}, dict.fromkeys(("thr", "list", "slots"))
    )
    for c in range(T):
        k = _mkey(score[c, : vis[c]])
        th = np.sort(k)[::-1][KS - 1] if vis[c] > KS else 1
        assert (out["thr"][c] == np.uint32(th).view(np.int32)).all(), c
        pos = np.arange(vis[c])
        want = np.concatenate([pos[k > th], pos[k == th]])[:KS]
        n = len(want)
        np.testing.assert_array_equal(out["list"][c, :n], want, err_msg=f"column {c}")
        # lane 7 (one member per column): the position of the last tie listed, every tie, or none
        st, ties = int((k > th).sum()), pos[k == th]
        cut = 0x7FFFFFFF if st + len(ties) <= KS else -1 if st >= KS else int(ties[KS - st - 1])
        rec = [0 if vis[c] else -1, H * c, KS, 0, n, c, 0, cut if NM == 1 else 0]
        np.testing.assert_array_equal(out["slots"][c], rec, err_msg=f"column {c}")


def test_prefill_masked_attention(device, recipes):
    """ix_mask -> ix_unmask -> mla_slot_pm -> mla_slot_am over two key slots against numpy's masked
    softmax. Column 0 keeps no key in the first slot, column 1 none in the second; column 2 sees only
    some keys."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    rng = np.random.default_rng(3)
    C, T, H, K, KS, LAT = 2048, 16, 8, 4096, 512, 64
    M, scale = H * T, 1 / 24.0
    vis = np.full(T, K, np.int32)
    vis[2], vis[3] = 2500, 3000
    score = rng.standard_normal((T, K)).astype(np.float32)
    score[0, :C], score[1, C:] = -1e6, -1e6
    th = np.ones(T, np.uint32)
    for c in range(T):
        k = _mkey(score[c, : vis[c]])
        if vis[c] > KS:
            th[c] = np.sort(k)[::-1][KS - 1]
    thr = np.repeat(th[:, None], 64, axis=1).view(np.int32)
    keep = (_mkey(score) >= th[:, None]) & (np.arange(K) < vis[:, None])

    g = Graph(device, "t_ix_mask")
    vi, bits = g.input("vis", (T,), I32), g.tensor((K // 32, T), I32)
    g.node(
        "rg_ix_mask_f32",
        [g.input("score", (K, T), F32), vi, g.input("thr", (64, T), I32)],
        [bits],
    )
    g.node("rg_ix_unmask_f32", [bits, vi], [g.input("mask", (K, T), F32)])
    mask = Run(device, recipes, g)(
        {"score": score, "vis": vis, "thr": thr}, {"mask": None}
    )["mask"].copy()
    np.testing.assert_array_equal(mask == 0, keep)

    sc = (rng.standard_normal((M, K)) * 3 / scale).astype(np.float32)
    values = rng.standard_normal((K, LAT)).astype(np.float32)
    g = Graph(device, "t_slot_pm")
    g.node(
        "rg_mla_slot_pm_f32",
        [
            g.input("sc", (C, M, 1), F32),
            g.input("slots", (8, 1), I32),
            g.input("mask", (K, T), F32),
        ],
        [
            g.input("P", (C, M, 1), BF16),
            g.input("mx", (M, 1), F32),
            g.input("l", (M, 1), F32),
        ],
        struct.pack("fi", scale, H),
    )
    pm = Run(device, recipes, g)

    def buf(a=None, nbytes=0):
        address = device.malloc(max(nbytes or a.nbytes, 256))
        if a is not None:
            device.h2d(address, np.ascontiguousarray(a))
        return address

    acc = {
        "acc_o": buf(nbytes=4 * LAT * M),
        "acc_m": buf(nbytes=4 * M),
        "acc_l": buf(nbytes=4 * M),
        "sink": buf(nbytes=256),
    }
    shapes = {"acc_o": (LAT, M, 1), "acc_m": (M, 1), "acc_l": (M, 1)}
    g = Graph(device, "t_acc_init")
    g.node(
        "rg_mla_acc_init_f32",
        [g.input(n, s, F32) for n, s in shapes.items()],
        [g.input("sink", (64, 1), I32)],
    )
    init = recipes.compile(g)
    for n, a in acc.items():
        init.bind(n, a)
    init.launch()
    device.sync()
    g = Graph(device, "t_slot_am")
    g.node(
        "rg_mla_slot_am_f32",
        [
            g.input("O", (LAT, M, 1), F32),
            g.input("mx", (M, 1), F32),
            g.input("l", (M, 1), F32),
            g.input("slots", (8, 1), I32),
            *(g.input(n, s, F32) for n, s in shapes.items()),
        ],
        [g.input("sink", (64, 1), I32)],
    )
    am = recipes.compile(g)
    for n, a in acc.items():
        am.bind(n, a)
    for key0 in (0, C):
        slots = np.array([[0, 0, K, key0, K, 0, 0, 0]], np.int32)
        r = pm(
            {"sc": sc[None, :, key0 : key0 + C], "slots": slots, "mask": mask},
            dict.fromkeys(("P", "mx", "l")),
        )
        P = (r["P"].astype(np.uint32) << 16).view(np.float32)[0]
        mx, ls = r["mx"].copy(), r["l"].copy()
        assert np.isfinite(P).all() and np.isfinite(mx).all() and np.isfinite(ls).all()
        partial = P @ values[key0 : key0 + C]
        addresses = [buf(partial[None]), buf(mx), buf(ls), buf(slots)]
        for n, a in zip(("O", "mx", "l", "slots"), addresses, strict=True):
            am.bind(n, a)
        am.launch()
        device.sync()
        for a in addresses:
            device.free(a)
    out = device.d2h(acc["acc_o"], (M, LAT), np.float32).copy()
    assert np.isfinite(out).all()
    ref = np.zeros_like(out)
    for m in range(M):
        kept = keep[m // H]
        s = sc[m, kept] * scale
        p = np.exp(s - s.max())
        ref[m] = (p / p.sum()) @ values[kept]
    err = np.abs(out - ref).max() / np.abs(ref).max()
    assert err < 2e-2, err
    for a in acc.values():
        device.free(a)


# ------------------------------------------------------------------ prefill (dequantized weights)
def _dims(**kw):
    from resin_models_gaudi.mistral_large4.layers import Dims

    base = dict(
        hidden=1024, heads=8, qk=128, latent=512, rope=64, v=128, dense=512, expert=256,
        shared=256, experts=128, top=4, vocab=1024, index_heads=32, index_dim=128,
        eps=1e-5, index_eps=1e-6, scale=128**-0.5,
    )  # fmt: skip
    base.update(kw)
    return Dims(**base)


def test_dense_dequant_gemm(device, recipes):
    from resin_models_gaudi.gguf.planar import gguf_planar
    from resin_models_gaudi.gguf.quants import dequantize
    from resin_models_gaudi.mistral_large4.layers import LayerGraph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(30)
    M, K, T = 640, 1024, 256
    B = random_blocks(rng, "Q8_0", M, K)
    P = gguf_planar(B, "Q8_0").reshape(-1)
    x = rng.standard_normal((T, K)).astype(np.float32)
    lg = LayerGraph(device, "t_dense_mme", _dims(), T, mme=True)
    lg.gemv(
        "w",
        "Q8_0",
        lg.input("x", (K, T), F32),
        K,
        M,
        T,
        out=lg.input("y", (M * T,), F32),
    )
    out = Run(device, recipes, lg.g)(
        {"w": P, "x": x, "dep": np.zeros(64, np.float32)}, {"y": None}
    )["y"].reshape(T, M)
    ref = x.astype(np.float64) @ dequantize(B, "Q8_0").astype(np.float64).T
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() / np.abs(ref).max() < 2e-2


@pytest.mark.parametrize("qtype", ["Q4_K", "Q5_K"])
def test_paged_experts(device, recipes, qtype):
    """the paged MoE (gate, up and down, SwiGLU) against numpy, with padding tokens routed nowhere"""
    from resin_models_gaudi.gguf.planar import gguf_planar
    from resin_models_gaudi.gguf.quants import dequantize
    from resin_models_gaudi.mistral_large4.layers import (
        LayerGraph,
        paged_experts,
        paged_shape,
        row_chunks,
    )
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(31)
    T, E, F, H, K, real = 256, 128, 256, 6144, 4, 250
    d = _dims(hidden=H, expert=F, experts=E)
    P = paged_shape(d, T)[1]
    assert len(row_chunks(H, F, P)) == 1 and len(row_chunks(F, H, P)) == 1
    gate = [random_blocks(rng, "Q4_K", F, H) for _ in range(E)]
    up = [random_blocks(rng, "Q4_K", F, H) for _ in range(E)]
    down = [random_blocks(rng, qtype, H, F) for _ in range(E)]
    xu = np.stack(
        [
            np.concatenate(
                [gguf_planar(g, "Q4_K").ravel(), gguf_planar(u, "Q4_K").ravel()]
            )
            for g, u in zip(gate, up)
        ]
    )
    xd = np.stack([gguf_planar(b, qtype).ravel() for b in down])
    ids = np.zeros((T, 64), np.int32)
    ids[:, :K] = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    ids[real:, :K] = -1  # padding tokens (moe_route's ids for t >= n_real)
    h = (0.1 * rng.standard_normal((T, H))).astype(np.float32)
    lg = LayerGraph(device, f"t_paged_{qtype}", d, T, mme=True)
    yd = paged_experts(
        lg,
        lg.input("h", (H, T), F32),
        lg.input("ids", (64 * T,), I32),
        {"xu": "Q4_K", "xd": qtype},
    )
    lg.node("memcpy", [yd], [lg.input("y", (H, K * T), F32)])
    out = Run(device, recipes, lg.g)(
        {"xu": xu, "xd": xd, "ids": ids, "h": h, "dep": np.zeros(64, np.float32)},
        {"y": None},
    )["y"].reshape(T, K, H)
    assert np.isfinite(out).all()
    for t in (0, 1, 100, real - 1):
        for s in range(K):
            e = ids[t, s]
            g = dequantize(gate[e], "Q4_K") @ h[t]
            u = dequantize(up[e], "Q4_K") @ h[t]
            a = g / (1 + np.exp(-g)) * u
            ref = dequantize(down[e], qtype) @ a
            assert np.abs(out[t, s] - ref).max() < 3e-2 * np.abs(ref).max(), (t, s)
    assert not out[real:].any()
