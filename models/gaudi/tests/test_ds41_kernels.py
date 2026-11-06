"""DeepSeek-V4.1 kernels on one card against numpy: the MXFP4 expert dequant."""

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

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


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


def fp8_byte(v: np.ndarray, code: np.ndarray) -> np.ndarray:
    """Round to E4M3 bias 7 (+-240, subnormals to 2^-9), ties to even, saturating. Code -0 gives 0x00,
    a negative value that rounds to zero gives 0x80."""
    a = np.abs(v)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0))).astype(np.int64)
    e = np.clip(e, -6, None)
    q = np.round(a / np.exp2(e - 3.0)).astype(
        np.int64
    )  # 3 mantissa bits; numpy rounds ties to even
    up = q == 16
    e, q = np.where(up, e + 1, e), np.where(up, 8, q)
    normal = q >= 8
    byte = np.where(normal, ((e + 7) << 3) | (q - 8), q)
    byte = np.where(a > 0, byte, 0)
    byte = np.where((e > 7) | (byte > 0x77), 0x77, byte)
    negative = (v < 0) | ((code & 8) != 0) & (code != 8)
    byte = np.where(negative & (code != 8), byte | 0x80, byte)
    return byte.astype(np.uint8)


def mx4_ref(codes: np.ndarray, scales: np.ndarray, k: int) -> np.ndarray:
    """numpy model of mx4_dec: codes (M, K/2) low nibble first, scales (M, K/32) -> FP8 bytes (M, K)."""
    nib = (
        np.stack([codes & 15, codes >> 4], -1)
        .reshape(codes.shape[0], -1)
        .astype(np.int64)
    )
    exponent = scales.astype(np.int64).repeat(32, 1)[:, : nib.shape[1]] - 127 - k
    value = E2M1[nib] * np.exp2(exponent.astype(np.float64))
    return fp8_byte(value, nib)


def run_dec(device, recipes, codes, scales, k, *, flat, ids=None, stale=None):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, I32, U8

    m, half = codes.shape
    n = 2 * half
    rows = m if ids is None else len(ids)
    g = Graph(
        device, f"t_mx4_{'s' if ids is not None else ''}{'f' if flat else '2'}_{m}x{n}"
    )
    if flat:
        c = g.input("codes", (m * half, 1), U8)
        s = g.input("scales", (m * n // 32, 1), U8)
        o = g.input("out", (m * n, 1), F8)
    else:
        c = g.input("codes", (half, m), U8)
        s = g.input("scales", (n // 32, m), U8)
        o = g.input("out", (n, rows), F8)
    if ids is None:
        g.node("rg_mx4_dec_f32", [c, s], [o], struct.pack("<i", k))
    else:
        i, kt = g.input("ids", (rows,), I32), g.input("ktab", (m,), I32)
        g.node("rg_mx4_decs_f32", [c, s, i, kt], [o], struct.pack("<ii", 0, 0))
    recipe = recipes.compile(g)
    allocated = []
    try:

        def put(name, array):
            a = np.ascontiguousarray(array)
            address = device.malloc(max(a.nbytes, 256))
            allocated.append(address)
            device.h2d(address, a)
            recipe.bind(name, address)

        put("codes", codes)
        put("scales", scales)
        if ids is not None:
            put("ids", np.asarray(ids, np.int32))
            put("ktab", np.broadcast_to(np.asarray(k, np.int32), (m,)).copy())
        out = np.full((rows, n), 0xEE, np.uint8) if stale is None else stale
        put("out", out)
        address = allocated[-1]
        recipe.launch()
        device.sync()
        return device.d2h(address, (rows, n), np.uint8)
    finally:
        for address in allocated:
            device.free(address)
        g.destroy()


def exhaustive_inputs(
    k_len: int, rows: int, shift: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every block holds all 16 codes; scales cycle through all 256 bytes across the rows' blocks."""
    blocks = k_len // 32
    nib = np.tile(np.arange(32) % 16, blocks)
    nib = np.stack([np.roll(nib, (r * 3 + shift) % 32) for r in range(rows)])
    codes = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    scales = (
        (np.arange(rows * blocks).reshape(rows, blocks) * 7 + shift) % 256
    ).astype(np.uint8)
    return codes, scales


@pytest.mark.parametrize("flat", [True, False])
@pytest.mark.parametrize("k", [0, -20, -8, 13])
def test_mx4_dec_every_code_and_scale(device, recipes, flat, k):
    # K 34816: the 32-vector loop and the 8-vector tail
    codes, scales = exhaustive_inputs(34816, 3, k & 31)
    out = run_dec(device, recipes, codes, scales, k, flat=flat)
    np.testing.assert_array_equal(out, mx4_ref(codes, scales, k))
    nib = np.stack([codes & 15, codes >> 4], -1).reshape(3, -1)
    pairs = {
        (int(c), int(s)) for c, s in zip(nib.ravel(), scales.repeat(32, 1).ravel())
    }
    assert len(pairs) == 4096
    assert not np.any((out & 0x78) == 0x78), "an exponent-15 byte"


def test_mx4_dec_short_rows(device, recipes):
    rng = np.random.default_rng(3)
    for k_len in (32, 2048, 2080, 6144):
        codes = rng.integers(0, 256, (5, k_len // 2), dtype=np.uint8)
        scales = rng.integers(110, 135, (5, k_len // 32), dtype=np.uint8)
        k = int(scales.max()) - 127 - 5
        out = run_dec(device, recipes, codes, scales, k, flat=False)
        np.testing.assert_array_equal(
            out, mx4_ref(codes, scales, k), err_msg=f"K {k_len}"
        )


def test_mx4_decs_rows_follow_ids(device, recipes):
    rng = np.random.default_rng(4)
    experts, k_len = 6, 10240
    codes = rng.integers(0, 256, (experts, k_len // 2), dtype=np.uint8)
    spread = 3 * np.arange(experts)[:, None]
    scales = (rng.integers(115, 125, (experts, k_len // 32)) + spread).astype(np.uint8)
    ks = scales.max(axis=1).astype(np.int64) - 127 - 5  # one k per expert
    ids = [3, -1, 0, 3, 5, 1]
    stale = np.full((len(ids), k_len), 0xEE, np.uint8)
    out = run_dec(device, recipes, codes, scales, ks, flat=False, ids=ids, stale=stale)
    for row, expert in enumerate(ids):
        if expert < 0:
            np.testing.assert_array_equal(out[row], stale[row], err_msg="padding row")
            continue
        one = slice(expert, expert + 1)
        want = mx4_ref(codes[one], scales[one], int(ks[expert]))[0]
        np.testing.assert_array_equal(out[row], want, err_msg=f"row {row}")


class Run:
    """Compile a graph, bind numpy inputs and output buffers by name, launch once."""

    def __init__(self, device, recipes, graph) -> None:
        self.device, self.graph = device, graph
        self.recipe = recipes.compile(graph)

    def __call__(
        self, inputs: dict[str, np.ndarray], outputs: list[str]
    ) -> dict[str, np.ndarray]:
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


def fp8_value(b: np.ndarray) -> np.ndarray:
    """E4M3 bias 7 bytes -> values."""
    b = b.astype(np.int64)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * np.exp2(e - 10.0), m * np.exp2(-9.0))
    return np.where(b & 0x80, -v, v)


def pow2_scale(amax: float) -> float:
    """The smallest power of two s with amax <= 240 s."""
    s = 2.0 ** np.ceil(np.log2(max(amax, 1e-30) / 240.0))
    return s * 2 if amax > 240 * s else (s / 2 if amax <= 120 * s else s)


def mx4_values(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    nib = (
        np.stack([codes & 15, codes >> 4], -1)
        .reshape(codes.shape[0], -1)
        .astype(np.int64)
    )
    return E2M1[nib] * np.exp2(scales.astype(np.float64).repeat(32, 1) - 127)


@pytest.mark.parametrize(
    "lo, hi", [(117, 125), (106, 113)], ids=["clamped", "unclamped"]
)
def test_expert_block_matches_numpy(device, recipes, lo, hi):
    """Routed experts of one rank: FP8 input, mx4_decs into (token, route) slots, NT batch_gemm, clamped
    SwiGLU (natural order, tail rows in y), mx4_decs, NT batch_gemm, weighted sum. Scales lo..hi set
    whether the clamp at 10 acts."""
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    rng = np.random.default_rng(5)
    H, w0, t, E, T, R, lim = 5120, 256, 32, 8, 3, 6, 10.0
    w, S = w0 + t, R * T
    # w13 slot rows: gate 0..w0 | up 0..w0 | gate w0..w | up w0..w; w2 slot: (H rows, w inputs)
    c13 = rng.integers(0, 256, (E, 2 * w * H // 2), dtype=np.uint8)
    s13 = rng.integers(lo, hi, (E, 2 * w * H // 32), dtype=np.uint8)
    c2 = rng.integers(0, 256, (E, H * w // 2), dtype=np.uint8)
    s2 = rng.integers(lo, hi, (E, H * w // 32), dtype=np.uint8)
    k13, k2 = int(s13.max()) - 132, int(s2.max()) - 132
    x = rng.standard_normal((T, H)).astype(np.float32)
    ids = np.full((T, 64), -1, np.int32)
    ids[:, :R] = np.stack([rng.choice(E, R, replace=False) for _ in range(T)])
    wts = np.zeros((T, 64), np.float32)
    wts[:, :R] = rng.uniform(0.1, 1.0, (T, R))

    from resin_models_gaudi.deepseek_v41 import layers as L

    d = L.Dims(
        hidden=H, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=w, experts=E, top=R,
        vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=lim, route_scale=1.5, ranks=1,
    )  # fmt: skip
    g = L.LayerGraph(device, "t_ds41_experts", d, T)
    X = g.input("x", (H, T), F32)
    xq, sx = g.tensor((H, 1, S), F8), g.tensor((1, T), F32)
    g.node("rg_ds_xq_lin_f32", [X], [xq, sx], struct.pack("i", R))
    IDS, WTS = g.input("ids", (64, T), I32), g.input("wts", (64, T), F32)
    W2 = g.input("c2", (H * w // 2, E), U8)
    S2 = g.input("s2", (H * w // 32, E), U8)
    K2 = g.input("k2", (E,), I32)
    y13 = L._w13(g, xq, IDS, R)  # each slot's rows 0..w-1 and w..2w-1 apart
    C13 = g.input("cc13", (2 * w, E), F32)
    hq, hs = g.tensor((w, 1, S), F8), g.tensor((1, S), F32)
    g.node(
        "rg_ds_xact_lin_f32",
        [y13, y13, C13, sx, IDS],
        [hq, hs],
        struct.pack("ifii", R, lim, t, 0),
    )
    d2 = g.tensor((H * w, S), F8)
    g.node("rg_mx4_decs_f32", [W2, S2, IDS, K2], [d2], struct.pack("ii", R, 0))
    v2 = g.tensor((w, H, S), F8)
    g.node("reshape", [d2], [v2])
    y2 = g.tensor((H, 1, S), F32)
    g.node("batch_gemm", [hq, v2], [y2], struct.pack("??", False, True))
    C2 = g.input("cc2", (H, E), F32)
    SH, SHS = g.input("sh", (H, T), F32), g.input("shs", (1, T), F32)
    g.node(
        "rg_ds_xsum_f32",
        [y2, C2, hs, IDS, WTS, SH, SHS],
        [g.input("out", (H, T), F32)],
        struct.pack("i", R),
    )
    out = Run(device, recipes, g.g)(
        {"x13": c13, "x13.s": s13, "x": x, "ids": ids, "wts": wts, "c2": c2, "x13.k": np.full(E, k13, np.int32), "k2": np.full(E, k2, np.int32), "s2": s2,
         "cc13": np.full((E, 2 * w), 2.0**k13, np.float32), "cc2": np.full((E, H), 2.0**k2, np.float32),
         "sh": np.zeros((T, H), np.float32), "shs": np.ones((1, T), np.float32)},
        ["out"],
    )["out"]  # fmt: skip

    # numpy: the same FP8 roundings of the activations, f64 sums
    want = np.zeros((T, H))
    for tk in range(T):
        scale = pow2_scale(float(np.abs(x[tk]).max()))
        xv = fp8_value(fp8_byte(x[tk] / scale, np.zeros(H, np.int64))) * scale
        for r in range(R):
            e = ids[tk, r]
            W = mx4_values(
                c13[e].reshape(2 * w, H // 2), s13[e].reshape(2 * w, H // 32)
            )
            y = W @ xv
            gate = np.concatenate([y[:w0], y[2 * w0 : 2 * w0 + t]])
            up = np.concatenate([y[w0 : 2 * w0], y[2 * w0 + t :]])
            gate = np.minimum(gate, lim)
            h = gate / (1 + np.exp(-gate)) * np.clip(up, -lim, lim)
            hsc = pow2_scale(float(np.abs(h).max()))
            hv = fp8_value(fp8_byte(h / hsc, np.zeros(w, np.int64))) * hsc
            W2v = mx4_values(c2[e].reshape(H, w // 2), s2[e].reshape(H, w // 32))
            want[tk] += wts[tk, r] * (W2v @ hv)
    err = np.linalg.norm(out - want) / np.linalg.norm(want)
    clamped = (lo, hi) == (117, 125)
    assert err < 2e-3, f"relative error {err:.2e}"
    print(f"relative error {err:.2e} ({'clamped' if clamped else 'unclamped'})")


def test_ds_q8_p2_power_of_two_scales(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32

    rng = np.random.default_rng(6)
    K, T = 5120, 5
    x = (rng.standard_normal((T, K)) * np.exp2(rng.integers(-20, 20, (T, 1)))).astype(
        np.float32
    )
    g = Graph(device, "t_ds_q8_p2")
    X, SIN, NW = (
        g.input("x", (K, T), F32),
        g.input("sin", (1, T), F32),
        g.input("nw", (K,), F32),
    )
    q, sc = g.input("q", (K, T), F8), g.input("s", (1, T), F32)
    g.node("rg_ds_q8_p2_f32", [X, SIN, NW], [q, sc], struct.pack("iff", 1, 1e-6, 10.0))
    out = Run(device, recipes, g)(
        {"x": x, "sin": np.ones((1, T), np.float32), "nw": np.ones(K, np.float32)},
        ["q", "s"],
    )
    for t in range(T):
        s = float(out["s"][t, 0])
        assert s == pow2_scale(float(np.abs(x[t]).max())), t
        # tau order: out[4 i + j] = x[64 j + i] within each 256 block
        tau = x[t].reshape(-1, 4, 64).transpose(0, 2, 1).reshape(-1)
        np.testing.assert_array_equal(
            out["q"][t], fp8_byte(tau.astype(np.float64) / s, np.zeros(K, np.int64))
        )


def bf16_round(x: np.ndarray) -> np.ndarray:
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne, bf16_to_f32

    return bf16_to_f32(bf16_rne(np.asarray(x, np.float32)))


def test_window_attention_with_sink(device, recipes):
    """One prefill chunk: keys into pages (mla_store_p2), a slot over them, windowed softmax (ds_slot_pw) and the
    output with the sink (ds_slot_out), against numpy."""
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    rng = np.random.default_rng(7)
    D, hpg, P, C, win, PR = 512, 8, 200, 256, 128, 1024
    W, M = hpg * P, hpg * P
    Mp = -(-M // 64) * 64
    scale = D**-0.5
    keys = (rng.standard_normal((P, D)) * 2).astype(np.float32)
    q = rng.standard_normal((P, hpg, D)).astype(np.float32)
    sink = rng.uniform(-2, 4, hpg).astype(np.float32)
    rows = np.full((P, 8), -1, np.int32)
    n = np.arange(P)
    rows[:, 0] = 129 * (n // 128) + n % 128
    rows[:, 3] = 129 * (n // 128) + 128
    rows[:, 4] = n % 128
    slots = np.array([[0, 0, 0, 0, P, 0, 0, win]], np.int32)
    lat_pages = np.array([[0, 129]], np.int32)

    g = Graph(device, "t_ds41_window")
    arena = g.input("arena", (512, PR, 1), U8)
    sink_t = g.tensor((64, 1), I32)
    g.node(
        "rg_mla_store_p2_f32",
        [
            arena,
            g.input("rows", (8, P), I32),
            g.input("kn", (D, P), F32),
            g.input("ix", (128, P), BF16),
        ],
        [sink_t],
    )
    table = g.input("slots", (8, 1), I32)
    lp, rp = (
        g.input("lat_pages", (C // 128, 1), I32),
        g.input("rope_pages", (C // 128, 1), I32),
    )
    qs = g.tensor((D, M, 1), BF16)
    g.node("rg_mla_slot_q_f32", [g.input("q_all", (D, W), BF16), table], [qs])
    ks = g.tensor((D, C, 1), BF16)
    g.node(
        "rg_mla_gather_f32", [arena, lp, rp, table, sink_t], [ks], struct.pack("i", 0)
    )
    sc = g.tensor((C, M, 1), F32)
    g.node("batch_gemm", [qs, ks], [sc], struct.pack("??", False, True))
    Pm, mx, ls = (
        g.tensor((C, M, 1), BF16),
        g.tensor((Mp, 1), F32),
        g.tensor((Mp, 1), F32),
    )
    g.node(
        "rg_ds_slot_pw_f32", [sc, table], [Pm, mx, ls], struct.pack("fi", scale, hpg)
    )
    pv = g.tensor((D, M, 1), F32)
    g.node("batch_gemm", [Pm, ks], [pv], struct.pack("??", False, False))
    out = g.input("out", (D, W, 1), F32)
    g.node(
        "rg_ds_slot_out_f32",
        [pv, mx, ls, table, g.input("sink", (hpg,), F32)],
        [out],
        struct.pack("i", hpg),
    )
    got = Run(device, recipes, g)(
        {"arena": np.zeros((1, PR, 512), np.uint8), "rows": rows, "kn": keys,
         "ix": np.zeros((P, 128), np.uint16), "slots": slots, "lat_pages": lat_pages,
         "rope_pages": np.zeros((1, C // 128), np.int32),
         "q_all": bf16_rne(q.reshape(W, D)), "sink": sink},
        ["out"],
    )["out"].reshape(W, D)  # fmt: skip

    stored = np.empty_like(keys, np.float64)
    for t in range(P):
        s = pow2_scale(float(np.abs(keys[t]).max()))
        stored[t] = (
            fp8_value(fp8_byte(keys[t].astype(np.float64) / s, np.zeros(D, np.int64)))
            * s
        )
    qb = bf16_round(q).astype(np.float64)
    want = np.zeros((P, hpg, D))
    for t in range(P):
        k = stored[max(0, t - win + 1) : t + 1]
        s = np.einsum("hd,kd->hk", qb[t], k) * scale
        m = s.max(axis=1, keepdims=True)
        e = np.exp(s - m)
        want[t] = (bf16_round(e) @ k) / (
            e.sum(axis=1, keepdims=True) + np.exp(sink[:, None] - m)
        )
    err = np.linalg.norm(got - want.reshape(W, D)) / np.linalg.norm(want)
    print(f"window attention relative error {err:.2e}")
    assert err < 3e-3


def test_router_sqrt_softplus(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(8)
    E, K, T, scale = 384, 6, 9, 1.5
    logits = rng.standard_normal((T, E)).astype(np.float32) * 3
    bias = (rng.standard_normal(E) * 0.3).astype(np.float32)
    g = Graph(device, "t_ds41_route")
    ids, wts = g.input("ids", (64, T), I32), g.input("wts", (64, T), F32)
    g.node(
        "rg_moe_route_ssp_f32",
        [
            g.input("lg", (E, T), F32),
            g.input("bias", (E,), F32),
            g.input("meta", (4,), I32),
        ],
        [ids, wts],
        struct.pack("iif", K, E, scale),
    )
    out = Run(device, recipes, g)(
        {"lg": logits, "bias": bias, "meta": np.array([0, T, 0, 0], np.int32)},
        ["ids", "wts"],
    )
    s = np.sqrt(np.log1p(np.exp(logits.astype(np.float64))))
    for t in range(T):
        pick = np.argsort(-(s[t] + bias), kind="stable")[:K]
        assert sorted(out["ids"][t, :K].tolist()) == sorted(pick.tolist()), t
        w = s[t, out["ids"][t, :K]]
        np.testing.assert_allclose(
            out["wts"][t, :K], scale * w / (w.sum() + 1e-20), rtol=2e-5
        )


def test_engram_gather_and_gate(device, recipes):
    from resin_models_gaudi.deepseek_v41.weights import OCP, inc_values, mx8_rows
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32

    rng = np.random.default_rng(9)
    R, J, T, H, row0 = 1024, 24, 5, 5120, 3000
    raw = rng.integers(0, 256, (R, 256), dtype=np.uint8)
    raw = np.where((raw & 0x7F) == 0x7F, raw - 1, raw).astype(np.uint8)
    rscale = rng.integers(120, 124, (R, 8), dtype=np.uint8)
    conv = mx8_rows(raw, rscale, tau=False)
    idx = rng.integers(row0 - 500, row0 + R + 500, (T, J)).astype(np.int32)
    X = rng.standard_normal((T, 4 * H)).astype(np.float32)
    kv = rng.standard_normal((T, 5 * H)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, 4 * H).astype(np.float32)
    g = Graph(device, "t_ds41_engram")
    emb = g.input("emb", (256 * J, T), F32)
    table, rs, ix = (
        g.input("table", (256, R), F8),
        g.input("rs", (R,), F32),
        g.input("idx", (J, T), I32),
    )
    g.node("rg_eg_gather_f32", [table, rs, ix], [emb], struct.pack("i", row0))
    Xt, kvt, wt = (
        g.input("X", (4 * H, T), F32),
        g.input("kv", (5 * H, T), F32),
        g.input("w", (4 * H,), F32),
    )
    g.node(
        "rg_eg_gate_f32",
        [Xt, kvt, wt],
        [g.input("out", (4 * H, T), F32)],
        struct.pack("f", 1e-20),
    )
    out = Run(device, recipes, g)(
        {
            "table": conv.bytes,
            "rs": conv.row_scales,
            "idx": idx,
            "X": X,
            "kv": kv,
            "w": w,
        },
        ["emb", "out"],
    )
    rows = (
        inc_values(conv.bytes) * conv.row_scales[:, None]
    )  # the converted table (random bytes round a little)
    assert (
        np.count_nonzero(
            rows != OCP[raw] * np.exp2(rscale.astype(np.float64) - 127).repeat(32, 1)
        )
        == conv.inexact
    )
    want = np.zeros((T, J, 256))
    local = idx - row0
    have = (local >= 0) & (local < R)
    want[have] = rows[local[have]]
    np.testing.assert_array_equal(
        out["emb"].reshape(T, J, 256), want.astype(np.float32)
    )
    h = X.reshape(T, 4, H).astype(np.float64)
    key, value = (
        kv[:, : 4 * H].reshape(T, 4, H).astype(np.float64),
        kv[:, 4 * H :].astype(np.float64),
    )
    r = 1 / np.sqrt((h**2).mean(-1) + 1e-20) / np.sqrt((key**2).mean(-1) + 1e-20)
    dot = (h * w.reshape(4, H) * key).sum(-1) * r / np.sqrt(H)
    gate = 1 / (1 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)))
    ref = h + gate[..., None] * value[:, None, :]
    err = np.abs(out["out"].reshape(T, 4, H) - ref).max() / np.abs(ref).max()
    assert err < 1e-5, err


@pytest.mark.parametrize("inverse", [False, True])
def test_rope_adjacent_pairs(device, recipes, inverse):
    """ml4_rope with DeepSeek-V4.1 YaRN tables on the last 64 dims of each 512-dim head."""
    from resin_models_gaudi.deepseek_v41.rope import cs_table, frequencies
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(10)
    D, hpt, T = 512, 8, 6
    pos = np.array([0, 1, 127, 128, 4095, 70000])
    freqs = frequencies(64, 160000.0, 65536, 16.0, 32.0, 1.0)
    x = rng.standard_normal((T * hpt, D)).astype(np.float32)
    g = Graph(device, f"t_ds41_rope{int(inverse)}")
    g.node(
        "rg_ml4_rope_f32",
        [g.input("x", (D, T * hpt), F32), g.input("cs", (128, T), F32), g.input("s", (hpt,), F32),
         g.input("b", (hpt,), F32), g.input("lp", (T,), F32)],
        [g.input("y", (D, T * hpt), F32)],
        struct.pack("iii", hpt, D - 64, 0),
    )  # fmt: skip
    y = Run(device, recipes, g)(
        {"x": x, "cs": cs_table(pos, freqs, inverse), "s": np.zeros(hpt, np.float32),
         "b": np.ones(hpt, np.float32), "lp": np.zeros(T, np.float32)},
        ["y"],
    )["y"]  # fmt: skip
    angle = pos.astype(np.float32)[:, None] * freqs[None, :]
    rot = np.exp(1j * angle.astype(np.float64) * (-1 if inverse else 1))  # (T, 32)
    z = x[:, D - 64 :].astype(np.float64).reshape(T, hpt, 32, 2)
    zc = (z[..., 0] + 1j * z[..., 1]) * rot[:, None, :]
    want = x.astype(np.float64).copy()
    want[:, D - 64 :] = np.stack([zc.real, zc.imag], -1).reshape(T * hpt, 64)
    np.testing.assert_allclose(y, want, rtol=0, atol=2e-5)


E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_ref(x: np.ndarray, group: int, e4m3_scale: bool) -> np.ndarray:
    """The checkpoint's fp4_quant_kernel, in place: per group scale, nearest E2M1 value, ties to the even code."""
    xg = x.astype(np.float64).reshape(x.shape[0], -1, group)
    a = np.abs(xg).max(-1, keepdims=True)
    if e4m3_scale:
        v = np.maximum(a, 6 * 2.0**-9) / 6
        e = np.maximum(np.floor(np.log2(v)), -6)
        s = np.minimum(
            np.round(v / 2 ** (e - 3)) * 2 ** (e - 3), 448
        )  # numpy rounds ties to even
    else:
        s = 2 ** np.ceil(np.log2(np.maximum(a, 6 * 2.0**-126) / 6))
    r = np.clip(xg / s, -6, 6)
    d = np.abs(np.abs(r)[..., None] - E2M1_GRID)
    best = d.min(-1, keepdims=True)
    tie = (d == best).sum(-1) > 1
    code = np.argmin(d, -1)
    code = np.where(tie, np.where(code % 2 == 0, code, code + 1), code)
    return (np.sign(r) * E2M1_GRID[code] * s).reshape(x.shape)


@pytest.mark.parametrize("mode", [0, 1])
def test_fp4_rounding(device, recipes, mode):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(11)
    D, T = 512, 7
    x = (
        rng.standard_normal((T, D))
        * np.exp2(rng.integers(-8, 8, (T, D // 16)).repeat(16, 1))
    ).astype(np.float32)
    x[0, :32] = np.array(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4, np.float32
    )  # ties at scale 1
    g = Graph(device, f"t_ds41_fq4_{mode}")
    g.node(
        "rg_ds_fq4_f32",
        [g.input("x", (D, T), F32)],
        [g.input("y", (D, T), F32)],
        struct.pack("i", mode),
    )
    y = Run(device, recipes, g)({"x": x}, ["y"])["y"]
    want = fp4_ref(x, 32 if mode == 0 else 16, mode == 1)
    np.testing.assert_array_equal(y, want.astype(np.float32))


def test_compressor_pool_and_state(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(12)
    T, R = 6, 3
    kv = rng.standard_normal((T, 512)).astype(np.float32)
    score = (rng.standard_normal((T, 512)) * 3).astype(np.float32)
    state = rng.standard_normal((R, 1024)).astype(np.float32)
    # positions 5 .. 10: groups (4, 5) with 4 pending in state row 1, (6, 7), (8, 9); 10 stays open -> row 2
    groups = np.array([[-1, 0, 1, 0], [1, 2, 0, 0], [3, 4, 0, 0]], np.int32)
    rows = np.array([-1, -1, -1, -1, -1, 2], np.int32)
    g = Graph(device, "t_ds41_compress")
    KV, SC, ST = (
        g.input("kv", (512, T), F32),
        g.input("score", (512, T), F32),
        g.input("state", (1024, R), F32),
    )
    lat = g.input("latent", (512, 3), F32)
    g.node("rg_ds_cpool_f32", [KV, SC, ST, g.input("groups", (4, 3), I32)], [lat])
    sink = g.input(
        "sink", (64, 1), I32
    )  # a persistent sink: the node writes in place
    g.node("rg_ds_cstate_f32", [KV, SC, g.input("rows", (T,), I32), ST, lat], [sink])
    recipe = recipes.compile(g)
    arrays = {"kv": kv, "score": score, "state": state, "groups": groups, "rows": rows,
              "latent": np.zeros((3, 512), np.float32), "sink": np.zeros((1, 64), np.int32)}  # fmt: skip
    addr = {}
    try:
        for name, a in arrays.items():
            addr[name] = device.malloc(max(a.nbytes, 256))
            device.h2d(addr[name], np.ascontiguousarray(a))
            recipe.bind(name, addr[name])
        recipe.launch()
        device.sync()
        latent = device.d2h(addr["latent"], (3, 512), np.float32)
        new_state = device.d2h(addr["state"], (R, 1024), np.float32)
    finally:
        for a in addr.values():
            device.free(a)
    pairs = [(state[1, :512], state[1, 512:], kv[0], score[0]), (kv[1], score[1], kv[2], score[2]),
             (kv[3], score[3], kv[4], score[4])]  # fmt: skip
    for j, (ka, sa, kb, sb) in enumerate(pairs):
        m = np.maximum(sa, sb)
        ea, eb = np.exp(sa - m), np.exp(sb - m)
        np.testing.assert_allclose(
            latent[j], (ka * ea + kb * eb) / (ea + eb), rtol=2e-5, atol=1e-6
        )
    np.testing.assert_array_equal(new_state[2], np.concatenate([kv[5], score[5]]))
    np.testing.assert_array_equal(new_state[:2], state[:2])


def test_two_part_attention_with_sink(device, recipes):
    """A windowed slot over the chunk's own keys and a slot over other keys (all visible), each with its column max
    and sum (ds_slot_ml), merged with the sink (ds_attn_comb): one softmax over both key sets."""
    from resin_models_gaudi.qwen3_5.checkpoint import bf16_rne
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    rng = np.random.default_rng(13)
    D, hpg, P, NB, win, PR = 512, 8, 40, 300, 16, 2048
    W = hpg * P
    Mp = -(-W // 64) * 64
    scale = D**-0.5
    ka = (rng.standard_normal((P, D)) * 2).astype(np.float32)
    kb = (rng.standard_normal((NB, D)) * 2).astype(np.float32)
    q = rng.standard_normal((P, hpg, D)).astype(np.float32)
    sink = rng.uniform(-2, 4, hpg).astype(np.float32)
    # set A in page 0 (rows 0 .. 128), set B in pages 1 .. 3 (rows 129 ..)
    keys = np.concatenate([ka, kb])
    rows = np.full((P + NB, 8), -1, np.int32)
    n = np.concatenate(
        [np.arange(P), 128 + np.arange(NB)]
    )  # slot positions of each stored key
    rows[:, 0] = 129 * (n // 128) + n % 128
    rows[:, 3] = 129 * (n // 128) + 128
    rows[:, 4] = n % 128
    g = Graph(device, "t_ds41_two_part")
    arena = g.input("arena", (512, PR, 1), U8)
    stored = g.tensor((64, 1), I32)
    g.node("rg_mla_store_p2_f32", [arena, g.input("rows", (8, P + NB), I32), g.input("kn", (D, P + NB), F32),
                                   g.input("ix", (128, P + NB), BF16)], [stored])  # fmt: skip
    q_all = g.input("q_all", (D, W), BF16)
    parts = []
    for name, C, slot in (
        ("a", 128, [0, 0, 0, 0, P, 0, 0, win]),
        ("b", 384, [0, 0, 10**6, 0, NB, 0, 0, 0]),
    ):
        table = g.input(f"slots_{name}", (8, 1), I32)
        lp = g.input(f"lat_{name}", (C // 128, 1), I32)
        qs = g.tensor((D, W, 1), BF16)
        g.node("rg_mla_slot_q_f32", [q_all, table], [qs])
        ks = g.tensor((D, C, 1), BF16)
        g.node(
            "rg_mla_gather_f32",
            [arena, lp, lp, table, stored],
            [ks],
            struct.pack("i", 0),
        )
        sc = g.tensor((C, W, 1), F32)
        g.node("batch_gemm", [qs, ks], [sc], struct.pack("??", False, True))
        Pm, mx, ls = (
            g.tensor((C, W, 1), BF16),
            g.tensor((Mp, 1), F32),
            g.tensor((Mp, 1), F32),
        )
        g.node(
            "rg_ds_slot_pw_f32",
            [sc, table],
            [Pm, mx, ls],
            struct.pack("fi", scale, hpg),
        )
        pv = g.tensor((D, W, 1), F32)
        g.node("batch_gemm", [Pm, ks], [pv], struct.pack("??", False, False))
        o, mc, lc = (
            g.tensor((D, W, 1), F32),
            g.tensor((W, 1), F32),
            g.tensor((W, 1), F32),
        )
        g.node("rg_ds_slot_ml_f32", [pv, mx, ls, table], [o, mc, lc])
        parts += [o, mc, lc]
    out = g.input("out", (D, W, 1), F32)
    g.node(
        "rg_ds_attn_comb_f32",
        [*parts, g.input("sink", (hpg,), F32)],
        [out],
        struct.pack("i", hpg),
    )
    got = Run(device, recipes, g)(
        {"arena": np.zeros((1, PR, 512), np.uint8), "rows": rows, "kn": keys,
         "ix": np.zeros((P + NB, 128), np.uint16), "q_all": bf16_rne(q.reshape(W, D)), "sink": sink,
         "slots_a": np.array([[0, 0, 0, 0, P, 0, 0, win]], np.int32), "lat_a": np.array([[0]], np.int32),
         "slots_b": np.array([[0, 0, 10**6, 0, NB, 0, 0, 0]], np.int32),
         "lat_b": np.array([[129, 258, 387]], np.int32)},
        ["out"],
    )["out"].reshape(W, D)  # fmt: skip

    def stored_keys(k):
        out = np.empty_like(k, np.float64)
        for t in range(len(k)):
            s = pow2_scale(float(np.abs(k[t]).max()))
            out[t] = (
                fp8_value(fp8_byte(k[t].astype(np.float64) / s, np.zeros(D, np.int64)))
                * s
            )
        return out

    sa, sb = stored_keys(ka), stored_keys(kb)
    qb = bf16_round(q).astype(np.float64)
    want = np.zeros((P, hpg, D))
    for t in range(P):
        k = np.concatenate([sa[max(0, t - win + 1) : t + 1], sb])
        s = np.einsum("hd,kd->hk", qb[t], k) * scale
        m = np.maximum(s.max(axis=1, keepdims=True), sink[:, None])
        e = np.exp(s - m)
        want[t] = (bf16_round(e) @ k) / (
            e.sum(axis=1, keepdims=True) + np.exp(sink[:, None] - m)
        )
    err = np.linalg.norm(got - want.reshape(W, D)) / np.linalg.norm(want)
    print(f"two-part attention relative error {err:.2e}")
    assert err < 3e-3


def test_indexer_selection_ratio2(device, recipes):
    """Scores over compressed keys with ratio-2 visibility (ds_ix_slot_acc), then the top-512 lists (ix_thr, ix_bcnt,
    ix_list) in a second launch, against numpy."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(14)
    NH, T, K, C, top, real = 32, 5, 4096, 4096, 512, 1003
    pos = np.arange(2000, 2000 + T)
    S = rng.standard_normal((1, NH * T, C)).astype(np.float32)
    w = rng.standard_normal((T, NH)).astype(np.float32)
    vis = np.minimum((pos + 1) // 2, real).astype(np.int32)
    g1 = Graph(device, "t_ds41_index_score")
    g1.node("rg_ds_ix_slot_acc_f32", [g1.input("S", (C, NH * T, 1), F32), g1.input("w", (NH, T), F32),
                                       g1.input("slots", (8, 1), I32), g1.input("score", (K, T), F32)],
            [g1.input("ix_sink", (64, 1), I32)], struct.pack("f", 1.0))  # fmt: skip
    g2 = Graph(device, "t_ds41_index_select")
    score, V = g2.input("score", (K, T), F32), g2.input("vis", (T,), I32)
    thr = g2.tensor((64, T), I32)
    g2.node("rg_ix_thr_f32", [score, V], [thr], struct.pack("i", top))
    cnt, gsum = g2.tensor((K // 64, T), I32), g2.tensor((K // 4096, T), I32)
    g2.node("rg_ix_bcnt_f32", [score, V, thr], [cnt, gsum])
    g2.node("rg_ix_list_f32", [score, V, thr, cnt, gsum], [g2.input("list", (top, T), I32), g2.input("lslots", (8, T), I32)],
            struct.pack("iii", max(1, 48 // T), top, 8))  # fmt: skip
    arrays = {"S": S, "w": w, "slots": np.array([[0, 0, int(pos[0]), 0, real, 0, 2, 0]], np.int32),
              "score": np.zeros((T, K), np.float32), "vis": vis, "ix_sink": np.zeros((1, 64), np.int32),
              "list": np.zeros((T, top), np.int32), "lslots": np.zeros((T, 8), np.int32)}  # fmt: skip
    addr = {}
    try:
        for name, a in arrays.items():
            addr[name] = device.malloc(max(a.nbytes, 256))
            device.h2d(addr[name], np.ascontiguousarray(a))
        for graph in (g1, g2):
            recipe = recipes.compile(graph)
            for name in recipe.persistent:
                recipe.bind(name, addr[name])
            recipe.launch()
            device.sync()
        out = {
            n: device.d2h(addr[n], arrays[n].shape, arrays[n].dtype)
            for n in ("score", "list", "lslots")
        }
    finally:
        for a in addr.values():
            device.free(a)
    for t in range(T):
        sc = (np.maximum(S[0, NH * t : NH * (t + 1)], 0) * w[t][:, None]).sum(0)
        np.testing.assert_allclose(
            out["score"][t, : vis[t]], sc[: vis[t]], rtol=1e-5, atol=1e-5
        )
        assert np.all(out["score"][t, vis[t] : real] < -1e38), t
        order = np.argsort(-sc[: vis[t]], kind="stable")
        count = out["lslots"][t, 4]
        assert count == top, (t, count)
        got = out["list"][t, :count]
        assert len(set(got.tolist())) == count, (
            "a key listed twice"
        )  # strict keys ascending, then the ties
        # the device sums in f32: keys may trade places only at the boundary, within rounding of the 512th score
        diff = set(got.tolist()) ^ set(order[:top].tolist())
        assert len(diff) <= 4, (t, sorted(diff))
        edge = sc[order[top - 1]]
        assert all(abs(sc[k] - edge) <= 1e-5 * max(1.0, abs(edge)) for k in diff), t


def test_paged_experts_match_slots(device, recipes):
    """The MoE sub-layer with each expert decoded once per page of its rows equals the one decoded per
    (token, route)."""
    from dataclasses import replace

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import NUMPY_DTYPE

    rng = np.random.default_rng(12)
    d = L.Dims(
        hidden=5120, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=288, experts=24, top=6,
        vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0, route_scale=1.5, ranks=1,
    )  # fmt: skip
    T = 20
    inputs: dict[str, np.ndarray] = {}
    graphs = [L.ffn_graph(device, d, T), L.ffn_graph(device, replace(d), T, paged=True)]
    runs = [Run(device, recipes, g) for g in graphs]
    for name, t in runs[1].recipe.persistent.items():
        if name in inputs or name in ("p_out", "X_out", "c_out"):
            continue
        shape, dtype = tuple(reversed(t.shape)), NUMPY_DTYPE[t.dtype]
        if name == "meta":
            a = np.array([0, T, 0, 0], np.int32)
        elif name.endswith(".k"):
            a = np.full(shape, -10, np.int32)
        elif name.endswith((".c", "one")) or name.endswith("_scale"):
            a = np.full(shape, 2.0**-10 if name.endswith(".c") else 1.0, np.float32)
        elif name in ("x13.s", "x2.s"):
            a = rng.integers(118, 124, shape).astype(np.uint8)
        elif dtype == np.uint8:
            a = rng.integers(0, 256, shape).astype(np.uint8)
            if name in ("sh_w13", "sh_w2"):
                a = np.where((a & 0x78) == 0x78, 0, a).astype(np.uint8)
        elif name == "c":
            a = np.zeros(shape, np.float32)
            a[0] = 1.0
            a[8 + np.arange(4) * 5] = 1.0
        elif name.endswith(".s"):
            a = np.full(shape, 2.0**-8, np.float32)
        else:
            a = (0.1 * rng.standard_normal(shape)).astype(dtype)
        inputs[name] = a
    outs = [
        run(
            {k: v for k, v in inputs.items() if k in run.recipe.persistent},
            ["p_out", "X_out", "c_out"],
        )["p_out"]
        for run in runs
    ]
    for g in graphs:
        g.destroy()
    err = np.linalg.norm(outs[1] - outs[0]) / np.linalg.norm(outs[0])
    assert np.linalg.norm(outs[0]) > 0
    assert err < 1e-5, f"paged vs slots {err:.2e}"


def test_engram_hash_on_device(device, recipes):
    """eg_hash: the newest token's product from a table, XOR the older tokens' (from the host), 64-bit mod each
    lane's prime, plus its row offset; equal to EngramHash for the checkpoint's sizes."""
    from resin.models.deepseek_v41.engram import EngramHash, next_prime
    from resin_models_gaudi.deepseek_v41.weights import engram_hash_tables, split64
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import I32

    rng = np.random.default_rng(3)
    layers, n, heads, V, vocab = 2, 4, 8, 129280, 99092
    primes, seen = np.empty((layers, n - 1, heads), np.int64), set()
    for k in range(layers):
        for g in range(n - 1):
            cur = 16_000_000 - 1
            for h in range(heads):
                cur = next_prime(cur, seen)
                seen.add(cur)
                primes[k, g, h] = cur
    flat = primes.reshape(layers, -1)
    offsets = np.concatenate(
        [np.zeros((layers, 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1
    )
    bound = (np.iinfo(np.int64).max // vocab) // 2
    mult = np.stack(
        [rng.integers(0, bound, n, dtype=np.int64) * 2 + 1 for _ in range(layers)]
    )
    token_map = rng.integers(0, vocab, V).astype(np.int64)
    eh = EngramHash(token_map, int(token_map[2]), mult, primes, offsets)
    T = 200
    tok = rng.integers(0, V, T).astype(np.int32)
    older = token_map[rng.integers(0, V, (T, n - 1))]
    older[:7, 2] = eh.pad_id  # positions near the start: blocked tokens are the pad
    older[:3, 1] = eh.pad_id
    windows = np.concatenate([token_map[tok][:, None], older], 1)
    want = eh.hash(windows)
    xor = eh.older(older)
    for k in range(layers):
        g = Graph(device, "t_eg_hash")
        g.node(
            "rg_eg_hash_f32",
            [g.input("tok", (T,), I32), g.input("a0", (2, V), I32), g.input("xh", (64, T), I32),
             g.input("xl", (64, T), I32), g.input("hp", (64, 4), I32)],
            [g.input("rows", (24, T), I32)],
        )  # fmt: skip
        tables = engram_hash_tables(eh, k)
        lanes = np.zeros((T, 64), np.int64)
        lanes[:, :24] = xor[:, k].repeat(heads, 1)
        x = split64(lanes)
        got = Run(device, recipes, g)(
            {
                "tok": tok,
                "a0": tables["eg_a0"],
                "xh": x[..., 0],
                "xl": x[..., 1],
                "hp": tables["eg_hp"],
            },
            ["rows"],
        )["rows"]
        g.destroy()
        bad = np.argwhere(got != want[:, k])
        assert not len(bad), (k, bad[:5], got[tuple(bad[0])], want[:, k][tuple(bad[0])])


@pytest.mark.parametrize("D, hpt", [(512, 8), (128, 32)])
@pytest.mark.parametrize("mode", range(8))
def test_ds_rope_norm_scale_bf16(device, recipes, D, hpt, mode):
    """ds_rope: optional RMSNorm, optional per-token scale, RoPE on the last 64 rows (adjacent pairs), f32 or
    natural-order bf16, against numpy."""
    from resin_models_gaudi.deepseek_v41.rope import cs_table
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    rng = np.random.default_rng(mode)
    T, eps = 3, 1e-6
    N = T * hpt
    x = rng.standard_normal((N, D)).astype(np.float32)
    nw = rng.uniform(0.5, 1.5, D).astype(np.float32)
    ts = np.exp2(rng.integers(-4, 4, T)).astype(np.float32)
    freqs = (1.0 / 10000 ** (np.arange(32) / 32)).astype(np.float32)
    cs = cs_table(np.array([5, 900, 4097]), freqs)
    g = Graph(device, f"t_ds_rope_{D}_{mode}")
    g.node(
        "rg_ds_rope_f32",
        [g.input("x", (D, N), F32), g.input("cs", (128, T), F32), g.input("ts", (T,), F32), g.input("nw", (D,), F32)],
        [g.input("y", (D, N), BF16 if mode & 4 else F32)],
        struct.pack("iif", hpt, mode, eps),
    )  # fmt: skip
    got = Run(device, recipes, g)({"x": x, "cs": cs, "ts": ts, "nw": nw}, ["y"])["y"]
    g.destroy()
    if mode & 4:
        got = (got.astype(np.uint32) << 16).view(np.float32)
    v = x.astype(np.float64)
    if mode & 1:
        v = v / np.sqrt((v * v).mean(1, keepdims=True) + eps) * nw
    if mode & 2:
        v = v * ts[np.arange(N) // hpt, None]
    t = np.arange(N) // hpt
    co, si = cs[t, :64].astype(np.float64), cs[t, 64:].astype(np.float64)
    r = v[:, D - 64 :]
    swap = r.reshape(N, 32, 2)[:, :, ::-1].reshape(N, 64)
    v[:, D - 64 :] = r * co + swap * si
    tol = 8e-3 if mode & 4 else 1e-5
    err = np.abs(got - v).max() / np.abs(v).max()
    assert err < tol, f"mode {mode}: {err:.2e}"


@pytest.mark.parametrize(
    "build, mode, K",
    [("z", 0, 4096), ("n", 2, 1024), ("s", 5, 288), ("z", 8, 1024), ("s", 13, 288)],
)
def test_ds_q8_builds_equal_the_generic_kernel(device, recipes, build, mode, K):
    """ds_q8z/n/s_p2 (one mode each) write the same bytes and scales as ds_q8_p2."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32

    rng = np.random.default_rng(mode)
    T = 3
    rows = 2 * K if mode & 4 else K
    inputs = {
        "x": (rng.standard_normal((T, rows)) * 3).astype(np.float32),
        "sin": np.exp2(rng.integers(-3, 3, (1, T))).astype(np.float32),
        "nw": rng.uniform(0.5, 1.5, K).astype(np.float32),
    }
    outs = []
    for guid in (f"rg_ds_q8{build}_p2_f32", "rg_ds_q8_p2_f32"):
        g = Graph(device, f"t_q8_{guid}_{mode}_{K}")
        g.node(
            guid,
            [
                g.input("x", (rows, T), F32),
                g.input("sin", (1, T), F32),
                g.input("nw", (K,), F32),
            ],
            [g.input("q", (-(-K // 256) * 256, T), F8), g.input("s", (1, T), F32)],
            struct.pack("iff", mode, 1e-6, 10.0),
        )
        outs.append(Run(device, recipes, g)(inputs, ["q", "s"]))
        g.destroy()
    np.testing.assert_array_equal(outs[0]["q"], outs[1]["q"])
    np.testing.assert_array_equal(outs[0]["s"], outs[1]["s"])


@pytest.mark.parametrize(
    "pair",
    [
        ("rg_ds_slot_pw1_f32", "rg_ds_slot_pw_f32"),
        ("rg_ds_slot_p1_f32", "rg_mla_slot_p_f32"),
    ],
)
def test_slot_softmax_per_column_builds(device, recipes, pair):
    """ds_slot_pw1 / ds_slot_p1 (one member per column) write what ds_slot_pw / mla_slot_p write."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    rng = np.random.default_rng(4)
    C, M, S, hpg = 384, 8, 3, 8
    slots = np.zeros((S, 8), np.int32)
    for s in range(S):
        q = 300 + 50 * s
        slots[s] = (
            0,
            M * s,
            q,
            (q - 127) // 128 * 128,
            q + 1,
            0,
            0,
            128 if "pw" in pair[0] else 0,
        )
    inputs = {
        "sc": (rng.standard_normal((S, M, C)) * 4).astype(np.float32),
        "slots": slots,
    }
    outs = []
    for guid in pair:
        g = Graph(device, f"t_{guid}")
        g.node(
            guid,
            [g.input("sc", (C, M, S), F32), g.input("slots", (8, S), I32)],
            [
                g.input("P", (C, M, S), BF16),
                g.input("mx", (64, S), F32),
                g.input("l", (64, S), F32),
            ],
            struct.pack("fi", 0.044, hpg),
        )
        outs.append(Run(device, recipes, g)(inputs, ["P", "mx", "l"]))
        g.destroy()
    np.testing.assert_array_equal(outs[0]["P"], outs[1]["P"])
    np.testing.assert_array_equal(outs[0]["mx"][:, :M], outs[1]["mx"][:, :M])
    np.testing.assert_array_equal(outs[0]["l"][:, :M], outs[1]["l"][:, :M])


def test_slot_merge_per_column_build(device, recipes):
    """ds_slot_ml1 (one member per column) writes what ds_slot_ml writes: decode slots, one per column."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(5)
    D, M, S = 512, 8, 3
    W = M * S
    slots = np.zeros((S, 8), np.int32)
    for s in range(S):
        slots[s, :2] = (0, M * s)
    inputs = {
        "O": rng.standard_normal((S, M, D)).astype(np.float32),
        "mx": rng.standard_normal((S, 64)).astype(np.float32),
        "l": rng.uniform(0.5, 4, (S, 64)).astype(np.float32),
        "slots": slots,
    }
    outs = []
    for guid in ("rg_ds_slot_ml1_f32", "rg_ds_slot_ml_f32"):
        g = Graph(device, f"t_{guid}")
        g.node(
            guid,
            [g.input("O", (D, M, S), F32), g.input("mx", (64, S), F32), g.input("l", (64, S), F32),
             g.input("slots", (8, S), I32)],
            [g.input("out", (D, W, 1), F32), g.input("m", (W, 1), F32), g.input("lc", (W, 1), F32)],
        )  # fmt: skip
        outs.append(Run(device, recipes, g)(inputs, ["out", "m", "lc"]))
        g.destroy()
    for k in ("out", "m", "lc"):
        np.testing.assert_array_equal(outs[0][k], outs[1][k])


def test_attention_merge_with_rope(device, recipes):
    """ds_attn_comb_r writes what ds_attn_comb followed by ml4_rope (no scale) writes."""
    from resin_models_gaudi.deepseek_v41.rope import cs_table
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(6)
    D, nh, T = 512, 8, 3
    W = nh * T
    freqs = (1.0 / 10000 ** (np.arange(32) / 32)).astype(np.float32)
    inputs = {
        "o1": rng.standard_normal((1, W, D)).astype(np.float32),
        "o2": rng.standard_normal((1, W, D)).astype(np.float32),
        "m1": rng.standard_normal((1, W)).astype(np.float32),
        "m2": rng.standard_normal((1, W)).astype(np.float32),
        "l1": rng.uniform(0.5, 3, (1, W)).astype(np.float32),
        "l2": rng.uniform(0.5, 3, (1, W)).astype(np.float32),
        "sink": rng.standard_normal(nh).astype(np.float32),
        "cs": cs_table(np.array([7, 300, 5000]), freqs, True),
        "rs": np.zeros(nh, np.float32),
        "rb": np.ones(nh, np.float32),
        "rlp": np.zeros(T, np.float32),
    }

    def parts(g):
        return [g.input("o1", (D, W, 1), F32), g.input("m1", (W, 1), F32), g.input("l1", (W, 1), F32),
                g.input("o2", (D, W, 1), F32), g.input("m2", (W, 1), F32), g.input("l2", (W, 1), F32),
                g.input("sink", (nh,), F32)]  # fmt: skip

    g = Graph(device, "t_comb_r")
    g.node(
        "rg_ds_attn_comb_r_f32",
        [*parts(g), g.input("cs", (128, T), F32)],
        [g.input("y", (D, W, 1), F32)],
        struct.pack("i", nh),
    )
    got = Run(device, recipes, g)(
        {k: v for k, v in inputs.items() if k not in ("rs", "rb", "rlp")}, ["y"]
    )["y"]
    g.destroy()
    g = Graph(device, "t_comb_then_rope")
    o = g.tensor((D, W, 1), F32)
    g.node("rg_ds_attn_comb_f32", parts(g), [o], struct.pack("i", nh))
    flat = g.tensor((D, W), F32)
    g.node("reshape", [o], [flat])
    g.node(
        "rg_ml4_rope_f32",
        [
            flat,
            g.input("cs", (128, T), F32),
            g.input("rs", (nh,), F32),
            g.input("rb", (nh,), F32),
            g.input("rlp", (T,), F32),
        ],
        [g.input("y", (D, W), F32)],
        struct.pack("iii", nh, D - 64, 0),
    )
    want = Run(device, recipes, g)(inputs, ["y"])["y"]
    g.destroy()
    np.testing.assert_array_equal(got.reshape(want.shape), want)


def test_mx4_decs_unit_offset(device, recipes):
    """mx4_decs with a unit offset u0 decodes elements 2048 u0 .. of each selected row: the matching slice of a
    decode from 0."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, I32, U8

    rng = np.random.default_rng(8)
    E, K, R, T = 5, 3 * 2048 * 4, 2, 3
    codes = rng.integers(0, 256, (E, K // 2), dtype=np.uint8)
    scales = rng.integers(118, 126, (E, K // 32), dtype=np.uint8)
    ids = np.full((T, 64), -1, np.int32)
    ids[:, :R] = rng.integers(0, E, (T, R))
    outs = {}
    for u0, n in ((0, K), (4, K // 3), (8, K // 3)):
        g = Graph(device, f"t_decs_u{u0}")
        g.node(
            "rg_mx4_decs_f32",
            [g.input("c", (K // 2, E), U8), g.input("s", (K // 32, E), U8), g.input("ids", (64, T), I32),
             g.input("k", (E,), I32)],
            [g.input("o", (n, R * T), F8)],
            struct.pack("ii", R, u0),
        )  # fmt: skip
        outs[u0] = Run(device, recipes, g)(
            {"c": codes, "s": scales, "ids": ids, "k": np.full(E, -3, np.int32)}, ["o"]
        )["o"]
        g.destroy()
    for u0 in (4, 8):
        np.testing.assert_array_equal(
            outs[u0], outs[0][:, 2048 * u0 : 2048 * u0 + K // 3]
        )


def test_expert_product_in_parts_equals_whole(device, recipes):
    """_halves: a slot's rows decoded and multiplied in two parts equal the whole product (w2 at T <= 4)."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import F8, F32, I32

    rng = np.random.default_rng(11)
    H, w, E, R, T = 5120, 288, 6, 6, 1
    d = L.Dims(
        hidden=H, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=w, experts=E, top=R,
        vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0, route_scale=1.5, ranks=1,
    )  # fmt: skip
    hq = rng.integers(0, 0x70, (R * T, 1, w)).astype(np.uint8)
    inputs = {
        "h": hq, "ids": np.tile(np.arange(64, dtype=np.int32), (T, 1)) % E,
        "x2": rng.integers(0, 256, (E, H * w // 2), dtype=np.uint8),
        "x2.s": rng.integers(118, 124, (E, H * w // 32), dtype=np.uint8), "x2.k": np.full(E, -10, np.int32),
    }  # fmt: skip
    outs = []
    for n in (1, 2):
        g = L.LayerGraph(device, f"t_parts_{n}", d, T)
        h = g.input("h", (w, 1, R * T), F8)
        ids = g.input("ids", (64, T), I32)
        y = L._halves(g, h, ids, R, "x2", H * w, (w, H // n), n)
        g.node("memcpy", [y], [g.input("y", (H, 1, R * T), F32)])
        outs.append(Run(device, recipes, g.g)(inputs, ["y"])["y"])
        g.g.destroy()
    assert np.isfinite(outs[0]).all()
    np.testing.assert_array_equal(outs[1], outs[0])


def test_engram_gate_over_gathered_slices(device, recipes):
    """eg_gate_g (kv as the ranks' row slices, all-gathered; a member per stream) equals eg_gate."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(13)
    H, T, R = 5120, 3, 8
    C = 5 * H // R
    X = rng.standard_normal((T, 4 * H)).astype(np.float32)
    kv = rng.standard_normal((T, 5 * H)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, 4 * H).astype(np.float32)
    # all-gather of the slices: rank r's (C, T) block, ranks outermost -> numpy (R, T, C)
    gathered = np.ascontiguousarray(kv.reshape(T, R, C).transpose(1, 0, 2))
    outs = []
    for guid, kv_in, shape in (("rg_eg_gate_f32", kv, (5 * H, T)), ("rg_eg_gate_g_f32", gathered, (C, T, R))):
        g = Graph(device, f"t_{guid}")
        g.node(
            guid,
            [g.input("X", (4 * H, T), F32), g.input("kv", shape, F32), g.input("w", (4 * H,), F32)],
            [g.input("out", (4 * H, T), F32)],
            struct.pack("f", 1e-6),
        )
        outs.append(Run(device, recipes, g)({"X": X, "kv": kv_in, "w": w}, ["out"])["out"])
        g.destroy()
    np.testing.assert_array_equal(outs[1], outs[0])


def candidate_blocks_ref(score: np.ndarray, vis: int, top: int, block: int = 8) -> np.ndarray:
    """The reference's select_candidate_blocks for one decode column: a mask over the first ``vis`` entries."""
    s = np.where(np.arange(len(score)) < vis, score, -np.inf)[: -(-vis // block) * block]
    bm = s.reshape(-1, block).max(1)
    bm[(vis - 1) // block] = np.inf
    keep = np.zeros(len(bm), bool)
    order = np.argsort(-bm, kind="stable")[:top]
    keep[order] = bm[order] > -np.inf
    return np.repeat(keep, block)[:vis]


def test_candidate_blocks_mask(device, recipes):
    """ix_bmax (block maxima, newest block pinned), ix_thr over the blocks (top 2048) and ix_cmask against the
    reference's two-level selection, for columns with more, exactly and fewer than 2048 visible blocks."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(20)
    K, top, NINF = 32768, 2048, -3.0e38
    vis = np.array([20001, 16384, 5000, 32768], np.int32)
    T = len(vis)
    visb = np.minimum(-(-(-(-vis // 8)) // 64) * 64, K // 8).astype(np.int32)
    score = rng.standard_normal((T, K)).astype(np.float32)
    g = Graph(device, "t_ds41_candidates")
    sc, V, VB = g.input("score", (K, T), F32), g.input("vis", (T,), I32), g.input("visb", (T,), I32)
    bmax, cthr = g.input("bmax", (K // 8, T), F32), g.input("cthr", (64, T), I32)
    g.node("rg_ix_bmax_f32", [sc, V], [bmax])
    g.node("rg_ix_thr_f32", [bmax, VB], [cthr], struct.pack("i", top))
    g.node("rg_ix_cmask_f32", [sc, V, bmax, cthr], [g.input("out", (K, T), F32)])
    arrays = {"score": score, "vis": vis, "visb": visb, "bmax": np.zeros((T, K // 8), np.float32),
              "cthr": np.zeros((T, 64), np.int32), "out": np.zeros((T, K), np.float32)}  # fmt: skip
    addr = {}
    try:
        for name, a in arrays.items():
            addr[name] = device.malloc(max(a.nbytes, 256))
            device.h2d(addr[name], np.ascontiguousarray(a))
        recipe = recipes.compile(g)
        for name in recipe.persistent:
            recipe.bind(name, addr[name])
        recipe.launch()
        device.sync()
        out = {n: device.d2h(addr[n], arrays[n].shape, arrays[n].dtype) for n in ("bmax", "out")}
    finally:
        for a in addr.values():
            device.free(a)
    b = np.arange(K // 8)
    lane = 64 * (b // 64) + 8 * (b % 8) + (b % 64) // 8  # where block b is stored
    for t in range(T):
        n = vis[t]
        s = np.where(np.arange(K) < n, score[t], NINF).reshape(-1, 8).max(1)
        s[(n - 1) // 8] = 3.0e38
        np.testing.assert_array_equal(out["bmax"][t, lane], s.astype(np.float32))
        keep = candidate_blocks_ref(score[t], n, top)
        assert keep.sum() == min(top, -(-n // 8)) * 8 - (-n % 8), t
        np.testing.assert_array_equal(out["out"][t, :n], np.where(keep, score[t, :n], np.float32(NINF)))


def test_ds_gather_list_equals_mla_gather_list(device, recipes):
    """ds_gather_list (8 keys a member, table reads first) writes the same slab as mla_gather_list, for full,
    partial and empty lists of positions spread over many pages."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, I32, U8

    rng = np.random.default_rng(41)
    PR, NB, C, P = 256, 64, 512, 32
    T = 3
    arena = rng.integers(0, 256, (NB, PR, 512), dtype=np.uint8)
    arena[(arena & 0x78) == 0x78] &= 0xF7  # no Inf / NaN bytes
    rows = arena.reshape(NB * PR, 512)
    blocks = (129 * (np.arange(T * P).reshape(T, P) + 1)).astype(np.int32)
    for b in blocks.ravel():
        rows[b + 128] = rng.standard_normal(128).astype(np.float32).view(np.uint8)
    counts = [C, 300, 0]
    lst = np.zeros((T, C), np.int32)
    slots = np.zeros((T, 8), np.int32)
    for t, n in enumerate(counts):
        lst[t, :n] = np.sort(rng.choice(P * 128, n, replace=False))
        slots[t, 4] = n
    loff = np.zeros(64, np.int32)
    loff[1] = -1
    outs = []
    for guid in ("rg_mla_gather_list_f32", "rg_ds_gather_list_f32"):
        g = Graph(device, f"t_{guid}")
        g.node(
            guid,
            [g.input("arena", (512, PR, NB), U8), g.input("list", (C, T), I32), g.input("blocks", (P, T), I32),
             g.input("loff", (64,), I32), g.input("slots", (8, T), I32), g.input("after", (64,), I32)],
            [g.input("out", (512, C, T), BF16)],
        )  # fmt: skip
        arrays = {"arena": arena, "list": lst, "blocks": blocks, "loff": loff, "slots": slots,
                  "after": np.zeros(64, np.int32), "out": np.full((T, C, 512), 0x7FC0, np.uint16)}  # fmt: skip
        addr = {}
        try:
            for name, a in arrays.items():
                addr[name] = device.malloc(max(a.nbytes, 256))
                device.h2d(addr[name], np.ascontiguousarray(a))
            recipe = recipes.compile(g)
            for name in recipe.persistent:
                recipe.bind(name, addr[name])
            recipe.launch()
            device.sync()
            outs.append(device.d2h(addr["out"], (T, C, 512), np.uint16))
        finally:
            for a in addr.values():
                device.free(a)
        g.destroy()
    np.testing.assert_array_equal(outs[1], outs[0])
    assert not np.any(outs[0][2]) and not np.any(outs[0][1, 300:])


def test_radix_threshold_equals_bisection(device, recipes):
    """Eight ix_rcnt / ix_rsel passes give ix_thr's threshold, with ties, zeros of both signs, a column that sees
    at most KS keys and members of uneven sizes."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(7)
    K, top, M = 32768, 512, 5
    vis = np.array([30001, 512, 20000, 4097], np.int32)
    T = len(vis)
    score = rng.standard_normal((T, K)).astype(np.float32)
    score[0, :3000] = np.round(score[0, :3000], 1)  # many exact ties
    score[2, ::7] = 0.0
    score[2, 1::7] = -0.0
    score[3] = np.abs(score[3]) * 1e-30
    g = Graph(device, "t_ds41_radix")
    sc, V = g.input("score", (K, T), F32), g.input("vis", (T,), I32)
    g.node("rg_ix_thr_f32", [sc, V], [g.input("thr", (64, T), I32)], struct.pack("i", top))
    state = g.input("zero", (64, T), I32)
    for sh in range(28, -1, -4):
        part = g.tensor((64, M, T), I32)
        g.node("rg_ix_rcnt_f32", [sc, V, state], [part], struct.pack("ii", sh, top))
        nxt = g.input("rthr", (64, T), I32) if sh == 0 else g.tensor((64, T), I32)
        g.node("rg_ix_rsel_f32", [part, V, state], [nxt], struct.pack("ii", sh, top))
        state = nxt
    arrays = {"score": score, "vis": vis, "thr": np.zeros((T, 64), np.int32), "zero": np.zeros((T, 64), np.int32),
              "rthr": np.zeros((T, 64), np.int32)}  # fmt: skip
    addr = {}
    try:
        for name, a in arrays.items():
            addr[name] = device.malloc(max(a.nbytes, 256))
            device.h2d(addr[name], np.ascontiguousarray(a))
        recipe = recipes.compile(g)
        for name in recipe.persistent:
            recipe.bind(name, addr[name])
        recipe.launch()
        device.sync()
        out = {n: device.d2h(addr[n], (T, 64), np.int32) for n in ("thr", "rthr")}
    finally:
        for a in addr.values():
            device.free(a)
    np.testing.assert_array_equal(out["rthr"], out["thr"])
    assert out["thr"][1, 0] == 1 and len(set(out["thr"][0])) == 1


def run_graph(device, recipes, g, arrays, outs):
    addr = {}
    try:
        for name, a in arrays.items():
            addr[name] = device.malloc(max(a.nbytes, 256))
            device.h2d(addr[name], np.ascontiguousarray(a))
        recipe = recipes.compile(g)
        for name in recipe.persistent:
            recipe.bind(name, addr[name])
        recipe.launch()
        device.sync()
        return {n: device.d2h(addr[n], arrays[n].shape, arrays[n].dtype) for n in outs}
    finally:
        for a in addr.values():
            device.free(a)


def test_candidate_gather(device, recipes):
    """ix_cgather: the FP8 index keys and scales of each slot's candidate blocks (ix_bmax's block order in the
    list) and their positions, for a full list and a short one."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    rng = np.random.default_rng(9)
    PR, NB, P, T, KB, KC, ooff, page_rows = 256, 64, 40, 2, 1024, 2048, 7, 129
    arena = rng.integers(0, 256, (NB, PR, 512), dtype=np.uint8)
    rows = arena.reshape(NB * PR, 512)
    blocks = (200 * (np.arange(T * P).reshape(T, P) + 1)).astype(np.int32)
    counts = [256, 200]
    cand_b = [rng.choice(P * 16, n, replace=False) for n in counts]
    cand = np.zeros((T, KB), np.int32)
    cslots = np.zeros((T, 8), np.int32)
    for c, b in enumerate(cand_b):
        cand[c, : len(b)] = (b & ~63) + 8 * (b & 7) + ((b >> 3) & 7)
        cslots[c, 4] = len(b)
    loff = np.zeros(64, np.int32)
    loff[0] = ooff
    slots = np.zeros((T, 8), np.int32)
    slots[:, 5] = np.arange(T)
    g = Graph(device, "t_ds41_cgather")
    g.node(
        "rg_ix_cgather_f32",
        [g.input("arena", (512, PR, NB), U8), g.input("cand", (KB, T), I32), g.input("cslots", (8, T), I32),
         g.input("blocks", (P, T), I32), g.input("loff", (64,), I32), g.input("slots", (8, T), I32),
         g.input("after", (64,), I32)],
        [g.input("out", (256, 1024, T), F8), g.input("scales", (2048, T), F32), g.input("cpos", (KC, T), I32)],
        struct.pack("i", page_rows),
    )  # fmt: skip
    out = run_graph(device, recipes, g, {
        "arena": arena, "cand": cand, "cslots": cslots, "blocks": blocks, "loff": loff, "slots": slots,
        "after": np.zeros(64, np.int32), "out": np.zeros((T, 1024, 256), np.uint8),
        "scales": np.zeros((T, 2048), np.float32), "cpos": np.full((T, KC), -1, np.int32)},
        ["out", "scales", "cpos"])  # fmt: skip
    out["out"] = out["out"].reshape(T, 2048, 128)
    for c in range(T):
        b = np.zeros(256, np.int64)
        b[: counts[c]] = cand_b[c]
        p = (8 * b[:, None] + np.arange(8)).ravel()
        np.testing.assert_array_equal(out["cpos"][c, :2048], p)
        r0 = blocks[c, p >> 7].astype(np.int64) + ooff + page_rows  # index page rows
        e = p & 127
        want = rows[r0 + (e >> 2)].reshape(-1, 4, 128)[np.arange(len(e)), e & 3]
        np.testing.assert_array_equal(out["out"][c], want)
        sc = rows[r0 + 32].view(np.float32)[np.arange(len(e)), e]
        np.testing.assert_array_equal(out["scales"][c].view(np.uint32), sc.view(np.uint32))


def test_candidate_select_maps_positions(device, recipes):
    """ix_cfix (entries at or past the visible positions dropped), ix_thr, ix_bcnt and ix_list_map: the top 512
    compact entries listed as their positions."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(10)
    KC, T, top = 4096, 3, 512
    score = rng.standard_normal((T, KC)).astype(np.float32)
    cpos = np.stack([rng.permutation(50000)[:KC] for _ in range(T)]).astype(np.int32)
    vis = np.array([40000, 50000, 10000], np.int32)
    g = Graph(device, "t_ds41_cselect")
    sc, cp, V = g.input("score", (KC, T), F32), g.input("cpos", (KC, T), I32), g.input("vis", (T,), I32)
    VC = g.input("visc", (T,), I32)
    m = g.tensor((KC, T), F32)
    g.node("rg_ix_cfix_f32", [sc, cp, V], [m])
    thr = g.tensor((64, T), I32)
    g.node("rg_ix_thr_f32", [m, VC], [thr], struct.pack("i", top))
    cnt, gsum = g.tensor((KC // 64, T), I32), g.tensor((KC // 4096, T), I32)
    g.node("rg_ix_bcnt_f32", [m, VC, thr], [cnt, gsum])
    g.node("rg_ix_list_map_f32", [m, VC, thr, cnt, gsum, cp],
           [g.input("list", (top, T), I32), g.input("lslots", (8, T), I32)], struct.pack("iii", max(1, 48 // T), top, 8))  # fmt: skip
    out = run_graph(device, recipes, g, {
        "score": score, "cpos": cpos, "vis": vis, "visc": np.full(T, KC, np.int32),
        "list": np.zeros((T, top), np.int32), "lslots": np.zeros((T, 8), np.int32)}, ["list", "lslots"])  # fmt: skip
    for t in range(T):
        live = cpos[t] < vis[t]
        order = np.argsort(-np.where(live, score[t], -np.inf), kind="stable")[:top]
        want = set(cpos[t, order[live[order]]].tolist())
        n = out["lslots"][t, 4]
        assert n == min(top, live.sum()) or n == top, (t, n)
        got = out["list"][t, :n]
        assert set(got.tolist()) == want, t


def fp8_rows(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keys (n, 128) as FP8 bytes with a power-of-two scale each (mla_store_i8's storage)."""
    s = np.array([pow2_scale(float(np.abs(k).max())) for k in keys])
    v = keys / s[:, None]
    return fp8_byte(v, np.zeros_like(v, np.int64)), s.astype(np.float32)


def test_index_keys_fp8_store_and_gather(device, recipes):
    """mla_store_i8 writes FP8 index keys and their power-of-two scales into an index page; ds_ix_gather reads a
    slot's keys and scales back (keys past klen: zeros). FP4-valued keys (the reference's rounding) come back
    exactly."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    rng = np.random.default_rng(12)
    PR, NB, B, T = 256, 8, 1000, 6
    keys = fp4_ref(rng.standard_normal((T, 128)) * np.exp2(rng.integers(-6, 6, (T, 1))), 32, False)
    e = np.array([0, 1, 2, 5, 64, 127])
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 2] = 4 * B + e
    rows[:, 5] = B + 32
    rows[:, 6] = e
    g = Graph(device, "t_ds41_ix8")
    arena = g.input("arena", (512, PR, NB), U8)
    sink = g.input("sink", (64, 1), I32)
    g.node("rg_mla_store_i8_f32", [arena, g.input("rows", (8, T), I32), g.input("kn", (512, T), F32),
                                   g.input("ix", (128, T), F32)], [sink])  # fmt: skip
    g.node("rg_ds_ix_gather_f32", [arena, g.input("pages", (1, 1), I32), g.input("slots", (8, 1), I32), sink],
           [g.input("out", (256, 64, 1), F8), g.input("scales", (128, 1), F32)])  # fmt: skip
    klen = 100
    out = run_graph(device, recipes, g, {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32), "rows": rows,
        "kn": np.zeros((T, 512), np.float32), "ix": keys.astype(np.float32), "pages": np.array([[B]], np.int32),
        "slots": np.array([[0, 0, 0, 0, klen, 0, 1, 0]], np.int32), "out": np.full((1, 64, 256), 0x55, np.uint8),
        "scales": np.full((1, 128), -1.0, np.float32)}, ["out", "scales"])  # fmt: skip
    out["out"] = out["out"].reshape(1, 128, 128)
    got = fp8_value(out["out"][0]) * out["scales"][0][:, None]
    for t in range(T):
        if e[t] < klen:
            np.testing.assert_array_equal(got[e[t]], keys[t])
            assert out["scales"][0, e[t]] == pow2_scale(float(np.abs(keys[t]).max()))
    assert not np.any(out["out"][0, klen:]) and not np.any(out["scales"][0, klen:])


def test_index_scores_fp8(device, recipes):
    """ix_slot_q8, ds_ix_gather, the FP8 batch_gemm and ds_ix_slot_acc8 against f64: per key, the sum over heads of
    w relu(q . k) / 64, with FP4-valued queries and keys stored as FP8 with power-of-two scales."""
    from resin_models_gaudi.glm5_3_flash.layers import GEMM_NT
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    rng = np.random.default_rng(13)
    PR, NB, NH, T, C, S = 256, 8, 32, 2, 2048, 2
    nkeys = C * S
    keys = fp4_ref(rng.standard_normal((nkeys, 128)), 32, False)
    kb, ks = fp8_rows(keys)
    arena = np.zeros((NB, PR, 512), np.uint8)
    flat = arena.reshape(-1, 512)
    pages = np.zeros((S, C // 128), np.int32)
    for p in range(nkeys // 128):
        r0 = 40 * p + 3
        pages.reshape(-1)[p] = r0
        flat[r0 : r0 + 32] = kb[128 * p : 128 * (p + 1)].reshape(32, 512)
        flat[r0 + 32, :512] = ks[128 * p : 128 * (p + 1)].view(np.uint8)
    q = fp4_ref(rng.standard_normal((NH * T, 128)), 32, False)
    qb, qs = fp8_rows(q)
    w = rng.standard_normal((T, NH)).astype(np.float32)
    # slot s: column s against keys C s .. C s + C - 1, all visible
    slots = np.array([[0, NH * s, 10**6, C * s, C * S, s, 1, 0] for s in range(S)], np.int32)
    g = Graph(device, "t_ds41_ix_score8")
    tab = g.input("slots", (8, S), I32)
    q8, qsc = g.tensor((128, NH, S), F8), g.tensor((NH, S), F32)
    g.node("rg_ix_slot_q8_f32", [g.input("q", (128, NH * T), F8), g.input("qs", (NH * T,), F32), tab], [q8, qsc])
    k8w, k8, ksc = g.tensor((256, C // 2, S), F8), g.tensor((128, C, S), F8), g.tensor((C, S), F32)
    g.node("rg_ds_ix_gather_f32", [g.input("arena", (512, PR, NB), U8), g.input("pages", (C // 128, S), I32), tab,
                                   g.input("dep", (64,), I32)], [k8w, ksc])  # fmt: skip
    g.node("reshape", [k8w], [k8])
    sc = g.tensor((C, NH, S), F32)
    g.node("batch_gemm", [q8, k8], [sc], GEMM_NT)
    g.node("rg_ds_ix_slot_acc8_f32", [sc, g.input("w", (NH, T), F32), tab, g.input("score", (C * S, T), F32),
                                      qsc, ksc], [g.input("sink", (64, 1), I32)], struct.pack("f", 1.0 / 64))  # fmt: skip
    out = run_graph(device, recipes, g, {
        "slots": slots, "q": qb, "qs": qs, "arena": arena, "pages": pages, "dep": np.zeros(64, np.int32),
        "w": w, "score": np.zeros((T, C * S), np.float32), "sink": np.zeros((1, 64), np.int32)}, ["score"])  # fmt: skip
    for s in range(S):
        kk = keys[C * s : C * (s + 1)]
        ref = (np.maximum(q[NH * s : NH * (s + 1)] @ kk.T, 0) * w[s][:, None]).sum(0) / 64
        np.testing.assert_allclose(out["score"][s, C * s : C * (s + 1)], ref, rtol=1e-5, atol=1e-5)


def test_index_queries_fp8(device, recipes):
    """ix_q8: each 128-dim column as natural-order FP8 with a power-of-two scale; FP4-valued queries exactly."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32

    rng = np.random.default_rng(15)
    W = 70
    x = fp4_ref(rng.standard_normal((W, 128)) * np.exp2(rng.integers(-8, 8, (W, 1))), 32, False)
    g = Graph(device, "t_ds41_ix_q8")
    g.node("rg_ix_q8_f32", [g.input("x", (128, W), F32)], [g.input("q", (128, W), F8), g.input("s", (1, W), F32)])
    out = run_graph(device, recipes, g, {"x": x.astype(np.float32), "q": np.zeros((W, 128), np.uint8),
                                          "s": np.zeros((W, 1), np.float32)}, ["q", "s"])  # fmt: skip
    np.testing.assert_array_equal(fp8_value(out["q"]) * out["s"], x)
    assert all(out["s"][c, 0] == pow2_scale(float(np.abs(x[c]).max())) for c in range(W))


def test_stream_mean_and_argmax(device, recipes):
    """ds_smean: the 4 streams' mean per column; ds_argmax: each column's first largest row below V."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(16)
    H, T, V, Vp = 320, 5, 1000, 1024
    X = rng.standard_normal((T, 4 * H)).astype(np.float32)
    x = rng.standard_normal((T, Vp)).astype(np.float32)
    x[1, 700] = x[1, 300] = 50.0  # a tie: the lower index
    x[2, V:] = 99.0  # past V: ignored
    g = Graph(device, "t_ds41_smean_argmax")
    g.node("rg_ds_smean_f32", [g.input("X", (4 * H, T), F32)], [g.input("y", (H, T), F32)])
    g.node("rg_ds_argmax_f32", [g.input("x", (Vp, T), F32)], [g.input("out", (64, T), I32), g.input("idx", (T,), I32)],
           struct.pack("i", V))  # fmt: skip
    out = run_graph(device, recipes, g, {"X": X, "y": np.zeros((T, H), np.float32), "x": x,
                                          "out": np.zeros((T, 64), np.int32), "idx": np.zeros(T, np.int32)},
                    ["y", "out", "idx"])  # fmt: skip
    np.testing.assert_allclose(out["y"], X.reshape(T, 4, H).mean(1), rtol=1e-6, atol=1e-6)
    want = np.argmax(x[:, :V], 1)
    assert want[1] == 300
    np.testing.assert_array_equal(out["idx"], want)
    np.testing.assert_array_equal(out["out"], np.repeat(want[:, None], 64, 1))


def test_markov_bias_add(device, recipes):
    """ds_mk_add: row b's column b block + i of the ranks' all-gathered logits, in vocabulary order, plus the bias;
    the last vector of a rank's slice does not spill into the next rank's entries."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    rng = np.random.default_rng(17)
    Vr, R, T, B, K, i = 100, 3, 8, 2, 4, 2
    L = rng.standard_normal((R, T, Vr)).astype(np.float32)
    bias = rng.standard_normal((B, R * Vr)).astype(np.float32)
    g = Graph(device, "t_ds41_mk_add")
    g.node("rg_ds_mk_add_f32", [g.input("L", (Vr, T, R), F32), g.input("bias", (R * Vr, B), F32)],
           [g.input("z", (R * Vr, B), F32)], struct.pack("ii", i, K))  # fmt: skip
    out = run_graph(device, recipes, g, {"L": L, "bias": bias, "z": np.zeros((B, R * Vr), np.float32)}, ["z"])
    want = np.stack([L[:, b * K + i, :].reshape(-1) for b in range(B)]) + bias
    np.testing.assert_array_equal(out["z"], want)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit patterns, rounding to nearest even"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_value(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def test_prefill_index_scores_from_bf16_products(device, recipes):
    """ds_ix_qacc: each prefill slot's scores from bf16 head products, sum_h w * scale * sq * relu(S) times the key's
    scale for visible keys (ratio-aware), -3e38 for the rest; entries outside the slots stay as they were."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    rng = np.random.default_rng(18)
    NH, C, Q, S, T, K, scale = 32, 2048, 16, 2, 32, 4096, 1.0 / 64
    slots = np.array(
        [[0, 0, 1000, 0, 1010, 0, 1, 0], [0, NH * Q, 3000, 2048, 1500 + 2048, Q, 2, 0]],
        np.int32,
    )
    Sb = bf16_bits(rng.standard_normal((S, NH * Q, C)) * 3)
    w = rng.standard_normal((T, NH)).astype(np.float32)
    sq = rng.uniform(0.5, 2, (S, NH * Q)).astype(np.float32)
    sk = rng.uniform(0.5, 2, (S, C)).astype(np.float32)
    score = rng.standard_normal((T, K)).astype(np.float32)
    g = Graph(device, "t_ds41_ix_qacc")
    g.node("rg_ds_ix_qacc_f32",
           [g.input("S", (C, NH * Q, S), BF16), g.input("w", (NH, T), F32), g.input("slots", (8, S), I32),
            g.input("score", (K, T), F32), g.input("sq", (NH * Q, S), F32), g.input("sk", (C, S), F32)],
           [g.input("sink", (64, 1), I32)], struct.pack("f", scale))  # fmt: skip
    out = run_graph(device, recipes, g, {"S": Sb, "w": w, "slots": slots, "score": score, "sq": sq, "sk": sk,
                                          "sink": np.zeros((1, 64), np.int32)}, ["score"])["score"]  # fmt: skip
    want = score.astype(np.float64).copy()
    mag = np.zeros_like(want)  # the terms' magnitudes summed: f32 sums agree to a few ulps of it
    Sv = bf16_value(Sb).astype(np.float64)
    for s in range(S):
        _, _, qpos0, key0, klen, tok0, ratio, _ = slots[s]
        for t in range(Q):
            col, lim = tok0 + t, min((qpos0 + t + 1) // ratio, klen)
            wt = w[col].astype(np.float64) * scale * sq[s, NH * t : NH * (t + 1)]
            terms = np.maximum(Sv[s, NH * t : NH * (t + 1)], 0) * wt[:, None]
            keys = key0 + np.arange(C)
            want[col, key0 : key0 + C] = np.where(keys < lim, terms.sum(0) * sk[s], np.float32(-3.0e38))
            mag[col, key0 : key0 + C] = np.abs(terms).sum(0) * sk[s]
    assert np.all(np.abs(out - want) <= 1e-6 * mag)
    assert np.sum(out == np.float32(-3.0e38)) == np.sum(want == np.float32(-3.0e38)) > 0


def ds41_dims(ranks: int = 1):
    from resin_models_gaudi.deepseek_v41 import layers as L

    return L.Dims(
        hidden=5120, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=288, experts=24, top=6,
        vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0, route_scale=1.5, ranks=ranks,
    )  # fmt: skip


@pytest.mark.parametrize("cand", ["", "source", "user"])
def test_prefilter_selection_lists_the_plain_entries(device, recipes, cand):
    """index_select_graph with the block prefilter (the entries of each column's 576 blocks of 8 with the largest
    maxima, packed by ix_bgather) lists the same entries as the plain selection, with ties."""
    from resin_models_gaudi.deepseek_v41 import layers as L

    rng = np.random.default_rng(19)
    T, K, PF, top = 48, 32768, 576, 2048
    KB = -(-K // 8 // 4096) * 4096
    score = (rng.standard_normal((T, K)) ** 3).astype(np.float32)
    score[:, ::5] = np.round(score[:, ::5], 1)  # ties
    vis = (12000 + 400 * np.arange(T)).astype(np.int32)
    nb = -(-vis // 8)
    base = {"ix_score": score, "ix_vis": vis, "ix_visb": np.minimum(-(-nb // 64) * 64, K // 8).astype(np.int32),
            "ix_visc": np.full(T, 8 * PF, np.int32)}  # fmt: skip
    d = ds41_dims()
    if cand == "user":  # the candidate blocks of a source over other scores
        src = L.index_select_graph(device, d, T, K, "source", top, cand_blocks=KB)
        other = (rng.standard_normal((T, K)) ** 3).astype(np.float32)
        run_src = Run(device, recipes, src)
        r = run_src(
            {
                k: v
                for k, v in {**base, "ix_score": other}.items()
                if k in run_src.recipe.persistent
            },
            ["cand_bmax", "cand_thr", "ix_list", "ix_lslots"],
        )
        src.destroy()
        base.update(cand_bmax=r["cand_bmax"], cand_thr=r["cand_thr"])
    outs = []
    for pf in (0, PF):
        g = L.index_select_graph(
            device, d, T, K, cand, top, cand_blocks=KB, prefilter=pf
        )
        run = Run(device, recipes, g)
        want = [n for n in ("ix_list", "ix_lslots", "cand_bmax", "cand_thr", "ix_thr") if n in run.recipe.persistent
                and n not in base]  # fmt: skip
        outs.append(
            run({k: v for k, v in base.items() if k in run.recipe.persistent}, want)
        )
        g.destroy()
    plain, pre = outs
    np.testing.assert_array_equal(plain["ix_lslots"][:, 4], pre["ix_lslots"][:, 4])
    for t in range(T):
        n = plain["ix_lslots"][t, 4]
        assert n == 512
        assert set(plain["ix_list"][t, :n].tolist()) == set(
            pre["ix_list"][t, :n].tolist()
        ), t
    if cand == "source":
        np.testing.assert_array_equal(plain["cand_bmax"], pre["cand_bmax"])
        np.testing.assert_array_equal(plain["cand_thr"], pre["cand_thr"])


def mkey_np(x: np.ndarray) -> np.ndarray:
    """the order-preserving u32 key of an f32 score (ix_thr's mkey)"""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return np.where(x < 0, ~b, b | np.uint32(0x80000000)).astype(np.uint32)


def test_list_tie_cut(device, recipes):
    """ix_list records in slot lane 7 the position of the last tie it lists: the listed entries are those above the
    threshold and the ties at or before the cut (0x7FFFFFFF: every tie, -1: none)."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(20)
    T, K, top = 48, 4096, 512
    score = rng.standard_normal((T, K)).astype(np.float32)
    score[0, :400] = 10 + rng.random(400)  # 400 above, 300 tied: 112 ties listed
    score[0, 1000:1300] = 5.0
    score[0, 1300:] = -5.0 - rng.random(K - 1300)
    score[1, :600] = 7.0  # 600 tied, nothing above: the 512 lowest positions
    vis = np.full(T, K, np.int32)
    vis[2] = 300  # sees fewer than 512: everything listed
    g = Graph(device, "t_ds41_tiecut")
    sc, V = g.input("score", (K, T), F32), g.input("vis", (T,), I32)
    thr = g.input("thr", (64, T), I32)
    g.node("rg_ix_thr_f32", [sc, V], [thr], struct.pack("i", top))
    cnt, gsum = g.tensor((K // 64, T), I32), g.tensor((K // 4096, T), I32)
    g.node("rg_ix_bcnt_f32", [sc, V, thr], [cnt, gsum])
    g.node("rg_ix_list_f32", [sc, V, thr, cnt, gsum], [g.input("list", (top, T), I32), g.input("lslots", (8, T), I32)],
           struct.pack("iii", max(1, 48 // T), top, 8))  # fmt: skip
    out = run_graph(device, recipes, g, {"score": score, "vis": vis, "thr": np.zeros((T, 64), np.int32),
                                          "list": np.zeros((T, top), np.int32), "lslots": np.zeros((T, 8), np.int32)},
                    ["thr", "list", "lslots"])  # fmt: skip
    for t in range(T):
        n, cut, th = (
            out["lslots"][t, 4],
            out["lslots"][t, 7],
            np.uint32(out["thr"][t, 0]),
        )
        k = mkey_np(score[t, : vis[t]])
        pos = np.arange(vis[t])
        kept = (k > th) | ((k == th) & (pos <= (cut if cut >= 0 else -1)))
        assert set(out["list"][t, :n].tolist()) == set(np.flatnonzero(kept).tolist()), t
    assert out["lslots"][0, 7] == 1000 + 111 and out["lslots"][1, 7] == 511
    assert out["lslots"][2, 7] == 0x7FFFFFFF


def test_dense_softmax_keeps_the_listed_entries(device, recipes):
    """ds_slot_pk over every visible compressed entry of a query block keeps exactly each column's listed entries
    (threshold and tie cut from ix_thr / ix_list): P, max and sum as a softmax over those entries."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    rng = np.random.default_rng(21)
    C, Q, hpg, S, top, scale = 2048, 16, 8, 2, 512, 0.044
    M, T = hpg * Q, Q * S
    slots = np.array(
        [[0, 0, 900, 0, (900 + Q) // 2, 0, 2, 0], [0, M, 1800, 0, 1800 + Q, Q, 1, 0]],
        np.int32,
    )
    vis = np.array([min((slots[s, 2] + t + 1) // slots[s, 6], slots[s, 4]) for s in range(S) for t in range(Q)],
                   np.int32)  # fmt: skip
    score = np.round(rng.standard_normal((T, C)), 1).astype(np.float32)  # many ties
    sc = (rng.standard_normal((S, M, C)) * 4).astype(np.float32)
    g = Graph(device, "t_ds41_slot_pk")
    sv, V = g.input("score", (C, T), F32), g.input("vis", (T,), I32)
    thr = g.input("thr", (64, T), I32)
    g.node("rg_ix_thr_f32", [sv, V], [thr], struct.pack("i", top))
    cnt, gsum = g.tensor((C // 64, T), I32), g.tensor((C // 4096 + 1, T), I32)
    g.node("rg_ix_bcnt_f32", [sv, V, thr], [cnt, gsum])
    lsl = g.input("lslots", (8, T), I32)
    g.node("rg_ix_list_f32", [sv, V, thr, cnt, gsum], [g.input("list", (top, T), I32), lsl],
           struct.pack("iii", max(1, 48 // T), top, hpg))  # fmt: skip
    g.node("rg_ds_slot_pk_f32", [g.input("sc", (C, M, S), F32), g.input("slots", (8, S), I32), sv, thr, lsl],
           [g.input("P", (C, M, S), BF16), g.input("mx", (M, S), F32), g.input("l", (M, S), F32)],
           struct.pack("fi", scale, hpg))  # fmt: skip
    out = run_graph(device, recipes, g, {
        "score": score, "vis": vis, "thr": np.zeros((T, 64), np.int32), "lslots": np.zeros((T, 8), np.int32),
        "list": np.zeros((T, top), np.int32), "sc": sc, "slots": slots, "P": np.zeros((S, M, C), np.uint16),
        "mx": np.zeros((S, M), np.float32), "l": np.zeros((S, M), np.float32)}, ["list", "lslots", "P", "mx", "l"])  # fmt: skip
    P = bf16_value(out["P"])
    for s in range(S):
        for m in range(M):
            col = slots[s, 5] + m // hpg
            listed = np.zeros(C, bool)
            listed[out["list"][col, : out["lslots"][col, 4]]] = True
            assert listed.sum() == min(top, vis[col])
            z = sc[s, m].astype(np.float64)
            mx = z[listed].max() * scale
            e = np.where(listed, np.exp(z * scale - mx), 0.0)
            assert not np.any(P[s, m][~listed]), (s, m)
            np.testing.assert_allclose(out["mx"][s, m], mx, rtol=1e-6)
            np.testing.assert_allclose(out["l"][s, m], e.sum(), rtol=1e-4)
            np.testing.assert_allclose(P[s, m], e, rtol=1e-2, atol=1e-6)


def test_direct_slot_merge_equals_the_scan(device, recipes):
    """ds_slot_mld (column c's slot is c / M) writes what ds_slot_ml writes when slot s covers columns M s .. M s +
    M - 1, a slot of group -1 included."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(22)
    D, M, S = 512, 8, 40
    W = M * S
    slots = np.zeros((S, 8), np.int32)
    slots[:, 1] = M * np.arange(S)
    slots[7, 0] = -1
    inputs = {
        "O": rng.standard_normal((S, M, D)).astype(np.float32),
        "mx": rng.standard_normal((S, 64)).astype(np.float32),
        "l": rng.uniform(0.5, 4, (S, 64)).astype(np.float32),
        "slots": slots,
    }
    outs = []
    for guid in ("rg_ds_slot_mld_f32", "rg_ds_slot_ml_f32"):
        g = Graph(device, f"t_{guid}_direct")
        g.node(
            guid,
            [g.input("O", (D, M, S), F32), g.input("mx", (64, S), F32), g.input("l", (64, S), F32),
             g.input("slots", (8, S), I32)],
            [g.input("out", (D, W, 1), F32), g.input("m", (W, 1), F32), g.input("lc", (W, 1), F32)],
        )  # fmt: skip
        outs.append(Run(device, recipes, g)(inputs, ["out", "m", "lc"]))
        g.destroy()
    for k in ("out", "m", "lc"):
        np.testing.assert_array_equal(outs[0][k], outs[1][k])
