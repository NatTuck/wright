"""DeepSeek-V4.1 fp4 code+scale KV store: the value-exact gate (one card, numpy oracle).

The checkpoint (NVIDIA fp4_act_quant inplace) stores the bf16 round-tripped value
``bf16(f32(e2m1_code) * f32(scale))``.  e2m1 (1 mantissa bit) x e4m3 scale (3 bits) needs <= 5
significant bits, so it fits bf16 (7 stored) exactly: storing code+scale and reproducing that product
in the gather is bit-for-bit the reference.  The current FP8 store re-quantizes the fp4-round-tripped
value to fp8 e4m3 (3 mantissa bits) with a power-of-two token scale -- a SECOND rounding -- so it is
NOT value-exact.  This gate must FAIL on the fp8 store (RED) and PASS on the fp4 store (GREEN).

The oracle is the independent numpy ``fp4_ref`` (the checkpoint's fp4_quant_kernel), not the code
under test.  Per-element, no accumulation: a bit-exact comparison is valid here.
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

E2M1_GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
E2M1_SIGNED = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float64)


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


def fp4_ref(x: np.ndarray, group: int, e4m3_scale: bool) -> np.ndarray:
    """The checkpoint's fp4_quant_kernel, inplace: per-group scale, nearest E2M1, ties to the even code.
    Returns the f32 round-tripped value (== the bf16 value, exact)."""
    xg = x.astype(np.float64).reshape(x.shape[0], -1, group)
    a = np.abs(xg).max(-1, keepdims=True)
    if e4m3_scale:
        v = np.maximum(a, 6 * 2.0**-9) / 6
        e = np.maximum(np.floor(np.log2(v)), -6)
        s = np.minimum(np.round(v / 2 ** (e - 3)) * 2 ** (e - 3), 448)
    else:
        s = 2 ** np.ceil(np.log2(np.maximum(a, 6 * 2.0**-126) / 6))
    r = np.clip(xg / s, -6, 6)
    d = np.abs(np.abs(r)[..., None] - E2M1_GRID)
    best = d.min(-1, keepdims=True)
    tie = (d == best).sum(-1) > 1
    code = np.argmin(d, -1)
    code = np.where(tie, np.where(code % 2 == 0, code, code + 1), code)
    return (np.sign(r) * E2M1_GRID[code] * s).reshape(x.shape)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


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


def _latent_inputs(rng, T):
    """Latents (T, 512) with a spread of group magnitudes (so e4m3 scales vary), plus ties."""
    x = (
        rng.standard_normal((T, 512))
        * np.exp2(rng.integers(-6, 6, (T, 32)).repeat(16, 1))
    ).astype(np.float32)
    x[0, :16] = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2, np.float32)
    return x


def _fp8_roundtrip(device, recipes, x):
    """The CURRENT production compress-latent store+gather: fq4(mode 1) -> mla_store_i8 (FP8 + pow2
    token scale) -> ds_gather_list (FP8 -> bf16).  Returns the gathered bf16 bits (T, C, 512)."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32, U8

    T = x.shape[0]
    PR, NB, B = 256, 8, 200  # latent page base row
    g = Graph(device, "t_fp4kv_red_fp8")
    arena = g.input("arena", (512, PR, NB), U8)
    sink = g.input("sink", (64, 1), I32)
    lq = g.tensor((512, T), F32)
    g.node("rg_ds_fq4_f32", [g.input("x", (512, T), F32)], [lq], struct.pack("i", 1))
    # cmp_store lanes: [lat row, rope(-1), index(-1), scale row, scale entry, ...]
    g.node(
        "rg_mla_store_i8_f32",
        [arena, g.input("rows", (8, T), I32), lq, g.input("ix_none", (128, T), F32)],
        [sink],
    )
    g.node(
        "rg_ds_gather_list_f32",
        [arena, g.input("list", (T, T), I32), g.input("blocks", (1, T), I32),
         g.input("loff", (64,), I32), g.input("slots", (8, T), I32), sink],
        [g.input("out", (512, T, T), F32)],
    )
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = B + np.arange(T)  # latent row
    rows[:, 3] = B + 128  # scale row
    rows[:, 4] = np.arange(T)  # scale entry
    lst = np.tile(np.arange(T, dtype=np.int32), (T, 1))  # column t gathers entries 0..T-1
    blocks = np.full((T, 1), B, np.int32)
    slots = np.zeros((T, 8), np.int32)
    slots[:, 4] = T
    out = run_graph(device, recipes, g, {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32),
        "x": np.ascontiguousarray(x), "rows": rows, "ix_none": np.zeros((T, 128), np.float32),
        "list": lst, "blocks": blocks, "loff": np.zeros(64, np.int32), "slots": slots,
        "out": np.zeros((T, T, 512), np.float32)}, ["out"])["out"]
    return bf16_bits(out)  # (T, C=T, 512) bf16 bits of the dequanted latent


def test_fp8_store_is_not_value_exact(device, recipes):
    """RED control: the current FP8 store adds a second rounding, so the gathered latent is NOT
    bit-for-bit the checkpoint's bf16 round-tripped value.  This is the defect the fp4 store removes."""
    rng = np.random.default_rng(101)
    T = 6
    x = _latent_inputs(rng, T)
    want = bf16_bits(fp4_ref(x, 16, True))  # (T, 512) reference bf16 bits
    got = _fp8_roundtrip(device, recipes, x)  # (T, C, 512)
    # column t gathers entries 0..T-1 in order; entry k's latent is x[k]
    mism = 0
    for t in range(T):
        for k in range(T):
            mism += int(np.sum(got[t, k] != want[k]))
    assert mism > 0, "fp8 store unexpectedly value-exact (the RED control must fail to be exact)"
    print(f"  fp8 store: {mism} / {T*T*512} elements differ from the checkpoint bf16 value (RED)")


def test_fp4_kv_value_exact(device, recipes):
    """THE GATE: stored->dequanted latent == bf16(f32(code)*f32(scale)) bit-for-bit.  RED on the fp8
    store (second rounding); GREEN on the fp4 code+scale store.  Per-element, no accumulation."""
    rng = np.random.default_rng(101)
    T = 6
    x = _latent_inputs(rng, T)
    want = bf16_bits(fp4_ref(x, 16, True))  # (T, 512)
    got = _fp4_roundtrip(device, recipes, x)  # fp4 store+gather (GREEN); fp8 RED control above
    for t in range(T):
        for k in range(T):
            np.testing.assert_array_equal(
                got[t, k], want[k], err_msg=f"entry {k} col {t}: fp4 store not value-exact"
            )


def e4m3_value(b: np.ndarray) -> np.ndarray:
    """OCP E4M3 (bias 7) bytes -> values, magnitudes to 448 (0x7F/0xFF read as 0)."""
    b = b.astype(np.int64)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * np.exp2(e - 10.0), m * np.exp2(-9.0))
    v = np.where((b & 0x7F) == 0x7F, 0.0, v)
    return np.where(b & 0x80, -v, v)


def _store_f4(device, recipes, lat, ix):
    """Store lat (T,512) and ix (T,128) via ds_store_f4 into a fresh arena; return the arena flat rows."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32, U8

    T = lat.shape[0]
    PR, NB, B = 256, 8, 200
    g = Graph(device, "t_fp4kv_store")
    arena = g.input("arena", (512, PR, NB), U8)
    sink = g.input("sink", (64, 1), I32)
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = B  # lat0 (page first row)
    rows[:, 1] = np.arange(T)  # entry e
    g.node(
        "rg_ds_store_f4_f32",
        [arena, g.input("rows", (8, T), I32), g.input("lat", (512, T), F32),
         g.input("ix", (128, T), F32)],
        [sink],
    )
    out = run_graph(device, recipes, g, {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32),
        "rows": rows, "lat": np.ascontiguousarray(lat), "ix": np.ascontiguousarray(ix)}, ["arena"])["arena"]
    return out.reshape(NB * PR, 512), B  # PR=256: ROW(r) maps to flat row r


def _decode_latent(flat, B, e):
    codes = flat[B + (e >> 1), (e & 1) * 256:(e & 1) * 256 + 256]
    nib = np.stack([codes & 15, codes >> 4], -1).reshape(-1).astype(np.int64)
    sc = flat[B + 64 + (e >> 3), (e & 7) * 32:(e & 7) * 32 + 32]
    s = e4m3_value(sc).repeat(16)
    return E2M1_SIGNED[nib] * s


def _decode_index(flat, B, e):
    ix0 = B + 80
    codes = flat[ix0 + (e >> 3), (e & 7) * 64:(e & 7) * 64 + 64]
    nib = np.stack([codes & 15, codes >> 4], -1).reshape(-1).astype(np.int64)
    sc = flat[ix0 + 16, 4 * e:4 * e + 4]
    s = np.exp2(sc.astype(np.float64) - 127).repeat(32)
    return E2M1_SIGNED[nib] * s


def test_fp4_store_roundtrip(device, recipes):
    """ds_store_f4 codes+scales decode bit-for-bit to the checkpoint fp4 value (latent e4m3/16,
    index e8m0/32).  Verifies quantization, nibble packing and scale encoding independent of the gather."""
    rng = np.random.default_rng(202)
    T = 6
    lat = _latent_inputs(rng, T)
    ix = (rng.standard_normal((T, 128)) * np.exp2(rng.integers(-6, 6, (T, 4)).repeat(32, 1))).astype(np.float32)
    flat, B = _store_f4(device, recipes, lat, ix)
    want_lat = fp4_ref(lat, 16, True)
    want_ix = fp4_ref(ix, 32, False)
    for e in range(T):
        np.testing.assert_array_equal(_decode_latent(flat, B, e), want_lat[e], err_msg=f"latent entry {e}")
        np.testing.assert_array_equal(_decode_index(flat, B, e), want_ix[e], err_msg=f"index entry {e}")


def _fp4_roundtrip(device, recipes, x):
    """The fp4 production path: ds_store_f4 (codes+scales) -> ds_gather_list_f4 (fp4 -> bf16).
    Returns the gathered bf16 bits (T, C, 512)."""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8

    T = x.shape[0]
    C = 16
    PR, NB, B = 256, 8, 200
    g = Graph(device, "t_fp4kv_green")
    arena = g.input("arena", (512, PR, NB), U8)
    sink = g.input("sink", (64, 1), I32)
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = B
    rows[:, 1] = np.arange(T)
    g.node(
        "rg_ds_store_f4_f32",
        [arena, g.input("rows", (8, T), I32), g.input("lat", (512, T), F32),
         g.input("ix", (128, T), F32)],
        [sink],
    )
    codes = g.tensor((256, C, T), U8)
    scales = g.tensor((32, C, T), U8)
    g.node(
        "rg_ds_gather_list_f4_f32",
        [arena, g.input("list", (C, T), I32), g.input("blocks", (1, T), I32),
         g.input("loff", (64,), I32), g.input("slots", (8, T), I32), sink],
        [codes, scales],
    )
    codes2 = g.tensor((256, C * T), U8)
    scales2 = g.tensor((32, C * T), U8)
    out2 = g.tensor((512, C * T), BF16)
    g.node("reshape", [codes], [codes2])
    g.node("reshape", [scales], [scales2])
    g.node("rg_ds_ldec_f4_f32", [codes2, scales2], [out2])
    g.node("reshape", [out2], [g.input("out", (512, C, T), BF16)])
    lst = np.zeros((T, C), np.int32)
    lst[:, :T] = np.arange(T, dtype=np.int32)
    blocks = np.full((T, 1), B, np.int32)
    slots = np.zeros((T, 8), np.int32)
    slots[:, 4] = T
    out = run_graph(device, recipes, g, {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32),
        "rows": rows, "lat": np.ascontiguousarray(x), "ix": np.zeros((T, 128), np.float32),
        "list": lst, "blocks": blocks, "loff": np.zeros(64, np.int32), "slots": slots,
        "out": np.full((T, C, 512), 0x7FC0, np.uint16)}, ["out"])["out"]
    return out  # (T, C, 512) bf16 bits


def test_fp4_store_gather_value_exact(device, recipes):
    """GREEN GATE: fp4 store -> fp4 gather dequant == bf16(f32(code)*f32(scale)) bit-for-bit."""
    rng = np.random.default_rng(101)
    T = 6
    x = _latent_inputs(rng, T)
    want = bf16_bits(fp4_ref(x, 16, True))
    got = _fp4_roundtrip(device, recipes, x)
    for t in range(T):
        for k in range(T):
            np.testing.assert_array_equal(
                got[t, k], want[k], err_msg=f"entry {k} col {t}: fp4 store+gather not value-exact"
            )


def _index_gather_dequant(device, recipes, ix, T, C):
    """The production indexer fp4 path: ds_store_f4 (index codes+e8m0) -> ds_ix_gather_f4 (codes, scales,
    per-key sk=2^k, ktab=k, ids) -> mx4_decs (per-row k, fp4->fp8 NORMALIZED).  Returns (fp8 value (C,128),
    sk (C,)).  The effective key value is fp8_value(dec) * sk."""
    from test_ds41_kernels import fp8_value

    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F8, F32, I32, U8

    PR, NB, B = 256, 8, 200
    g = Graph(device, "t_fp4ix")
    arena = g.input("arena", (512, PR, NB), U8)
    sink = g.input("sink", (64, 1), I32)
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = B
    rows[:, 1] = np.arange(T)
    g.node("rg_ds_store_f4_f32",
           [arena, g.input("rows", (8, T), I32), g.input("lat", (512, T), F32), g.input("ix", (128, T), F32)],
           [sink])
    codes = g.tensor((64, C, 1), U8)
    scales = g.tensor((4, C, 1), U8)
    sk = g.input("sk", (C, 1), F32)
    ktab = g.tensor((C, 1), I32)
    ids = g.tensor((C, 1), I32)
    g.node("rg_ds_ix_gather_f4_f32",
           [arena, g.input("cmp_blocks", (1, 1), I32), g.input("slots", (8, 1), I32), sink,
            g.input("loff", (64,), I32)],
           [codes, scales, sk, ktab, ids], struct.pack("i", 80))
    codes2 = g.tensor((64, C), U8)
    scales2 = g.tensor((4, C), U8)
    ktab2 = g.tensor((C,), I32)
    ids2 = g.tensor((C,), I32)
    g.node("reshape", [codes], [codes2])
    g.node("reshape", [scales], [scales2])
    g.node("reshape", [ktab], [ktab2])
    g.node("reshape", [ids], [ids2])
    dec2 = g.tensor((128, C), F8)
    g.node("rg_mx4_decs_f32", [codes2, scales2, ids2, ktab2], [dec2], struct.pack("ii", 0, 0))
    dec = g.input("dec", (128, C, 1), F8)
    g.node("reshape", [dec2], [dec])
    slots = np.array([[0, 0, 0, 0, T, 0, 1, 0]], np.int32)
    out = run_graph(device, recipes, g, {
        "arena": np.zeros((NB, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32),
        "rows": rows, "lat": np.zeros((T, 512), np.float32), "ix": np.ascontiguousarray(ix),
        "cmp_blocks": np.full((1, 1), B, np.int32), "slots": slots,
        "loff": np.zeros(64, np.int32), "dec": np.zeros((1, C, 128), np.uint8),
        "sk": np.zeros((1, C), np.float32)}, ["dec", "sk"])
    return fp8_value(out["dec"][0]), out["sk"][0]  # (C, 128), (C,)


def test_fp4_index_store_gather(device, recipes):
    """Indexer fp4 path: ds_store_f4 (index codes+e8m0) -> ds_ix_gather_f4 (staging + per-key sk) ->
    mx4_decs (per-row k, fp4->fp8 normalized).  fp8_value(dec) * sk == the checkpoint fp4 index key value."""
    rng = np.random.default_rng(303)
    T = 6
    C = 32
    ix = (rng.standard_normal((T, 128)) * np.exp2(rng.integers(-4, 4, (T, 4)).repeat(32, 1))).astype(np.float32)
    decv, sk = _index_gather_dequant(device, recipes, ix, T, C)
    got = decv * sk[:, None]
    want = fp4_ref(ix, 32, False)
    for k in range(T):
        np.testing.assert_array_equal(got[k], want[k], err_msg=f"index key {k}")


def test_fp4_index_large_scale(device, recipes):
    """RED->GREEN: index keys with LARGE e8m0 group scales (2^8..2^12).  The fp4->fp8 dequant must
    normalize by a per-key tensor scale k (sk = 2^k) so the fp8 code stays < 240; the effective value
    fp8_value(dec) * sk must equal the checkpoint fp4 value exactly.  With k=0 / sk=1 the fp8 saturates
    (or wraps to poison) for large scales -> RED.  The small-scale test above cannot catch this."""
    rng = np.random.default_rng(404)
    T = 6
    C = 32
    ix = (rng.standard_normal((T, 128)) * np.exp2(rng.integers(8, 13, (T, 4)).repeat(32, 1))).astype(np.float32)
    decv, sk = _index_gather_dequant(device, recipes, ix, T, C)
    got = decv * sk[:, None]
    want = fp4_ref(ix, 32, False)
    mism = 0
    for k in range(T):
        mism += int(np.sum(got[k] != want[k]))
    print(f"  fp4 index large-scale: {mism}/{T*128} elems differ from the checkpoint fp4 value")
    assert mism == 0, f"large-scale index dequant not value-exact ({mism} elems): k=0 fp8 saturation"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))