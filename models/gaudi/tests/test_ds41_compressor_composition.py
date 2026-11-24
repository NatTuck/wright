"""Compressor store composition oracle (DeepSeek-V4.1-Flash, one rank, ratio 2 KV owner).

owner_store_graph: residual + collapse8 + _compress_store (cmp_wkv/cmp_wgate f32 GEMMs -> ds_cpool
softmax pooling -> cmp_norm -> ix_wk bf16 GEMM + ix_knorm RoPE -> fq4 FP4 -> mla_store_i8 FP8). Reads the
stored latents and index keys back from the arena and compares to an independent numpy compressor.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


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


def _report(name, got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    print(f"  {name:12s} max_abs={np.abs(got - want).max():.4e} norm_rel={rel:.4e}")
    return rel


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def _e2m1(x, s):
    a = np.abs(x)
    q = np.full_like(a, 6.0)
    q = np.where(a <= 5.0 * s, 4.0, q)
    q = np.where(a < 3.5 * s, 3.0, q)
    q = np.where(a <= 2.5 * s, 2.0, q)
    q = np.where(a < 1.75 * s, 1.5, q)
    q = np.where(a <= 1.25 * s, 1.0, q)
    q = np.where(a < 0.75 * s, 0.5, q)
    q = np.where(a < 0.25 * s, 0.0, q)
    return np.where(x < 0, -q, q) * s


def _e4m3(x):
    from resin_models_gaudi.loading.convert import bf16_rne  # noqa: F401
    x = np.asarray(x, np.float64)
    bits = x.astype(np.float32).view(np.int32).astype(np.int64)
    e = np.maximum(((bits >> 23) & 255) - 127, -6)
    quantum = np.exp2((e - 3).astype(np.float64))
    q = np.round(x / quantum)  # ties to even
    return np.minimum(q * quantum, 448.0)


def _fq4(x, mode):
    """x (..., D) last axis grouped; mode 0 G=32 pow2 scale, mode 1 G=16 e4m3 scale."""
    G = 32 if mode == 0 else 16
    shp = x.shape
    xg = x.reshape(*shp[:-1], shp[-1] // G, G)
    a = np.abs(xg).max(axis=-1, keepdims=True)
    if mode == 0:
        m = np.maximum(a, 6.0 * 1.1754944e-38)
        r = m / 6.0
        s = np.exp2(np.ceil(np.log2(np.maximum(r, 1e-30))))
        s = np.where(m > s * 6.0, s * 2.0, s)
        s = np.where(m <= s * 3.0, s * 0.5, s)
    else:
        m = np.maximum(a, 6.0 * 0.001953125)
        s = _e4m3(m / 6.0)
    return _e2m1(xg, s).reshape(shp)


def _rope_swap(v, cs):
    """v (..., D), cs (128,) for one position: adjacent-pair RoPE on last 64 (cleared formula)."""
    v = v.copy().astype(np.float64)
    D = v.shape[-1]
    co, si = cs[:64].astype(np.float64), cs[64:].astype(np.float64)
    r = v[..., D - 64 :]
    swap = r.reshape(*r.shape[:-1], 32, 2)[..., ::-1].reshape(*r.shape)
    v[..., D - 64 :] = r * co + swap * si
    return v


def _cs_row(pos, freqs):
    ang = pos * freqs
    co = np.cos(ang.astype(np.float64)).astype(np.float32).repeat(2)
    si = np.sin(ang.astype(np.float64)).astype(np.float32).repeat(2)
    sign = np.where(np.arange(64) % 2 == 1, 1.0, -1.0).astype(np.float32)
    return np.concatenate([co, si * sign])


def test_compressor_store_matches_numpy(device, recipes):
    from test_ds41_kernels import Run, fp8_byte, fp8_value, pow2_scale

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.loading.convert import bf16_rne

    rng = np.random.default_rng(777)
    H, D, T, ratio, G, eps = 5120, 512, 4, 2, 2, 1e-20
    ID = 128
    freqs = (1.0 / (40000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64.0))).astype(np.float32)
    X0 = (rng.standard_normal((T, H)) * 0.5).astype(np.float32)
    X = np.zeros((T, 4 * H), np.float32)
    X[:, :H] = X0
    c = np.zeros((24, T), np.float32)
    c[0] = 1.0
    c[8 + np.arange(4) * 5] = 1.0
    attn_norm = (1.0 + 0.02 * rng.standard_normal(H)).astype(np.float32)
    cmp_wkv = (rng.standard_normal((D, H)) * 0.02).astype(np.float32)
    cmp_wgate = (rng.standard_normal((D, H)) * 0.02).astype(np.float32)
    cmp_norm = (1.0 + 0.02 * rng.standard_normal(D)).astype(np.float32)
    ix_wk = bf16_rne((rng.standard_normal((ID, D)) * 0.02).astype(np.float32))
    ix_wk_f32 = (ix_wk.astype(np.uint32) << 16).view(np.float32)
    ix_knorm = (1.0 + 0.02 * rng.standard_normal(ID)).astype(np.float32)

    # arena tables (fp4): latent page base lat0=(g//128)*97, entry e=g%128; index at lat0+80
    PR = 1024
    cmp_store = np.full((8, G), -1, np.int32)
    cmp_store[0] = (np.arange(G) // 128) * 97  # latent page base row
    cmp_store[1] = np.arange(G) % 128          # entry within page
    cmp_groups = np.full((4, G), -1, np.int32)
    cmp_groups[0] = np.arange(G) * ratio  # ca
    cmp_groups[1] = np.arange(G) * ratio + 1  # cb
    cs_g = np.stack([_cs_row(g * ratio, freqs) for g in range(G)], axis=1)  # (128,G)

    d = L.Dims(hidden=H, heads=8, head_dim=D, q_lora=1280, o_lora=1024, groups=1, expert=576,
               experts=64, top=6, vocab=16160, eps=eps, hc_iters=20, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    g = L.owner_store_graph(device, d, T, 1, ratio=ratio, groups=G, state_rows=1)
    arena = np.zeros((1, PR, 512), np.uint8)
    dev = {
        "X": X, "p": np.zeros((1, T, H), np.float32), "c": c,
        "X_out": np.zeros((T, 4 * H), np.float32), "attn_norm": attn_norm,
        "cmp_wkv": cmp_wkv, "cmp_wgate": cmp_wgate, "cmp_norm": cmp_norm,
        "ix_wk": ix_wk, "ix_knorm": ix_knorm, "cs_g": np.ascontiguousarray(cs_g.T),
        "cmp_store": np.ascontiguousarray(cmp_store.T), "cmp_groups": np.ascontiguousarray(cmp_groups.T),
        "cmp_state": np.zeros((1, 2 * D), np.float32), "cmp_rows": np.arange(T, dtype=np.int32),
        "cmp_sink": np.zeros((1, 64), np.int32),
        "cmp_stored": np.zeros((1, 64), np.int32),
        "rope_s1": np.zeros(1, np.float32), "rope_b1": np.ones(1, np.float32),
        "rope_lp2": np.zeros(G, np.float32),
        "arena": arena, "one": np.ones(1, np.float32), "dep": np.zeros(64, np.float32),
    }
    dev = {k: v for k, v in dev.items() if k in g.tensors}
    got = Run(device, recipes, g)(dev, ["arena"])
    g.destroy()
    ar = got["arena"].reshape(PR, 512)  # (rows, bytes)

    # numpy compressor
    cv = X0.astype(np.float64)
    r = 1.0 / np.sqrt((cv * cv).mean(axis=1, keepdims=True) + eps)
    h_norm = cv * r * attn_norm
    kv = h_norm @ cmp_wkv.T  # (T,D)
    score = h_norm @ cmp_wgate.T
    lat = np.zeros((G, D), np.float64)
    for gg in range(G):
        a, b = gg * ratio, gg * ratio + 1
        m = np.maximum(score[a], score[b])
        ea, eb = np.exp(score[a] - m), np.exp(score[b] - m)
        lat[gg] = (kv[a] * ea + kv[b] * eb) / (ea + eb)
    latn = np.zeros((G, D), np.float64)
    for gg in range(G):
        rr = 1.0 / np.sqrt((lat[gg] ** 2).mean() + eps)
        latn[gg] = lat[gg] * rr * cmp_norm
    print("\n=== compressor store (fp4 code+scale) ===")
    from test_ds41_fp4_kv import _decode_latent, _decode_index
    # stored latents: fp4 codes + e4m3 scales; value = fp4_ref(rope(latn), mode 1)
    for gg in range(G):
        want = _fq4(_rope_swap(latn[gg], cs_g[:, gg]), 1)
        lat0 = int(cmp_store[0, gg]); e = int(cmp_store[1, gg])
        gotlat = _decode_latent(ar, lat0, e)
        rl = _report(f"latent{gg}", gotlat, want)
        assert rl < 2e-3, f"latent{gg} diverges {rl:.3e}"
    # stored index keys: fp4 codes + e8m0 scales; value = fp4_ref(ds_rope(...), mode 0)
    for gg in range(G):
        from test_ds41_kernels import bf16_round
        k = (bf16_round(latn[gg]).astype(np.float32) @ ix_wk_f32.T)  # bf16 latn x bf16 ix_wk -> f32
        k = k.astype(np.float32)
        rr = 1.0 / np.sqrt((k.astype(np.float64) ** 2).mean() + eps)
        kn = k * rr * ix_knorm
        want = _fq4(_rope_swap(kn, cs_g[:, gg]), 0)
        lat0 = int(cmp_store[0, gg]); e = int(cmp_store[1, gg])
        ix0 = lat0 + 80
        gotix = _decode_index(ar, lat0, e)
        ri = _report(f"index{gg}", gotix, want)
        assert ri < 2e-3, f"index{gg} diverges {ri:.3e}"