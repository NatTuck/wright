"""Dense compressed-attention output composition oracle (DeepSeek-V4.1-Flash, one rank, prefill).

attention_out_graph(dense): gather_dense (FP8 compressed latents) -> batch_gemm scores -> ds_slot_pk
(index-masked softmax) -> batch_gemm -> ds_slot_ml -> ds_attn_comb_r (merge window part + sink + inverse
RoPE) -> wo_a/wo_b -> p_attn. Feeds known q_all, a zero window part, known FP8 latents and a
full-visibility index mask, and compares p_attn to an independent numpy compressed attention.
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


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _report(name, got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    print(f"  {name:10s} max_abs={np.abs(got - want).max():.4e} norm_rel={rel:.4e}")
    return rel


def _cs_row(pos, freqs, inverse=False):
    ang = pos * freqs
    co = np.cos(ang.astype(np.float64)).astype(np.float32).repeat(2)
    si = np.sin(ang.astype(np.float64)).astype(np.float32).repeat(2)
    sign = np.where(np.arange(64) % 2 == 1, 1.0, -1.0).astype(np.float32)
    return np.concatenate([co, si * (-sign if inverse else sign)])


def _rope_swap(v, cs):
    v = v.copy().astype(np.float64)
    D = v.shape[-1]
    co, si = cs[:64].astype(np.float64), cs[64:].astype(np.float64)
    r = v[..., D - 64 :]
    swap = r.reshape(*r.shape[:-1], 32, 2)[..., ::-1].reshape(*r.shape)
    v[..., D - 64 :] = r * co + swap * si
    return v


def _build_staged_dense(device, d, T, keys, nh, D):
    import struct

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.glm5_3_flash.layers import GEMM_NN, GEMM_NT
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    W = nh * T
    S = T // 128
    M = nh * 128
    lg = L.LayerGraph(device, f"t_dense_T{T}", d, T)
    snaps = {}

    def sn(name, ten):
        o = lg.input("s_" + name, ten.shape, ten.dtype)
        lg.node("memcpy", [ten], [o])
        snaps[name] = o
        return ten

    ks = L._gather_dense(lg, 1, keys)
    sn("ks", ks)
    table = lg.input("cslots", (8, S), I32)
    qs = lg.view(lg.input("q_all", (D, W), BF16), (D, M, S))
    sc = lg.tensor((keys, M, S), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    sn("sc", sc)
    P, mx, ls = lg.tensor((keys, M, S), BF16), lg.tensor((M, S), F32), lg.tensor((M, S), F32)
    lg.node("rg_ds_slot_pk_f32",
            [sc, table, lg.input("ix_score", (keys, T), F32), lg.input("ix_thr", (64, T), I32),
             lg.input("ix_lslots", (8, T), I32)],
            [P, mx, ls], struct.pack("fi", d.scale, nh))
    sn("P", P)
    sn("mx", mx)
    sn("ls", ls)
    pv = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    o2, m2, l2 = lg.tensor((D, W, 1), F32), lg.tensor((W, 1), F32), lg.tensor((W, 1), F32)
    lg.node("rg_ds_slot_ml_f32", [pv, mx, ls, table], [o2, m2, l2])
    sn("o2", o2)
    sn("m2", m2)
    sn("l2", l2)
    Wd = nh * T
    o = lg.tensor((D, Wd, 1), F32)
    lg.node("rg_ds_attn_comb_r_f32",
            [lg.input("o1", (D, Wd, 1), F32), lg.input("m1", (Wd, 1), F32), lg.input("l1", (Wd, 1), F32),
             o2, m2, l2, lg.input("attn_sink", (nh,), F32), lg.input("cs_inv", (128, T), F32)],
            [o], struct.pack("i", nh))
    sn("o", o)
    L._attention_out(lg, lg.view(o, (D, Wd)))
    return lg.g, snaps


def test_dense_compressed_attention(device, recipes):
    from test_ds41_kernels import Run, bf16_round, fp8_byte, fp8_value, pow2_scale

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.deepseek_v41.weights import mx8_rows
    from resin_models_gaudi.loading.convert import bf16_rne

    rng = np.random.default_rng(4242)
    H, D, nh, T, olora, groups = 5120, 512, 8, 128, 1024, 1
    keys, ratio, eps = 256, 1, 1e-20
    W = nh * T
    per = nh // groups * D
    scale = D**-0.5
    freqs = (1.0 / (10000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64.0))).astype(np.float32)
    PR = 1024

    # known compressed latents, stored as true fp4 codes + e4m3 group scales (ds_store_f4 layout):
    # page p (128 entries) at latent base row p*97 (latent 80 rows + index 17 rows = PAIR_ROWS).
    from test_ds41_fp4_kv import fp4_ref
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32, U8

    lat = (rng.standard_normal((keys, D)) * 0.5).astype(np.float32)
    latd = fp4_ref(lat, 16, True)  # exact checkpoint fp4 value (e2m1 * e4m3/16)
    sg = Graph(device, "t_dense_store")
    sarena = sg.input("arena", (512, PR, 1), U8)
    ssink = sg.input("sink", (64, 1), I32)
    srows = np.full((keys, 8), -1, np.int32)
    srows[:, 0] = (np.arange(keys) // 128) * 97
    srows[:, 1] = np.arange(keys) % 128
    sg.node("rg_ds_store_f4_f32",
            [sarena, sg.input("rows", (8, keys), I32), sg.input("lat", (D, keys), F32),
             sg.input("ix", (128, keys), F32)], [ssink])
    sdev = {"arena": np.zeros((1, PR, 512), np.uint8), "sink": np.zeros((1, 64), np.int32),
            "rows": srows, "lat": np.ascontiguousarray(lat), "ix": np.zeros((keys, 128), np.float32)}
    arena = Run(device, recipes, sg)(sdev, ["arena"])["arena"]
    sg.destroy()
    cmp_lat = np.array([[0], [97]], np.int32)  # (keys//128, S) fp4 latent page base rows
    cslots = np.array([[0, 0, 0, 0, keys, 0, ratio, 0]], np.int32)  # (8,S)

    q_all = bf16_rne((rng.standard_normal((W, D)) * 0.5).astype(np.float32))  # (W,D)
    qf = _bf16_to_f32(q_all).astype(np.float64)
    cs_inv = np.stack([_cs_row(t, freqs, inverse=True) for t in range(T)], axis=1)  # (128,T)
    sink = np.full(nh, -1e30, np.float32)

    def mxfp8(rows, cols):
        b = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
        b[(b & 0x7F) == 0x7F] = 0
        s = rng.integers(113, 115, (rows // 32, cols // 32), dtype=np.uint8)
        return b, s

    from resin_models_gaudi.loading.convert import _permute_k

    def untau(b):
        return _permute_k(b, 64, 4)

    woa_b, woa_s = mxfp8(olora, per)
    wob_b, wob_s = mxfp8(H, olora * groups)
    mwoa, mwob = mx8_rows(woa_b, woa_s), mx8_rows(wob_b, wob_s)
    woa_eff = fp8_value(untau(mwoa.bytes)).astype(np.float64) * mwoa.row_scales[:, None]
    wob_eff = fp8_value(untau(mwob.bytes)).astype(np.float64) * mwob.row_scales[:, None]

    d = L.Dims(hidden=H, heads=nh, head_dim=D, q_lora=1280, o_lora=olora, groups=groups, expert=576,
               experts=64, top=6, vocab=16160, eps=eps, hc_iters=20, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    g, snaps = _build_staged_dense(device, d, T, keys, nh, D)
    ix_score = np.ones((keys, T), np.float32)  # all positive -> mkey > thr=0 -> kept
    ix_thr = np.zeros((64, T), np.int32)
    ix_lslots = np.zeros((8, T), np.int32)
    ix_lslots[7] = 2000000000  # tie cut > 16777216 -> always allow
    dev = {
        "arena": arena, "cmp_lat": cmp_lat, "cslots": cslots,
        "q_all": q_all, "o1": np.zeros((1, W, D), np.float32),
        "m1": np.full((1, W), -1e30, np.float32), "l1": np.zeros((1, W), np.float32),
        "ix_score": ix_score, "ix_thr": ix_thr, "ix_lslots": ix_lslots,
        "attn_sink": sink, "cs_inv": np.ascontiguousarray(cs_inv.T),
        "wo_a.0": mwoa.bytes, "wo_a.0.s": mwoa.row_scales,
        "wo_b": mwob.bytes, "wo_b.s": mwob.row_scales, "one": np.ones(1, np.float32),
        "dep": np.zeros(64, np.float32),
    }
    dev = {k: v for k, v in dev.items() if k in g.tensors}
    got = Run(device, recipes, g)(dev, ["p_out"] + ["s_" + n for n in snaps])
    g.destroy()
    p_attn = got["p_out"]  # (T,H)

    # numpy: compressed attention over visible latents (ratio=1 -> token t sees keys 0..t)
    o2 = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        lim = min((t + 1) // ratio, keys)
        kk = latd[:lim]
        for h in range(nh):
            q = qf[t * nh + h]
            sc = (kk @ q) * scale
            m = sc.max()
            e = np.exp(sc - m)
            o2[t, h] = (bf16_round(e) @ kk) / e.sum()
    # merge: window zero, sink -1e30 -> o = o2; inverse RoPE; output projection
    o_inv = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        for h in range(nh):
            o_inv[t, h] = _rope_swap(o2[t, h], cs_inv[:, t])
    def wo_proj(oinv):
        p = np.zeros((T, H), np.float64)
        for t in range(T):
            flat = oinv[t].reshape(nh * D)
            outs = []
            for gp in range(groups):
                gpart = flat[gp * per : (gp + 1) * per]
                sg = pow2_scale(float(np.abs(gpart).max()))
                og = fp8_value(fp8_byte(gpart / sg, np.zeros(per, np.int64))) * sg
                outs.append(og @ woa_eff.T)
            cat = np.concatenate(outs)
            sb = pow2_scale(float(np.abs(cat).max()))
            ob = fp8_value(fp8_byte(cat / sb, np.zeros(olora * groups, np.int64))) * sb
            p[t] = ob @ wob_eff.T
        return p

    p_want = wo_proj(o_inv)
    print("\n=== dense compressed attention ===")
    _report("ks", _bf16_to_f32(got["s_ks"]).reshape(keys, D), latd)
    _report("o2", got["s_o2"].reshape(T * nh, D), o2.reshape(T * nh, D))
    _report("o(roped)", got["s_o"].reshape(T * nh, D), o_inv.reshape(T * nh, D))
    rel = _report("p_attn", p_attn, p_want)
    # self-consistency of the fp8 output projection: numpy wo applied to the DEVICE o_inv.  The raw
    # p_attn carries fp8 pow2-grid sensitivity (a ~1e-7 o_inv difference can flip a pow2 scale boundary),
    # so the strong gate is this self-consistency; the full p_attn is bounded grossly (as the index test).
    o_dev = got["s_o"].reshape(T, nh, D).astype(np.float64)
    rwo = _report("p(wo dev-o)", p_attn, wo_proj(o_dev))
    print(f"  ||p_attn||={np.linalg.norm(p_attn):.4e} ||p_want||={np.linalg.norm(p_want):.4e}")
    assert rwo < 1e-3, f"output projection self-consistency diverges: {rwo:.3e}"
    assert rel < 2e-2, f"dense compressed attention diverges: norm_rel {rel:.3e}"