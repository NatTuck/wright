"""One-layer window attention composition oracle (DeepSeek-V4.1-Flash, one rank).

The window store/gather/softmax/merge and ds_rope are verified individually; this tests the COMPOSITION
in attention_window_graph: collapse8 -> w_a (q/kv) -> kv_norm+RoPE -> store -> q_norm -> wq_b -> q RoPE
-> window attention -> inverse RoPE -> wo_a/wo_b -> p_attn, against a complete independent numpy reference.
"""

from __future__ import annotations

import os
import struct

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
    ad = np.abs(got - want)
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    print(f"  {name:12s} max_abs={ad.max():.4e} norm_rel={rel:.4e}")
    return rel




def _build_staged_attn(device, d, T, S, C, win):
    """Faithful copy of attention_window_graph (residual=True), snapshotting intermediates."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.glm5_3_flash.layers import GEMM_NN, GEMM_NT
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    D, nh = d.head_dim, d.heads
    W, M = nh * T, nh * T
    Mp = -(-M // 64) * 64
    lg = L.LayerGraph(device, f"t_attncomp_T{T}", d, T)
    snaps = {}

    def sn(name, ten):
        o = lg.input("s_" + name, ten.shape, ten.dtype)
        lg.node("memcpy", [ten], [o])
        snaps[name] = o
        return ten

    X = lg.residual()
    _, h32, h8, hs = lg.collapse8(X, lg.input("c", (T, 24), F32), "attn_norm")
    sn("h32", h32)
    sn("h8", h8)
    sn("hs", hs)
    wa = lg.gemm8(h8, "w_a", d.q_lora + D)
    sn("wa", wa)
    qa, kva = lg.split0(wa, [d.q_lora, D])
    cs, cs_inv = lg.input("cs", (128, T), F32), lg.input("cs_inv", (128, T), F32)
    kn = lg.ds_rope(kva, cs, 1, norm="kv_norm")
    sn("kn", kn)
    arena = lg.arena(1)
    stored = lg.tensor((64, 1), I32)
    lg.node("rg_mla_store_p2_f32",
            [arena, lg.input("rows", (8, T), I32), kn, lg.input("ix_none", (128, T), BF16)], [stored])
    qr8, sq = lg.q8p(qa, norm="q_norm")
    sn("qr8", qr8)
    sn("sq", sq)
    q = lg.gemm8(qr8, "wq_b", nh * D)
    sn("q", q)
    q16 = lg.ds_rope(lg.view(q, (D, W)), cs, nh, ts=sq, bf16=True)
    sn("q16", q16)
    table = lg.input("slots", (8, S), I32)
    lp, rp = lg.input("lat_pages", (C // 128, S), I32), lg.input("rope_pages", (C // 128, S), I32)
    qs = lg.view(q16, (D, M, S))
    ks = lg.tensor((D, C, S), BF16)
    lg.node("rg_mla_gather_f32", [arena, lp, rp, table, stored], [ks], struct.pack("i", 0))
    sn("ks", ks)
    sc = lg.tensor((C, M, S), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    sn("sc", sc)
    P, mx, ls = lg.tensor((C, M, S), BF16), lg.tensor((Mp, S), F32), lg.tensor((Mp, S), F32)
    lg.node("rg_ds_slot_pw1_f32" if M <= 64 else "rg_ds_slot_pw_f32",
            [sc, table], [P, mx, ls], struct.pack("fi", d.scale, nh))
    pv = lg.tensor((D, M, S), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    o = lg.tensor((D, W, 1), F32)
    lg.node("rg_ds_slot_out_f32", [pv, mx, ls, table, lg.input("attn_sink", (nh,), F32)],
            [o], struct.pack("i", nh))
    sn("o", o)
    oinv = lg.rope(lg.view(o, (D, W)), cs_inv, nh)
    sn("oinv", oinv)
    L._attention_out(lg, oinv)
    lg.site(X, "hca")
    return lg.g, snaps


def test_window_attention_composition(device, recipes):
    from test_ds41_kernels import Run, bf16_round, fp8_byte, fp8_value, pow2_scale

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.deepseek_v41.weights import mx8_rows

    rng = np.random.default_rng(31337)
    H, nh, D, qlora, olora, groups = 5120, 8, 512, 1280, 1024, 1
    T, win, C, S, PR, eps = 4, 128, 256, 1, 1024, 1e-20
    per = nh // groups * D
    scale = D**-0.5
    freqs = (1.0 / (10000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64.0))).astype(np.float32)
    positions = np.arange(T, dtype=np.float32)

    def mxfp8(rows, cols, lo=113, hi=115):
        b = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
        b[(b & 0x7F) == 0x7F] = 0
        s = rng.integers(lo, hi, (rows // 32, cols // 32), dtype=np.uint8)
        return b, s


    # weights (checkpoint MXFP8 -> device fp8 via mx8_rows; oracle uses device-effective dequant)
    wa_b, wa_s = mxfp8(qlora + D, H)
    wqb_b, wqb_s = mxfp8(nh * D, qlora)
    woa_b, woa_s = mxfp8(olora, per)
    wob_b, wob_s = mxfp8(H, olora * groups)
    mwa, mwqb, mwoa, mwob = (mx8_rows(b, s) for b, s in ((wa_b, wa_s), (wqb_b, wqb_s), (woa_b, woa_s), (wob_b, wob_s)))
    attn_norm = (1.0 + 0.02 * rng.standard_normal(H)).astype(np.float32)
    kv_norm = (1.0 + 0.02 * rng.standard_normal(D)).astype(np.float32)
    q_norm = (1.0 + 0.02 * rng.standard_normal(qlora)).astype(np.float32)
    sink = rng.uniform(-2, 4, nh).astype(np.float32)

    # device-effective weights (natural K): the fp8 bytes the GEMM uses, dequanted independently.
    # tau order is _permute_k(groups=4, inner=64); its INVERSE is _permute_k(groups=64, inner=4)
    # (tperm is NOT an involution: transpose (4,64)->(64,4) applied twice is not identity).
    from resin_models_gaudi.loading.convert import _permute_k

    def untau(b):
        return _permute_k(b, 64, 4)

    def eff(m):
        return fp8_value(untau(m.bytes)).astype(np.float64) * m.row_scales[:, None]

    wa_eff, wqb_eff, woa_eff, wob_eff = eff(mwa), eff(mwqb), eff(mwoa), eff(mwob)

    # residual input: stream 0 carries the hidden state
    X0 = (rng.standard_normal((T, H)) * 0.5).astype(np.float32)
    X = np.zeros((T, 4 * H), np.float32)
    X[:, :H] = X0
    c = np.zeros((24, T), np.float32)
    c[0] = 1.0
    c[8 + np.arange(4) * 5] = 1.0

    # rope tables (T,128) -> device (128,T)
    def cs_table(inv):
        ang = positions[:, None] * freqs[None, :]
        co = np.cos(ang.astype(np.float64)).astype(np.float32).repeat(2, 1)
        si = np.sin(ang.astype(np.float64)).astype(np.float32).repeat(2, 1)
        sign = np.where(np.arange(64) % 2 == 1, 1.0, -1.0).astype(np.float32)
        return np.concatenate([co, si * (-sign if inv else sign)], 1)  # (T,128); device cs is (128,T)

    cs, cs_inv = cs_table(False), cs_table(True)
    n = np.arange(T)
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = 129 * (n // 128) + n % 128
    rows[:, 3] = 129 * (n // 128) + 128
    rows[:, 4] = n % 128
    slots = np.array([[0, 0, 0, 0, T, 0, 0, win]], np.int32)
    lat_pages = np.array([[0, 129]], np.int32)

    d = L.Dims(hidden=H, heads=nh, head_dim=D, q_lora=qlora, o_lora=olora, groups=groups, expert=576,
               experts=64, top=6, vocab=16160, eps=eps, hc_iters=20, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    g, snaps = _build_staged_attn(device, d, T, S, C, win)
    dev = {
        "X": X, "p": np.zeros((1, T, H), np.float32), "c": c,
        "X_out": np.zeros((T, 4 * H), np.float32), "c_out": np.zeros((24, T), np.float32),
        "attn_norm": attn_norm, "w_a": mwa.bytes, "w_a.s": mwa.row_scales,
        "kv_norm": kv_norm, "q_norm": q_norm, "wq_b": mwqb.bytes, "wq_b.s": mwqb.row_scales,
        "cs": cs, "cs_inv": cs_inv, "rows": rows, "ix_none": np.zeros((T, 128), np.uint16),
        "slots": slots, "lat_pages": lat_pages, "rope_pages": np.zeros((C // 128, S), np.int32),
        "arena": np.zeros((S, PR, 512), np.uint8), "attn_sink": sink,
        "wo_a.0": mwoa.bytes, "wo_a.0.s": mwoa.row_scales, "wo_b": mwob.bytes, "wo_b.s": mwob.row_scales,
        "one": np.ones(1, np.float32),
        "rope_s8": np.zeros(nh, np.float32), "rope_b8": np.ones(nh, np.float32),
        "rope_lp4": np.zeros(T, np.float32),
        "hca_fn": np.zeros((32, 4 * H), np.float32), "hca_base": np.zeros(24, np.float32),
        "hca_scale": np.ones(3, np.float32),
    }
    dev = {k: v for k, v in dev.items() if k in g.tensors}
    got = Run(device, recipes, g)(dev, ["p_out"] + ["s_" + n for n in snaps])
    g.destroy()
    p_attn = got["p_out"]  # (T,H)

    # ---- independent numpy reference
    cv = X0.astype(np.float64)
    r = 1.0 / np.sqrt((cv * cv).mean(axis=1, keepdims=True) + eps)
    h_norm = cv * r * attn_norm  # (T,H)
    hs = np.array([pow2_scale(float(np.abs(h_norm[t]).max())) for t in range(T)])
    h8v = np.stack([fp8_value(fp8_byte(h_norm[t] / hs[t], np.zeros(H, np.int64))) for t in range(T)])
    qa = h8v @ wa_eff[:qlora].T  # (T,qlora) sa=1
    kva = h8v @ wa_eff[qlora:].T  # (T,D)
    # keys: RMSNorm(kva)*kv_norm then RoPE (cleared ds_rope formula), then fp8 pow2 store
    kn = np.zeros((T, D), np.float64)
    vkn = np.zeros((T, D), np.float64)
    for t in range(T):
        rk = 1.0 / np.sqrt((kva[t] ** 2).mean() + eps)
        vkn[t] = kva[t] * rk * kv_norm
    co, si = cs[:, :64].astype(np.float64), cs[:, 64:].astype(np.float64)  # cs (T,128) as bound
    rr = vkn[:, D - 64 :]
    swap = rr.reshape(T, 32, 2)[:, :, ::-1].reshape(T, 64)
    kn[:, : D - 64] = vkn[:, : D - 64]
    kn[:, D - 64 :] = rr * co + swap * si
    stored = np.stack([fp8_value(fp8_byte(kn[t] / pow2_scale(float(np.abs(kn[t]).max())), np.zeros(D, np.int64)))
                       * pow2_scale(float(np.abs(kn[t]).max())) for t in range(T)])
    # queries: RMSNorm(qa)*q_norm -> fp8 -> wq_b -> RoPE (bf16)
    q16 = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        rq = 1.0 / np.sqrt((qa[t] ** 2).mean() + eps)
        nq = qa[t] * rq * q_norm
        sq = pow2_scale(float(np.abs(nq).max()))
        qr = fp8_value(fp8_byte(nq / sq, np.zeros(qlora, np.int64)))
        q = (qr @ wqb_eff.T) * sq  # (nh*D,)
        q = q.reshape(nh, D)
        rq_ = q[:, D - 64 :]
        sw_ = rq_.reshape(nh, 32, 2)[:, :, ::-1].reshape(nh, 64)
        q[:, D - 64 :] = rq_ * co[t] + sw_ * si[t]
        q16[t] = q
    q16 = bf16_round(q16)
    # window attention with sink
    o = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        k = stored[max(0, t - win +1) : t + 1]
        sc = np.einsum("hd,kd->hk", q16[t], k) * scale
        m = sc.max(axis=1, keepdims=True)
        e = np.exp(sc - m)
        o[t] = (bf16_round(e) @ k) / (e.sum(axis=1, keepdims=True) + np.exp(sink[:, None] - m))
    # inverse RoPE on the output, then wo_a/wo_b
    co_i, si_i = cs_inv[:, :64].astype(np.float64), cs_inv[:, 64:].astype(np.float64)
    o_inv = o.copy()
    for t in range(T):
        ri = o[t, :, D - 64 :]
        swi = ri.reshape(nh, 32, 2)[:, :, ::-1].reshape(nh, 64)
        o_inv[t, :, D - 64 :] = ri * co_i[t] + swi * si_i[t]
    p_want = np.zeros((T, H), np.float64)
    for t in range(T):
        flat = o_inv[t].reshape(nh * D)  # (per*groups,)
        parts = [flat[g * per : (g + 1) * per] for g in range(groups)]
        outs = []
        for gpart in parts:
            sg = pow2_scale(float(np.abs(gpart).max()))
            og = fp8_value(fp8_byte(gpart / sg, np.zeros(per, np.int64))) * sg
            outs.append(og @ woa_eff.T)
        cat = np.concatenate(outs)
        sb = pow2_scale(float(np.abs(cat).max()))
        ob = fp8_value(fp8_byte(cat / sb, np.zeros(olora * groups, np.int64))) * sb
        p_want[t] = ob @ wob_eff.T
    print("\n=== window attention composition (per stage) ===")
    for nm in snaps:
        print(f"  ||device {nm}|| = {np.linalg.norm(got['s_' + nm].astype(np.float64)):.4e}")
    _report("hs", got["s_hs"][:, 0], hs)
    _report("h32", got["s_h32"], h_norm)
    # isolate the w_a GEMM: reference from the DEVICE h8 (untau, dequant) and device-effective weights
    h8_dev = fp8_value(untau(got["s_h8"]))  # (T,H) natural fp8 values
    _report("h8", h8_dev, h8v)
    wa_ref_dev = h8_dev @ wa_eff.T
    _report("wa(devh8)", got["s_wa"], wa_ref_dev)
    _report("wa", got["s_wa"], np.concatenate([qa, kva], axis=1))
    _report("kn", got["s_kn"], kn)
    _report("q16", _bf16_to_f32(got["s_q16"]), q16.reshape(T * nh, D))
    _report("o", got["s_o"].reshape(T * nh, D), o.reshape(T * nh, D))
    _report("oinv", got["s_oinv"], o_inv.reshape(T * nh, D))
    rel = _report("p_attn", p_attn, p_want)
    print(f"  ||p_attn||={np.linalg.norm(p_attn):.4e} ||p_want||={np.linalg.norm(p_want):.4e}")
    assert rel < 3e-3, f"window attention composition diverges: norm_rel {rel:.3e}"