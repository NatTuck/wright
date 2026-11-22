"""One-layer full ffn_graph MoE composition oracle (DeepSeek-V4.1-Flash, one rank).

The component kernels are verified individually (test_ds41_kernels). This tests the COMPOSITION in
ffn_graph: collapse8 -> router gemm -> rg_moe_route_ssp -> shared expert -> routed experts -> combine.
It feeds a known residual X and real-format weights, runs the device graph, and compares every stage
against a COMPLETE independent numpy MoE reference (true MXFP8/MXFP4 dequant from the checkpoint-format
codes/scales, independent fp8 activation rounding, sqrt-softplus router, clamped SwiGLU, weighted combine).

The combine guid is parameterized so the same graph can run the f32 build (rg_ds_xsum_f32, what ffn_graph
currently calls) and the bf16 build (rg_ds_xsum_b16_f32). y2 and sh are BF16 products (023b10b), so the
f32 build reads them as f32 -- this localizes the first diverging stage.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
from test_ds41_kernels import (
    Run,
    bf16_round,
    fp8_byte,
    fp8_value,
    mx4_values,
    pow2_scale,
)


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


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _silu_clamp(gate: np.ndarray, up: np.ndarray, lim: float) -> np.ndarray:
    g = np.minimum(gate, lim)
    u = np.clip(up, -lim, lim)
    sig = np.where(g >= 0, 1.0 / (1.0 + np.exp(-np.abs(g))), np.exp(-np.abs(g)) / (1.0 + np.exp(-np.abs(g))))
    return g * sig * u


@pytest.fixture(scope="module")
def ffn_case():
    """Synthetic checkpoint-format weights + the device tensors the graph binds, and the numpy truth."""
    from resin_models_gaudi.deepseek_v41.weights import (
        OCP,
        even_odd_rows,
        expert_k,
        mx8_rows,
        slot_rows,
    )
    from resin_models_gaudi.loading.convert import bf16_rne, kperm

    rng = np.random.default_rng(20261122)
    H, E, K, T, w, lim, rscale, eps = 5120, 64, 6, 4, 576, 10.0, 1.5, 1e-20
    w0 = w // 128 * 128
    t = w - w0
    S = K * T

    # ---- residual input: stream 0 carries the hidden state; pre-mix selects stream 0 only.
    X0 = (rng.standard_normal((T, H)) * 0.5).astype(np.float32)
    X = np.zeros((T, 4 * H), np.float32)
    X[:, :H] = X0
    ffn_norm = (1.0 + 0.02 * rng.standard_normal(H)).astype(np.float32)
    # c (T,24): pre=[1,0,0,0], post=0, comb=identity (entries 8,13,18,23 = 1)
    c = np.zeros((T, 24), np.float32)
    c[:, 0] = 1.0
    for j in range(4):
        c[:, 8 + 4 * j + j] = 1.0

    # ---- router (bf16 gate weight, E x H) + bias
    gate = bf16_rne((rng.standard_normal((E, H)) * 0.02).astype(np.float32))
    router_bias = (rng.standard_normal(E) * 0.3).astype(np.float32)
    gate_f32 = _bf16_to_f32(gate)

    # ---- shared expert: MXFP8 (OCP E4M3 bytes + 32x32 E8M0 block scales)
    def mxfp8(rows, cols, lo=120, hi=128):
        b = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
        b[b == 0x7F] = 0
        b[b == 0xFF] = 0
        s = rng.integers(lo, hi, (rows // 32, cols // 32), dtype=np.uint8)
        return b, s

    w1b, w1s = mxfp8(w, H)
    w3b, w3s = mxfp8(w, H)
    w2b, w2s = mxfp8(H, w)
    m1, m3, m2 = mx8_rows(w1b, w1s), mx8_rows(w3b, w3s), mx8_rows(w2b, w2s)
    sh_w13 = np.concatenate([m1.bytes, m3.bytes])
    sh_w13_s = np.concatenate([m1.row_scales, m3.row_scales])
    order = even_odd_rows(H)
    sh_w2 = np.ascontiguousarray(m2.bytes[order])
    sh_w2_s = m2.row_scales[order]

    def true_mx8(b, s):
        ss = np.exp2(s.astype(np.float64) - 127).repeat(32, 0).repeat(32, 1)
        return OCP[b] * ss[: b.shape[0], : b.shape[1]]

    sh_w13_true = np.concatenate([true_mx8(w1b, w1s), true_mx8(w3b, w3s)])  # (2w,H) gate|up
    sh_w2_true = true_mx8(w2b, w2s)  # (H,w)

    # ---- routed experts: MXFP4 (E2M1 codes + per-32 E8M0 scales), per-expert exponent spread
    which, rows = slot_rows(w)
    c13 = np.empty((E, 2 * w, H // 2), np.uint8)
    s13 = np.empty((E, 2 * w, H // 32), np.uint8)
    c2n = np.empty((E, H, w // 2), np.uint8)
    s2n = np.empty((E, H, w // 32), np.uint8)
    for e in range(E):
        base = 112 + (e % 9)  # per-expert scale band -> per-expert k (real x13.c/x2.c vary)
        for m, proj in enumerate(("w1", "w3")):
            pick = rows[which == m]
            cb = rng.integers(0, 256, (w, H // 2), dtype=np.uint8)
            sb = rng.integers(base, base + 6, (w, H // 32), dtype=np.uint8)
            c13[e, which == m] = cb[pick]
            s13[e, which == m] = sb[pick]
        c2n[e] = rng.integers(0, 256, (H, w // 2), dtype=np.uint8)
        s2n[e] = rng.integers(base, base + 6, (H, w // 32), dtype=np.uint8)
    k13 = expert_k(s13.reshape(E, -1))
    k2 = expert_k(s2n.reshape(E, -1))
    c2 = np.ascontiguousarray(c2n[:, order]).reshape(E, -1)
    s2 = np.ascontiguousarray(s2n[:, order]).reshape(E, -1)
    x13c = np.repeat(np.exp2(k13.astype(np.float32))[:, None], 2 * w, 1)  # (E,2w)
    x2c = np.repeat(np.exp2(k2.astype(np.float32))[:, None], H, 1)  # (E,H)

    dev = {
        "X": X, "p": np.zeros((1, T, H), np.float32), "c": np.ascontiguousarray(c.T),
        "meta": np.array([0, T, 0, 0], np.int32), "one": np.ones(1, np.float32),
        "ffn_norm": ffn_norm, "router": kperm(gate), "router_bias": router_bias,
        "sh_w13": sh_w13, "sh_w13.s": sh_w13_s, "sh_w2": sh_w2, "sh_w2.s": sh_w2_s,
        "x13": c13.reshape(E, -1), "x13.s": s13.reshape(E, -1), "x13.k": k13,
        "x2": c2, "x2.s": s2, "x2.k": k2, "x13.c": x13c, "x2.c": x2c,
        "hcf_fn": np.zeros((4 * H, 32), np.float32), "hcf_base": np.zeros(24, np.float32),
        "hcf_scale": np.ones(3, np.float32), "dep": np.zeros(64, np.float32),
    }
    truth = dict(H=H, E=E, K=K, T=T, w=w, w0=w0, t=t, S=S, lim=lim, rscale=rscale, eps=eps,
                 X0=X0, ffn_norm=ffn_norm, gate_f32=gate_f32, router_bias=router_bias,
                 sh_w13_true=sh_w13_true, sh_w2_true=sh_w2_true,
                 c13=c13, s13=s13, c2n=c2n, s2n=s2n, order=order)
    return dev, truth


def _numpy_reference(truth, dev):
    """Complete independent MoE reference. Returns dict of per-stage numpy arrays (natural order)."""
    H, E, K, T, w, w0, t = (truth[k] for k in ("H", "E", "K", "T", "w", "w0", "t"))
    lim, rscale, eps = truth["lim"], truth["rscale"], truth["eps"]
    X0, ffn_norm = truth["X0"], truth["ffn_norm"]
    out = {}
    # collapse8 (pre=[1,0,0,0]): h32 = RMSNorm(X0)*w
    cv = X0.astype(np.float64)
    r = 1.0 / np.sqrt((cv * cv).mean(axis=1, keepdims=True) + eps)
    h32 = (cv * r * ffn_norm).astype(np.float32)
    out["h32"] = h32
    h16 = bf16_round(h32)
    hs = np.array([pow2_scale(float(np.abs(h32[t_]).max())) for t_ in range(T)], np.float32)
    out["hs"] = hs
    # router logits = bf16(h32) @ gate^T
    logits = (h16.astype(np.float64) @ truth["gate_f32"].T).astype(np.float32)
    out["logits"] = logits
    s = np.sqrt(np.log1p(np.exp(logits.astype(np.float64))))
    score = s + truth["router_bias"]
    ids = np.full((T, 64), -1, np.int32)
    wts = np.zeros((T, 64), np.float32)
    for t_ in range(T):
        pick = np.argsort(-score[t_], kind="stable")[:K]
        ids[t_, :K] = pick
        ww = s[t_, pick]
        wts[t_, :K] = rscale * ww / (ww.sum() + 1e-20)
    out["ids"], out["wts"] = ids, wts
    # shared expert
    sh_w13_true, sh_w2_true = truth["sh_w13_true"], truth["sh_w2_true"]
    g13 = np.zeros((T, 2 * w), np.float32)
    shs = np.zeros(T, np.float32)
    sh = np.zeros((T, H), np.float32)
    for t_ in range(T):
        h8v = fp8_value(fp8_byte(h32[t_].astype(np.float64) / hs[t_], np.zeros(H, np.int64)))
        g = (h8v @ sh_w13_true.T).astype(np.float32)  # sa=1
        g13[t_] = g
        gate = g[:w].astype(np.float64) * hs[t_]
        up = g[w:].astype(np.float64) * hs[t_]
        hsw = _silu_clamp(gate, up, lim)
        sc = pow2_scale(float(np.abs(hsw).max()))
        shs[t_] = sc
        a8v = fp8_value(fp8_byte(hsw / sc, np.zeros(w, np.int64)))
        sh[t_] = bf16_round((a8v @ sh_w2_true.T).astype(np.float32))
    out["g13"], out["shs"], out["sh"] = g13, shs, sh
    # routed experts (true MXFP4 weights, dequantized activations) -- as test_expert_block_matches_numpy
    sx = hs.copy()
    out["sx"] = sx
    y13 = np.zeros((T, K, 2 * w), np.float64)
    hsl = np.zeros((T, K), np.float32)
    y2 = np.zeros((T, K, H), np.float64)
    p_ffn = sh.astype(np.float64) * shs[:, None]  # shared contribution sh*shs
    for t_ in range(T):
        scale = sx[t_]
        xv = fp8_value(fp8_byte(h32[t_].astype(np.float64) / scale, np.zeros(H, np.int64))) * scale
        for k in range(K):
            e = ids[t_, k]
            W13 = mx4_values(truth["c13"][e].reshape(2 * w, H // 2), truth["s13"][e].reshape(2 * w, H // 32))
            yy = W13 @ xv
            yy = bf16_round(yy)  # dt=BF16 at w=576
            y13[t_, k] = yy
            gate = np.concatenate([yy[:w0], yy[2 * w0 : 2 * w0 + t]])
            up = np.concatenate([yy[w0 : 2 * w0], yy[2 * w0 + t :]])
            hsw = _silu_clamp(gate, up, lim)
            sc = pow2_scale(float(np.abs(hsw).max()))
            hsl[t_, k] = sc
            hv = fp8_value(fp8_byte(hsw / sc, np.zeros(w, np.int64))) * sc
            W2 = mx4_values(truth["c2n"][e].reshape(H, w // 2), truth["s2n"][e].reshape(H, w // 32))
            y2k = bf16_round(W2 @ hv)
            y2[t_, k] = y2k
            p_ffn[t_] += wts[t_, k] * y2k
    out["y13"], out["hsl"], out["y2"], out["p_ffn"] = y13, hsl, y2, p_ffn.astype(np.float32)
    return out


def _build_staged(device, dims, T, combine_guid, snap=True):
    """Faithful copy of ffn_graph's body (non-paged), feeding the collapse input X_out directly and
    omitting residual()/site() (they produce X_out and c_out, not p_out). Snapshots every stage."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32

    d = dims
    lg = L.LayerGraph(device, f"t_ffncomp_{combine_guid}_T{T}", d, T)
    H, K, E, w = d.hidden, d.top, d.experts, d.expert
    w0 = w // 128 * 128
    S = K * T
    Xout = lg.input("X", (4 * H, T), F32)
    snaps = {}

    def sn(name, ten):
        if snap:
            o = lg.input("s_" + name, ten.shape, ten.dtype)
            lg.node("memcpy", [ten], [o])
            snaps[name] = o
        return ten

    h16, h32, h8, hs = lg.collapse8(Xout, lg.input("c", (T, 24), F32), "ffn_norm")
    sn("h16", h16)
    sn("h32", h32)
    sn("h8", h8)
    sn("hs", hs)
    logits = lg.gemm(h16, "router", E)
    sn("logits", logits)
    g13 = lg.gemm8(h8, "sh_w13", 2 * w)
    sn("g13", g13)
    xq, sx = lg.tensor((H, 1, S), F8), lg.tensor((1, T), F32)
    lg.node("rg_ds_xq_lin_f32", [h32], [xq, sx], struct.pack("i", K))
    sn("xq", xq)
    sn("sx", sx)
    ids, wts = lg.tensor((64, T), I32), lg.tensor((64, T), F32)
    lg.node("rg_moe_route_ssp_f32",
            [logits, lg.input("router_bias", (E,), F32), lg.meta()],
            [ids, wts], struct.pack("iif", K, E, d.route_scale))
    sn("ids", ids)
    sn("wts", wts)
    a8, shs = lg.q8p(g13, sin=hs, swiglu=True)
    sn("a8", a8)
    sn("shs", shs)
    sh = lg.gemm8(a8, "sh_w2", H, dt=BF16)
    sn("sh", sh)
    dt = L.expert_out_dtype(lg)
    y13 = L._w13(lg, xq, ids, K, dt)
    sn("y13", y13)
    hq, hsl = lg.tensor((w, 1, S), F8), lg.tensor((1, S), F32)
    lg.node(L._xact_lin(dt),
            [y13, y13, lg.input("x13.c", (2 * w, E), F32), sx, ids],
            [hq, hsl], struct.pack("ifii", K, d.limit, w - w0, 0))
    sn("hq", hq)
    sn("hsl", hsl)
    y2 = L._w2(lg, hq, ids, K, BF16)
    sn("y2", y2)
    lg.node(combine_guid,
            [y2, lg.input("x2.c", (H, E), F32), hsl, ids, wts, sh, shs],
            [lg.p_out()], struct.pack("i", K))
    return lg.g, snaps


def _dims(truth):
    from resin_models_gaudi.deepseek_v41 import layers as L

    return L.Dims(hidden=truth["H"], heads=8, head_dim=512, q_lora=1280, o_lora=1024, groups=1,
                  expert=truth["w"], experts=truth["E"], top=truth["K"], vocab=16160, eps=truth["eps"],
                  hc_iters=20, hc_eps=1e-6, limit=truth["lim"], route_scale=truth["rscale"], ranks=1)


def _report(name, got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    ad = np.abs(got - want)
    denom = np.maximum(np.abs(want), 1e-6)
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    print(f"  {name:10s} max_abs={ad.max():.4e} max_rel={(ad/denom).max():.4e} norm_rel={rel:.4e}")
    return rel


def test_ffn_composition_localizes_combine(device, recipes, ffn_case):

    dev, truth = ffn_case
    T, H, K = truth["T"], truth["H"], truth["K"]
    ref = _numpy_reference(truth, dev)
    dims = _dims(truth)
    inv = np.argsort(truth["order"])  # undo even_odd row order on the hidden axis

    results = {}
    for guid in ("rg_ds_xsum_f32", "rg_ds_xsum_b16_f32"):
        g, snaps = _build_staged(device, dims, T, guid)
        inputs = {k: v for k, v in dev.items() if k in g.tensors}
        out_names = ["p_out"] + ["s_" + n for n in snaps]
        got = Run(device, recipes, g)(inputs, out_names)
        results[guid] = got
        g.destroy()
        print(f"\n=== combine guid {guid} ===")
        # Run returns arrays in reversed(device shape): device (H,T) -> (T,H) already token-major.
        _report("h32", got["s_h32"], ref["h32"])
        _report("hs", got["s_hs"][:, 0], ref["hs"])
        # router
        _report("logits", got["s_logits"], ref["logits"])
        ids_dev = got["s_ids"]  # (T,64)
        wts_dev = got["s_wts"]  # (T,64)
        ids_match = np.array_equal(np.sort(ids_dev[:, :K], axis=1), np.sort(ref["ids"][:, :K], axis=1))
        print(f"  ids match (top-K sets): {ids_match}")
        _report("wts", wts_dev[:, :K], ref["wts"][:, :K])
        # shared sh (bf16, even_odd rows -> natural)
        sh_dev = _bf16_to_f32(got["s_sh"])  # (T,H) even_odd on H axis
        sh_nat = sh_dev[:, inv]
        _report("sh", sh_nat, ref["sh"])
        _report("shs", got["s_shs"][:, 0], ref["shs"])
        # routed y2 (H,1,S) bf16 -> (S,H) even_odd -> natural. Device y2 is the RAW fp8 product (pre x2.c);
        # the reference y2 is the fully dequantized product, so scale device y2 by x2.c (=2^k2) and hsl.
        y2_dev = _bf16_to_f32(got["s_y2"]).reshape(T * K, H)  # (S,H) even_odd rows
        y2_nat = y2_dev[:, inv].reshape(T, K, H)
        x2c = dev["x2.c"]  # (E,H) = 2^k2 per expert
        hsl_dev = got["s_hsl"][:, 0].reshape(T, K)
        y2_scaled = np.empty_like(y2_nat)
        for t_ in range(T):
            for k in range(K):
                e = ids_dev[t_, k]
                y2_scaled[t_, k] = y2_nat[t_, k] * x2c[e, 0] * hsl_dev[t_, k]
        _report("y2", y2_scaled, ref["y2"])
        _report("hsl", hsl_dev, ref["hsl"])
        # final combine
        rel_p = _report("p_ffn", got["p_out"], ref["p_ffn"])
        results[guid + "_rel_p"] = rel_p

    # The bf16 build matches the full independent reference; the f32 build (what ffn_graph calls) does not.
    assert results["rg_ds_xsum_b16_f32_rel_p"] < 3e-3, "bf16 combine should match the numpy MoE reference"
    assert results["rg_ds_xsum_f32_rel_p"] > 0.1, "f32 combine on BF16 products should diverge (the bug)"


def test_staged_matches_genuine_ffn_graph(device, recipes, ffn_case):
    """Faithfulness: the staged reconstruction equals the genuine ffn_graph p_out bit-exactly (same combine
    guid as production: rg_ds_xsum_b16_f32 after the fix)."""
    dev, truth = ffn_case
    T = truth["T"]
    dims = _dims(truth)
    g, _ = _build_staged(device, dims, T, "rg_ds_xsum_b16_f32", snap=False)
    inputs = {k: v for k, v in dev.items() if k in g.tensors}
    staged = Run(device, recipes, g)(inputs, ["p_out"])["p_out"]
    g.destroy()
    genuine = _run_genuine(device, recipes, dev, truth)
    np.testing.assert_array_equal(staged, genuine)


def _run_genuine(device, recipes, dev, truth):
    from resin_models_gaudi.deepseek_v41 import layers as L

    dims = _dims(truth)
    gg = L.ffn_graph(device, dims, truth["T"])
    gin = {k: v for k, v in dev.items() if k in gg.tensors}
    # residual()/site() write the persistent intermediates X_out and c_out; bind scratch buffers for them.
    gin["X_out"] = np.zeros((truth["T"], 4 * truth["H"]), np.float32)
    gin["c_out"] = np.zeros((24, truth["T"]), np.float32)
    out = Run(device, recipes, gg)(gin, ["p_out"])["p_out"]
    gg.destroy()
    return out


def test_genuine_ffn_graph_matches_numpy(device, recipes, ffn_case):
    """REGRESSION GATE: the genuine ffn_graph (production combine guid) against the complete independent
    numpy MoE reference. RED on the unfixed code (rg_ds_xsum_f32 reading BF16 products as f32), GREEN once
    ffn_graph calls the bf16 combine build."""
    dev, truth = ffn_case
    ref = _numpy_reference(truth, dev)
    genuine = _run_genuine(device, recipes, dev, truth)
    rel = _report("genuine p_ffn", genuine, ref["p_ffn"])
    assert rel < 3e-3, f"genuine ffn_graph diverges from the numpy MoE reference: norm_rel {rel:.3e}"