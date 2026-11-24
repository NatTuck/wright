"""Prefill compressed-attention span oracle (DeepSeek-V4.1-Flash, one rank, T=32, ratio-2 owner+source).

The garbage boot's prefill is a short prompt (T=32 bucket, qb=32): NOT the dense path (qb != 128) but the
LISTED path. This runs the genuine graphs of one compressed source/owner layer at that shape, in production
order, with persistent device buffers between launches:

  attention_pre_graph (residual=False: collapse -> window keys store -> window part o1/m1/l1 -> compressor
  store -> index queries ix_q8/ix_qs/ix_w) -> index_gslot_graph -> index_score_graph (prefill bf16, 2 parts)
  -> tile copy -> index_merge_graph (ix_thr/ix_list/ix_lslots) -> attention_out_graph (listed:
  ds_gather_list -> ds_slot_p1 -> ds_slot_mld -> ds_attn_comb_r -> wo_a/wo_b -> p_out).

Every stage output is compared to an independent numpy reference (the checkpoint's Indexer/Compressor/
Attention math). The host tables are the numpy transcription of the Rust fill (crates/resin-gaudi/src/ds41)
for this shape. Two seeds: same shapes, different data.
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
    x = np.asarray(x, np.float64)
    bits = x.astype(np.float32).view(np.int32).astype(np.int64)
    e = np.maximum(((bits >> 23) & 255) - 127, -6)
    quantum = np.exp2((e - 3).astype(np.float64))
    q = np.round(x / quantum)
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


def _cs_row(pos, freqs, inverse=False):
    ang = pos * freqs
    co = np.cos(ang.astype(np.float64)).astype(np.float32).repeat(2)
    si = np.sin(ang.astype(np.float64)).astype(np.float32).repeat(2)
    sign = np.where(np.arange(64) % 2 == 1, 1.0, -1.0).astype(np.float32)
    return np.concatenate([co, si * (-sign if inverse else sign)])


def _build_staged_listed_out(device, d, T, blocks, arena_blocks):
    """Faithful copy of attention_out_graph's LISTED path. o2/m2/l2/o are written DIRECTLY to persistent
    outputs (dedicated buffers, no memcpy of reuse-prone internals) so they are trustworthy stage gates."""
    import struct

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.glm5_3_flash.layers import GEMM_NN, GEMM_NT
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    D, nh = d.head_dim, d.heads
    W = nh * T
    K = L.INDEX_TOP
    lg = L.LayerGraph(device, f"t_listed_out_T{T}", d, T)
    ks = L._gather_listed(lg, arena_blocks, blocks)
    lslots = lg.input("ix_lslots", (8, T), I32)
    qs = lg.view(lg.input("q_all", (D, W), BF16), (D, nh, T))
    sc = lg.tensor((K, nh, T), F32)
    lg.node("batch_gemm", [qs, ks], [sc], GEMM_NT)
    Mp = -(-nh // 64) * 64
    P, mx, ls = lg.tensor((K, nh, T), BF16), lg.tensor((Mp, T), F32), lg.tensor((Mp, T), F32)
    lg.node("rg_ds_slot_p1_f32", [sc, lslots], [P, mx, ls], struct.pack("fi", d.scale, nh))
    pv = lg.tensor((D, nh, T), F32)
    lg.node("batch_gemm", [P, ks], [pv], GEMM_NN)
    o2 = lg.input("o2_dbg", (D, W, 1), F32)
    m2 = lg.input("m2_dbg", (W, 1), F32)
    l2 = lg.input("l2_dbg", (W, 1), F32)
    lg.node(L._slot_merge(W, T), [pv, mx, ls, lslots], [o2, m2, l2])
    o = lg.input("o_dbg", (D, W, 1), F32)
    lg.node("rg_ds_attn_comb_r_f32",
            [lg.input("o1", (D, W, 1), F32), lg.input("m1", (W, 1), F32), lg.input("l1", (W, 1), F32),
             o2, m2, l2, lg.input("attn_sink", (nh,), F32), lg.input("cs_inv", (128, T), F32)],
            [o], struct.pack("i", nh))
    L._attention_out(lg, lg.view(o, (D, W)))
    return lg.g


class Chain:
    """Several genuine graphs launched in order over shared persistent device buffers."""

    def __init__(self, device, recipes, graphs):
        self.device = device
        self.recipes = [recipes.compile(g) for g in graphs]
        self.bufs: dict[str, tuple[int, int]] = {}
        for r in self.recipes:
            for n, t in r.persistent.items():
                nb = max(t.nbytes, 256)
                if n in self.bufs:
                    if self.bufs[n][1] != nb:
                        raise ValueError(f"buffer {n}: {self.bufs[n][1]} vs {nb} bytes")
                else:
                    self.bufs[n] = (device.malloc(nb), nb)

    def upload(self, name, array):
        a = np.ascontiguousarray(array)
        addr, nb = self.bufs[name]
        if a.nbytes > nb:
            raise ValueError(f"{name}: {a.nbytes} > {nb}")
        self.device.h2d(addr, a)

    def launch(self, i, extra=None):
        r = self.recipes[i]
        for n in r.persistent:
            addr = (extra or {}).get(n, self.bufs[n][0])
            r.bind(n, addr)
        r.launch()

    def sync(self):
        self.device.sync()

    def download(self, name, shape, dtype):
        return self.device.d2h(self.bufs[name][0], shape, dtype)

    def free(self):
        for addr, _ in self.bufs.values():
            self.device.free(addr)
        self.bufs.clear()


@pytest.mark.parametrize("seed", [911, 912])
def test_prefill_compressed_span(device, recipes, seed):
    from test_ds41_kernels import bf16_round, fp8_byte, fp8_value, pow2_scale

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.deepseek_v41.weights import mx8_rows
    from resin_models_gaudi.loading.convert import _permute_k, bf16_rne, kperm

    rng = np.random.default_rng(seed)
    H, D, nh, T, qlora, olora, groups = 5120, 512, 8, 32, 1280, 1024, 1
    ID, IH, ratio, eps = 128, 32, 2, 1e-20
    qb, C, G2, PR, cpages = 32, 384, 17, 1024, 16
    ixrows, run, K = 16384, 512, 512
    W, M = nh * T, nh * qb
    per = nh // groups * D
    scale = D**-0.5
    freqs = (1.0 / (10000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64.0))).astype(np.float32)
    pos = np.arange(T)
    vis = (pos + 1) // ratio  # compressed entries each token sees (<= 16)
    CMP = 300  # arena base row of the compressed page (latent 300..427, scale 428, index 429..461)

    def mxfp8(rows, cols):
        b = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
        b[(b & 0x7F) == 0x7F] = 0
        s = rng.integers(113, 115, (rows // 32, cols // 32), dtype=np.uint8)
        return b, s

    def untau(b):
        return _permute_k(b, 64, 4)

    def eff(m):
        return fp8_value(untau(m.bytes)).astype(np.float64) * m.row_scales[:, None]

    wa_b, wa_s = mxfp8(qlora + D, H)
    wqb_b, wqb_s = mxfp8(nh * D, qlora)
    woa_b, woa_s = mxfp8(olora, per)
    wob_b, wob_s = mxfp8(H, olora * groups)
    iwq_b, iwq_s = mxfp8(IH * ID, qlora)
    mwa, mwqb, mwoa, mwob, miwq = (
        mx8_rows(b, s) for b, s in ((wa_b, wa_s), (wqb_b, wqb_s), (woa_b, woa_s), (wob_b, wob_s), (iwq_b, iwq_s))
    )
    wa_eff, wqb_eff, woa_eff, wob_eff, iwq_eff = eff(mwa), eff(mwqb), eff(mwoa), eff(mwob), eff(miwq)
    attn_norm = (1.0 + 0.02 * rng.standard_normal(H)).astype(np.float32)
    kv_norm = (1.0 + 0.02 * rng.standard_normal(D)).astype(np.float32)
    q_norm = (1.0 + 0.02 * rng.standard_normal(qlora)).astype(np.float32)
    cmp_wkv = (rng.standard_normal((D, H)) * 0.02).astype(np.float32)
    cmp_wgate = (rng.standard_normal((D, H)) * 0.02).astype(np.float32)
    cmp_norm = (1.0 + 0.02 * rng.standard_normal(D)).astype(np.float32)
    ix_wk = bf16_rne((rng.standard_normal((ID, D)) * 0.02).astype(np.float32))
    ix_wk_f32 = _bf16_to_f32(ix_wk)
    ix_knorm = (1.0 + 0.02 * rng.standard_normal(ID)).astype(np.float32)
    wproj = bf16_rne((rng.standard_normal((IH, H)) * 0.02).astype(np.float32))
    sink = rng.uniform(-2, 4, nh).astype(np.float32)

    X0 = (rng.standard_normal((T, H)) * 0.5).astype(np.float32)
    X = np.zeros((T, 4 * H), np.float32)
    X[:, :H] = X0
    c = np.zeros((24, T), np.float32)
    c[0] = 1.0
    c[8 + np.arange(4) * 5] = 1.0
    cs = np.stack([_cs_row(t, freqs) for t in pos])  # (T,128)
    cs_inv = np.stack([_cs_row(t, freqs, inverse=True) for t in pos])
    cs_g = np.stack([_cs_row(2 * g, freqs) for g in range(G2)])  # (G2,128); pad group pos 0

    # ---- host tables (numpy transcription of the Rust fill for this shape)
    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = 129 * (pos // 128) + pos % 128
    rows[:, 3] = 129 * (pos // 128) + 128
    rows[:, 4] = pos % 128
    slots = np.array([[0, 0, 0, 0, T, 0, 0, 128]], np.int32)  # one window slot, win 128
    lat_pages = np.array([[0, 129, 258]], np.int32)
    cmp_store = np.full((G2, 8), -1, np.int32)
    for g in range(T // 2):
        e = g  # fp4: all entries in the page at lat0=CMP, entry e=g
        cmp_store[g] = [CMP, e, 0, 0, 0, 0, 0, 0]
    cmp_groups = np.full((G2, 4), -1, np.int32)
    for g in range(T // 2):
        cmp_groups[g] = [2 * g, 2 * g + 1, 0, 0]
    cmp_rows = np.full(T, -1, np.int32)  # last position 31 odd: no pending store
    blk = np.zeros((T, 2), np.int32)  # one query block: q=0, seen=16 (ratio 2; the fill's last ratio wins)
    blk[0] = [0, T // ratio]
    ix_vis = vis.astype(np.int32)
    cmp_blocks = np.full((T, cpages), -1, np.int32)
    cmp_blocks[:, 0] = CMP
    cmp_loff = np.zeros(64, np.int32)  # owner offset 0 (CMP already in cmp_blocks)
    ix_gbase = np.zeros(1, np.int32)
    ix_pos = np.zeros((T, run), np.int32)

    d = L.Dims(hidden=H, heads=nh, head_dim=D, q_lora=qlora, o_lora=olora, groups=groups, expert=576,
               experts=64, top=6, vocab=16160, eps=eps, hc_iters=20, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    g_pre = L.attention_pre_graph(
        device, d, T, 1, C, 1, ratio=ratio, owner=True, source=True, groups=G2, state_rows=1,
        residual=False, query_block=qb,
    )
    g_gslot = L.index_gslot_graph(device, d, T, 8, 1, 8, 1, qb, ratio, 1, 2048, run, 128)
    g_score = L.index_score_graph(device, d, T, 8, 2048, 1, ixrows, query_block=qb, parts=2, cpages=cpages)
    g_merge = L.index_merge_graph(device, d, T, ixrows, run)
    g_out = L.attention_out_graph(device, d, T, 1, cpages)
    g_staged = _build_staged_listed_out(device, d, T, cpages, 1)
    graphs = [g_pre, g_gslot, g_score, g_merge, g_out, g_staged]
    chain = Chain(device, recipes, graphs)

    arena = np.zeros((1, PR, 512), np.uint8)
    dev = {
        "X": X, "c": c, "attn_norm": attn_norm,
        "w_a": mwa.bytes, "w_a.s": mwa.row_scales, "kv_norm": kv_norm, "q_norm": q_norm,
        "wq_b": mwqb.bytes, "wq_b.s": mwqb.row_scales, "cs": cs, "rows": rows,
        "ix_none": np.zeros((T, 128), np.uint16), "slots": slots, "lat_pages": lat_pages,
        "rope_pages": np.zeros((1, C // 128), np.int32), "arena": arena,
        "cmp_wkv": cmp_wkv, "cmp_wgate": cmp_wgate, "cmp_norm": cmp_norm,
        "ix_wk": ix_wk, "ix_knorm": ix_knorm, "cs_g": cs_g,
        "cmp_store": cmp_store, "cmp_groups": cmp_groups,
        "cmp_state": np.zeros((1, 2 * D), np.float32), "cmp_rows": cmp_rows,
        "ix_wq_b": miwq.bytes, "ix_wq_b.s": miwq.row_scales, "ix_wproj": kperm(wproj),
        "blk": blk, "ix_vis": ix_vis, "cmp_blocks": cmp_blocks, "cmp_loff": cmp_loff,
        "ix_gbase": ix_gbase, "ix_pos": ix_pos,
        "ix_merge": np.zeros((T, run + ixrows), np.float32),
        "o1": np.zeros((1, W, D), np.float32), "m1": np.zeros((1, W), np.float32),
        "l1": np.zeros((1, W), np.float32),
        "attn_sink": sink, "cs_inv": cs_inv,
        "wo_a.0": mwoa.bytes, "wo_a.0.s": mwoa.row_scales,
        "wo_b": mwob.bytes, "wo_b.s": mwob.row_scales,
        "one": np.ones(1, np.float32), "dep": np.zeros(64, np.float32),
        "rope_s1": np.zeros(1, np.float32), "rope_b1": np.ones(1, np.float32),
        "rope_lp17": np.zeros(G2, np.float32),
        "hca_fn": np.zeros((4 * H, 32), np.float32), "hca_base": np.zeros(24, np.float32),
        "hca_scale": np.ones(3, np.float32), "c_out": np.zeros((24, T), np.float32),
        "ix_list": np.zeros((T, K), np.int32), "ix_thr": np.zeros((T, 64), np.int32),
        "ix_lslots": np.zeros((T, 8), np.int32),
    }
    for n, a in dev.items():
        if n in chain.bufs:
            chain.upload(n, a)
    try:
        chain.launch(0)  # pre
        chain.launch(1)  # gslot
        # part p's slots table at byte 4*8*Sp*p of the launch's records (Sp = 4)
        base = chain.bufs["ix_rec"][0]
        chain.launch(2, {f"ix_slots.p{p}": base + 128 * p for p in range(2)})
        chain.sync()
        tile = chain.download("ix_score", (T, ixrows), np.float32)
        merge = np.zeros((T, run + ixrows), np.float32)
        merge[:, :ixrows] = tile  # g=0: the tile at rows 0.. (the production copy_multi)
        chain.upload("ix_merge", merge)
        chain.launch(3)  # merge
        chain.launch(4)  # out (listed)
        chain.sync()
        got = {
            "q_all": chain.download("q_all", (W, D), np.uint16),
            "ix_q8": chain.download("ix_q8", (IH * T, ID), np.uint8),
            "ix_qs": chain.download("ix_qs", (IH * T,), np.float32),
            "ix_w": chain.download("ix_w", (T, IH), np.float32),
            "o1": chain.download("o1", (1, W, D), np.float32)[0],
            "m1": chain.download("m1", (1, W), np.float32)[0],
            "l1": chain.download("l1", (1, W), np.float32)[0],
            "arena": chain.download("arena", (1, PR, 512), np.uint8)[0],
            "ix_score": tile,
            "ix_thr": chain.download("ix_thr", (T, 64), np.int32),
            "ix_list": chain.download("ix_list", (T, K), np.int32),
            "ix_lslots": chain.download("ix_lslots", (T, 8), np.int32),
            "p_out": chain.download("p_out", (T, H), np.float32),
        }
        chain.launch(5)  # staged listed out (direct persistent stage outputs; overwrites p_out)
        chain.sync()
        got["p_out_staged"] = chain.download("p_out", (T, H), np.float32)
        got["o2_dbg"] = chain.download("o2_dbg", (1, W, D), np.float32)[0]
        got["m2_dbg"] = chain.download("m2_dbg", (1, W), np.float32)[0]
        got["l2_dbg"] = chain.download("l2_dbg", (1, W), np.float32)[0]
        got["o_dbg"] = chain.download("o_dbg", (1, W, D), np.float32)[0]
    finally:
        chain.free()
        for g in graphs:
            g.destroy()

    # ---- independent numpy reference
    cv = X0.astype(np.float64)
    r = 1.0 / np.sqrt((cv * cv).mean(axis=1, keepdims=True) + eps)
    h_norm = cv * r * attn_norm
    hs = np.array([pow2_scale(float(np.abs(h_norm[t]).max())) for t in range(T)])
    h8v = np.stack([fp8_value(fp8_byte(h_norm[t] / hs[t], np.zeros(H, np.int64))) for t in range(T)])
    qa = h8v @ wa_eff[:qlora].T
    kva = h8v @ wa_eff[qlora:].T
    kn = np.zeros((T, D), np.float64)
    for t in range(T):
        rk = 1.0 / np.sqrt((kva[t] ** 2).mean() + eps)
        kn[t] = _rope_swap(kva[t] * rk * kv_norm, cs[t])
    wstored = np.stack([
        fp8_value(fp8_byte(kn[t] / pow2_scale(float(np.abs(kn[t]).max())), np.zeros(D, np.int64)))
        * pow2_scale(float(np.abs(kn[t]).max())) for t in range(T)
    ])
    q16 = np.zeros((T, nh, D), np.float64)
    qr_n = np.zeros((T, qlora), np.float64)
    qsq = np.zeros(T)
    for t in range(T):
        rq = 1.0 / np.sqrt((qa[t] ** 2).mean() + eps)
        nq = qa[t] * rq * q_norm
        sq = pow2_scale(float(np.abs(nq).max()))
        qsq[t] = sq
        qr_n[t] = fp8_value(fp8_byte(nq / sq, np.zeros(qlora, np.int64)))
        q = (qr_n[t] @ wqb_eff.T) * sq
        q = q.reshape(nh, D)
        q16[t] = _rope_swap(q, cs[t])
    q16 = bf16_round(q16)
    # window partials (no sink): keys max(0, t-127) .. t
    o1 = np.zeros((T, nh, D), np.float64)
    m1 = np.zeros((T, nh))
    l1 = np.zeros((T, nh))
    for t in range(T):
        kk = wstored[max(0, t - 127) : t + 1]
        sc = np.einsum("hd,kd->hk", q16[t], kk) * scale
        m = sc.max(axis=1, keepdims=True)
        e = bf16_round(np.exp(sc - m))
        l1[t] = e.sum(axis=1)
        m1[t] = m[:, 0]  # sc already carries *scale: the exported max is in scaled-logit units
        o1[t] = (e @ kk) / l1[t, :, None]
    # compressor (ratio 2): latents + index keys
    kv = h_norm @ cmp_wkv.T
    gate = h_norm @ cmp_wgate.T
    lat = np.zeros((T // 2, D), np.float64)
    for g in range(T // 2):
        a, b = 2 * g, 2 * g + 1
        m = np.maximum(gate[a], gate[b])
        ea, eb = np.exp(gate[a] - m), np.exp(gate[b] - m)
        lat[g] = (kv[a] * ea + kv[b] * eb) / (ea + eb)
    latn = np.zeros((T // 2, D), np.float64)
    for g in range(T // 2):
        rr = 1.0 / np.sqrt((lat[g] ** 2).mean() + eps)
        latn[g] = lat[g] * rr * cmp_norm
    latd = np.zeros((T // 2, D), np.float64)
    k8n = np.zeros((T // 2, ID), np.int64)
    sk = np.zeros(T // 2)
    for g in range(T // 2):
        lq = _fq4(_rope_swap(latn[g], cs_g[g]), 1)
        latd[g] = lq  # fp4 value (exact); store keeps codes+scales
        k = bf16_round(latn[g]).astype(np.float32) @ ix_wk_f32.T
        rr = 1.0 / np.sqrt((k.astype(np.float64) ** 2).mean() + eps)
        kni = k * rr * ix_knorm
        ix32 = _fq4(_rope_swap(kni, cs_g[g]), 0)
        sk[g] = 1.0  # fp4 index: e8m0 scale folded into the fp8 dequant in the gather
        k8n[g] = fp8_byte(ix32, np.zeros(ID, np.int64))  # fp8(ix32) exact (fp4 <=5 sig bits)
    k8v = fp8_value(k8n.astype(np.uint8))
    # index queries: wq_b -> RoPE -> fq4(mode 0) -> fp8 pow2 per (token, head) column
    qi = (qr_n @ iwq_eff.T) * qsq[:, None]  # (T, IH*ID)
    qi = qi.reshape(T, IH, ID)
    q8n = np.zeros((T * IH, ID), np.int64)
    qsc = np.zeros(T * IH)
    for t in range(T):
        for h in range(IH):
            v = _fq4(_rope_swap(qi[t, h], cs[t]), 0)
            s = pow2_scale(float(np.abs(v).max()))
            qsc[t * IH + h] = s
            q8n[t * IH + h] = fp8_byte(v / s, np.zeros(ID, np.int64))
    q8v = fp8_value(q8n.astype(np.uint8))
    ix_w = bf16_round(h_norm).astype(np.float32) @ _bf16_to_f32(wproj).T  # (T, IH) f32
    # index scores: bf16 head products, relu, weights * 1/64 * qscale, * kscale; visibility j < (t+1)/2
    score = np.full((T, ixrows), -3.0e38, np.float32)
    for t in range(T):
        score[t, : vis[t]] = 0.0
        for h in range(IH):
            dots = k8v[: vis[t]] @ q8v[t * IH + h]  # raw fp8 dot products
            sb = bf16_round(dots).astype(np.float64)
            wt = float(ix_w[t, h]) * (1.0 / 64.0) * qsc[t * IH + h]
            score[t, : vis[t]] += (np.maximum(sb, 0.0) * wt).astype(np.float32)
        score[t, : vis[t]] *= sk[: vis[t]].astype(np.float32)
    # top-512 list: vis <= 16 < 512 -> every visible entry, position order
    ix_list = np.zeros((T, K), np.int32)
    ix_lslots = np.zeros((T, 8), np.int32)
    for t in range(T):
        ix_list[t, : vis[t]] = np.arange(vis[t])
        ix_lslots[t] = [0 if vis[t] > 0 else -1, nh * t, K, 0, vis[t], t, 0, 0x7FFFFFFF]
    # listed compressed attention + merge with the window part + sink, inverse RoPE, wo_a/wo_b
    o2 = np.zeros((T, nh, D), np.float64)
    m2 = np.full((T, nh), -1.0e30)
    l2 = np.zeros((T, nh))
    for t in range(T):
        if vis[t] == 0:
            continue
        kk = latd[: vis[t]]
        sc = np.einsum("hd,kd->hk", q16[t], kk) * scale
        m = sc.max(axis=1, keepdims=True)
        e = bf16_round(np.exp(sc - m))
        l2[t] = e.sum(axis=1)
        m2[t] = m[:, 0]
        o2[t] = (e @ kk) / l2[t, :, None]
    o = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        for h in range(nh):
            M = max(m1[t, h], m2[t, h], float(sink[h]))
            a1 = l1[t, h] * np.exp(m1[t, h] - M)
            a2 = l2[t, h] * np.exp(m2[t, h] - M)
            den = a1 + a2 + np.exp(float(sink[h]) - M)
            o[t, h] = (a1 * o1[t, h] + a2 * o2[t, h]) / den
    o_inv = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        for h in range(nh):
            o_inv[t, h] = _rope_swap(o[t, h], cs_inv[t])
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

    print(f"\n=== prefill compressed span T={T} seed={seed} ===")
    ar = got["arena"]
    from test_ds41_fp4_kv import _decode_latent, _decode_index
    lat_ok, ix_ok = True, True
    for g in range(T // 2):
        lat0 = int(cmp_store[g, 0]); e = int(cmp_store[g, 1])
        gotlat = _decode_latent(ar, lat0, e)
        rl = _report(f"latent{g}", gotlat, latd[g])
        lat_ok &= rl < 2e-3
        gotix = _decode_index(ar, lat0, e)
        ri = _report(f"index{g}", gotix, k8v[g] * sk[g])
        ix_ok &= ri < 2e-3
    assert lat_ok, "stored latents diverge"
    assert ix_ok, "stored index keys diverge"
    rq = _report("q_all", _bf16_to_f32(got["q_all"]), q16.reshape(W, D))
    r1 = _report("o1", got["o1"].reshape(W, D), o1.reshape(W, D))
    r2 = _report("m1", got["m1"], m1.reshape(W))
    r3 = _report("l1", got["l1"], l1.reshape(W))
    rq8 = _report("ix_q8", fp8_value(got["ix_q8"]) * got["ix_qs"][:, None], q8v * qsc[:, None])
    rw = _report("ix_w", got["ix_w"], ix_w)
    # scores: compare the visible region (the rest is NINF both sides); ragged per token -> flatten
    sv_got = np.concatenate([got["ix_score"][t, : vis[t]] for t in range(T) if vis[t] > 0])
    sv_want = np.concatenate([score[t, : vis[t]] for t in range(T) if vis[t] > 0])
    rs = _report("ix_score", sv_got, sv_want)
    ninf_ok = all(np.all(got["ix_score"][t, vis[t] : 2048] < -1e30) for t in range(T))
    list_ok = all(
        np.array_equal(got["ix_list"][t, : vis[t]], np.arange(vis[t])) for t in range(T)
    )
    cnt_ok = all(int(got["ix_lslots"][t, 4]) == vis[t] for t in range(T))
    rel = _report("p_attn", got["p_out"], p_want)
    print(f"  ||p_attn||={np.linalg.norm(got['p_out']):.4e} ||p_want||={np.linalg.norm(p_want):.4e}")
    print(f"  ninf_ok={ninf_ok} list_ok={list_ok} cnt_ok={cnt_ok}")
    # staged listed-out stage outputs (direct persistent buffers -> trustworthy)
    _report("o2", got["o2_dbg"], o2.reshape(W, D))
    _report("m2", got["m2_dbg"], m2.reshape(W))
    _report("l2", got["l2_dbg"], l2.reshape(W))
    ro = _report("o(merge+rope)", got["o_dbg"], o_inv.reshape(W, D))
    o_dev = got["o_dbg"].reshape(T, nh, D).astype(np.float64)
    rwo = _report("p(wo dev-o)", got["p_out"], wo_proj(o_dev))
    rstag = _report("p_staged", got["p_out_staged"], got["p_out"])
    print(f"  staged-vs-genuine p_out bit-rel {rstag:.3e}")
    fails = []
    if rq >= 2e-3:
        fails.append(f"q_all {rq:.3e}")
    if max(r1, r2, r3) >= 3e-3:
        fails.append(f"window {r1:.3e}/{r2:.3e}/{r3:.3e}")
    if rq8 >= 5e-3:  # fp4 tie-rounding noise (<= ~20/131072 elems)
        fails.append(f"ix_q8 {rq8:.3e}")
    if rw >= 2e-3:
        fails.append(f"ix_w {rw:.3e}")
    if rs >= 4e-3:
        fails.append(f"ix_score {rs:.3e}")
    if not ninf_ok:
        fails.append("ninf")
    if not list_ok:
        fails.append("list")
    if not cnt_ok:
        fails.append("cnt")
    if ro >= 2e-3:
        fails.append(f"o(merge+rope) {ro:.3e}")
    if rwo >= 1e-3:
        fails.append(f"wo self-consistency {rwo:.3e}")
    if rstag >= 1e-6:
        fails.append(f"staged!=genuine {rstag:.3e}")
    # raw p_attn carries fp8 pow2-grid sensitivity in the sink-dominated regime; bound gross garbage only
    if rel >= 2e-2:
        fails.append(f"p_attn {rel:.3e}")
    assert not fails, "diverging stages: " + ", ".join(fails)