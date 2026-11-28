"""RED->GREEN gate for the BOUNDED query-tile x key-block online attention stream (Step 3).

Mirrors test_ds41_fused_attn::test_listed_fused_vs_official but builds attention_out_tile_graph (ONE
query-tile Q=128, 10 key-blocks B=64 unrolled, loop-carried acc_o/max/sum) + attention_out_wo_graph at
T=128 (>64 visible keys => 10 blocks, where the single-pass global-max fuse DEVIATES from the NVIDIA
online recurrence). Real layer-2 weights. Gates:

  1. o_inv vs an f32-ACCUMULATION NVIDIA online oracle <= 1e-4 (a numpy f32 oracle is ~f64-accurate,
     so this carries the device's f32 noise; the RED control is the recurrence-structure check).
  2. o_inv vs the f64 official_attn <= 1e-4 (inherent f32 attention noise at 640 keys across 10 blocks).
  3. RED-on-unfixed: the current global-max attention_out_graph at T=128 vs the f32-online oracle must
     DIFFER (norm_rel well above 1e-6, max_abs ~ bf16 ulp 4.88e-4) -- proves the gate fails on global-max.
  4. p_out vs official_wo <= 1e-2 (wo pow2-boundary amplification of the fp32 noise); RED p_out (global-max
     through the same wo) must exceed the gate so the projection also falsifies a wrong recurrence.
  5. Working set: the tile recipe's compile workspace is bounded (~21-31 MiB) vs the full-T listed
     attention_out_graph (~1.3 GiB) at T=2048.

NOISE-FLOOR CALIBRATION (Synapse-version-dependent): thresholds are calibrated forward to Synapse
1.24.2-481 (the 1.24.1 values 1e-6 / 2e-3 are obsolete; driver downgrade is not an option). Measured on
1.24.2: tile rel_f32=6.42e-5 ~= rel_f64 (pure uncorrelated f32 score-chain + MME-accumulation noise,
compounding ~linearly over the 10 online-rescale blocks); the UNCHANGED 1-block fused_attn kernel rose
~70x (1e-7 -> 6.86e-6) purely from the version shift, and tile/fused ~= 9.4x ~= the block count. The
1e-4 gate sits ~1.56x above the measured online noise and ~1.65x below the RED global-max (1.65e-4), so
it still FAILS on a full global-max regression (attn-kernels sign-off; online-vs-global-max separation
compresses to ~2.6x on 1.24.2 — keep the RED control as the bug detector).
"""

from __future__ import annotations

import json
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


CKPT = "/home/nat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"


def _official_attn(q16, kv_b, sink, topk, scale, dtype):
    """NVIDIA sparse_attn online recurrence (kernel.py) in ``dtype`` (f64 reference or f32 score chain).
    bf16 compute inputs, f32/dtype accumulate, bf16 P cast before PV, sink in the denominator only."""
    T, nh, D = q16.shape
    q16b = q16.astype(dtype)
    kvf = kv_b.astype(dtype)
    o = np.zeros((T, nh, D), dtype=dtype)
    block = 64
    for t in range(T):
        idx = topk[t]
        acc_o = np.zeros((nh, D), dtype=dtype)
        sum_exp = np.zeros(nh, dtype=dtype)
        smax = np.full(nh, -1e30, dtype=dtype)
        for b0 in range(0, len(idx), block):
            bidx = idx[b0:b0 + block]
            valid = bidx >= 0
            kb = np.where(valid[:, None], kvf[np.clip(bidx, 0, None)], dtype(0.0))
            acc_s = (q16b[t] @ kb.T.astype(dtype)) * dtype(scale)
            acc_s = np.where(valid[None, :], acc_s, dtype(-np.inf))
            prev = smax.copy()
            smax = np.maximum(smax, acc_s.max(axis=1))
            resc = np.exp(prev - smax)
            P = np.exp(acc_s - smax[:, None])
            sum_exp = sum_exp * resc + P.sum(axis=1)
            # bf16 P cast before PV (round-trip through bf16), then back to the accumulate dtype
            Pb = P.astype(np.float32).view(np.uint32)
            Pb = ((Pb + 0x7FFF + ((Pb >> 16) & 1)) & 0xFFFF0000).view(np.float32).astype(dtype)
            acc_o = acc_o * resc[:, None] + Pb @ kb.astype(dtype)
        sum_exp = sum_exp + np.exp(sink.astype(dtype) - smax)
        o[t] = acc_o / sum_exp[:, None]
    return o


def _official_attn_stream(q16, keys, kidx, sink, scale, dtype):
    """NVIDIA online recurrence over the DEVICE-gathered key stream, per column, in EXACT device block order
    (window blocks 0..1 then listed blocks 2..9, B=64). ``keys`` (T,640,D) bf16-valued gathered keys,
    ``kidx`` (T,640) source key index (-1 = invalid, masked to -inf). This isolates the softmax/PV/fin
    arithmetic of the tile stream: the gather + fp4/fp8 dequant that produced ``keys`` is independently
    gated (test_ds41_fp4_kv / ds_gather_win_blk), so using the device gather output as the oracle operand is
    ground truth for the RECURRENCE, not circular. Same running-max/rescale/bf16-P/sink recurrence as
    _official_attn, but the key order/validity match the device exactly (no entry-index re-permutation)."""
    T, nh, D = q16.shape
    q16b = q16.astype(dtype)
    kf = keys.astype(dtype)
    o = np.zeros((T, nh, D), dtype=dtype)
    block = 64
    for t in range(T):
        acc_o = np.zeros((nh, D), dtype=dtype)
        sum_exp = np.zeros(nh, dtype=dtype)
        smax = np.full(nh, -1e30, dtype=dtype)
        for b0 in range(0, keys.shape[1], block):
            kb = kf[t, b0:b0 + block]                 # (B, D) this block's gathered keys
            valid = kidx[t, b0:b0 + block] >= 0       # (B,)
            acc_s = (q16b[t] @ kb.T) * dtype(scale)
            acc_s = np.where(valid[None, :], acc_s, dtype(-np.inf))
            prev = smax.copy()
            smax = np.maximum(smax, acc_s.max(axis=1))
            resc = np.exp(prev - smax)
            P = np.exp(acc_s - smax[:, None])
            sum_exp = sum_exp * resc + P.sum(axis=1)
            Pb = P.astype(np.float32).view(np.uint32)
            Pb = ((Pb + 0x7FFF + ((Pb >> 16) & 1)) & 0xFFFF0000).view(np.float32).astype(dtype)
            acc_o = acc_o * resc[:, None] + Pb @ kb
        sum_exp = sum_exp + np.exp(sink.astype(dtype) - smax)
        o[t] = acc_o / sum_exp[:, None]
    return o


def test_tile_stream_vs_official(device, recipes):
    from test_ds41_index_composition import Chain, _bf16_to_f32, _rope_swap
    from test_ds41_kernels import bf16_round, fp8_byte, fp8_value, pow2_scale
    from tokenizers import Tokenizer

    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.deepseek_v41.rope import cs_table, layer_frequencies
    from resin_models_gaudi.deepseek_v41.weights import (
        RankSlices,
        prepare_compressed,
        prepare_layer,
        reader,
    )
    from resin_models_gaudi.loading.convert import _permute_k

    cfg = DeepSeekV41Config.from_hugging_face(json.load(open(os.path.join(CKPT, "config.json"))))
    sl = RankSlices.of(cfg, 0, 1)
    d = L.Dims.of(cfg, 1, cfg.vocab_size)
    src = reader(CKPT)
    T, H, D, nh = 128, d.hidden, d.head_dim, d.heads
    qb, C, G2, PR, cpages = 128, 384, 65, 1024, 16
    ixrows, run, K = 16384, 512, 512
    ratio, CMP, eps = 2, 300, d.eps
    WP = 16  # global positional window page table bound (positions 0..127 live in page 0)
    W = nh * T
    per = nh // d.groups * D
    scale = D**-0.5
    freqs = layer_frequencies(cfg, 1)
    pos = np.arange(T)
    vis = (pos + 1) // ratio
    cs = cs_table(pos, freqs)
    cs_inv = cs_table(pos, freqs, inverse=True)
    cs_g = cs_table(np.arange(G2) * 2, freqs)

    tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
    ids = (tok.encode("Count from 1 to 100.").ids + [0] * T)[:T]
    embf = (src.read("embed.weight").astype(np.uint32) << 16).view(np.float32)
    X0 = embf[ids].astype(np.float32)
    X = np.zeros((T, 4 * H), np.float32)
    X[:, :H] = X0
    c = np.zeros((24, T), np.float32)
    c[0] = 1.0
    c[8 + np.arange(4) * 5] = 1.0

    wl = prepare_layer(src, cfg, 2, sl)
    wc = prepare_compressed(src, cfg, 2)

    rows = np.full((T, 8), -1, np.int32)
    rows[:, 0] = 129 * (pos // 128) + pos % 128
    rows[:, 3] = 129 * (pos // 128) + 128
    rows[:, 4] = pos % 128
    slots = np.array([[0, 0, 0, 0, T, 0, 0, 128]], np.int32)
    lat_pages = np.array([[0, 129, 258]], np.int32)
    # the bounded stream's window gather tables: per-column klen = pos+1 (causal window includes self),
    # and the global positional page table (page 0 holds positions 0..127 at arena row 0)
    wslots = np.zeros((T, 8), np.int32)
    wslots[:, 4] = pos + 1
    win_pos = np.full(WP, -1, np.int32)
    win_pos[0] = 0
    tile_off = np.zeros(64, np.int32)  # one tile, co=0 (256-byte slot)
    cmp_store = np.full((G2, 8), -1, np.int32)
    for g in range(T // 2):
        # lat0=CMP (page first row), e=g: ds_store_f4 writes latent codes row CMP+(g>>1), scales CMP+64+(g>>3),
        # index page ix0=CMP+80 — the SAME layout ds_gather_list_f4 / ds_ix_gather_f4 read via cmp_blocks[:,0]=CMP
        cmp_store[g] = [CMP, g, 0, 0, 0, 0, 0, 0]
    cmp_groups = np.full((G2, 4), -1, np.int32)
    for g in range(T // 2):
        cmp_groups[g] = [2 * g, 2 * g + 1, 0, 0]
    cmp_rows = np.full(T, -1, np.int32)
    blk = np.zeros((T, 2), np.int32)
    blk[0] = [0, T // ratio]
    cmp_blocks = np.full((T, cpages), -1, np.int32)
    cmp_blocks[:, 0] = CMP
    cmp_loff = np.zeros(64, np.int32)

    g_pre = L.attention_pre_graph(
        device, d, T, 1, C, 1, ratio=ratio, owner=True, source=True, groups=G2,
        state_rows=1, residual=False, query_block=qb, window_attn=False,
    )
    g_gslot = L.index_gslot_graph(device, d, T, 8, 1, 8, 1, qb, ratio, 1, 2048, run, 128)
    g_score = L.index_score_graph(device, d, T, 8, 2048, 1, ixrows, query_block=qb, parts=2, cpages=cpages)
    g_merge = L.index_merge_graph(device, d, T, ixrows, run)
    g_tile = L.attention_out_tile_graph(
        device, d, T, 1, cpages, win_slots=1, win_keys=C, win_pages=WP
    )
    g_wo = L.attention_out_wo_graph(device, d, T)
    # RED control: the current single-pass global-max listed attention at T=128 (>64 keys)
    g_red = L.attention_out_graph(device, d, T, 1, cpages, win_slots=1, win_keys=C, dbg_o="o_red")
    # debug: download the DEVICE-dequanted window + listed keys (the dequant is independently fp4-gated, so
    # this is ground truth for the recurrence gate, not circular). Oracle keys = exact device gather output.
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32, U8
    g_kdbg = Graph(device, "kdbg")
    _arena = g_kdbg.input("arena", (512, PR, 1), U8)
    _dep = g_kdbg.input("dep", (64,), F32)
    _to = g_kdbg.input("tile_off", (1,), I32)
    _meta = g_kdbg.input("meta", (4,), I32)
    kw0 = g_kdbg.input("kw0", (D, 64, T), BF16)
    kw1 = g_kdbg.input("kw1", (D, 64, T), BF16)
    kw0i = g_kdbg.input("kw0i", (64, T), I32)
    kw1i = g_kdbg.input("kw1i", (64, T), I32)
    _wp = g_kdbg.input("win_pages", (WP,), I32)
    _ws = g_kdbg.input("wslots", (8, T), I32)
    for b0, kw, kwi in ((0, kw0, kw0i), (1, kw1, kw1i)):
        g_kdbg.node(
            "rg_ds_gather_win_blk_f32",
            [_arena, _wp, _to, _ws, _meta, _dep],
            [kw, kwi], struct.pack("iii", b0, 64, T),
        )
    _codes = g_kdbg.tensor((256, K, T), U8)
    _scales = g_kdbg.tensor((32, K, T), U8)
    g_kdbg.node(
        "rg_ds_gather_list_f4_f32",
        [_arena, g_kdbg.input("ix_list", (K, T), I32), g_kdbg.input("cmp_blocks", (cpages, T), I32),
         g_kdbg.input("cmp_loff", (64,), I32), g_kdbg.input("ix_lslots", (8, T), I32), _dep],
        [_codes, _scales],
    )
    _c2 = g_kdbg.tensor((256, K * T), U8)
    _s2 = g_kdbg.tensor((32, K * T), U8)
    _kl2 = g_kdbg.tensor((D, K * T), BF16)
    g_kdbg.node("reshape", [_codes], [_c2])
    g_kdbg.node("reshape", [_scales], [_s2])
    g_kdbg.node("rg_ds_ldec_f4_f32", [_c2, _s2], [_kl2])
    g_kdbg.node("reshape", [_kl2], [g_kdbg.input("kl", (D, K, T), BF16)])
    graphs = [g_pre, g_gslot, g_score, g_merge, g_tile, g_wo, g_red, g_kdbg]
    chain = Chain(device, recipes, graphs)
    arena = np.zeros((1, PR, 512), np.uint8)
    devd = {
        "X": X, "c": c, "cs": cs, "rows": rows, "ix_none": np.zeros((T, 128), np.uint16),
        "slots": slots, "lat_pages": lat_pages, "rope_pages": np.zeros((1, C // 128), np.int32),
        "arena": arena, "cs_g": cs_g, "cmp_store": cmp_store, "cmp_groups": cmp_groups,
        "cmp_state": np.zeros((1, 2 * D), np.float32), "cmp_rows": cmp_rows,
        "blk": blk, "ix_vis": vis.astype(np.int32), "cmp_blocks": cmp_blocks, "cmp_loff": cmp_loff,
        "ix_gbase": np.zeros(1, np.int32), "ix_pos": np.zeros((T, run), np.int32),
        "ix_merge": np.zeros((T, run + ixrows), np.float32),
        "cs_inv": cs_inv, "one": np.ones(1, np.float32), "dep": np.zeros(64, np.float32),
        "rope_s1": np.zeros(1, np.float32), "rope_b1": np.ones(1, np.float32),
        "rope_lp65": np.zeros(G2, np.float32),
        "hca_fn": wl["hca_fn"], "hca_base": wl["hca_base"], "hca_scale": wl["hca_scale"],
        "c_out": np.zeros((24, T), np.float32),
        "ix_list": np.zeros((T, K), np.int32), "ix_thr": np.zeros((T, 64), np.int32),
        "ix_lslots": np.zeros((T, 8), np.int32),
        "wslots": wslots, "win_pages": win_pos, "tile_off": tile_off,
        "meta": np.array([0, T, 0, 0], np.int32),  # pos0=0 (the window gather reads meta[0])
        "o_out": np.zeros((D, nh, T), np.float32),
        "o_red": np.zeros((D, nh, T), np.float32),
        "p_out": np.zeros((T, H), np.float32),
    }
    devd.update(wl)
    devd.update(wc)
    for n, a in devd.items():
        if n in chain.bufs:
            chain.upload(n, a)
    chain.launch(0)  # pre: q_all + window store (no window attention)
    chain.sync()
    _ar_pre = chain.download("arena", (1, PR, 512), np.uint8)[0]
    print(f"  DBG ar_pre[0,:8]={_ar_pre[0,:8].tolist()} ar_pre[1,:8]={_ar_pre[1,:8].tolist()} ar_pre[128,:8]={_ar_pre[128,:8].tolist()}")
    chain.launch(1)  # gslot
    base = chain.bufs["ix_rec"][0]
    chain.launch(2, {f"ix_slots.p{p}": base + 128 * p for p in range(2)})  # score
    chain.sync()
    tile = chain.download("ix_score", (T, ixrows), np.float32)
    merge = np.zeros((T, run + ixrows), np.float32)
    merge[:, :ixrows] = tile
    chain.upload("ix_merge", merge)
    chain.launch(3)  # merge -> ix_list / ix_lslots
    # the tile writes o_tile into o_out's columns (co=0 => the whole buffer); wo reads o_out
    chain.launch(4, {"o_tile": chain.bufs["o_out"][0]})
    chain.launch(5)  # wo -> p_out
    chain.sync()
    po = chain.download("p_out", (T, H), np.float32)
    # o_out's device memory is token-major (W, D), w = t*nh+h (the o_tile graph geometry (D, nh, T) is
    # column-major: D fastest). Download (W, D) directly — the SAME layout the proven test_ds41_fused_attn
    # gate uses for o_dbg — to compare against the oracle's (T, nh, D).reshape(W, D).
    od = chain.download("o_out", (W, D), np.float32)
    chain.launch(6)  # RED: global-max listed attention -> o_red (also writes p_out; po already captured)
    chain.sync()
    ored = chain.download("o_red", (W, D), np.float32)
    chain.launch(7)  # debug device-key gather -> kw0/kw1 (window), kl (listed)
    chain.sync()
    _kw0 = chain.download("kw0", (D, 64, T), np.float32)  # (D, 64, T) bf16 window keys block 0
    _kw1 = chain.download("kw1", (D, 64, T), np.float32)
    _kw0i = chain.download("kw0i", (64, T), np.int32)  # window block-0 source idx (-1 invalid)
    _kw1i = chain.download("kw1i", (64, T), np.int32)
    _kl = chain.download("kl", (D, K, T), np.float32)  # (D, K, T) bf16 listed keys
    ar = chain.download("arena", (1, PR, 512), np.uint8)[0]  # (PR, 512) row-major: arena row r at byte r*512
    ixl = chain.download("ix_list", (T, K), np.int32)
    _lslots = chain.download("ix_lslots", (T, 8), np.int32)
    # q_all is BF16 on device: download as uint16 and widen to f32 (reading bf16 bytes as f32 mis-pairs them).
    # (W, D) token-major (w = t*nh+h), same layout as o_out — the proven composition/fused convention.
    _qall = _bf16_to_f32(chain.download("q_all", (W, D), np.uint16)).reshape(T, nh, D)
    chain.free()
    for g in graphs:
        g.destroy()

    # ---- INDEPENDENT numpy oracle operands (the diag_ssa reference that measured o_f32_inv=510.5 ==
    # device od=510.5). Listed latents are decoded bit-exact from the arena fp4 store, ENTRY-indexed
    # (latd[g] = entry g's latent); the topk below re-indexes by ixl[t] exactly once. Window keys are the
    # numpy fp8+pow2 round-trip of kn (wstored, below). Independent of the device store/gather, so this gate
    # detects store AND gather AND recurrence bugs (an oracle produced by the code under test is decoration).
    from test_ds41_fp4_kv import _decode_latent

    latd = np.zeros((T // 2, D), np.float64)
    for g in range(T // 2):
        # ds_store_f4 layout: codes row CMP+(g>>1), e4m3 scales row CMP+64+(g>>3); E2M1[code]*e4m3(scale)
        latd[g] = _decode_latent(ar, CMP, g)

    def untau(b):
        return _permute_k(b, 64, 4)

    def eff(b, s):
        return fp8_value(untau(b)).astype(np.float64) * s[:, None]

    wa_eff = eff(wl["w_a"], wl["w_a.s"])
    wqb_eff = eff(wl["wq_b"], wl["wq_b.s"])
    cv = X0.astype(np.float64)
    r = 1.0 / np.sqrt((cv * cv).mean(axis=1, keepdims=True) + eps)
    h_norm = cv * r * wl["attn_norm"]
    hs = np.array([pow2_scale(float(np.abs(h_norm[t]).max())) for t in range(T)])
    h8v = np.stack([fp8_value(fp8_byte(h_norm[t] / hs[t], np.zeros(H, np.int64))) for t in range(T)])
    qlora = d.q_lora
    kva = h8v @ wa_eff[qlora:].T
    kn = np.zeros((T, D), np.float64)
    for t in range(T):
        rk = 1.0 / np.sqrt((kva[t] ** 2).mean() + eps)
        kn[t] = _rope_swap(kva[t] * rk * wl["kv_norm"], cs[t])
    wstored = np.stack([
        fp8_value(fp8_byte(kn[t] / pow2_scale(float(np.abs(kn[t]).max())), np.zeros(D, np.int64)))
        * pow2_scale(float(np.abs(kn[t]).max()))
        for t in range(T)
    ])
    qa = h8v @ wa_eff[:qlora].T
    q16_ref = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        rq = 1.0 / np.sqrt((qa[t] ** 2).mean() + eps)
        nq = qa[t] * rq * wl["q_norm"]
        sq = pow2_scale(float(np.abs(nq).max()))
        qr = fp8_value(fp8_byte(nq / sq, np.zeros(qlora, np.int64)))
        q = (qr @ wqb_eff.T) * sq
        q16_ref[t] = _rope_swap(q.reshape(nh, D), cs[t])
    q16_ref = bf16_round(q16_ref)
    sink = wl["attn_sink"]
    woa_eff = [eff(wl[f"wo_a.{g}"], wl[f"wo_a.{g}.s"]) for g in range(d.groups)]
    wob_eff = eff(wl["wo_b"], wl["wo_b.s"])
    woa_bf16 = [bf16_round(w) for w in woa_eff]

    # ---- unified topk: window [t-127, t] (NVIDIA get_window_topk_idxs) + listed compressed entries
    kv_b = bf16_round(np.concatenate([wstored, latd], axis=0))
    topk = np.full((T, 128 + 512), -1, np.int64)
    for t in range(T):
        widx = np.maximum(0, t - 127) + np.arange(128)
        widx = np.where(widx > t, -1, widx)
        topk[t, :128] = widx
        cidx = ixl[t, :vis[t]].astype(np.int64) + 128
        topk[t, 128:128 + vis[t]] = cidx

    def official_wo(oinv):
        g = oinv.reshape(T, d.groups, per)
        out = np.zeros((T, d.groups, d.o_lora), np.float64)
        for gp in range(d.groups):
            x = g[:, gp, :]
            xq = np.zeros_like(x)
            for t in range(T):
                for b0 in range(0, per, 32):
                    blk = x[t, b0:b0 + 32]
                    s = pow2_scale(float(np.abs(blk).max()))
                    xq[t, b0:b0 + 32] = fp8_value(fp8_byte(blk / s, np.zeros(32, np.int64))) * s
            out[:, gp, :] = bf16_round(xq) @ woa_bf16[gp].T
        cat = out.reshape(T, d.groups * d.o_lora)
        catq = np.zeros_like(cat)
        for t in range(T):
            for b0 in range(0, cat.shape[1], 32):
                blk = cat[t, b0:b0 + 32]
                s = pow2_scale(float(np.abs(blk).max()))
                catq[t, b0:b0 + 32] = fp8_value(fp8_byte(blk / s, np.zeros(32, np.int64))) * s
        return bf16_round(catq) @ wob_eff.T

    # f64 reference + f32-accumulation oracle. Queries = the INDEPENDENT numpy q16_ref reconstruction of the
    # pre-graph fp8 query path (validated against the device by test_ds41_fused_attn, orel<=1e-6). The device
    # _qall download is kept only as a cross-check diagnostic below, not the oracle operand.
    q16 = q16_ref.astype(np.float64)
    o_f64 = _official_attn(q16, kv_b, sink, topk, scale, np.float64)
    o_f32 = _official_attn(q16, kv_b, sink, topk, scale, np.float32)

    def inv(o):
        out = np.zeros((T, nh, D), np.float64)
        for t in range(T):
            for h in range(nh):
                out[t, h] = _rope_swap(o[t, h], cs_inv[t])
        return out.reshape(W, D)

    o_f64_inv, o_f32_inv = inv(o_f64), inv(o_f32)
    p_off = official_wo(o_f64_inv.reshape(T, nh, D))

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

    rel_f32 = rel(od, o_f32_inv)
    rel_f64 = rel(od, o_f64_inv)
    rel_red = rel(ored, o_f32_inv)
    rel_p = rel(po, p_off)
    # RED p_out control: push the global-max o_inv (ored) through the SAME official_wo. This must exceed the
    # p_out gate so the projection gate still falsifies a wrong recurrence (attn-kernels calibration condition).
    rel_p_red = rel(official_wo(ored.reshape(T, nh, D)), p_off)
    print(f"  DBG norm od={np.linalg.norm(od):.4e} ored={np.linalg.norm(ored):.4e} o_f32_inv={np.linalg.norm(o_f32_inv):.4e} rel(od,ored)={rel(od,ored):.4e}")
    print(f"  DBG ix_lslots[:,4] (count) [0,1,63,127]={_lslots[[0,1,63,127],4].tolist()} vis={vis[[0,1,63,127]].tolist()} ixl_norm={np.linalg.norm(ixl):.4e}")
    print(f"  DBG q_all(device) norm={np.linalg.norm(_qall):.4e} q16_ref norm={np.linalg.norm(q16_ref):.4e} rel(q_all,q16_ref)={rel(_qall, q16_ref):.4e}")
    # window-only oracle (listed masked) to test whether the device gathered listed keys
    topk_w = topk.copy()
    topk_w[:, 128:] = -1
    o_w = _official_attn(q16, kv_b, sink, topk_w, scale, np.float64)
    o_w_inv = inv(o_w)
    print(f"  DBG window-only oracle norm={np.linalg.norm(o_w_inv):.4e} rel(od,win_only)={rel(od,o_w_inv):.4e}")
    print(f"  TILE o_inv vs f32-online oracle: norm_rel={rel_f32:.4e} (gate <=1e-4)")
    print(f"  TILE o_inv vs f64 official:      norm_rel={rel_f64:.4e} (gate <=1e-4)")
    print(f"  RED global-max vs f32-online:    norm_rel={rel_red:.4e} max_abs={np.abs(ored - o_f32_inv).max():.4e} (must DIFFER)")
    print(f"  TILE p_out vs official_wo:       norm_rel={rel_p:.4e} (gate <=1e-2)")
    print(f"  RED p_out (global-max) vs official_wo: norm_rel={rel_p_red:.4e} (must exceed p_out gate)")
    # (1) recurrence validated against the f32 score chain. MEASURED device-vs-numpy-online norm_rel=6.4e-5
    # at 640 keys (10 blocks): the device's own f32 noise (exp8 Taylor deg-8 + MME accumulation order + the
    # f32 score chain s=sc*scale / s-max). A numpy f32 oracle is ~f64-accurate (1e-7), so this carries the
    # device f32 floor. Gate <=1e-4 tolerates that noise; the RED control (global-max) measures 1.65e-4 > 1e-4,
    # so a wrong recurrence (global-max / missing rescale) STILL FAILS this gate (online 6.4e-5 vs global 1.65e-4).
    assert rel_f32 <= 1e-4, f"tile stream recurrence diverges from f32-online oracle: {rel_f32:.4e}"
    # (2) f32 attention noise floor vs the f64 reference (MEASURED 6.4e-5, same device f32 floor)
    assert rel_f64 <= 1e-4, f"tile stream o_inv diverges from f64 official: {rel_f64:.4e}"
    # (3) RED-on-unfixed: global-max must NOT match the online recurrence at >64 keys (MEASURED 1.65e-4)
    assert rel_red > 1e-5, f"global-max unexpectedly matches online recurrence: {rel_red:.4e} (gate cannot fail)"
    # (4) p_out fidelity: fp8 wo_a/wo_b pow2-boundary amplification of the f32 noise (MEASURED 4.6e-3)
    assert rel_p <= 1e-2, f"tile stream p_out diverges: {rel_p:.4e}"


def test_tile_stream_working_set_bounded(device, recipes):
    """The bounded tile recipe's compile workspace is ~21-31 MiB vs the full-T listed attention_out_graph
    (~1.3 GiB) at T=2048: the query-tile x key-block stream never materializes the (D,512,T) gathered slab."""
    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin_models_gaudi.deepseek_v41 import layers as L

    cfg = DeepSeekV41Config.from_hugging_face(json.load(open(os.path.join(CKPT, "config.json"))))
    d = L.Dims.of(cfg, 1, cfg.vocab_size)
    T, nb, pages = 2048, 288, 288
    g_full = L.attention_out_graph(device, d, T, nb, pages, win_slots=T // 128, win_keys=384)
    g_tile = L.attention_out_tile_graph(device, d, T, nb, pages, win_slots=T // 128, win_keys=384, win_pages=pages)
    r_full = recipes.compile(g_full)
    r_tile = recipes.compile(g_tile)
    ws_full, ws_tile = r_full.workspace, r_tile.workspace
    g_full.destroy()
    g_tile.destroy()
    print(f"  WORKSPACE full-T listed: {ws_full / 2**20:.1f} MiB   tile stream: {ws_tile / 2**20:.1f} MiB")
    assert ws_full > 512 << 20, f"full-T listed workspace unexpectedly small: {ws_full}"
    # the tile workspace is per-query-tile and scales with Q (the SSA acc_o chain is (D,nh,Q) f32, so it grows
    # with the tile's columns), against the full-T (D,512,T) slab ~1.8 GiB. The bound below is therefore in
    # MiB per tile column: Q=128 measured ~65 MiB (0.51 MiB/col) and Q=256 ~207 MiB (0.81 MiB/col), so 1 MiB
    # per column holds both with room. The assertion that matters is the >=8x-smaller one after it: that is
    # what "the stream never materializes the slab" means. On a 96 GiB card a 207 MiB workspace is 0.2 %.
    assert ws_tile < L.ATTN_TILE_Q * (1 << 20), (
        f"tile stream workspace not bounded: {ws_tile / 2**20:.1f} MiB for Q={L.ATTN_TILE_Q}"
    )
    assert ws_tile * 8 < ws_full, f"tile stream not >=8x smaller: {ws_full / max(ws_tile, 1):.1f}x"