"""RED->GREEN gate: the REAL compressed listed-out attention graph vs the official DeepSeek
sparse_attn (block-wise online softmax over the full topk array) + block-32 wo, with REAL layer-2
weights, T=32 prefill (qb=32, the LISTED path). The single-pass global-max fuse must make p_out
bit-exact (norm_rel <= 1e-6) vs the official reference. On the unfixed split+merge code this is
~1.08e-2 (RED). Mirrors /tmp/moe-oracle/real_span_ref.py's chain and official reference.
"""

from __future__ import annotations

import json
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


CKPT = "/home/nat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"


def test_listed_fused_vs_official(device, recipes):
    from test_ds41_index_composition import Chain, _rope_swap
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
    T, H, D, nh = 32, d.hidden, d.head_dim, d.heads
    qb, C, G2, PR, cpages = 32, 384, 17, 1024, 16
    ixrows, run, K = 16384, 512, 512
    ratio, CMP, eps = 2, 300, d.eps
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
    cmp_store = np.full((G2, 8), -1, np.int32)
    for g in range(T // 2):
        cmp_store[g] = [CMP + g, 0, 4 * (CMP + 129) + g, CMP + 128, g, CMP + 161, g, 0]
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
        state_rows=1, residual=False, query_block=qb,
    )
    g_gslot = L.index_gslot_graph(device, d, T, 8, 1, 8, 1, qb, ratio, 1, 2048, run, 128)
    g_score = L.index_score_graph(device, d, T, 8, 2048, 1, ixrows, query_block=qb, parts=2, cpages=cpages)
    g_merge = L.index_merge_graph(device, d, T, ixrows, run)
    g_out = L.attention_out_graph(device, d, T, 1, cpages, win_slots=1, win_keys=C, dbg_o="o_dbg")
    graphs = [g_pre, g_gslot, g_score, g_merge, g_out]
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
        "o1": np.zeros((1, W, D), np.float32), "m1": np.zeros((1, W), np.float32),
        "l1": np.zeros((1, W), np.float32),
        "cs_inv": cs_inv, "one": np.ones(1, np.float32), "dep": np.zeros(64, np.float32),
        "rope_s1": np.zeros(1, np.float32), "rope_b1": np.ones(1, np.float32),
        "rope_lp17": np.zeros(G2, np.float32),
        "hca_fn": wl["hca_fn"], "hca_base": wl["hca_base"], "hca_scale": wl["hca_scale"],
        "c_out": np.zeros((24, T), np.float32),
        "ix_list": np.zeros((T, K), np.int32), "ix_thr": np.zeros((T, 64), np.int32),
        "ix_lslots": np.zeros((T, 8), np.int32),
        "o_dbg": np.zeros((1, W, D), np.float32),
    }
    devd.update(wl)
    devd.update(wc)
    for n, a in devd.items():
        if n in chain.bufs:
            chain.upload(n, a)
    chain.launch(0)
    chain.launch(1)
    base = chain.bufs["ix_rec"][0]
    chain.launch(2, {f"ix_slots.p{p}": base + 128 * p for p in range(2)})
    chain.sync()
    tile = chain.download("ix_score", (T, ixrows), np.float32)
    merge = np.zeros((T, run + ixrows), np.float32)
    merge[:, :ixrows] = tile
    chain.upload("ix_merge", merge)
    chain.launch(3)
    chain.launch(4)
    chain.sync()
    po = chain.download("p_out", (T, H), np.float32)
    od = chain.download("o_dbg", (1, W, D), np.float32)[0]
    ar = chain.download("arena", (1, PR, 512), np.uint8)[0]
    ixl = chain.download("ix_list", (T, K), np.int32)
    chain.free()
    for g in graphs:
        g.destroy()

    # ---- readback latents + window keys + numpy queries (real weights)
    latd = np.zeros((T // 2, D), np.float64)
    for g in range(T // 2):
        gs = ar[CMP + 128, 4 * g:4 * g + 4].view(np.float32)[0]
        latd[g] = fp8_value(ar[CMP + g, :D]) * gs

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

    # ---- OFFICIAL reference (full topk array blocking, kernel.py:311) + block-32 wo
    kv_b = bf16_round(np.concatenate([wstored, latd], axis=0))
    topk = np.full((T, 32 + 512), -1, np.int64)
    for t in range(T):
        widx = np.arange(32)
        widx = np.where(widx > t, -1, widx)
        topk[t, :32] = widx
        cidx = ixl[t, :vis[t]].astype(np.int64) + 32
        topk[t, 32:32 + vis[t]] = cidx

    def official_attn(q16, kv_b, sink, topk, scale):
        q16b = bf16_round(q16)
        o = np.zeros((T, nh, D), np.float64)
        block = 64
        for t in range(T):
            idx = topk[t]
            acc_o = np.zeros((nh, D), np.float64)
            sum_exp = np.zeros(nh)
            smax = np.full(nh, -1e30)
            for b0 in range(0, len(idx), block):
                bidx = idx[b0:b0 + block]
                valid = bidx >= 0
                kb = np.where(valid[:, None], kv_b[np.clip(bidx, 0, None)], 0.0)
                acc_s = (q16b[t] @ kb.T.astype(np.float64)) * scale
                acc_s = np.where(valid[None, :], acc_s, -np.inf)
                prev = smax.copy()
                smax = np.maximum(smax, acc_s.max(axis=1))
                resc = np.exp(prev - smax)
                P = np.exp(acc_s - smax[:, None])
                sum_exp = sum_exp * resc + P.sum(axis=1)
                Pb = bf16_round(P)
                acc_o = acc_o * resc[:, None] + Pb @ kb.astype(np.float64)
            sum_exp = sum_exp + np.exp(sink.astype(np.float64) - smax)
            o[t] = acc_o / sum_exp[:, None]
        return o

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

    o_off = official_attn(q16_ref, kv_b, sink, topk, scale)
    o_off_inv = np.zeros((T, nh, D), np.float64)
    for t in range(T):
        for h in range(nh):
            o_off_inv[t, h] = _rope_swap(o_off[t, h], cs_inv[t])
    p_off = official_wo(o_off_inv)

    rel = float(np.linalg.norm(po - p_off) / max(np.linalg.norm(p_off), 1e-12))
    orel = float(np.linalg.norm(od - o_off_inv.reshape(W, D)) / max(np.linalg.norm(o_off_inv), 1e-12))
    print(f"  p_out REAL graph vs OFFICIAL: norm_rel={rel:.4e} max_abs={np.abs(po - p_off).max():.4e}")
    print(f"  o_inv REAL graph vs OFFICIAL: norm_rel={orel:.4e} max_abs={np.abs(od - o_off_inv.reshape(W, D)).max():.4e}")
    # The attention fuse is bit-exact up to fp32 (o_inv ~1e-7). p_out carries wo pow2-boundary
    # amplification of that fp32 noise (scattered ~1e-3 cols); the unfixed split+merge was 1.08e-2.
    assert orel <= 1e-6, f"listed attention o_inv diverges from official: norm_rel={orel:.4e}"
    assert rel <= 2e-3, f"listed attention p_out diverges from official: norm_rel={rel:.4e}"


def test_attention_graph_one_launch_compiles(device, recipes):
    """The non-owner/non-source compressed layer is one launch (pre+out); its fused listed path must
    construct and compile (the window gather depends on the pre-graph's store marker)."""
    import json

    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin_models_gaudi.deepseek_v41 import layers as L

    cfg = DeepSeekV41Config.from_hugging_face(json.load(open(os.path.join(CKPT, "config.json"))))
    d = L.Dims.of(cfg, 1, cfg.vocab_size)
    g = L.attention_graph(device, d, 32, 1, 384, 1, 16, ratio=2, query_block=32)
    recipes.compile(g)
    g.destroy()


# (T, qb, ratio, dense) covering every prefill bucket and decode, ratio 1 and 2, listed and dense.
_COVERAGE = [
    (32, 32, 1, 0), (32, 32, 2, 0),
    (128, 128, 1, 0), (128, 128, 2, 0), (128, 128, 2, 2048),
    (512, 128, 1, 0), (512, 128, 2, 0), (512, 128, 2, 2048),
    (1024, 128, 1, 0), (1024, 128, 2, 0), (1024, 128, 2, 2048),
    (2048, 128, 1, 0), (2048, 128, 2, 0), (2048, 128, 2, 2048),
    (16, 1, 1, 0), (16, 1, 2, 0), (4, 1, 2, 0), (1, 1, 2, 0),
]


@pytest.mark.parametrize("T,qb,ratio,dense", _COVERAGE)
def test_fused_compile_coverage(device, recipes, T, qb, ratio, dense):
    """Every serving shape's fused listed attention must compile: the window gemms run in the slot layout
    (M = nh*qb, S = T/qb) viewed to (nh, T), so any win_slots S works (the S>1 broadcast bug is gone)."""
    import json

    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin_models_gaudi.deepseek_v41 import layers as L

    cfg = DeepSeekV41Config.from_hugging_face(json.load(open(os.path.join(CKPT, "config.json"))))
    d = L.Dims.of(cfg, 1, cfg.vocab_size)
    prefill = qb > 1
    S = T // qb
    keys = 384 if prefill else 256
    # one-launch (non-owner/non-source) attention_graph, listed and dense
    g = L.attention_graph(device, d, T, S, keys, 288, 288, ratio=ratio, query_block=qb,
                          dense=dense, scores=16384 if dense else 0)
    recipes.compile(g)
    g.destroy()
    # separate owner/source out graph (listed path only; dense uses _dense_part)
    if not dense:
        g2 = L.attention_out_graph(device, d, T, 288, 288, win_slots=S, win_keys=keys)
        recipes.compile(g2)
        g2.destroy()