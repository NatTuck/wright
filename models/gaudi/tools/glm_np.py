"""Numpy models of the GLM-5.3-Flash operators, checked against the transformers oracle dumps
(tools/glm_ref_hf.py). They define the algorithms the kernels implement.

  python3 tools/glm_np.py ORACLE.npz        # checks every model against the dumps it finds

KDA (Kimi delta attention, per-channel decay). Recurrence per head (S: (dk, dv)):
  S <- diag(exp(g_t)) S;  S <- S + k_t (beta_t (v_t - S^T k_t))^T;  o_t = S^T q_t
with q, k l2-normalized, q scaled by dk^-0.5, g_t in [-5, 0] per channel.
Chunked form (chunks of C = 64, G = inclusive cumsum of g inside the chunk): Gram products use a
reference point per 16-token sub-chunk s of the ROW index: rows i in s scale by exp(G_i - G_r), keys j
by exp(G_r - G_j), and keys past the sub-chunk are zeroed.
"""
import os
import sys

import numpy as np

F = np.float32


def l2n(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


# ------------------------------------------------------------------ KDA
def kda_recurrent(q, k, v, g, beta, S0=None):
    """q, k, v, g: (T, NH, D); beta: (T, NH). Returns o (T, NH, D), S (NH, D, D)."""
    T, NH, D = q.shape
    q = l2n(q.astype(np.float64)) * D ** -0.5
    k = l2n(k.astype(np.float64))
    v, g, beta = v.astype(np.float64), g.astype(np.float64), beta.astype(np.float64)
    S = np.zeros((NH, D, D)) if S0 is None else S0.astype(np.float64).copy()
    o = np.zeros((T, NH, D))
    for t in range(T):
        S = S * np.exp(g[t])[:, :, None]
        kv = np.einsum("hkv,hk->hv", S, k[t])
        d = (v[t] - kv) * beta[t][:, None]
        S = S + k[t][:, :, None] * d[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", S, q[t])
    return o, S


def kda_blocks(q, k, v, g, beta, S0=None, C=64, SUB=16):
    """The kernel algorithm (float32 throughout). Shapes as kda_recurrent; T padded to C inside."""
    T, NH, D = q.shape
    P = (-T) % C
    pad = lambda x: np.concatenate([x, np.zeros((P,) + x.shape[1:], x.dtype)]) if P else x
    q = pad(l2n(q.astype(F)) * F(D ** -0.5)); k = pad(l2n(k.astype(F)))
    v, g, beta = pad(v.astype(F)), pad(g.astype(F)), pad(beta.astype(F))
    NC = (T + P) // C
    S = np.zeros((NH, D, D), F) if S0 is None else S0.astype(F).copy()
    o = np.zeros_like(v)
    tri = np.tril(np.ones((C, C), bool), -1)                     # strictly lower
    lowq = np.tril(np.ones((C, C), bool))                        # lower incl. diagonal
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        qc, kc, vc, gc, bc = q[sl], k[sl], v[sl], g[sl], beta[sl]  # (C, NH, D) / (C, NH)
        G = np.cumsum(gc, 0)                                      # (C, NH, D), <= 0, decreasing
        # Gram matrices with per-row-sub-chunk reference points
        KK = np.zeros((NH, C, C), F); QK = np.zeros((NH, C, C), F)
        for s in range(C // SUB):
            r = s * SUB
            rows = slice(r, r + SUB)
            Gr = G[r]                                             # (NH, D)
            er = np.exp(G[rows] - Gr)                             # (SUB, NH, D) <= 1
            ek = np.exp(np.minimum(Gr - G[:r + SUB], 80.0))       # (r+SUB, NH, D)
            Ks = kc[:r + SUB] * ek
            KK[:, rows, :r + SUB] = np.einsum("ihd,jhd->hij", kc[rows] * er, Ks)
            QK[:, rows, :r + SUB] = np.einsum("ihd,jhd->hij", qc[rows] * er, Ks)
        A = -(KK * bc.T[:, :, None]) * tri                        # A_ij = -beta_i k_i.k_j decay, i > j
        # T = (I - A)^-1 ... as the reference: forward substitution rows, then + I
        Tm = A.copy()
        for i in range(1, C):
            Tm[:, i, :i] = Tm[:, i, :i] + np.einsum("hj,hjm->hm", Tm[:, i, :i], Tm[:, :i, :i])
        Tm = Tm + np.eye(C, dtype=F)
        vb = vc * bc[:, :, None]
        kbg = kc * bc[:, :, None] * np.exp(G)                     # beta k exp(G_i) (<= 1)
        U = np.einsum("hij,jhd->ihd", Tm, vb)
        W = np.einsum("hij,jhd->ihd", Tm, kbg)
        vnew = U - np.einsum("ihk,hkv->ihv", W, S)
        oc = np.einsum("ihk,hkv->ihv", qc * np.exp(G), S) + np.einsum("hij,jhv->ihv", QK * lowq, vnew)
        o[sl] = oc
        Gl = G[-1]                                                # (NH, D)
        S = S * np.exp(Gl)[:, :, None] + np.einsum("jhk,jhv->hkv", kc * np.exp(Gl - G), vnew)
    return o[:T], S


# ------------------------------------------------------------------ mHC
def mhc(X, fn, base, scale, iters=20, eps=1e-6, rms_eps=1e-5):
    """X: (T, 4, H) streams. Returns post (T,4), comb (T,4,4), collapsed (T,H)."""
    T, NS, H = X.shape
    flat = X.reshape(T, NS * H).astype(np.float64)
    flat = flat / np.sqrt((flat ** 2).mean(-1, keepdims=True) + rms_eps)
    m = flat @ fn.astype(np.float64).T
    pre_w, post_w, comb_w = m[:, :NS], m[:, NS:2 * NS], m[:, 2 * NS:]
    pre = 1 / (1 + np.exp(-(pre_w * scale[0] + base[:NS]))) + eps
    post = 2 / (1 + np.exp(-(post_w * scale[1] + base[NS:2 * NS])))
    cl = comb_w.reshape(T, NS, NS) * scale[2] + base[2 * NS:].reshape(NS, NS)
    cl = np.exp(cl - cl.max(-1, keepdims=True)); comb = cl / cl.sum(-1, keepdims=True) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    coll = (pre[:, :, None] * X).sum(1)
    return post, comb, coll


def mhc_update(X, y, post, comb):
    """streams after the sub-layer: post (x) y + comb^T X"""
    return post[:, :, None] * y[:, None, :] + np.einsum("tij,tih->tjh", comb, X)


# ------------------------------------------------------------------ module models (weights from Ckpt)
def rms(x, w, eps=1e-5):
    x = x.astype(np.float64)
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def silu(x):
    return x / (1 + np.exp(-x))


def kda_module(h, ck, i, NH=64, D=128, lb=-5.0):
    """h: (T, H) the normed module input. Returns out (T, H) + the KDA kernel inputs."""
    L = f"model.language_model.layers.{i}.self_attn."
    W = lambda n: ck.f32(L + n).astype(np.float64)
    T = h.shape[0]
    x = np.concatenate([h @ W(f"{c}_proj.weight").T for c in "qkv"], 1)            # (T, 3 NH D)
    cw = np.concatenate([W(f"{c}_conv1d.weight")[:, 0] for c in "qkv"])            # (3 NH D, 4)
    xp = np.concatenate([np.zeros((3, x.shape[1])), x])
    conv = sum(xp[j:j + T] * cw[:, j] for j in range(4))                             # causal, window 4
    conv = silu(conv)
    q, k, v = (conv[:, n * NH * D:(n + 1) * NH * D].reshape(T, NH, D) for n in range(3))
    f = (h @ W("f_a_proj.weight").T) @ W("f_b_proj.weight").T + W("dt_bias")
    g = lb / (1 + np.exp(-(np.exp(W("A_log"))[None, :, None] * f.reshape(T, NH, D))))
    beta = 1 / (1 + np.exp(-(h @ W("b_proj.weight").T)))
    o, S = kda_recurrent(q, k, v, g, beta)
    gate = ((h @ W("g_a_proj.weight").T) @ W("g_b_proj.weight").T).reshape(T, NH, D)
    on = o / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-5) * W("o_norm.weight") / (1 + np.exp(-gate))
    return on.reshape(T, NH * D) @ W("o_proj.weight").T, dict(q=q, k=k, v=v, g=g, beta=beta, o=o, S=S)


def router(h, ck, i, K=8, scale=2.5):
    L = f"model.language_model.layers.{i}.mlp.gate."
    lg = h.astype(np.float64) @ ck.f32(L + "weight").astype(np.float64).T
    sc = 1 / (1 + np.exp(-lg))
    ch = sc + ck.f32(L + "e_score_correction_bias")
    ids = np.argsort(-ch, axis=1, kind="stable")[:, :K]
    w = np.take_along_axis(sc, ids, 1)
    return ids, w / (w.sum(1, keepdims=True) + 1e-20) * scale


def mlp(h, ck, pfx, limit=10.0):
    W = lambda n: ck.f32(pfx + n).astype(np.float64)
    g = np.minimum(h @ W("gate_proj.weight").T, limit)
    u = np.clip(h @ W("up_proj.weight").T, -limit, limit)
    return (silu(g) * u) @ W("down_proj.weight").T


def moe(h, ck, i):
    L = f"model.language_model.layers.{i}.mlp."
    ids, w = router(h, ck, i)
    y = mlp(h, ck, L + "shared_experts.")
    for e in np.unique(ids):
        t, sl = np.nonzero(ids == e)
        y[t] += mlp(h[t], ck, f"{L}experts.{e}.") * w[t, sl][:, None]
    return y, ids, w


def indexer(h, qr, ck, i, NHI=32, DI=128, pool=4, topk=2048):
    """h: (T, H) normed input, qr: (T, 1536) q_a_layernorm(q_a_proj(h)). Returns token indices (T, topk + 3),
    -1 padded, as Glm5NextTextIndexer (prefill of a fresh sequence, no padding)."""
    L = f"model.language_model.layers.{i}.self_attn.indexer."
    W = lambda n: ck.f32(L + n).astype(np.float64)
    T = h.shape[0]
    q = (qr @ W("wq_b.weight").T).reshape(T, NHI, DI)
    kr = h @ W("wk.weight").T
    mu, var = kr.mean(-1, keepdims=True), kr.var(-1, keepdims=True)
    k = (kr - mu) / np.sqrt(var + 1e-6) * W("k_norm.weight") + W("k_norm.bias")
    gs = h @ W("index_kpool_compress_gate").T                                     # (T, DI)
    NP = T // pool                                                                 # complete pools
    lg = gs[:NP * pool].reshape(NP, pool, DI) + W("index_kpool_compress_ape")
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    pk = (pr * k[:NP * pool].reshape(NP, pool, DI)).sum(1)                         # (NP, DI)
    sc = np.maximum(np.einsum("thd,pd->thp", q, pk) * DI ** -0.5, 0)
    wts = (h @ W("weights_proj.weight").T) * NHI ** -0.5
    isc = np.einsum("th,thp->tp", wts, sc)
    out = np.full((T, topk + pool - 1), -1, np.int64)
    for t in range(T):
        nvis = NP_t = (t + 1) // pool                                              # pools ending <= t
        sel = np.argsort(-isc[t, :NP_t], kind="stable")[:min(topk // pool, NP_t)] if NP_t else np.zeros(0, int)
        toks = (sel[:, None] * pool + np.arange(pool)).reshape(-1)
        tail = np.arange(NP_t * pool, t + 1)
        out[t, :len(toks)] = toks
        out[t, len(toks):len(toks) + len(tail)] = tail
    return out, isc


def mla(h, ck, i, NH=64, DN=256, DV=256, R=512):
    """MLA without rope over the indexer's selection. Returns out (T, H), idx."""
    L = f"model.language_model.layers.{i}.self_attn."
    W = lambda n: ck.f32(L + n).astype(np.float64)
    T = h.shape[0]
    qr = rms(h @ W("q_a_proj.weight").T, W("q_a_layernorm.weight"))
    q = (qr @ W("q_b_proj.weight").T).reshape(T, NH, DN)
    c = rms(h @ W("kv_a_proj_with_mqa.weight").T, W("kv_a_layernorm.weight"))       # (T, R) latent
    kv = (c @ W("kv_b_proj.weight").T).reshape(T, NH, DN + DV)
    kk, vv = kv[..., :DN], kv[..., DN:]
    idx, _ = indexer(h, qr, ck, i)
    m = np.full((T, T), -np.inf)
    for t in range(T):
        sel = idx[t][idx[t] >= 0]
        m[t, sel] = 0
    s = np.einsum("thd,uhd->htu", q, kk) * (DN ** -0.5) + m
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    o = np.einsum("htu,uhd->thd", p, vv).reshape(T, NH * DV)
    return o @ W("o_proj.weight").T, idx


# ------------------------------------------------------------------ checkpoint access (numpy only)
class Ckpt:
    def __init__(self, path=os.environ.get("CKPT", "GLM-5.3-Flash")):
        import json
        import os
        import struct
        self.path, self.idx = path, json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.h, self.m, self.os, self.json, self.struct = {}, {}, os, json, struct
        u = np.arange(256)
        sg = np.where(u & 0x80, -1.0, 1.0); e = (u >> 3) & 15; mm = u & 7
        self.lut = (sg * np.where(e == 0, mm / 8.0 * 2.0 ** -6, (1 + mm / 8.0) * 2.0 ** (e - 7.0))).astype(F)

    def raw(self, name):
        f = self.os.path.join(self.path, self.idx[name])
        if f not in self.h:
            with open(f, "rb") as fh:
                n = self.struct.unpack("<Q", fh.read(8))[0]
                self.h[f] = (self.json.loads(fh.read(n)), 8 + n)
            self.m[f] = np.memmap(f, np.uint8, "r")
        hd, base = self.h[f]
        v = hd[name]
        a, b = v["data_offsets"]
        dt = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F32": np.float32, "I64": np.int64}[v["dtype"]]
        return self.m[f][base + a:base + b].view(dt).reshape(v["shape"]), v["dtype"]

    def f32(self, name):
        t, dt = self.raw(name)
        if dt == "BF16":
            return (t.astype(np.uint32) << 16).view(F)
        if dt == "F8_E4M3":
            s = self.f32(name + "_scale_inv")
            w = self.lut[t]
            Mb, Kb = s.shape
            return (w.reshape(Mb, w.shape[0] // Mb, Kb, w.shape[1] // Kb) * s[:, None, :, None]).reshape(t.shape)
        return np.asarray(t, F)


# ------------------------------------------------------------------ checks against the oracle
def err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return f"max|d| {np.abs(a - b).max():.2e} (scale {np.abs(b).max():.2e})"


def check(path, ckpt=None):
    d = np.load(path)
    ck = Ckpt()
    L = "model.language_model.layers."
    for i in range(1, 64):
        if f"l{i}.hca_post" not in d.files or f"x{i - 1}" not in d.files:
            continue
        X = d[f"x{i - 1}"]
        T = X.shape[0]
        hp = lambda tag: (ck.f32(f"{L}{i}.hc_{tag}_fn"), ck.f32(f"{L}{i}.hc_{tag}_base"), ck.f32(f"{L}{i}.hc_{tag}_scale"))
        post, comb, coll = mhc(X, *hp("attn"))
        Xm = mhc_update(X, d[f"l{i}.attn_out"], post, comb)
        post2, comb2, coll2 = mhc(Xm, *hp("ffn"))
        Xo = mhc_update(Xm, d[f"l{i}.mlp_out"], post2, comb2)
        hn = rms(d[f"l{i}.hca_in"], ck.f32(f"{L}{i}.input_layernorm.weight"))
        if f"l{i}.kda_q" in d.files:
            out, kin = kda_module(hn, ck, i)
            print(f"layer {i} KDA module: out {err(out, d[f'l{i}.attn_out'])} q {err(kin['q'].reshape(-1, 128), d[f'l{i}.kda_q'])} "
                  f"g {err(kin['g'].reshape(-1, 128), d[f'l{i}.kda_g'])} beta {err(kin['beta'], d[f'l{i}.kda_beta'])}")
        elif f"l{i}.idx" in d.files:
            out, idx = mla(hn, ck, i)
            ref_idx = d[f"l{i}.idx"].astype(np.int64)
            same = sum(set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()) for a, b in zip(idx, ref_idx))
            print(f"layer {i} MLA+indexer: out {err(out, d[f'l{i}.attn_out'])} selections equal {same}/{len(idx)}")
        hm = rms(d[f"l{i}.hcf_in"], ck.f32(f"{L}{i}.post_attention_layernorm.weight"))
        if f"l{i}.router_ids" in d.files:
            y, ids, w = moe(hm, ck, i)
            ri = d[f"l{i}.router_ids"].astype(np.int64)
            same = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(ids, ri))
            print(f"layer {i} MoE: out {err(y, d[f'l{i}.mlp_out'])} routing equal {same}/{len(ids)}")
        else:
            print(f"layer {i} MLP: out {err(mlp(hm, ck, f'{L}{i}.mlp.'), d[f'l{i}.mlp_out'])}")
        print(f"layer {i} mHC: post {err(post, d[f'l{i}.hca_post'])} comb {err(comb, d[f'l{i}.hca_comb'].reshape(T, 4, 4))} "
              f"in {err(coll, d[f'l{i}.hca_in'])} | ffn site post {err(post2, d[f'l{i}.hcf_post'])} in {err(coll2, d[f'l{i}.hcf_in'])} "
              f"| streams out {err(Xo, d[f'x{i}'])}")
    for i in range(64):
        if f"l{i}.kda_q" not in d.files:
            continue
        NH, D = 64, 128
        q, k, v, g = (d[f"l{i}.kda_{n}"].reshape(-1, NH, D) for n in "qkvg")
        beta = d[f"l{i}.kda_beta"]
        o_ref, S_ref = d[f"l{i}.kda_o"].reshape(-1, NH, D), d[f"l{i}.kda_state"].reshape(NH, D, D)
        o_r, S_r = kda_recurrent(q, k, v, g, beta)
        o_c, S_c = kda_blocks(q, k, v, g, beta)
        print(f"layer {i} KDA (T={q.shape[0]}): recurrent vs oracle o {err(o_r, o_ref)} S {err(S_r, S_ref)} | "
              f"chunked(f32) vs oracle o {err(o_c, o_ref)} S {err(S_c, S_ref)}")


if __name__ == "__main__":
    check(sys.argv[1])
