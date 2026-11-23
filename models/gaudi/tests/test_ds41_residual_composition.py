"""One-layer hyper-connection residual + site composition oracle (DeepSeek-V4.1-Flash, one rank).

The collapse (mhc_pre8_p2) is verified; the POST sub-layer residual update (mhc_res / mhc_post) and the
site coefficient projection (mhc_mix* + mhc_coef*) are not. This feeds a known stream X, incoming
coefficients c and partial p, runs residual() + site(), and compares X_out and c_out against the
checkpoint's own inference/model.py math (hc_post, hc_mixes, hc_split_sinkhorn).
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
    ad = np.abs(got - want)
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    print(f"  {name:12s} max_abs={ad.max():.4e} norm_rel={rel:.4e}")
    return rel


def _sinkhorn(mix, scale, base, iters, eps):
    """The reference hc_split_sinkhorn (inference/model.py): pre | post | comb (4x4, row-major)."""
    pre = 1.0 / (1.0 + np.exp(-(mix[:4] * scale[0] + base[:4]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mix[4:8] * scale[1] + base[4:8])))
    comb = (mix[8:24] * scale[2] + base[8:24]).reshape(4, 4)
    comb = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = comb / comb.sum(axis=1, keepdims=True) + eps
    comb = comb / (comb.sum(axis=0, keepdims=True) + eps)  # columns first
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + eps)  # rows
        comb = comb / (comb.sum(axis=0, keepdims=True) + eps)  # columns
    return pre, post, comb


@pytest.mark.parametrize("T", [1, 4, 8, 16, 32, 64])
def test_residual_and_site_match_reference(device, recipes, T):
    from test_ds41_kernels import Run

    from resin_models_gaudi.deepseek_v41 import layers as L

    rng = np.random.default_rng(4242 + T)
    H, eps, hc_eps, iters = 5120, 1e-20, 1e-6, 20
    X = (rng.standard_normal((T, 4 * H)) * 0.5).astype(np.float32)
    p = (rng.standard_normal((T, H)) * 0.3).astype(np.float32)
    # incoming coefficients c (T,24): pre | post | comb
    c = np.zeros((24, T), np.float32)
    c[:4] = rng.uniform(0.2, 0.8, (4, T))  # pre
    c[4:8] = rng.uniform(0.2, 1.5, (4, T))  # post
    cm = rng.uniform(0.1, 1.0, (T, 4, 4))
    cm = cm / cm.sum(axis=2, keepdims=True)  # row-stochastic-ish
    c[8:24] = cm.reshape(T, 16).T
    fn = (rng.standard_normal((24, 4 * H)) * 0.02).astype(np.float32)
    fn_pad = np.zeros((32, 4 * H), np.float32)
    fn_pad[:24] = fn
    base = (rng.standard_normal(24) * 0.5).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, 3).astype(np.float32)

    d = L.Dims(hidden=H, heads=8, head_dim=512, q_lora=1280, o_lora=1024, groups=1, expert=576,
               experts=64, top=6, vocab=16160, eps=eps, hc_iters=iters, hc_eps=hc_eps, limit=10.0,
               route_scale=1.5, ranks=1)
    lg = L.LayerGraph(device, f"t_res_site_T{T}", d, T)
    Xo = lg.residual()
    lg.site(Xo, "hca")
    inputs = {
        "X": X, "p": p.reshape(1, T, H), "c": c,
        "hca_fn": fn_pad, "hca_base": base, "hca_scale": scale,
    }
    inputs = {k: v for k, v in inputs.items() if k in lg.g.tensors}
    got = Run(device, recipes, lg.g)(inputs, ["X_out", "c_out"])
    lg.g.destroy()
    X_out = got["X_out"]  # (T,4H)
    c_out = got["c_out"].reshape(24, T)  # T=1 site returns (24,), T>1 returns (24,T)

    # numpy hc_post: X_out_j = post_j*p + sum_s comb[4*s+j]*X_s
    want_X = np.zeros((T, 4 * H), np.float64)
    post = c[4:8].T  # (T,4)
    comb = c[8:24].T.reshape(T, 4, 4)  # comb[t,s,j] = c[8+4*s+j,t]
    for t in range(T):
        for j in range(4):
            acc = post[t, j] * p[t].astype(np.float64)
            for s in range(4):
                acc = acc + comb[t, s, j] * X[t, s * H : (s + 1) * H].astype(np.float64)
            want_X[t, j * H : (j + 1) * H] = acc
    print(f"\n=== T={T} residual + site ===")
    rel_x = _report("X_out", X_out, want_X)

    # numpy hc_mixes + hc_split_sinkhorn on X_out
    want_c = np.zeros((24, T), np.float64)
    xf = want_X  # flattened streams (T,4H)
    r = 1.0 / np.sqrt((xf * xf).mean(axis=1, keepdims=True) + eps)
    mix = (xf @ fn.T) * r  # (T,24)
    for t in range(T):
        pre, post_, comb_ = _sinkhorn(mix[t], scale, base, iters, hc_eps)
        want_c[:4, t] = pre
        want_c[4:8, t] = post_
        want_c[8:24, t] = comb_.reshape(16)
    rel_pre = _report("c_out pre", c_out[:4], want_c[:4])
    rel_post = _report("c_out post", c_out[4:8], want_c[4:8])
    rel_comb = _report("c_out comb", c_out[8:24], want_c[8:24])
    assert rel_x < 1e-4, f"residual update diverges: {rel_x:.3e}"
    assert rel_pre < 1e-3 and rel_post < 1e-3 and rel_comb < 1e-3, (
        f"site coefficients diverge: pre {rel_pre:.3e} post {rel_post:.3e} comb {rel_comb:.3e}"
    )