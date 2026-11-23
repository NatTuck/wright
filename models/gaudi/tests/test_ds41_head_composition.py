"""Head composition oracle (DeepSeek-V4.1-Flash, one rank): the last ffn's residual update, the collapse
with its pre-mix (mhc_pre mode 0), the final RMSNorm and the bf16 lm_head GEMM -> logits, vs numpy.
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
    print(f"  {name:10s} max_abs={np.abs(got - want).max():.4e} norm_rel={rel:.4e}")
    return rel


@pytest.mark.parametrize("T", [1, 4])
def test_head_matches_numpy(device, recipes, T):
    from test_ds41_kernels import Run, bf16_round

    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.loading.convert import bf16_rne, kperm

    rng = np.random.default_rng(99 + T)
    H, V, eps = 5120, 256, 1e-20
    X = (rng.standard_normal((T, 4 * H)) * 0.5).astype(np.float32)
    p = (rng.standard_normal((T, H)) * 0.3).astype(np.float32)
    c = np.zeros((24, T), np.float32)
    pre = rng.uniform(0.2, 0.8, (4, T))
    pre = pre / pre.sum(axis=0, keepdims=True)
    c[:4] = pre
    c[4:8] = rng.uniform(0.2, 1.5, (4, T))
    cm = rng.uniform(0.1, 1.0, (T, 4, 4))
    cm = cm / cm.sum(axis=2, keepdims=True)
    c[8:24] = cm.reshape(T, 16).T
    norm = (1.0 + 0.02 * rng.standard_normal(H)).astype(np.float32)
    head = bf16_rne((rng.standard_normal((V, H)) * 0.02).astype(np.float32))
    head_f32 = (head.astype(np.uint32) << 16).view(np.float32)

    d = L.Dims(hidden=H, heads=8, head_dim=512, q_lora=1280, o_lora=1024, groups=1, expert=576,
               experts=64, top=6, vocab=V, eps=eps, hc_iters=20, hc_eps=1e-6, limit=10.0,
               route_scale=1.5, ranks=1)
    g = L.head_graph(device, d, T)
    dev = {
        "X": X, "p": p.reshape(1, T, H), "c": c, "norm": norm, "head": kperm(head),
        "X_out": np.zeros((T, 4 * H), np.float32),
    }
    dev = {k: v for k, v in dev.items() if k in g.tensors}
    got = Run(device, recipes, g)(dev, ["logits", "X_out"])
    g.destroy()

    # numpy: hc_post -> collapse(pre) -> RMSNorm(norm) -> bf16 -> head GEMM
    Xo = np.zeros((T, 4 * H), np.float64)
    post = c[4:8].T
    comb = c[8:24].T.reshape(T, 4, 4)
    for t in range(T):
        for j in range(4):
            acc = post[t, j] * p[t].astype(np.float64)
            for s in range(4):
                acc = acc + comb[t, s, j] * X[t, s * H : (s + 1) * H].astype(np.float64)
            Xo[t, j * H : (j + 1) * H] = acc
    cv = np.zeros((T, H), np.float64)
    for t in range(T):
        for s in range(4):
            cv[t] += pre[s, t] * Xo[t, s * H : (s + 1) * H]
    r = 1.0 / np.sqrt((cv * cv).mean(axis=1, keepdims=True) + eps)
    h_norm = cv * r * norm
    h16 = bf16_round(h_norm)
    logits = h16 @ head_f32.T
    print(f"\n=== head T={T} ===")
    _report("X_out", got["X_out"], Xo)
    rel = _report("logits", got["logits"], logits)
    assert rel < 1e-3, f"head diverges: norm_rel {rel:.3e}"