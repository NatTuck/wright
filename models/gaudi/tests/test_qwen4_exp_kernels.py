"""Qwen3.8-Flash-Next kernels on one card against numpy: hyper connections (hc_combine in its three modes, hc_lo,
hc_mix), the router, the routed experts (per-token, work-list and paged paths over block-FP8 experts packed by
``pack_experts``, some routes on another rank), the PLE (hash, gather conversion, gate, conv), and the QSA indexer
(block keys, queries, selection, prefill masks)."""

from __future__ import annotations

import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

H, NS, EPS, HCR = 2560, 4, 1e-6, 384
E_ALL, K, F = 512, 10, 640
GEMM_NN = struct.pack("??", False, False)


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


class Run:
    """Compile a graph, bind numpy inputs and output buffers by name, launch once."""

    def __init__(self, device, recipes, graph) -> None:
        self.device, self.graph = device, graph
        self.recipe = recipes.compile(graph)

    def __call__(self, inputs: dict[str, np.ndarray], outputs: list[str], stale=None):
        from resin_models_gaudi.synapse.runtime import NUMPY_DTYPE

        d, allocated = self.device, []
        try:
            for name, array in inputs.items():
                a = np.ascontiguousarray(array)
                address = d.malloc(max(a.nbytes, 256))
                allocated.append(address)
                d.h2d(address, a)
                self.recipe.bind(name, address)
            outs = {}
            for name in outputs:
                t = self.recipe.persistent[name]
                address = d.malloc(max(t.nbytes, 256))
                allocated.append(address)
                if stale and name in stale:
                    d.h2d(address, np.ascontiguousarray(stale[name]))
                self.recipe.bind(name, address)
                outs[name] = (address, t)
            self.recipe.launch()
            d.sync()
            return {
                name: d.d2h(a, tuple(reversed(t.shape)), NUMPY_DTYPE[t.dtype])
                for name, (a, t) in outs.items()
            }
        finally:
            for address in allocated:
                d.free(address)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def sig(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sig(x)


def rms_s(x, w, T):
    """per-stream RMSNorm of (T, 4H)"""
    xs = x.reshape(T, NS, H)
    return (xs / np.sqrt((xs * xs).mean(-1, keepdims=True) + EPS)).reshape(
        T, NS * H
    ) * w


def unsig(bits):
    """(T, C) bf16 bits in sigma order -> natural f32"""
    from resin_models_gaudi.loading.convert import bf16_to_f32, sigma_perm

    out = np.empty(bits.shape, np.float32)
    out[:, sigma_perm(bits.shape[1])] = bf16_to_f32(bits)
    return out


def sig_pack(a):
    """(T, C) f32 -> bf16 bits in sigma order"""
    from resin_models_gaudi.loading.convert import bf16_rne, sigma_perm

    return bf16_rne(np.ascontiguousarray(a[:, sigma_perm(a.shape[1])]))


# ------------------------------------------------------------------ hyper connections
@pytest.mark.parametrize("kernel", ["rg_hc_combine_f32", "rg_hc_combine_t_f32"])
@pytest.mark.parametrize("mode", [1, 2, 0])
@pytest.mark.parametrize("T", [3, 128])
def test_hc_combine(device, recipes, kernel, mode, T):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, NS * H)).astype(np.float32)
    y = rng.standard_normal((T, H)).astype(np.float32)
    lo = rng.standard_normal((T, HCR)).astype(np.float32)
    w = (1 + 0.1 * rng.standard_normal(NS * H)).astype(np.float32)
    g = Graph(device, f"t_hcc{mode}_{kernel[3:7]}_{T}")
    ins = [
        g.input("x", (NS * H, T), F32),
        g.input("y", (H, T, 1), F32),
        g.input("lo", (HCR, T), F32),
        g.input("w", (NS * H,), F32),
    ]
    outs = [
        g.input("xo", (NS * H, T), F32),
        g.input("xn", (NS * H, T), BF16),
        g.input("r", (64, NS, T), F32),
    ]
    g.node(kernel, ins, outs, struct.pack("fi", EPS, mode))
    res = Run(device, recipes, g)({"x": x, "y": y, "lo": lo, "w": w}, ["xo", "xn", "r"])
    a = 2 * sig(lo[:, 320:324] / 4)
    if mode == 1:
        ref = x + (y[:, None, :] * a[:, :, None]).reshape(T, -1)
    elif mode == 2:
        ref = np.tile(y, (1, NS))
    else:
        ref = x
    rr = 1 / np.sqrt((ref.reshape(T, NS, H) ** 2).mean(-1) + EPS)
    assert rel(res["xo"], ref) < 1e-6
    assert rel(unsig(res["xn"]), rms_s(ref, w, T)) < 1e-2
    assert rel(res["r"][:, :, 0], rr) < 1e-5


def test_hc_combine_sums_rank_slices(device, recipes):
    """y (H, T, NR): the all-gathered partials of NR ranks are summed in the kernel"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    rng = np.random.default_rng(3)
    T, NR = 2, 4
    x = rng.standard_normal((T, NS * H)).astype(np.float32)
    y = rng.standard_normal((NR, T, H)).astype(np.float32)
    lo = rng.standard_normal((T, HCR)).astype(np.float32)
    w = np.ones(NS * H, np.float32)
    g = Graph(device, "t_hcc_nr")
    g.node(
        "rg_hc_combine_f32",
        [
            g.input("x", (NS * H, T), F32),
            g.input("y", (H, T, NR), F32),
            g.input("lo", (HCR, T), F32),
            g.input("w", (NS * H,), F32),
        ],
        [
            g.input("xo", (NS * H, T), F32),
            g.input("xn", (NS * H, T), BF16),
            g.input("r", (64, NS, T), F32),
        ],
        struct.pack("fi", EPS, 1),
    )
    res = Run(device, recipes, g)({"x": x, "y": y, "lo": lo, "w": w}, ["xo", "xn", "r"])
    a = 2 * sig(lo[:, 320:324] / 4)
    ref = x + (y.sum(0)[:, None, :] * a[:, :, None]).reshape(T, -1)
    assert rel(res["xo"], ref) < 1e-6


def test_hc_lo_and_mix(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32

    rng = np.random.default_rng(1)
    T = 3
    lo = rng.standard_normal((T, HCR)).astype(np.float32)
    g = Graph(device, "t_hclo")
    g.node(
        "rg_hc_lo_f32", [g.input("lo", (HCR, T), F32)], [g.input("o", (HCR, T), BF16)]
    )
    res = Run(device, recipes, g)({"lo": lo}, ["o"])
    assert rel(unsig(res["o"]), silu(lo / 4)) < 1e-2
    x = rng.standard_normal((T, NS * H)).astype(np.float32)
    u = rng.standard_normal((T, NS * H)).astype(np.float32)
    w = (1 + 0.1 * rng.standard_normal(NS * H)).astype(np.float32)
    g = Graph(device, "t_hcmix")
    g.node(
        "rg_hc_mix_f32",
        [
            g.input("x", (NS * H, T), F32),
            g.input("w", (NS * H,), F32),
            g.input("u", (NS * H, T), F32),
            g.input("r", (64, NS, T), F32),
        ],
        [g.input("m16", (H, T), BF16), g.input("m32", (H, T), F32)],
    )
    rin = np.repeat(
        (1 / np.sqrt((x.reshape(T, NS, H) ** 2).mean(-1) + EPS))[:, :, None], 64, 2
    ).astype(np.float32)
    res = Run(device, recipes, g)({"x": x, "w": w, "u": u, "r": rin}, ["m16", "m32"])
    ref = (sig(u) * rms_s(x, w, T)).reshape(T, NS, H).mean(1)
    assert rel(res["m32"], ref) < 1e-5
    assert rel(unsig(res["m16"]), ref) < 1e-2


# ------------------------------------------------------------------ MoE
def test_moe_route_top_k_softmax(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(2)
    T = 3
    lg = rng.standard_normal((T, E_ALL + 1)).astype(np.float32)
    lg[0, 7] = lg[0, 300] = lg[0].max() + 1  # a tie: lowest index first
    g = Graph(device, "t_route")
    g.node(
        "rg_moe_route_f32",
        [g.input("lg", (E_ALL + 1, T), F32), g.input("meta", (4,), I32)],
        [g.input("ids", (64, T), I32), g.input("wts", (64, T), F32)],
        struct.pack("i", K),
    )
    res = Run(device, recipes, g)(
        {"lg": lg, "meta": np.array([0, T, 0, 0], np.int32)}, ["ids", "wts"]
    )
    rid = np.argsort(-lg[:, :E_ALL], axis=1, kind="stable")[:, :K]
    rl = np.take_along_axis(lg, rid, 1)
    rw = np.exp(rl - rl[:, :1])
    rw /= rw.sum(1, keepdims=True)
    assert np.array_equal(res["ids"][:, :K], rid)
    assert not res["ids"][:, K:].any() and not res["wts"][:, K:].any()
    assert rel(res["wts"][:, :K], rw) < 1e-5


class Experts:
    """E local experts (global ids LO .. LO + E - 1) in block FP8, their dense values, routing of T tokens."""

    def __init__(self, T: int, E: int = 6, LO: int = 2, seed: int = 4) -> None:
        from resin_models_gaudi.loading.convert import E4M3, e4m3_encode
        from resin_models_gaudi.qwen4_exp.checkpoint import pack_experts

        rng = np.random.default_rng(seed)
        self.T, self.E, self.LO = T, E, LO

        def q8(shape):
            return e4m3_encode(
                np.clip(rng.standard_normal(shape).astype(np.float32) * 40, -240, 240)
            )

        gate, up, down = q8((E, F, H)), q8((E, F, H)), q8((E, H, F))
        gs = (rng.random((E, F // 128, H // 128)) * 1e-3 + 1e-4).astype(np.float32)
        us = (rng.random((E, F // 128, H // 128)) * 1e-3 + 1e-4).astype(np.float32)
        ds = (rng.random((E, H // 128, F // 128)) * 1e-3 + 1e-4).astype(np.float32)
        self.W12, self.S12, self.W3, self.S3 = pack_experts(gate, gs, up, us, down, ds)

        def deq(b, s):
            v = E4M3[b]
            M, N = v.shape[1:]
            return (
                v.reshape(E, M // 128, 128, N // 128, 128) * s[:, :, None, :, None]
            ).reshape(E, M, N)

        self.Gf, self.Uf, self.Df = deq(gate, gs), deq(up, us), deq(down, ds)
        self.ids = np.zeros((T, 64), np.int32)
        for t in range(T):
            self.ids[t, :K] = rng.choice(
                np.arange(E + 6), K, replace=False
            )  # some routes not local
        self.wts = np.zeros((T, 64), np.float32)
        self.wts[:, :K] = rng.random((T, K)).astype(np.float32)
        self.x = rng.standard_normal((T, H)).astype(np.float32)
        self.sh = rng.standard_normal((T, H)).astype(np.float32)
        self.lg = rng.standard_normal((T, E_ALL + 1)).astype(np.float32)

    def reference(self) -> np.ndarray:
        ref = sig(self.lg[:, E_ALL : E_ALL + 1]) * self.sh
        for t in range(self.T):
            for k in range(K):
                e = self.ids[t, k] - self.LO
                if 0 <= e < self.E:
                    hk = silu(self.Gf[e] @ self.x[t]) * (self.Uf[e] @ self.x[t])
                    ref[t] += self.wts[t, k] * (self.Df[e] @ hk)
        return ref

    def local_pairs(self) -> int:
        return int(
            ((self.ids[:, :K] >= self.LO) & (self.ids[:, :K] < self.LO + self.E)).sum()
        )

    def inputs(self) -> dict:
        return {
            "x": self.x,
            "W12": self.W12,
            "S12": self.S12,
            "ids": self.ids,
            "W3": self.W3,
            "S3": self.S3,
            "wts": self.wts,
            "sh": self.sh,
            "lg": self.lg,
        }

    def declare(self, g):
        from resin_models_gaudi.synapse.runtime import F8, F32, I32

        E, NB, NO = self.E, F // 128, H // 256
        return dict(
            x=g.input("x", (H, self.T), F32),
            W12=g.input("W12", (256, 128, NB, H // 128, E), F8),
            S12=g.input("S12", (64, 2, NB, H // 128, E), F32),
            ids=g.input("ids", (64, self.T), I32),
            W3=g.input("W3", (256, 128, NO, F // 128, E), F8),
            S3=g.input("S3", (64, 2, NO, F // 128, E), F32),
            wts=g.input("wts", (64, self.T), F32),
            sh=g.input("sh", (H, self.T), F32),
            lg=g.input("lg", (E_ALL + 1, self.T), F32),
        )


@pytest.mark.parametrize("T,real", [(3, 3), (128, 13)])
def test_moe_per_token_kernels(device, recipes, T, real):
    """``real`` < T: the padding columns' routes are -1 (as moe_route makes them) and add nothing"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    ex = Experts(T=T)
    ex.ids[real:] = -1
    ex.wts[real:] = 0
    NB = F // 128
    g = Graph(device, f"t_moe_tok_{T}")
    i = ex.declare(g)
    hh = g.tensor((128, NB, K, ex.T), F32)
    g.node(
        "rg_moe_up_f32",
        [i["x"], i["W12"], i["S12"], i["ids"]],
        [hh],
        struct.pack("ii", ex.LO, K),
    )
    g.node(
        "rg_moe_down_f32",
        [hh, i["W3"], i["S3"], i["ids"], i["wts"], i["sh"], i["lg"]],
        [g.input("y", (H, ex.T), F32)],
        struct.pack("iiii", ex.LO, K, 1, E_ALL),
    )
    res = Run(device, recipes, g)(ex.inputs(), ["y"])
    assert ex.local_pairs() > 0
    assert rel(res["y"][:real], ex.reference()[:real]) < 2e-2


def test_moe_work_list_kernels(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32

    ex = Experts(T=3, seed=5)
    NB, NI = F // 128, H // 128
    g = Graph(device, "t_moe_w")
    i = ex.declare(g)
    hp = g.tensor((256, NB, NI, K, ex.T), F32)
    g.node(
        "rg_moe_up_w_f32",
        [i["x"], i["W12"], i["S12"], i["ids"]],
        [hp],
        struct.pack("ii", ex.LO, K),
    )
    yk = g.tensor((H, NB * K, ex.T), F32)
    g.node(
        "rg_moe_down_w_f32",
        [hp, i["W3"], i["S3"], i["ids"], i["wts"]],
        [yk],
        struct.pack("ii", ex.LO, K),
    )
    g.node(
        "rg_moe_sum_w_f32",
        [yk, i["sh"], i["lg"], i["ids"]],
        [g.input("y", (H, ex.T), F32)],
        struct.pack("iiiii", 1, E_ALL, ex.LO, ex.E, K),
    )
    res = Run(device, recipes, g)(ex.inputs(), ["y"])
    assert rel(res["y"], ex.reference()) < 2e-2


def test_moe_paged_prefill_path(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, F32, I32

    ex = Experts(T=7, seed=6)
    T, E, LO = ex.T, ex.E, ex.LO
    NB = F // 128
    PG, PSH = 8, 6
    PS = 1 << PSH
    g = Graph(device, "t_moe_pg")
    i = ex.declare(g)
    meta = g.input("meta", (4,), I32)
    cntm = g.tensor((E, 24), I32)
    g.node("rg_moe_cnt_f32", [i["ids"], meta], [cntm], struct.pack("ii", LO, K))
    po, pe = g.tensor((E,), I32), g.tensor((PG,), I32)
    g.node("rg_moe_pages_f32", [cntm], [po, pe], struct.pack("i", PSH))
    gidx, bm = g.tensor((PS * PG,), I32), g.tensor((64, T), I32)
    g.node(
        "rg_moe_asg_f32",
        [i["ids"], meta, cntm, po, pe],
        [gidx, bm],
        struct.pack("iii", LO, K, PSH),
    )
    wb12, wb3 = g.tensor((2 * F, H, PG), BF16), g.tensor((H, F, PG), BF16)
    g.node(
        "rg_moe_deq_p_f32", [i["W12"], i["S12"], pe], [wb12], struct.pack("ii", 128, F)
    )
    g.node(
        "rg_moe_deq_p_f32", [i["W3"], i["S3"], pe], [wb3], struct.pack("ii", 256, 128)
    )
    X2 = g.tensor((H, PS * PG), BF16)
    g.node(
        "gather_fwd_bf16",
        [g.input("xpad", (H, T + 1), BF16), gidx],
        [X2],
        struct.pack("i", 1),
    )
    X3 = g.tensor((H, PS, PG), BF16)
    g.node("reshape", [X2], [X3])
    y1 = g.tensor((2 * F, PS, PG), BF16)
    g.node("batch_gemm", [X3, wb12], [y1], GEMM_NN)
    y1f = g.tensor((2 * F, PS * PG), BF16)
    g.node("reshape", [y1], [y1f])
    hh2 = g.tensor((F, PS * PG), BF16)
    g.node("rg_moe_act_g_f32", [y1f, gidx, meta], [hh2])
    hh3 = g.tensor((F, PS, PG), BF16)
    g.node("reshape", [hh2], [hh3])
    y2 = g.tensor((H, PS, PG), BF16)
    g.node("batch_gemm", [hh3, wb3], [y2], GEMM_NN)
    ido = g.tensor((64, T), I32)
    g.node("rg_moe_ovf_f32", [i["ids"], bm, meta], [ido], struct.pack("iii", LO, E, K))
    hov = g.tensor((128, NB, K, T), F32)
    g.node(
        "rg_moe_up_f32",
        [i["x"], i["W12"], i["S12"], ido],
        [hov],
        struct.pack("ii", LO, K),
    )
    yo = g.tensor((H, T), F32)
    g.node(
        "rg_moe_down_f32",
        [hov, i["W3"], i["S3"], ido, i["wts"], i["sh"], i["lg"]],
        [yo],
        struct.pack("iiii", LO, K, 1, E_ALL),
    )
    g.node(
        "rg_moe_combine_p_f32",
        [y2, yo, i["ids"], bm, i["wts"], meta],
        [g.input("y", (H, T), F32)],
        struct.pack("iii", LO, E, K),
    )
    xpad = np.concatenate([sig_pack(ex.x), np.zeros((1, H), np.uint16)], 0)
    res = Run(device, recipes, g)(
        {**ex.inputs(), "xpad": xpad, "meta": np.array([0, T, 0, 0], np.int32)}, ["y"]
    )
    assert rel(res["y"], ex.reference()) < 3e-2


# ------------------------------------------------------------------ PLE
def test_ple_gate_and_conv(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(7)
    T = 3
    NR = T - 1  # real tokens of the chunk
    x = rng.standard_normal((T, NS * H)).astype(np.float32)
    kv = rng.standard_normal((T, 5 * H)).astype(np.float32)
    nk, nq, nc = [
        (1 + 0.1 * rng.standard_normal(NS * H)).astype(np.float32) for _ in range(3)
    ]
    cw = rng.standard_normal((4, NS * H)).astype(np.float32) * 0.5
    st = rng.standard_normal((9, NS * H)).astype(np.float32)
    g = Graph(device, "t_ple")
    xi = g.input("x", (NS * H, T), F32)
    gv, gvn = g.tensor((NS * H, T), F32), g.tensor((NS * H, T), F32)
    g.node(
        "rg_ple_gate_f32",
        [
            xi,
            g.input("kv", (5 * H, T), F32),
            g.input("nk", (NS * H,), F32),
            g.input("nq", (NS * H,), F32),
            g.input("nc", (NS * H,), F32),
        ],
        [gv, gvn],
        struct.pack("f", EPS),
    )
    g.node(
        "rg_ple_conv_f32",
        [
            xi,
            gv,
            gvn,
            g.input("cw", (NS * H, 4), F32),
            g.input("st", (NS * H, 9), F32),
            g.input("meta", (4,), I32),
        ],
        [g.input("xo", (NS * H, T), F32), g.input("sto", (NS * H, 9), F32)],
    )
    res = Run(device, recipes, g)(
        {
            "x": x,
            "kv": kv,
            "nk": nk,
            "nq": nq,
            "nc": nc,
            "cw": cw,
            "st": st,
            "meta": np.array([0, NR, 0, 0], np.int32),
        },
        ["xo", "sto"],
    )
    key, val = kv[:, : NS * H], kv[:, NS * H :]
    gd = (rms_s(key, nk, T) * rms_s(x, nq, T)).reshape(T, NS, H).sum(-1) / np.sqrt(H)
    gd = np.sign(gd) * np.sqrt(np.maximum(np.abs(gd), 1e-6))
    gvr = (sig(gd)[:, :, None] * val[:, None, :]).reshape(T, NS * H)
    gvnr = rms_s(gvr, nc, T)
    hist = np.concatenate([st, gvnr[:NR]], 0)
    conv = sum(cw[k] * hist[9 - 3 * (3 - k) + np.arange(NR)] for k in range(4))
    xr = x + gvr
    xr[:NR] += silu(conv)
    assert rel(res["xo"], xr) < 1e-4
    assert rel(res["sto"], hist[-9:]) < 1e-6


def test_ple_hash_matches_host(device, recipes):
    """ple_hash: the device rows of a token after a history equal NGram.rows, local rows or the zero row"""
    from resin_models_gaudi.qwen4_exp.ngram import NGram
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import I32

    V, EOS = 248320, 248044
    ng = NGram(V, 3, 8, 20_000_000, EOS)
    lo, n_local = 40_000_000, 100_000_000  # this rank's rows [lo, lo + n_local)
    A, HP = ng.device_tables(V, lo)
    g = Graph(device, "t_ple_hash")
    g.node(
        "rg_ple_hash_f32",
        [
            g.input("tok", (1,), I32),
            g.input("hin", (64,), I32),
            g.input("A", (8, V), I32),
            g.input("HP", (64, 4), I32),
        ],
        [g.input("rows", (ng.heads,), I32), g.input("hout", (64,), I32)],
        struct.pack("ii", EOS, n_local),
    )
    run = Run(device, recipes, g)
    for tok, hist in (
        (1234, [EOS, EOS]),
        (5, [7, 9]),
        (248000, [EOS, 3]),
        (17, [EOS, EOS]),
    ):
        hin = np.zeros(64, np.int32)
        hin[0], hin[1] = hist[1], hist[0]  # [t-1, t-2]
        res = run(
            {"tok": np.array([tok], np.int32), "hin": hin, "A": A, "HP": HP},
            ["rows", "hout"],
        )
        want = ng.rows([tok], advance=False, history=hist)[0] - lo
        want = np.where((want >= 0) & (want < n_local), want, n_local)
        assert np.array_equal(res["rows"], want), (tok, hist)
        assert res["hout"][0] == tok and res["hout"][1] == hist[1]


def test_ple_cvt_layout(device, recipes):
    from resin_models_gaudi.loading.convert import E4M3, HALVE, bf16_to_f32
    from resin_models_gaudi.qwen4_exp.checkpoint import ple_cvt_source
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import BF16, U8

    rng = np.random.default_rng(8)
    T, NH, D = 2, 16, 160
    # OCP e4m3 values within Gaudi's range, fed to the kernel in Gaudi's encoding (HALVE)
    ocp = rng.integers(0, 256, (NH * T, D), dtype=np.uint8)
    ocp = np.where(np.isnan(E4M3[ocp]) | (np.abs(E4M3[ocp]) > 240), 1, ocp).astype(
        np.uint8
    )
    e = HALVE[ocp]
    g = Graph(device, "t_ple_cvt")
    g.node(
        "rg_ple_cvt_f32",
        [g.input("e", (D, NH * T), U8)],
        [g.input("o", (256 * NH, T), BF16)],
    )
    res = Run(device, recipes, g)({"e": e}, ["o"])
    src = ple_cvt_source(D)
    want = np.zeros((T, 256 * NH), np.float32)
    for t in range(T):
        for h in range(NH):
            ok = src >= 0
            # Gaudi's encoding holds half the OCP value (the table's scale carries the 2)
            want[t, 256 * h + np.nonzero(ok)[0]] = E4M3[ocp[NH * t + h, src[ok]]] / 2
    assert rel(bf16_to_f32(res["o"]), want) < 1e-2


# ------------------------------------------------------------------ QSA indexer
def rope_tab(n, base=1e7, rd=64):
    half = rd // 2
    inv = base ** (-np.arange(half) * 2.0 / rd)
    ang = np.arange(n)[:, None] * inv[None]
    return np.concatenate(
        [np.cos(ang), np.cos(ang), -np.sin(ang), np.sin(ang)], 1
    ).astype(np.float32)


def rope(x, row):
    y = x.copy()
    a = x[:64]
    sw = np.concatenate([a[32:], a[:32]])
    y[:64] = a * row[:64] + sw * row[64:]
    return y


def rms(x, w):
    return x / np.sqrt((x * x).mean() + EPS) * w


def test_qsa_block_keys_and_queries(device, recipes):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(9)
    cs = rope_tab(4096)
    ixn = (1 + 0.1 * rng.standard_normal(256)).astype(np.float32)
    T, pos0, n, KOFS, P = 16, 6, 9, 640, 900
    p = rng.standard_normal((T, P)).astype(np.float32)
    ikc = np.zeros((64, 128), np.float32)
    ikc[:pos0] = rng.standard_normal((pos0, 128))
    nbc = T // 4 + 1
    g = Graph(device, "t_qsa_kb")
    g.node(
        "rg_qsa_kb_f32",
        [
            g.input("p", (P, T), F32),
            g.input("ikc", (128, 64), F32),
            g.input("cs", (128, 4096), F32),
            g.input("ixn", (256,), F32),
            g.input("meta", (4,), I32),
        ],
        [g.input("iks", (128, T), F32), g.input("kbs", (128, 4 * nbc), F32)],
        struct.pack("fi", EPS, KOFS),
    )
    res = Run(device, recipes, g)(
        {
            "p": p,
            "ikc": ikc,
            "cs": cs,
            "ixn": ixn,
            "meta": np.array([pos0, n, 0, 0], np.int32),
        },
        ["iks", "kbs"],
    )
    raw = np.concatenate([ikc[:pos0], p[:n, KOFS : KOFS + 128]], 0)
    assert rel(res["iks"][:n], p[:n, KOFS : KOFS + 128]) < 1e-6
    for q in range(nbc):
        b = pos0 // 4 + q
        ks = [
            raw[u] if u < pos0 + n else np.zeros(128, np.float32)
            for u in range(4 * b, 4 * b + 4)
        ]
        ref = rope(rms(np.mean(ks, 0), ixn[128:]), cs[4 * b])
        for j in range(4):
            assert rel(res["kbs"][4 * q + j], ref) < 1e-4
    g = Graph(device, "t_qsa_q")
    g.node(
        "rg_qsa_q_f32",
        [
            g.input("p", (P, 2), F32),
            g.input("ixn", (256,), F32),
            g.input("cs", (128, 2), F32),
        ],
        [g.input("q", (128, 8), F32)],
        struct.pack("fi", EPS, 128),
    )
    res = Run(device, recipes, g)({"p": p[:2], "ixn": ixn, "cs": cs[100:102]}, ["q"])
    for t in range(2):
        for h in range(4):
            assert (
                rel(
                    res["q"][h + 4 * t],
                    rope(
                        rms(p[t, 128 + 128 * h : 256 + 128 * h], ixn[:128]), cs[100 + t]
                    ),
                )
                < 1e-4
            )


def _select_ref(s1, q4, kb, pp, pos0, KS, Lc):
    nk = 4 * ((pp + 1) // 4)
    nb = nk // 4
    b0 = pos0 // 4
    bs = np.maximum(s1[::4], 0).sum(1)[:nb].copy()
    fresh = nk > 4 * b0
    if fresh:
        bs[b0] = np.maximum(q4 @ kb, 0).sum()
    order = sorted(range(nb), key=lambda b: (-bs[b], b))
    sel = set(order[: KS // 4])
    km = np.zeros(Lc, np.float32)
    for i in range(min(nk, pos0)):
        if i // 4 not in sel:
            km[i] = -1e30
    own = [
        (-1e30 if (fresh and pos0 + j < nk and b0 not in sel) else 0.0)
        for j in range(2)
    ]
    return km, own


@pytest.mark.parametrize(
    "pos0,T",
    [(300, 1), (303, 1), (40, 1), (63, 1), (307, 1), (302, 2), (303, 2), (401, 2)],
)
def test_qsa_select(device, recipes, pos0, T):
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(10 + pos0)
    KS, NH, Lc, R = 64, 4, 512, 24
    s = rng.standard_normal((Lc // 4, NH * T)).astype(np.float32)
    s[::7] = -np.abs(s[::7])
    s = np.repeat(s, 4, 0)
    q4 = rng.standard_normal((NH * T, 128)).astype(np.float32)
    kb = rng.standard_normal(128).astype(np.float32)
    if pos0 == 307:
        kb = -q4[:NH].sum(0) * 10
    hpg = R if T == 1 else 12
    Tn, RR = (1, R) if T == 1 else (T, hpg * T)
    g = Graph(device, f"t_qsa_sel_{pos0}_{T}")
    ins = [
        g.input("s", (Lc, NH * T), F32),
        g.input("q", (128, NH * T), F32),
        g.input("kbs", (128, 4), F32),
        g.input("meta", (4,), I32),
    ]
    kx = g.tensor((Lc, T), I32)
    g.node("rg_qsa_keys_f32", ins, [kx])
    g.node(
        "rg_qsa_select_f32",
        [kx] + ins[1:],
        [g.input("km", (Lc, T), F32), g.input("kn", (Tn, RR), F32)],
        struct.pack("ii", KS, hpg),
    )
    res = Run(device, recipes, g)(
        {
            "s": s.T,
            "q": q4,
            "kbs": np.tile(kb, (4, 1)),
            "meta": np.array([pos0, T, 0, 0], np.int32),
        },
        ["km", "kn"],
    )
    km, kn = res["km"], res["kn"]
    for t in range(T):
        kref, own = _select_ref(
            s[:, NH * t : NH * (t + 1)],
            q4[NH * t : NH * (t + 1)],
            kb,
            pos0 + t,
            pos0,
            KS,
            Lc,
        )
        assert np.array_equal(km[t, :pos0], kref[:pos0]), (pos0, T, t)
        rows = range(RR) if T == 1 else range(hpg * t, hpg * (t + 1))
        for rr in rows:
            for j in range(min(Tn, t + 1)):
                assert kn[rr, j] == own[j], (pos0, T, t, rr, j)


@pytest.mark.parametrize(
    "L,T,pos0,n,KS",
    [(512, 128, 301, 100, 64), (1024, 128, 1024, 128, 256), (512, 256, 510, 256, 128)],
)
def test_qsa_prefill_mask_keeps_top_blocks(device, recipes, L, T, pos0, n, KS):
    """qsa_keys_pf + qsa_mask_pk: every real query keeps exactly its visible top blocks' keys and its tail"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(11)
    NH = 4
    nbc = T // 4 + 1
    sx = rng.standard_normal((NH * T, L + 4 * nbc)).astype(np.float32)
    sx[:, ::5] = -np.abs(sx[:, ::5])
    g = Graph(device, f"t_qsa_mpf_{L}_{T}_{pos0}")
    si, mi = g.input("s", (L + 4 * nbc, NH * T), F32), g.input("meta", (4,), I32)
    kx = g.tensor((L + T, T), I32)
    g.node("rg_qsa_keys_pf_f32", [si, mi], [kx], struct.pack("i", L))
    g.node(
        "rg_qsa_mask_pk_f32",
        [kx, mi],
        [g.input("mask", (L + T, T), F32)],
        struct.pack("ii", L, KS),
    )
    mask = Run(device, recipes, g)(
        {"s": sx, "meta": np.array([pos0, n, 0, 0], np.int32)}, ["mask"]
    )["mask"]
    # reference (the kernels' contract): key position k < nk = 4 floor((p + 1) / 4) is a complete block's key,
    # scored by the relu-summed head scores of its block key: row k of s when k < kb0 = 4 floor(pos0 / 4), else
    # row L + (k - kb0) (the chunk's fresh block keys). Query t keeps the top KS keys of those (ties: lowest
    # blocks) and every key of its incomplete tail. Mask columns: past key k < pos0 at k, chunk key u at L + u.
    kb0 = 4 * (pos0 // 4)
    for t in range(0, n, max(1, n // 7)):
        p = pos0 + t
        nk = 4 * ((p + 1) // 4)
        sc = sx[NH * t : NH * (t + 1)]  # (NH, L + 4 nbc)

        def score(k):
            return np.maximum(sc[:, k] if k < kb0 else sc[:, L + k - kb0], 0).sum()

        # the kernel ranks keys: the top KS keys below nk by score, ties to the lowest key
        order = sorted(range(nk), key=lambda k: (-score(k), k))
        kept = set(order[:KS])
        row = mask[t]
        for q in range(0, p + 1, max(1, (p + 1) // 40)):
            index = q if q < pos0 else L + (q - pos0)
            visible = q >= nk or q in kept
            assert (row[index] == 0.0) == visible, (t, q)
        assert (row[L + t + 1 : L + T] < 0).all()


def test_ple_conv_batched_rows(device, recipes):
    """ple_conv_b: B rows of one token, each with its own conv state slot, equals ple_conv row by row"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(12)
    B = 5
    x = rng.standard_normal((B, NS * H)).astype(np.float32)
    gv = rng.standard_normal((B, NS * H)).astype(np.float32)
    gvn = rng.standard_normal((B, NS * H)).astype(np.float32)
    cw = rng.standard_normal((4, NS * H)).astype(np.float32) * 0.5
    st = rng.standard_normal((B, 9, NS * H)).astype(np.float32)
    g = Graph(device, "t_ple_conv_b")
    g.node(
        "rg_ple_conv_b_f32",
        [
            g.input("x", (NS * H, B), F32),
            g.input("gv", (NS * H, B), F32),
            g.input("gvn", (NS * H, B), F32),
            g.input("cw", (NS * H, 4), F32),
            g.input("st", (NS * H, 9, B), F32),
            g.input("meta", (4,), I32),
        ],
        [g.input("xo", (NS * H, B), F32), g.input("sto", (NS * H, 9, B), F32)],
    )
    real = B - 1  # the last row is padding
    res = Run(device, recipes, g)(
        {
            "x": x,
            "gv": gv,
            "gvn": gvn,
            "cw": cw,
            "st": st,
            "meta": np.array([0, real, 0, 0], np.int32),
        },
        ["xo", "sto"],
    )
    for b in range(B):
        hist = np.concatenate([st[b], gvn[b : b + 1]], 0)  # (10, C)
        conv = sum(cw[k] * hist[9 - 3 * (3 - k)] for k in range(4))
        want = x[b] + gv[b] + (silu(conv) if b < real else 0)
        assert rel(res["xo"][b], want) < 1e-4, b
        assert rel(res["sto"][b], hist[1:]) < 1e-6, b


def test_qsa_block_keys_batched_rows(device, recipes):
    """qsa_kb_b: each row's raw key and its block key from the block's earlier keys plus the new one"""
    from resin_models_gaudi.synapse.graph import Graph
    from resin_models_gaudi.synapse.runtime import F32, I32

    rng = np.random.default_rng(13)
    cs = rope_tab(4096)
    ixn = (1 + 0.1 * rng.standard_normal(256)).astype(np.float32)
    B, P, KOFS = 4, 900, 640
    positions = np.array([300, 303, 41, 7], np.int32)
    p = rng.standard_normal((B, P)).astype(np.float32)
    tail = rng.standard_normal((B, 4, 128)).astype(np.float32)
    g = Graph(device, "t_qsa_kb_b")
    g.node(
        "rg_qsa_kb_b_f32",
        [
            g.input("p", (P, B), F32),
            g.input("tail", (128, 4, B), F32),
            g.input("pos", (B,), I32),
            g.input("cs", (128, 4096), F32),
            g.input("ixn", (256,), F32),
        ],
        [g.input("iks", (128, B), F32), g.input("kbs", (128, 4, B), F32)],
        struct.pack("fi", EPS, KOFS),
    )
    res = Run(device, recipes, g)(
        {"p": p, "tail": tail, "pos": positions, "cs": cs, "ixn": ixn}, ["iks", "kbs"]
    )
    for b in range(B):
        new = p[b, KOFS : KOFS + 128]
        assert rel(res["iks"][b], new) < 1e-6
        have = int(positions[b]) % 4
        keys = [tail[b, j] for j in range(have)] + [new]
        ref = rope(
            rms(np.sum(keys, 0) / 4, ixn[128:]), cs[4 * (int(positions[b]) // 4)]
        )
        for j in range(4):
            assert rel(res["kbs"][b, j], ref) < 1e-4, (b, j)
