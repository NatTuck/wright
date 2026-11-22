"""The MoE expert block's dequant -> batch_gemm as ONE call, against the two-half form it replaced.

``_w13`` / ``_w2`` used to split each slot's rows into two halves, decode and multiply each apart and
``cat`` the two products together.  One ``rg_mx4_decs_f32`` over the whole slot row plus one
``batch_gemm`` gives the same product bit-for-bit (the dequant kernel's per-launch ``u0`` unit offset is
what made the split possible, not necessary), and the graph compiler turns the single MME operand into
two bundles and 18 MB of SRAM instead of four bundles and 30 MB -- the dequant tile is SRAM-resident
either way, which is the part that matters: see ``cases/wright/notes/moe-dequant-tile-sram.md``.

These tests pin the equivalence for both weights.  They are a gate on the *restructure*, so they must
fail if the one-call form stops agreeing with the two-half form.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    try:
        opened = open_device(None)
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


@pytest.fixture(scope="module")
def recipes(device, tmp_path_factory):
    from resin_models_gaudi.synapse.graph import RecipeCache

    return RecipeCache(device, tmp_path_factory.mktemp("recipes"))


class Run:
    """Compile a graph, bind numpy inputs and output buffers by name, launch once (as in
    test_ds41_kernels.py)."""

    def __init__(self, device, recipes, graph) -> None:
        self.device, self.graph = device, graph
        self.recipe = recipes.compile(graph)

    def __call__(self, inputs, outputs):
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


# name, the slot's element count, the (contraction, rows) of the whole product, and of each half
CASES = {
    "x2": (lambda H, w: H * w, lambda H, w: (w, H), lambda H, w: (w, H // 2)),
    "x13": (lambda H, w: 2 * w * H, lambda H, w: (H, 2 * w), lambda H, w: (H, w)),
}


def slot_product(device, recipes, which, H, w, E, R, T, n, *, act=None, weights=None, dt=None):
    """One ``_halves`` of ``which`` with ``n`` parts of dtype ``dt`` (F32 by default), launched on ``act``
    against ``weights``."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import F8, F32, I32

    size_of, whole_of, half_of = CASES[which]
    part = whole_of(H, w) if n == 1 else half_of(H, w)
    d = L.Dims(
        hidden=H, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=w, experts=E,
        top=R, vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0, route_scale=1.5, ranks=1,
    )  # fmt: skip
    g = L.LayerGraph(device, f"t_whole_{which}_{n}_{dt}", d, T)
    h = g.input("h", act.shape, F8)
    ids = g.input("ids", (64, T), I32)
    y = L._halves(g, h, ids, R, which, size_of(H, w), part, n, dt)
    g.node("memcpy", [y], [g.input("y", y.shape, y.dtype)])
    out = Run(device, recipes, g.g)(
        {
            "h": act,
            "ids": np.tile(np.arange(64, dtype=np.int32), (T, 1)) % E,
            which: weights[0],
            which + ".s": weights[1],
            which + ".k": np.full(E, -10, np.int32),
        },
        ["y"],
    )["y"]
    g.g.destroy()
    return out


@pytest.mark.parametrize("which", ["x2", "x13"])
def test_slot_product_whole_equals_halves(device, recipes, which):
    """The whole product in one ``_halves`` part is bit-identical to the two-half form it replaced."""
    rng = np.random.default_rng(11)
    H, w, E, R, T = 5120, 288, 6, 6, 1
    size_of = CASES[which][0]
    # an activation that cannot decode to an fp8 poison byte (0x78..0x7E / 0xF8..0xFE)
    act = rng.integers(0, 0x70, (H, 1, R * T) if which == "x13" else (w, 1, R * T)).astype(np.uint8)
    weights = (
        rng.integers(0, 256, (E, size_of(H, w) // 2), dtype=np.uint8),
        rng.integers(118, 124, (E, size_of(H, w) // 32), dtype=np.uint8),
    )
    whole = slot_product(device, recipes, which, H, w, E, R, T, 1, act=act, weights=weights)
    halves = slot_product(device, recipes, which, H, w, E, R, T, 2, act=act, weights=weights)
    assert np.isfinite(whole).all() and np.isfinite(halves).all()
    np.testing.assert_array_equal(whole, halves)


def bf16_to_f32(u):
    """bf16 bits -> f32 values (the upper half of the f32 word)."""
    return (np.asarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    """f32 -> bf16 bits, round half to even."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


@pytest.mark.parametrize("which", ["x2", "x13"])
def test_slot_product_bf16_is_the_rounded_f32(device, recipes, which):
    """The BF16 product is the f32 product rounded once to bf16 -- the paired differential against the f32 arm.

    The reference's expert GEMM is ``fp8_gemm(..., out_dtype=BF16)`` added into an f32 accumulator, so this
    rounding is what the model already does; the claim this test pins is that the bf16 arm is *the same
    product* (one rounding of it), not a different accumulation.  It fails on a mis-declared tensor, a changed
    accumulation order, or a kernel that reads the tile as something other than bf16.
    """
    from resin_models_gaudi.synapse.runtime import BF16, F32

    rng = np.random.default_rng(12)
    H, w, E, R, T = 5120, 576, 384, 6, 2
    size_of = CASES[which][0]
    act = rng.integers(0, 0x70, (H, 1, R * T) if which == "x13" else (w, 1, R * T)).astype(np.uint8)
    weights = (
        rng.integers(0, 256, (E, size_of(H, w) // 2), dtype=np.uint8),
        rng.integers(118, 124, (E, size_of(H, w) // 32), dtype=np.uint8),
    )
    f32 = slot_product(device, recipes, which, H, w, E, R, T, 1, act=act, weights=weights, dt=F32)
    b16 = bf16_to_f32(
        slot_product(device, recipes, which, H, w, E, R, T, 1, act=act, weights=weights, dt=BF16)
    )
    assert np.isfinite(b16).all(), "the bf16 product overflowed"
    np.testing.assert_array_equal(b16, bf16_to_f32(f32_to_bf16(f32)))
    rel = np.abs(b16 - f32) / np.maximum(np.abs(f32), 1e-30)
    assert rel.max() <= 2.0**-8, f"bf16 rounding exceeded 2**-8: {rel.max():.3e}"
    print(
        f"{which}: bf16 is the f32 product rounded to bf16, max rel {rel.max():.3e}, "
        f"mean {rel.mean():.3e} over {rel.size} elements"
    )


def test_expert_product_dtype_and_kernel(device, recipes):
    """``_w13`` / ``_w2`` emit a BF16 product and the SwiGLU kernel that reads a BF16 tile, for the 4-rank
    width; the 8-rank width (a 32-row tail, which a 128-element bf16 load cannot address) keeps f32."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import BF16, F32

    def dims(w):
        return L.Dims(
            hidden=5120, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=w,
            experts=384, top=6, vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0,
            route_scale=1.5, ranks=1,
        )  # fmt: skip

    for w, want in ((576, BF16), (288, F32)):
        d = dims(w)
        g = L.LayerGraph(device, f"t_prod_dtype_{w}", d, 2)
        xq, ids = g.input("x", (5120, 1, 12), 8192), g.input("ids", (64, 2), 16)
        before = g.g._nodes
        y13 = L._w13(g, xq, ids, 6, L.expert_out_dtype(g))
        assert y13.dtype == want, (w, y13.dtype)
        hq = g.input("h", (w, 1, 12), 8192)
        y2 = L._w2(g, hq, ids, 6, L.expert_out_dtype(g))
        assert y2.dtype == want, (w, y2.dtype)
        assert g.g._nodes - before == 6
        assert L._xact_lin(want) == (
            "rg_ds_xact_lin_b16_f32" if want == BF16 else "rg_ds_xact_lin_f32"
        )
        g.g.destroy()


def test_w13_and_w2_are_one_call(device, recipes):
    """``_w13`` / ``_w2`` emit one dequant and one batch_gemm each -- no ``concat``, one weight operand.

    The node count is the instrument: one dequant + its reshape + one batch_gemm is 3 nodes; the two-half
    form it replaced is 2 x 3 + a ``concat`` = 7.
    """
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import F8, I32

    H, w, E, R, T = 5120, 288, 6, 6, 3
    d = L.Dims(
        hidden=H, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=w, experts=E,
        top=R, vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0, route_scale=1.5, ranks=1,
    )  # fmt: skip
    g = L.LayerGraph(device, "t_one_call", d, T)
    S = R * T
    xq = g.input("x", (H, 1, S), F8)
    ids = g.input("ids", (64, T), I32)
    before = g.g._nodes
    y13 = L._w13(g, xq, ids, R)
    assert g.g._nodes - before == 3, "w13 must be one dequant, one reshape and one batch_gemm"
    hq = g.input("h", (w, 1, S), F8)
    before = g.g._nodes
    y2 = L._w2(g, hq, ids, R)
    assert g.g._nodes - before == 3, "w2 must be one dequant, one reshape and one batch_gemm"
    assert y13.shape == (2 * w, 1, S) and y2.shape == (H, 1, S)
    # The negative control: the two-half form the change replaced counts 7, so the assertions above can
    # fail.  A count that could not distinguish the two would be decoration.
    before = g.g._nodes
    L._halves(g, xq, ids, R, "x13", 2 * w * H, (H, w), 2)
    assert g.g._nodes - before == 7, "the two-half form must count 7, not 3"
