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


def slot_product(device, recipes, which, H, w, E, R, T, n, *, act=None, weights=None):
    """One ``_halves`` of ``which`` with ``n`` parts, launched on ``act`` against ``weights``."""
    from resin_models_gaudi.deepseek_v41 import layers as L
    from resin_models_gaudi.synapse.runtime import F8, F32, I32

    size_of, whole_of, half_of = CASES[which]
    part = whole_of(H, w) if n == 1 else half_of(H, w)
    d = L.Dims(
        hidden=H, heads=8, head_dim=512, q_lora=1024, o_lora=1024, groups=1, expert=w, experts=E,
        top=R, vocab=16160, eps=1e-6, hc_iters=20, hc_eps=1e-6, limit=10.0, route_scale=1.5, ranks=1,
    )  # fmt: skip
    g = L.LayerGraph(device, f"t_whole_{which}_{n}", d, T)
    h = g.input("h", act.shape, F8)
    ids = g.input("ids", (64, T), I32)
    y = L._halves(g, h, ids, R, which, size_of(H, w), part, n)
    g.node("memcpy", [y], [g.input("y", y.shape, F32)])
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
