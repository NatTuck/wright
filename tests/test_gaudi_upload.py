"""The shared Gaudi weight uploader on the card-free device: allocation sizes, the kinds,
the requantization graphs, ownership and errors."""

from __future__ import annotations

import numpy as np
import pytest

from resin_models_gaudi.loading.upload import Fp8, WeightLoader
from resin_models_gaudi.synapse.dry import DryRecipeCache, dry_device
from resin_models_gaudi.synapse.graph import Graph
from resin_models_gaudi.synapse.runtime import F8, F32


def _loader(min_alloc: int = 256):
    device = dry_device(keep_data=True)
    recipes = DryRecipeCache(device)
    return device, recipes, WeightLoader(device, recipes, min_alloc=min_alloc)


def _launches(device) -> list[tuple[str, dict[str, int]]]:
    return [(c[1], c[2]) for c in device.lib.calls if c[0] == "launch"]


@pytest.mark.parametrize("min_alloc", [0, 256])
def test_plain_items_upload_as_they_are_with_the_model_minimum(min_alloc):
    device, _, loader = _loader(min_alloc)
    small = np.arange(4, dtype=np.float32)  # 16 bytes
    large = np.arange(1000, dtype=np.uint16)
    loader.load([("small", small, "f32"), ("large", large, "planar:Q8_0")])
    assert device.lib.allocations[loader.w["small"]] == max(16, min_alloc)
    assert device.lib.allocations[loader.w["large"]] == 2000
    assert device.lib.read(loader.w["small"], 16) == small.tobytes()
    assert device.lib.read(loader.w["large"], 2000) == large.tobytes()
    assert loader.kinds == {"small": "f32", "large": "planar:Q8_0"}
    assert loader.nbytes == 16 + 2000


def _fp8(M: int, K: int, rng) -> Fp8:
    return Fp8(
        rng.integers(0, 0x7E, (M, K), dtype=np.uint8),
        rng.uniform(0.5, 2.0, (M // 128, K // 128)).astype(np.float32),
    )


def test_fp8_halves_on_the_card_then_requantizes():
    device, recipes, loader = _loader()
    rng = np.random.default_rng(0)
    w = _fp8(256, 384, rng)
    loader.load([("l0.w", w, "fp8")])
    assert recipes.compiled == ["fp8_inc_384", "requant_fp8_384x256"]
    (inc, inc_binds), (requant, binds) = _launches(device)
    assert inc == "fp8_inc_384" and requant == "requant_fp8_384x256"
    assert binds["w"] == inc_binds["o"]  # the halved bytes feed the requantization
    assert binds["o"] == loader.w["l0.w"] and binds["rs"] == loader.w["l0.w.s"]
    assert device.lib.allocations[loader.w["l0.w"]] == 256 * 384
    assert device.lib.allocations[loader.w["l0.w.s"]] == 256 * 4
    # the staging buffers are freed when the load ends: only the outputs remain
    assert sorted(device.lib.allocations) == sorted(loader.w.values())


def test_inc8_only_requantizes_and_x8_requantizes_experts():
    device, recipes, loader = _loader()
    rng = np.random.default_rng(1)
    loader.load([("d.w", _fp8(128, 256, rng), "inc8")])
    assert recipes.compiled == ["requant_fp8_256x128"]
    experts = Fp8(
        rng.integers(0, 0x7E, (2, 256, 128), dtype=np.uint8),
        rng.uniform(0.5, 2.0, (2, 2, 1)).astype(np.float32),
    )
    loader.load([("l1.x13", experts, "x8")])
    assert recipes.compiled[1:] == ["fp8_inc_256", "x8_requant_128x256x2"]
    assert device.lib.allocations[loader.w["l1.x13.c"]] == 4 * 128 * 2
    assert loader.kinds == {"d.w": "inc8", "l1.x13": "x8"}


def test_the_requantization_graph_is_the_one_both_old_loaders_built():
    """Its definition is part of the recipe's cache key: a change would recompile at startup."""
    device, recipes, loader = _loader()
    loader.load([("w", _fp8(128, 256, np.random.default_rng(2)), "inc8")])
    reference = Graph(device, "requant_fp8_256x128")
    w = reference.input("w", (256, 128), F8)
    bs = reference.input("bs", (2, 1), F32)
    out = reference.input("o", (256, 128), F8)
    rs = reference.input("rs", (1, 128), F32)
    reference.node("rg_requant_fp8_f32", [w, bs], [out, rs])
    built = loader._recipes[("requant", 256, 128)]
    assert built.identity == reference.identity()


def test_stream_items_upload_themselves():
    device, _, loader = _loader()

    class Table:
        nbytes = 4096

        def upload(self, address, h2d, memset):
            h2d(address, np.full(1024, 7, np.uint8))
            memset(address + 1024, 3072)

    loader.load([("ple.tab", Table(), "stream")])
    data = device.lib.read(loader.w["ple.tab"], 4096)
    assert data[:1024] == bytes([7]) * 1024 and data[1024:] == bytes(3072)
    assert loader.kinds["ple.tab"] == "stream"


def test_close_frees_only_the_loader_allocations():
    device, _, loader = _loader()
    loader.load([("a", np.zeros(512, np.uint8), "u8")])
    loader.w["a.alias"] = loader.w["a"] + 128  # a model's own name for part of a weight
    extra = loader.malloc(64)
    other = device.malloc(64)  # someone else's allocation on the same card
    loader.close()
    assert list(device.lib.allocations) == [other]
    assert loader.w == {} and extra not in device.lib.allocations


def test_a_name_twice_and_a_failing_preparation_stop_the_load():
    device, _, loader = _loader()
    with pytest.raises(ValueError, match="uploaded twice"):
        loader.load([("a", np.zeros(4, np.float32), "f32")] * 2)

    def broken():
        yield "b", _fp8(128, 256, np.random.default_rng(3)), "fp8"
        raise RuntimeError("bad shard")

    with pytest.raises(RuntimeError, match="bad shard"):
        loader.load(broken())
    # staging is freed even when the load fails
    assert sorted(device.lib.allocations) == sorted(loader.w.values())
