"""The Qwen3.5/3.8 KV page formats (``qwen3_5/cache.py``): bf16 pages of 128 rows, FP8 pages of 65 rows, and the
``kv_rows`` entries that ``kv_store8`` takes (an FP8 page's keys as 256-byte units)."""

from __future__ import annotations

import numpy as np
import pytest

from resin_models_gaudi.common.attention import SlotRow, kv_rows
from resin_models_gaudi.loading.convert import sigma_inv
from resin_models_gaudi.qwen3_5.cache import CacheGeometry, kv8_bits


def geometry(kv8: int, groups: int = 4) -> CacheGeometry:
    # Qwen3.8-27B: 16 full-attention layers, 48 GDN layers
    return CacheGeometry(16, 48, groups, 6144, 256, 40, mtp=True, kv8=kv8)


def test_the_types_name_the_fp8_pages():
    assert kv8_bits(("bf16", "bf16")) == 0
    assert kv8_bits(("fp8", "bf16")) == 1
    assert kv8_bits(("bf16", "fp8")) == 2
    assert kv8_bits(("fp8", "fp8")) == 3
    with pytest.raises(ValueError, match="cache type 'q8_0'"):
        kv8_bits(("q8_0", "bf16"))


@pytest.mark.parametrize("kv8", [0, 1, 2, 3])
def test_a_layer_holds_its_k_then_its_v_pages(kv8):
    g = geometry(kv8)
    k_rows, v_rows = (65 if kv8 & bit else 128 for bit in (1, 2))
    assert (g.k_page_rows, g.v_page_rows) == (k_rows, v_rows)
    assert g.kv_layer_rows == 4 * (k_rows + v_rows)
    assert g.mtp_layer_rows == g.kv_layer_rows
    k, v = g.kv_pages(np.array([0, 100_000]))
    np.testing.assert_array_equal(k[:, 0], np.arange(4) * k_rows)
    np.testing.assert_array_equal(v[:, 0], 4 * k_rows + np.arange(4) * v_rows)
    np.testing.assert_array_equal(k[:, 1] - k[:, 0], 100_000)
    # the pages of a layer do not overlap and end at the layer's rows
    assert v[3, 0] + v_rows == g.kv_layer_rows


@pytest.mark.parametrize("kv8", [0, 1, 2, 3])
@pytest.mark.parametrize("groups", [4, 2, 1])
def test_a_slab_is_whole_arena_blocks(kv8, groups):
    g = geometry(kv8, groups)
    assert g.own_slab_rows % 1024 == 0 and g.own_slab_rows >= g.state_block_rows
    assert g.kv_blocks_per_slab * g.kv_block_rows <= g.own_slab_rows
    g.arena_blocks(100)


def test_fp8_halves_the_kv_block_and_doubles_the_tokens_of_a_slab():
    bf16, fp8 = geometry(0), geometry(3)
    assert fp8.kv_block_rows * 128 == bf16.kv_block_rows * 65
    # a slab holds a STATE block, so its size stays; it holds about twice the KV blocks
    assert abs(fp8.own_slab_rows - bf16.own_slab_rows) < bf16.kv_block_rows
    assert fp8.kv_blocks_per_slab >= 1.9 * bf16.kv_blocks_per_slab


@pytest.mark.parametrize("kv8", [0, 1, 2, 3])
def test_kv_rows_give_fp8_keys_as_units(kv8):
    g = geometry(kv8, groups=2)
    blocks = np.array([1000, 5000, 9000])
    k, v = g.kv_pages(blocks)
    row = SlotRow(column=3, pos0=200, real=100, k_pages=k, v_pages=v)
    out = kv_rows(128, [row], kv8)
    pos = 200 + np.arange(100)
    page, slot = pos // 128, pos % 128
    for grp in range(2):
        kp, vp = k[grp, page], v[grp, page]
        want_k = 2 * kp + slot if kv8 & 1 else kp + slot
        want_v = 2 * vp + sigma_inv(slot) if kv8 & 2 else vp + sigma_inv(slot)
        np.testing.assert_array_equal(out[3:103, grp], want_k)
        np.testing.assert_array_equal(out[3:103, 4 + grp], want_v)
    assert (
        (out[:3] == -1).all() and (out[103:] == -1).all() and (out[:, 2:4] == -1).all()
    )
