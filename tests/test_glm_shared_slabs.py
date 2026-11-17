"""GLM-5.3-Flash's cache geometry in slabs that another model's slabs share: a slab is a multiple of whole KV
blocks and arena blocks, it holds as many KV and STATE blocks as fit, and every address stays inside it."""

from __future__ import annotations

import math

import pytest

from resin_models_gaudi.glm5_3_flash.cache import ARENA_BLOCK_ROWS, CacheGeometry

# GLM-5.3-Flash at TP4: 11 sparse-attention layers and 34 KDA layers with 16 local heads
OWN = CacheGeometry(tuple(range(11)), tuple(range(11, 45)), 2048, 144, 32)
QWEN_9B_SLAB = 106_496  # rows of a Qwen3.5-9B slab


def test_own_slabs_hold_one_state_block():
    assert OWN.kv_block_rows == 1792 and OWN.state_block_rows == 78_592
    assert OWN.slab_unit == math.lcm(1792, ARENA_BLOCK_ROWS) == 7168
    assert OWN.slab_rows == OWN.own_slab_rows == 78_848
    kv, state = OWN.layout(10).groups
    assert (kv.blocks_per_slab, state.blocks_per_slab) == (44, 1)


def test_a_shared_slab_is_a_multiple_of_both_models_units():
    # the serving rule: the largest slab, in whole units of every model's
    unit = math.lcm(ARENA_BLOCK_ROWS, OWN.slab_unit, ARENA_BLOCK_ROWS)
    rows = -(-max(OWN.own_slab_rows, QWEN_9B_SLAB) // unit) * unit
    assert rows == 107_520
    shared = OWN.with_slab_rows(rows)
    assert shared.slab_rows == rows and shared.own_slab_rows == 78_848
    kv, state = shared.layout(10).groups
    assert (kv.blocks_per_slab, state.blocks_per_slab) == (60, 1)
    assert shared.arena_blocks(10) == 11 * rows // ARENA_BLOCK_ROWS
    # the last KV block and the STATE block of a slab end inside it
    assert shared.kv_block_row([3], [59])[0] + shared.kv_block_rows <= 4 * rows
    assert shared.state_block_row(3, 0) + shared.state_block_rows <= 4 * rows
    assert shared.capacity_for(100 * shared.slab_bytes, shared.slab_bytes) == 99


def test_a_shared_slab_must_hold_this_models_slab_in_whole_units():
    for rows in (QWEN_9B_SLAB, 71_680):  # not whole KV blocks; smaller than a STATE block
        with pytest.raises(ValueError, match="a shared slab"):
            OWN.with_slab_rows(rows)
