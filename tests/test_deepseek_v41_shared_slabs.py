"""DeepSeek-V4.1-Flash's cache geometry in slabs that another model's slabs share: each slab holds as many WINDOW
or HISTORY blocks as fit, and every address stays inside its slab."""

from __future__ import annotations

import numpy as np
import pytest

from resin_models_gaudi.deepseek_v41.cache import ARENA_BLOCK_ROWS, CacheGeometry

# DeepSeek-V4.1-Flash: 40 layers; KV owners 2, 8, 14, 20 with compress ratios 2, 2, 2, 1
OWN = CacheGeometry(40, (2, 8, 14, 20), (2, 2, 2, 1), 128, 128)
QWEN_9B_SLAB = 106_496  # rows of a Qwen3.5-9B slab


def test_own_slabs_hold_one_window_block():
    assert OWN.window_block_rows == 5248 and OWN.history_block_rows == 896
    assert OWN.slab_rows == OWN.own_slab_rows == 5248
    window, history = OWN.layout(10).groups
    assert (window.blocks_per_slab, history.blocks_per_slab) == (1, 5)


def test_a_shared_slab_holds_as_many_blocks_as_fit():
    shared = OWN.with_slab_rows(QWEN_9B_SLAB)
    assert shared.slab_rows == QWEN_9B_SLAB and shared.own_slab_rows == 5248
    window, history = shared.layout(10).groups
    assert (window.blocks_per_slab, history.blocks_per_slab) == (20, 118)
    assert shared.arena_blocks(10) == 11 * QWEN_9B_SLAB // ARENA_BLOCK_ROWS
    assert shared.window_blocks(10) == 11 * 20
    assert shared.state_bytes(10) == OWN.state_bytes(10) * 20
    # the last block of a slab ends inside it
    last_window = shared.window_block_row([3], [19])[0] + shared.window_block_rows
    last_history = shared.history_block_row([3], [117])[0] + shared.history_block_rows
    assert last_window <= 4 * QWEN_9B_SLAB and last_history <= 4 * QWEN_9B_SLAB
    assert np.array_equal(shared.window_block_id([2, 3], [0, 19]), [40, 79])
    # a capacity fits the arena and the pending state of a slab
    per = QWEN_9B_SLAB * 512 + 20 * shared.state_rows_per_block * 1024 * 4
    assert shared.capacity_for(100 * per) == 98


@pytest.mark.parametrize("rows", [5120, 5248 + 64])
def test_a_shared_slab_must_hold_a_window_block_in_whole_units(rows):
    with pytest.raises(ValueError, match="shared slab"):
        OWN.with_slab_rows(rows)
