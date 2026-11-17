"""Qwen3.8-Flash-Next's cache geometry in slabs that another model's slabs share: a slab is a multiple of whole KV
blocks and arena blocks, and holds as many KV and STATE blocks as fit."""

from __future__ import annotations

import pytest

from resin_models_gaudi.qwen4_exp.cache import CacheGeometry

GEO = CacheGeometry(
    full_layers=4, linear_layers=12, kv_groups=1, ssm_rows=1024, conv_rows=48, ple_rows=40
)


def test_a_shared_slab_holds_as_many_blocks_as_fit():
    assert GEO.kv_block_rows == 4 * 4 * 128
    assert GEO.slab_unit == 2048 and GEO.own_slab_rows % GEO.slab_unit == 0
    assert GEO.slab_rows == GEO.own_slab_rows
    rows = 4 * GEO.own_slab_rows
    shared = GEO.with_slab_rows(rows)
    assert shared.slab_rows == rows and shared.own_slab_rows == GEO.own_slab_rows
    assert shared.kv_blocks_per_slab == rows // GEO.kv_block_rows
    assert shared.state_blocks_per_slab == rows // GEO.state_block_rows
    assert shared.arena_blocks(3) == 4 * rows // 1024


def test_a_shared_slab_must_hold_this_models_slab_in_whole_units():
    for rows in (GEO.own_slab_rows + 1024, GEO.own_slab_rows - GEO.slab_unit):
        with pytest.raises(ValueError, match="a shared slab"):
            GEO.with_slab_rows(rows)
