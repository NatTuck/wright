"""The cache arena that several models of one card share, planned on the card-free device."""

from __future__ import annotations

import pytest

from resin_models_gaudi.arena import (
    ARENA_BLOCK_ROWS,
    ARENA_PAD_BLOCKS,
    ROW_BYTES,
    plan_shared_arena,
    shared_slab_rows,
)
from resin_models_gaudi.qwen3_5.cache import CacheGeometry
from resin_models_gaudi.synapse.dry import DryLib, dry_device

GiB = 1 << 30


def test_the_shared_slab_is_the_largest_in_whole_arena_blocks() -> None:
    assert shared_slab_rows([3 * 1024, 5 * 1024 + 1, 2048]) == 6 * 1024
    assert shared_slab_rows([311_296]) == 311_296


def test_the_arena_holds_the_slabs_then_each_model_s_scratch() -> None:
    device = dry_device(total_bytes=96 * GiB)
    device.malloc(60 * GiB)  # the models' weights and buffers
    arena = plan_shared_arena(
        device,
        [311_296, 100_352],
        [0, 3000],
        memory_utilization=0.9,
        reserve=2 * GiB,
        quantum=8,
    )
    slab_bytes = 311_296 * ROW_BYTES
    assert arena.slab_rows == 311_296
    assert arena.parent_capacity % 8 == 0 and arena.parent_capacity > 0
    slabs_end = (arena.parent_capacity + 1) * arena.slab_rows
    assert arena.scratch == (slabs_end, slabs_end)  # the first model has none
    # the second model's scratch (3 blocks), then the padding
    assert arena.blocks * ARENA_BLOCK_ROWS == slabs_end + (3 + ARENA_PAD_BLOCKS) * ARENA_BLOCK_ROWS
    # it fits what the weights left, with the 10% kept free and the workspace reserve
    left = 96 * GiB - 60 * GiB - int(96 * GiB * 0.1) - 2 * GiB
    assert arena.nbytes <= left + (3 + ARENA_PAD_BLOCKS) * ARENA_BLOCK_ROWS * ROW_BYTES + slab_bytes
    lib = device.lib
    assert isinstance(lib, DryLib)
    assert lib.allocations[arena.address] == arena.nbytes


def test_a_requested_capacity_is_kept() -> None:
    arena = plan_shared_arena(dry_device(), [2048, 4096], [0, 0], parent_capacity=10)
    assert arena.parent_capacity == 10
    assert arena.blocks == 11 * 4096 // ARENA_BLOCK_ROWS + ARENA_PAD_BLOCKS


def test_a_qwen_geometry_takes_the_shared_slab() -> None:
    from dataclasses import replace

    geometry = CacheGeometry(
        full_layers=4,
        linear_layers=12,
        kv_groups=2,
        ssm_rows=4096,
        conv_rows=96,
        hidden_rows=40,
    )
    own = geometry.own_slab_rows
    assert geometry.slab_rows == own
    shared = replace(geometry, shared_slab_rows=own * 3)
    assert shared.slab_rows == own * 3
    assert shared.kv_blocks_per_slab == 3 * geometry.kv_blocks_per_slab
    assert shared.kv_block_row(2, 1) == 2 * own * 3 + shared.kv_block_rows
    with pytest.raises(ValueError, match="smaller than this model's"):
        _ = replace(geometry, shared_slab_rows=own - 1024).slab_rows
