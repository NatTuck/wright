"""One cache arena for the models of a card.

Each model's cache is rows of 512 bytes in one device allocation, its arena. When several models
share the card, they share one arena: the slabs of one cache pool (``resin.CachePool``) and,
after the slabs, each model's scratch rows. A slab is as large as the largest slab of any of the
models, and every model's recipes declare the whole arena.

Order of a start with several models: load every model's weights and fixed buffers
(``defer_cache=True``), then ``plan_shared_arena`` with what the weights leave free, then
``attach`` every model to the arena.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from resin.backend.capacity import plan_cache_capacity
from resin_models_gaudi.synapse.runtime import Device

ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024  # rows of the arena's dim 1
ARENA_PAD_BLOCKS = 1
"""Blocks after the slabs and scratch of an arena: a model's view of a layer that starts part of a block into the
arena ends past the slabs (``qwen3_5.cache.CacheGeometry.attention_blocks``)."""


@dataclass(frozen=True, slots=True)
class SharedArena:
    address: int
    blocks: int
    """The arena's dim 2 (1024-row blocks) in every model's recipes."""
    slab_rows: int
    parent_capacity: int
    """Slabs of the pool, without the null slab (slab 0)."""
    scratch: tuple[int, ...]
    """Each model's first scratch row (after the slabs)."""

    @property
    def nbytes(self) -> int:
        return self.blocks * ARENA_BLOCK_ROWS * ROW_BYTES


def shared_slab_rows(slab_rows: Sequence[int]) -> int:
    """The slab of the shared pool: the largest of the models', in whole arena blocks."""
    largest = max(slab_rows)
    return -(-largest // ARENA_BLOCK_ROWS) * ARENA_BLOCK_ROWS


def plan_shared_arena(
    device: Device,
    slab_rows: Sequence[int],
    scratch_rows: Sequence[int],
    *,
    memory_utilization: float = 0.9,
    reserve: int = 2 << 30,
    quantum: int = 8,
    parent_capacity: int | None = None,
) -> SharedArena:
    """Allocate and clear the arena of models whose own slabs are ``slab_rows`` and whose scratch
    is ``scratch_rows``, from the card memory that is free now (their weights and buffers are
    loaded). ``reserve``: the largest recipe workspace any of them keeps free. ``quantum``: the
    capacity is a multiple of it, so small changes of free memory do not change it (and with it
    every recipe's arena shape)."""
    if not slab_rows or len(slab_rows) != len(scratch_rows):
        raise ValueError("one slab size and one scratch size for each model")
    slab = shared_slab_rows(slab_rows)
    scratch_blocks = [-(-rows // ARENA_BLOCK_ROWS) for rows in scratch_rows]
    scratch_bytes = sum(scratch_blocks) * ARENA_BLOCK_ROWS * ROW_BYTES
    if parent_capacity is None:
        free, total = device.memory_info()
        planned = plan_cache_capacity(
            total,
            free - scratch_bytes,
            memory_utilization,
            slab * ROW_BYTES,
            overhead_bytes=reserve,
        ).parent_capacity
        parent_capacity = max(1, planned // quantum * quantum)
    slab_blocks = (parent_capacity + 1) * slab // ARENA_BLOCK_ROWS
    first, scratch = slab_blocks * ARENA_BLOCK_ROWS, []
    for blocks in scratch_blocks:
        scratch.append(first)
        first += blocks * ARENA_BLOCK_ROWS
    blocks = slab_blocks + sum(scratch_blocks) + ARENA_PAD_BLOCKS
    nbytes = blocks * ARENA_BLOCK_ROWS * ROW_BYTES
    address = device.malloc(nbytes)
    device.memset32(address, 0, nbytes // 4)
    return SharedArena(address, blocks, slab, parent_capacity, tuple(scratch))
