"""Cache geometry of Mistral-Large-4 on Gaudi: the Engine's paged KV group in one arena of
512-byte rows, replicated on every tensor-parallel rank (every head reads the shared latent).

A KV block (128 tokens) holds, for each layer that owns a cache (63 of 72; the KV-sharing
layers read their source's pages):

- a latent page of 129 rows: token r's 512-dim latent as Gaudi2 FP8 E4M3 in row ``r`` (natural
  order), and row 128 holding the 128 tokens' f32 scales (max |latent| / 240, bf16 values);
- in rope layers, a rotary page of 32 rows: token r's 64 bf16 in 128-byte unit ``r`` (four per
  row);
- in layers that own a DSA indexer (44), an index-key page of 64 rows: token r's 128 bf16 in
  256-byte unit ``r`` (two per row).

With the DSpark drafter (``draft_layers``), each of its layers adds a latent and a rotary page.

Pages are laid out latent pages first (in owner order), then index pages, then rotary pages, then
the drafter's latent and rotary pages.
The arena's total rows are a multiple of 1024.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resin import CacheGroupLayout, CacheLayout
from resin.models.mistral_large4.config import MistralLarge4Config

PAGE_TOKENS = 128
ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024
LATENT_PAGE_ROWS = 129  # 128 FP8 token rows and one row of their scales
ROPE_PAGE_ROWS = 32  # 128-byte units, four per row
INDEX_PAGE_ROWS = 64  # 256-byte units, two per row
STORE_FIELDS = 8  # mla_store's per-token destination record
KV_GROUP = 0
SLAB_MULTIPLE = (
    8  # (capacity + 1) slabs round to this, so the arena is whole 1024-row blocks
)


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    kv_owners: tuple[int, ...]
    """layers that own a latent cache"""
    index_owners: tuple[int, ...]
    rope_owners: tuple[int, ...]
    kv_sources: tuple[int, ...]
    draft_layers: int = 0
    """the DSpark drafter's layers (each a latent and a rotary page per block)"""

    @classmethod
    def from_config(
        cls, cfg: MistralLarge4Config, draft_layers: int = 0
    ) -> CacheGeometry:
        owners = cfg.kv_owner_layers
        return cls(
            owners,
            cfg.index_owner_layers,
            tuple(i for i in owners if cfg.rope_layers[i]),
            cfg.kv_sources,
            draft_layers,
        )

    @property
    def _target_rows(self) -> int:
        return (
            len(self.kv_owners) * LATENT_PAGE_ROWS
            + len(self.index_owners) * INDEX_PAGE_ROWS
            + len(self.rope_owners) * ROPE_PAGE_ROWS
        )

    @property
    def block_rows(self) -> int:
        """rows of one KV block, rounded up so that SLAB_MULTIPLE blocks fill whole arena blocks
        (22400 for the full model: no padding)"""
        rows = self._target_rows + self.draft_layers * (
            LATENT_PAGE_ROWS + ROPE_PAGE_ROWS
        )
        unit = ARENA_BLOCK_ROWS // SLAB_MULTIPLE
        return -(-rows // unit) * unit

    @property
    def block_bytes(self) -> int:
        return self.block_rows * ROW_BYTES

    def latent_offset(self, layer: int) -> int:
        """rows from a block's first row to ``layer``'s latent page (its source's, if shared)"""
        return self.kv_owners.index(self.kv_sources[layer]) * LATENT_PAGE_ROWS

    def rope_offset(self, layer: int) -> int:
        """rows to the rotary page of ``layer``'s cache (-1: a NoPE cache)"""
        source = self.kv_sources[layer]
        if source not in self.rope_owners:
            return -1
        base = (
            len(self.kv_owners) * LATENT_PAGE_ROWS
            + len(self.index_owners) * INDEX_PAGE_ROWS
        )
        return base + self.rope_owners.index(source) * ROPE_PAGE_ROWS

    def index_offset(self, layer: int) -> int:
        """rows to the index-key page of an indexer-owning ``layer`` (-1: none)"""
        if layer not in self.index_owners:
            return -1
        base = len(self.kv_owners) * LATENT_PAGE_ROWS
        return base + self.index_owners.index(layer) * INDEX_PAGE_ROWS

    def draft_latent_offset(self, layer: int) -> int:
        """rows from a block's first row to drafter layer ``layer``'s latent page"""
        assert 0 <= layer < self.draft_layers
        return self._target_rows + layer * LATENT_PAGE_ROWS

    def draft_rope_offset(self, layer: int) -> int:
        """rows to drafter layer ``layer``'s rotary page"""
        assert 0 <= layer < self.draft_layers
        base = self._target_rows + self.draft_layers * LATENT_PAGE_ROWS
        return base + layer * ROPE_PAGE_ROWS

    def draft_store_rows(
        self, layer: int, block_rows: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """``store_rows`` for drafter layer ``layer``: its latent, rotary and scale entries"""
        T = len(positions)
        out = np.full((T, STORE_FIELDS), -1, np.int32)
        b = np.asarray(block_rows, np.int64)
        r = np.asarray(positions, np.int64) % PAGE_TOKENS
        live = b >= 0
        lat = b[live] + self.draft_latent_offset(layer)
        out[live, 0] = lat + r[live]
        out[live, 3] = lat + PAGE_TOKENS
        out[live, 4] = r[live]
        out[live, 1] = 4 * (b[live] + self.draft_rope_offset(layer)) + r[live]
        return out

    def layout(self, parent_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity=parent_capacity,
            groups=[CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=1)],
        )

    @staticmethod
    def capacity_for(free_bytes: int, block_bytes: int) -> int:
        """the largest parent capacity whose arena fits ``free_bytes`` (slab 0 is null)"""
        slabs = free_bytes // block_bytes // SLAB_MULTIPLE * SLAB_MULTIPLE
        return max(0, slabs - 1)

    def arena_blocks(self, parent_capacity: int) -> int:
        """the arena's dim 2 (1024-row blocks): ``parent_capacity`` slabs plus the null slab"""
        slabs = parent_capacity + 1
        if slabs % SLAB_MULTIPLE:
            raise ValueError(f"{slabs} slabs: not a multiple of {SLAB_MULTIPLE}")
        rows = slabs * self.block_rows
        assert rows % ARENA_BLOCK_ROWS == 0 and rows < 2**31
        return rows // ARENA_BLOCK_ROWS

    # ---- per-token destinations (arena rows / half-row units)
    def block_row(self, slabs) -> np.ndarray:
        """first rows of KV blocks (one block per slab)"""
        return np.asarray(slabs, np.int64) * self.block_rows

    def store_rows(
        self, layer: int, block_rows: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """(T, 8) i32 mla_store destinations of tokens at ``positions`` whose blocks start at
        ``block_rows`` (-1 entries skip): [latent row, rotary unit (128 B), index unit (256 B),
        scale row, the scale's index in it, -1 ...]. Layers reading another layer's KV cache store no latent or
        rotary key (they may still own an indexer: layer 39 does)."""
        T = len(positions)
        out = np.full((T, STORE_FIELDS), -1, np.int32)
        b, r = (
            np.asarray(block_rows, np.int64),
            np.asarray(positions, np.int64) % PAGE_TOKENS,
        )
        live = b >= 0
        if self.kv_sources[layer] == layer:
            lat = b[live] + self.latent_offset(layer)
            out[live, 0] = lat + r[live]
            out[live, 3] = lat + PAGE_TOKENS
            out[live, 4] = r[live]
            if (ro := self.rope_offset(layer)) >= 0:
                out[live, 1] = 4 * (b[live] + ro) + r[live]
        if (io := self.index_offset(layer)) >= 0:
            out[live, 2] = 2 * (b[live] + io) + r[live]
        return out
