"""Cache geometry of Qwen3.5/3.8 on Gaudi: the Engine's paged KV and STATE groups laid out in
one arena of 512-byte rows.

The arena holds ``parent_capacity + 1`` slabs (slab 0 is the null slab). A slab packs either
KV blocks or STATE blocks:

- a KV block (128 tokens) holds every full-attention layer's pages; layer a's K page of KV
  group g starts at row ``a * 2G * 128 + g * 128`` of the block and its V page at
  ``a * 2G * 128 + (G + g) * 128``. A page is 128 token-major rows (V rows in sigma_inv order).
- a STATE block (4096 tokens) holds every GDN layer's f32 recurrent state (128 * nv rows) and
  f32 conv state (conv_dim * 3 / 128 rows) at a stride of ``state_layer_rows``, then the f32
  final residual of the block's last position (the hidden the MTP layer of a continuing prefill
  reads). The stride is a multiple of 256 rows.
- with MTP, an MTP KV block (128 tokens) holds the MTP layer's pages, laid out as one layer of
  a KV block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.qwen3_5.layers import ARENA_BLOCK_ROWS, Dims

PAGE_TOKENS = 128
STATE_TOKENS = 4096
ROW_BYTES = 512
KV_GROUP = 0
STATE_GROUP = 1
MTP_GROUP = 2


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    full_layers: int
    linear_layers: int
    kv_groups: int
    ssm_rows: int
    conv_rows: int
    hidden_rows: int
    mtp: bool = False

    @classmethod
    def from_config(cls, config: Qwen35Config, *, mtp: bool = False) -> CacheGeometry:
        d = Dims.from_config(config)
        return cls(
            config.full_layer_count,
            config.linear_layer_count,
            d.kv_groups,
            d.ssm_rows,
            d.conv_rows,
            d.hidden * 4 // ROW_BYTES,
            mtp,
        )

    @property
    def kv_layer_rows(self) -> int:
        return 2 * self.kv_groups * PAGE_TOKENS

    @property
    def kv_block_rows(self) -> int:
        return self.full_layers * self.kv_layer_rows

    @property
    def state_layer_rows(self) -> int:
        return -(-(self.ssm_rows + self.conv_rows) // 256) * 256

    @property
    def state_block_rows(self) -> int:
        return (
            self.linear_layers * self.state_layer_rows
            + -(-self.hidden_rows // 256) * 256
        )

    @property
    def slab_rows(self) -> int:
        blocks = -(-self.state_block_rows // self.kv_block_rows)
        return blocks * self.kv_block_rows

    @property
    def slab_bytes(self) -> int:
        return self.slab_rows * ROW_BYTES

    @property
    def kv_blocks_per_slab(self) -> int:
        return self.slab_rows // self.kv_block_rows

    @property
    def state_blocks_per_slab(self) -> int:
        return self.slab_rows // self.state_block_rows

    @property
    def mtp_blocks_per_slab(self) -> int:
        return self.slab_rows // self.kv_layer_rows

    def layout(self, parent_capacity: int) -> CacheLayout:
        groups = [
            CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.kv_blocks_per_slab),
            CacheGroupLayout(
                STATE_TOKENS,
                blocks_per_slab=self.state_blocks_per_slab,
                family=CacheGroupFamily.STATE,
            ),
        ]
        if self.mtp:
            groups.append(
                CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.mtp_blocks_per_slab)
            )
        return CacheLayout(parent_capacity=parent_capacity, groups=groups)

    def arena_blocks(self, parent_capacity: int) -> int:
        """The arena's dim 2 (1024-row blocks): ``parent_capacity`` slabs plus the null slab."""
        rows = (parent_capacity + 1) * self.slab_rows
        assert rows % ARENA_BLOCK_ROWS == 0 and rows < 2**31
        return rows // ARENA_BLOCK_ROWS

    def attention_blocks(self, parent_capacity: int) -> int:
        """The arena's dim 2 as full-attention layers declare it: layer a binds the arena
        ``a * kv_layer_rows`` rows in."""
        shift = -(-(self.full_layers - 1) * self.kv_layer_rows // ARENA_BLOCK_ROWS)
        return self.arena_blocks(parent_capacity) - shift

    # ---- addresses (arena rows)
    def kv_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.kv_block_rows
        )

    def mtp_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.kv_layer_rows
        )

    def state_block_row(self, slab: int, slot: int) -> int:
        return slab * self.slab_rows + slot * self.state_block_rows

    def hidden_row(self, block_row: int) -> int:
        """The first row of a STATE block's hidden slot (-1: none)."""
        if block_row < 0:
            return -1
        return block_row + self.linear_layers * self.state_layer_rows

    def kv_pages(self, block_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Layer 0's (K, V) page rows (G, pages) of a sequence's KV blocks; layer a adds
        ``a * kv_layer_rows``."""
        g = np.arange(self.kv_groups, dtype=np.int64)[:, None] * PAGE_TOKENS
        k = np.asarray(block_rows, np.int64)[None, :] + g
        return k, k + self.kv_groups * PAGE_TOKENS

    def state_rows(self, block_row: int, layer: int) -> tuple[int, int]:
        """(ssm, conv) first rows of the ``layer``-th GDN layer in a STATE block (-1: none)."""
        if block_row < 0:
            return -1, -1
        ssm = block_row + layer * self.state_layer_rows
        return ssm, ssm + self.ssm_rows
