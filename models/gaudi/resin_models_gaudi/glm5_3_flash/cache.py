"""Cache geometry of GLM-5.3-Flash on Gaudi: the Engine's paged KV and STATE groups in one arena of
512-byte rows. Every head reads the shared latent, so KV is replicated on every tensor-parallel rank;
recurrent state is per head, so it scales with the rank's heads.

The arena holds ``parent_capacity + 1`` slabs (slab 0 is null). A slab packs KV or STATE blocks:

- a KV block (128 tokens) holds, for each sparse-attention layer in order (``layer_rows`` apart):
  - a latent page of 129 rows: token r's 512-dim latent as Gaudi2 FP8 in row r, and row 128 holding
    the 128 tokens' f32 scales (``mla_store``, as Mistral-Large-4);
  - an index page of 16 rows: pool P's 128 bf16 in 256-byte unit P (two per row, 32 pools per block);
  - a tail of 12 rows: the raw f32 [k | gate] (2 rows) of tokens 0, 1, 2 of a pool, for two consecutive pools
    (position modulo 8): a verify step stores its drafts' tails, and a rejected draft in the next pool must not
    overwrite what this pool's restart reads. A pool never crosses a block, and prefixes publish on block
    boundaries, where no pool is open;
- a STATE block (``STATE_TOKENS``) holds each KDA layer's f32 recurrent state (128 rows per local head) and
  conv state at a stride of ``state_layer_rows``, then the f32 final residual of the block's last position
  (the MTP layer's input in a continuing prefill);
- with MTP, an MTP KV block (128 tokens) holds the MTP layer's pages, laid out as one layer of a KV block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.models.glm5_3_flash.config import Glm53Config

PAGE_TOKENS = 128
STATE_TOKENS = 4096
ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024
LATENT_PAGE_ROWS = 129
INDEX_PAGE_ROWS = 16
TAIL_ROWS = 12
LAYER_ROWS = LATENT_PAGE_ROWS + INDEX_PAGE_ROWS + TAIL_ROWS
POOL = 4
KV_GROUP = 0
STATE_GROUP = 1
MTP_GROUP = 2
BLOCK_UNIT = 128  # KV blocks round up to this many rows


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    dsa_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    ssm_rows: int
    conv_rows: int
    hidden_rows: int
    mtp: bool = False

    @classmethod
    def from_config(
        cls, cfg: Glm53Config, heads: int, *, layers: int | None = None, mtp: bool = False
    ) -> CacheGeometry:
        n = cfg.num_hidden_layers if layers is None else layers
        kinds = cfg.layer_types[:n]
        return cls(
            tuple(i for i, k in enumerate(kinds) if k == "deepseek_sparse_attention"),
            tuple(i for i, k in enumerate(kinds) if k == "linear_attention"),
            128 * heads,
            # 3 C * 3 / 128 = 9 heads rows, padded
            -(-9 * heads // 16) * 16,
            cfg.hidden_size * 4 // ROW_BYTES,
            mtp,
        )

    @property
    def kv_block_rows(self) -> int:
        rows = max(len(self.dsa_layers), 1) * LAYER_ROWS
        return -(-rows // BLOCK_UNIT) * BLOCK_UNIT

    @property
    def mtp_block_rows(self) -> int:
        return -(-LAYER_ROWS // BLOCK_UNIT) * BLOCK_UNIT

    @property
    def state_layer_rows(self) -> int:
        return -(-(self.ssm_rows + self.conv_rows) // 256) * 256

    @property
    def state_block_rows(self) -> int:
        return (
            max(len(self.kda_layers), 1) * self.state_layer_rows
            + -(-self.hidden_rows // 256) * 256
        )

    @property
    def slab_rows(self) -> int:
        """whole KV blocks covering one STATE block, and whole 1024-row arena blocks"""
        unit = np.lcm(self.kv_block_rows, ARENA_BLOCK_ROWS)
        return int(-(-self.state_block_rows // unit) * unit)

    @property
    def slab_bytes(self) -> int:
        return self.slab_rows * ROW_BYTES

    def layout(self, parent_capacity: int) -> CacheLayout:
        groups = [
            CacheGroupLayout(
                PAGE_TOKENS, blocks_per_slab=self.slab_rows // self.kv_block_rows
            ),
            CacheGroupLayout(
                STATE_TOKENS,
                blocks_per_slab=self.slab_rows // self.state_block_rows,
                family=CacheGroupFamily.STATE,
            ),
        ]
        if self.mtp:
            groups.append(
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.slab_rows // self.mtp_block_rows
                )
            )
        return CacheLayout(parent_capacity=parent_capacity, groups=groups)

    @staticmethod
    def capacity_for(free_bytes: int, slab_bytes: int) -> int:
        """the largest parent capacity whose arena fits ``free_bytes`` (slab 0 is null)"""
        return max(0, free_bytes // slab_bytes - 1)

    def arena_blocks(self, parent_capacity: int) -> int:
        """The arena's dim 2 (1024-row blocks): ``parent_capacity`` slabs plus the null slab."""
        rows = (parent_capacity + 1) * self.slab_rows
        assert rows % ARENA_BLOCK_ROWS == 0 and rows < 2**31
        return rows // ARENA_BLOCK_ROWS

    # ---- addresses (arena rows)
    def kv_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.kv_block_rows
        )

    def mtp_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.mtp_block_rows
        )

    def state_block_row(self, slab: int, slot: int) -> int:
        return slab * self.slab_rows + slot * self.state_block_rows

    def hidden_row(self, block_row: int) -> int:
        """The first row of a STATE block's hidden slot (-1: none)."""
        if block_row < 0:
            return -1
        return block_row + len(self.kda_layers) * self.state_layer_rows

    def state_rows(self, block_row: int, nth: int) -> tuple[int, int]:
        """(ssm, conv) first rows of the ``nth`` KDA layer in a STATE block (-1: none)."""
        if block_row < 0:
            return -1, -1
        ssm = block_row + nth * self.state_layer_rows
        return ssm, ssm + self.ssm_rows

    def layer_offset(self, layer: int) -> int:
        """rows from a KV block's first row to DSA layer ``layer``'s pages"""
        return self.dsa_layers.index(layer) * LAYER_ROWS


def index_rows(block_row: int) -> int:
    """a layer's index page, from its latent page's first row"""
    return block_row + LATENT_PAGE_ROWS


def tail_rows(position):
    """rows into the tail of a pool's token at ``position`` (not the pool's last; 2 rows each): two consecutive
    pools never share them. Up to K = 5 drafts stay inside the next pool's rows."""
    p = np.asarray(position) % (2 * POOL)
    return 2 * (p - (p >= POOL))
