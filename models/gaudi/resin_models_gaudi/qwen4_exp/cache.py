"""Cache geometry of Qwen3.8-Flash-Next on Gaudi: the Engine's paged KV and STATE groups in one arena of 512-byte
rows (``parent_capacity + 1`` slabs; slab 0 is the null slab).

- a KV block (128 tokens) holds every attention layer's pages, ``kv_layer_rows`` apart: K pages (one per KV group
  held, 128 token-major bf16 rows of 256), V pages (sigma_inv order within the page), the QSA indexer's raw-key page
  (128 f32 rows: a token's 128-dim indexer key) and its block-key page (128 f32 rows: the key of the token's block
  of 4, duplicated over the block's 4 rows, as the scoring GEMM and the kernels read them);
- a STATE block (4096 tokens) holds every delta-net layer's f32 recurrent state (128 nv rows) and conv state
  (C * 3 / 128 rows) at a stride of ``state_layer_rows`` (a multiple of 256: the kernels take a state base that is
  a multiple of 128), then the PLE conv state (4H x 9 f32).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin_models_gaudi.common.layers import ARENA_BLOCK_ROWS
from resin_models_gaudi.qwen4_exp.layers import NS, Dims

PAGE_TOKENS = 128
STATE_TOKENS = 4096
ROW_BYTES = 512
KV_GROUP = 0
STATE_GROUP = 1
POOL = 4  # QSA keys per block


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    full_layers: int
    linear_layers: int
    kv_groups: int  # KV groups held by this rank
    ssm_rows: int
    conv_rows: int
    ple_rows: int  # the PLE conv state (0: no PLE layer)
    shared_slab_rows: int = 0  # the slab of a pool that other models share (0: this model's own)

    @classmethod
    def from_dims(
        cls, d: Dims, *, full_layers: int, linear_layers: int, ple: bool
    ) -> CacheGeometry:
        return cls(
            full_layers,
            linear_layers,
            d.kv_groups,
            d.ssm_rows,
            d.conv_rows,
            d.ple_rows if ple else 0,
        )

    # ---- KV blocks
    @property
    def kv_layer_rows(self) -> int:
        """K pages, V pages, the raw-key page and the block-key page"""
        return (2 * self.kv_groups + 2) * PAGE_TOKENS

    @property
    def kv_block_rows(self) -> int:
        return max(self.full_layers, 1) * self.kv_layer_rows

    # ---- STATE blocks
    @property
    def state_layer_rows(self) -> int:
        return -(-(self.ssm_rows + self.conv_rows) // 256) * 256

    @property
    def state_block_rows(self) -> int:
        return (
            self.linear_layers * self.state_layer_rows + -(-self.ple_rows // 256) * 256
        )

    # ---- slabs
    @property
    def slab_unit(self) -> int:
        """a slab is a multiple of this many rows: whole KV blocks and whole 1024-row arena blocks"""
        return int(np.lcm(self.kv_block_rows, ARENA_BLOCK_ROWS))

    @property
    def own_slab_rows(self) -> int:
        """whole KV blocks covering one STATE block, rounded to whole arena blocks"""
        blocks = -(-self.state_block_rows // self.kv_block_rows)
        unit = self.slab_unit
        return -(-blocks * self.kv_block_rows // unit) * unit

    @property
    def slab_rows(self) -> int:
        return self.shared_slab_rows or self.own_slab_rows

    def with_slab_rows(self, rows: int) -> CacheGeometry:
        """The geometry in slabs of ``rows`` rows, which a slab of another model sets: each holds as many KV
        blocks, and as many STATE blocks, as fit."""
        if rows < self.own_slab_rows or rows % self.slab_unit:
            raise ValueError(
                f"a shared slab of {rows} rows: at least {self.own_slab_rows}, in units of {self.slab_unit}"
            )
        return replace(self, shared_slab_rows=rows)

    @property
    def slab_bytes(self) -> int:
        return self.slab_rows * ROW_BYTES

    @property
    def kv_blocks_per_slab(self) -> int:
        return self.slab_rows // self.kv_block_rows

    @property
    def state_blocks_per_slab(self) -> int:
        return self.slab_rows // self.state_block_rows

    def layout(self, parent_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity=parent_capacity,
            groups=[
                CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.kv_blocks_per_slab),
                CacheGroupLayout(
                    STATE_TOKENS,
                    blocks_per_slab=self.state_blocks_per_slab,
                    family=CacheGroupFamily.STATE,
                ),
            ],
        )

    def arena_blocks(self, parent_capacity: int) -> int:
        """the arena's dim 2 (1024-row blocks): ``parent_capacity`` slabs plus the null slab"""
        rows = (parent_capacity + 1) * self.slab_rows
        assert rows % ARENA_BLOCK_ROWS == 0 and rows < 2**31
        return rows // ARENA_BLOCK_ROWS

    # ---- addresses (arena rows)
    def kv_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.kv_block_rows
        )

    def state_block_row(self, slab: int, slot: int) -> int:
        return slab * self.slab_rows + slot * self.state_block_rows

    def kv_pages(
        self, block_rows: np.ndarray, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """the ``layer``-th attention layer's (K, V) page rows (G, pages) of a sequence's KV blocks"""
        base = np.asarray(block_rows, np.int64)[None, :] + layer * self.kv_layer_rows
        g = np.arange(self.kv_groups, dtype=np.int64)[:, None] * PAGE_TOKENS
        return base + g, base + self.kv_groups * PAGE_TOKENS + g

    def ix_pages(
        self, block_rows: np.ndarray, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """the ``layer``-th attention layer's (raw-key, block-key) page rows (pages,)"""
        base = (
            np.asarray(block_rows, np.int64)
            + layer * self.kv_layer_rows
            + 2 * self.kv_groups * PAGE_TOKENS
        )
        return base, base + PAGE_TOKENS

    def state_rows(self, block_row: int, layer: int) -> tuple[int, int]:
        """(ssm, conv) first rows of the ``layer``-th delta-net layer in a STATE block (-1: none)"""
        if block_row < 0:
            return -1, -1
        ssm = block_row + layer * self.state_layer_rows
        return ssm, ssm + self.ssm_rows

    def ple_row(self, block_row: int) -> int:
        """the first row of the PLE conv state in a STATE block (-1: none)"""
        if block_row < 0 or not self.ple_rows:
            return -1
        return block_row + self.linear_layers * self.state_layer_rows


def ple_state_rows(hidden: int) -> int:
    return NS * hidden * 9 // 128


__all__ = [
    "KV_GROUP",
    "PAGE_TOKENS",
    "POOL",
    "ROW_BYTES",
    "STATE_GROUP",
    "STATE_TOKENS",
    "CacheGeometry",
    "ple_state_rows",
]
