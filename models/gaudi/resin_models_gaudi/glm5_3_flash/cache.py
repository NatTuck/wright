"""Cache geometry of GLM-5.3-Flash on Gaudi: the Engine's paged KV and STATE groups in one arena of
512-byte rows. Every head reads the shared latent, so KV is replicated on every tensor-parallel rank;
recurrent state is per head, so it scales with the rank's heads.

The arena holds ``parent_capacity + 1`` slabs (slab 0 is null). A slab packs KV or STATE blocks:

- a KV block (128 tokens) holds, for each sparse-attention layer in order (``layer_rows`` apart):
  - a latent page of 129 rows: token r's 512-dim latent as Gaudi2 FP8 in row r, and row 128 holding
    the 128 tokens' f32 scales (``mla_store``, as Mistral-Large-4);
  - an index page of 16 rows: pool P's 128 bf16 in 256-byte unit P (two per row, 32 pools per block);
  - a tail of 6 rows per ``tail_pools``: the raw f32 [k | gate] (2 rows) of tokens 0, 1, 2 of a pool, for
    ``tail_pools`` consecutive pools (position modulo 4 ``tail_pools``): a verify step stores its drafts' tails,
    and a rejected draft in a later pool must not overwrite what this pool's restart reads. Two pools cover up
    to 5 drafts a step, three up to 9. A pool never crosses a block, and prefixes publish on block boundaries,
    where no pool is open;
- a STATE block (``STATE_TOKENS``) holds each KDA layer's f32 recurrent state (128 rows per local head) and
  conv state at a stride of ``state_layer_rows``, then the f32 final residual of the block's last position
  (the MTP layer's input in a continuing prefill);
- with MTP, an MTP KV block (128 tokens) holds the MTP layer's pages, laid out as one layer of a KV block;
- with a DFlash 2 drafter (``glm5_3_flash.dflash``), a DRAFT block (128 tokens) holds the drafter's layers'
  K and V pages (``draft_layer_rows`` rows per layer), retained for the last ``draft_window`` positions plus a
  step's writes (the Engine's WINDOW retention). It takes the MTP group's index: MTP and the drafter are
  alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout, CacheRetention
from resin.models.glm5_3_flash.config import Glm53Config

PAGE_TOKENS = 128
STATE_TOKENS = 4096
ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024
LATENT_PAGE_ROWS = 129
INDEX_PAGE_ROWS = 16
TAIL_POOLS = 2  # pools whose tails a layer keeps apart (``tail_pools_for``)
TAIL_ROWS = 6 * TAIL_POOLS
LAYER_ROWS = LATENT_PAGE_ROWS + INDEX_PAGE_ROWS + TAIL_ROWS  # with the default tails
POOL = 4
KV_GROUP = 0
STATE_GROUP = 1
MTP_GROUP = 2
DRAFT_GROUP = 2  # the DFlash drafter's cache takes the MTP group's place
BLOCK_UNIT = 128  # KV blocks round up to this many rows


def tail_pools_for(draft_tokens: int) -> int:
    """The tail pools a speculative step of ``draft_tokens`` drafts needs: a rejected draft up to 3 pools past
    the anchor's must not overwrite the tails its restart reads (``tail_rows``)."""
    return 2 if draft_tokens <= 5 else 3


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    dsa_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    ssm_rows: int
    conv_rows: int
    hidden_rows: int
    mtp: bool = False
    tail_pools: int = TAIL_POOLS
    draft_layers: int = 0  # DFlash drafter layers in a DRAFT block (0: none)
    draft_layer_rows: int = 0  # rows of one drafter layer's pages
    draft_window: int = 0  # positions the drafter reads
    draft_write: int = 0  # the widest step the drafter's cache takes (prefill chunk)
    shared_slab_rows: int = 0  # the slab of a pool that other models share (0: this model's own)

    @classmethod
    def from_config(
        cls,
        cfg: Glm53Config,
        heads: int,
        *,
        layers: int | None = None,
        mtp: bool = False,
        tail_pools: int = TAIL_POOLS,
        draft_layers: int = 0,
        draft_layer_rows: int = 0,
        draft_window: int = 0,
        draft_write: int = 0,
    ) -> CacheGeometry:
        if mtp and draft_layers:
            raise ValueError("MTP and a DFlash drafter are alternatives")
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
            tail_pools,
            draft_layers,
            draft_layer_rows,
            draft_window,
            draft_write,
        )

    @property
    def layer_rows(self) -> int:
        """rows of one DSA layer's pages in a KV block: latent, index, tails"""
        return LATENT_PAGE_ROWS + INDEX_PAGE_ROWS + 6 * self.tail_pools

    @property
    def kv_block_rows(self) -> int:
        rows = max(len(self.dsa_layers), 1) * self.layer_rows
        return -(-rows // BLOCK_UNIT) * BLOCK_UNIT

    @property
    def mtp_block_rows(self) -> int:
        return -(-self.layer_rows // BLOCK_UNIT) * BLOCK_UNIT

    @property
    def draft_block_rows(self) -> int:
        return self.draft_layers * self.draft_layer_rows

    @property
    def draft_blocks_per_slab(self) -> int:
        return self.slab_rows // self.draft_block_rows

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
    def slab_unit(self) -> int:
        """a slab is a multiple of this many rows: whole KV blocks and whole 1024-row arena blocks"""
        return int(np.lcm(self.kv_block_rows, ARENA_BLOCK_ROWS))

    @property
    def own_slab_rows(self) -> int:
        """whole KV blocks covering one STATE block, and whole 1024-row arena blocks"""
        unit = self.slab_unit
        return -(-self.state_block_rows // unit) * unit

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
        if self.draft_layers:
            if self.draft_blocks_per_slab < 1:
                raise ValueError("a DRAFT block does not fit a slab")
            groups.append(
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.draft_blocks_per_slab,
                    retention=CacheRetention.WINDOW,
                    read_window_tokens=self.draft_window,
                    window_tokens=self.draft_window + self.draft_write - 1,
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

    def draft_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.draft_block_rows
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
        return self.dsa_layers.index(layer) * self.layer_rows


def index_rows(block_row: int) -> int:
    """a layer's index page, from its latent page's first row"""
    return block_row + LATENT_PAGE_ROWS


def tail_rows(position, pools: int = TAIL_POOLS):
    """rows into the tail of a pool's token at ``position`` (not the pool's last; 2 rows each): ``pools``
    consecutive pools never share them. Two pools cover up to 5 drafts a step, three up to 9
    (``tail_pools_for``)."""
    p = np.asarray(position) % (pools * POOL)
    return 2 * (p - p // POOL)
