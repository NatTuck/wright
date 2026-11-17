"""Cache geometry of Qwen3.5/3.8 on Gaudi: the Engine's paged KV and STATE groups laid out in
one arena of 512-byte rows.

The arena holds ``parent_capacity + 1`` slabs (slab 0 is the null slab). A slab packs either
KV blocks or STATE blocks:

- a KV block (128 tokens) holds every full-attention layer's pages; layer a's K page of KV
  group g starts at row ``a * G * (Pk + Pv) + g * Pk`` of the block and its V page at
  ``a * G * (Pk + Pv) + G * Pk + g * Pv``, for K pages of Pk rows and V pages of Pv rows. A page
  holds 128 key slots, token-major (V keys in sigma_inv order). A bf16 page is 128 rows, a key a
  row; an FP8 page (``KV_TYPES``) is 65 rows: two keys' 256 FP8 values a row, then a row of the
  128 keys' f32 scales (``kernels/kv_store8.c``).
- a STATE block (4096 tokens) holds every GDN layer's f32 recurrent state (128 * nv rows) and
  f32 conv state (conv_dim * 3 / 128 rows) at a stride of ``state_layer_rows``, then the f32
  final residual of the block's last position (the hidden the MTP layer of a continuing prefill
  reads). The stride is a multiple of 256 rows.
- with MTP, an MTP KV block (128 tokens) holds the MTP layer's pages, laid out as one layer of
  a KV block (of all its KV groups: a tensor-parallel rank keeps the whole MTP layer).
- with a DFlash 2 drafter, a DRAFT block (128 tokens) holds the drafter's five layers' pages (K
  and V pages per KV pair, ``draft_layer_rows`` rows per layer), retained for the last
  ``draft_window`` positions plus a step's writes (the Engine's WINDOW retention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout, CacheRetention
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.common.layers import ARENA_BLOCK_ROWS, Dims

PAGE_TOKENS = 128
STATE_TOKENS = 4096
ROW_BYTES = 512
KV_GROUP = 0
STATE_GROUP = 1
MTP_GROUP = 2
DRAFT_GROUP = 2  # the DFlash drafter's cache takes the MTP group's place
KV_TYPES = ("bf16", "fp8")
"""The types of the K and V pages: ``--cache-type-k`` and ``--cache-type-v``."""
FP8_PAGE_ROWS = 65


def kv8_bits(cache_types: tuple[str, str]) -> int:
    """``cache_types`` (K, V) as the FP8 bits of the kernels: bit 0 the K pages, bit 1 the V pages."""
    for kind in cache_types:
        if kind not in KV_TYPES:
            raise ValueError(f"cache type {kind!r}: one of {', '.join(KV_TYPES)}")
    return int(cache_types[0] == "fp8") | 2 * int(cache_types[1] == "fp8")


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    full_layers: int
    linear_layers: int
    kv_groups: int
    ssm_rows: int
    conv_rows: int
    hidden_rows: int
    mtp: bool = False
    draft_layers: int = 0  # DFlash drafter layers in a DRAFT block (0: none)
    draft_layer_rows: int = 0  # rows of one drafter layer's pages
    draft_window: int = 0  # positions the drafter reads
    draft_write: int = 0  # the widest step the drafter's cache takes (prefill chunk)
    shared_slab_rows: int = 0
    """The slab of a cache pool that other models share (0: this model's own slab). It is at
    least this model's slab; each group packs as many of its blocks as fit."""
    mtp_kv_groups: int = 0
    """The MTP layer's KV groups (0: ``kv_groups``): a tensor-parallel rank keeps the whole MTP layer."""
    mtp_short_view: bool = False
    """The MTP layer takes the target's attention recipes (one card): their arena view, bound at the arena's start,
    ends ``attention_shift_rows`` before the last slab's end, so no MTP block may sit in a slab's last
    ``attention_shift_rows`` rows (``mtp_blocks_per_slab``)."""
    kv8: int = 0
    """FP8 pages: bit 0 the K pages, bit 1 the V pages (``kv8_bits``). The MTP layer's pages are the same."""

    @classmethod
    def from_config(
        cls,
        config: Qwen35Config,
        *,
        mtp: bool = False,
        draft_layers: int = 0,
        draft_layer_rows: int = 0,
        draft_window: int = 0,
        draft_write: int = 0,
        shard=None,
        mtp_kv_groups: int = 0,
        mtp_short_view: bool = False,
        kv8: int = 0,
    ) -> CacheGeometry:
        """``shard``: a tensor-parallel rank's geometry (its KV groups and GDN heads); ``mtp_kv_groups``: the MTP
        layer's KV groups (0: the rank's); ``mtp_short_view``: the MTP layer takes the target's attention recipes."""
        d = Dims.from_config(config)
        if shard is not None:
            d = d.shard(shard)
        if mtp and draft_layers:
            raise ValueError("MTP and a DFlash drafter are alternatives")
        return cls(
            config.full_layer_count,
            config.linear_layer_count,
            d.kv_groups,
            d.ssm_rows,
            d.conv_rows,
            d.hidden * 4 // ROW_BYTES,
            mtp,
            draft_layers,
            draft_layer_rows,
            draft_window,
            draft_write,
            mtp_kv_groups=mtp_kv_groups,
            mtp_short_view=mtp_short_view,
            kv8=kv8,
        )

    @property
    def k_page_rows(self) -> int:
        return FP8_PAGE_ROWS if self.kv8 & 1 else PAGE_TOKENS

    @property
    def v_page_rows(self) -> int:
        return FP8_PAGE_ROWS if self.kv8 & 2 else PAGE_TOKENS

    @property
    def kv_layer_rows(self) -> int:
        return self.kv_groups * (self.k_page_rows + self.v_page_rows)

    @property
    def mtp_layer_rows(self) -> int:
        """Rows of an MTP KV block: the MTP layer's K and V pages."""
        return (self.mtp_kv_groups or self.kv_groups) * (
            self.k_page_rows + self.v_page_rows
        )

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
    def own_slab_rows(self) -> int:
        """The smallest slab of whole KV blocks that holds a STATE block, in whole arena blocks (FP8 pages make a
        KV block that is not a multiple of one)."""
        blocks = -(-self.state_block_rows // self.kv_block_rows)
        rows = blocks * self.kv_block_rows
        return -(-rows // ARENA_BLOCK_ROWS) * ARENA_BLOCK_ROWS

    @property
    def slab_rows(self) -> int:
        if self.shared_slab_rows:
            if self.shared_slab_rows < self.own_slab_rows:
                raise ValueError(
                    f"a shared slab of {self.shared_slab_rows} rows is smaller than this "
                    f"model's {self.own_slab_rows}"
                )
            return self.shared_slab_rows
        return self.own_slab_rows

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
        """MTP KV blocks of a slab: with a short view (``mtp_short_view``), none in a slab's last
        ``attention_shift_rows`` rows, which the last slab's view does not reach."""
        rows = self.slab_rows
        if self.mtp_short_view:
            rows -= self.attention_shift_rows
        return rows // self.mtp_layer_rows

    @property
    def draft_block_rows(self) -> int:
        return self.draft_layers * self.draft_layer_rows

    @property
    def draft_blocks_per_slab(self) -> int:
        return self.slab_rows // self.draft_block_rows

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
        if self.draft_layers:
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

    def arena_blocks(self, parent_capacity: int) -> int:
        """The arena's dim 2 (1024-row blocks): ``parent_capacity`` slabs plus the null slab."""
        rows = (parent_capacity + 1) * self.slab_rows
        assert rows % ARENA_BLOCK_ROWS == 0 and rows < 2**31
        return rows // ARENA_BLOCK_ROWS

    def attention_blocks(self, parent_capacity: int) -> int:
        """The arena's dim 2 as full-attention layers declare it: layer a binds the arena
        ``a * kv_layer_rows`` rows in, and its view holds that layer's rows of every slab. When
        ``kv_layer_rows`` is not a multiple of a block (a tensor-parallel rank's fewer KV groups),
        the last layer's view ends up to a block past the slabs: the arena keeps a block of padding
        after them (``ARENA_PAD_BLOCKS``)."""
        return self.arena_blocks(parent_capacity) - self.attention_shift_rows // ARENA_BLOCK_ROWS

    @property
    def attention_shift_rows(self) -> int:
        """The rows by which a full-attention layer's arena view (``attention_blocks``) is shorter than the arena:
        the last layer's offset, in whole blocks."""
        return (self.full_layers - 1) * self.kv_layer_rows // ARENA_BLOCK_ROWS * ARENA_BLOCK_ROWS

    # ---- addresses (arena rows)
    def kv_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.kv_block_rows
        )

    def mtp_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.mtp_layer_rows
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
        return block_row + self.linear_layers * self.state_layer_rows

    def kv_pages(
        self, block_rows: np.ndarray, kv_groups: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Layer 0's (K, V) page rows (G, pages) of a sequence's KV blocks (or an MTP layer's of
        ``kv_groups`` groups); layer a adds ``a * kv_layer_rows``."""
        G = self.kv_groups if kv_groups is None else kv_groups
        g = np.arange(G, dtype=np.int64)[:, None]
        block = np.asarray(block_rows, np.int64)[None, :]
        k = block + g * self.k_page_rows
        return k, block + G * self.k_page_rows + g * self.v_page_rows

    def state_rows(self, block_row: int, layer: int) -> tuple[int, int]:
        """(ssm, conv) first rows of the ``layer``-th GDN layer in a STATE block (-1: none)."""
        if block_row < 0:
            return -1, -1
        ssm = block_row + layer * self.state_layer_rows
        return ssm, ssm + self.ssm_rows
