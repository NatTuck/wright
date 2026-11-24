"""Cache geometry of DeepSeek-V4.1-Flash on Gaudi: the Engine's paged WINDOW and HISTORY groups in one arena of
512-byte rows, replicated on every tensor-parallel rank (every head reads the same keys).

The arena holds ``parent_capacity + 1`` slabs (slab 0 is null); a slab is one WINDOW block or HISTORY blocks.

- A WINDOW block (128 tokens, retained while a query can see it) holds, for every layer in order, a latent
  page of 129 rows: token r's 512-dim key as Gaudi2 FP8 in row r and the 128 tokens' f32 scales in row 128.
- A HISTORY block (256 tokens) holds, for each KV owner in order, its compressed entries: per 128 entries a
  latent page (129 rows, as above) and an index page (33 rows: four entries' 128 FP8 index keys per row, then a
  row of their f32 power-of-two scales).
  A ratio-2 owner has one such page pair per block, a ratio-1 owner two.

The ratio-2 compressor's pending token (the first of a pair whose second has not run) is f32 [kv | score]
(1024 values) in a separate state buffer: PENDING_SLOTS rows per (WINDOW block, ratio-2 owner), in the block of
the pending token's position, slot (position // 2) % PENDING_SLOTS. A verify step stores the state of every even
column (a rejected draft can end the committed range at any of them), so the slots of the even positions one
step can hold and of the position before it must differ: a step writes at most 2 * PENDING_SLOTS - 2 positions
of a row (``CacheGeometry.max_pending_columns``).

The Engine's prefix cache shares blocks at multiples of 256 tokens. There every compressor group is complete,
so a sequence that starts from cached blocks needs from them only its keys and entries. The tokens that Engram
hashes before its first position are on the host (the native executor's ``Ds41Context``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from resin import CacheGroupLayout, CacheLayout, CacheRetention

PAGE_TOKENS = 128
HISTORY_TOKENS = 256
ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024
PAGE_ROWS = 129  # WINDOW latent page: 128 fp8 rows + 1 f32-scale row (unchanged)
CMP_PAGE_ROWS = 80  # fp4 compressed latent page: 64 code rows (2 entries/row) + 16 scale rows (8/row)
INDEX_ROWS = 17  # fp4 index page: 16 code rows (8 keys/row) + 1 e8m0-scale row
PAIR_ROWS = CMP_PAGE_ROWS + INDEX_ROWS
BLOCK_UNIT = 128
WINDOW_GROUP, HISTORY_GROUP = 0, 1
STATE_FLOATS = 1024
PENDING_SLOTS = 4


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    layers: int
    owners: tuple[int, ...]
    """KV owners (compressor layers) in order"""
    ratios: tuple[int, ...]
    """each owner's compress ratio"""
    window: int
    max_write: int
    """tokens one step may write to a row (its window blocks must all be resident)"""
    draft_layers: int = 0
    """DSpark stages after the backbone, each with a latent page in every WINDOW block"""
    shared_slab_rows: int = 0
    """the rows of a slab that another model's slabs share (0: this model's own, ``own_slab_rows``)"""

    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        layers: int | None = None,
        max_write: int = 128,
        draft_layers: int = 0,
    ) -> CacheGeometry:
        n = cfg.num_hidden_layers if layers is None else layers
        owners = tuple(o for o in cfg.kv_source_layers if o < n)
        return cls(
            n,
            owners,
            tuple(cfg.compress_ratios[o] for o in owners),
            cfg.sliding_window,
            max_write,
            draft_layers,
        )

    # ---- block shapes (rows)
    @property
    def window_block_rows(self) -> int:
        return (
            -(-(self.layers + self.draft_layers) * PAGE_ROWS // BLOCK_UNIT) * BLOCK_UNIT
        )

    def owner_pages(self, owner: int) -> int:
        """page pairs per HISTORY block"""
        return HISTORY_TOKENS // self.ratios[self.owners.index(owner)] // PAGE_TOKENS

    def owner_offset(self, owner: int) -> int:
        """rows from a HISTORY block's first row to the owner's first latent page"""
        k = self.owners.index(owner)
        return sum(PAIR_ROWS * self.owner_pages(o) for o in self.owners[:k])

    @property
    def history_block_rows(self) -> int:
        rows = sum(PAIR_ROWS * self.owner_pages(o) for o in self.owners)
        return max(BLOCK_UNIT, -(-rows // BLOCK_UNIT) * BLOCK_UNIT)

    @property
    def own_slab_rows(self) -> int:
        """the smallest slab: one WINDOW block or the HISTORY blocks that fit beside it"""
        return max(self.window_block_rows, self.history_block_rows)

    @property
    def slab_rows(self) -> int:
        return self.shared_slab_rows or self.own_slab_rows

    def with_slab_rows(self, rows: int) -> CacheGeometry:
        """The geometry in slabs of ``rows`` rows, which a slab of another model sets: each holds as many WINDOW
        or HISTORY blocks as fit."""
        if rows < self.own_slab_rows or rows % BLOCK_UNIT:
            raise ValueError(
                f"a shared slab of {rows} rows: at least {self.own_slab_rows}, in units of {BLOCK_UNIT}"
            )
        return replace(self, shared_slab_rows=rows)

    @property
    def slab_bytes(self) -> int:
        return self.slab_rows * ROW_BYTES

    @property
    def pending_owners(self) -> tuple[int, ...]:
        """owners with a pending state (ratio 2)"""
        return tuple(o for o, r in zip(self.owners, self.ratios, strict=True) if r > 1)

    @property
    def state_rows_per_block(self) -> int:
        """pending-state rows of one WINDOW block: PENDING_SLOTS per ratio-2 owner (at least one row)"""
        return max(len(self.pending_owners), 1) * PENDING_SLOTS

    @property
    def max_pending_columns(self) -> int:
        """the most positions of one row a decode or verify step may write: the slots of its even positions and of
        the position before it must differ"""
        return 2 * PENDING_SLOTS - 2

    def state_bytes(self, parent_capacity: int) -> int:
        return (
            self.window_blocks(parent_capacity)
            * self.state_rows_per_block
            * STATE_FLOATS
            * 4
        )

    def layout(self, parent_capacity: int, replay: int = 0) -> CacheLayout:
        """``replay``: bounded replay's kept span; a prefix hit then leaves at least that many prompt tokens to
        compute (the layers after the cut have no windows before it)."""
        read = self.window
        return CacheLayout(
            parent_capacity=parent_capacity,
            replay_suffix_tokens=replay,
            groups=[
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.slab_rows // self.window_block_rows,
                    retention=CacheRetention.WINDOW,
                    read_window_tokens=read,
                    window_tokens=read + self.max_write - 1,
                ),
                CacheGroupLayout(
                    HISTORY_TOKENS,
                    blocks_per_slab=self.slab_rows // self.history_block_rows,
                ),
            ],
        )

    def capacity_for(self, free_bytes: int) -> int:
        """the largest parent capacity whose arena and state fit ``free_bytes`` (slab 0 is null)"""
        per = (
            self.slab_bytes
            + (self.slab_rows // self.window_block_rows)
            * self.state_rows_per_block
            * STATE_FLOATS
            * 4
        )
        return max(0, free_bytes // per - 2)

    def arena_blocks(self, parent_capacity: int) -> int:
        """the arena's dim 2 (1024-row blocks) for ``parent_capacity`` slabs plus the null slab"""
        rows = (parent_capacity + 1) * self.slab_rows
        return -(-rows // ARENA_BLOCK_ROWS)

    # ---- addresses
    def window_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.window_block_rows
        )

    def history_block_row(self, slabs, slots) -> np.ndarray:
        return (
            np.asarray(slabs, np.int64) * self.slab_rows
            + np.asarray(slots, np.int64) * self.history_block_rows
        )

    def window_blocks(self, parent_capacity: int) -> int:
        """the number of WINDOW block ids (``window_block_id``) of ``parent_capacity`` slabs and the null slab"""
        return (parent_capacity + 1) * (self.slab_rows // self.window_block_rows)

    def window_block_id(self, slabs, slots) -> np.ndarray:
        """a WINDOW block's index among every slab's window blocks (its state rows)"""
        per = self.slab_rows // self.window_block_rows
        return np.asarray(slabs, np.int64) * per + np.asarray(slots, np.int64)

    def layer_offset(self, layer: int) -> int:
        """rows from a WINDOW block's first row to the layer's latent page (DSpark stage k: layer layers + k)"""
        return layer * PAGE_ROWS

    def state_row(self, block_id, owner: int, position):
        """the pending-state row of a ratio-2 owner for the token at ``position`` in WINDOW block ``block_id``"""
        return self.pending_row(block_id, self.pending_owners.index(owner), position)

    def pending_row(self, block_id, index: int, position):
        """``state_row`` with the owner's index among ``pending_owners``"""
        n = len(self.pending_owners)
        slot = np.asarray(position, np.int64) // 2 % PENDING_SLOTS
        return (np.asarray(block_id, np.int64) * n + index) * PENDING_SLOTS + slot

    def entry_pages(self, owner: int, history_rows: np.ndarray) -> np.ndarray:
        """the first rows of an owner's latent pages (128 entries each) in entry order, from its row's HISTORY
        block rows in token order"""
        pages = self.owner_pages(owner)
        rows = (
            np.asarray(history_rows, np.int64)[:, None]
            + self.owner_offset(owner)
            + PAIR_ROWS * np.arange(pages)
        )
        return rows.reshape(-1)




