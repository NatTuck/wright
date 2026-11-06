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
(1024 values) in a separate state buffer: one row per (WINDOW block, ratio-2 owner), in the block of the
pending token's position.

The Engine's prefix cache shares blocks at multiples of 256 tokens. There every compressor group is complete,
so a sequence that starts from cached blocks needs from them only its keys and entries. The tokens that Engram
hashes before its first position are on the host (``EngramContext``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from resin import CacheGroupLayout, CacheLayout, CacheRetention

PAGE_TOKENS = 128
HISTORY_TOKENS = 256
ROW_BYTES = 512
ARENA_BLOCK_ROWS = 1024
PAGE_ROWS = 129
INDEX_ROWS = 33
PAIR_ROWS = PAGE_ROWS + INDEX_ROWS
BLOCK_UNIT = 128
WINDOW_GROUP, HISTORY_GROUP = 0, 1
STATE_FLOATS = 1024


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
    def slab_rows(self) -> int:
        return max(self.window_block_rows, self.history_block_rows)

    @property
    def slab_bytes(self) -> int:
        return self.slab_rows * ROW_BYTES

    @property
    def pending_owners(self) -> tuple[int, ...]:
        """owners with a pending state (ratio 2)"""
        return tuple(o for o, r in zip(self.owners, self.ratios, strict=True) if r > 1)

    def state_bytes(self, parent_capacity: int) -> int:
        return (
            (parent_capacity + 1) * max(len(self.pending_owners), 1) * STATE_FLOATS * 4
        )

    def layout(self, parent_capacity: int) -> CacheLayout:
        read = self.window
        return CacheLayout(
            parent_capacity=parent_capacity,
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
        per = self.slab_bytes + max(len(self.pending_owners), 1) * STATE_FLOATS * 4
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

    def state_row(self, block_id, owner: int):
        """the pending-state row of a ratio-2 owner in a WINDOW block"""
        n = len(self.pending_owners)
        return np.asarray(block_id, np.int64) * n + self.pending_owners.index(owner)

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


_NONE = np.zeros(0, np.int64)
logger = logging.getLogger("wright")


class EngramContext:
    """The ``n`` tokens before each position that Engram hashes, for every sequence of the runner.

    ``seen`` records the tokens of a sequence as they are committed, from its first token. It also records the
    last ``n`` tokens of each WINDOW block that they complete, by window block id. A sequence that the Engine
    starts from cached blocks (a prefix hit, at a block boundary) has no record of its own. ``before`` then takes
    its tokens from the record of the block before its first position, which is in its WINDOW table.

    The Engine does not change a shared block, and the new owner of a free block writes it again before the
    block can be shared. Thus the record of a block that a plan names is the record of that block's tokens."""

    def __init__(self, n: int, window_blocks: int) -> None:
        if n < 1:
            raise ValueError("Engram hashes at least one token before a position")
        self.n = n
        self.last: dict[int, np.ndarray] = {}
        """sequence id -> its last ``n`` committed tokens"""
        self._windows: dict[int, tuple[int, np.ndarray]] = {}
        """sequence id -> its WINDOW table of the latest plan: first logical block and the window block ids"""
        self._tails = np.full((window_blocks, n), -1, np.int64)
        """window block id -> the last ``n`` tokens of the block (-1: not recorded)"""

    def before(
        self, sequence: int, pos0: int, first: int, ids: np.ndarray
    ) -> np.ndarray:
        """The tokens before ``pos0`` that Engram hashes (those that exist), for a row whose WINDOW table starts
        at logical block ``first`` with window block ids ``ids``."""
        ids = np.asarray(ids, np.int64)
        self._windows[sequence] = (first, ids)
        if pos0 == 0:
            return _NONE
        last = self.last.get(sequence)
        if last is None:
            last = self._restore(sequence, pos0, first, ids)
        return last[-min(self.n, pos0) :]

    def _restore(
        self, sequence: int, pos0: int, first: int, ids: np.ndarray
    ) -> np.ndarray:
        block = pos0 // PAGE_TOKENS - 1 - first
        if pos0 % PAGE_TOKENS or not 0 <= block < len(ids):
            raise RuntimeError(
                f"sequence {sequence} starts at position {pos0}, which is not after a block in its WINDOW table"
            )
        tail = self._tails[ids[block]]
        if (tail < 0).any():
            raise RuntimeError(
                f"sequence {sequence} starts at position {pos0} after a block with no recorded tokens"
            )
        self.last[sequence] = tail = tail[-min(self.n, pos0) :].copy()
        logger.debug(
            f"sequence {sequence} starts from cached blocks at position {pos0} after tokens {tail.tolist()}"
        )
        return tail

    def seen(self, sequence: int, pos0: int, tokens) -> None:
        """Record ``tokens``, committed at positions ``pos0`` and after. Call after ``before`` for a row of this
        sequence whose WINDOW table covers those positions."""
        last = self.last.get(sequence, _NONE) if pos0 else _NONE
        full = np.concatenate([last, np.asarray(tokens, np.int64)])
        start = pos0 - len(last)  # the position of full[0]
        window = self._windows.get(sequence)
        if window is not None:
            first, ids = window
            end = start + len(full)
            for e in range(
                -(-(pos0 + 1) // PAGE_TOKENS) * PAGE_TOKENS, end + 1, PAGE_TOKENS
            ):
                block = e // PAGE_TOKENS - 1 - first
                if e - self.n >= start and 0 <= block < len(ids):
                    self._tails[ids[block]] = full[e - self.n - start : e - start]
        self.last[sequence] = full[-self.n :]

    def release(self, sequences) -> None:
        """Forget released sequences. The records of their blocks stay: cached blocks outlive their sequence."""
        for sequence in sequences:
            self.last.pop(int(sequence), None)
            self._windows.pop(int(sequence), None)
