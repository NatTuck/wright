"""Host side of slot attention: cache rows for a chunk's new K/V and the slot tables of its
attention launches (see ``layers.attn_slots_graph``).

A chunk's columns belong to one or more sequences (rows): a prefill chunk is one sequence's
tokens, a decode batch one token of each sequence. A row's K and V history for one KV group is a
list of 128-token pages, each given by its first arena row, in position order. Page rows are
token-major; V rows are sigma_inv-ordered within a page.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from resin_models_gaudi.qwen3_5.checkpoint import sigma_inv
from resin_models_gaudi.qwen3_5.layers import SLOT_KEYS

PAGE_TOKENS = 128


@dataclass(frozen=True, slots=True)
class SlotRow:
    """One sequence of a chunk: its first column, first position and real tokens, and its
    (G, pages) K and V page rows (one layer's; the caller offsets them per layer)."""

    column: int
    pos0: int
    real: int
    k_pages: np.ndarray
    v_pages: np.ndarray


def kv_rows(tokens: int, rows: Sequence[SlotRow]) -> np.ndarray:
    """(T, 8) i32 destination rows of a chunk's K (groups in columns 0..3) then V rows (4..7); -1
    for padding and for groups past G."""
    G = rows[0].k_pages.shape[0]
    out = np.full((tokens, 8), -1, np.int32)
    for row in rows:
        pos = row.pos0 + np.arange(row.real)
        page, r = pos // PAGE_TOKENS, pos % PAGE_TOKENS
        cols = slice(row.column, row.column + row.real)
        out[cols, :G] = (row.k_pages[:, page] + r).T
        out[cols, 4 : 4 + G] = (row.v_pages[:, page] + sigma_inv(r)).T
    return out


@dataclass(frozen=True, slots=True)
class SlotLaunch:
    """Inputs of one slot launch: ``slots`` (S, 8) [group, qcol0, qpos0, key0, klen, 0, 0, 0],
    ``k_pages`` / ``v_pages`` (S, C / 128) first arena rows (-1: zeros)."""

    slots: np.ndarray
    k_pages: np.ndarray
    v_pages: np.ndarray

    def offset(self, rows: int) -> SlotLaunch:
        """The same launch for pages ``rows`` further into the arena (another layer)."""
        return SlotLaunch(
            self.slots,
            np.where(self.k_pages >= 0, self.k_pages + rows, -1).astype(np.int32),
            np.where(self.v_pages >= 0, self.v_pages + rows, -1).astype(np.int32),
        )


def plan_slots(
    rows: Sequence[SlotRow], query_block: int, hpg: int, per_launch: int
) -> list[SlotLaunch]:
    """Slots of a chunk: for every row, every (query block, key chunk, group) whose keys some
    query of the block can see, packed ``per_launch`` at a time; the last launch is padded with
    group -1 slots. A slot's table entry is [group, qcol0, qpos0, key0, klen, qtok0, 0, 0]
    (qtok0: the block's first column token). Order: rows, query blocks, key chunks, groups."""
    per_chunk = SLOT_KEYS // PAGE_TOKENS
    slots, kps, vps = [], [], []
    lane = np.arange(per_chunk)
    for row in rows:
        G, pages = row.k_pages.shape
        klen = row.pos0 + row.real
        t0 = np.arange(0, row.real, query_block)
        visible = np.minimum(row.pos0 + t0 + query_block, klen)  # keys a block may see
        chunks = -(-visible // SLOT_KEYS)
        block = np.repeat(np.arange(len(t0)), chunks)  # each (query block, key chunk)
        chunk = np.arange(len(block)) - np.repeat(np.cumsum(chunks) - chunks, chunks)
        key0 = chunk * SLOT_KEYS
        first = key0 // PAGE_TOKENS
        live = np.minimum(per_chunk, -(-(visible[block] - key0) // PAGE_TOKENS))
        assert pages >= int((first + live).max(initial=0))
        index = first[:, None] + lane[None, :]  # (E, per_chunk)
        valid = lane[None, :] < live[:, None]
        index = np.where(valid, index, 0)
        # (E, G, per_chunk) pages; entries ordered (query block, key chunk, group)
        kp = np.where(valid[:, None, :], row.k_pages[:, index].transpose(1, 0, 2), -1)
        vp = np.where(valid[:, None, :], row.v_pages[:, index].transpose(1, 0, 2), -1)
        E = len(block)
        fields = np.zeros((E, G, 8), np.int64)
        fields[:, :, 0] = np.arange(G)[None, :]
        fields[:, :, 1] = (hpg * (row.column + t0[block]))[:, None]
        fields[:, :, 2] = (row.pos0 + t0[block])[:, None]
        fields[:, :, 3] = key0[:, None]
        fields[:, :, 4] = klen
        fields[:, :, 5] = (row.column + t0[block])[:, None]
        slots.append(fields.reshape(E * G, 8))
        kps.append(kp.reshape(E * G, per_chunk))
        vps.append(vp.reshape(E * G, per_chunk))
    slots = np.concatenate(slots)
    kps, vps = np.concatenate(kps), np.concatenate(vps)
    pad = -(-len(slots) // per_launch) * per_launch - len(slots)
    if pad:
        fill = np.zeros((pad, 8), np.int64)
        fill[:, 0] = -1
        slots = np.concatenate([slots, fill])
        kps = np.concatenate([kps, np.full((pad, per_chunk), -1)])
        vps = np.concatenate([vps, np.full((pad, per_chunk), -1)])
    n = len(slots) // per_launch
    slots = slots.astype(np.int32).reshape(n, per_launch, 8)
    kps = kps.astype(np.int32).reshape(n, per_launch, per_chunk)
    vps = vps.astype(np.int32).reshape(n, per_launch, per_chunk)
    return [SlotLaunch(slots[i], kps[i], vps[i]) for i in range(n)]
