"""One Qwen3.5/3.8 sequence stepped through a runner's native executor with plans made here (tests and tools): the
cache rule that the Engine follows, over a fixed range of the runner's slabs, without the Engine.

KV blocks come in order from the first slab up and STATE blocks from the last slab down as the sequence grows, one
STATE block per 4096 positions: a step reads the block of position pos0 - 1 and writes the block of its last
position. With the MTP layer, its KV blocks sit in one slab below at most three STATE blocks; with a drafter, a
ring of DRAFT blocks over its window sits in the slabs below them. A speculative step (``verify``) leases its
lookahead in every group, and the executor carries its drafts to the sequence's next speculative step."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from resin import BatchPhase, SamplingParams
from resin._resin import BatchPlan, CacheGroupPlan
from resin_models_gaudi.qwen3_5.cache import PAGE_TOKENS, STATE_TOKENS
from resin_models_gaudi.qwen3_5.model import SPEC_ROWS

GREEDY = SamplingParams(temperature=0.0, seed=0)


class PlanSequence:
    """One sequence of ``runner`` (a ``Qwen3_5GaudiRunner``) over ``slabs`` (default: all of them). ``prefill``,
    ``decode`` and ``verify`` make one step each through the runner's native executor and return its tokens."""

    _ids = iter(range(1 << 40, 1 << 41))  # sequence ids no Engine uses

    def __init__(self, runner, slabs: range | None = None) -> None:
        self.runner, model = runner, runner.model
        self.geometry = g = runner.geometry
        self.slabs = slabs or range(1, model.parent_capacity + 1)
        self.mtp_slab = self.slabs[-4] if model.mtp else None
        self.ring: list[tuple[int, int]] | None = (
            None  # (slab, slot) of each DRAFT block of the ring
        )
        if model.dfl_cfg is not None:
            per = g.draft_blocks_per_slab
            top = self.slabs[-4]
            # the Engine's resident DRAFT blocks: the window and a prefill chunk's writes (the exact range that
            # the Python executor's plan check takes)
            count = runner.cache_layout.groups[2].window_resident_block_count
            self.ring = [(top - i // per, i % per) for i in range(count)]
            self.ring_slabs = (count - 1) // per + 1
        self.reset()

    def reset(self) -> None:
        """A fresh sequence (a new sequence id: no carried drafts)."""
        self.sequence = next(self._ids)
        self.pos = 0
        self.state_blocks: dict[int, int] = {}  # logical STATE block -> slab

    def _state(self, logical: int) -> int:
        if logical not in self.state_blocks:
            self.state_blocks[logical] = self.slabs[-1 - len(self.state_blocks)]
        return self.state_blocks[logical]

    def _groups(self, n: int, lookahead: int) -> list[CacheGroupPlan]:
        """The step's cache groups for n tokens from ``pos`` and ``lookahead`` positions past them."""
        g = self.geometry
        reach = self.pos + n - 1 + lookahead
        pages = reach // PAGE_TOKENS + 1
        positions = self.pos + np.arange(n)
        page = positions // PAGE_TOKENS
        offsets = positions % PAGE_TOKENS
        per = g.kv_blocks_per_slab
        kv = [(self.slabs[0] + b // per, b % per) for b in range(pages)]
        groups = [_group(0, kv, 0, [kv[p] for p in page], offsets, per)]
        # the Engine's STATE table: the block of the last position leased and the one before it
        first = max(0, reach // STATE_TOKENS - 1)
        states = [(self._state(k), 0) for k in range(first, reach // STATE_TOKENS + 1)]
        written = [(self._state(int(p) // STATE_TOKENS), 0) for p in positions]
        groups.append(
            _group(
                1,
                states,
                first,
                written,
                positions % STATE_TOKENS,
                g.state_blocks_per_slab,
            )
        )
        top = self.slabs[-1] - len(self.state_blocks)
        if self.mtp_slab is not None:
            mtp = [(self.mtp_slab, b) for b in range(pages)]
            groups.append(
                _group(
                    2, mtp, 0, [mtp[p] for p in page], offsets, g.mtp_blocks_per_slab
                )
            )
            top = min(top, self.mtp_slab - 1)
        if self.ring is not None:
            count = len(self.ring)
            draft_first = max(0, pages - count)
            ring = [self.ring[k % count] for k in range(draft_first, pages)]
            written = [self.ring[int(p) % count] for p in page]
            groups.append(
                _group(2, ring, draft_first, written, offsets, g.draft_blocks_per_slab)
            )
            top = min(top, self.slabs[-4] - self.ring_slabs)
        if (self.mtp_slab is not None or self.ring is not None) and len(
            self.state_blocks
        ) > 3:
            raise MemoryError("the sequence exceeds its slabs")
        if kv[-1][0] > top:
            raise MemoryError("the sequence exceeds its slabs")
        return groups

    def _step(
        self,
        ids: Sequence[int],
        phase: BatchPhase,
        lookahead: int,
        params: SamplingParams,
    ):
        plan = BatchPlan(
            sequence_ids=[self.sequence],
            context_lengths=[self.pos],
            token_ids=[int(t) for t in ids],
            token_indptr=[0, len(ids)],
            phase=phase,
            cache_groups=self._groups(len(ids), lookahead),
            sampling_mask=[True],
            sampling_params=[params],
            media=[None],
        )
        return self.runner.forward(plan)

    def prefill(self, ids: Sequence[int], params: SamplingParams = GREEDY) -> int:
        """The prompt's next token (the executor carries the drafts after it)."""
        out = self._step(ids, BatchPhase.PREFILL, 0, params)
        self.pos += len(ids)
        return int(out.token_ids[0])

    def decode(self, token: int, params: SamplingParams = GREEDY) -> int:
        """One plain step from ``token`` (the last emitted), its plan leasing the speculative lookahead (as the
        Engine's)."""
        mode = self.runner.policy.mode
        self.runner.policy.mode = "never"
        try:
            out = self._step([token], BatchPhase.DECODE, self.runner.lookahead, params)
        finally:
            self.runner.policy.mode = mode
        self.pos += 1
        return int(out.token_ids[0])

    def verify(self, token: int, params: SamplingParams = GREEDY) -> list[int]:
        """One speculative step from ``token`` (the last emitted): the tokens it emits."""
        mode = self.runner.policy.mode
        self.runner.policy.mode = "always"
        try:
            out = self._step([token], BatchPhase.DECODE, self.runner.lookahead, params)
        finally:
            self.runner.policy.mode = mode
        tokens = [int(t) for t in out.token_ids]
        self.pos += len(tokens)
        return tokens

    @property
    def drafts(self) -> np.ndarray | None:
        """The drafts carried to this sequence's next speculative step."""
        carried = self.runner.carried(self.sequence)
        return None if carried is None else np.asarray(carried[2], np.int64)

    def logits(self) -> np.ndarray:
        """The last plain step's logits (a prefill's last token)."""
        return self.runner.model.logits()


def _group(
    index: int, blocks, first: int, written, offsets, per: int
) -> CacheGroupPlan:
    """A cache group of one row: its (slab, slot) blocks from logical block ``first``, and each token's (slab, slot)
    and offset in its block."""
    return CacheGroupPlan(
        group_index=index,
        block_indptr=[0, len(blocks)],
        block_slab_ids=[int(slab) for slab, _ in blocks],
        block_slotnums=[int(slot) for _, slot in blocks],
        block_table_start_indices=[int(first)],
        append_slab_ids=[int(slab) for slab, _ in written],
        append_slotnums=[int(slot) for _, slot in written],
        append_block_offsets=[int(offset) for offset in offsets],
        blocks_per_slab=int(per),
    )


def decode_together(
    sequences: Sequence[PlanSequence], tokens: Sequence[int]
) -> list[int]:
    """One plain decode step of several sequences of one runner as one batch: each one's next token."""
    runner = sequences[0].runner
    # the Engine leases a decode step of up to SPEC_ROWS rows its speculative lookahead
    lookahead = runner.lookahead if len(sequences) <= SPEC_ROWS else 0
    groups = [s._groups(1, lookahead) for s in sequences]
    merged = []
    for k in range(len(groups[0])):
        parts = [g[k] for g in groups]
        indptr = [0]
        for part in parts:
            indptr.append(indptr[-1] + len(part.block_slab_ids))
        merged.append(
            CacheGroupPlan(
                group_index=parts[0].group_index,
                block_indptr=indptr,
                block_slab_ids=np.concatenate([p.block_slab_ids for p in parts]),
                block_slotnums=np.concatenate([p.block_slotnums for p in parts]),
                block_table_start_indices=np.concatenate(
                    [p.block_table_start_indices for p in parts]
                ),
                append_slab_ids=np.concatenate([p.append_slab_ids for p in parts]),
                append_slotnums=np.concatenate([p.append_slotnums for p in parts]),
                append_block_offsets=np.concatenate(
                    [p.append_block_offsets for p in parts]
                ),
                blocks_per_slab=parts[0].blocks_per_slab,
            )
        )
    n = len(sequences)
    plan = BatchPlan(
        sequence_ids=[s.sequence for s in sequences],
        context_lengths=[s.pos for s in sequences],
        token_ids=[int(t) for t in tokens],
        token_indptr=list(range(n + 1)),
        phase=BatchPhase.DECODE,
        cache_groups=merged,
        sampling_mask=[True] * n,
        sampling_params=[GREEDY] * n,
        media=[None] * n,
    )
    mode = runner.policy.mode
    runner.policy.mode = "never"
    try:
        out = runner.forward(plan)
    finally:
        runner.policy.mode = mode
    for s in sequences:
        s.pos += 1
    return [int(t) for t in out.token_ids]
