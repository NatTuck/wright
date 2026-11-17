"""Which part of a dense hybrid decoder's weights one tensor-parallel rank holds (Qwen3.5/3.8).

Rank ``r`` of ``n`` holds:

- attention: query heads [r H/n, (r+1) H/n) and the KV groups they read. With more ranks than KV groups, each
  group sits on ``n / G`` ranks (the KV heads are replicated);
- GDN: key heads [r nk/n, (r+1) nk/n) and the value heads that read them: in the checkpoint's order a value
  head o reads key head o // (nv / nk), so they are one run [r nv/n, (r+1) nv/n);
- MLP: a run of the 256-wide blocks of the intermediate size, split as evenly as possible (the first ranks take
  one block more when the blocks do not divide; FP8 GEMMs permute K inside 256-element blocks);
- replicated: the embeddings, the LM head, the norms, and the MTP layer.

The row-parallel output projections (attention ``o_proj``, GDN ``out_proj``, MLP ``down_proj``) give every rank
a partial sum of the hidden vector, all-reduced across the ranks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FFN_BLOCK = 256  # MLP channels a shard boundary must be a multiple of


@dataclass(frozen=True, slots=True)
class DenseShard:
    """Rank ``rank`` of ``nranks``: index sets of its heads and channels in checkpoint order."""

    rank: int
    nranks: int
    heads: int
    groups: int
    nk: int
    nv: int
    ffn: int

    def __post_init__(self) -> None:
        n = self.nranks
        problems = []
        if not 0 <= self.rank < n:
            problems.append(f"rank {self.rank} of {n}")
        if self.heads % n:
            problems.append(f"{n} ranks do not divide {self.heads} attention heads")
        if not (self.groups % n == 0 or n % self.groups == 0):
            problems.append(
                f"{n} ranks and {self.groups} KV groups do not divide one another"
            )
        elif n > self.groups and (self.heads // self.groups) % (n // self.groups):
            problems.append("a KV group's query heads do not split over its ranks")
        if self.nk % n:
            problems.append(f"{n} ranks do not divide {self.nk} GDN key heads")
        if self.nv % self.nk:
            problems.append("value heads are not a multiple of the key heads")
        if self.ffn % FFN_BLOCK or self.ffn // FFN_BLOCK < n:
            problems.append(
                f"{self.ffn} MLP channels do not split into {n} runs of {FFN_BLOCK}"
            )
        if problems:
            raise ValueError("; ".join(problems))

    # ---- attention
    @property
    def hpg(self) -> int:
        """query heads per KV group (of the whole model)"""
        return self.heads // self.groups

    @property
    def attn_heads(self) -> np.ndarray:
        h = self.heads // self.nranks
        return np.arange(self.rank * h, (self.rank + 1) * h)

    @property
    def attn_groups(self) -> np.ndarray:
        return np.unique(self.attn_heads // self.hpg)

    # ---- GDN
    @property
    def key_heads(self) -> np.ndarray:
        k = self.nk // self.nranks
        return np.arange(self.rank * k, (self.rank + 1) * k)

    @property
    def value_heads(self) -> np.ndarray:
        """the value heads that read this rank's key heads, in checkpoint order"""
        r = self.nv // self.nk
        return np.arange(self.key_heads[0] * r, (self.key_heads[-1] + 1) * r)

    # ---- MLP
    @property
    def ffn_range(self) -> tuple[int, int]:
        blocks, n = self.ffn // FFN_BLOCK, self.nranks
        per, extra = divmod(blocks, n)
        first = self.rank * per + min(self.rank, extra)
        count = per + (self.rank < extra)
        return first * FFN_BLOCK, (first + count) * FFN_BLOCK

    # ---- local dimensions
    @property
    def local_heads(self) -> int:
        return len(self.attn_heads)

    @property
    def local_groups(self) -> int:
        return len(self.attn_groups)

    @property
    def local_nk(self) -> int:
        return len(self.key_heads)

    @property
    def local_nv(self) -> int:
        return len(self.value_heads)

    @property
    def local_ffn(self) -> int:
        a, b = self.ffn_range
        return b - a


def channels(heads: np.ndarray, width: int) -> np.ndarray:
    """The channels of ``heads`` of ``width`` channels each, in order."""
    return (np.asarray(heads)[:, None] * width + np.arange(width)[None, :]).reshape(-1)
