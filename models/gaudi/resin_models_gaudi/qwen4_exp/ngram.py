"""Hashed n-gram embedding rows of Qwen3.8-Flash-Next's PLE, on the host.

Token t_i gets ``(ngram_size - 1) * heads_per_ngram`` table rows: for n in 2 .. ngram_size,
``mixed = (t_i m_0) ^ (t_(i-1) m_1) ^ ...`` (uint64, wrapping), ``row_h = mixed % p_h + offset_h``. A
predecessor before the sequence start, or at or behind an EOS, reads as EOS (the token's own EOS does
not cut its context). Matches transformers ``Qwen4ExpTextNGramEmbedding``;
``NGram.device_tables`` feeds the ``ple_hash`` kernel, which does the same arithmetic on the card.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

M64 = (1 << 64) - 1
_G, _M1, _M2, _P1 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix64(v: int) -> int:
    v = (v + _G) & M64
    v = ((v ^ (v >> 30)) * _M1) & M64
    v = ((v ^ (v >> 27)) * _M2) & M64
    return (v ^ (v >> 31)) & M64


def multipliers(vocab: int, n: int, ple_index: int = 0, seed: int = 1234) -> list[int]:
    """The odd 64-bit multipliers of the n context positions (the checkpoint stores the same)."""
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + _P1 * ple_index
    return [2 * (_splitmix64((base + _G * (i + 1)) & M64) % half) + 1 for i in range(n)]


def _is_prime(v: int) -> bool:
    if v < 2 or v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def head_sizes(
    base: int, n_heads: int, ple_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(prime table size per head (u64), the heads' first rows (i64)): consecutive primes from ``base``."""
    sizes, p = [], base - 1
    for _ in range(ple_index * n_heads):
        p += 1
        while not _is_prime(p):
            p += 1
    for _ in range(n_heads):
        p += 1
        while not _is_prime(p):
            p += 1
        sizes.append(p)
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    return np.array(sizes, np.uint64), offs


class NGram:
    """The hashing of one PLE, with a running history of the last ``ngram_size - 1`` tokens."""

    def __init__(
        self,
        vocab: int,
        ngram_size: int,
        heads_per_ngram: int,
        base: int,
        eos: int,
        mult: Sequence[int] | None = None,
        ple_index: int = 0,
        seed: int = 1234,
    ) -> None:
        self.n, self.hpn, self.eos = ngram_size, heads_per_ngram, eos
        self.heads = (ngram_size - 1) * heads_per_ngram
        m = (
            mult
            if mult is not None
            else multipliers(vocab, ngram_size, ple_index, seed)
        )
        self.mult = np.array([int(x) & M64 for x in m], np.uint64)
        self.sizes, self.offs = head_sizes(base, self.heads, ple_index)
        self.hist: list[int] = [eos] * (ngram_size - 1)

    @property
    def rows_total(self) -> int:
        return int(self.sizes.sum())

    def reset(self) -> None:
        self.hist = [self.eos] * (self.n - 1)

    def rows(
        self,
        tokens: Sequence[int],
        advance: bool = True,
        history: Sequence[int] | None = None,
    ) -> np.ndarray:
        """(T, heads) int64 table rows of ``tokens`` after ``history`` (default: the stored one)."""
        k = self.n - 1
        hist = list(self.hist if history is None else history)
        t = np.array(hist + list(tokens), np.int64)
        T = len(tokens)
        ctx = np.empty((self.n, T), np.int64)
        ctx[0] = t[k:]
        cut = np.zeros(T, bool)
        for s in range(1, self.n):
            p = t[k - s : k - s + T]
            cut |= p == self.eos
            ctx[s] = np.where(cut, self.eos, p)
        c = ctx.astype(np.uint64)
        out = np.empty((T, self.heads), np.int64)
        with np.errstate(over="ignore"):
            mixed = c[0] * self.mult[0]
            for g in range(2, self.n + 1):
                mixed = mixed ^ (c[g - 1] * self.mult[g - 1])
                h0 = (g - 2) * self.hpn
                sz = self.sizes[h0 : h0 + self.hpn]
                out[:, h0 : h0 + self.hpn] = (mixed[:, None] % sz[None]).astype(
                    np.int64
                ) + self.offs[None, h0 : h0 + self.hpn]
        if advance and history is None:
            self.hist = (hist + list(tokens))[-k:]
        return out

    def device_tables(
        self, vocab: int, table_lo: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tables for ``ple_hash``: ``A`` (V, 8) i32 = (hi, lo) of t m_j for j = 0..2 (+ 2 pad), and ``HP``
        (4, 64) i32 per head lane: p, head offset - table_lo, 2^32 mod p, f32 bits of 1 / p."""
        if self.n != 3 or self.heads > 64:
            raise ValueError("ple_hash takes trigrams over at most 64 heads")
        t = np.arange(vocab, dtype=np.uint64)
        A = np.zeros((vocab, 8), np.uint32)
        with np.errstate(over="ignore"):
            for j in range(3):
                v = t * self.mult[j]
                A[:, 2 * j] = (v >> np.uint64(32)).astype(np.uint32)
                A[:, 2 * j + 1] = (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        HP = np.zeros((4, 64), np.int32)
        p = self.sizes.astype(np.int64)
        HP[0, : self.heads] = p
        HP[1, : self.heads] = self.offs - table_lo
        HP[2, : self.heads] = (1 << 32) % p
        HP[3, : self.heads] = (
            (1.0 / p.astype(np.float64)).astype(np.float32).view(np.int32)
        )
        return np.asarray(A.view(np.int32)), HP


__all__ = ["NGram", "head_sizes", "multipliers"]
