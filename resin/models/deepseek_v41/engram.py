# Adapted from DeepSeek-V4.1-Flash-1017 inference/engram.py, Copyright (c) 2023 DeepSeek, MIT License
# (resin/models/deepseek_v41/LICENSE).
"""Engram n-gram hash ids of DeepSeek-V4.1, on the host.

Each position is hashed with the tokens before it as n-grams of 2 .. max_ngram_size tokens, each over
n_heads heads; every (n-gram size, head) pair has its own prime-sized range of rows in its layer's table.
Ids first map to a compressed vocabulary where tokens that normalize alike are one id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEAD = -1
"""A token that no n-gram may span (an image position)."""


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above ``start`` that is not in ``seen``."""
    n = start + 1
    while not is_prime(n) or n in seen:
        n += 1
    return n


def compressed_token_map(backend) -> tuple[np.ndarray, int]:
    """Token id -> compressed id, for a ``tokenizers.Tokenizer``: tokens whose text normalizes alike
    (NFKC, NFD, accents stripped, lower case, runs of white space as one space, stripped) share an id;
    partial UTF-8 tokens are keyed by their raw form. Returns the map and the compressed vocabulary size."""
    from tokenizers import Regex, normalizers

    sentinel = ""  # a token that is exactly one space keeps a key of its own
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    vocab = backend.get_vocab_size(with_added_tokens=True)
    keys: dict[str, int] = {}
    out = np.empty(vocab, np.int64)
    for token in range(vocab):
        text = backend.decode([token], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token)
        else:
            key = normalizer.normalize_str(text) or text
        out[token] = keys.setdefault(key, len(keys))
    return out, len(keys)


@dataclass(frozen=True)
class EngramHash:
    """Hash ids for the Engram layers: ``ids(history, positions)`` -> (positions, layers, columns)."""

    token_map: np.ndarray
    pad_id: int
    multipliers: np.ndarray
    """(layers, max_ngram_size) int64"""
    primes: np.ndarray
    """(layers, max_ngram_size - 1, heads) int64"""
    offsets: np.ndarray
    """(layers, (max_ngram_size - 1) heads) int64: each column's first row in its layer's table"""

    @classmethod
    def build(cls, cfg, backend) -> EngramHash:
        token_map, vocab = compressed_token_map(backend)
        if vocab != cfg.engram_compressed_vocab_size:
            raise ValueError(
                f"compressed vocabulary of {vocab}, the checkpoint expects {cfg.engram_compressed_vocab_size}"
            )
        layers, n, heads = (
            len(cfg.engram_layer_ids),
            cfg.engram_max_ngram_size,
            cfg.engram_n_heads,
        )
        primes, seen = np.empty((layers, n - 1, heads), np.int64), set()
        for layer in range(layers):
            for gram in range(n - 1):
                current = cfg.engram_vocab_size - 1
                for head in range(heads):
                    current = next_prime(current, seen)
                    seen.add(current)
                    primes[layer, gram, head] = current
        flat = primes.reshape(layers, -1)
        offsets = np.concatenate(
            [np.zeros((layers, 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1
        )
        bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
        multipliers = np.stack(
            [
                np.random.default_rng(10007 * layer_id).integers(
                    0, bound, n, dtype=np.int64
                )
                * 2
                + 1
                for layer_id in cfg.engram_layer_ids
            ]
        )
        return cls(
            token_map,
            int(token_map[cfg.engram_pad_token_id]),
            multipliers,
            primes,
            offsets,
        )

    def compress(
        self, tokens: np.ndarray, live: np.ndarray | None = None
    ) -> np.ndarray:
        """Compressed ids of a sequence's tokens (``live`` False: DEAD)."""
        out = self.token_map[np.asarray(tokens, np.int64)]
        return out if live is None else np.where(live, out, DEAD)

    def ids(
        self, history: np.ndarray, positions: np.ndarray, offset: int = 0
    ) -> np.ndarray:
        """Hash ids at ``positions`` of a sequence whose compressed ids from position ``offset`` on are
        ``history`` (through the last position, and from max_ngram_size - 1 positions before the first or the
        sequence's start): (len(positions), layers, (max_ngram_size - 1) heads)."""
        n = self.multipliers.shape[1]
        positions = np.asarray(positions, np.int64)
        tokens = np.empty((len(positions), n), np.int64)
        blocked = np.zeros(len(positions), bool)
        for shift in range(n):
            source = history[np.maximum(positions - shift - offset, 0)]
            blocked |= (positions < shift) | (source == DEAD)
            tokens[:, shift] = np.where(blocked, self.pad_id, source)
        return self.hash(tokens)

    def newest(self) -> np.ndarray:
        """(layers, vocabulary) int64: every raw token's compressed id times the layer's first multiplier. With
        ``older``: ``hash(w)[:, k, g * heads + h] = (newest[k, w[:, 0]] ^ older(w[:, 1:])[:, k, g]) % primes[k, g, h]
        + offsets[k, g * heads + h]`` (products stay below 2^63)."""
        return self.token_map[None, :] * self.multipliers[:, :1]

    def older(self, tokens: np.ndarray) -> np.ndarray:
        """(P, layers, max_ngram_size - 1) int64: for each n-gram size 2 .. max_ngram_size, the XOR of the
        products of a position's older tokens ``tokens`` (P, max_ngram_size - 1) (compressed ids, newest first,
        ``pad_id`` where blocked)."""
        products = tokens[:, None, :] * self.multipliers[None, :, 1:]
        return np.bitwise_xor.accumulate(products, axis=-1)

    def hash(self, tokens: np.ndarray) -> np.ndarray:
        """Hash ids of n-gram windows: ``tokens`` (P, max_ngram_size) holds each position's compressed id and
        the ids before it, newest first (``pad_id`` where blocked) -> (P, layers, (max_ngram_size - 1) heads)."""
        n = self.multipliers.shape[1]
        products = tokens[:, None, :] * self.multipliers[None]  # (P, layers, n)
        rolling, out = products[..., 0], []
        for i in range(1, n):
            rolling = np.bitwise_xor(rolling, products[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, -1) + self.offsets[None]
