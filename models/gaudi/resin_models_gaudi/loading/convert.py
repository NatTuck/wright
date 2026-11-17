"""Host conversions that several models share: FP8 encodings, the K-axis layouts of the Gaudi
GEMMs, BF16 rounding, threaded copies and the rotary table.

Layout rules:

* bf16 GEMM weights have K in "sigma order" inside each 128-element block (``kperm``).
* FP8 GEMM weights have K in "tau" order (``tperm``).
* OCP-range FP8 bytes (``needs_halving``) are halved for the Gaudi range (``halve``) and their
  scales doubled.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FP8_MAX = 240.0
BLOCK = 128

POOL = ThreadPoolExecutor(max(4, min(32, (os.cpu_count() or 8) // 2)))
"""Threads for host conversions. They start on first use, so they keep the NUMA placement of the
rank that loads."""


def _e4m3_values() -> np.ndarray:
    u = np.arange(256, dtype=np.int32)
    sign = np.where(u & 0x80, -1.0, 1.0)
    exponent, mantissa = (u >> 3) & 15, u & 7
    value = sign * np.where(
        exponent == 0,
        mantissa / 8.0 * 2.0**-6,
        (1 + mantissa / 8.0) * 2.0 ** (exponent - 7.0),
    )
    value[(u & 0x7F) == 0x7F] = np.nan
    return value.astype(np.float32)


E4M3 = _e4m3_values()


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """float32 to bf16 bits, rounding to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """float32 with |x| <= 240 to e4m3 bytes, rounding to nearest even.

    From the float32 bits: from 2^-6 up, the 23 mantissa bits round to 3 (a carry goes into the exponent) and
    the exponent bias 127 becomes 7; below, the byte is the number of 2^-9 steps (8 of them are the smallest
    normal's byte)."""
    x = np.ascontiguousarray(x, np.float32)
    bits = x.view(np.uint32)
    magnitude = bits & np.uint32(0x7FFFFFFF)
    rounded = magnitude >> np.uint32(20)
    rounded &= np.uint32(1)
    rounded += magnitude
    rounded += np.uint32(0x7FFFF)
    rounded >>= np.uint32(20)
    rounded -= np.uint32(120 << 3)
    out = rounded.astype(np.uint8)
    small = magnitude < np.uint32(0x3C800000)  # below 2^-6
    if small.any():
        out[small] = np.rint(np.abs(x[small]) * np.float32(512)).astype(np.uint8)
    out |= (bits >> np.uint32(24)).astype(np.uint8) & np.uint8(0x80)
    return out


def _halve_table() -> np.ndarray:
    values = E4M3.copy()
    values[np.isnan(values)] = 0
    return e4m3_encode(values / 2)


HALVE = _halve_table()


def block_f32(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(M, K) block-FP8 bytes and their (M/128, K/128) scales -> float32"""
    M, K = w.shape
    values = np.nan_to_num(E4M3[w])
    return (
        values.reshape(M // BLOCK, BLOCK, K // BLOCK, BLOCK) * s[:, None, :, None]
    ).reshape(M, K)


def needs_halving(weight: np.ndarray) -> bool:
    """Whether block-FP8 bytes use the OCP range (|x| up to 448): a value in the first 256 rows
    is above the Gaudi maximum of 240."""
    return bool(np.nanmax(np.abs(E4M3[np.asarray(weight)[:256]])) > FP8_MAX)


def _row_chunks(n: int, parts: int = 32) -> tuple[range, int]:
    step = max(1, -(-n // parts))
    return range(0, n, step), step


def halve(w: np.ndarray) -> np.ndarray:
    """``HALVE[w]`` in row chunks on many threads. A byte with an exponent field of 2 or more halves
    by taking 1 from the exponent (8 from the byte); subnormals, zero and NaN use the table."""
    out = np.empty(w.shape, np.uint8)
    rows, step = _row_chunks(w.shape[0])

    def run(i: int) -> None:
        src, dst = w[i : i + step], out[i : i + step]
        np.subtract(src, np.uint8(8), out=dst)
        rest = ((src & 0x70) == 0) | ((src & 0x7F) == 0x7F)
        dst[rest] = HALVE[src[rest]]

    list(POOL.map(run, rows))
    return out


def pcopy(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Row-chunked parallel copy (pages memory-mapped data in with many threads)."""
    rows, step = _row_chunks(dst.shape[0])
    list(POOL.map(lambda i: np.copyto(dst[i : i + step], src[i : i + step]), rows))
    return dst


def pcat(parts: list[np.ndarray]) -> np.ndarray:
    out = np.empty(
        (sum(p.shape[0] for p in parts),) + parts[0].shape[1:], parts[0].dtype
    )
    offset = 0
    for part in parts:
        pcopy(out[offset : offset + part.shape[0]], part)
        offset += part.shape[0]
    return out


def sigma_perm(n: int) -> np.ndarray:
    """perm[s] = source index of stored slot s, per 128-element block."""
    s = np.arange(128)
    base = np.where(s % 2 == 0, s // 2, 64 + s // 2)
    return (np.arange(n // 128)[:, None] * 128 + base[None, :]).reshape(-1)


def sigma_inv(p: np.ndarray) -> np.ndarray:
    """Stored slot of logical index p (inverse of ``sigma_perm``)."""
    p = np.asarray(p)
    block, r = p & ~127, p & 127
    return block + np.where(r < 64, 2 * r, 2 * (r - 64) + 1)


def tau_perm(n: int) -> np.ndarray:
    """perm[s] = source index of stored slot s, per 256-element block of tau order."""
    s = np.arange(256)
    return (
        np.arange(n // 256)[:, None] * 256 + ((s % 4) * 64 + s // 4)[None, :]
    ).reshape(-1)


def _permute_k(w: np.ndarray, groups: int, inner: int) -> np.ndarray:
    m, k = w.shape
    if k % (groups * inner):
        raise ValueError(f"K {k} is not a multiple of {groups * inner}: pad it first")
    out = np.empty((m, k), w.dtype)
    rows, step = _row_chunks(m)

    def run(i: int) -> None:
        out[i : i + step] = (
            w[i : i + step]
            .reshape(-1, k // (groups * inner), groups, inner)
            .transpose(0, 1, 3, 2)
            .reshape(-1, k)
        )

    list(POOL.map(run, rows))
    return out


def kperm(w: np.ndarray) -> np.ndarray:
    """K axis into sigma order."""
    return _permute_k(w, 2, 64)


def tperm(w: np.ndarray) -> np.ndarray:
    """K axis into tau order."""
    return _permute_k(w, 4, 64)


def rope_table(positions: int, config) -> np.ndarray:
    """(positions, 128) f32 rows ``[cos | cos | -sin | sin]`` (NeoX over ``config.rotary_dim``,
    base ``config.rope_theta``)."""
    rotary = config.rotary_dim
    half = rotary // 2
    inverse = config.rope_theta ** (-np.arange(half, dtype=np.float64) * 2.0 / rotary)
    angle = np.arange(positions, dtype=np.float64)[:, None] * inverse[None, :]
    c, s = np.cos(angle), np.sin(angle)
    return np.concatenate([c, c, -s, s], 1).astype(np.float32)


def disable_hugepage_madvise() -> None:
    """Stop NumPy from asking for transparent huge pages on large arrays (process-wide)."""
    try:
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except AttributeError:  # numpy < 2
        np.core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
