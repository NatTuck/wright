"""DeepSeek-V4.1 rotary tables for ``ml4_rope``: adjacent pairs (2i, 2i + 1) of the last 64 dims of a head.

Layers that compress (ratio > 0) use YaRN over ``compress_rope_theta``; window-only layers use plain
``rope_theta``. Frequencies and angles are f32, as the reference computes them.
"""

from __future__ import annotations

import math

import numpy as np


def frequencies(
    dim: int,
    base: float,
    original: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
):
    """(dim / 2,) f32 rotary frequencies; YaRN when ``original`` > 0."""
    freqs = (
        1.0
        / (
            np.float32(base)
            ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
        )
    ).astype(np.float32)
    if original > 0:

        def corrected(rotations: float) -> float:
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), dim - 1)
        ramp = np.clip(
            (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1
        )
        smooth = (1 - ramp).astype(np.float32)
        freqs = (freqs / np.float32(factor) * (1 - smooth) + freqs * smooth).astype(
            np.float32
        )
    return freqs


def layer_frequencies(cfg, ratio: int) -> np.ndarray:
    if ratio:
        return frequencies(
            cfg.rope_head_dim, cfg.compress_rope_theta, cfg.rope_original_max_position, cfg.rope_factor,
            cfg.rope_beta_fast, cfg.rope_beta_slow,
        )  # fmt: skip
    return frequencies(cfg.rope_head_dim, cfg.rope_theta, 0, 1.0, 0.0, 0.0)


def cs_table(
    positions: np.ndarray, freqs: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """(T, 128) f32 for ``ml4_rope``: lane l < 64 holds cos of pair l >> 1; lane 64 + l holds its sin, negated
    for even l (inverse: the opposite sign)."""
    angle = np.asarray(positions, np.float32)[:, None] * freqs[None, :]
    cos = np.cos(angle.astype(np.float64)).astype(np.float32).repeat(2, 1)
    sin = np.sin(angle.astype(np.float64)).astype(np.float32).repeat(2, 1)
    sign = np.where(np.arange(64) % 2 == 1, 1.0, -1.0).astype(np.float32)
    return np.concatenate([cos, sin * (-sign if inverse else sign)], 1)
