"""Multimodal rotary positions (mrope) for Qwen3.5/3.8.

Text tokens have one position. Image tokens have three: (temporal, height, width).
Qwen3.5 splits the rotary frequencies between them, so the cos/sin row of a token
is not a function of a single position index. That is why the runner's
position-indexed ``cs_tab`` cannot express an image prompt, and why this module
exists: it builds the per-token rows the tower's prompt needs.

The checkpoint sets ``mrope_interleaved: true`` with
``mrope_section: [11, 11, 10]``, which sums to ``rotary_dim // 2`` (32). Interleaved
means frequency ``f`` uses position stream ``f % 3`` (t, h, w), and the section
sizes are just the resulting counts: 11 t, 11 h, 10 w.

For a text-only prompt all three streams are equal, and the rows must match
``checkpoint.rope_table`` exactly. `test_qwen_mrope` asserts that, which is the
check that the interleaving is wired the right way round.
"""

from __future__ import annotations

import numpy as np

STREAMS = 3  # temporal, height, width


def inverse_frequencies(rotary: int, theta: float) -> np.ndarray:
    """``theta ** (-2i / rotary)`` for i < rotary // 2."""
    half = rotary // 2
    return theta ** (-np.arange(half, dtype=np.float64) * 2.0 / rotary)


def stream_of_frequency(rotary: int, sections: tuple[int, int, int]) -> np.ndarray:
    """Which position stream each frequency reads, under the interleaved layout."""
    half = rotary // 2
    if len(sections) != STREAMS or any(section <= 0 for section in sections):
        raise ValueError(f"mrope sections must be {STREAMS} positive values, got {sections}")
    if sum(sections) != half:
        raise ValueError(f"mrope sections {sections} must sum to {half}")
    return np.arange(half) % STREAMS


def mrope_rows(
    position_ids: np.ndarray, *, rotary: int, theta: float, sections: tuple[int, int, int]
) -> np.ndarray:
    """``(tokens, 2 * rotary)`` f32 rows ``[cos | cos | -sin | sin]`` (NeoX layout).

    ``position_ids`` is ``(3, tokens)``: temporal, height and width positions.
    """
    ids = np.asarray(position_ids)
    if ids.ndim != 2 or ids.shape[0] != STREAMS:
        raise ValueError(f"position_ids must be (3, tokens), got {ids.shape}")
    if ids.shape[1] == 0:
        raise ValueError("position_ids has no tokens")
    if ids.min() < 0:
        raise ValueError("position_ids must be non-negative")
    stream = stream_of_frequency(rotary, sections)
    inverse = inverse_frequencies(rotary, theta)
    angle = ids[stream].T * inverse[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    return np.concatenate([cos, cos, -sin, sin], 1).astype(np.float32)


def rope_rows(config, position_ids: np.ndarray) -> np.ndarray:
    """`mrope_rows` using a ``Qwen35Config``'s rotary settings."""
    sections = config.mrope_section
    if sections is None:
        raise ValueError("this checkpoint has no mrope sections")
    return mrope_rows(
        position_ids,
        rotary=config.rotary_dim,
        theta=config.rope_theta,
        sections=tuple(int(section) for section in sections),
    )


def text_position_ids(positions: np.ndarray) -> np.ndarray:
    """The three streams of a text-only prompt: every stream is the token position."""
    positions = np.asarray(positions, np.int64).reshape(1, -1)
    return np.repeat(positions, STREAMS, axis=0)
