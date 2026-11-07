from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from resin_models_gaudi.qwen3_5.checkpoint import rope_table
from resin_models_gaudi.qwen3_5.mrope import (
    STREAMS,
    inverse_frequencies,
    mrope_rows,
    stream_of_frequency,
    text_position_ids,
)

# The checkpoint's settings: rotary_dim 64, theta 1e7, sections [11, 11, 10].
ROTARY = 64
THETA = 10_000_000.0
SECTIONS = (11, 11, 10)
CONFIG = SimpleNamespace(rotary_dim=ROTARY, rope_theta=THETA)


def test_interleaved_sections_are_the_checkpoint_counts() -> None:
    stream = stream_of_frequency(ROTARY, SECTIONS)
    counts = [int((stream == i).sum()) for i in range(STREAMS)]
    assert counts == list(SECTIONS)


def test_sections_must_partition_half_the_rotary_dimension() -> None:
    with pytest.raises(ValueError, match="must sum to"):
        stream_of_frequency(ROTARY, (10, 11, 10))
    with pytest.raises(ValueError, match="positive"):
        stream_of_frequency(ROTARY, (0, 32, 0))


def test_text_positions_match_the_position_indexed_rope_table() -> None:
    """All three streams equal: mrope rows must equal the existing 1-D table."""
    tokens = 64
    positions = np.arange(tokens)
    rows = mrope_rows(
        text_position_ids(positions), rotary=ROTARY, theta=THETA, sections=SECTIONS
    )
    reference = rope_table(tokens, CONFIG)
    np.testing.assert_allclose(rows, reference, rtol=1e-6, atol=1e-6)


def test_image_positions_differ_per_stream() -> None:
    """An image token's row is not the row of any single position."""
    # Two image tokens with distinct (t, h, w), surrounded by text.
    position_ids = np.array(
        [
            [0, 1, 1, 1, 1, 2],  # temporal
            [0, 0, 0, 1, 1, 2],  # height
            [0, 0, 1, 0, 1, 2],  # width
        ]
    )
    rows = mrope_rows(position_ids, rotary=ROTARY, theta=THETA, sections=SECTIONS)
    assert rows.shape == (6, 2 * ROTARY)
    assert np.isfinite(rows).all()
    # The two image tokens must not share a row, and neither may equal its
    # same-index text row.
    assert not np.allclose(rows[1], rows[2])
    flat = np.repeat(position_ids[0:1], STREAMS, axis=0)
    text = mrope_rows(flat, rotary=ROTARY, theta=THETA, sections=SECTIONS)
    assert not np.allclose(rows[1], text[1])


def test_rope_rows_are_unit_amplitude_and_duplicated() -> None:
    rows = mrope_rows(
        text_position_ids(np.arange(8)), rotary=ROTARY, theta=THETA, sections=SECTIONS
    )
    half = ROTARY // 2
    cos, neg_sin, sin = rows[:, :half], rows[:, 2 * half : 3 * half], rows[:, 3 * half :]
    np.testing.assert_allclose(cos**2 + sin**2, 1.0, atol=1e-6)
    np.testing.assert_allclose(rows[:, half : 2 * half], cos)
    np.testing.assert_allclose(neg_sin, -sin)


def test_inverse_frequencies_are_monotone() -> None:
    inverse = inverse_frequencies(ROTARY, THETA)
    assert inverse.shape == (ROTARY // 2,)
    assert np.all(np.diff(inverse) < 0)
    np.testing.assert_allclose(inverse[0], 1.0)


def test_position_ids_are_validated() -> None:
    with pytest.raises(ValueError, match=r"\(3, tokens\)"):
        mrope_rows(np.zeros((2, 4)), rotary=ROTARY, theta=THETA, sections=SECTIONS)
    with pytest.raises(ValueError, match="no tokens"):
        mrope_rows(np.zeros((3, 0)), rotary=ROTARY, theta=THETA, sections=SECTIONS)
    with pytest.raises(ValueError, match="non-negative"):
        mrope_rows(-np.ones((3, 4)), rotary=ROTARY, theta=THETA, sections=SECTIONS)


def test_decode_positions_are_offset_by_the_mrope_delta() -> None:
    """After an image prompt the model rotates at the token position plus the prompt's delta.

    An image's tokens occupy one position per merged cell, so its compressed positions fall
    behind its token count: `delta = max(position_ids) + 1 - tokens = merged - image_tokens`.
    HF adds that to every later position, so a decode step after an image is not at the token
    index. `RowCache.pos_delta` is the only place the two are allowed to differ: the cache
    stays indexed by token position and the rope is offset.
    """
    import numpy as np

    from resin_models_gaudi.qwen3_5.model import RowCache
    from resin_models_gaudi.qwen3_5.mrope import rope_rows

    # a text prompt of 10 tokens, an 8x8 merged grid (64 image tokens), 5 more tokens
    before, merged, after = 10, 8, 5
    image_tokens = merged * merged
    tokens = before + image_tokens + after
    position_ids = np.zeros((3, tokens), np.int64)
    position_ids[:, :before] = np.arange(before)
    rows, cols = np.divmod(np.arange(image_tokens), merged)
    for axis, values in enumerate((0, rows, cols)):
        position_ids[axis, before : before + image_tokens] = values + before
    position_ids[:, before + image_tokens :] = before + merged + np.arange(after)

    delta = int(position_ids.max()) + 1 - tokens
    assert delta == merged - image_tokens == -56

    row = RowCache(84, np.zeros(0, np.int64), -1, 0, None, 0, (), delta)
    assert row.pos0 == 84          # the cache's index, uncompressed
    assert row.rope_at == 84 - 56  # the position the model rotates as
    # a row with no delta is its own rope position, which is every text prompt
    assert RowCache(84, np.zeros(0, np.int64), -1, 0).rope_at == 84

    # the rope rows a media plan uploads are the prompt's own, so they are absolute already:
    # the delta is applied to *later* steps, never to the prompt's rows
    config = SimpleNamespace(rotary_dim=ROTARY, rope_theta=THETA, mrope_section=SECTIONS)
    rows_table = rope_rows(config, position_ids)
    assert rows_table.shape == (tokens, 2 * ROTARY)
    # the last row's angle is the prompt's last compressed position, which the delta explains
    assert int(position_ids.max()) + 1 - tokens == delta
