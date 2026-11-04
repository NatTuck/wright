"""Qwen3.5/3.8 checkpoint preparation (host only)."""

from __future__ import annotations

import numpy as np

from resin_models_gaudi.qwen3_5.checkpoint import HALVE, halve


def test_halve_matches_the_table_for_every_byte():
    w = np.arange(256 * 70, dtype=np.int64).astype(np.uint8).reshape(70, 256)
    assert np.array_equal(halve(w), HALVE[w])


def test_halve_leaves_its_input_alone():
    w = np.random.default_rng(0).integers(0, 256, (33, 128), dtype=np.uint8)
    before = w.copy()
    out = halve(w)
    assert np.array_equal(w, before)
    assert np.array_equal(out, HALVE[before])
