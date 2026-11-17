"""Prepared weights (``Fp8Weight``, ``DenseWeight``) as items of the shared uploader
(``loading.upload``).

Block-FP8 weights are already in the Gaudi range (``qwen3_5.checkpoint.Qwen35Checkpoint`` halves
OCP-range bytes on the host), so they only requantize on the card to per-row FP8: ``name`` holds
the FP8 bytes and ``name + ".s"`` the row scales. Dense tensors upload as prepared.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from resin_models_gaudi.loading.upload import Fp8


@dataclass(frozen=True, slots=True)
class Fp8Weight:
    """Block-FP8 bytes (M, K), K tau-permuted, with f32 block scales (M/128, K/128)."""

    bytes: np.ndarray
    block_scales: np.ndarray


@dataclass(frozen=True, slots=True)
class DenseWeight:
    """A bf16 (uint16 bits) or f32 tensor uploaded as-is."""

    array: np.ndarray


def weight_items(
    weights: Iterable[tuple[str, Fp8Weight | DenseWeight]],
) -> Iterator[tuple[str, object, str]]:
    """``(name, weight)`` pairs (a checkpoint's or a drafter's) as ``(name, payload, kind)``."""
    for name, weight in weights:
        if isinstance(weight, Fp8Weight):
            yield name, Fp8(weight.bytes, weight.block_scales), "inc8"
        else:
            assert isinstance(weight, DenseWeight), name
            yield name, weight.array, "plain"
