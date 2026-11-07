"""Turn a prepared media prompt into what the Qwen3.5 runner needs.

The recipe (Rust, `crates/resin-recipe/src/media/qwen.rs`) already produces the
hard part: ``pixel_values``, ``image_grid_thw``, per-token ``position_ids`` and
``rope_delta``. This module joins that to a vision tower - `DeviceTower`
(`vision_device.py`) on the card by default, `VisionTower` (`vision.py`) as the
CPU oracle - and the mrope row builder, so the runner has one call to make once
its graph can consume the result.

The device side the towers feed is in `layers.py` (the `media` variant of
``embed_graph`` splices the image rows into the embedding before the input
norm) and `model.py` (``_splice_media`` uploads this prompt's rope rows into
the plan's own table). The investigation that got this working is in
`docs/vision.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from resin_models_gaudi.qwen3_5.mrope import rope_rows
from resin_models_gaudi.qwen3_5.vision import VisionTower

VISION_SWITCH = "RESIN_GAUDI_VISION"


@dataclass(frozen=True, slots=True)
class ImagePrompt:
    """The device-side inputs an image prompt needs, in prompt order."""

    features: np.ndarray  # (image tokens, out_hidden) f32
    rope: np.ndarray  # (tokens, 2 * rotary) f32
    tokens: int
    # the image placeholders' span in the prompt: `image_positions` from the recipe's media
    # payload, which the Rust side has already validated as contiguous and grid-consistent.
    # It is NOT found by scanning the token ids for a run of identical placeholders: the
    # runner only ever sees one prefill chunk's tokens, and a chunk boundary (or a
    # prefix-cache restore) can start or end inside the run, which is exactly the
    # "0 maximal runs" crash the scan produced on chunked prompts.
    first: int = 0
    # `max(position_ids) + 1 - tokens`: how far the prompt's compressed positions fall behind
    # its token count. HF adds this to every later position, so a decode step after an image
    # sits at token index + delta, not at the token index. It is negative, and its magnitude
    # is the image's token count minus the merged grid's larger side.
    delta: int = 0

    @property
    def image_tokens(self) -> int:
        return int(self.features.shape[0])


def tower_from_checkpoint(path, config, device=None):
    """The tower `RESIN_GAUDI_VISION` asks for.

    ``device`` runs it on the card (`vision_device.DeviceTower`); ``cpu`` runs the torch
    oracle in `vision.py`. The default is ``device``: the card path is the one the accuracy
    set was measured with (19/19), and the CPU oracle is 181-1241 ms per image against
    19-74 ms on the card. Both return an object with the same ``forward`` and
    ``config.merge``, which is all `prepare_image_prompt` uses.
    """
    which = os.environ.get(VISION_SWITCH, "device").strip().lower()
    if which == "cpu":
        return VisionTower.from_checkpoint(path, config)
    if which == "device":
        from resin_models_gaudi.qwen3_5.vision_device import DeviceTower

        return DeviceTower.from_checkpoint(path, config, device=device)
    raise ValueError(f"{VISION_SWITCH}={which!r}: expected 'device' or 'cpu'")


def prepare_image_prompt(media, tower, config) -> ImagePrompt:
    """Run the tower over ``media`` and build its rope rows.

    ``media`` is the recipe's payload: ``pixel_values``, ``image_grid_thw``,
    ``position_ids`` and ``token_ids``.
    """
    pixel_values = np.asarray(media.pixel_values)
    grids = np.asarray(media.image_grid_thw)
    position_ids = np.asarray(media.position_ids)
    tokens = len(media.token_ids)

    features = tower.forward(pixel_values, grids)
    rows = rope_rows(config, position_ids)

    if rows.shape[0] != tokens:
        raise ValueError(
            f"{rows.shape[0]} rope rows for {tokens} prompt tokens; "
            "the recipe and the tower disagree on the prompt"
        )
    # HF's `mrope_position_deltas`: the decode positions after this prompt are the token
    # index plus this, because the image's 64 tokens occupy only 8 positions.
    delta = int(position_ids.max()) + 1 - tokens
    expected = int(np.prod(grids[:, 1:]).sum()) // (tower.config.merge**2)
    if features.shape[0] != expected:
        raise ValueError(f"{features.shape[0]} image tokens, expected {expected} from the grid")
    positions = np.asarray(media.image_positions, np.int64).reshape(-1)
    if positions.size == 0:
        raise ValueError("the media payload has no image positions")
    steps = np.diff(positions)
    if positions[0] < 0 or positions[-1] >= tokens or (steps != 1).any():
        raise ValueError(
            f"image positions {positions[:8]}... are not one contiguous run inside the "
            f"{tokens}-token prompt; the payload and the prompt disagree"
        )
    if positions.size != features.shape[0]:
        raise ValueError(
            f"{positions.size} placeholder positions for {features.shape[0]} image tokens"
        )
    return ImagePrompt(
        features=features, rope=rows, tokens=tokens, first=int(positions[0]), delta=delta
    )
