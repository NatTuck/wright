"""Image preprocessing on the branches that resample, through the serving recipe path.

A 256x256 image takes the *identity* path: `min_pixels` is 256 squared, so neither the
upscale nor the downscale branch fires and no filter runs at all. That is why the original
"preprocessing is bit-identical" check could not have detected a resize difference - it was
measured on the one input class that resizes nothing. Every other size a user sends takes a
branch that does work, so these are all five:

| input | branch |
| --- | --- |
| 512x384 | identity, but with structure (the control) |
| 100x100 | below `min_pixels`: the upscale branch |
| 1000x64 | `min_pixels` again, extreme aspect: largest single-axis scaling |
| 300x500 | no branch: `h_bar` and `w_bar` rounding only, non-square |
| 300x300 | above `max_pixels`: the downscale branch |

Each case asserts the grid `resize_shape` gives, that no value leaves the source range, that
every value lands on an 8-bit level, and a fingerprint of the tensor.

What each assertion can and cannot prove. The grid check fails if `resize_shape` moves. The
level check is the one that separates the fixed path from the pre-fix one: HF's processor
resizes uint8 tensors, so torchvision quantizes to 8 bits between the two separable passes,
and a float resize leaves values between levels - on the pre-fix path 66% of the values of
the 100x100 case sit up to 0.5 levels off, against 7e-06 here. The fingerprint is a
regression lock captured from the implementation whose output matched HF's `image_processor`
to at most one grey level on every branch (`docs/vision.md`, and a3acbfe for the two-pass
uint8 explanation); it proves the path has not changed since that comparison and cannot by
itself prove the comparison was right. The independent HF comparison needs transformers in a
separate interpreter and stays in `models/gaudi/tools/vision_preprocess.py`, which is where the
recorded before/after table comes from.

The downscale branch is the expensive one: the production bounds reach it only above
`max_pixels` = 16,777,216, and a 6000x6000 image costs 32 s to encode and resample. The last
case therefore builds the same recipe with a lowered `max_pixels`, which fires the same
branch - same filter, same antialias, a smaller output - on a 300x300 image in milliseconds.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from resin.recipe import (
    Image as RImage,
)
from resin.recipe import (
    Message,
    RenderOptions,
    Role,
    Text,
    get_native_recipe,
)

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None or not (Path(CHECKPOINT) / "preprocessor_config.json").exists(),
    reason="requires RESIN_GAUDI_QWEN35, a Qwen3.8-27B FP8 checkpoint",
)

# element positions the fingerprint reads: the first row, one mid row, and the last row
SAMPLES = ((0, 0), (0, 1), (0, 1535), (100, 700), (-1, 0), (-1, -1))


class Case(NamedTuple):
    width: int
    height: int
    grid: tuple[int, int, int]
    shape: tuple[int, int]
    mean: float
    low: float
    high: float
    samples: tuple[float, ...]


CASES = {
    "identity": Case(
        512, 384, (1, 24, 32), (768, 1536),
        -0.24918301030993462, -1.0, 1.0,
        (1.0, 0.7254902124404907, -0.7647058963775635, -0.7647058963775635,
         -0.7647058963775635, 0.7254902124404907),
    ),
    "upscale": Case(
        100, 100, (1, 16, 16), (256, 1536),
        -0.1658598848929008, -1.0, 1.0,
        (1.0, 0.9843137264251709, -0.9137254953384399, -0.843137264251709,
         -0.7647058963775635, 0.7254902124404907),
    ),
    "extreme_aspect": Case(
        1000, 64, (1, 6, 64), (384, 1536),
        -0.2577818662005787, -1.0, 1.0,
        (1.0, 0.7333333492279053, -0.7647058963775635, -0.7647058963775635,
         -0.7647058963775635, 0.7254902124404907),
    ),
    "rounding": Case(
        300, 500, (1, 32, 18), (576, 1536),
        -0.23385984236719431, -0.9921568632125854, 0.9921568632125854,
        (0.9921568632125854, 0.7254902124404907, -0.7647058963775635,
         -0.7647058963775635, -0.7647058963775635, 0.7254902124404907),
    ),
    "downscale": Case(
        300, 300, (1, 16, 16), (256, 1536),
        -0.23395629258205494, -0.9686274528503418, 0.9607843160629272,
        (0.9607843160629272, 0.7176470756530762, -0.929411768913269,
         -0.7647058963775635, -0.7647058963775635, 0.7254902124404907),
    ),
}  # fmt: skip


def picture(width: int, height: int):
    """Structure on both axes: a hard vertical edge plus a horizontal ramp."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, height), (220, 30, 30))
    im.paste(
        Image.new("RGB", (width // 2, height), (30, 30, 220)),
        (width // 2, 0, width, height),
    )
    draw = ImageDraw.Draw(im)
    for x in range(0, width, max(2, width // 16)):
        draw.line((x, 0, x, height), fill=(255, 255, 0), width=1)
    return im


def data_url(width: int, height: int) -> str:
    buffer = io.BytesIO()
    picture(width, height).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def pixels(recipe, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rendered = recipe.render(
        [
            Message(
                Role.USER,
                [Text("Describe this image."), RImage(source=data_url(width, height))],
            )
        ],
        options=RenderOptions(enable_thinking=False),
    )
    return (
        np.asarray(rendered.media.pixel_values, np.float32),
        np.asarray(rendered.media.image_grid_thw),
    )


@pytest.fixture(scope="module")
def recipe():
    """The recipe serving builds: the media config straight from the checkpoint."""
    from deploy.models import create_recipe, get_model

    return create_recipe(replace(get_model("qwen3_5-gaudi2"), media=True), CHECKPOINT)


@pytest.fixture(scope="module")
def bounded():
    """The same recipe, with bounds that put a 300x300 image on the downscale branch.

    `create_recipe` takes `min_pixels`/`max_pixels` from the checkpoint, where both are
    absent and the processor defaults (256 squared, 4096 squared) apply. Lowering them here
    reaches the same branch on an image small enough to encode in a test; the name is the
    one `create_recipe` would pick for this family, since the spec leaves `recipe` unset.
    """
    from tokenizers import Tokenizer

    from deploy.models import get_model

    spec = get_model("qwen3_5-gaudi2")
    config = json.loads((Path(CHECKPOINT) / "preprocessor_config.json").read_text())
    return get_native_recipe(
        spec.recipe or "qwen3_8",
        Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json")),
        media_config={**config, "min_pixels": 1024, "max_pixels": 65536},
        **dict(spec.recipe_options),
    )


@pytest.fixture(scope="module")
def rendered(recipe, bounded):
    """Rendered (pixel_values, grid) by case name, so each image is processed once."""
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in cache:
            case = CASES[name]
            built = bounded if name == "downscale" else recipe
            cache[name] = pixels(built, case.width, case.height)
        return cache[name]

    return get


@pytest.mark.parametrize("name", list(CASES))
def test_the_grid_is_the_processor_resize_shape(rendered, name):
    case = CASES[name]
    _, grid = rendered(name)
    assert grid.shape == (1, 3)
    assert tuple(int(x) for x in grid[0]) == case.grid


@pytest.mark.parametrize("name", list(CASES))
def test_the_pixel_tensor_stays_in_the_source_range(rendered, name):
    """An invariant, not a discriminator: torchvision's uint8 path clamps, so no value can
    leave [-1, 1]. Kept because a filter that lets the cubic's overshoot through would break
    it - though the image crate's CatmullRom, the pre-fix path, does not overshoot on these
    images, so `test_every_value_is_a_uint8_level` is what separates the two.
    """
    pv, _ = rendered(name)
    assert float(np.abs(pv).max()) <= 1.0 + 1e-6


@pytest.mark.parametrize("name", list(CASES))
def test_every_value_is_a_uint8_level(rendered, name):
    """The two-pass quantization, visible from here: normalization maps levels to integers."""
    pv, _ = rendered(name)
    level = (pv.astype(np.float64) + 1.0) * 127.5
    off = np.abs(level - np.round(level))
    assert off.max() < 1e-4, (
        f"{name}: {int((off > 1e-4).sum())} of {off.size} values are not on a uint8 level "
        f"(worst {off.max():.6f} off); a resize that does not quantize between its passes "
        "does this"
    )


@pytest.mark.parametrize("name", list(CASES))
def test_the_resampled_values_are_unchanged(rendered, name):
    """The lock: the resize, the patch layout and the normalization, per branch."""
    case = CASES[name]
    pv, _ = rendered(name)
    assert pv.shape == case.shape
    got = [
        float(pv.astype(np.float64).mean()),
        float(pv.min()),
        float(pv.max()),
        *(float(pv[i, j]) for i, j in SAMPLES),
    ]
    want = [case.mean, case.low, case.high, *case.samples]
    np.testing.assert_allclose(
        got, want, rtol=0, atol=1e-6, err_msg=f"{name}: the preprocessing path changed"
    )
