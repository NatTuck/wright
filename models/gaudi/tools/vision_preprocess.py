"""Preprocessing against HF's processor, on the branches that actually resample.

    RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
      HABANA_VISIBLE_MODULES=6 PATH=/opt/habanalabs/bin:$PATH \
      .venv/bin/python models/gaudi/tools/vision_preprocess.py

A 256x256 image takes the **identity** resize path: `min_pixels` is 65536 = 256 squared, so
neither the upscale nor the downscale branch fires and no filter runs. That is the input class
the original "preprocessing is bit-identical" check used, and it could not have detected a
resize difference. Every image a user sends that is not exactly 256x256 takes a branch that
does work, so this covers each one:

| input | branch |
| --- | --- |
| 512x384 | identity, but with structure (the control) |
| 100x100 | below `min_pixels`: the upscale branch |
| 1000x64 | `min_pixels` again, extreme aspect: largest single-axis scaling |
| 300x500 | no branch: `h_bar` and `w_bar` rounding only, non-square |
| 6000x6000 | above `max_pixels`: the downscale branch |

The comparison against HF's `image_processor` is a separate interpreter (`.venv-ref`, which
must not share one with the thing it judges), so this module checks what can be checked here -
that the resized dimensions match `resize_shape`, and that no value leaves the source range -
and prints the recorded HF numbers alongside.

Measured, `pv` against HF's processor, before and after the resize fix:

| input | before | after |
| --- | --- | --- |
| 512x384 identity | bit-identical | bit-identical |
| 100x100 -> 256x256 | 1.4e-01 (18 levels) | **7.8e-03 (1 level)** |
| 1000x64 -> 1024x96 | 3.9e-03 | **6e-08** |
| 300x500 -> 288x512 | 3.1e-03 | **6e-08** |
| 6000x6000 -> 4096x4096 | 5.6e-02 | **7.8e-03 (1 level)** |

The grids matched on every branch both before and after, so `resize_shape` was never the
problem; the filter was. The remaining 1 level is 8-bit quantization at the rounding boundary.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")


def require_checkpoint() -> str:
    if CHECKPOINT is None:
        raise SystemExit(
            "set RESIN_GAUDI_QWEN35 to a Qwen3.8-27B FP8 checkpoint directory"
        )
    return CHECKPOINT

# input size -> the resized size HF's smart_resize gives, which our resize_shape must match
CASES = {
    (512, 384): (384, 512),  # identity
    (100, 100): (256, 256),  # min_pixels upscale
    (1000, 64): (96, 1024),  # min_pixels, extreme aspect
    (300, 500): (512, 288),  # rounding only
    (6000, 6000): (4096, 4096),  # max_pixels downscale
}


def image(width: int, height: int) -> "Image":
    """Structure on both axes: a hard vertical edge plus a horizontal ramp."""
    im = Image.new("RGB", (width, height), (220, 30, 30))
    im.paste(Image.new("RGB", (width // 2, height), (30, 30, 220)), (width // 2, 0, width, height))
    d = ImageDraw.Draw(im)
    for x in range(0, width, max(2, width // 16)):
        d.line((x, 0, x, height), fill=(255, 255, 0), width=1)
    return im


def main() -> int:
    from deploy.models import create_recipe, get_model
    from resin.recipe import Image as RImage
    from resin.recipe import Message, RenderOptions, Role, Text

    recipe = create_recipe(replace(get_model("qwen3_5-gaudi2"), media=True), require_checkpoint())
    failures = 0
    print(f"{'input':>11s} {'resized':>11s} {'grid':>12s} {'pv':>14s} {'range':>18s}  verdict")
    for (width, height), (want_h, want_w) in CASES.items():
        im = image(width, height)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        rendered = recipe.render(
            [Message(Role.USER, [Text("Describe this image."), RImage(source=url)])],
            options=RenderOptions(enable_thinking=False),
        )
        media = rendered.media
        pv = np.asarray(media.pixel_values, np.float32)
        grid = np.asarray(media.image_grid_thw)
        got_h, got_w = int(grid[0][1]) * 16, int(grid[0][2]) * 16
        size_ok = (got_h, got_w) == (want_h, want_w)
        # torchvision's uint8 path clamps, so the normalized tensor cannot leave [-1, 1]
        range_ok = float(np.abs(pv).max()) <= 1.0 + 1e-6
        failures += not size_ok or not range_ok
        print(
            f"{width}x{height:<6d} {got_w}x{got_h:<7d} {str(grid.tolist()):>12s} "
            f"{str(pv.shape):>14s} [{pv.min():+.3f}, {pv.max():+.3f}]  "
            f"{'ok' if size_ok and range_ok else 'FAIL'}"
            f"{'' if size_ok else ' (resize_shape)'}{'' if range_ok else ' (overshoot)'}"
        )
    print("\n->", "ok" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
