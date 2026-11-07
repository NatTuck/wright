"""Diff `VisionTower` against the reference implementation, stage by stage.

    RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \\
      PATH=/opt/habanalabs/bin:$PATH ./.venv/bin/python models/gaudi/tools/vision_reference.py

This exists because for the whole project `vision.py` was the oracle *for itself*: the device
graph was diffed against it, so both agreed and neither had been checked against the model
that the checkpoint actually describes. The device matching it to 2.9e-06 meant nothing about
fidelity. This module compares the CPU tower to HF's `Qwen3VLVisionModel` on one fixed image
and reports the first stage that diverges.

It also checks the numpy twins the serving path computes with (`vision_rope(np, ...)`,
`interpolate_positions_np`) against the torch oracle, since the device tower must not
depend on torch.

Stages: patch embed + position, after block 0, after the last block, and the merger.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from resin_models_gaudi.qwen3_5.checkpoint import Safetensors
from resin_models_gaudi.qwen3_5.vision import (
    VisionConfig,
    VisionTower,
    _f32,
    interpolate_positions,
    interpolate_positions_np,
    vision_rope,
)

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")
GRID = 16


def require_checkpoint() -> str:
    if CHECKPOINT is None:
        raise SystemExit(
            "set RESIN_GAUDI_QWEN35 to a Qwen3.8-27B FP8 checkpoint directory"
        )
    return CHECKPOINT


def reference(raw: dict) -> Qwen3VLVisionModel:
    """HF's vision tower with the checkpoint's weights loaded into it."""
    model = Qwen3VLVisionModel(Qwen3VLVisionConfig(**raw["vision_config"])).eval()
    files = Safetensors(require_checkpoint())
    state = {}
    for name in files.meta:
        if name.startswith("model.visual."):
            state[name[len("model.visual.") :]] = torch.from_numpy(
                np.ascontiguousarray(_f32(files, name))
            )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weights do not line up: missing {missing[:4]}, extra {unexpected[:4]}")
    return model


def image(grid: int, seed: int = 0) -> np.ndarray:
    """A structured still, in the recipe's patch layout: (patches, 3*2*16*16)."""
    axis = np.linspace(-1, 1, grid, dtype=np.float32)
    plane = np.broadcast_to(axis[None], (grid, grid))
    rgb = np.stack((plane, plane.T, 1 - np.abs(plane)), axis=-1).reshape(grid * grid, 3)
    patch = np.repeat(rgb, 16 * 16, axis=1)
    return np.concatenate([patch, patch], axis=1).astype(np.float32)


def relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / np.abs(want).max())


def device_check(pixels: np.ndarray, want_merged: np.ndarray) -> int:
    """The same image through the device tower, diffed against the reference's merger output.

    This is the comparison that should have existed from the start: the card against HF, not
    the card against a file we wrote. It goes through `DeviceTower`, so it also covers the
    splice-free path end to end (graph, weights, merge kernel, output readback).
    """
    from resin_models_gaudi.qwen3_5.vision_device import DeviceTower
    from resin_models_gaudi.synapse.runtime import open_device

    ckpt = require_checkpoint()
    device = open_device(None)
    tower = DeviceTower.from_checkpoint(ckpt, json.loads((Path(ckpt) / "config.json").read_text()), device=device)
    grid = int(round(pixels.shape[0] ** 0.5))
    got = tower.forward(pixels, [[1, grid, grid]])
    err = relative(got, want_merged)
    print(f"  device merger output vs the reference: relative {err:.3e}  "
          f"({'ok' if err < 1e-3 else 'MISMATCH'})")
    tower.close()
    return 0 if err < 1e-3 else 2


def numpy_parity() -> bool:
    """The numpy twins the serving path computes with, against the torch oracle.

    The device tower must not depend on torch, so `vision_rope(np, ...)` and
    `interpolate_positions_np` are what serving runs; this is the check that they
    did not drift from the torch arithmetic the oracle is validated with.
    """
    hidden = 32
    table = np.random.default_rng(0).normal(size=(48 * 48, hidden)).astype(np.float32)
    worst = 0.0
    for height, width in ((48, 48), (16, 16), (10, 14)):
        want = interpolate_positions(
            torch, torch.from_numpy(table), 48, height, width, hidden
        ).numpy()
        got = interpolate_positions_np(table, 48, height, width, hidden)
        worst = max(worst, float(np.abs(got - want).max()))
    for height, width in ((16, 16), (10, 14)):
        cos_t, sin_t = vision_rope(torch, 72, height, width, 2)
        cos_n, sin_n = vision_rope(np, 72, height, width, 2)
        worst = max(
            worst,
            float(np.abs(cos_t.numpy() - cos_n).max()),
            float(np.abs(sin_t.numpy() - sin_n).max()),
        )
    ok = worst < 1e-5
    print(f"  numpy twins vs the torch oracle: max abs {worst:.3e}  "
          f"({'ok' if ok else 'MISMATCH'})")
    return ok


def main() -> int:
    print("numpy twins:")
    ok = numpy_parity()
    raw = json.loads((Path(require_checkpoint()) / "config.json").read_text())
    config = VisionConfig.from_hugging_face(raw["vision_config"])
    pixels = image(GRID)
    patches = pixels.shape[0]

    hf = reference(raw)
    captured = {}

    def grab(module, args, kwargs=None):
        """Capture block 0's input: patch embed plus position, before any attention."""
        captured["patch"] = (args[0] if args else kwargs["hidden_states"]).detach()

    handle = hf.blocks[0].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        out = hf(hidden_states=torch.from_numpy(pixels), grid_thw=torch.tensor([[1, GRID, GRID]]))
    handle.remove()
    want_patch = captured["patch"].numpy()
    want_blocks = out.last_hidden_state.numpy()
    want_merged = out.pooler_output.numpy()

    tower = VisionTower.from_checkpoint(require_checkpoint(), raw)
    x = tower._linear(torch.from_numpy(pixels), "patch_embed.proj") + tower._pos_embed(GRID, GRID)
    got_patch = x.numpy()
    rope = vision_rope(torch, config.head_dim, GRID, GRID, config.merge)
    after_first = None
    for index in range(config.depth):
        x = tower._block(x, index, rope)
        if index == 0:
            after_first = x.numpy()
    got_blocks = x.numpy()
    got_merged = tower._merger(x, GRID, GRID).numpy()

    print(f"patches {patches}, merged {got_merged.shape[0]}, depth {config.depth}")
    rows = [
        ("patch embed + position", got_patch, want_patch),
        ("after block 0", after_first, want_blocks if config.depth == 1 else None),
        ("after the last block", got_blocks, want_blocks),
        ("merger output", got_merged, want_merged),
    ]
    for label, got, want in rows:
        if want is None:
            print(f"  {label:24s} skipped (needs a depth-1 run)")
            continue
        err = relative(got, want)
        ok = ok and err < 1e-3
        print(f"  {label:24s} relative {err:.3e}  ({'ok' if err < 1e-3 else 'MISMATCH'})")
    print("\n->", "the tower matches the reference" if ok else "the tower does NOT match the reference")
    if "--device" in __import__("sys").argv:
        print("\ndevice:")
        ok = device_check(pixels, want_merged) == 0 and ok
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
