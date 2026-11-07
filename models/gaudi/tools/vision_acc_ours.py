"""Accuracy on the 19-case set, through the engine.

    RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
      HABANA_VISIBLE_MODULES=6 .venv/bin/python models/gaudi/tools/vision_acc_ours.py

Exits non-zero if any case misses, so it can be used as a check rather than only as a report.
The cases come from `vision_accuracy_cases`, which is the same set `vision_acc_hf` runs.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer
from vision_accuracy_cases import cases  # sibling: run this file from the repo root

from deploy.models import create_recipe, get_model
from resin.recipe import Image as RImage
from resin.recipe import Message, RenderOptions, Role, Text
from resin_models_gaudi.qwen3_5.media import prepare_image_prompt
from resin_models_gaudi.qwen3_5.model import Qwen35Gaudi, SingleSequence
from resin_models_gaudi.qwen3_5.vision_device import DeviceTower
from resin_models_gaudi.synapse.runtime import open_device

CKPT = os.environ.get("RESIN_GAUDI_QWEN35")


def require_ckpt() -> str:
    if CKPT is None:
        raise SystemExit(
            "set RESIN_GAUDI_QWEN35 to a Qwen3.8-27B FP8 checkpoint directory"
        )
    return CKPT


def data_url(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ckpt = require_ckpt()
    raw = json.loads((Path(ckpt) / "config.json").read_text())
    tok = Tokenizer.from_file(str(Path(ckpt) / "tokenizer.json"))
    device = open_device(None)
    engine = Qwen35Gaudi(device, ckpt, parent_capacity=12, media=True)
    try:
        tower = DeviceTower.from_checkpoint(ckpt, raw, device=device)
        recipe = create_recipe(replace(get_model("qwen3_5-gaudi2"), media=True), ckpt)
        seq = SingleSequence(engine, range(1, 7))
        right = total = 0
        for name, im, q, expect in cases():
            prompt = recipe.render(
                [Message(Role.USER, [Text(q), RImage(source=data_url(im))])],
                options=RenderOptions(enable_thinking=False),
            )
            m = prompt.media
            ids = list(m.token_ids)
            p = prepare_image_prompt(m, tower, engine.config)
            span = p.first
            seq.reset()
            out = [seq.prefill(ids, media=(p.features, span, p.image_tokens, p.rope, p.delta))]
            for _ in range(5):
                out.append(seq.decode(out[-1]))
            answer = tok.decode(out).strip()
            ok = expect.lower() in answer.lower()
            right += ok
            total += 1
            print(
                f"  {name:22s} expect {expect:8s} got {answer!r:18s} "
                f"{'ok' if ok else 'MISS'}",
                flush=True,
            )
        print(f"\nours: {right}/{total}")
        return 0 if right == total else 1
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
