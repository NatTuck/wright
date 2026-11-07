"""Accuracy on the 19-case set, through HF, on the same cases the engine runs.

    RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
      .venv-ref/bin/python models/gaudi/tools/vision_acc_hf.py

Exits non-zero if any case misses. This is the reference the engine is compared against, so a
disagreement between the two totals is the thing to look at rather than either number alone.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from vision_accuracy_cases import cases  # sibling: run this file from the repo root

CKPT = os.environ.get("RESIN_GAUDI_QWEN35")


def require_ckpt() -> str:
    if CKPT is None:
        raise SystemExit(
            "set RESIN_GAUDI_QWEN35 to a Qwen3.8-27B FP8 checkpoint directory"
        )
    return CKPT


def main() -> int:
    ckpt = require_ckpt()
    proc = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, dtype="auto", local_files_only=True
    ).eval()
    right = total = 0
    for name, im, q, expect in cases():
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im}, {"type": "text", "text": q}],
            }
        ]
        inputs = proc.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        answer = proc.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        ok = expect.lower() in answer.lower()
        right += ok
        total += 1
        print(
            f"  {name:22s} expect {expect:8s} got {answer!r:18s} "
            f"{'ok' if ok else 'MISS'}",
            flush=True,
        )
    print(f"\nHF: {right}/{total}", flush=True)
    return 0 if right == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
