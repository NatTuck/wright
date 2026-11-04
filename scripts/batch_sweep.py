"""Run the batch throughput sweep on real text, one scripts/bench.py run per load. This gives the
draft results by batch size. scripts/sweep.py gives the prefill and decode results by context length.

uv run --no-default-groups --group gaudi scripts/batch_sweep.py \
    --qwen /path/to/Qwen3.8-27B-FP8 \
    --mistral /path/to/Mistral-Large-4-1T-A52B-GGUF \
    --mistral-tokenizer /path/to/Mistral-Large-4-tokenizer \
    --mistral-drafter /path/to/Mistral-Large-4-drafter \
    --glm /path/to/GLM-5.3-Flash \
    --corpus bench/corpus --output bench/sweep

Each run appends to its own JSONL file. A run whose file exists is skipped, so a stopped sweep
continues where it stopped. Use --pilot for one point per model, to set --steps and --seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Run:
    """One load: a model at one draft setting, across its batch sizes and context lengths."""

    name: str
    model: str
    draft_tokens: int
    batches: tuple[int, ...]
    contexts: tuple[int, ...]
    modes: tuple[str, ...] = ("sampled",)
    extra: tuple[str, ...] = ()
    render: dict = field(default_factory=dict)


NO_THINKING = {"enable_thinking": False}
_QWEN: dict[str, Any] = {"model": "qwen3_5-gaudi2", "render": NO_THINKING}
# The GLM template always opens a reasoning block, so it can only shorten it.
LOW_REASONING = {"reasoning_effort": "low"}
_GLM: dict[str, Any] = {"model": "glm5_3_flash-gaudi2", "render": LOW_REASONING}
_MISTRAL: dict[str, Any] = {"model": "mistral_large4-gaudi2", "render": LOW_REASONING}

SWEEP = (
    Run(
        "qwen-k0",
        draft_tokens=0,
        batches=(1, 8, 32),
        contexts=(8192, 16384, 32768),
        **_QWEN,
    ),
    Run(
        "qwen-k2",
        draft_tokens=2,
        batches=(1, 8, 32),
        contexts=(8192, 16384, 32768),
        modes=("sampled", "greedy"),
        **_QWEN,
    ),
    Run("qwen-k1", draft_tokens=1, batches=(1, 32), contexts=(8192,), **_QWEN),
    Run("qwen-k3", draft_tokens=3, batches=(1, 32), contexts=(8192,), **_QWEN),
    Run(
        "mistral-k0",
        draft_tokens=0,
        batches=(1, 4, 16),
        contexts=(2048, 8192, 16384, 32768, 65536),
        **_MISTRAL,
    ),
    Run(
        "mistral-k3",
        draft_tokens=3,
        batches=(1, 2, 4),
        contexts=(2048, 8192, 16384, 32768, 65536),
        modes=("sampled", "greedy"),
        **_MISTRAL,
    ),
    Run(
        "glm-k0",
        draft_tokens=0,
        batches=(1, 4, 16),
        contexts=(8192, 16384, 65536),
        **_GLM,
    ),
    Run(
        "glm-k2",
        draft_tokens=2,
        batches=(1, 4, 8, 16),
        contexts=(8192, 16384, 65536),
        modes=("sampled", "greedy"),
        **_GLM,
    ),
    Run("glm-k1", draft_tokens=1, batches=(1, 16), contexts=(8192,), **_GLM),
    Run("glm-k3", draft_tokens=3, batches=(1, 16), contexts=(8192,), **_GLM),
    # Where drafts stop paying: the batch sizes around the crossover.
    Run("glm-k0-mid", draft_tokens=0, batches=(2, 8), contexts=(8192,), **_GLM),
    Run("glm-k1-mid", draft_tokens=1, batches=(2, 4, 8), contexts=(8192,), **_GLM),
    Run(
        "glm-tp4-k0",
        model="glm5_3_flash-gaudi2-tp4",
        draft_tokens=0,
        batches=(1, 16),
        contexts=(8192,),
        render=LOW_REASONING,
    ),
    Run(
        "glm-tp4-k2",
        model="glm5_3_flash-gaudi2-tp4",
        draft_tokens=2,
        batches=(1, 16),
        contexts=(8192,),
        render=LOW_REASONING,
    ),
    Run(
        "glm-256k-k0",
        model="glm5_3_flash-gaudi2-256k",
        draft_tokens=0,
        batches=(1,),
        contexts=(65536, 262144),
        render=LOW_REASONING,
    ),
    Run(
        "glm-256k-k2",
        model="glm5_3_flash-gaudi2-256k",
        draft_tokens=2,
        batches=(1,),
        contexts=(65536, 262144),
        render=LOW_REASONING,
    ),
)

PILOT = (
    Run("pilot-qwen", draft_tokens=2, batches=(8,), contexts=(8192,), **_QWEN),
    Run("pilot-mistral", draft_tokens=3, batches=(4,), contexts=(8192,), **_MISTRAL),
    Run("pilot-glm", draft_tokens=2, batches=(8,), contexts=(8192,), **_GLM),
)


def sampling(checkpoint: Path) -> dict:
    """The checkpoint's own sampling defaults, which stay fixed for every run of that model."""
    config = checkpoint / "generation_config.json"
    values = json.loads(config.read_text()) if config.exists() else {}
    return {
        "temperature": values.get("temperature", 1.0),
        "top_k": values.get("top_k"),
        "top_p": values.get("top_p", 1.0),
    }


def command(
    run: Run, args, checkpoint: Path, tokenizer: Path | None, output: Path
) -> list[str]:
    values = sampling(tokenizer or checkpoint)
    line = [
        sys.executable,
        str(ROOT / "scripts" / "bench.py"),
        "--model",
        run.model,
        "--checkpoint",
        str(checkpoint),
        "--phases",
        "decode",
        "--decode-batch-sizes",
        ",".join(map(str, run.batches)),
        "--decode-context-lengths",
        ",".join(map(str, run.contexts)),
        "--draft-tokens",
        str(run.draft_tokens),
        "--prompt-file",
        f"english={args.corpus / 'english.jsonl'}",
        "--prompt-file",
        f"coding={args.corpus / 'coding.jsonl'}",
        "--text-modes",
        ",".join(run.modes),
        "--temperature",
        str(values["temperature"]),
        "--top-p",
        str(values["top_p"]),
        "--seeds",
        args.seeds,
        "--text-steps",
        str(args.steps),
        "--text-output-tokens",
        str(args.output_tokens),
        "--render-options",
        json.dumps(run.render),
        "--skip-unfit",
        "--output",
        str(output),
        *run.extra,
    ]
    if values["top_k"] is not None:
        line += ["--top-k", str(values["top_k"])]
    if tokenizer is not None:
        line += ["--tokenizer-checkpoint", str(tokenizer)]
    if run.model.startswith("mistral") and run.draft_tokens:
        line += ["--assistant-checkpoint", str(args.mistral_drafter)]
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--mistral", type=Path, required=True)
    parser.add_argument("--mistral-tokenizer", type=Path, required=True)
    parser.add_argument("--mistral-drafter", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("bench/corpus"))
    parser.add_argument("--output", type=Path, default=Path("bench/sweep"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--only", help="comma-separated run names")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "qwen": (args.qwen, None),
        "mistral": (args.mistral, args.mistral_tokenizer),
        "glm": (args.glm, None),
    }
    only = None if args.only is None else set(args.only.split(","))
    failed: list[str] = []
    for run in PILOT if args.pilot else SWEEP:
        if only is not None and run.name not in only:
            continue
        output = args.output / f"{run.name}.jsonl"
        if output.exists():
            print(f"[sweep] {run.name}: already done", flush=True)
            continue
        checkpoint, tokenizer = next(
            paths
            for family, paths in checkpoints.items()
            if run.model.startswith(family)
        )
        partial = output.with_suffix(".partial")
        partial.unlink(missing_ok=True)
        print(f"[sweep] {run.name}: start", flush=True)
        result = subprocess.run(
            command(run, args, checkpoint, tokenizer, partial), cwd=ROOT
        )
        if result.returncode:
            print(f"[sweep] {run.name}: failed with {result.returncode}", flush=True)
            failed.append(run.name)
            continue
        partial.replace(output)
        print(f"[sweep] {run.name}: done", flush=True)
    if failed:
        raise SystemExit(
            f"[sweep] failed: {', '.join(failed)}. Run again to retry them"
        )


if __name__ == "__main__":
    main()
