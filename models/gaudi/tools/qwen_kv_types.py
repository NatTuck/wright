"""Compare Qwen3.5/3.8 outputs on FP8 and bf16 KV caches (one card).

    python tools/qwen_kv_types.py CHECKPOINT OUT.json [--types bf16,bf16 fp8,bf16 fp8,fp8] [--lengths 2048 8192 32768]

For each cache type pair (K, V) the tool loads the model, and for each text (English: the docs of this checkout;
code: its Rust sources) and length it records the prompt logprobs of the last 1024 positions (the logprob of each
prompt token and the top 20) and 256 greedy tokens after the prompt. Then it compares each pair with the first:
the mean and largest change of the prompt tokens' logprobs, KL(first || pair) over the first's top 20, the top-1
agreement, and the greedy tokens up to their first difference. It prints the cache capacity of each pair too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCORED, TOP, GREEDY_TOKENS = 1024, 20, 256


def texts(tokenizer) -> dict[str, list[int]]:
    english = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    english += "\n\n" + (ROOT / "README.md").read_text()
    code = "\n\n".join(p.read_text() for p in sorted((ROOT / "crates").rglob("*.rs")))
    return {
        "english": tokenizer.encode(english).ids,
        "code": tokenizer.encode(code).ids,
    }


def run_types(
    checkpoint: str, types: tuple[str, str], prompts: dict, lengths: list[int]
) -> dict:
    from resin import Engine, SamplingParams
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner

    card = os.environ.get("RESIN_GAUDI_CARD")
    t0 = time.time()
    runner = Qwen3_5GaudiRunner(
        checkpoint,
        card=None if card is None else int(card),
        max_batch_size=8,
        max_seq_len=max(lengths) + GREEDY_TOKENS + 64,
        cache_types=types,
    )
    g = runner.model.geometry
    capacity = runner.cache_layout.parent_capacity
    out = {
        "types": list(types),
        "load_s": round(time.time() - t0, 1),
        "slabs": capacity,
        "slab_mib": round(g.slab_bytes / 2**20, 1),
        "kv_tokens_per_slab": g.kv_blocks_per_slab * 128,
        "kv_tokens": capacity * g.kv_blocks_per_slab * 128,
        "runs": {},
    }
    print(
        f"{types}: {capacity} slabs of {out['slab_mib']} MiB, {out['kv_tokens']:,} KV tokens",
        flush=True,
    )
    engine = Engine(
        runner,
        max_running_requests=8,
        context_length=max(lengths) + GREEDY_TOKENS + 64,
        prefix_cache=False,
        enable_overlap=True,
    )
    for name, ids in prompts.items():
        for length in lengths:
            prompt = ids[:length]
            params = SamplingParams(
                max_output_tokens=1,
                temperature=0.0,
                prompt_logprobs=True,
                prompt_top_logprobs=TOP,
            )
            lease = engine.submit(prompt, sampling_params=params)
            while engine.has_work():
                engine.step()
            lease.take_result()
            scores = lease.prompt_logprobs
            assert scores is not None
            scored = [e for e in scores[length - SCORED :] if e is not None]
            lease = engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    max_output_tokens=GREEDY_TOKENS, temperature=0.0
                ),
            )
            while engine.has_work():
                engine.step()
            out["runs"][f"{name}/{length}"] = {
                "logprob": [e.logprob for e in scored],
                "top": [
                    [[int(t), float(v)] for t, v in e.top_logprobs] for e in scored
                ],
                "greedy": lease.take_result(),
            }
            print(f"  {name}/{length} done", flush=True)
    engine.close()
    runner.close()
    return out


def compare(ref: dict, other: dict) -> dict:
    result = {}
    for key, a in ref["runs"].items():
        b = other["runs"][key]
        diffs = [abs(x - y) for x, y in zip(a["logprob"], b["logprob"], strict=True)]
        kls, top1 = [], 0
        for ta, tb in zip(a["top"], b["top"], strict=True):
            pb = dict((t, v) for t, v in tb)
            floor = tb[-1][
                1
            ]  # a token past the other's top 20 has at most its 20th logprob
            kl = sum(math.exp(v) * (v - pb.get(t, floor)) for t, v in ta)
            kls.append(max(kl, 0.0))
            top1 += ta[0][0] == tb[0][0]
        first = next(
            (i for i, (x, y) in enumerate(zip(a["greedy"], b["greedy"])) if x != y),
            len(a["greedy"]),
        )
        result[key] = {
            "mean_abs_dlogprob": round(sum(diffs) / len(diffs), 5),
            "max_abs_dlogprob": round(max(diffs), 4),
            "mean_kl_top20": round(sum(kls) / len(kls), 6),
            "top1_agree": round(top1 / len(a["top"]), 4),
            "greedy_same_until": first,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--types", nargs="+", default=["bf16,bf16", "fp8,bf16", "fp8,fp8"]
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 32768])
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    prompts = texts(tokenizer)
    for name, ids in prompts.items():
        if len(ids) < max(args.lengths):
            raise SystemExit(
                f"the {name} text has {len(ids)} tokens, fewer than {max(args.lengths)}"
            )
    results = [
        run_types(args.checkpoint, tuple(t.split(",")), prompts, args.lengths)
        for t in args.types
    ]
    report = {
        "capacity": {
            ",".join(r["types"]): {k: r[k] for k in ("slabs", "slab_mib", "kv_tokens")}
            for r in results
        },
        "against": ",".join(results[0]["types"]),
        "compare": {",".join(r["types"]): compare(results[0], r) for r in results[1:]},
    }
    args.out.write_text(json.dumps({"report": report, "runs": results}))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
