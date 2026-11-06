# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1"]
# ///
"""Measure how long one new prompt stops the streams that decode.

uv run scripts/gap_bench.py URL --streams 15 --prompt-tokens 6400 13000 26000 \\
    --output artifacts/gap-bench.json

For each prompt length, the script starts --streams streaming requests and waits until all of
them decode. It then sends one prompt of about that many tokens and records:

- ttft_s: the time from the send to the first token of the prompt;
- longest_gap_s: the longest time between two chunks of one stream, while the prompt prefills;
- baseline_gap_s: the median time between two chunks before the prompt;
- stream_tokens_per_s: the mean rate of one stream while the prompt prefills.

Each prompt starts with a new random word, so the prefix cache cannot hold it. Run the server
with --stream-interval 1, so that each chunk is one token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

PASSAGE = (
    "The river ran past the old mill, and the wheel turned through the night. Each morning "
    "the miller walked the bank, counted the sacks, and wrote the numbers in a book. "
)
STREAM_PROMPT = (
    "Count from 1 to 100000. Write each number on its own line, and nothing else."
)


@dataclass
class Stream:
    arrivals: list[float] = field(default_factory=list)
    finished: bool = False
    error: str | None = None


def chat(model: str, content: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def model_name(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(f"{url}/v1/models")
    response.raise_for_status()
    return response.json()["data"][0]["id"]


async def long_prompt(
    client: httpx.AsyncClient, url: str, model: str, tokens: int
) -> str:
    """Text of about `tokens` prompt tokens, from the server's own tokenizer."""
    sample = PASSAGE * 64
    response = await client.post(f"{url}/tokenize", json=chat(model, sample, 1))
    response.raise_for_status()
    per_passage = response.json()["count"] / 64
    repeats = max(1, round(tokens / per_passage))
    return (
        f"{secrets.token_hex(8)}. Summarize this text in one sentence.\n\n"
        + PASSAGE * repeats
    )


async def consume(
    client: httpx.AsyncClient, url: str, body: dict, stream: Stream, usage: dict
) -> None:
    try:
        async with client.stream(
            "POST", f"{url}/v1/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("error"):
                    stream.error = chunk["error"].get("message", str(chunk["error"]))
                if chunk.get("usage"):
                    usage.update(chunk["usage"])
                choices = chunk.get("choices") or []
                if choices and any(
                    choices[0].get("delta", {}).get(key)
                    for key in ("content", "reasoning_content", "reasoning")
                ):
                    stream.arrivals.append(time.perf_counter())
    except (httpx.HTTPError, asyncio.CancelledError) as error:
        if not isinstance(error, asyncio.CancelledError):
            stream.error = repr(error)
        raise
    finally:
        stream.finished = True


def gaps_in(arrivals: list[float], start: float, end: float) -> list[float]:
    """Intervals between consecutive chunks that overlap [start, end]."""
    return [
        later - earlier
        for earlier, later in zip(arrivals, arrivals[1:])
        if later >= start and earlier <= end
    ]


async def measure(args, client: httpx.AsyncClient, model: str, tokens: int) -> dict:
    prompt = await long_prompt(client, args.url, model, tokens)
    streams = [Stream() for _ in range(args.streams)]
    tasks = [
        asyncio.create_task(
            consume(
                client, args.url, chat(model, STREAM_PROMPT, args.stream_tokens), s, {}
            )
        )
        for s in streams
    ]
    deadline = time.perf_counter() + args.timeout
    while any(len(s.arrivals) < args.warmup_chunks for s in streams):
        if time.perf_counter() > deadline or any(s.finished for s in streams):
            for task in tasks:
                task.cancel()
            errors = sorted({s.error for s in streams if s.error})
            raise RuntimeError(
                f"the streams did not all decode before the prompt: {errors or 'no error'}"
            )
        await asyncio.sleep(0.05)
    # Keep a short baseline of steady decode before the prompt arrives.
    await asyncio.sleep(args.baseline_seconds)

    usage: dict = {}
    probe = Stream()
    sent = time.perf_counter()
    prompt_task = asyncio.create_task(
        consume(client, args.url, chat(model, prompt, 1), probe, usage)
    )
    await asyncio.wait_for(prompt_task, args.timeout)
    first = probe.arrivals[0] if probe.arrivals else time.perf_counter()
    ended_early = sum(1 for s in streams if s.finished)
    await asyncio.sleep(args.tail_seconds)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    longest = []
    baseline = []
    rates = []
    window_s = max(first - sent, 1e-9)
    steady_start = sent - args.baseline_seconds
    steady_tokens = sum(
        1 for s in streams for t in s.arrivals if steady_start <= t < sent
    )
    window_tokens = sum(1 for s in streams for t in s.arrivals if sent <= t <= first)
    for s in streams:
        window = gaps_in(s.arrivals, sent, first)
        longest.append(max(window, default=first - sent))
        baseline += gaps_in([t for t in s.arrivals if t < sent], 0.0, sent)
        inside = sum(1 for t in s.arrivals if sent <= t <= first)
        rates.append(inside / max(first - sent, 1e-9))
    return {
        "prompt_tokens_requested": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "streams": args.streams,
        "ttft_s": first - sent,
        "longest_gap_s": max(longest),
        "median_longest_gap_s": statistics.median(longest),
        "baseline_gap_s": statistics.median(baseline) if baseline else None,
        "stream_tokens_per_s": statistics.mean(rates),
        # Throughput: all streams together before the prompt, and while it prefills.
        "steady_decode_tokens_per_s": steady_tokens / args.baseline_seconds,
        "window_decode_tokens": window_tokens,
        "window_decode_tokens_per_s": window_tokens / window_s,
        "prefill_tokens_per_s": (usage.get("prompt_tokens") or tokens) / window_s,
        # Streams that ended before the window closed make the decode totals low.
        "streams_done_early": ended_early,
        "stream_errors": [s.error for s in streams if s.error],
    }


async def main(args) -> None:
    limits = httpx.Limits(max_connections=args.streams + 4)
    timeout = httpx.Timeout(args.timeout, connect=30)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        model = args.model or await model_name(client, args.url)
        report = {"url": args.url, "model": model, "label": args.label, "runs": []}
        for tokens in args.prompt_tokens:
            for _ in range(args.repeats):
                run = await measure(args, client, model, tokens)
                report["runs"].append(run)
                print(
                    f"{tokens:>7} tokens: TTFT {run['ttft_s']:7.2f} s, longest gap "
                    f"{run['longest_gap_s']:6.2f} s, baseline gap "
                    f"{(run['baseline_gap_s'] or 0) * 1000:5.0f} ms, stream rate "
                    f"{run['stream_tokens_per_s']:5.1f} tok/s, all streams "
                    f"{run['window_decode_tokens_per_s']:6.1f} tok/s while prefilling "
                    f"(steady {run['steady_decode_tokens_per_s']:6.1f})",
                    flush=True,
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("url", help="server root, such as http://127.0.0.1:8080")
    parser.add_argument("--model", help="model name (default: the first in /v1/models)")
    parser.add_argument("--streams", type=int, default=15)
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[6400, 13000, 26000]
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--stream-tokens", type=int, default=8192, help="max_tokens of each stream"
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=8,
        help="chunks per stream before the prompt",
    )
    parser.add_argument("--baseline-seconds", type=float, default=2.0)
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--label", help="a name for this run in the report")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if min(arguments.streams, arguments.repeats, *arguments.prompt_tokens) < 1:
        parser.error("streams, repeats and prompt tokens must be positive")
    asyncio.run(main(arguments))
