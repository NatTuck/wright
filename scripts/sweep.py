"""Measure prefill and decode by context length on real text: llama-benchy against a real server.

uv run --no-default-groups --group gaudi scripts/sweep.py \
    --qwen /path/to/Qwen3.8-27B-FP8 \
    --mistral /path/to/Mistral-Large-4-1T-A52B-GGUF \
    --mistral-tokenizer /path/to/Mistral-Large-4-1T-A52B-NVFP4 \
    --mistral-drafter /path/to/Mistral-Large-4-drafter \
    --glm /path/to/GLM-5.3-Flash \
    --qwen35 /path/to/qwen35-checkpoints \
    --output bench/sweep

The script starts one server for each configuration, and runs llama-benchy with sampling at each
context length, concurrency and text. Concurrency goes in powers of 2 up to the preset's default
maximum batch. Each configuration appends to its own JSONL file, and finished cells are skipped, so a
stopped sweep continues where it stopped. --gpqa runs GPQA Diamond from
inspect_evals in place of the sweep. --report prints the result tables.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTEXTS = tuple(1 << n for n in range(10, 18))  # 1K to 128K
TEXTS = {
    "english": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "code": "https://raw.githubusercontent.com/python/cpython/v3.13.0/Objects/unicodeobject.c",
}
NO_THINKING = {"enable_thinking": False}
LOW_REASONING = {"reasoning_effort": "low"}


@dataclass(frozen=True)
class Config:
    """One server: a model on some cards with a fixed number of draft tokens."""

    name: str
    family: str
    preset: str
    draft_tokens: int | None = None  # None keeps the preset's value
    body: dict = field(default_factory=dict)
    server: tuple[str, ...] = ()
    folder: str | None = None  # the checkpoint's folder under --qwen35
    sampling: dict | None = None  # None reads the checkpoint's generation_config.json


_QWEN: dict[str, Any] = dict(family="qwen", preset="qwen3_5-gaudi2", body=NO_THINKING)
# The Qwen3.5 dense checkpoints have no generation_config.json. These values are the ones that the
# model cards give for thinking mode, and that the Qwen3.8-27B checkpoint gives.
_QWEN35_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
_MISTRAL: dict[str, Any] = dict(
    family="mistral", preset="mistral_large4-gaudi2", body=LOW_REASONING
)
_GLM8: dict[str, Any] = dict(
    family="glm", preset="glm5_3_flash-gaudi2", body=LOW_REASONING
)
_GLM4: dict[str, Any] = dict(
    family="glm", preset="glm5_3_flash-gaudi2-tp4", body=LOW_REASONING
)

CONFIGS = (
    Config("qwen-mtp0", draft_tokens=0, **_QWEN),
    Config("qwen-mtp1", draft_tokens=1, **_QWEN),
    Config("qwen-mtp2", draft_tokens=2, **_QWEN),
    Config("glm-8cards-mtp0", draft_tokens=0, **_GLM8),
    Config("glm-8cards-mtp1", draft_tokens=1, **_GLM8),
    Config("glm-8cards-mtp2", draft_tokens=2, **_GLM8),
    Config("glm-4cards-mtp0", draft_tokens=0, **_GLM4),
    Config("glm-4cards-mtp1", draft_tokens=1, **_GLM4),
    Config("glm-4cards-mtp2", draft_tokens=2, **_GLM4),
    # Mistral-Large-4 has no MTP layer. Its drafter takes only 3 draft tokens.
    Config("mistral-nodraft", draft_tokens=0, **_MISTRAL),
    Config("mistral-dspark3", draft_tokens=3, **_MISTRAL),
    *(
        Config(
            f"qwen35-{size.lower()}-mtp0",
            "qwen35",
            f"qwen3_5-{size.lower()}-gaudi2",
            draft_tokens=0,
            body=NO_THINKING,
            folder=f"Qwen3.5-{size}-FP8",
            sampling=_QWEN35_SAMPLING,
        )
        for size in ("9B", "4B", "2B", "0.8B")
    ),
)

# Each row names the score of the same model on https://artificialanalysis.ai/evaluations/gpqa-diamond,
# read on 2026-11-01. The sweep results choose the draft tokens and the request count.
GPQA = (
    (
        Config(
            "qwen-xhigh", "qwen", "qwen3_5-gaudi2", body={"reasoning_effort": "xhigh"}
        ),
        32,
        0.905,
    ),
    (
        Config("qwen-no-reasoning", "qwen", "qwen3_5-gaudi2", body=NO_THINKING),
        32,
        0.818,
    ),
    (Config("glm-8cards", "glm", "glm5_3_flash-gaudi2"), 16, 0.912),
    (Config("glm-4cards", "glm", "glm5_3_flash-gaudi2-tp4"), 16, 0.912),
    (
        Config(
            "mistral-high",
            "mistral",
            "mistral_large4-gaudi2",
            body={"reasoning_effort": "high"},
        ),
        16,
        None,
    ),
)


def checkpoint_paths(config: Config, args) -> tuple[Path, Path]:
    """The checkpoint, and the directory that holds its tokenizer."""
    if config.family == "qwen":
        return args.qwen, args.qwen
    if config.family == "glm":
        return args.glm, args.glm
    if config.family == "qwen35":
        path = args.qwen35 / config.folder
        return path, path
    return args.mistral, args.mistral_tokenizer


def sampling(tokenizer: Path) -> dict:
    """The checkpoint's own sampling defaults, which stay fixed for every run of that model."""
    path = tokenizer / "generation_config.json"
    values = json.loads(path.read_text()) if path.exists() else {}
    fields = {
        "temperature": values.get("temperature", 1.0),
        "top_p": values.get("top_p", 1.0),
    }
    if values.get("top_k") is not None:
        fields["top_k"] = values["top_k"]
    return fields


def server_command(config: Config, args, context: int | None = None) -> list[str]:
    checkpoint, tokenizer = checkpoint_paths(config, args)
    line = [
        sys.executable,
        str(ROOT / "scripts" / "serve_local.py"),
        "--model",
        config.preset,
        "--checkpoint",
        str(checkpoint),
        "--port",
        str(args.port),
        "--drain-timeout",
        "5",
        *config.server,
    ]
    if config.draft_tokens is not None:
        line += ["--draft-tokens", str(config.draft_tokens)]
    if context is not None:  # a throughput run: every prompt must miss the cache
        line += ["--context-length", str(context), "--no-prefix-cache"]
    if args.max_batch_size:
        line += ["--max-batch-size", str(args.max_batch_size)]
    if args.gpu:
        line += ["--gpu", args.gpu]
    if config.family == "mistral":
        line += ["--tokenizer", str(tokenizer)]
        if config.draft_tokens:
            line += ["--assistant-checkpoint", str(args.mistral_drafter)]
    return line


def fetch(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def wait_until_ready(process: subprocess.Popen, base: str, limit: float) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"the server stopped with exit code {process.returncode}"
            )
        try:
            fetch(f"{base}/health")
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise RuntimeError(f"the server was not ready after {limit:.0f} s")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def draft_counters(base: str) -> dict[str, float]:
    """Accepted and proposed drafts from /metrics, which rise over the life of the server."""
    totals: dict[str, float] = {}
    for line in fetch(f"{base}/metrics").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        key = name.split("{")[0]
        if "spec_draft" in key or "spec_accepted" in key:
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def server_counters(base: str) -> dict[str, float]:
    """Requests that run now, and requests that have lost their cache since the server started."""
    values: dict[str, float] = {}
    for line in fetch(f"{base}/metrics").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        key = name.split("{")[0]
        for short, part in (
            ("running", "num_running_reqs"),
            ("retracted", "retracted"),
        ):
            if part in key:
                values[short] = values.get(short, 0.0) + float(value)
    return values


def levels(config: Config, args) -> tuple[int, ...]:
    """Powers of 2 up to the preset's default maximum batch."""
    if args.concurrency:
        return args.concurrency
    from deploy.models import get_model

    limit = get_model(config.preset).max_batch_size
    return tuple(1 << n for n in range(16) if 1 << n <= limit)


class Watcher(threading.Thread):
    """Poll /metrics during a run: how many requests ran at once, and how many lost their cache."""

    def __init__(self, base: str) -> None:
        super().__init__(daemon=True)
        self.base = base
        self.finished = threading.Event()
        self.peak_running = 0.0
        self.retracted = 0.0

    def run(self) -> None:
        first = None
        while not self.finished.wait(0.5):
            try:
                values = server_counters(self.base)
            except OSError:
                continue
            self.peak_running = max(self.peak_running, values.get("running", 0.0))
            first = values.get("retracted", 0.0) if first is None else first
            self.retracted = values.get("retracted", 0.0) - first


def measure(
    config: Config,
    args,
    text: str,
    context: int,
    concurrency: int,
    base: str,
    model: str,
) -> dict:
    """One llama-benchy cell. The prompt leaves room for the reply. A cell that repeats its prompts
    many times costs minutes, so large cells run once, with no warmup."""
    _, tokenizer = checkpoint_paths(config, args)
    body = {**(config.sampling or sampling(tokenizer)), **config.body}
    prompt = context - args.output_tokens - 16
    runs = max(1, min(args.runs, args.run_budget // (context * concurrency)))
    with tempfile.TemporaryDirectory() as directory:
        result = Path(directory) / "result.json"
        line = [
            *shlex.split(args.benchy),
            "--base-url",
            f"{base}/v1",
            "--model",
            model,
            "--served-model-name",
            model,
            "--tokenizer",
            str(tokenizer),
            "--pp",
            str(prompt),
            "--tg",
            str(args.output_tokens),
            "--concurrency",
            str(concurrency),
            "--runs",
            str(runs),
            "--warmup-runs",
            "1" if runs > 1 else "0",
            "--no-cache",
            "--latency-mode",
            "generation",
            "--skip-coherence",
            "--book-url",
            TEXTS[text],
            "--format",
            "json",
            "--save-result",
            str(result),
            "--extra-body",
            ",".join(f"{key}={json.dumps(value)}" for key, value in body.items()),
        ]
        before = draft_counters(base)
        started = time.monotonic()
        watcher = Watcher(base)
        watcher.start()
        run = subprocess.run(
            line, capture_output=True, text=True, timeout=args.run_timeout
        )
        watcher.finished.set()
        watcher.join()
        record = {
            "config": config.name,
            "preset": config.preset,
            "draft_tokens": config.draft_tokens,
            "text": text,
            "context": context,
            "concurrency": concurrency,
            "runs": runs,
            "prompt_tokens_requested": prompt,
            "output_tokens_requested": args.output_tokens,
            "sampling": body,
            "seconds": time.monotonic() - started,
            "returncode": run.returncode,
            "peak_running": watcher.peak_running,
            "retracted": watcher.retracted,
        }
        if result.exists():
            record["benchy"] = json.loads(result.read_text())
        else:
            record["error"] = run.stderr[-2000:] or run.stdout[-2000:]
        after = draft_counters(base)
        record["drafts"] = {key: after[key] - before.get(key, 0.0) for key in after}
        return record


def did_not_fit(record: dict) -> bool:
    """The cache could not hold every request at once. Below this size, a low peak is only a short
    run that the poll missed."""
    if record.get("retracted"):
        return True
    size = record["context"] * record["concurrency"]
    return size >= 1 << 18 and record.get("peak_running", 0) < record["concurrency"]


def run_config(config: Config, args) -> bool:
    """Start the server, measure every cell that is still missing, and stop the server. A cell at
    least as large as one that did not fit is recorded as skipped."""
    output = args.output / f"{config.name}.jsonl"
    done = {}
    if output.exists():
        for line in output.read_text().splitlines():
            record = json.loads(line)
            done[record["text"], record["context"], record["concurrency"]] = record
    cells = [
        (text, context, concurrency)
        for text in args.texts
        for context in args.contexts
        for concurrency in levels(config, args)
    ]
    if all(cell in done for cell in cells):
        print(f"[sweep] {config.name}: already done", flush=True)
        return True
    base = f"http://127.0.0.1:{args.port}"
    log = (args.output / f"{config.name}.log").open("a")
    print(f"[sweep] {config.name}: start", flush=True)
    top = max(args.contexts)
    while True:
        process = subprocess.Popen(
            server_command(config, args, top),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_until_ready(process, base, args.load_timeout)
            break
        except RuntimeError as error:
            stop(process)
            shorter = [context for context in args.contexts if context < top]
            if not shorter:
                print(f"[sweep] {config.name}: failed: {error}", flush=True)
                log.close()
                return False
            print(f"[sweep] {config.name}: no start at {top}: {error}", flush=True)
            top = max(shorter)
    try:
        model = json.loads(fetch(f"{base}/v1/models"))["data"][0]["id"]
        limit = {}
        for record in done.values():
            if did_not_fit(record):
                size = record["context"] * record["concurrency"]
                limit[record["text"]] = min(limit.get(record["text"], size), size)
        for text, context, concurrency in cells:
            if (text, context, concurrency) in done:
                continue
            size = context * concurrency
            reason = None
            if context > top:
                reason = "the server did not start at this context"
            elif size >= limit.get(text, size + 1):
                reason = "did not fit the cache"
            if reason:
                record = {
                    "config": config.name,
                    "text": text,
                    "context": context,
                    "concurrency": concurrency,
                    "skipped": reason,
                }
            else:
                record = measure(config, args, text, context, concurrency, base, model)
                if did_not_fit(record):
                    limit[text] = min(limit.get(text, size), size)
            with output.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"[sweep] {config.name}: {summary(record)}", flush=True)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
        print(f"[sweep] {config.name}: failed: {error}", flush=True)
        return False
    finally:
        stop(process)
        log.close()
    print(f"[sweep] {config.name}: done", flush=True)
    return True


def best_settings(config: Config, args) -> tuple[int | None, int | None, float | None]:
    """The draft tokens and request count with the highest decode throughput in the sweep, on the
    same preset and English at --gpqa-context, whose prompts are short like the questions. The
    settings must also hold --gpqa-max-tokens per request, the length an answer can reach. Draft
    tokens do not change the output tokens."""
    draft_tokens = {
        c.name: c.draft_tokens for c in CONFIGS if c.preset == config.preset
    }
    records = [
        record
        for path in sorted(args.output.glob("*.jsonl"))
        for record in map(json.loads, path.read_text().splitlines())
        if record["config"] in draft_tokens and record["text"] == "english"
    ]
    too_long = {
        (draft_tokens[record["config"]], record["concurrency"])
        for record in records
        if record["context"] == next(c for c in CONTEXTS if c >= args.gpqa_max_tokens)
        and ("skipped" in record or did_not_fit(record))
    }
    best = (None, None, None)
    for record in records:
        settings = (draft_tokens[record["config"]], record["concurrency"])
        if (
            record["context"] != args.gpqa_context
            or "skipped" in record
            or settings in too_long
        ):
            continue
        _, decode, _, errors = rates(record)
        if decode and not errors and (best[2] is None or decode > best[2]):
            best = (*settings, decode)
    return best


def run_gpqa(config: Config, connections: int, args) -> bool:
    """Score one model on GPQA Diamond at its highest throughput. The default temperature of 0.6 is
    the Artificial Analysis value for reasoning models whose lab recommends no other."""
    draft_tokens, concurrency, decode = best_settings(config, args)
    if concurrency is not None:
        config = replace(config, draft_tokens=draft_tokens)
        connections = concurrency
    if args.gpqa_connections:
        connections = args.gpqa_connections
    directory = args.output / "gpqa"
    if args.gpqa_subdir:
        directory = directory / args.gpqa_subdir
    output = directory / f"{config.name}.json"
    if output.exists():
        print(f"[gpqa] {config.name}: already done", flush=True)
        return True
    logs = directory / config.name
    logs.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    log = (directory / f"{config.name}.log").open("w")
    print(
        f"[gpqa] {config.name}: start, draft tokens {config.draft_tokens}, "
        f"{connections} requests at once, sweep decode {decode}",
        flush=True,
    )
    process = subprocess.Popen(
        server_command(config, args, args.gpqa_server_context),
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_until_ready(process, base, args.load_timeout)
        model = json.loads(fetch(f"{base}/v1/models"))["data"][0]["id"]
        stopped = sorted(logs.glob("*.json"))
        if stopped:  # keeps the answers it has, and its settings
            line = [
                *shlex.split(args.inspect),
                "eval-retry",
                str(stopped[-1].resolve()),
                "--log-dir",
                str(logs.resolve()),
                "--display",
                "plain",
            ]
        else:
            line = [
                *shlex.split(args.inspect),
                "eval",
                "inspect_evals/gpqa_diamond",
                "--model",
                f"openai-api/wright/{model}",
                "-M",
                f"client_timeout={args.gpqa_timeout}",
                "--temperature",
                str(args.gpqa_temperature),
                "--max-tokens",
                str(args.gpqa_max_tokens),
                "--epochs",
                str(args.gpqa_epochs),
                "--max-connections",
                str(connections),
                "--seed",
                "1",
                "--log-dir",
                str(logs.resolve()),
                "--log-format",
                "json",
                "--display",
                "plain",
            ]
            if args.gpqa_limit:
                line += ["--limit", str(args.gpqa_limit)]
            if args.gpqa_top_p is not None:
                line += ["--top-p", str(args.gpqa_top_p)]
            if args.gpqa_sample_id:
                line += ["--sample-id", args.gpqa_sample_id]
            if "reasoning_effort" in config.body:
                line += ["--reasoning-effort", config.body["reasoning_effort"]]
            extra = {k: v for k, v in config.body.items() if k != "reasoning_effort"}
            if extra:
                line += ["--extra-body", json.dumps(extra)]
        environment = {
            **os.environ,
            "WRIGHT_BASE_URL": f"{base}/v1",
            "WRIGHT_API_KEY": "EMPTY",
            "INSPECT_LOG_FORMAT": "json",
        }
        subprocess.run(
            line, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
        newest = max(logs.glob("*.json"), key=lambda path: path.stat().st_mtime)
        result = json.loads(newest.read_text())
        score = result["results"]["scores"][0]["metrics"]
        usage = result["stats"]["model_usage"]
        summary_row = {
            "config": config.name,
            "preset": config.preset,
            "status": result["status"],
            "accuracy": score["accuracy"]["value"],
            "stderr": score.get("stderr", {}).get("value"),
            "samples": result["results"].get("completed_samples"),
            "epochs": args.gpqa_epochs,
            "draft_tokens": config.draft_tokens,
            "connections": connections,
            "max_tokens": args.gpqa_max_tokens,
            "usage": usage,
            "log": newest.name,
        }
        output.write_text(json.dumps(summary_row, indent=2) + "\n")
        print(f"[gpqa] {config.name}: {summary_row['accuracy']:.3f}", flush=True)
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        KeyError,
        ValueError,
        OSError,
    ) as error:
        print(f"[gpqa] {config.name}: failed: {error}", flush=True)
        return False
    finally:
        stop(process)
        log.close()
    return True


def gpqa_report(args) -> None:
    rows = [
        (config, published, args.output / "gpqa" / f"{config.name}.json")
        for config, _, published in GPQA
    ]
    if not any(path.exists() for _, _, path in rows):
        return
    print("\n**GPQA Diamond**\n")
    print(
        "| Run | Draft tokens | Requests at once | Wright | Standard error | Artificial Analysis |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for config, published, path in rows:
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        aa = "No score" if published is None else f"{published:.1%}"
        print(
            f"| {config.name} | {row['draft_tokens']} | {row['connections']} "
            f"| {row['accuracy']:.1%} | {row['stderr'] or 0:.1%} | {aa} |"
        )


def rates(record: dict) -> tuple[float | None, float | None, float | None, int]:
    """Prefill tok/s, mean decode tok/s, peak decode tok/s in one second, and errors. The server
    sends its first chunk before the prefill starts, so llama-benchy's prefill rate for one request
    counts almost no time. For one request, prefill runs to the first token instead."""
    benchy = record.get("benchy")
    if not benchy:
        return None, None, None, 1
    row = benchy["benchmarks"][0]
    mean = lambda name: (row.get(name) or {}).get("mean")  # noqa: E731
    errors = row.get("error_count", 0) or 0
    prefill = mean("pp_throughput")
    if record["concurrency"] == 1:
        first_token_ms = mean("e2e_ttft")
        prefill = row["prompt_size"] * 1000 / first_token_ms if first_token_ms else None
    return prefill, mean("tg_throughput"), mean("peak_throughput"), errors


def accept_rate(record: dict) -> float | None:
    drafts = record.get("drafts") or {}
    proposed = sum(value for key, value in drafts.items() if "spec_draft" in key)
    accepted = sum(value for key, value in drafts.items() if "spec_accepted" in key)
    return accepted / proposed if proposed else None


def summary(record: dict) -> str:
    where = f"{record['text']} {record['context']} x{record['concurrency']}"
    if "skipped" in record:
        return f"{where}: skipped, {record['skipped']}"
    prefill, decode, peak, errors = rates(record)
    show = lambda value: "-" if value is None else f"{value:.0f}"  # noqa: E731
    return (
        f"{where}: prefill {show(prefill)} decode {show(decode)} peak {show(peak)} tok/s, "
        f"errors {errors}, "
        f"peak {record['peak_running']:.0f}, retracted {record['retracted']:.0f}"
    )


def report(args) -> None:
    """One table per configuration and text: a row per context length, a column per concurrency.
    Prefill and decode are totals across the requests of a run."""
    records: dict[str, list[dict]] = {}
    for path in sorted(args.output.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            records.setdefault(record["config"], []).append(record)
    for config in CONFIGS:
        rows = records.get(config.name)
        if not rows:
            continue
        columns = sorted({row["concurrency"] for row in rows})
        for text in args.texts:
            if not any(row["text"] == text for row in rows):
                continue
            print(
                f"\n**{config.name}, {text}: prefill / decode / peak decode tok/s**\n"
            )
            print("| Context | " + " | ".join(str(c) for c in columns) + " |")
            print("|---:|" + "---:|" * len(columns))
            for context in args.contexts:
                cells = []
                for concurrency in columns:
                    row = next(
                        (
                            r
                            for r in rows
                            if (r["text"], r["context"], r["concurrency"])
                            == (text, context, concurrency)
                        ),
                        None,
                    )
                    if row is None:
                        cells.append("Not measured")
                    elif "skipped" in row:
                        cells.append(
                            "No fit" if "fit" in row["skipped"] else "No start"
                        )
                    else:
                        prefill, decode, peak, errors = rates(row)
                        cells.append(
                            "Failed"
                            if prefill is None or decode is None or errors
                            else f"{prefill:.0f} / {decode:.0f} / {peak:.0f}"
                        )
                print(f"| {context // 1024}K | " + " | ".join(cells) + " |")
    gpqa_report(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--mistral", type=Path)
    parser.add_argument("--mistral-tokenizer", type=Path)
    parser.add_argument("--mistral-drafter", type=Path)
    parser.add_argument("--glm", type=Path)
    parser.add_argument(
        "--qwen35",
        type=Path,
        help="folder with one Qwen3.5 dense checkpoint per size, named as on Hugging Face, "
        "such as Qwen3.5-9B-FP8",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/sweep"))
    parser.add_argument(
        "--benchy",
        default="uvx --from git+https://github.com/eugr/llama-benchy@e9be344578cec17745066b220798b80a0d2686d3 llama-benchy",
        help="command that runs llama-benchy (default: %(default)s)",
    )
    parser.add_argument(
        "--contexts",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        default=CONTEXTS,
        help="comma-separated context lengths (default: powers of 2 from 1K to 128K)",
    )
    parser.add_argument(
        "--texts", type=lambda text: tuple(text.split(",")), default=tuple(TEXTS)
    )
    parser.add_argument(
        "--concurrency",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        help="comma-separated request counts (default: powers of 2 up to the preset's maximum batch)",
    )
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3, help="runs per cell, at most")
    parser.add_argument(
        "--run-budget",
        type=int,
        default=200_000,
        help="prompt tokens that one cell may repeat before it runs fewer times",
    )
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument(
        "--gpu",
        help="module IDs that the server may use, such as 0,1 (default: the server's)",
    )
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--load-timeout", type=float, default=3600.0)
    parser.add_argument("--run-timeout", type=float, default=3600.0)
    parser.add_argument("--only", help="comma-separated configuration names")
    parser.add_argument(
        "--gpqa", action="store_true", help="run GPQA Diamond in place of the sweep"
    )
    parser.add_argument(
        "--inspect",
        default="uvx --from inspect-evals==0.27.0 --with openai inspect",
        help="command that runs Inspect (default: %(default)s)",
    )
    parser.add_argument("--gpqa-epochs", type=int, default=4)
    parser.add_argument(
        "--gpqa-context",
        type=int,
        default=1024,
        help="context length whose sweep results choose the GPQA settings (default: %(default)s)",
    )
    parser.add_argument("--gpqa-max-tokens", type=int, default=16384)
    parser.add_argument(
        "--gpqa-server-context",
        type=int,
        help="server context length, which must hold the longest prompt plus --gpqa-max-tokens "
        "(default: the preset's)",
    )
    parser.add_argument(
        "--gpqa-timeout",
        type=int,
        default=7200,
        help="seconds one answer may take; a shorter limit retries long answers from the start "
        "(default: %(default)s)",
    )
    parser.add_argument("--gpqa-limit", type=int, help="ask only this many questions")
    parser.add_argument("--gpqa-temperature", type=float, default=0.6)
    parser.add_argument("--gpqa-top-p", type=float)
    parser.add_argument(
        "--gpqa-connections",
        type=int,
        help="requests at the same time (default: the fastest that fits, from the sweep)",
    )
    parser.add_argument(
        "--gpqa-sample-id",
        help="comma-separated questions to ask, in place of all of them",
    )
    parser.add_argument(
        "--gpqa-subdir",
        help="folder under gpqa/ for this run, which keeps it apart from the main results",
    )
    parser.add_argument(
        "--report", action="store_true", help="print the tables and stop"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args)
        return
    only = None if args.only is None else set(args.only.split(","))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if args.gpqa:
        failed = [
            config.name
            for config, connections, _ in GPQA
            if (only is None or config.name in only)
            and not run_gpqa(config, connections, args)
        ]
        if failed:
            raise SystemExit(f"[gpqa] failed: {', '.join(failed)}")
        return
    failed = [
        config.name
        for config in CONFIGS
        if (only is None or config.name in only) and not run_config(config, args)
    ]
    if failed:
        raise SystemExit(
            f"[sweep] failed: {', '.join(failed)}. Run again to retry them"
        )


if __name__ == "__main__":
    main()
