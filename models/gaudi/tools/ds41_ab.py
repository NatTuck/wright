"""DeepSeek-V4.1 scenarios through Resin's Engine, for A/B runs of two builds (e.g. the Python step path and the
native executor): every request's tokens, logprobs and top lists, and every runner call's kind and shape, as JSON.

usage (run with the venv whose resin you test; card jobs through the card lock script):

    .venv/bin/python ds41_ab.py --fake --out A.json          # one card as rank 0 of 8
    .venv/bin/python ds41_ab.py --ranks 8 --out B.json       # the checkpoint on 8 cards
    python ds41_ab.py --compare A.json B.json

``--fake`` uses ``ds41_fake``'s placeholder weights (no checkpoint read). With ``--ranks 8`` every rank runs the
same plans (Gloo groups and the socket step channel at startup, as ``wright`` serves); rank 0 writes the JSON.
Two runs are comparable when their call shapes agree (the kind, phase, rows and tokens of every runner call):
bucket-dependent numerics make outputs of different shapes differ.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from datetime import timedelta
from pathlib import Path

CHECKPOINT = Path(os.environ.get("DS41_CHECKPOINT", "DeepSeek-V4.1-Flash"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def corpus(tokenizer) -> list[int]:
    root = Path(__file__).resolve().parents[3]
    text = "\n\n".join(p.read_text() for p in sorted((root / "docs").glob("*.md")))
    text += "\n\n" + (root / "README.md").read_text()
    return tokenizer.encode(text).ids


def scenarios(text: list[int], draft_tokens: int):
    """(name, [(prompt, params kwargs)], together): requests submitted together or one after the other."""
    greedy = dict(temperature=0.0)
    scored = dict(temperature=0.0, logprobs=True, top_logprobs=3)
    sampled = dict(
        temperature=0.8, top_p=0.9, top_k=40, seed=7, logprobs=True, top_logprobs=2
    )
    out = [
        ("greedy alone", [(text[0:37], dict(greedy, max_output_tokens=24)),
                          (text[100:300], dict(greedy, max_output_tokens=24)),
                          (text[300:1351], dict(greedy, max_output_tokens=24)),
                          (text[400:2500], dict(greedy, max_output_tokens=24)),
                          (text[0:5199], dict(greedy, max_output_tokens=16))], False),
        ("three together", [(text[i * 50: i * 50 + 700], dict(scored, max_output_tokens=32)) for i in range(3)],
         True),
        ("eight together", [(text[i * 211: i * 211 + 400 + 37 * i], dict(greedy, max_output_tokens=20))
                            for i in range(8)], True),
        ("sampled", [(text[900:1900], dict(sampled, max_output_tokens=24)),
                     (text[2000:2100], dict(sampled, seed=11, max_output_tokens=24))], True),
        ("prefix hit", [(text[0:5199], dict(greedy, max_output_tokens=16)),
                        (text[0:1054], dict(greedy, max_output_tokens=8)),
                        (text[0:1054], dict(greedy, max_output_tokens=8))], False),
        ("long decode", [(text[3000:4000], dict(greedy, max_output_tokens=300))], False),
        ("wide", [(text[i * 37: i * 37 + 64], dict(greedy, max_output_tokens=12)) for i in range(40)], True),
        ("prompt logprobs", [(text[0:300], dict(greedy, max_output_tokens=2, prompt_logprobs=True,
                                                 prompt_top_logprobs=3)),
                             (text[500:2600], dict(greedy, max_output_tokens=2, prompt_logprobs=True,
                                                   prompt_top_logprobs=0))], False),
        # prefill time alone: one request at a time, one token each (not in a default run: --only)
        ("prefill 2K", [(text[i * 97: i * 97 + 2048], dict(greedy, max_output_tokens=1)) for i in range(12)],
         False),
        ("prefill 8K", [(text[i * 97: i * 97 + 8192], dict(greedy, max_output_tokens=1)) for i in range(6)],
         False),
    ]  # fmt: skip
    return out


TIMED = ("prefill 2K", "prefill 8K")


class Recorder:
    """Wraps a runner's forward and enqueue: each call's kind, phase, rows and tokens."""

    def __init__(self, runner) -> None:
        self.calls: list[list] = []
        forward, enqueue = runner.forward, runner.enqueue

        def record(kind, plan, feedback=None):
            self.calls.append([kind, str(plan.phase).rsplit(".", 1)[-1], plan.num_sequences,
                               len(plan.token_ids), feedback is not None])  # fmt: skip

        def fwd(plan, **kwargs):
            if not kwargs:  # the Engine's call (a runner's own fallback passes its decision)
                record("forward", plan)
            return forward(plan, **kwargs)

        def enq(plan, *, feedback=None):
            record("enqueue", plan, feedback)
            return enqueue(plan, feedback=feedback)

        runner.forward, runner.enqueue = fwd, enq


def generate(engine, requests, together: bool) -> list[dict]:
    from resin import SamplingParams

    out = []

    def run(batch):
        leases = [
            engine.submit(list(p), sampling_params=SamplingParams(**kw))
            for p, kw in batch
        ]
        logs = {lease.request_id: [] for lease in leases}
        while engine.has_work():
            for event in engine.step():
                if event.logprobs is not None:
                    lp = event.logprobs
                    top = [[int(t), float(v)] for t, v in (lp.top_logprobs or ())]
                    logs[event.request_id].append([float(lp.logprob), top])
        for lease in leases:
            result = lease.take_result()
            prompt = []
            for entry in getattr(lease, "prompt_logprobs", None) or ():
                if entry is None:
                    prompt.append(None)
                else:
                    top = [[int(t), float(v)] for t, v in (entry.top_logprobs or ())]
                    prompt.append([float(entry.logprob), top])
            out.append({"tokens": [int(t) for t in result], "logprobs": logs[lease.request_id], "prompt": prompt,
                        "cached": int(getattr(lease, "cached_tokens", 0) or 0)})  # fmt: skip

    if together:
        run(requests)
    else:
        for request in requests:
            run([request])
    return out


def build(args, rank: int, groups, native_ranks=None):
    import ds41_fake

    from resin_models_gaudi.deepseek_v41.runner import DeepSeekV41GaudiRunner

    if args.fake:
        ds41_fake.patch_runner()
        if args.ranks == 1:
            ds41_fake.one_rank_of(args.rank, 8)
    cards = os.environ.get("VT_HPU_CARDS")
    card = int(cards.split(",")[rank]) if cards else None  # accel indices
    return DeepSeekV41GaudiRunner(
        str(args.checkpoint), groups=groups, card=card, max_batch_size=64,
        max_seq_len=args.max_seq_len, parent_capacity=args.slabs, draft_tokens=args.draft_tokens,
        replay=args.replay, **({} if native_ranks is None else {"native_ranks": native_ranks}),
    )  # fmt: skip


def run(
    rank: int,
    args,
    port: int | None = None,
    path: str | None = None,
    token: bytes = b"",
) -> None:
    from tokenizers import Tokenizer

    from resin import Engine, Parallelism

    groups = transport = native = None
    if args.ranks > 1 and args.native_ranks:
        from resin._resin import GaudiRanks

        native = GaudiRanks(path, token, rank, args.ranks, timeout=3600.0)
    elif args.ranks > 1:
        import torch.distributed as dist

        from resin.backend.step_transport import create_step_transport
        from resin.backend.topology import ParallelGroups

        timeout = timedelta(minutes=60)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=args.ranks, timeout=timeout)  # fmt: skip
        topology = Parallelism(
            world_size=args.ranks, attn_dp=1, attn_tp=args.ranks, ep=1
        )
        groups = ParallelGroups.create(topology, device_backend="gloo", timeout=timeout)
        transport = create_step_transport(
            groups, "socket", timeout=timeout.total_seconds()
        )
    t0 = time.time()
    runner = build(args, rank, groups, native)
    built = time.time() - t0
    if (
        args.fixed is not None
    ):  # placeholder weights accept nothing: a fixed acceptance commits drafts
        runner._speculation.benchmark_accepted_drafts = args.fixed
    executor = getattr(runner, "executor", None)
    if native is not None:
        if rank > 0:
            calls = runner.serve()
            print(f"rank {rank}: served {calls} calls", flush=True)
            runner.close()
            return
        recorder = Recorder(runner)
        engine = Engine(runner, max_running_requests=64, context_length=args.max_seq_len,
                        prefix_cache=True, enable_overlap=True)  # fmt: skip
    elif args.ranks > 1:
        from resin.backend.distributed import DistributedEngine, DistributedWorker

        worker = DistributedWorker(runner, groups, transport=transport)
        if not groups.is_driver:
            worker.serve()
            return
        recorder = Recorder(runner)
        engine = DistributedEngine(worker, max_running_requests=64, context_length=args.max_seq_len,
                                   prefix_cache=True, enable_overlap=True)  # fmt: skip
    else:
        recorder = Recorder(runner)
        engine = Engine(runner, max_running_requests=64, context_length=args.max_seq_len,
                        prefix_cache=True, enable_overlap=True)  # fmt: skip
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    text = corpus(tokenizer)
    result = {"build_s": built, "executor": executor is not None, "scenarios": {}}
    for name, requests, together in scenarios(text, args.draft_tokens):
        if args.only and name not in args.only or not args.only and name in TIMED:
            continue
        start, calls = time.time(), len(recorder.calls)
        before = dict(executor.timings) if executor is not None else {}
        result["scenarios"][name] = {
            "outputs": generate(engine, requests, together),
            "calls": recorder.calls[calls:],
            "seconds": time.time() - start,
        }
        if executor is not None:  # the driver's executor time in this scenario, ns
            after = executor.timings
            result["scenarios"][name]["timings"] = {k: after[k] - before.get(k, 0) for k in after}
        print(f"{name}: {time.time() - start:.1f} s", flush=True)
    result["compiles"] = runner.compiles_after_start
    result["speculation"] = runner.speculative_stats
    if executor is not None:
        result["timings"] = dict(executor.timings)
        result["native_steps"] = executor.native_steps
    engine.close()
    Path(args.out).write_text(json.dumps(result))
    runner.close()


def ulps(a, b) -> str:
    """How far two prompt-logprob lists are apart: the largest f32 ULP distance of a logprob, and whether the
    top lists' ids agree."""
    import numpy as np

    worst, ids = 0, True
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            ids &= x is y
            continue
        values = [x[0]] + [v for _, v in x[1]], [y[0]] + [v for _, v in y[1]]
        u = np.array(values, np.float32).view(np.int32).astype(np.int64)
        worst = max(worst, int(np.abs(u[0] - u[1]).max()))
        ids &= [t for t, _ in x[1]] == [t for t, _ in y[1]]
    return f"max {worst} ulp, top ids {'equal' if ids else 'differ'}"


def compare(a: str, b: str) -> int:
    """Exit status 0 when every scenario's call shapes and outputs agree."""
    A, B = (json.loads(Path(p).read_text())["scenarios"] for p in (a, b))
    bad = 0
    for name in A:
        if name not in B:
            print(f"{name}: only in {a}")
            continue
        sa, sb = A[name], B[name]
        if sa["calls"] != sb["calls"]:
            first = next(i for i, (x, y) in enumerate(zip(sa["calls"], sb["calls"])) if x != y) \
                if any(x != y for x, y in zip(sa["calls"], sb["calls"])) else min(len(sa["calls"]), len(sb["calls"]))  # fmt: skip
            print(f"{name}: call shapes differ at call {first}: "
                  f"{sa['calls'][first:first + 2]} vs {sb['calls'][first:first + 2]}")  # fmt: skip
            bad += 1
            continue
        for i, (x, y) in enumerate(zip(sa["outputs"], sb["outputs"], strict=True)):
            if x["tokens"] != y["tokens"]:
                part = next(
                    (
                        j
                        for j, (s, t) in enumerate(zip(x["tokens"], y["tokens"]))
                        if s != t
                    ),
                    None,
                )
                print(f"{name} request {i}: tokens differ from {part}")
                bad += 1
            elif x["logprobs"] != y["logprobs"]:
                print(f"{name} request {i}: logprobs differ")
                bad += 1
            elif x.get("prompt") != y.get("prompt"):
                print(
                    f"{name} request {i}: prompt logprobs differ: {ulps(x['prompt'], y['prompt'])}"
                )
                bad += 1
        print(f"{name}: {len(sa['calls'])} calls, {len(sa['outputs'])} requests"
              f"{'' if bad else ' identical'}")  # fmt: skip
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument(
        "--fake", action="store_true", help="placeholder weights (ds41_fake)"
    )
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument(
        "--native-ranks", action="store_true", help="the native rank group (no Gloo)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="one card: stand for this rank of 8 (--fake)",
    )
    parser.add_argument("--slabs", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--draft-tokens", type=int, default=0)
    parser.add_argument("--replay", type=int, default=None)
    parser.add_argument(
        "--fixed",
        type=int,
        help="accept this many drafts of every row (benchmark mode)",
    )
    parser.add_argument("--only", nargs="*", help="run these scenarios only")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        raise SystemExit(compare(*args.compare))
    if args.ranks == 1:
        run(0, args)
        return
    from resin.process import spawn

    cards = os.environ["VT_HPU_CARDS"].split(",")[: args.ranks]
    bdf = [
        os.path.basename(os.path.realpath(f"/sys/class/accel/accel{i}/device"))
        for i in cards
    ]
    port = 20000 + secrets.randbelow(20000)
    import tempfile

    path = os.path.join(tempfile.mkdtemp(prefix="ds41-ab-"), "group")
    spawn(
        _worker,
        args=(args, port, path, secrets.token_bytes(16)),
        devices=[f"pci:{b}" for b in bdf],
    )


def _worker(rank: int, args, port: int, path: str, token: bytes) -> None:
    run(rank, args, port, path, token)


if __name__ == "__main__":
    main()
