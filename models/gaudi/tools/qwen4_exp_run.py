"""Qwen3.8-Flash-Next through ``Qwen4ExpGaudi`` on R cards: greedy generation from a prompt, with timings.

usage: qwen4_exp_run.py CHECKPOINT [--ranks R] [--layers N] [--no-table] [--tokens K] [--prompt TEXT] [--raw]
                        [--batch B[,B..]] [--replicate-dense]

The prompt is rendered as one user turn of the Qwen chat template (``--raw``: tokenized as is), prefilled, then K
tokens are decoded greedily; with ``--batch B`` the same prompt runs as B rows of one decode batch (the step time
is per batch; several sizes run one after the other). Rank 0 prints the text, the prefill time and the decode
rate; ``RESIN_FNEXT_PROFILE=1`` adds the host / device split of a decode step. The cards come from
``VT_HPU_CARDS`` (a queue script's grant), else the first R accel devices. One process per card.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path


def _worker(rank: int, port: int, cards: list[str], args: dict, out: str) -> None:
    import numpy as np
    from tokenizers import Tokenizer

    from resin_models_gaudi import ranks as par
    from resin_models_gaudi.qwen4_exp.model import Qwen4ExpGaudi
    from resin_models_gaudi.synapse.runtime import open_device

    n = len(cards)
    if n > 1:
        par.init_gloo(rank, n, port)
    dev = open_device(par.accel_index(cards[rank]))
    if n > 1:
        par.hccl_rendezvous(dev, rank, n)
    tk = Tokenizer.from_file(f"{args['checkpoint']}/tokenizer.json")
    text = args["prompt"]
    if not args["raw"]:
        text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ids = tk.encode(text, add_special_tokens=False).ids
    log = print if rank == 0 else (lambda _: None)
    batches = args["batches"]
    t0 = time.time()
    m = Qwen4ExpGaudi(
        dev,
        args["checkpoint"],
        layers=args["layers"],
        table=not args["no_table"],
        max_positions=2048,
        parent_capacity=3 * max(batches)
        + 1,  # a 2048-token test row: up to 2 KV slabs and a STATE slab
        rank=rank,
        nranks=n,
        replicate_dense=args["replicate_dense"],
        decode_buckets=tuple(sorted(set(batches))),
        log=log,
    )
    if n > 1:
        par.barrier()
    log(f"model up in {time.time() - t0:.0f} s; prompt {len(ids)} tokens")
    if os.environ.get("RESIN_FNEXT_PROFILE") == "1":
        m.profile = []
    eos = m.cfg.eos_token_id
    results = []
    for B in batches:
        rows = [m.sequence_rows(b, 2048) for b in range(B)]
        t0 = time.time()
        toks = [m.prefill(rows[b], ids) for b in range(B)]
        dev.sync()
        prefill_s = (time.time() - t0) / B
        rows = [r.advanced(len(ids), ids) for r in rows]
        outs = [[t] for t in toks]
        t0 = time.time()
        steps = 0
        for _ in range(args["tokens"] - 1):
            nxt = m.decode(rows, toks)
            rows = [r.advanced(1, [t]) for r, t in zip(rows, toks, strict=True)]
            toks = [int(t) for t in nxt]
            for b in range(B):
                outs[b].append(toks[b])
            steps += 1
            if B == 1 and toks[0] == eos:
                break
        dev.sync()
        decode_s = time.time() - t0
        result = {
            "batch": B,
            "prompt_tokens": len(ids),
            "prefill_s": prefill_s,
            "decode_steps": steps,
            "decode_ms_per_step": 1000 * decode_s / max(steps, 1),
            "tokens_per_s": steps * B / decode_s if decode_s else 0.0,
            "tokens": outs[0],
            "text": tk.decode(outs[0], skip_special_tokens=False),
            "compiles": m.compiles_after_start,
        }
        if m.profile:
            dec = np.array([p[1:] for p in m.profile if p[0] == B])
            result["decode_profile_ms"] = dict(
                zip(
                    ("tables", "launches", "wait"),
                    (1000 * dec.mean(0)).round(3).tolist(),
                    strict=True,
                )
            )
            m.profile = []
        results.append(result)
    if rank == 0:
        Path(out).write_text(json.dumps(results))
    if n > 1:
        par.barrier()
    dev.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--batch",
        default="1",
        help="rows of a decode batch, or several comma-separated: each runs",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function that parses an ISO 8601 date string.",
    )
    parser.add_argument("--raw", action="store_true")
    parser.add_argument(
        "--replicate-dense",
        action="store_true",
        help="the dense weights whole on every rank",
    )
    parser.add_argument("--out", default=None, help="write the result JSON here too")
    a = parser.parse_args()
    from resin.process import spawn
    from resin_models_gaudi import features

    if features.enabled("skipvalidation"):  # (RESIN_GAUDI_FEATURES) the ranks inherit the preload
        from deploy.gaudi_cli import preload_skip_validation

        preload_skip_validation()
    picked = re.split(r"[ ,]+", os.environ.get("VT_HPU_CARDS", "").strip())
    if picked != [""]:
        entries = [Path(f"/sys/class/accel/accel{i}") for i in picked][: a.ranks]
    else:
        entries = sorted(Path("/sys/class/accel").glob("accel[0-9]*"))[: a.ranks]
    if len(entries) != a.ranks:
        sys.exit(f"{a.ranks} cards needed, {len(entries)} available")
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = a.out or f"/tmp/qwen4_exp_run_{os.getpid()}.json"
    args = dict(
        checkpoint=a.checkpoint, layers=a.layers, no_table=a.no_table, tokens=a.tokens, prompt=a.prompt, raw=a.raw,
        replicate_dense=a.replicate_dense, batches=[int(b) for b in a.batch.split(",")],
    )  # fmt: skip
    spawn(_worker, args=(port, cards, args, out), devices=[f"pci:{c}" for c in cards])
    for result in json.loads(Path(out).read_text()):
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("text", "tokens")},
                indent=1,
            )
        )
    print("---")
    print(json.loads(Path(out).read_text())[0]["text"])


if __name__ == "__main__":
    main()
