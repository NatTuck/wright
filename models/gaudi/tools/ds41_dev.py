"""A warm DeepSeek-V4.1 process: the weights stay on the cards while scripts run against them.

usage: ds41_dev.py CHECKPOINT DIR [--layers N] [--ranks R | --slice R/N] [--engram host|card]
                   [--profile PDIR] [--dspark]

Each rank loads the weights once, then runs every new DIR/*.py in name order with ``exec``. A script sees
``device``, ``cfg``, ``rank``, ``nranks``, ``weights`` (name -> device address), ``build()`` (a fresh
``DeepSeekV41Gaudi`` over those weights from the current source: modules are reloaded first) and ``say`` (rank
0's print). Rank 0 writes DIR/<script>.out; a script's exception is printed and the process carries on. DIR/stop
ends it. ``--slice R/N`` loads rank R of N on one card with no collectives (scripts launch the graphs
themselves); ``--profile`` turns the Synapse profiler on (``device.prof_start`` / ``prof_stop``).
Each rank takes the first free card that HABANA_VISIBLE_MODULES allows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("dir")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--engram", default="host")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--slice")
    parser.add_argument("--profile")
    parser.add_argument("--dspark", action="store_true", help="load the DSpark stages too")
    args = parser.parse_args()
    if args.slice:
        rank, nranks = (int(x) for x in args.slice.split("/"))
        run(rank, nranks, args)
        return
    if args.ranks == 1:
        run(0, 1, args)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=run, args=(r, args.ranks, args)) for r in range(args.ranks)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def run(rank: int, nranks: int, args) -> None:
    import numpy as np

    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin_models_gaudi.synapse.graph import RecipeCache
    from resin_models_gaudi.synapse.runtime import open_device

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DeepSeekV41Config.from_hugging_face(
        json.load(open(os.path.join(args.checkpoint, "config.json")))
    )
    if args.profile:
        from resin_models_gaudi.synapse.trace import enable_profiler

        enable_profiler(args.profile)
    device = open_device()
    barrier = None
    if nranks > 1 and not args.slice:
        from resin_models_gaudi.mistral_large4 import parallel

        parallel.init_gloo(rank, nranks, 29645)
        parallel.hccl_rendezvous(device, rank, nranks)
        barrier = parallel.barrier
    recipes = RecipeCache(device, os.path.expanduser("~/.cache/resin/ds41-recipes"))
    log = (
        (lambda m: print(m, flush=True))
        if rank == 0 or args.slice
        else (lambda m: None)
    )

    def make(weights=None, **options):
        from resin_models_gaudi.deepseek_v41 import cache, layers, model, rope
        from resin_models_gaudi.deepseek_v41 import weights as wmod

        for m in (wmod, rope, cache, layers, model):
            importlib.reload(m)
        from resin_models_gaudi.deepseek_v41.model import DeepSeekV41Gaudi

        return DeepSeekV41Gaudi(
            device, args.checkpoint, cfg, capacity=args.capacity, max_seq_len=args.max_seq_len, rank=rank,
            nranks=nranks, layers=args.layers, engram=args.engram, weights=weights, barrier=barrier, recipes=recipes,
            log=log, **{"dspark": args.dspark, **options},
        )  # fmt: skip

    first = make()
    weights = first.w
    log(f"ready: {len(weights)} tensors; drop scripts in {out}")
    done: set[str] = set(p.name for p in out.glob("*.py"))
    while not (out / "stop").exists():
        new = sorted(p for p in out.glob("*.py") if p.name not in done)
        if not new:
            time.sleep(1)
            continue
        for script in new:
            done.add(script.name)
            lines: list[str] = []

            def say(*a, **k):
                if rank == 0 or args.slice:
                    lines.append(" ".join(str(x) for x in a))
                    print(*a, flush=True)

            env = {
                "np": np, "device": device, "cfg": cfg, "rank": rank, "nranks": nranks, "weights": weights,
                "build": lambda **options: make(weights, **options), "say": say, "first": first, "args": args,
            }  # fmt: skip
            t0 = time.time()
            try:
                exec(compile(script.read_text(), str(script), "exec"), env)
            except Exception:
                say(traceback.format_exc())
            say(f"[{script.name} {time.time() - t0:.1f} s]")
            if rank == 0 or args.slice:
                (out / (script.name + ".out")).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
