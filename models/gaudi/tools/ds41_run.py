"""DeepSeek-V4.1 steps through ``DeepSeekV41Gaudi`` on N cards: chunked prefill and decode agree.

usage: ds41_run.py CHECKPOINT IDS [--layers N] [--ranks R] [--engram host|card]
                   [--chunks 128,37] [--decode K] [--batch] [--ref REF.npz] [--time]

IDS is a file of token ids. The prompt is prefilled in chunks of each size in ``--chunks`` (each from an
empty cache); with ``--decode K`` its first n - K tokens are prefilled and the last K decoded one at a time
(teacher-forced); with ``--batch`` the decode runs as one row of a two-row batch beside a second prompt. Every
run's logits are compared with the first chunk size's, and with REF (a ds41_ref.py dump of the same layers,
``--last`` covering the prompt) when given. ``--time`` times decode steps.
Each rank takes the first free card that HABANA_VISIBLE_MODULES allows.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from resin.models.deepseek_v41 import DeepSeekV41Config
from resin_models_gaudi.deepseek_v41.cache import HISTORY_TOKENS, PAGE_TOKENS
from resin_models_gaudi.deepseek_v41.model import DeepSeekV41Gaudi, Row
from resin_models_gaudi.synapse.graph import RecipeCache
from resin_models_gaudi.synapse.runtime import open_device


class Sequence:
    """One sequence's cache blocks from a free list of slabs (every block kept)."""

    def __init__(
        self, model: DeepSeekV41Gaudi, free: list[int], tokens: np.ndarray
    ) -> None:
        self.model, self.free, self.tokens = model, free, tokens
        self.window: list[int] = []
        self.history: list[tuple[int, int]] = []

    def ensure(self, end: int) -> None:
        geo = self.model.geo
        while len(self.window) * PAGE_TOKENS < end:
            self.window.append(self.free.pop())
        per = geo.slab_rows // geo.history_block_rows
        while len(self.history) * HISTORY_TOKENS < end:
            k = len(self.history)
            if k % per == 0:
                self._slab = self.free.pop()
            self.history.append((self._slab, k % per))

    def row(self, column: int, pos0: int, real: int) -> Row:
        self.ensure(pos0 + real)
        geo = self.model.geo
        slabs = np.array(self.window)
        hs = np.array(self.history).reshape(-1, 2)
        return Row(
            column, pos0, real,
            geo.window_block_row(slabs, 0), geo.window_block_id(slabs, 0), 0,
            geo.history_block_row(hs[:, 0], hs[:, 1]),
            self.tokens[max(pos0 - 3, 0) : pos0],
        )  # fmt: skip

    def release(self) -> None:
        self.free.extend(self.window)
        self.free.extend({s for s, _ in self.history})


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("ids")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--engram", default="host")
    parser.add_argument("--chunks", default="128")
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--ref")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    args = parser.parse_args()
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
    if any(p.exitcode for p in procs):
        raise SystemExit("a rank failed")


def run(rank: int, nranks: int, args) -> None:
    say = print if rank == 0 else (lambda *a, **k: None)
    cfg = DeepSeekV41Config.from_hugging_face(
        json.load(open(os.path.join(args.checkpoint, "config.json")))
    )
    prompt = np.array([int(x) for x in open(args.ids).read().split()], np.int64)
    device = open_device()
    barrier = None
    if nranks > 1:
        from resin_models_gaudi.mistral_large4.parallel import barrier as gloo_barrier
        from resin_models_gaudi.mistral_large4.parallel import (
            hccl_rendezvous,
            init_gloo,
        )

        init_gloo(rank, nranks, 29643)
        hccl_rendezvous(device, rank, nranks)
        barrier = gloo_barrier
    model = DeepSeekV41Gaudi(
        device, args.checkpoint, cfg, capacity=256, max_seq_len=args.max_seq_len, rank=rank, nranks=nranks,
        layers=args.layers, engram=args.engram, barrier=barrier,
        recipes=RecipeCache(device, os.path.expanduser("~/.cache/resin/ds41-recipes")),
        log=say,
    )  # fmt: skip
    free = list(range(model.capacity, 0, -1))
    n = len(prompt)

    def prefill(seq: Sequence, upto: int, chunk: int) -> np.ndarray:
        out = []
        for s in range(0, upto, chunk):
            m = min(chunk, upto - s)
            row = seq.row(0, s, m)
            out.append(
                model.forward(
                    seq.tokens[s : s + m], s + np.arange(m), [row], prefill=True
                )
            )
        return np.concatenate(out)

    results: dict[str, np.ndarray] = {}
    for chunk in [int(c) for c in args.chunks.split(",")]:
        seq = Sequence(model, free, prompt)
        t0 = time.time()
        results[f"chunks of {chunk}"] = prefill(seq, n, chunk)
        say(
            f"prefill {n} tokens in chunks of {chunk}: {time.time() - t0:.1f} s",
            flush=True,
        )
        seq.release()
    if args.decode:
        K = args.decode
        seq = Sequence(model, free, prompt)
        logits = [prefill(seq, n - K, 128)]
        other = None
        if args.batch:
            other = Sequence(model, free, prompt[::-1].copy())
            prefill(other, n - K, 128)
        times = []
        for p in range(n - K, n):
            rows = [seq.row(0, p, 1)]
            toks = [prompt[p]]
            if other is not None:
                rows.append(other.row(1, p, 1))
                toks.append(other.tokens[p])
            t0 = time.perf_counter()
            out = model.forward(np.array(toks), np.array([p] * len(rows)), rows)
            times.append(time.perf_counter() - t0)
            logits.append(out[:1])
        results[
            f"prefill {n - K} + decode {K}" + (" (batch of 2)" if other else "")
        ] = np.concatenate(logits)
        say(
            f"decode steps: median {np.median(times) * 1e3:.1f} ms (host wall time, synchronous)",
            flush=True,
        )
        seq.release()
        if other is not None:
            other.release()
    if rank:
        return
    names = list(results)
    first = results[names[0]]
    for name in names:
        got = results[name]
        rel = np.linalg.norm(got - first, axis=1) / np.linalg.norm(first, axis=1)
        same = int((got.argmax(1) == first.argmax(1)).sum())
        say(
            f"{name:40s} vs {names[0]}: argmax {same}/{n}, logits rel max {rel.max():.2e} mean {rel.mean():.2e}"
        )
    if args.ref:
        ref = np.load(args.ref)
        want = ref["logits"][:n]
        for name in names:
            got = results[name][: len(want)]
            rel = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
            same = int((got.argmax(1) == ref["argmax"][: len(got)]).sum())
            lp = got - got.max(1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(1, keepdims=True))
            lq = want - want.max(1, keepdims=True)
            lq -= np.log(np.exp(lq).sum(1, keepdims=True))
            kl = (np.exp(lq) * (lq - lp)).sum(1)
            say(f"{name:40s} vs reference: argmax {same}/{len(got)}, logits rel mean {rel.mean():.2e}, "
                f"KL mean {kl.mean():.3f} median {np.median(kl):.4f}")  # fmt: skip
        tail = ref["argmax"][: n - 1]
        say(f"reference next-token accuracy {(tail == prompt[1:n]).mean():.3f}; "
            f"Wright {(first.argmax(1)[: n - 1] == prompt[1:n]).mean():.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
