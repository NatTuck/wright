"""DeepSeek-V4.1-Flash on Gaudi with placeholder weights: the real config, tokenizer, shapes, tables, cache, graphs
and recipes, and no weight reads (a build takes about a minute once the recipes are cached).

Every tensor that a recipe binds and the checkpoint would provide is allocated on the card when the model binds
it, and filled whole with values that depend only on its name and the rank, so two processes get the same bytes:
FP8 and FP4 codes and E8M0 scales in ranges that keep the 40 layers finite, f32 norms at 1, the rest small.
Engram tables keep each rank's real row count (hash ids hit the rows a rank holds, as with the checkpoint).
Random weights accept no drafts.

Use from a harness::

    from ds41_fake import FakeDeepSeekV41Gaudi, fake_runner, patch_runner
    model = FakeDeepSeekV41Gaudi(device, CHECKPOINT, cfg, capacity=slabs, rank=r, nranks=8, ...)
    patch_runner()  # DeepSeekV41GaudiRunner builds FakeDeepSeekV41Gaudi from here on
    runner = fake_runner(groups=..., parent_capacity=slabs, draft_tokens=3)

One card can stand for rank r of 8 (``--rank r``): the shapes, slices and tables of that rank, with a one-rank
HCCL communicator, so each all-reduce leaves the rank's partial and each all-gather fills the rank's part only
(the other ranks' parts stay zero). ``--ranks 8`` spawns eight processes with real HCCL.

usage (run with the venv whose resin you test; every card job through the card lock script):

    .venv/bin/python ds41_fake.py [--rank 0]      # one card
    .venv/bin/python ds41_fake.py --ranks 8       # 8 cards

It builds the runner (its native executor and native rank group with ``--ranks``), runs prompts of 37, 1051 and
2100 tokens and decode steps of 1, 8 and 64 rows through the Engine (overlapped; DSpark steps at 1 and 2 rows with
``--draft-tokens``), prints each kind's median ms per Engine step, and ends with a digest of every token (equal
digests: identical runs).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

CHECKPOINT = Path(os.environ.get("DS41_CHECKPOINT", "DeepSeek-V4.1-Flash"))
POOL_BYTES = (
    64 << 20
)  # each kind's device pool of random values; a tensor copies from it at a seeded offset


# ------------------------------------------------------------------ values
def _seed(*parts) -> int:
    return int.from_bytes(
        hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "little"
    )


def _bf16(v: np.ndarray) -> np.ndarray:
    bits = np.asarray(v, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _pool(kind: str) -> np.ndarray:
    """POOL_BYTES of valid random values of ``kind`` (the same bytes in every process)."""
    rng = np.random.default_rng(_seed("pool", kind))
    n = POOL_BYTES
    if (
        kind == "fp8"
    ):  # E4M3 bias 7: sign, exponent 6 or 7, any mantissa: +-[0.5, 1.875]
        sign = rng.integers(0, 2, n, dtype=np.uint8) << 7
        exp = rng.integers(6, 8, n, dtype=np.uint8) << 3
        return sign | exp | rng.integers(0, 8, n, dtype=np.uint8)
    if kind == "u8":  # E2M1 codes, two per byte: every nibble is a number
        return rng.integers(0, 256, n, dtype=np.uint8)
    if kind == "bf16":
        return _bf16(rng.uniform(-1, 1, n // 2) * 2.0**-5).view(np.uint8)
    if kind == "bf16_emb":
        return _bf16(rng.uniform(-1, 1, n // 2)).view(np.uint8)
    if kind == "f32":
        return (rng.uniform(-1, 1, n // 4) * 2.0**-6).astype(np.float32).view(np.uint8)
    raise ValueError(kind)


# expert dequant: E8M0 scale bytes E8M0 and exponent K, so that codes decode to code * 2^(E8M0 - 127 - K) =
# code * 32 in FP8 (|code| <= 6: at most 192) and ``.c`` = 2^K scales the GEMM's output back
E8M0, K = 121, -11
_F32_ONE, _F32 = 0x3F800000, lambda x: int(np.float32(x).view(np.uint32))


def kind_of(name: str, dtype: int) -> tuple[str, int | None]:
    """How to fill weight ``name`` of synapse dtype ``dtype``: (random pool kind, None) or ("const", 32-bit
    pattern)."""
    from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, I64, U8

    base = name.rsplit(".", 1)[-1] if name.startswith("l") and "." in name else name
    if dtype == F8 or base == "eg_table":
        return "fp8", None
    if dtype == U8:
        return ("const", E8M0 * 0x01010101) if name.endswith(".s") else ("u8", None)
    if dtype == BF16:
        if base.endswith("norm"):
            return "const", 0x3F803F80
        return ("bf16_emb", None) if base in ("emb", "mk_embed") else ("bf16", None)
    if dtype == F32:
        if base.endswith("norm") or name.endswith("norm"):
            return "const", _F32_ONE
        if base in ("attn_sink", "router_bias") or base.endswith("_base"):
            return "const", 0
        if name.endswith(".c"):
            return "const", _F32(2.0**K)
        if name.endswith(".s") or base == "eg_rs":
            return "const", _F32(2.0**-6)
        if base.endswith("_scale"):
            return "const", _F32(2.0**-4)
        return "f32", None
    if dtype == I32:
        return ("const", K & 0xFFFFFFFF) if name.endswith(".k") else ("const", 0)
    if dtype == I64:
        return "const", 0
    raise ValueError(f"{name}: dtype {dtype}")


class FakeWeights:
    """Placeholder weights on one card: ``alloc`` allocates and fills a tensor; ``allocated`` lists them."""

    def __init__(self, device, rank: int) -> None:
        self.device, self.rank = device, rank
        self.pools: dict[str, int] = {}
        self.allocated: dict[
            str, tuple[int, int, str]
        ] = {}  # name: (address, bytes, kind)

    def _pool(self, kind: str) -> int:
        if kind not in self.pools:
            address = self.device.malloc(POOL_BYTES)
            self.device.h2d(address, _pool(kind))
            self.pools[kind] = address
        return self.pools[kind]

    def fill(self, address: int, nbytes: int, name: str, dtype: int) -> str:
        kind, pattern = kind_of(name, dtype)
        dev = self.device
        if kind == "const":
            words = -(-nbytes // 4)
            dev.memset32(address, pattern, words)
            return f"const {pattern:#x}"
        pool = self._pool(kind)
        # whole tensors: copy the pool from a 256-byte aligned seeded offset, wrapping, until every byte is set
        off = (_seed(name, self.rank) % (POOL_BYTES // 256)) * 256
        done = 0
        while done < nbytes:
            n = min(nbytes - done, POOL_BYTES - off)
            dev.d2d(address + done, pool + off, n)
            done += n
            off = 0
        return kind

    def alloc(self, name: str, nbytes: int, dtype: int) -> int:
        nbytes = max(256, -(-nbytes // 256) * 256)
        address = self.device.malloc(nbytes)
        kind = self.fill(address, nbytes, name, dtype)
        self.allocated[name] = (address, nbytes, kind)
        return address

    @property
    def nbytes(self) -> int:
        return sum(n for _, n, _ in self.allocated.values()) + POOL_BYTES * len(
            self.pools
        )


# ------------------------------------------------------------------ the model
def _model_class():
    from resin_models_gaudi.deepseek_v41.model import DeepSeekV41Gaudi
    from resin_models_gaudi.deepseek_v41.weights import reader, table_rows

    class FakeDeepSeekV41Gaudi(DeepSeekV41Gaudi):
        """``DeepSeekV41Gaudi`` with placeholder weights (``FakeWeights``), allocated as the programs bind them;
        ``attach`` binds every shape's program, so the weights exist before the first step. ``capacity`` must
        leave room for them (an int, or a callable given the free bytes less ``weight_reserve``)."""

        # rank 0's placeholder weights with DSpark (61.9 GiB), plus room
        weight_reserve = 64 << 30

        def __init__(
            self, device, checkpoint, cfg, *, rank: int = 0, nranks: int = 1, **kw
        ) -> None:
            kw.pop("weights", None)
            layers = kw.get("layers")
            n = cfg.num_hidden_layers if layers is None else layers
            src = reader(checkpoint)
            w: dict[str, int] = {}
            for layer in (e for e in cfg.engram_layer_ids if e < n):
                total = src.info(f"layers.{layer}.engram.embed.weight")[1][0]
                rows = table_rows(total, rank, nranks)
                w[f"l{layer}.eg_row0"], w[f"l{layer}.eg_rows"] = (
                    rows.start,
                    rows.stop - rows.start,
                )
            self.fake = FakeWeights(device, rank)
            super().__init__(
                device, checkpoint, cfg, rank=rank, nranks=nranks, weights=w, **kw
            )

        def attach(self, capacity, **kw) -> None:
            if callable(capacity):
                fit = capacity
                capacity = lambda free: fit(free - self.weight_reserve)  # noqa: E731
            super().attach(capacity, **kw)
            # the partial and logit all-gathers' destinations: zero where a one-rank communicator writes nothing
            for address, nbytes in (
                *((p, 4 * self.d.hidden * 16 * self.nranks) for p in self.pall),
                (self.logits_all, 4 * self.d.vocab * 2048 * self.nranks),
                (self.io["eg_kv_all"], 4 * self.eg_rows_r * 64 * self.d.ranks),
            ):
                self.device.memset32(address, 0, nbytes // 4)
            t0 = time.time()
            for s in self.shapes:
                self._program(s)
                if self.stages and ("dsp_attn", s) in self.r:
                    self._draft_program(s)
            self.device.sync()
            self.log(
                f"ds41 fake: {len(self.fake.allocated)} placeholder weights, {self.fake.nbytes / 2**30:.1f} GiB, "
                f"in {time.time() - t0:.1f} s"
            )

        def _op(self, recipe, layer, names):
            for n, tensor in recipe.persistent.items():
                if (
                    n in names
                    or (layer is not None and f"l{layer}.{n}" in self.w)
                    or n in self.w
                ):
                    continue
                if n in self.io or n in self.consts:
                    continue
                key = n if layer is None else f"l{layer}.{n}"
                self.w[key] = self.fake.alloc(key, tensor.nbytes, tensor.dtype)
            return super()._op(recipe, layer, names)

    return FakeDeepSeekV41Gaudi


def __getattr__(name: str):
    if name == "FakeDeepSeekV41Gaudi":
        return _model_class()
    raise AttributeError(name)


def patch_runner(**model_options) -> None:
    """Make ``DeepSeekV41GaudiRunner`` build the placeholder model (with ``model_options`` added)."""
    import resin_models_gaudi.deepseek_v41.runner as runner_module

    cls = _model_class()
    runner_module.DeepSeekV41Gaudi = (
        functools.partial(cls, **model_options) if model_options else cls
    )


def one_rank_of(rank: int, nranks: int = 8) -> None:
    """Make the runner's ``open_rank`` stand one card for rank ``rank`` of ``nranks`` (one-rank communicator)."""
    import resin_models_gaudi.deepseek_v41.runner as runner_module
    from resin_models_gaudi import ranks

    def open_rank(groups, card, model):
        from resin_models_gaudi.synapse.runtime import open_device

        device = open_device(card)
        device.ccl_init(device.ccl_unique_id(), 1, 0)
        return device, rank, nranks

    runner_module.ranks = _Ranks(ranks, open_rank)


class _Ranks:
    """``resin_models_gaudi.ranks`` with another ``open_rank``"""

    def __init__(self, base, open_rank) -> None:
        self._base, self.open_rank = base, open_rank

    def __getattr__(self, name):
        return getattr(self._base, name)


def fake_runner(checkpoint: str | Path = CHECKPOINT, **runner_kwargs):
    """A ``DeepSeekV41GaudiRunner`` over placeholder weights (``patch_runner`` first)."""
    from resin_models_gaudi.deepseek_v41.runner import DeepSeekV41GaudiRunner

    patch_runner()
    return DeepSeekV41GaudiRunner(str(checkpoint), **runner_kwargs)


# ------------------------------------------------------------------ the CLI
def run(
    rank: int, nranks: int, args, path: str | None = None, token: bytes | None = None
) -> dict:
    """Engine steps of the placeholder model through its runner and native executor: prompts of 37, 1051 and
    2100 tokens (bounded replay as configured), then decode steps of 1, 8 and 64 rows (overlapped), and DSpark
    steps of 1 and 2 rows. Returns each kind's median wall ms per Engine step and a digest of every token."""
    from tokenizers import Tokenizer

    from resin import Engine, SamplingParams

    cards = os.environ.get("VT_HPU_CARDS")
    card = int(cards.split(",")[rank]) if cards else None
    group = None
    if nranks > 1:
        from resin._resin import GaudiRanks

        group = GaudiRanks(path, token, rank, nranks, timeout=3600.0)
    else:
        one_rank_of(args.rank, 8)  # rank args.rank of 8 on one card
    say = print if rank == 0 else (lambda *a, **k: None)
    t0 = time.time()
    runner = fake_runner(
        card=card, native_ranks=group, max_batch_size=64, max_seq_len=args.max_seq_len,
        parent_capacity=args.slabs, draft_tokens=args.draft_tokens, replay=args.replay,
    )  # fmt: skip
    build = time.time() - t0
    say(f"built in {build:.1f} s")
    if group is not None and rank > 0:
        runner.serve()
        runner.close()
        return {}
    engine = Engine(runner, max_running_requests=64, context_length=args.max_seq_len,
                    prefix_cache=False, enable_overlap=True)  # fmt: skip
    tok = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    readme = Path(__file__).resolve().parents[3] / "README.md"
    text = tok.encode(readme.read_text() * 4).ids
    digest = hashlib.blake2b(digest_size=16)
    times: dict[str, list[float]] = {}

    def generate(name: str, prompts: list[list[int]], tokens: int) -> None:
        leases = [engine.submit(p, sampling_params=SamplingParams(max_output_tokens=tokens, temperature=0.0))
                  for p in prompts]  # fmt: skip
        steps = []
        while engine.has_work():
            start = time.perf_counter()
            events = engine.step()
            steps.append((time.perf_counter() - start, len(events)))
        # decode steps: those that give every row a token (after the prompts' prefill)
        decode = [1e3 * dt for dt, n in steps if n >= len(prompts)]
        times[name] = decode or [1e3 * dt for dt, _ in steps]
        for lease in leases:
            digest.update(np.array(lease.take_result(), np.int64).tobytes())

    for n in (37, 1051, 2100):
        generate(f"prefill {n}", [text[:n]], 1)
    for B in (1, 8, 64):
        generate(f"decode B={B}", [text[i * 7 : i * 7 + args.context] for i in range(B)], args.steps)
    result = {
        "rank": rank,
        "build_s": round(build, 1),
        "ms": {k: round(float(np.median(v)), 3) for k, v in times.items()},
        "speculation": runner.speculative_stats,
        "digest": digest.hexdigest(),
        "compiles_after_start": runner.compiles_after_start,
        "timings_ms": {k: round(v / 1e6, 1) for k, v in runner.executor.timings.items()},
    }
    say(json.dumps(result, indent=1))
    if args.out:
        Path(f"{args.out}.{rank}.json").write_text(json.dumps(result))
    engine.close()
    runner.close()
    return result


def _worker(rank: int, nranks: int, args, path: str, token: bytes, cards: str) -> None:
    os.environ["VT_HPU_CARDS"] = cards
    run(rank, nranks, args, path, token)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="1: one card as rank --rank of 8; else N cards with HCCL",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--slabs", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--replay", type=int, default=256)
    parser.add_argument(
        "--context", type=int, default=1024, help="decode rows' prompt length"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--out", help="write each rank's result to OUT.<rank>.json")
    args = parser.parse_args()
    if args.ranks == 1:
        run(0, 1, args)
        return
    from resin.process import spawn

    cards = os.environ["VT_HPU_CARDS"]
    accel = cards.split(",")[: args.ranks]
    bdf = [
        os.path.basename(os.path.realpath(f"/sys/class/accel/accel{i}/device"))
        for i in accel
    ]
    path = os.path.join(tempfile.mkdtemp(prefix="ds41-fake-"), "group")
    try:
        spawn(_worker, args=(args.ranks, args, path, secrets.token_bytes(16), cards),
              devices=[f"pci:{b}" for b in bdf])  # fmt: skip
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


if __name__ == "__main__":
    main()
