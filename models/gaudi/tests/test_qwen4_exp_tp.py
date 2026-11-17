"""Qwen3.8-Flash-Next over RANKS Gaudi cards (one process each, HCCL over the ranks): the tensor-parallel model's
logits after a prompt and over greedy decode steps match the single-card model's, and every rank emits the same
tokens. Same environment as ``test_qwen4_exp_ref.py`` (``RESIN_GAUDI_FNEXT``, ``RESIN_FNEXT_LAYERS``); the cards
come from ``VT_HPU_CARDS`` (a queue script's grant), else the first RANKS accel devices."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_GAUDI_FNEXT")
N = int(os.environ.get("RESIN_FNEXT_LAYERS", "4"))
RANKS = int(os.environ.get("RESIN_FNEXT_RANKS", "4"))
GATHER = int(
    os.environ.get("RESIN_FNEXT_GATHER", "16")
)  # 0: every launch all-reduces in place
DECODE_ONLY = os.environ.get("RESIN_FNEXT_DECODE_ONLY") == "1"
REPLICATE = (
    os.environ.get("RESIN_FNEXT_REPLICATE_DENSE") == "1"
)  # the dense weights whole on every rank
NDEC = 4
NEAR_TIE = 1.0  # logits: a single card's top-two margin below this may flip under another summation order
PROMPT = "Write a Python function that parses an ISO 8601 date string and returns a datetime object. Here it is:"

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_FNEXT, the Synapse SDK and tpc-clang",
)


def _run(m, ids: list[int], forced: list[int] | None = None):
    """the logits after the prompt and after NDEC decode steps (greedy, or teacher-forced with ``forced``), and
    the tokens the device sampled"""
    row = m.sequence_rows(0, 2048)
    if DECODE_ONLY:  # the prompt token by token: the decode path alone
        for tok in ids:
            last = int(m.decode([row], [tok])[0])
            row = row.advanced(1, [tok])
        tokens = [last]
    else:
        tokens = [m.prefill(row, ids)]
        row = row.advanced(len(ids), ids)
    logits = [m.last_logits()[0]]
    for k in range(NDEC):
        fed = tokens[-1] if forced is None else forced[k]
        tokens.append(int(m.decode([row], [fed])[0]))
        row = row.advanced(1, [fed])
        logits.append(m.last_logits()[0])
    return [lg.astype(np.float64) for lg in logits], [int(t) for t in tokens]


def _worker(rank: int, port: int, cards: list[str], out: str) -> None:
    import torch.distributed as dist
    from tokenizers import Tokenizer

    from resin_models_gaudi import ranks as par
    from resin_models_gaudi.qwen4_exp.model import Qwen4ExpGaudi
    from resin_models_gaudi.synapse.runtime import open_device

    if (
        os.environ.get("RESIN_FNEXT_NO_WORKLIST") == "1"
    ):  # decode MoE through the per-token kernels
        from resin_models_gaudi.qwen4_exp import layers as L

        L.WORK_LIST_PAIRS = 0
    n = len(cards)
    par.init_gloo(rank, n, port)
    dev = open_device(par.accel_index(cards[rank]))
    par.hccl_rendezvous(dev, rank, n)
    ids = Tokenizer.from_file(f"{CHECKPOINT}/tokenizer.json").encode(PROMPT).ids
    common = dict(
        layers=N, table=False, max_positions=2048, parent_capacity=8, log=lambda _: None
    )
    # rank 0's card also runs the single-rank model first: its greedy tokens teacher-force every rank's run
    box: list = [None]
    ref_logits = None
    if rank == 0:
        single = Qwen4ExpGaudi(dev, CHECKPOINT, **common)
        ref_logits, ref_tokens = _run(single, ids)
        box = [ref_tokens]
    dist.broadcast_object_list(box, src=0)
    ref_tokens = box[0]
    m = Qwen4ExpGaudi(
        dev,
        CHECKPOINT,
        rank=rank,
        nranks=n,
        gather_tokens=GATHER,
        replicate_dense=REPLICATE,
        **common,
    )
    par.barrier()
    compiles = dev.compile_count
    logits, tokens = _run(m, ids, forced=ref_tokens)
    results: dict[str, object] = {
        "tokens": tokens,
        "compiles": dev.compile_count - compiles,
    }
    if rank == 0:
        assert ref_logits is not None
        results["ref_tokens"] = ref_tokens
        results["corr"] = [
            float(np.corrcoef(a, b)[0, 1])
            for a, b in zip(logits, ref_logits, strict=True)
        ]
        results["argmax"] = [
            [int(a.argmax()), int(b.argmax())]
            for a, b in zip(logits, ref_logits, strict=True)
        ]
        # the single card's margin between its top two logits: a near tie may fall the other way under the
        # ranks' summation order
        results["margin"] = [float(np.diff(np.sort(b)[-2:])[0]) for b in ref_logits]
    par.barrier()
    Path(out).with_suffix(f".{rank}.json").write_text(json.dumps(results))
    dev.close()


def test_tensor_parallel_matches_one_card(tmp_path):
    from resin.process import spawn

    picked = re.split(r"[ ,]+", os.environ.get("VT_HPU_CARDS", "").strip())
    if picked != [""]:
        entries = [Path(f"/sys/class/accel/accel{i}") for i in picked][:RANKS]
    else:
        entries = sorted(Path("/sys/class/accel").glob("accel[0-9]*"))[:RANKS]
    assert len(entries) == RANKS, f"{RANKS} cards needed"
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "results.json"
    spawn(_worker, args=(port, cards, str(out)), devices=[f"pci:{c}" for c in cards])
    results = [
        json.loads(out.with_suffix(f".{r}.json").read_text()) for r in range(RANKS)
    ]
    assert all(r["tokens"] == results[0]["tokens"] for r in results), (
        "the ranks emitted different tokens"
    )
    assert all(r["compiles"] == 0 for r in results)
    first = results[0]
    print("corr", first["corr"], "argmax", first["argmax"], "margin", first["margin"])
    assert min(first["corr"]) > 0.9999, first["corr"]
    for (a, b), margin in zip(first["argmax"], first["margin"], strict=True):
        assert a == b or margin < NEAR_TIE, (a, b, margin)
    assert sum(a == b for a, b in first["argmax"]) >= NDEC, first["argmax"]
