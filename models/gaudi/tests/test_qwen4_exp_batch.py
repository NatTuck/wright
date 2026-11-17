"""Qwen3.8-Flash-Next batched decode on one card: several sequences decoded together in one step (their own KV
and STATE blocks of the paged arena, the delta-net states, QSA keys and PLE state read and written per row) give
the logits each sequence gets decoded alone, and a prompt prefilled in two chunks gives the one-chunk logits.
Same environment as ``test_qwen4_exp_ref.py`` (``RESIN_GAUDI_FNEXT``, ``RESIN_FNEXT_LAYERS``)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_GAUDI_FNEXT")
N = int(os.environ.get("RESIN_FNEXT_LAYERS", "4"))
NDEC = 6
PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string and returns a datetime object. Here it is:",
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
]

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_FNEXT, the Synapse SDK and tpc-clang",
)


def close(a: np.ndarray, b: np.ndarray, what: str) -> None:
    corr = float(np.corrcoef(a, b)[0, 1])
    sys.stderr.write(
        f"{what}: corr {corr:.6f} max|d| {np.abs(a - b).max():.4f} argmax {a.argmax()} vs {b.argmax()}\n"
    )
    assert corr > 0.9999 and a.argmax() == b.argmax(), what


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from resin_models_gaudi.qwen4_exp.model import Qwen4ExpGaudi
    from resin_models_gaudi.synapse.graph import RecipeCache
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        device = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    m = Qwen4ExpGaudi(
        device,
        CHECKPOINT,
        layers=N,
        table=False,
        max_positions=4096,
        parent_capacity=12,
        decode_buckets=(1, 4),
        prefill_buckets=(128, 256),
        recipes=RecipeCache(device, tmp_path_factory.mktemp("recipes")),
        log=lambda _: None,
    )
    m.check_overlaps = True
    yield m
    device.close()


def test_batched_decode_matches_single_rows(model):
    from tokenizers import Tokenizer

    m = model
    tk = Tokenizer.from_file(f"{CHECKPOINT}/tokenizer.json")
    prompts = [tk.encode(p).ids for p in PROMPTS]
    # alone: every sequence prefilled and decoded at bucket 1
    alone = []
    for s, ids in enumerate(prompts):
        row = m.sequence_rows(s, 1024)
        tok = m.prefill(row, ids)
        row = row.advanced(len(ids), ids)
        steps = [m.last_logits()[0]]
        for _ in range(NDEC):
            m.decode([row], [tok])
            row = row.advanced(1, [tok])
            steps.append(m.last_logits()[0])
            tok = int(steps[-1].argmax())
        alone.append(steps)
    # together: fresh rows in other slabs, prefilled alone, then decoded as one batch
    rows, toks, together = [], [], [[] for _ in prompts]
    for s, ids in enumerate(prompts):
        row = m.sequence_rows(len(prompts) + s, 1024)
        toks.append(m.prefill(row, ids))
        rows.append(row.advanced(len(ids), ids))
        together[s].append(m.last_logits()[0])
    for _ in range(NDEC):
        m.decode(rows, toks)
        lg = m.last_logits()
        for s in range(len(prompts)):
            together[s].append(lg[s])
            rows[s] = rows[s].advanced(1, [toks[s]])
        toks = [int(lg[s].argmax()) for s in range(len(prompts))]
    for s in range(len(prompts)):
        for k in range(NDEC + 1):
            close(together[s][k], alone[s][k], f"sequence {s} step {k}")
    assert m.compiles_after_start == 0


def test_two_chunk_prefill_matches_one(model):
    from tokenizers import Tokenizer

    m = model
    ids = (
        Tokenizer.from_file(f"{CHECKPOINT}/tokenizer.json")
        .encode(" ".join(PROMPTS) * 2)
        .ids
    )
    assert 64 < len(ids) <= 256
    row = m.sequence_rows(0, 1024)
    m.prefill(row, ids)
    whole = m.last_logits()[0]
    row = m.sequence_rows(1, 1024)
    m.prefill(row, ids[:64])
    m.prefill(row.advanced(64, ids[:64]), ids[64:])
    chunked = m.last_logits()[0]
    close(chunked, whole, "two-chunk prefill")
    # every column's logits of a chunk run on its own against the ones after a prefix: the chunk's first columns
    row = m.sequence_rows(2, 1024)
    cols = m.logits_of(row, ids[:64])
    m.prefill(m.sequence_rows(3, 1024), ids[:63])
    close(m.last_logits()[0], cols[62], "column 62 of a 64-token chunk")
