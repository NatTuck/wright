"""Qwen3.8-Flash-Next truncated to N layers on one card against the transformers implementation on the CPU
(``tools/qwen4_exp_ref_hf.py``, run with ``RESIN_FNEXT_REF_PYTHON``, a venv with torch and a transformers that
has ``qwen4_exp``): the logits after the prompt and after NDEC greedy decode steps, teacher forced with the
device's own tokens: the prompt's within 1e-4 of correlation, every argmax equal, the decode steps close (the
chunked prefill and the recurrent decode differ in rounding, which the MoE routing amplifies at near ties).
``RESIN_GAUDI_FNEXT`` names the checkpoint (a truncated download works); ``RESIN_FNEXT_LAYERS`` the layers
(default 4: delta net, the PLE layer, attention); ``RESIN_FNEXT_TABLE=1`` uses the n-gram table (the default
compares without it, both sides zero)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_GAUDI_FNEXT")
REF_PY = os.environ.get("RESIN_FNEXT_REF_PYTHON", "")
N = int(os.environ.get("RESIN_FNEXT_LAYERS", "4"))
TABLE = os.environ.get("RESIN_FNEXT_TABLE") == "1"
NDEC = 4
TOOLS = Path(__file__).resolve().parents[1] / "tools"
# (the fox prompt's first generated token sits at an MoE routing near-tie: 0.975 there, same argmax)
PROMPT = os.environ.get(
    "RESIN_FNEXT_PROMPT",
    "Write a Python function that parses an ISO 8601 date string and returns a datetime object. Here it is:",
)

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not REF_PY
    or not Path(REF_PY).exists()
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_FNEXT, the reference venv, the Synapse SDK and tpc-clang",
)


def reference(seq: list[int], out: Path) -> np.ndarray:
    args = [
        REF_PY,
        "-I",
        str(TOOLS / "qwen4_exp_ref_hf.py"),
        CHECKPOINT,
        str(N),
        str(out),
    ]
    if not TABLE:
        args.append("--no-table")
    subprocess.run(
        args + [str(t) for t in seq], check=True, capture_output=True, text=True
    )
    return np.load(out)["logits"]


def test_truncated_model_matches_transformers(tmp_path):
    from tokenizers import Tokenizer

    from resin_models_gaudi.qwen4_exp.model import Qwen4ExpGaudi
    from resin_models_gaudi.synapse.graph import RecipeCache
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        device = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    ids = Tokenizer.from_file(f"{CHECKPOINT}/tokenizer.json").encode(PROMPT).ids
    m = Qwen4ExpGaudi(
        device,
        CHECKPOINT,
        layers=N,
        table=TABLE,
        max_positions=4096,
        recipes=RecipeCache(device, tmp_path / "recipes"),
        log=lambda _: None,
    )
    row = m.sequence_rows(0, 2048)
    sampled = []
    if (
        os.environ.get("RESIN_FNEXT_DECODE_ONLY") == "1"
    ):  # the prompt token by token (the decode path alone)
        for tok in ids:
            sampled = list(m.decode([row], [tok]))
            row = row.advanced(1, [tok])
    else:
        sampled = [m.prefill(row, ids)]
        row = row.advanced(len(ids), ids)
    logits = [m.last_logits()[0]]
    seq = list(ids) + [int(logits[-1].argmax())]
    assert sampled[0] == seq[-1]  # the greedy sampler picks the argmax
    for _ in range(NDEC):
        sampled = m.decode([row], [seq[-1]])
        row = row.advanced(1, [seq[-1]])
        logits.append(m.last_logits()[0])
        seq.append(int(logits[-1].argmax()))
        assert sampled[0] == seq[-1]
    assert m.compiles_after_start == 0
    ref = reference(seq[:-1], tmp_path / "ref.npz")
    P = len(ids)
    report = []
    for k, lg in enumerate(logits):
        r = ref[P - 1 + k]
        corr = float(np.corrcoef(lg, r)[0, 1])
        report.append(
            (k, corr, float(np.abs(lg - r).max()), int(lg.argmax()), int(r.argmax()))
        )
    sys.stderr.write(
        "\n".join(
            f"step {k}: corr {c:.6f} max|d| {d:.3f} argmax {a} vs {b}"
            for k, c, d, a, b in report
        )
        + "\n"
    )
    # the prefill is one bucket's arithmetic, the reference another: a step or two after the prompt can sit at
    # a routing near-tie of the MoE; every argmax must agree
    for k, corr, _, a, b in report:
        assert corr > (0.9999 if k == 0 else 0.98), report
        assert a == b, report
    assert sum(c for _, c, _, _, _ in report[1:]) / NDEC > 0.995, report
    device.close()
