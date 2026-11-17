"""The DFlash 2 drafter of GLM-5.3-Flash on the native executor, tensor parallel on ``RESIN_GAUDI_GLM_RANKS`` cards
(default 4) with a native rank group; set ``RESIN_GAUDI_GLM53`` to the FP8 release and ``RESIN_GAUDI_GLM_DFLASH`` to the
``GLM-5.3-Flash-DFlash2`` directory. Every rank loads the whole model, so this takes minutes.

Through the Engine on rank 0: the block after a 256-token prompt agrees with the CPU reference
(``tools/qwen_dflash2_ref.py``) fed the device's own tap projections; speculative decoding emits exactly the tokens of
verify-step decoding that accepts no draft (the same arithmetic, so every accepted draft is a free token), and plain
decoding's tokens until the first near tie; every rank accepts the same drafts; the drafter earns its keep; nothing
compiles after construction.

A verify step's bucket rounds differently from a one-token decode step, so plain and speculative decoding part at a
near tie sooner or later: not a defect, and the same holds for a batch of one against a batch of eight.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHECKPOINT = os.environ.get("RESIN_GAUDI_GLM53")
DRAFT = os.environ.get("RESIN_GAUDI_GLM_DFLASH")
RANKS = int(os.environ.get("RESIN_GAUDI_GLM_RANKS", "4"))
PROMPT = 256
STEPS = 48
NEAR_TIE = 1.0  # logits: the margin below which the verify bucket's rounding may pick the other token

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or DRAFT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None
    or len(list(Path("/sys/class/accel").glob("accel[0-9]*"))) < RANKS,
    reason=f"requires RESIN_GAUDI_GLM53, RESIN_GAUDI_GLM_DFLASH, the Synapse SDK, tpc-clang and {RANKS} cards",
)


def text_ids(n: int) -> list[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json"))
    docs = Path(__file__).resolve().parents[3] / "docs"
    text = "\n\n".join(p.read_text() for p in sorted(docs.glob("*.md")))
    return tokenizer.encode(text).ids[:n]


def generate(engine, prompt: list[int], n: int) -> tuple[list[int], list[float]]:
    """Greedy decoding of ``n`` tokens -> (tokens, the logprob margin of each token's top two)"""
    from resin import SamplingParams

    params = SamplingParams(
        temperature=0.0, max_output_tokens=n, logprobs=True, top_logprobs=2
    )
    lease = engine.submit(list(prompt), sampling_params=params)
    margins = []
    while engine.has_work():
        for event in engine.step():
            if event.logprobs is not None:
                top = [float(v) for _, v in (event.logprobs.top_logprobs or ())]
                margins.append(top[0] - top[1] if len(top) == 2 else float("inf"))
    return [int(t) for t in lease.take_result()], margins


def _worker(rank: int, path: str, token: bytes, cards: list[str], out: str) -> None:
    import torch

    def log(*what) -> None:
        """Progress on stderr (``pytest -s``): a load takes minutes, and a card fault names no phase."""
        print(
            f"[{time.strftime('%H:%M:%S')} rank {rank}]",
            *what,
            file=sys.stderr,
            flush=True,
        )

    from resin import Engine, SamplingParams
    from resin._resin import GaudiRanks
    from resin_models_gaudi.glm5_3_flash.runner import Glm53FlashGaudiRunner

    group = GaudiRanks(path, token, rank, len(cards), timeout=3600.0)
    runner = Glm53FlashGaudiRunner(
        CHECKPOINT, native_ranks=group, card=int(cards[rank]), max_batch_size=4, max_seq_len=8192,
        parent_capacity=48, drafter=DRAFT, log=lambda _: None,
    )  # fmt: skip
    ex = runner.executor
    log("built")
    compiles = runner.compiles_after_start
    if rank > 0:
        runner.serve()
        Path(f"{out}.{rank}").write_text(json.dumps(list(ex.speculation)))
        runner.close()
        return
    engine = Engine(
        runner,
        max_running_requests=4,
        context_length=8192,
        prefix_cache=False,
        enable_overlap=True,
    )
    ids = text_ids(4200)
    results: dict[str, object] = {}
    # ---- the block after the prompt against the CPU reference, on the taps the device projected
    m, cfg = runner.model, runner.config
    drafter, dcfg = m.drafter, m.dfl_cfg
    ex.policy = "always"
    lease = engine.submit(
        ids[:PROMPT],
        sampling_params=SamplingParams(temperature=0.0, max_output_tokens=2),
    )
    log("prefill 256")
    engine.step()  # the prompt's prefill and the drafts after its sampled token
    carried = ex.carried_all()
    assert len(carried) == 1, carried
    _, token, drafts = next(iter(carried.values()))
    H = cfg.hidden_size
    projected = sum(
        m.device.d2h(drafter.feat[k], (PROMPT, H), np.float32).astype(np.float64)
        for k in range(len(dcfg.taps))
    )
    while engine.has_work():
        engine.step()
    lease.take_result()
    sys.path.insert(0, str(TOOLS))
    from qwen_dflash2_ref import DFlash2Config, DFlash2Ref, Weights

    log("reference")
    ref = DFlash2Ref(
        DFlash2Config.from_json(Path(DRAFT) / "config.json"),
        Weights.load(Path(DRAFT), Path(CHECKPOINT)),
    )
    ref.inject_projected(
        torch.tensor(projected, dtype=torch.float32), torch.arange(PROMPT)
    )
    hidden = ref.block_forward(token, PROMPT)
    logits = hidden[1:] @ ref.w.head.float().T
    ref_drafts = ref.select(hidden[1:], logits, token).numpy()
    drafts = np.asarray(drafts)
    results["drafts"], results["ref_drafts"] = drafts.tolist(), ref_drafts.tolist()
    results["agree"] = int((drafts == ref_drafts).sum())
    results["first_agrees"] = bool(drafts[0] == ref_drafts[0])
    # ---- speculative decoding emits verify-step decoding's tokens
    cases = {}
    for name, prompt in (
        ("short", ids[:37]),
        ("page", ids[:250]),
        ("state", ids[:4086]),
    ):
        count = STEPS if name != "state" else 24
        log(name, "plain")
        ex.policy, ex.fixed_acceptance = "never", None
        plain, margins = generate(engine, prompt, count)
        ex.policy, ex.fixed_acceptance = "always", 0
        log(name, "column")
        column, _ = generate(engine, prompt, count)
        log(name, "spec")
        ex.fixed_acceptance = None
        before = ex.speculation
        spec, _ = generate(engine, prompt, count)
        after = ex.speculation
        cases[name] = dict(
            plain=plain, margins=margins, column=column, spec=spec,
            accepted_per_block=(after[2] - before[2]) / max(1, after[0] - before[0]),
        )  # fmt: skip
    results["cases"] = cases
    results["compiles"] = runner.compiles_after_start - compiles
    results["speculation"] = list(ex.speculation)
    engine.close()
    runner.close()
    Path(out).write_text(json.dumps(results))


def test_drafter_across_ranks(tmp_path):
    from resin.process import spawn

    # the cards a queue script granted (VT_HPU_CARDS: accel indices), else the first RANKS
    picked = re.split(r"[ ,]+", os.environ.get("VT_HPU_CARDS", "").strip())
    if picked == [""]:
        picked = sorted(
            p.name.removeprefix("accel")
            for p in Path("/sys/class/accel").glob("accel[0-9]*")
        )
    cards = picked[:RANKS]
    bdf = [
        os.path.basename(os.path.realpath(f"/sys/class/accel/accel{i}/device"))
        for i in cards
    ]
    out = tmp_path / "results.json"
    path = os.path.join(tempfile.mkdtemp(prefix="glm-dflash-"), "group")
    spawn(
        _worker,
        args=(path, secrets.token_bytes(16), cards, str(out)),
        devices=[f"pci:{b}" for b in bdf],
    )
    results = json.loads(out.read_text())
    followers = [json.loads(Path(f"{out}.{r}").read_text()) for r in range(1, RANKS)]
    assert all(f == results["speculation"] for f in followers), (
        "the ranks accepted different drafts"
    )
    assert results["compiles"] == 0
    assert results["first_agrees"] and results["agree"] >= 5, (
        results["drafts"],
        results["ref_drafts"],
    )
    for name, c in results["cases"].items():
        plain, spec, column = c["plain"], c["spec"], c["column"]
        assert spec == column, (name, column, spec)
        parted = next((i for i, (p, s) in enumerate(zip(plain, spec)) if p != s), None)
        if (
            parted is not None
        ):  # the verify bucket's rounding picked the other side of a near tie
            assert parted > 0 and c["margins"][parted] < NEAR_TIE, (
                name,
                parted,
                c["margins"][parted],
                plain,
                spec,
            )
        if name != "state":
            assert c["accepted_per_block"] > 1.0, (name, c["accepted_per_block"])
