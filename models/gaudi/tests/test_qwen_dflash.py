"""The DFlash 2 drafter of Qwen3.8-27B on a Gaudi card (set ``RESIN_GAUDI_QWEN35`` to the FP8
checkpoint directory and ``RESIN_GAUDI_DFLASH`` to the ``Qwen3.8-27B-DFlash2`` directory; the
model takes a few minutes to load and compile).

The drafter's block after a prompt agrees with the CPU reference (``tools/qwen_dflash2_ref.py``)
fed the same target features; speculative decoding emits the tokens of plain decoding, alone and
across pages; nothing compiles after construction.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

# the K and V page types: RESIN_GAUDI_CACHE_TYPES=fp8,fp8 runs the tests on an FP8 cache
CACHE_TYPES = tuple(os.environ.get("RESIN_GAUDI_CACHE_TYPES", "bf16,bf16").split(","))

TOOLS = Path(__file__).resolve().parents[1] / "tools"

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")
DRAFT = os.environ.get("RESIN_GAUDI_DFLASH")

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or DRAFT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_QWEN35, RESIN_GAUDI_DFLASH, the Synapse SDK and tpc-clang",
)


@pytest.fixture(scope="module")
def runner():
    """The runner with the drafter (its native executor makes the steps)."""
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = Qwen3_5GaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=4,
            max_seq_len=8192,
            parent_capacity=24,
            drafter=DRAFT,
            cache_types=CACHE_TYPES,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield built
    built.close()


@pytest.fixture(scope="module")
def engine(runner):
    """The model the runner built."""
    return runner.model


@pytest.fixture(scope="module")
def model(runner):
    """One sequence over the first twelve slabs."""
    from qwen_sequence import PlanSequence

    return PlanSequence(runner, range(1, 13))


@pytest.fixture(scope="module")
def text_ids() -> list[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json"))
    docs = Path(__file__).resolve().parents[3] / "docs"
    text = "\n\n".join(p.read_text() for p in sorted(docs.glob("*.md")))
    return tokenizer.encode(text).ids


def plain(model, prompt: list[int], n: int) -> list[int]:
    model.reset()
    out = [model.prefill(prompt)]
    for _ in range(n - 1):
        out.append(model.decode(out[-1]))
    return out


def speculative(model, prompt: list[int], n: int) -> list[int]:
    model.reset()
    out = [model.prefill(prompt)]
    while len(out) < n:
        out += model.verify(out[-1])
    return out[:n]


def test_drafter_matches_the_reference(engine, model, text_ids):
    """The block after a 256-token prompt (one prefill chunk, so the device's ``feat`` buffers
    still hold every prompt column's projected taps): the device drafter against the CPU
    reference on those taps."""
    torch = pytest.importorskip("torch")
    from resin_models_gaudi.loading.convert import sigma_perm

    sys.path.insert(0, str(TOOLS))
    from qwen_dflash2_ref import (
        DFlash2Config,
        DFlash2Ref,
        Weights,
    )

    prompt = text_ids[:256]  # one prefill chunk of 256
    token = plain(model, prompt, 1)[0]
    drafter, cfg = engine.drafter, engine.dfl_cfg
    assert model.drafts is not None and len(model.drafts) == cfg.drafts
    T = 256
    H = engine.dims.hidden
    projected = sum(
        engine.device.d2h(drafter.feat[k], (T, H), np.float32)[: len(prompt)].astype(
            np.float64
        )
        for k in range(len(cfg.taps))
    )
    ref = DFlash2Ref(
        DFlash2Config.from_json(Path(DRAFT) / "config.json"),
        Weights.load(Path(DRAFT), Path(CHECKPOINT)),
    )
    ref.inject_projected(
        torch.tensor(projected, dtype=torch.float32), torch.arange(len(prompt))
    )
    hidden = ref.block_forward(token, len(prompt))
    logits = hidden[1:] @ ref.w.head.float().T
    ref_drafts = ref.select(hidden[1:], logits, token).numpy()
    # the block's final hidden state (the LM head's operand) tracks the reference
    h16 = engine.device.d2h(
        drafter.carry[cfg.layers % 2]["h16"], (cfg.block, H), np.uint16
    )
    got = (h16.astype(np.uint32) << 16).view(np.float32)[:, np.argsort(sigma_perm(H))]
    want = hidden.numpy()
    corr = np.corrcoef(got[1:].ravel(), want[1:].ravel())[0, 1]
    assert corr > 0.99, corr
    agree = int((model.drafts == ref_drafts).sum())
    assert model.drafts[0] == ref_drafts[0] and agree >= 5, (model.drafts, ref_drafts)


def test_speculation_equals_plain_decoding(model, text_ids):
    prompt = text_ids[:37]
    assert speculative(model, prompt, 48) == plain(model, prompt, 48)


def test_blocks_crossing_pages_and_the_state_block(model, text_ids):
    """Decoding from a prompt that ends just before a 128-token page boundary, and across the
    4096-token STATE boundary. (The 4090-token prefix of the docs sits on a greedy near-tie at
    position 4093, where plain decoding itself continues differently after a prefill than after
    decoding the same tokens; the 8-column verify step falls on the other side of it.)"""
    prompt = text_ids[:250]
    assert speculative(model, prompt, 30) == plain(model, prompt, 30)
    prompt = text_ids[:4086]
    assert speculative(model, prompt, 24) == plain(model, prompt, 24)


def test_accepts_some_drafts(runner, model, text_ids):
    before = runner.speculative_stats
    speculative(model, text_ids[:200], 64)
    after = runner.speculative_stats
    accepted = after["accepted"] - before["accepted"]
    blocks = after["blocks"] - before["blocks"]
    assert blocks > 0 and accepted / blocks > 1.0, (accepted, blocks)


def test_no_compiles_after_start(runner):
    assert runner.compiles_after_start == 0
