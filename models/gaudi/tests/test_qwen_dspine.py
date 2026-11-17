"""The DSpine drafter of Qwen3.8-27B on a Gaudi card (set ``RESIN_GAUDI_QWEN35`` to the FP8
checkpoint directory and ``RESIN_GAUDI_DSPINE`` to the ``Qwen3.8-27B-DSpine-b16`` directory; the
model takes a few minutes to load and compile).

The drafter's block after a prompt agrees with the CPU reference (``tools/qwen_dspine_ref.py``)
fed the same target features; speculative decoding emits the tokens of plain decoding, alone and
across pages, at every verify width; nothing compiles after construction. A step's verify width
(4, 8 or 16 columns) changes how a token's logits round, so a run that changes width can part from
plain decoding where its top two logits nearly tie (``NEAR_TIE``), as plain decoding itself does
between a prefill and decoding the same tokens.
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
DRAFT = os.environ.get("RESIN_GAUDI_DSPINE")
NEAR_TIE = (
    1.0  # logits: a near tie of the top two (drafter vs reference, or plain decoding)
)

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or DRAFT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_QWEN35, RESIN_GAUDI_DSPINE, the Synapse SDK and tpc-clang",
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


def plain_margins(model, prompt: list[int], n: int) -> tuple[list[int], list[float]]:
    """Plain decoding and each token's top-2 logit margin (the prefill's token: inf)."""
    model.reset()
    out, margins = [model.prefill(prompt)], [float("inf")]
    for _ in range(n - 1):
        out.append(model.decode(out[-1]))
        top = np.sort(np.asarray(model.logits()).reshape(-1))[-2:]
        margins.append(float(top[1] - top[0]))
    return out, margins


def assert_parts_at_a_near_tie(got: list[int], want: list[int], margins: list[float]):
    """Equal, or equal up to a first difference where plain decoding's top two logits are
    within NEAR_TIE: a verify step of another width rounds to the other side (module doc)."""
    diff = next((i for i in range(len(want)) if got[i] != want[i]), None)
    assert diff is None or margins[diff] < NEAR_TIE, (diff, margins[diff], got, want)


def speculative(model, prompt: list[int], n: int) -> list[int]:
    model.reset()
    out = [model.prefill(prompt)]
    while len(out) < n:
        out += model.verify(out[-1])
    return out[:n]


def test_drafter_matches_the_reference(engine, model, text_ids):
    """The block after a 256-token prompt (one prefill chunk, so the device's ``feat`` buffers
    still hold every prompt column's projected taps): the device drafter against the CPU
    reference on those taps, and the device's walk against the reference's walk over the
    device's own candidates."""
    torch = pytest.importorskip("torch")
    from resin_models_gaudi.loading.convert import sigma_perm

    sys.path.insert(0, str(TOOLS))
    from qwen_dspine_ref import DSpineConfig, DSpineRef, Weights

    prompt = text_ids[:256]  # one prefill chunk of 256
    token = plain(model, prompt, 1)[0]
    drafter, cfg = engine.drafter, engine.dfl_cfg
    assert cfg.kind == "dspine" and cfg.drafts == 15
    assert model.drafts is not None and len(model.drafts) == cfg.drafts
    T = 256
    H = engine.dims.hidden
    projected = sum(
        engine.device.d2h(drafter.feat[k], (T, H), np.float32)[: len(prompt)].astype(
            np.float64
        )
        for k in range(len(cfg.taps))
    )
    ref = DSpineRef(
        DSpineConfig.from_json(Path(DRAFT) / "config.json"),
        Weights.load(Path(DRAFT), Path(CHECKPOINT)),
    )
    ref.inject_projected(
        torch.tensor(projected, dtype=torch.float32), torch.arange(len(prompt))
    )
    hidden, stash = ref.block_forward(token, len(prompt))
    ref_drafts = ref.refine(stash, ref.candidates(hidden), token).numpy()
    # the block's final hidden state (the LM head's operand) tracks the reference
    h16 = engine.device.d2h(
        drafter.carry[cfg.layers % 2]["h16"], (cfg.block, H), np.uint16
    )
    got = (h16.astype(np.uint32) << 16).view(np.float32)[:, np.argsort(sigma_perm(H))]
    want = hidden.numpy()
    corr = np.corrcoef(got[1:].ravel(), want[1:].ravel())[0, 1]
    assert corr > 0.99, corr
    # the last write's scale and codes the walk reads
    ws = engine.device.d2h(drafter.dsp_ws, (cfg.block,), np.float32)
    np.testing.assert_allclose(ws[1:], stash.rms[1:].numpy(), rtol=0.05)
    assert ws[0] == 0.0
    # the walks agree up to their first fork, which must sit at a near tie of the reference's
    # refined logits (FP8 against f32), then the reference follows the device's choice
    fork = next(
        (i for i in range(cfg.drafts) if model.drafts[i] != ref_drafts[i]), None
    )
    assert fork is None or fork >= 4, (model.drafts, ref_drafts)
    from resin_models_gaudi.common.dflash import drafter_is_fp8

    # an fp8_inc drafter (per-tensor FP8 norms and codebook) parts further from the f32
    # reference after the fork; only the walks' common start is checked for it
    if fork is not None and not drafter_is_fp8(DRAFT):
        cands = ref.candidates(hidden)
        scores: list = []
        ref.refine(stash, cands, token, path=list(model.drafts), scores=scores)
        row = cands[fork].tolist()
        assert int(model.drafts[fork]) in row, (fork, model.drafts, row)
        s = scores[fork]
        margin = float(s.max() - s[row.index(int(model.drafts[fork]))])
        assert margin < NEAR_TIE, (fork, margin, model.drafts, ref_drafts)


def test_speculation_equals_plain_decoding(model, text_ids):
    prompt = text_ids[:37]
    assert speculative(model, prompt, 64) == plain(model, prompt, 64)


def test_blocks_crossing_pages_and_the_state_block(model, text_ids):
    """Decoding from a prompt that ends just before a 128-token page boundary, and across the
    4096-token STATE boundary (the 4086-token prefix: see test_qwen_dflash)."""
    prompt = text_ids[:250]
    assert speculative(model, prompt, 40) == plain(model, prompt, 40)
    prompt = text_ids[:4086]
    want, margins = plain_margins(model, prompt, 32)
    assert_parts_at_a_near_tie(speculative(model, prompt, 32), want, margins)


def test_every_width_equals_plain_decoding(engine, runner, model, text_ids):
    """Each verify width alone (the choice forced) gives plain decoding's tokens exactly."""
    for prompt in (text_ids[:37], text_ids[:4086]):
        want = plain(model, prompt, 40)
        for w in engine.verify_widths:
            runner.force_width = w
            try:
                model.reset()
                out = [model.prefill(prompt)]
                while len(out) < 40:
                    out += model.verify(out[-1])
            finally:
                runner.force_width = None
            assert out[:40] == want, (len(prompt), w)


def test_accepts_some_drafts(runner, model, text_ids):
    before = runner.speculative_stats
    speculative(model, text_ids[:200], 96)
    after = runner.speculative_stats
    accepted = after["accepted"] - before["accepted"]
    blocks = after["blocks"] - before["blocks"]
    assert blocks > 0 and accepted / blocks > 1.5, (accepted, blocks)


def test_no_compiles_after_start(runner):
    assert runner.compiles_after_start == 0
