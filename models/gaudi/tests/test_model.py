"""The full FP8 model on a Gaudi card, one sequence at a time through the native executor with plans made here
(``qwen_sequence``; set ``RESIN_GAUDI_QWEN35`` to a Qwen3.8-27B FP8 checkpoint directory; the model takes ~3
minutes to load and compile).

Greedy tokens of a 37-token prompt match the expected tokens; chunked and continued prefill and
decode equal single-shot prefill; a decode batch equals its sequences decoded alone; nothing
compiles after construction.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

# the K and V page types: RESIN_GAUDI_CACHE_TYPES=fp8,fp8 runs the tests on an FP8 cache
CACHE_TYPES = tuple(os.environ.get("RESIN_GAUDI_CACHE_TYPES", "bf16,bf16").split(","))

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_QWEN35, the Synapse SDK and tpc-clang",
)

# the first 37 tokens of resin's docs (Qwen3.8 tokenizer) and their greedy continuation
PROMPT = [2, 36593, 271, 1026, 258, 59666, 1622, 321, 4939, 14834, 494, 1558, 18336, 10993, 13,
          198, 760, 12654, 94411, 1510, 408, 258, 22728, 63, 17189, 5435, 279, 4434, 7499, 278,
          198, 63, 4422, 5218, 20942, 42181, 45924]  # fmt: skip
EXPECTED_TOKENS = [11, 1345, 1510, 408, 258]


@pytest.fixture(scope="module")
def engine():
    """The runner (its native executor makes the steps)."""
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = Qwen3_5GaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=4,
            max_seq_len=8192,
            parent_capacity=12,
            cache_types=CACHE_TYPES,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield built
    built.close()


@pytest.fixture(scope="module")
def model(engine):
    """One sequence over the first six slabs."""
    from qwen_sequence import PlanSequence

    return PlanSequence(engine, range(1, 7))


@pytest.fixture(scope="module")
def text_ids() -> list[int]:
    """Resin's docs as Qwen3.8 tokens (real text keeps greedy margins away from ties)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json"))
    docs = Path(__file__).resolve().parents[3] / "docs"
    text = "\n\n".join(p.read_text() for p in sorted(docs.glob("*.md")))
    ids = tokenizer.encode(text).ids
    assert len(ids) > 2100
    return ids


def logits_after(model, pieces, ids) -> np.ndarray:
    """Logits after ``pieces``: ("p", n) prefills the next n ids, ("d", n) decodes them."""
    model.reset()
    i = 0
    for kind, n in pieces:
        if kind == "p":
            model.prefill(ids[i : i + n])
        else:
            for k in range(n):
                model.decode(ids[i + k])
        i += n
    return model.logits()


def test_greedy_matches_expected_tokens(model):
    model.reset()
    tokens = [model.prefill(PROMPT)]
    for _ in range(len(EXPECTED_TOKENS) - 1):
        tokens.append(model.decode(tokens[-1]))
    assert tokens == EXPECTED_TOKENS
    assert np.isfinite(model.logits()).all()


@pytest.mark.parametrize(
    ("single", "split"),
    [
        ([("p", 337)], [("p", 37), ("p", 300)]),  # unaligned continuation
        ([("p", 129)], [("p", 126), ("d", 3)]),  # decode across a page boundary
        (
            [("p", 2060)],
            [("p", 2040), ("p", 20)],
        ),  # a chunk across a key-chunk boundary
    ],
)
def test_continuation_equals_single_shot(model, text_ids, single, split):
    """Split prefill and decode give logits close to single-shot prefill, with the same argmax."""
    a, b = logits_after(model, single, text_ids), logits_after(model, split, text_ids)
    assert a.argmax() == b.argmax()
    assert np.corrcoef(a, b)[0, 1] > 0.995


def test_aligned_chunks_are_exact(model, text_ids):
    """Chunks on 64-token boundaries give the same logits as one prefill, and runs are deterministic."""
    single = logits_after(model, [("p", 300)], text_ids)  # 256 + 44
    assert np.array_equal(
        single, logits_after(model, [("p", 128), ("p", 172)], text_ids)
    )
    assert np.array_equal(single, logits_after(model, [("p", 300)], text_ids))


def test_decode_batch_equals_single(engine, text_ids):
    """Three sequences at different lengths decoded together (padded to a batch of 4) give the
    tokens and logits they give decoded one at a time."""
    from qwen_sequence import PlanSequence, decode_together

    lengths, offsets = (37, 300, 2100), (0, 500, 1000)
    seqs = [PlanSequence(engine, range(1 + 4 * i, 5 + 4 * i)) for i in range(3)]

    def start():
        firsts = []
        for seq, n, o in zip(seqs, lengths, offsets, strict=True):
            seq.reset()
            firsts.append(seq.prefill(text_ids[o : o + n]))
        return firsts

    tokens = start()
    batched_tokens, batched_logits = [], []
    for _ in range(3):
        tokens = decode_together(seqs, tokens)
        batched_logits.append(engine.model.logits(len(seqs)))
        batched_tokens.append(tokens)

    tokens = start()
    for step in range(3):
        logits = []
        for i, seq in enumerate(seqs):
            tokens[i] = seq.decode(tokens[i])
            logits.append(seq.logits())
        assert tokens == batched_tokens[step]
        np.testing.assert_allclose(np.stack(logits), batched_logits[step], atol=0.02)


def test_no_compiles_after_start(engine, model):
    model.reset()
    model.prefill(PROMPT * 3)
    model.decode(1)
    assert engine.compiles_after_start == 0
