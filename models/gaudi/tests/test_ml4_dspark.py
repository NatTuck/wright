"""DSpark speculative decoding of Mistral-Large-4 under Resin's Engine (set ``RESIN_ML4_GGUF`` and
``RESIN_ML4_DRAFT``, the release's directory with params.json, tokenizer.json, the safetensors
index and the drafter's shard), on a 4-layer target.

Every emitted token is the target's sample at its position, computed to the bit as plain decoding
computes it, so speculative output must equal plain decoding exactly: greedy or seeded, alone and
batched, with a stop token inside an accepted block, across page boundaries and past the indexer's
2048 keys. A 4-layer target cannot feed the real drafter (its taps are at layers up to 71), so
acceptance paths also run with stubbed drafts: the plain tokens (all accepted) and partly wrong ones.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_ML4_GGUF")
DRAFT = os.environ.get("RESIN_ML4_DRAFT")
LAYERS = 4

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or DRAFT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_ML4_GGUF, RESIN_ML4_DRAFT, the Synapse SDK and tpc-clang",
)


@pytest.fixture(scope="module")
def runner():
    from resin_models_gaudi.mistral_large4.runner import MistralLarge4GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = MistralLarge4GaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=8,
            max_seq_len=8192 + 3,
            parent_capacity=255,
            num_layers=LAYERS,
            draft_checkpoint=DRAFT,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield built
    built.close()


@pytest.fixture(scope="module")
def engine(runner):
    from resin import Engine

    built = Engine(
        runner,
        max_running_requests=8,
        context_length=8192,
        prefix_cache=True,
        enable_overlap=True,
    )
    yield built
    built.close()


@pytest.fixture(scope="module")
def text_ids() -> list[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(DRAFT) / "tokenizer.json"))
    docs = Path(__file__).resolve().parents[3] / "docs"
    text = "\n\n".join(p.read_text() for p in sorted(docs.glob("*.md")))
    return [1] + tokenizer.encode(text, add_special_tokens=False).ids


def generate(engine, prompts, settings) -> list[list[int]]:
    leases = [
        engine.submit(p, sampling_params=q)
        for p, q in zip(prompts, settings, strict=True)
    ]
    while engine.has_work():
        engine.step()
    return [lease.take_result() for lease in leases]


def params(seed=None, **overrides):
    from resin import SamplingParams

    values = dict(max_output_tokens=24, temperature=0.0)
    if seed is not None:
        values.update(temperature=0.8, top_p=0.9, top_k=40, seed=seed)
    values.update(overrides)
    return SamplingParams(**values)


def plain(engine, runner, prompts, settings) -> list[list[int]]:
    """Each prompt alone, decoded one token per step (no speculation)"""
    runner.speculate = False
    try:
        out = []
        for p, q in zip(prompts, settings, strict=True):
            engine.clear_prefix_cache()
            out += generate(engine, [p], [q])
    finally:
        runner.speculate = True
    engine.clear_prefix_cache()
    return out


class Drafts:
    """``Drafter.draft`` replaced by drafts read from the plain outputs: a row whose anchor is
    token i of an output (at position prompt + i) gets the next tokens, with every ``wrong``-th
    draft replaced (0: none)."""

    def __init__(self, runner, prompts, outputs, wrong: int = 0) -> None:
        self.drafter = runner.model.drafter
        self.lengths, self.outputs, self.wrong = (
            [len(p) for p in prompts],
            outputs,
            wrong,
        )
        self.calls = 0

    def __enter__(self):
        self.real, self.drafter.draft = self.drafter.draft, self
        return self

    def __exit__(self, *_):
        self.drafter.draft = self.real

    def __call__(self, rows) -> np.ndarray:
        drafts = np.zeros((len(rows), 3), np.int32)
        for r, (token, position, _) in enumerate(rows):
            for length, out in zip(self.lengths, self.outputs, strict=True):
                i = position - length
                if 0 <= i < len(out) and out[i] == token:
                    next_tokens = out[i + 1 : i + 4]
                    drafts[r, : len(next_tokens)] = next_tokens
                    break
            self.calls += 1
            if self.wrong and self.calls % self.wrong == 0:
                drafts[r, self.calls % 3] += 1
        return drafts


@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize(
    "wrong", [None, 0, 3]
)  # the real drafter; all accepted; partly wrong
def test_speculation_equals_plain_decoding(engine, runner, text_ids, seed, wrong):
    prompts = [
        text_ids[:37],
        text_ids[500:800],
        text_ids[1000:1129],
        text_ids[3000:3090],
    ]
    settings = [params(None if seed is None else seed + i) for i in range(len(prompts))]
    expected = plain(engine, runner, prompts, settings)
    before = runner.speculative_stats["accepted"]
    if wrong is None:
        alone = [
            generate(engine, [p], [q])[0]
            for p, q in zip(prompts, settings, strict=True)
        ]
        engine.clear_prefix_cache()
        together = generate(engine, prompts, settings)
    else:
        with Drafts(runner, prompts, expected, wrong):
            alone = [
                generate(engine, [p], [q])[0]
                for p, q in zip(prompts, settings, strict=True)
            ]
            engine.clear_prefix_cache()
            together = generate(engine, prompts, settings)
    assert alone == expected
    assert together == expected
    if wrong is not None:  # the stubbed drafts really exercise multi-token commits
        assert runner.speculative_stats["accepted"] > before


def test_stop_inside_an_accepted_block(engine, runner, text_ids):
    prompt = text_ids[:37]
    expected = plain(engine, runner, [prompt], [params()])
    stop = expected[0][9]
    with Drafts(runner, [prompt], expected):
        out = generate(engine, [prompt], [params(stop_token_ids=[stop])])[0]
    assert out == expected[0][: expected[0].index(stop) + 1]


def test_blocks_crossing_a_page(engine, runner, text_ids):
    """Verify columns on both sides of a 128-token page boundary"""
    prompt = text_ids[:120]
    expected = plain(engine, runner, [prompt], [params()])
    with Drafts(runner, [prompt], expected):
        assert generate(engine, [prompt], [params()]) == expected


def test_past_the_indexer_threshold(engine, runner, text_ids):
    """Sparse attention (past 2048 keys) in verify columns, and a block on the threshold"""
    assert len(text_ids) > 2100
    prompt = text_ids[:2040]
    expected = plain(engine, runner, [prompt], [params()])
    with Drafts(runner, [prompt], expected, wrong=4):
        assert generate(engine, [prompt], [params()]) == expected
    engine.clear_prefix_cache()
    prompt = text_ids[:2100]
    expected = plain(engine, runner, [prompt], [params(max_output_tokens=12)])
    assert generate(engine, [prompt], [params(max_output_tokens=12)]) == expected


def test_rejections_past_the_indexer_threshold(engine, runner, text_ids):
    """Drafts rejected at every other step leave later steps exact, in two rows past the indexer's
    threshold."""
    prompts = [text_ids[:2100], text_ids[100:2300]]
    settings = [params(max_output_tokens=128) for _ in prompts]
    expected = plain(engine, runner, prompts, settings)
    before = runner.speculative_stats["accepted"]
    with Drafts(runner, prompts, expected, wrong=2):
        assert generate(engine, prompts, settings) == expected
    assert runner.speculative_stats["accepted"] > before
    engine.clear_prefix_cache()


def test_prefix_hit_matches_fresh(engine, runner, text_ids):
    prompt = text_ids[:300]
    expected = plain(engine, runner, [prompt], [params()])
    assert (
        generate(engine, [prompt], [params()]) == expected
    )  # fresh, cached afterwards
    assert generate(engine, [prompt], [params()]) == expected


def test_speculative_logprobs(engine, runner, text_ids):
    """Several tokens per step, each with its own column's logprobs: greedy tokens head their
    top lists."""
    prompt = text_ids[:37]
    expected = plain(engine, runner, [prompt], [params()])[0]
    with Drafts(runner, [prompt], [expected]):
        lease = engine.submit(
            prompt, sampling_params=params(logprobs=True, top_logprobs=3)
        )
        events = []
        while engine.has_work():
            events.extend(engine.step())
    assert lease.take_result() == expected
    scored = [e for e in events if e.logprobs is not None]
    assert [e.token_id for e in scored] == expected
    for event in scored:
        top = event.logprobs.top_logprobs
        assert len(top) == 3 and top[0][0] == event.token_id
        assert abs(top[0][1] - event.logprobs.logprob) < 1e-6


def test_drafter_matches_the_reference(engine, runner):
    """The drafter's hidden states and base logits against the CPU reference
    (``tools/ml4_dspark_ref.py``) on random target taps, past the 2048-position window: injection,
    the block, the window's lower bound (slot field ``kmin``) and the LM head."""
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from ml4_dspark_ref import DSparkRef

    from resin_models_gaudi.mistral_large4.dspark import DRAFT_TAPS
    from resin_models_gaudi.mistral_large4.model import prefill_chunks

    engine.clear_prefix_cache()
    model, dev = runner.model, runner.device
    H, L = model.d.hidden, 2300
    rng = np.random.default_rng(0)
    taps = rng.standard_normal((len(DRAFT_TAPS), L, H)).astype(np.float32)
    blocks = model.geo.block_row(np.arange(1, 1 + 19))
    chunks = prefill_chunks(L, model.prefill_columns)
    for s in chunks:
        n = min(L - s, chunks.step)
        T = model.bucket(n)
        buf = np.zeros((len(DRAFT_TAPS), T, H), np.float32)
        buf[:, :n] = taps[:, s : s + n]
        dev.h2d(model.hidden, buf)
        positions = np.arange(s, s + n)
        model.drafter.inject(T, positions, blocks[positions // 128])
    anchor = 1000
    drafts = model.drafter.draft([(anchor, L, blocks)])[0]
    hf = dev.d2h(model.drafter.hf, (4, H), np.float32)[:3]
    base = dev.d2h(model.drafter.logits, (4, model.d.vocab), np.float32)[:3]
    base = 30 * np.tanh(base / 30)

    def corr(a, b):
        return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]

    ref = DSparkRef(DRAFT, weight_dtype=torch.bfloat16)
    ctx = ref.new_context(L + 8)
    ref.inject(ctx, torch.as_tensor(taps), L)
    want = ref.draft_block(ctx, anchor, L)
    assert corr(hf, want["hidden"].float().numpy()) > 0.999
    assert corr(base, want["base_logits"].float().numpy()) > 0.999
    assert len(drafts) == 3
    # the window matters here: without it the reference's slots see different keys
    ref.window = None
    unwindowed = ref.draft_block(ctx, anchor, L)["hidden"].float().numpy()
    assert corr(hf, unwindowed) < corr(hf, want["hidden"].float().numpy())


def test_speculative_steps_overlap(engine, runner, text_ids):
    """Under the Engine's overlap a speculative step continues from the previous one's feedback"""
    prompt = text_ids[:37]
    expected = plain(engine, runner, [prompt], [params()])
    before = runner.continuations
    with Drafts(runner, [prompt], expected):
        assert generate(engine, [prompt], [params()]) == expected
    assert runner.continuations > before


def test_no_compiles_after_start(runner):
    assert runner.compiles_after_start == 0
