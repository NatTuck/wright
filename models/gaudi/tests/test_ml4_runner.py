"""The Mistral-Large-4 Gaudi runner under Resin's Engine, on one card with the first 4 layers
(set ``RESIN_ML4_GGUF`` to the UD-Q4_K_XL GGUF directory).

Sequences decoded together match each run alone (including a
prompt past the DSA top-k window); prefix-cache hits match fresh prefill; a NaN-filled cache
arena changes nothing; seeded sampling is invariant to batching; logprobs agree with the
softcapped logits; nothing compiles after construction across unseen lengths and batch sizes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_ML4_GGUF")
TOKENIZER = Path(os.environ.get("RESIN_ML4_TOKENIZER", "/nonexistent"))
LAYERS = 4

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_ML4_GGUF, the Synapse SDK and tpc-clang",
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
            max_batch_size=32,
            max_seq_len=8192,
            parent_capacity=255,
            num_layers=LAYERS,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    arena = built.model.arena  # stale cache contents must never matter
    built.device.memset32(arena.address, 0x7FC00000, arena.nbytes // 4)
    yield built
    built.close()


@pytest.fixture(scope="module")
def engine(runner):
    from resin import Engine

    built = Engine(
        runner, max_running_requests=32, context_length=8192, prefix_cache=True
    )
    yield built
    built.close()


@pytest.fixture(scope="module")
def text_ids() -> list[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    docs = Path(__file__).resolve().parents[3] / "docs"
    text = "\n\n".join(p.read_text() for p in sorted(docs.glob("*.md")))
    return [1] + tokenizer.encode(text, add_special_tokens=False).ids


def generate(engine, prompts, tokens=6, params=None, events=None) -> list[list[int]]:
    from resin import SamplingParams

    if params is None:
        params = [SamplingParams(max_output_tokens=tokens, temperature=0.0)] * len(
            prompts
        )
    leases = [
        engine.submit(p, sampling_params=q)
        for p, q in zip(prompts, params, strict=True)
    ]
    while engine.has_work():
        for event in engine.step():
            if events is not None:
                events.append(event)
    return [lease.take_result() for lease in leases]


def sampled(seed, **overrides):
    from resin import SamplingParams

    values = dict(max_output_tokens=8, temperature=0.8, top_p=0.9, top_k=40, seed=seed)
    values.update(overrides)
    return SamplingParams(**values)


def test_batch_first_tokens_equal_alone_and_order_invariant(engine, text_ids):
    """A batch's rows never mix: first tokens (including a prompt past the DSA window) equal each
    prompt alone, and outputs do not depend on the rows' order. Later tokens are compared at a
    matched batch shape."""
    prompts = [
        text_ids[o : o + n] for o, n in ((0, 37), (500, 300), (1000, 2300), (3000, 17))
    ]
    engine.clear_prefix_cache()
    alone = [generate(engine, [p], tokens=1)[0] for p in prompts]
    engine.clear_prefix_cache()
    together = generate(engine, prompts)
    assert [t[:1] for t in together] == alone
    engine.clear_prefix_cache()
    reversed_ = generate(engine, prompts[::-1])
    assert reversed_[::-1] == together


def test_prefix_hit_matches_fresh(engine, text_ids):
    prompt = text_ids[200 : 200 + 700]
    engine.clear_prefix_cache()
    fresh = generate(engine, [prompt])
    hit = generate(engine, [prompt])  # its 640 cached tokens are reused
    assert hit == fresh


def test_seeded_sampling_is_order_invariant(engine, text_ids):
    prompts = [text_ids[o : o + 50] for o in (0, 100, 200)]
    params = [sampled(7 + i) for i in range(3)]
    alone = [
        generate(engine, [p], params=[sampled(7 + i, max_output_tokens=1)])[0]
        for i, p in enumerate(prompts)
    ]
    together = generate(engine, prompts, params=params)
    assert [t[:1] for t in together] == alone  # prefill draws: per row
    again = generate(engine, prompts[::-1], params=params[::-1])
    assert again[::-1] == together


def test_logprobs_agree_with_logits(engine, runner, text_ids):
    from resin import SamplingParams
    from resin_models_gaudi.mistral_large4.model import Row

    prompt = text_ids[:40]
    events = []
    params = SamplingParams(
        max_output_tokens=1, temperature=0.0, logprobs=True, top_logprobs=5
    )
    engine.clear_prefix_cache()
    (tokens,) = generate(engine, [prompt], params=[params], events=events)
    (event,) = [e for e in events if e.logprobs is not None]
    from resin_models_gaudi.mistral_large4.model import prefill_chunks

    model = runner.model
    blocks = model.geo.block_row(np.arange(1, 4))
    chunks = prefill_chunks(len(prompt))
    for s in chunks:  # the runner's launches (same graphs, same numbers)
        chunk = np.asarray(prompt[s : s + chunks.step])
        logits = model.forward(
            chunk, s + np.arange(len(chunk)), [Row(0, s, len(chunk), blocks)]
        )
    cap = runner.config.final_logit_softcap
    z = cap * np.tanh(logits[-1].astype(np.float64) / cap)
    logp = z - z.max() - np.log(np.exp(z - z.max()).sum())
    order = np.argsort(-logp, kind="stable")[:5]
    assert tokens == [int(order[0])] and event.token_id == tokens[0]
    assert abs(event.logprobs.logprob - logp[order[0]]) < 2e-3
    top = event.logprobs.top_logprobs
    assert [t for t, _ in top] == order.tolist()
    np.testing.assert_allclose([v for _, v in top], logp[order], atol=2e-3)


def test_no_compiles_after_start(engine, runner, text_ids):
    prompts = [
        text_ids[o : o + n] for o, n in ((11, 3), (50, 61), (300, 129), (900, 1500))
    ]
    generate(engine, prompts, tokens=3)
    generate(engine, prompts[:3], tokens=2)
    generate(engine, [text_ids[5:20]] * 11, tokens=2)
    assert runner.compiles_after_start == 0


def assert_tops_close(tops, reference, atol):
    """equal top logprobs within atol; ids may differ only where values tie within it (also with
    a token just outside the list, at its last value)"""
    np.testing.assert_allclose(
        [v for _, v in tops], [v for _, v in reference], atol=atol
    )
    for (t, v), (r, w) in zip(tops, reference, strict=True):
        if t != r:
            near = dict(reference).get(t, reference[-1][1])
            assert abs(near - v) < atol, (tops, reference)


def prompt_scores(engine, prompt, top, targets=None):
    """The lease's prompt logprobs (index j: the distribution after prompt[:j])."""
    from resin import SamplingParams

    params = SamplingParams(
        max_output_tokens=1,
        temperature=0.0,
        prompt_logprobs=True,
        prompt_top_logprobs=top,
    )
    lease = engine.submit(
        prompt, sampling_params=params, prompt_target_logprobs=targets
    )
    while engine.has_work():
        engine.step()
    lease.take_result()
    return lease.prompt_logprobs


@pytest.mark.parametrize(
    "length,checks,atol",
    [(12, (6, 11), 1e-2), (40, (17, 30, 39), 5e-2), (80, (50, 65, 79), 1e-2)],
)
def test_prompt_logprobs_match_generation(engine, text_ids, length, checks, atol):
    """Position j's prompt score is the distribution after prompt[:j], which generating from
    that prefix reports too, within ``atol``."""
    from resin import SamplingParams

    prompt = text_ids[:length]
    engine.clear_prefix_cache()
    targets = [[]] + [[prompt[0], 7] for _ in prompt[1:]]
    scored = prompt_scores(engine, prompt, 5, targets)
    assert scored[0] is None and len(scored) == len(prompt)
    for j in checks:
        entry = scored[j]
        tops = entry.top_logprobs
        assert len(tops) == 5 and all(a[1] >= b[1] for a, b in zip(tops, tops[1:]))
        engine.clear_prefix_cache()
        events = []
        params = SamplingParams(
            max_output_tokens=1, temperature=0.0, logprobs=True, top_logprobs=5
        )
        generate(engine, [prompt[:j]], params=[params], events=events)
        reference = events[0].logprobs.top_logprobs
        assert_tops_close(tops, reference, atol)
        by_id = dict(tops)
        if prompt[j] in by_id:  # the prompt score and its top entries: one distribution
            assert abs(entry.logprob - by_id[prompt[j]]) < 1e-5
        assert [t for t, _ in entry.target_logprobs] == [prompt[0], 7]


def test_decode_batches_above_sixteen_equal_their_groups(engine, text_ids):
    """A batch of 21 rows equals its groups of 16 and 5 alone"""
    prompts = [text_ids[o : o + 20 + o % 7] for o in range(0, 40 * 21, 40)]
    engine.clear_prefix_cache()
    together = generate(engine, prompts, tokens=4)
    engine.clear_prefix_cache()
    first = generate(engine, prompts[:16], tokens=4)
    engine.clear_prefix_cache()
    rest = generate(engine, prompts[16:], tokens=4)
    assert together == first + rest


def test_sparse_decode_matches_dense_masked(engine, runner, text_ids):
    """Past 2048 keys a decode step attends to each column's DSA-listed keys only. A batch mixing a 2300-token row (a 2048-key list), a 1500-token row and a
    short row (every key listed) generates the same greedy tokens as dense attention masked by
    the same selection, at the same shapes."""
    assert len(text_ids) > 2400
    prompts = [text_ids[:2300], text_ids[100:1600], text_ids[50:80]]
    model = runner.model
    out = {}
    try:
        for sparse in (False, True):
            model.sparse = sparse
            engine.clear_prefix_cache()
            out[sparse] = generate(engine, prompts, tokens=8)
    finally:
        model.sparse = True
    assert out[True] == out[False]
