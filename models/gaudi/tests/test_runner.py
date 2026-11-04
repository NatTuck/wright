"""The Gaudi runner under Resin's Engine (set ``RESIN_GAUDI_QWEN35`` to a Qwen3.8-27B FP8
checkpoint directory).

Greedy outputs match the expected tokens; sequences decoded together match each run alone; prefix-cache hits
(KV pages and the 4096-token STATE snapshot) match fresh prefill; a NaN-filled cache pool changes
nothing; seeded sampling is invariant to batching; logprobs agree with the logits; prompt
logprobs agree with generating from each prefix; nothing compiles after construction across unseen lengths and batch
sizes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

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
def runner():
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = Qwen3_5GaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=8,
            max_seq_len=16384,
            parent_capacity=24,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    # stale cache contents must never matter
    model = built.model
    built.device.memset32(model.arena, 0x7FC00000, model.blocks * 1024 * 128)
    yield built
    built.close()


@pytest.fixture(scope="module")
def engine(runner):
    from resin import Engine

    built = Engine(
        runner,
        max_running_requests=8,
        context_length=16384,
        prefix_cache=True,
        enable_overlap=True,
    )
    yield built
    built.close()


@pytest.fixture(scope="module")
def text_ids() -> list[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json"))
    docs = Path(__file__).resolve().parents[3] / "docs"
    text = "\n\n".join(p.read_text() for p in sorted(docs.glob("*.md")))
    return tokenizer.encode(text).ids


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


def test_greedy_matches_expected_tokens(engine):
    engine.clear_prefix_cache()
    assert generate(engine, [PROMPT], len(EXPECTED_TOKENS)) == [EXPECTED_TOKENS]


def test_batch_equals_alone(engine, text_ids):
    prompts = [
        text_ids[o : o + n]
        for o, n in ((0, 37), (500, 300), (1000, 2100), (9000, 129), (7000, 5000))
    ]
    engine.clear_prefix_cache()
    alone = [generate(engine, [p])[0] for p in prompts]
    engine.clear_prefix_cache()
    assert generate(engine, prompts) == alone


def test_prefix_hits_equal_fresh(engine, text_ids):
    first = text_ids[:4500]
    second = (
        text_ids[:4300] + text_ids[6000:6200]
    )  # shares the 4096-token STATE snapshot
    engine.clear_prefix_cache()
    fresh = [generate(engine, [first])[0]]
    engine.clear_prefix_cache()
    fresh.append(generate(engine, [second])[0])
    engine.clear_prefix_cache()
    _, hits0, _, _ = engine.prefix_cache_stats
    cached = [generate(engine, [first])[0], generate(engine, [second])[0]]
    _, hits1, _, _ = engine.prefix_cache_stats
    assert hits1 > hits0
    assert cached == fresh


def test_seeded_sampling_is_invariant(engine, text_ids):
    prompts = [
        text_ids[o : o + n] for o, n in ((0, 37), (500, 300), (1000, 900), (3000, 64))
    ]
    params = [
        sampled(11),
        sampled(12, top_k=None),
        sampled(13, top_p=1.0),
        sampled(
            14, temperature=1.0, top_p=0.95, top_k=None
        ),  # nuclei past the candidates
    ]
    engine.clear_prefix_cache()
    alone = [
        generate(engine, [p], params=[q])[0]
        for p, q in zip(prompts, params, strict=True)
    ]
    engine.clear_prefix_cache()
    together = generate(engine, prompts, params=params)
    again = generate(engine, prompts, params=params)
    assert together == alone and again == alone
    assert alone != generate(engine, prompts)  # sampling is not greedy


def test_greedy_logprobs_agree_with_the_logits(engine, runner):
    from resin import SamplingParams

    events = []
    params = SamplingParams(
        max_output_tokens=len(EXPECTED_TOKENS), temperature=0.0, logprobs=True, top_logprobs=5
    )
    engine.clear_prefix_cache()
    assert generate(engine, [PROMPT], params=[params], events=events) == [EXPECTED_TOKENS]
    scored = [e for e in events if e.logprobs is not None]
    assert len(scored) == len(EXPECTED_TOKENS)
    for event in scored:
        top = event.logprobs.top_logprobs
        assert len(top) == 5 and top[0][0] == event.token_id
        assert abs(top[0][1] - event.logprobs.logprob) < 1e-6
        assert all(a[1] >= b[1] for a, b in zip(top, top[1:]))
    raw = runner.model.logits().astype(np.float64)  # the last step's distribution
    lsm = raw - raw.max()
    lsm -= np.log(np.exp(lsm).sum())
    order = np.argsort(-raw, kind="stable")[:5]
    last = scored[-1].logprobs.top_logprobs
    assert [t for t, _ in last] == order.tolist()
    np.testing.assert_allclose([v for _, v in last], lsm[order], atol=2e-4)


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


def first_distribution(engine, prefix, top=5):
    """The greedy first token's top logprobs after ``prefix``."""
    from resin import SamplingParams

    events = []
    params = SamplingParams(
        max_output_tokens=1, temperature=0.0, logprobs=True, top_logprobs=top
    )
    generate(engine, [prefix], params=[params], events=events)
    return events[0].logprobs.top_logprobs


@pytest.mark.parametrize("top", [5, 200])
def test_prompt_logprobs_match_generation(engine, top):
    """Position j's prompt score is the next-token distribution after prompt[:j], which
    generating from that prefix reports too (same chunk shape, so the same numbers)."""
    engine.clear_prefix_cache()
    targets = [[]] + [[PROMPT[0], 7] for _ in PROMPT[1:]]
    scored = prompt_scores(engine, PROMPT, top, targets)
    assert scored[0] is None and len(scored) == len(PROMPT)
    for j in (1, 6, 21, 36):
        entry = scored[j]
        tops = entry.top_logprobs
        assert len(tops) == top
        assert all(a[1] >= b[1] for a, b in zip(tops, tops[1:]))
        engine.clear_prefix_cache()
        reference = first_distribution(engine, PROMPT[:j])
        assert [t for t, _ in tops[:5]] == [t for t, _ in reference]
        np.testing.assert_allclose(
            [v for _, v in tops[:5]], [v for _, v in reference], atol=1e-5
        )
        # the prompt token's logprob and the requested targets come from the same distribution
        by_id = dict(tops)
        if PROMPT[j] in by_id:
            assert abs(entry.logprob - by_id[PROMPT[j]]) < 1e-5
        assert [t for t, _ in entry.target_logprobs] == [PROMPT[0], 7]
        for t, v in entry.target_logprobs:
            if t in by_id:
                assert abs(v - by_id[t]) < 1e-5
            assert v <= tops[0][1] + 1e-6


def test_prompt_logprobs_across_chunks(engine, text_ids):
    """Positions on both sides of the 4096-token chunk boundary, against the greedy token's
    logprob when generating from that prefix (the same chunk shapes)."""
    prompt = text_ids[:4200]
    checks = (4094, 4095, 4096, 4150)
    references = {}
    for j in checks:
        engine.clear_prefix_cache()
        token, value = first_distribution(engine, prompt[:j], top=1)[0]
        references[j] = (token, value)
    targets = [[] for _ in prompt]
    for j, (token, _) in references.items():
        targets[j] = [token]
    engine.clear_prefix_cache()
    scored = prompt_scores(engine, prompt, 0, targets)
    assert len(scored) == len(prompt) and scored[0] is None
    assert all(e.logprob <= 0 for e in scored[1:])
    for j, (token, value) in references.items():
        assert scored[j].target_logprobs[0][0] == token
        assert abs(scored[j].target_logprobs[0][1] - value) < 1e-5, j


def test_decode_overlaps(engine, text_ids):
    """Decode steps continue on the device while the Engine commits the previous one."""
    from resin import SamplingParams

    before = engine.overlap_stats[1]
    engine.clear_prefix_cache()
    lease = engine.submit(
        text_ids[:100],
        sampling_params=SamplingParams(max_output_tokens=40, temperature=0.0),
    )
    while engine.has_work():
        engine.step()
    lease.take_result()
    assert engine.overlap_stats[1] >= before + 3


def test_no_compiles_after_start(engine, runner, text_ids):
    engine.clear_prefix_cache()
    generate(engine, [text_ids[100:1100], text_ids[3000:3003], text_ids[20:777]], 3)
    assert runner.device.compile_count == runner.compiles_at_start
