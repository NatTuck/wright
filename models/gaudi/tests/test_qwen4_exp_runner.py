"""The Qwen3.8-Flash-Next Gaudi runner under Resin's Engine, on the layers ``RESIN_FNEXT_LAYERS`` (default 4) of
the checkpoint ``RESIN_GAUDI_FNEXT`` (a truncated download works): sequences decoded together match each run
alone; seeded sampling is invariant to batching; logprobs agree with the
logits; nothing compiles after construction."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_GAUDI_FNEXT")
N = int(os.environ.get("RESIN_FNEXT_LAYERS", "4"))

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_FNEXT, the Synapse SDK and tpc-clang",
)


@pytest.fixture(scope="module")
def runner():
    from resin_models_gaudi.qwen4_exp.runner import Qwen4ExpGaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = Qwen4ExpGaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=8,
            max_seq_len=2048,
            parent_capacity=16,
            layers=N,
            table=False,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    model = built.model  # stale cache contents must never matter
    built.device.memset32(model.arena, 0x7FC00000, model.arena_bytes // 4)
    yield built
    built.close()


@pytest.fixture(scope="module")
def engine(runner):
    from resin import Engine

    built = Engine(
        runner, max_running_requests=8, context_length=2048, prefix_cache=True
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


def generate(engine, prompts, tokens=6, params=None) -> list[list[int]]:
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
        engine.step()
    return [lease.take_result() for lease in leases]


def sampled(seed, **overrides):
    from resin import SamplingParams

    values = dict(max_output_tokens=8, temperature=0.8, top_p=0.9, top_k=40, seed=seed)
    values.update(overrides)
    return SamplingParams(**values)


def test_batch_equals_alone(engine, text_ids):
    prompts = [
        text_ids[o : o + n]
        for o, n in ((0, 37), (500, 300), (1000, 1100), (9000, 129), (3000, 700))
    ]
    engine.clear_prefix_cache()
    alone = [generate(engine, [p])[0] for p in prompts]
    engine.clear_prefix_cache()
    assert generate(engine, prompts) == alone


@pytest.mark.skip(
    reason="the prefix cache publishes at the 4096-token STATE block; contexts stay within the 2048-token "
    "indexer budget until the sparse QSA path is wired"
)
def test_prefix_hits_equal_fresh(engine, text_ids):
    first = text_ids[:1500]
    second = text_ids[:1200] + text_ids[6000:6200]
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
        sampled(14, temperature=1.0, top_p=0.95, top_k=None),
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


def test_greedy_logprobs_agree_with_the_logits(engine, runner, text_ids):
    from resin import SamplingParams

    events = []
    params = SamplingParams(
        max_output_tokens=3, temperature=0.0, logprobs=True, top_logprobs=5
    )
    engine.clear_prefix_cache()
    lease = engine.submit(text_ids[200:260], sampling_params=params)
    while engine.has_work():
        events += engine.step()
    lease.take_result()
    scored = [e for e in events if e.logprobs is not None]
    assert len(scored) == 3
    for event in scored:
        top = event.logprobs.top_logprobs
        assert len(top) == 5 and top[0][0] == event.token_id
        assert abs(top[0][1] - event.logprobs.logprob) < 1e-6
        assert all(a[1] >= b[1] for a, b in zip(top, top[1:]))
    raw = runner.model.last_logits()[0].astype(
        np.float64
    )  # the last step's distribution
    lsm = raw - raw.max()
    lsm -= np.log(np.exp(lsm).sum())
    order = np.argsort(-raw, kind="stable")[:5]
    last = scored[-1].logprobs.top_logprobs
    assert [t for t, _ in last] == order.tolist()
    np.testing.assert_allclose([v for _, v in last], lsm[order], atol=2e-4)


def test_no_compiles_after_start(engine, runner, text_ids):
    engine.clear_prefix_cache()
    generate(engine, [text_ids[100:1100], text_ids[3000:3003], text_ids[20:777]], 3)
    assert runner.device.compile_count == runner.compiles_at_start
