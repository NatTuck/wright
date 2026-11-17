"""Qwen3.8-27B decode in the 64-row bucket under Resin's Engine, with MTP-2 drafts (set
``RESIN_GAUDI_QWEN35`` to a Qwen3.8-27B FP8 checkpoint directory).

A batch of 33 to 64 rows decodes plainly in the 64-row bucket: each sequence's greedy tokens
follow the sequence decoded alone (with drafts) up to a near tie, and nothing compiles after
construction. Plain steps store the MTP layer's keys, so a row that speculates again after them
accepts its drafts as often as a row that speculated throughout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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

ROWS = 40  # past the 32-row bucket


@pytest.fixture(scope="module")
def runner():
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = Qwen3_5GaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=64,
            max_seq_len=8192 + Qwen3_5GaudiRunner.lookahead_tokens(2, False),
            mtp=2,
            cache_types=CACHE_TYPES,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    model = built.model
    built.device.memset32(
        model.arena, 0x7FC00000, model.blocks * 1024 * 128
    )  # stale = NaN
    yield built
    built.close()


@pytest.fixture(scope="module")
def engine(runner):
    from resin import Engine

    built = Engine(
        runner,
        max_running_requests=64,
        context_length=8192,
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


def generate(engine, prompts, tokens=8):
    """greedy tokens of each prompt and each step's top 20 (token, logprob)"""
    from resin import SamplingParams

    params = SamplingParams(
        max_output_tokens=tokens, temperature=0.0, logprobs=True, top_logprobs=20
    )
    leases = [engine.submit(p, sampling_params=params) for p in prompts]
    row = {int(lease): k for k, lease in enumerate(leases)}
    tops: list[list] = [[] for _ in prompts]
    while engine.has_work():
        for event in engine.step():
            if event.logprobs is not None and int(event.request_id) in row:
                tops[row[int(event.request_id)]].append(event.logprobs.top_logprobs)
    return [lease.take_result() for lease in leases], tops


def test_batch_of_40_follows_alone(engine, runner, text_ids):
    """40 sequences of different lengths (some past one 4096-token STATE block) decode in one
    64-row step each. A sequence's tokens equal the sequence run alone up to a near tie, as in
    the 32-row bucket: where a row first differs, the batch's token is within 0.75 nats of the
    solo run's top token there."""
    prompts = [
        text_ids[97 * i : 97 * i + n]
        for i, n in enumerate(
            [37 + 61 * k for k in range(36)] + [4090, 4100, 4500, 5000]
        )
    ]
    assert len(prompts) == ROWS
    engine.clear_prefix_cache()
    alone, tops = [], []
    for p in prompts:
        tokens, top = generate(engine, [p])
        alone += tokens
        tops += top
    engine.clear_prefix_cache()
    together, _ = generate(engine, prompts)
    differ = []
    for r, (a, b) in enumerate(zip(alone, together, strict=True)):
        if a == b:
            continue
        d = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        top = tops[r][d]
        differ.append(r)
        lp = dict(top)
        assert b[d] in lp and top[0][1] - lp[b[d]] < 0.75, (r, d, top[:5], b[d])
    assert len(differ) <= ROWS // 5, differ
    assert runner.device.compile_count == runner.compiles_at_start


def accepted_rate(runner, run) -> float:
    """drafts accepted per draft verified while ``run()`` runs (the native executor's counters)"""
    before = runner.speculative_stats
    run()
    after = runner.speculative_stats
    drafts = after["drafts"] - before["drafts"]
    return (after["accepted"] - before["accepted"]) / max(1, drafts)


def test_plain_steps_store_the_mtp_keys(engine, runner, text_ids):
    """A row that decodes 48 plain steps and then speculates accepts about as many drafts as a
    row that speculated throughout (its drafts attend over the MTP keys of the plain steps)."""
    from resin import SamplingParams

    prompt = text_ids[2000:2400]
    params = SamplingParams(max_output_tokens=112, temperature=0.0)

    def tail(mode_first: str):
        def run():
            lease = engine.submit(prompt, sampling_params=params)
            produced = 0
            runner.policy.mode = mode_first
            while engine.has_work():
                produced += len(engine.step())
                if produced >= 48:
                    runner.policy.mode = "always"
            lease.take_result()

        return run

    engine.clear_prefix_cache()
    runner.policy.mode = "always"
    throughout = accepted_rate(runner, tail("always"))
    engine.clear_prefix_cache()
    after_plain = accepted_rate(runner, tail("never"))
    runner.policy.mode = "auto"
    assert throughout > 0.3, throughout
    assert after_plain > 0.8 * throughout, (after_plain, throughout)
