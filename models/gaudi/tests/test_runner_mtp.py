"""MTP speculative decoding under Resin's Engine (set ``RESIN_GAUDI_QWEN35`` to a Qwen3.8-27B FP8
checkpoint directory with its MTP layer).

Every emitted token is the target's sample at its position, so speculative output must equal
non-speculative decoding exactly, greedy or seeded: alone, batched, with a stop token inside an
accepted block, and across a 4096-token STATE block boundary. References come from the same
model's plain decode path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")
K = 2

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_QWEN35, the Synapse SDK and tpc-clang",
)


@pytest.fixture(scope="module")
def runner():
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
            mtp=K,
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
        max_running_requests=4,
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


def generate(engine, prompts, params) -> list[list[int]]:
    leases = [
        engine.submit(p, sampling_params=q)
        for p, q in zip(prompts, params, strict=True)
    ]
    while engine.has_work():
        engine.step()
    return [lease.take_result() for lease in leases]


def reference(engine, runner, prompt, params) -> list[int]:
    """Non-speculative decoding of ``prompt`` on the runner's model (with the Engine's cache
    idle and its prefix cache cleared, so its slabs are free to use)."""
    from resin_models_gaudi.qwen3_5.model import Sampling, SingleSequence
    from resin_models_gaudi.qwen3_5.sampling import row_params

    engine.clear_prefix_cache()
    model = runner.model
    seq = SingleSequence(model, range(1, 13))
    vocab = model.dims.vocab

    def sampling(position):
        if not params.temperature:
            return None
        table = row_params(
            1,
            vocab,
            temperatures=[params.temperature],
            top_ps=[params.top_p],
            top_ks=[vocab if params.top_k is None else params.top_k],
            seeds=np.array([params.seed], np.uint64),
            offsets=np.array([position * 32], np.uint64),
        )
        return Sampling(
            table, np.array([params.seed], np.uint64), np.array([position * 32])
        )

    out = [seq.prefill(prompt, sampling(len(prompt)))]
    while len(out) < params.max_output_tokens:
        position = seq.pos + 1
        out.append(
            int(model.decode([seq.next_row()], [out[-1]], sampling(position))[0])
        )
        seq.advance()
    return out


def params(seed=None, **overrides):
    from resin import SamplingParams

    values = dict(max_output_tokens=24, temperature=0.0)
    if seed is not None:
        values.update(temperature=0.8, top_p=0.9, top_k=40, seed=seed)
    values.update(overrides)
    return SamplingParams(**values)


@pytest.mark.parametrize("seed", [None, 7])
def test_speculation_equals_plain_decoding(engine, runner, text_ids, seed):
    prompts = [
        text_ids[:37],
        text_ids[500:800],
        text_ids[1000:1129],
        text_ids[3000:3500],
    ]
    settings = [params(None if seed is None else seed + i) for i in range(len(prompts))]
    expected = [
        reference(engine, runner, p, q) for p, q in zip(prompts, settings, strict=True)
    ]
    engine.clear_prefix_cache()
    alone = [
        generate(engine, [p], [q])[0] for p, q in zip(prompts, settings, strict=True)
    ]
    engine.clear_prefix_cache()
    together = generate(engine, prompts, settings)
    assert alone == expected
    assert together == expected
    if seed is not None:  # the seeded streams really sample
        greedy = [reference(engine, runner, p, params()) for p in prompts]
        assert expected != greedy


def test_stop_inside_an_accepted_block(engine, runner, text_ids):
    prompt = text_ids[:37]
    plain = reference(engine, runner, prompt, params())
    stop = plain[9]
    engine.clear_prefix_cache()
    out = generate(engine, [prompt], [params(stop_token_ids=[stop])])[0]
    assert out == plain[: plain.index(stop) + 1]


def test_decoding_across_the_state_block_boundary(engine, runner, text_ids):
    prompt = text_ids[:4090]
    expected = reference(engine, runner, prompt, params(max_output_tokens=16))
    engine.clear_prefix_cache()
    assert generate(engine, [prompt], [params(max_output_tokens=16)]) == [expected]


def test_speculative_logprobs(engine, runner, text_ids):
    """Several tokens per step, each with its own column's logprobs: greedy tokens head their
    top lists."""
    prompt = text_ids[:37]
    expected = reference(engine, runner, prompt, params())
    engine.clear_prefix_cache()
    lease = engine.submit(prompt, sampling_params=params(logprobs=True, top_logprobs=3))
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
    generate(engine, [text_ids[100:1100], text_ids[3000:3003]], [params(3), params()])
    assert runner.device.compile_count == runner.compiles_at_start
