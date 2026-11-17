"""MTP speculative decoding under Resin's Engine (set ``RESIN_GAUDI_QWEN35`` to a Qwen3.8-27B FP8
checkpoint directory with its MTP layer), and the same with the DFlash 2 drafter (set
``RESIN_GAUDI_DFLASH`` to its directory; the ``dflash`` cases skip without it) and the DSpine
drafter (``RESIN_GAUDI_DSPINE``; the ``dspine`` cases).

Every emitted token is the target's sample at its position, so speculative output must equal
non-speculative decoding exactly, greedy or seeded: alone, batched, with a stop token inside an
accepted block, and across a 4096-token STATE block boundary. References come from the same
model's plain decode path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# the K and V page types: RESIN_GAUDI_CACHE_TYPES=fp8,fp8 runs the tests on an FP8 cache
CACHE_TYPES = tuple(os.environ.get("RESIN_GAUDI_CACHE_TYPES", "bf16,bf16").split(","))

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35")
DRAFT = os.environ.get("RESIN_GAUDI_DFLASH")
DSPINE = os.environ.get("RESIN_GAUDI_DSPINE")
K = 2
# drafter cases: (environment variable, directory, drafts a step verifies)
DRAFTERS = {
    "dflash": ("RESIN_GAUDI_DFLASH", DRAFT, 7),
    "dflash3": ("RESIN_GAUDI_DFLASH", DRAFT, 3),
    "dspine": ("RESIN_GAUDI_DSPINE", DSPINE, 15),
    "dspine4": ("RESIN_GAUDI_DSPINE", DSPINE, 4),
}

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires RESIN_GAUDI_QWEN35, the Synapse SDK and tpc-clang",
)


@pytest.fixture(scope="module", params=["mtp", *DRAFTERS])
def runner(request):
    """The MTP layer with 2 drafts, the DFlash drafter verifying all 7 of its drafts and 3 of
    them, the DSpine drafter all 15 and 4."""
    from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner
    from resin_models_gaudi.synapse.runtime import SynapseError

    drafter = request.param != "mtp"
    variable, directory, drafts = DRAFTERS.get(request.param, ("", None, K))
    if drafter and directory is None:
        pytest.skip(f"set {variable} to the drafter directory")
    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        built = Qwen3_5GaudiRunner(
            CHECKPOINT,
            card=None if card is None else int(card),
            max_batch_size=4,
            max_seq_len=8192
            + Qwen3_5GaudiRunner.lookahead_tokens(drafts if drafter else K, drafter),
            parent_capacity=24,
            mtp=drafts if drafter else K,
            drafter=directory if drafter else None,
            cache_types=CACHE_TYPES,
        )
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    built.policy.mode = "always"  # these tests are about the verify steps
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
    """Non-speculative decoding of ``prompt`` on the runner (plain steps through its native executor, with the
    Engine's cache idle and its prefix cache cleared, so its slabs are free to use)."""
    from qwen_sequence import PlanSequence

    from resin import SamplingParams

    engine.clear_prefix_cache()
    seq = PlanSequence(runner, range(1, 13))
    step = SamplingParams(
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        seed=params.seed if params.seed is not None else 0,
    )
    out = [seq.prefill(prompt, step)]
    while len(out) < params.max_output_tokens:
        out.append(seq.decode(out[-1], step))
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


def test_switching_between_drafts_and_plain_decoding(engine, runner, text_ids):
    """The draft policy switches a row between verify steps and plain steps (each way, chained
    or not, with the drafter's device acceptance): the tokens equal plain decoding."""
    prompts = [text_ids[:37], text_ids[1000:1129]]
    settings = [params() for _ in prompts]
    expected = [
        reference(engine, runner, p, q) for p, q in zip(prompts, settings, strict=True)
    ]
    engine.clear_prefix_cache()
    leases = [
        engine.submit(p, sampling_params=q)
        for p, q in zip(prompts, settings, strict=True)
    ]
    modes = ["never", "always", "always", "never", "always", "never", "never", "always"]
    steps = 0
    try:
        while engine.has_work():
            runner.policy.mode = modes[(steps // 3) % len(modes)]
            engine.step()
            steps += 1
    finally:
        runner.policy.mode = "always"
    assert [lease.take_result() for lease in leases] == expected


def test_stop_inside_an_accepted_block(engine, runner, text_ids):
    prompt = text_ids[:37]
    plain = reference(engine, runner, prompt, params())
    stop = plain[9]
    engine.clear_prefix_cache()
    out = generate(engine, [prompt], [params(stop_token_ids=[stop])])[0]
    assert out == plain[: plain.index(stop) + 1]


def test_decoding_across_the_state_block_boundary(engine, runner, text_ids):
    # the docs' 4090-token prefix sits on a greedy near-tie at position 4093 that the drafter's
    # 8-column verify step resolves differently from 1-column decoding (as does a plain prefill
    # ending there against decoding the same tokens), so the drafter cases start four earlier
    prompt = text_ids[: 4090 if runner.model.drafter is None else 4086]
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


def plain_margins(engine, runner, prompt, n) -> tuple[list[int], list[float]]:
    """Plain decoding of ``prompt`` (as ``reference``, greedy) and each token's top-2 logit margin (the prompt's
    token: inf)."""
    import numpy as np
    from qwen_sequence import PlanSequence

    engine.clear_prefix_cache()
    seq = PlanSequence(runner, range(1, 13))
    out, margins = [seq.prefill(prompt)], [float("inf")]
    while len(out) < n:
        out.append(seq.decode(out[-1]))
        top = np.sort(np.asarray(runner.model.logits()).reshape(-1))[-2:]
        margins.append(float(top[1] - top[0]))
    return out, margins


@pytest.mark.parametrize("length", [2040, 4086])
def test_wide_verify_steps_across_a_key_chunk_equal_plain_decoding(
    engine, runner, text_ids, length
):
    """A drafter's verify steps of 8 and 16 columns from just before a 2048-key chunk boundary (4096 is also the
    STATE block's) give plain decoding's tokens, with differences only where plain decoding's top two logits are
    within 1.0."""
    widths = [w for w in runner.model.verify_widths if w in (8, 16)]
    if runner.model.drafter is None or not widths:
        pytest.skip("a drafter's verify steps of 8 or 16 columns")
    prompt = text_ids[:length]
    want, margins = plain_margins(engine, runner, prompt, 40)
    for width in widths:
        runner.force_width = width
        try:
            engine.clear_prefix_cache()
            got = generate(engine, [prompt], [params(max_output_tokens=40)])[0]
        finally:
            runner.force_width = None
        diff = next((i for i in range(len(want)) if got[i] != want[i]), None)
        assert diff is None or margins[diff] < 1.0, (width, diff, margins[diff])
