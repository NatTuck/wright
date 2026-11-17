"""Qwen3.8-27B tensor parallel over two cards with native rank dispatch, against one card.

Rank 0's native executor sends each runner call to rank 1 over a ``GaudiRanks`` group (Unix sockets, no Gloo), and
rank 1 makes the same steps (``serve``). The model keeps its first LAYERS layers, so that it loads in minutes. On two
cards, without a drafter, with the DSpine drafter and with the MTP layer (MTP_DRAFTS drafts; every rank runs the whole
MTP layer), every step of up to 8 rows verifying drafts:

- without a drafter, the prefill logits of three prompts correlate with one card's above 0.9995, and the argmax agrees
  unless the one-card top-2 margin is under 1.0 (each rank's FP8 activation scales come from its shard; one card runs
  the fused layers, two cards the split ones). With a drafter the logits buffer also holds the drafter's work, so the
  drafter runs check decoding only;
- greedy decoding gives one card's tokens up to the first position where one card's top-2 margin is under 1.0;
- the follower makes every call of the driver, and both ranks end with the same logits;
- no graph compiles after the start.

The cards come from ``VT_HPU_CARDS`` (the grant of the card lock script): the test never takes cards by
itself. ``QWEN_TP_CHECKPOINT`` and ``QWEN_TP_DRAFTER`` give the checkpoint directories; the test skips without them."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# the K and V page types: RESIN_GAUDI_CACHE_TYPES=fp8,fp8 runs the tests on an FP8 cache
CACHE_TYPES = tuple(os.environ.get("RESIN_GAUDI_CACHE_TYPES", "bf16,bf16").split(","))

CARDS = [c for c in re.split(r"[ ,]+", os.environ.get("VT_HPU_CARDS", "").strip()) if c]
CHECKPOINT = Path(os.environ.get("QWEN_TP_CHECKPOINT", ""))
DRAFTER = Path(os.environ.get("QWEN_TP_DRAFTER", ""))
LAYERS = 4
LENGTHS = (37, 200, 1000)
MTP_DRAFTS = 2
DECODE = 24  # greedy tokens a row

pytestmark = pytest.mark.skipif(
    len(CARDS) < 2
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None
    or not os.environ.get("QWEN_TP_CHECKPOINT")
    or not os.environ.get("QWEN_TP_DRAFTER")
    or not (CHECKPOINT / "config.json").exists()
    or not (DRAFTER / "config.json").exists(),
    reason="requires two cards granted in VT_HPU_CARDS, the Synapse SDK, tpc-clang and the Qwen3.8-27B and "
    "DSpine checkpoints",
)


def _worker(
    rank: int,
    ranks: int,
    path: str,
    token: bytes,
    cards: list[str],
    out: str,
    drafter: str | None,
    mtp: int = 0,
) -> None:
    import functools

    from tokenizers import Tokenizer

    import resin_models_gaudi.qwen3_5.runner as qwen
    from resin import Engine, SamplingParams
    from resin._resin import GaudiRanks
    from resin_models_gaudi import ranks as par
    from resin_models_gaudi.qwen3_5.model import Qwen35Gaudi

    os.environ["RESIN_QWEN_DRAFT_POLICY"] = "always"
    qwen.Qwen35Gaudi = functools.partial(Qwen35Gaudi, layers=LAYERS)
    group = GaudiRanks(path, token, rank, ranks) if ranks > 1 else None
    runner = qwen.Qwen3_5GaudiRunner(
        CHECKPOINT,
        card=par.accel_index(cards[rank]),
        max_batch_size=4,
        max_seq_len=4096,
        parent_capacity=8,
        drafter=drafter,
        mtp=mtp,
        native_ranks=group,
        cache_types=CACHE_TYPES,
    )
    if rank > 0:
        calls = runner.serve()
        np.save(f"{out}.{rank}.npy", runner.model.logits())
        Path(f"{out}.{rank}.json").write_text(
            json.dumps({"calls": calls, "speculation": runner.speculative_stats})
        )
        runner.close()
        return
    engine = Engine(
        runner,
        max_running_requests=4,
        context_length=4096,
        prefix_cache=False,
        enable_overlap=True,
    )
    tokenizer = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    readme = Path(__file__).parents[3] / "README.md"
    text = tokenizer.encode(readme.read_text()).ids
    assert len(text) > 2 * max(LENGTHS)
    one = SamplingParams(max_output_tokens=1, temperature=0.0)
    logits = {}
    for length in LENGTHS:
        lease = engine.submit(text[length : 2 * length], sampling_params=one)
        while engine.has_work():
            engine.step()
        lease.take_result()
        logits[f"n{length}"] = runner.model.logits()
    greedy = SamplingParams(
        max_output_tokens=DECODE, temperature=0.0, logprobs=True, top_logprobs=2
    )
    leases = [
        engine.submit(text[start : start + 300], sampling_params=greedy)
        for start in (0, 500)
    ]
    margins: dict[int, list[float]] = {lease.request_id: [] for lease in leases}
    while engine.has_work():
        for event in engine.step():
            if event.logprobs is not None:
                top = [float(v) for _, v in event.logprobs.top_logprobs or ()]
                margins[event.request_id].append(
                    top[0] - top[1] if len(top) == 2 else float("inf")
                )
    decoded = [
        {
            "tokens": [int(t) for t in lease.take_result()],
            "margins": margins[lease.request_id],
        }
        for lease in leases
    ]
    # the last step is a prefill, so that every rank's logits come from the same step
    lease = engine.submit(text[50:90], sampling_params=one)
    while engine.has_work():
        engine.step()
    lease.take_result()
    np.save(f"{out}.0.npy", runner.model.logits())
    np.savez(f"{out}.logits.npz", **logits)
    result = {
        "decoded": decoded,
        "calls": None,
        "compiles": runner.compiles_after_start,
        "speculation": runner.speculative_stats,
    }
    engine.close()
    runner.close()  # rank 1 stops serving
    Path(f"{out}.0.json").write_text(json.dumps(result))


def _run(ranks: int, out: Path, drafter: Path | None, mtp: int = 0) -> None:
    from resin.process import spawn

    entries = [Path(f"/sys/class/accel/accel{i}") for i in CARDS[:ranks]]
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    path = os.path.join(tempfile.mkdtemp(prefix="qwen-tp-"), "group")
    try:
        spawn(
            _worker,
            args=(
                ranks,
                path,
                secrets.token_bytes(16),
                cards,
                str(out),
                None if drafter is None else str(drafter),
                mtp,
            ),
            devices=[f"pci:{c}" for c in cards],
        )
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def _decoded_alike(one: Path, two: Path) -> dict:
    """Greedy decoding agrees up to a near tie of one card; both runs compile nothing after the start, and the
    follower ends with the driver's logits. Returns the two-card run's result."""
    reference = json.loads(Path(f"{one}.0.json").read_text())
    result = json.loads(Path(f"{two}.0.json").read_text())
    for ref, got in zip(reference["decoded"], result["decoded"], strict=True):
        assert len(got["tokens"]) == DECODE
        part = next(
            (i for i, (x, y) in enumerate(zip(ref["tokens"], got["tokens"])) if x != y),
            None,
        )
        if part is not None:
            margin = ref["margins"][part]
            assert margin < 1.0, (part, ref["tokens"], got["tokens"], margin)
    assert (reference["compiles"], result["compiles"]) == (0, 0)
    follower = json.loads(Path(f"{two}.1.json").read_text())
    assert follower["calls"] > 0
    assert np.array_equal(np.load(f"{two}.0.npy"), np.load(f"{two}.1.npy"))
    return result


def test_two_cards_match_one_card(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    _run(1, one, None)
    _run(2, two, None)
    alone, split = np.load(f"{one}.logits.npz"), np.load(f"{two}.logits.npz")
    checks = []
    for key in alone.files:
        a, b = alone[key].astype(np.float64), split[key].astype(np.float64)
        first, second = np.sort(a)[-2:][::-1]
        corr = float(np.corrcoef(a, b)[0, 1])
        print(
            f"{key}: corr {corr:.6f}, argmax {int(a.argmax())} {int(b.argmax())}, "
            f"margin {first - second:.3f}"
        )
        checks.append((key, corr, a.argmax() == b.argmax() or first - second < 1.0))
    for key, corr, argmax in checks:
        assert corr > 0.9995, (key, corr)
        assert argmax, key
    _decoded_alike(one, two)


def test_two_cards_decode_with_the_drafter_like_one_card(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    _run(1, one, DRAFTER)
    _run(2, two, DRAFTER)
    result = _decoded_alike(one, two)
    assert result["speculation"]["blocks"] > 0, "no step verified drafts"


def test_two_cards_decode_with_the_mtp_layer_like_one_card(tmp_path):
    """Every rank runs the whole MTP layer from the full residual: the ranks take the same drafts and accept the
    same, and decoding agrees with one card up to its near ties."""
    one, two = tmp_path / "one", tmp_path / "two"
    _run(1, one, None, MTP_DRAFTS)
    _run(2, two, None, MTP_DRAFTS)
    result = _decoded_alike(one, two)
    assert result["speculation"]["blocks"] > 0, "no step verified drafts"
    follower = json.loads(Path(f"{two}.1.json").read_text())
    assert follower["speculation"] == result["speculation"]
