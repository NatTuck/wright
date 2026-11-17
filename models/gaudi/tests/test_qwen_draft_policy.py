"""Qwen3.8-27B's draft policy, on CPU: when a decode step verifies drafts, from the measured tokens per ms of verify
steps and of plain steps. Each test runs on both executors' policies: the Python executor's
(``qwen3_5.runner.DraftPolicy``) and the native executor's (``resin._resin.QwenDraftPolicy``)."""

from __future__ import annotations

import pytest

pytest.importorskip("resin_models_gaudi.qwen3_5.runner")


@pytest.fixture(params=["python", "native"])
def policy(request):
    """A factory of draft policies: policy(K=2, drafter=False, mode="auto")."""
    if request.param == "python":
        from resin_models_gaudi.qwen3_5.runner import DraftPolicy as make
    else:
        native = pytest.importorskip("resin._resin")
        if not hasattr(native, "QwenDraftPolicy"):
            pytest.skip("resin built without the gaudi feature")
        make = native.QwenDraftPolicy

    def build(K=2, drafter=False, mode="auto"):
        return make(K, drafter, mode)

    return build


def measure(p, b, plain_ms, verify_ms, tokens_per_row, n=20):
    for _ in range(n):
        p.record(("plain", b), b, plain_ms)
        p.record(("verify", b), b * tokens_per_row, verify_ms)


def test_modes_and_limits(policy):
    from resin_models_gaudi.qwen3_5.model import SPEC_ROWS

    assert not policy(K=0).speculate(1, (1,))
    assert not policy(mode="never").speculate(1, (3,))
    assert policy(mode="always").speculate(SPEC_ROWS, (3,))
    assert not policy(mode="always").speculate(SPEC_ROWS + 1, (3,))
    assert not policy().speculate(SPEC_ROWS + 1, (3,))
    with pytest.raises(ValueError):
        policy(mode="sometimes")


def test_measured_rates_decide(policy):
    """MTP-2 at 1 row (21.6 ms verify, 2.1 tokens per row-step, 17 ms plain) keeps its drafts;
    DFlash 2 at 16 rows (103 ms verify, 2.5 tokens, 33 ms plain) decodes plainly."""
    p = policy()
    measure(p, 1, 17.0, 21.6, 2.1)
    assert p.speculate(1, (3,))
    d = policy(K=7, drafter=True)
    measure(d, 16, 33.0, 103.0, 2.5)
    assert not d.speculate(16, (8,))


def test_drafts_stay_near_break_even(policy):
    """A bucket that verifies drafts keeps them until plain steps are MARGIN times faster, so a
    measurement near break-even does not switch it."""
    p = policy()
    measure(p, 1, 17.0, 28.0, 1.6)  # verify 0.057 tokens/ms, plain 0.059: 3% slower
    assert p.speculate(1, (3,))
    measure(p, 1, 17.0, 40.0, 1.6, n=40)  # verify 0.040: plain 1.47 times faster
    assert not p.speculate(1, (3,))


def test_a_drafter_is_judged_at_its_best_width(policy):
    """DSpine at 8 rows before any measurement: its full 16-column verify loses to plain
    decoding by the fits, its 4-column one wins."""
    p = policy(K=15, drafter=True)
    assert p.fit(8, 16) < p.rate(("plain", 8)) < p.fit(8, 4)
    assert p.speculate(8, (4, 8, 16))


def test_a_measured_verify_rate_overrides_the_width_fits(policy):
    """Once a bucket's verify steps are measured (at whatever widths the drafter chose), the fits
    of the drafter's widths no longer count: DSpine at 8 rows whose steps commit 1.3 tokens a row
    in 60 ms decodes plainly although its 4-column fit wins."""
    p = policy(K=15, drafter=True)
    assert p.speculate(8, (4, 8, 16))
    measure(p, 8, 24.0, 60.0, 1.3)
    assert not p.speculate(8, (4, 8, 16))


def test_probes_back_off(policy):
    """Drafts that rarely pass at 8 rows: each probe (whose steps are measured) keeps plain decoding
    and waits twice as long before the next."""
    p = policy()
    measure(p, 8, 24.0, 80.0, 1.2)  # plain decoding wins clearly
    decisions = []
    for _ in range(600):
        drafting = p.speculate(8, (3,))
        decisions.append(drafting)
        if drafting:
            p.record(("verify", 8), 8 * 1.2, 80.0)
        else:
            p.record(("plain", 8), 8, 24.0)
    starts = [i for i, d in enumerate(decisions) if d and (i == 0 or not decisions[i - 1])]
    assert len(starts) >= 3
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert gaps == sorted(gaps) and gaps[-1] > gaps[0]  # each probe that keeps plain waits longer
    assert sum(decisions) / len(decisions) < 0.15


def test_a_streak_of_one_token_steps_keeps_drafts_that_pay(policy):
    """DSpine on code at 1 row: verify steps of 24 ms commit 1.8 tokens on
    average, with runs of one-token steps; plain steps take 16.6 ms. Drafting pays (0.075 against
    0.060 tokens per ms) through every run."""
    p = policy(K=15, drafter=True)
    tokens = [3, 2, 1, 2, 2, 1, 1, 2, 1, 2, 4, 1, 3, 1, 1, 1, 4, 2, 1, 2] * 6
    tokens += [1] * 9 + [4, 3, 2]  # a run of nine one-token steps, then recovery
    for n in tokens:
        assert p.speculate(1, (4, 8, 16))
        p.record(("verify", 1), n, 24.0)
    assert p.speculate(1, (4, 8, 16))


def test_a_probe_measures_drafts_afresh(policy):
    """Drafts that stopped paying for a while (one token a step) make the bucket decode plainly; a
    probe whose drafts pay again (2 tokens a step: 0.083 against 0.060 tokens per ms) brings
    drafting back, though the earlier average was below plain."""
    p = policy(K=15, drafter=True)
    for _ in range(40):
        p.record(("verify", 1), 1, 24.0)
        p.record(("plain", 1), 1, 16.6)
    assert not p.speculate(1, (4, 8, 16))
    probe = []
    for _ in range(2 * p.PROBE_FIRST):
        drafting = p.speculate(1, (4, 8, 16))
        probe.append(drafting)
        if drafting:
            p.record(("verify", 1), 2.0, 24.0)
        else:
            p.record(("plain", 1), 1, 16.6)
        if probe[-3:] == [True, True, True] and len(probe) > p.PROBE_FIRST + p.PROBE_STEPS:
            break
    assert any(probe) and probe[-1], "the probe's measurement brought drafting back"


def test_spikes_and_averages(policy):
    p = policy()
    p.record(("plain", 8), 8, 25.0)
    p.record(("plain", 8), 8, 300.0)  # a host stall
    assert p.rate(("plain", 8)) == pytest.approx(8 / 25.0)
    p.record(("plain", 8), 8, 35.0)
    assert 8 / 35.0 < p.rate(("plain", 8)) < 8 / 25.0
    p.record(("verify", 8), 8, 25.0)
    p.record(("verify", 8), 1, 25.0)  # a step that commits little is counted, not a stall
    assert p.rate(("verify", 8)) == pytest.approx(4.5 / 25.0)
