"""DraftChoice: whether a decode step of a multi-card runner verifies drafts, from each row's recent acceptance and
measured step times, the same on every rank."""

from __future__ import annotations

import math

import numpy as np
import pytest

from resin_models_gaudi.draft_choice import (
    DraftChoice,
    RowAcceptance,
    StepCosts,
    StepTimes,
)

BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def glm_like(group: int = 2) -> StepCosts:
    """GLM-5.3-Flash on 4 cards with 7 DFlash 2 drafts: plain 10 ms, a verify launch of up to 2 rows 24.6 ms."""
    return StepCosts(
        lambda rows: 10.0 + 0.25 * (rows - 1),
        lambda rows: math.ceil(rows / group) * (24.6 + 3.0 * (min(rows, group) - 1)),
    )


def choice(mode: str = "auto", max_rows: int = 64, prior: float = 2.8, share=None) -> DraftChoice:
    return DraftChoice(
        7, glm_like(), max_rows=max_rows, prior=prior, buckets=BUCKETS, share=share, mode=mode
    )


def run(c: DraftChoice, rows: list[int], accepted: dict[int, int], steps: int) -> list[bool]:
    """``steps`` decode steps of ``rows``; each verify step accepts ``accepted[row]`` drafts of each row's own."""
    decisions = []
    for _ in range(steps):
        speculate = c.speculate(rows)
        decisions.append(speculate)
        if speculate:
            c.verified(rows, np.array([accepted[r] for r in rows]))
    return decisions


def test_a_row_that_accepts_many_drafts_verifies_and_one_that_accepts_few_decodes_plainly():
    # plain: 0.1 tokens/ms; a verify step pays above 1.46 accepted drafts
    code = run(choice(), [1], {1: 3}, 50)
    assert all(code)
    # the first steps verify (the row starts at the average of all rows) until the row's average falls; then only
    # the probes verify
    story = run(choice(), [2], {2: 1}, 200)
    assert story[0] and sum(story[40:]) <= 0.15 * len(story[40:])


def test_a_new_row_counts_at_the_average_of_all_rows():
    c = choice(prior=2.8)
    assert c.acceptance.expected(9) == 2.8
    run(c, [1], {1: 0}, 200)
    assert c.acceptance.expected(9) < 2.8
    prior = c.acceptance.prior
    c.acceptance.record(5, 0)  # one step moves a new row from the average of all rows, by BETA
    assert c.acceptance.expected(5) == pytest.approx(prior * (1 - RowAcceptance.BETA))


def test_rows_that_verified_zeros_say_nothing_of_their_acceptance():
    c = choice()
    c.verified([1, 2], np.array([0, 4]), own=np.array([False, True]))
    assert 1 not in c.acceptance.rows and 2 in c.acceptance.rows


def probe_gaps(decisions: list[bool], steps: int) -> list[int]:
    """Plain steps between the probes (runs of ``steps`` verify steps) after the first plain step"""
    first_plain = decisions.index(False)
    starts = [i for i in range(first_plain + 1, len(decisions)) if decisions[i] and not decisions[i - 1]]
    for start in starts:
        assert decisions[start : start + steps] == [True] * steps
    return [b - a - steps for a, b in zip(starts, starts[1:], strict=False)]


def test_probes_come_often_near_the_break_even_point_and_seldom_far_below_it():
    # a verify step takes 2.46 plain steps' time: drafts pay above 1.46 accepted drafts a step. A probe costs
    # 1.46 tokens for its first step (it verifies zeros) and 1.46 - accepted for each other step.
    far = probe_gaps(run(choice(), [3], {3: 0}, 1500), DraftChoice.PROBE_STEPS)
    near = probe_gaps(run(choice(), [3], {3: 1}, 1500), DraftChoice.PROBE_STEPS)
    ratio, share = 24.6 / 10.0, DraftChoice.PROBE_SHARE
    # 117 and 57 plain steps (1 less while the row's average has not reached its acceptance)
    assert math.ceil((ratio - 1 + 3 * (ratio - 1)) / share) - far[-1] in (0, 1)
    assert math.ceil((ratio - 1 + 3 * (ratio - 2)) / share) - near[-1] in (0, 1)


def test_a_row_gets_drafts_back_soon_after_a_long_plain_run():
    """The probes do not back off: after a long run of text that the drafter predicts badly, text that it predicts
    well (code after prose) gets drafts within one probe interval."""
    for before in (1000, 1500, 2000):
        c = choice()
        acceptance = {21: 0}
        run(c, [21], acceptance, before)
        acceptance[21] = 3
        interval = c.probe_after([21])
        decisions = run(c, [21], acceptance, 300)
        assert decisions.index(True) <= interval and sum(decisions) >= 0.6 * len(decisions)


def test_a_probe_that_finds_drafts_paying_again_keeps_them():
    c = choice()
    acceptance = {4: 0}
    decisions = run(c, [4], acceptance, 60)
    assert not decisions[-1]
    acceptance[4] = 6  # the row reaches a part that the drafter predicts well
    decisions = run(c, [4], acceptance, 200)
    assert all(decisions[-50:])


def test_a_bucket_that_verifies_keeps_drafts_within_the_margin():
    c = choice()
    # 1.46 drafts is the break-even; 1.3 is within the 10% margin of a bucket that verifies
    c.acceptance.rows[6] = 1.3
    spec, plain = c.rates([6])
    assert spec < plain <= spec * c.MARGIN
    assert c.speculate([6])  # drafting by default: keeps drafts
    c._drafting[1] = False
    assert not c.speculate([6])  # plain: needs spec > plain to start


def test_probes_cost_a_small_share_of_the_tokens_and_come_only_where_drafts_could_pay():
    c = choice()
    decisions = run(c, [3], {3: 0}, 2000)
    verify_ms, plain_ms = c.ms("verify", 1), c.ms("plain", 1)
    spent = sum(verify_ms if d else plain_ms for d in decisions[40:])
    # the tokens that plain steps commit in the probes' time, less the probes' tokens (one a step: none accepted)
    lost = sum(verify_ms / plain_ms - 1 for d in decisions[40:] if d)
    assert 0 < lost <= c.PROBE_SHARE * spent / plain_ms * 1.05
    # 16 rows: a verify step of 8 launches loses even with every draft accepted, so no probe
    rows = list(range(10, 26))
    decisions = run(choice(), rows, dict.fromkeys(rows, 0), 2000)
    assert not any(decisions[20:])


def test_large_batches_decode_plainly_when_verify_launches_run_one_after_another():
    c = choice()
    rows = list(range(10, 26))  # 16 rows: 8 verify launches of 2 rows
    decisions = run(c, rows, dict.fromkeys(rows, 5), 20)
    assert not any(decisions[:20])


class Ranks:
    """Two ranks of one model: share() gives every rank the first rank's table, as ranks.from_first_rank does."""

    def __init__(self) -> None:
        self.tables: list[dict] = []
        self.ranks = [choice(share=self.share), choice(share=self.share)]

    def share(self, table: dict) -> dict:
        self.tables.append(table)
        return self.tables[len(self.tables) - 1 - (len(self.tables) - 1) % 2]


def test_ranks_that_measure_different_times_decide_the_same():
    rng = np.random.default_rng(7)
    group = Ranks()
    first, second = group.ranks
    decisions_seen = set()
    for step in range(1500):
        rows = sorted(int(row) for row in rng.choice(40, size=int(rng.integers(1, 5)), replace=False))
        decisions = [c.speculate(rows) for c in group.ranks]
        assert decisions[0] == decisions[1], step
        decisions_seen.add(decisions[0])
        kind = "verify" if decisions[0] else "plain"
        # the second rank's clock says verify steps take much longer: only the first rank's times count
        first.measured(kind, len(rows), 20.0 if decisions[0] else 10.0)
        second.measured(kind, len(rows), 90.0 if decisions[0] else 10.0)
        if decisions[0]:
            accepted = rng.integers(0, 8, size=len(rows))
            own = rng.random(len(rows)) < 0.8
            for c in group.ranks:
                c.verified(rows, accepted, own)
        if rng.random() < 0.05:
            for c in group.ranks:
                c.forget(rows[:1])
    assert decisions_seen == {True, False}
    assert first.shared == second.shared and first.shared[("verify", 1)] == pytest.approx(20.0)


def test_measured_times_replace_the_fits():
    c = choice()
    assert c.ms("verify", 1) == pytest.approx(24.6)
    c.measured("verify", 1, 12.0)  # verify steps turn out faster than the fit
    assert c.ms("verify", 1) == 12.0
    # a row of 1 accepted draft: 2 tokens in 12 ms beat 1 token in 10 ms
    c.acceptance.rows[8] = 1.0
    spec, plain = c.rates([8])
    assert spec > plain and c.speculate([8])


def test_a_bucket_not_measured_yet_scales_the_fit_by_the_nearest_measured_bucket():
    c = choice()
    c.measured("plain", 8, 2 * glm_like().plain(8))  # plain steps take twice the fit
    assert c.ms("plain", 8) == pytest.approx(2 * glm_like().plain(8))
    assert c.ms("plain", 16) == pytest.approx(2 * glm_like().plain(16))
    assert c.ms("verify", 16) == pytest.approx(glm_like().verify(16))  # no verify step measured


def test_the_ranks_decide_with_the_fits_until_they_first_share_their_times():
    group = Ranks()
    first, second = group.ranks
    first.measured("verify", 1, 12.0)
    second.measured("verify", 1, 12.0)
    assert first.ms("verify", 1) == pytest.approx(24.6)  # not shared yet
    for _ in range(DraftChoice.SYNC_FIRST[0]):
        for c in group.ranks:
            c.speculate([1])
    assert first.ms("verify", 1) == second.ms("verify", 1) == 12.0


def test_modes_and_row_limits():
    assert run(choice("always"), [1], {1: 0}, 40) == [True] * 40
    assert run(choice("never"), [1], {1: 7}, 5) == [False] * 5
    assert not choice(max_rows=4).speculate([1, 2, 3, 4, 5])
    assert not DraftChoice(0, glm_like(), max_rows=8, prior=0, buckets=BUCKETS).speculate([1])
    with pytest.raises(ValueError, match="draft policy"):
        choice("sometimes")


def test_released_rows_are_forgotten_and_old_rows_drop_out():
    c = choice()
    c.verified([1, 2], np.array([5, 5]))
    c.forget([1])
    assert 1 not in c.acceptance.rows and 2 in c.acceptance.rows
    acceptance = RowAcceptance(1.0)
    for sequence in range(RowAcceptance.LIMIT + 10):
        acceptance.record(sequence, 1)
    assert len(acceptance.rows) == RowAcceptance.LIMIT and 0 not in acceptance.rows


def test_a_lasting_rise_in_step_time_is_measured_and_one_stall_is_not():
    times = StepTimes()
    for _ in range(30):
        times.record("plain", 1, 10.0)
    times.record("plain", 1, 100.0)  # one host stall
    assert times.table()[("plain", 1)] == pytest.approx(10.0)
    for _ in range(StepTimes.SPIKE_RUN + 30):  # the steps take longer from now on
        times.record("plain", 1, 40.0)
    assert times.table()[("plain", 1)] > 30.0
