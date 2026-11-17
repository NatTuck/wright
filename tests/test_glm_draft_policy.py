"""The native draft policy (``resin._resin.GlmDraftPolicy``, ``resin_gaudi::draft_choice``) against
``draft_choice.DraftChoice``: the same random traces of decode steps (rows joining and leaving, acceptance, step
times with spikes, shared tables) give the same decision at every step, in every mode, with and without sharing.
And the policy as the executor builds it: a native rank group's driver decides with the step times it measures, and
ranks without one with the first rank's."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from resin_models_gaudi.draft_choice import DraftChoice
from resin_models_gaudi.glm5_3_flash import native
from resin_models_gaudi.glm5_3_flash.model import DECODE_BUCKETS
from resin_models_gaudi.glm5_3_flash.runner import step_cost_terms, step_costs

_resin = pytest.importorskip("resin._resin", reason="the native extension")
pytestmark = pytest.mark.skipif(
    not hasattr(_resin, "GlmDraftPolicy"), reason="the native extension was built without the gaudi feature"
)


def pair(K: int, nranks: int, drafter: bool, group: int, mode: str, shared: bool):
    max_rows = 16 // (K + 1)
    share = (lambda table: dict(table)) if shared else None
    python = DraftChoice(
        K, step_costs(nranks, K, drafter, group), max_rows=max_rows, prior=0.4 * K, buckets=DECODE_BUCKETS,
        share=share, mode=mode,
    )  # fmt: skip
    native = _resin.GlmDraftPolicy(
        K, step_cost_terms(nranks, K, drafter, group), max_rows=max_rows, prior=0.4 * K,
        buckets=list(DECODE_BUCKETS), shared=shared, mode=mode,
    )  # fmt: skip
    return python, native


def step(python, native, rows: list[int]) -> tuple[bool, bool]:
    """One decode step's decision on both, the native side sharing its own table at its sync steps (as the first
    rank does)."""
    if native.sync_due():
        native.set_shared(native.table())
    return python.speculate(rows), native.speculate(rows)


@pytest.mark.parametrize("mode", ["auto", "always", "never"])
@pytest.mark.parametrize(("K", "nranks", "drafter"), [(1, 4, False), (3, 4, False), (3, 8, False), (7, 4, True)])
@pytest.mark.parametrize("shared", [False, True])
def test_the_native_policy_decides_as_python(mode: str, K: int, nranks: int, drafter: bool, shared: bool) -> None:
    rng = np.random.default_rng(K * 100 + nranks + shared)
    group = 2 if drafter else 16 // (K + 1)
    python, native = pair(K, nranks, drafter, group, mode, shared)
    live = list(range(4))
    next_id = 4
    mismatches = []
    for i in range(3000):
        if rng.random() < 0.02 and len(live) < 24:  # a row joins
            live.append(next_id)
            next_id += 1
        if rng.random() < 0.02 and len(live) > 1:  # a row leaves
            gone = live.pop(int(rng.integers(len(live))))
            python.forget([gone])
            native.forget([gone])
        rows = live[: int(rng.integers(1, len(live) + 1))]
        a, b = step(python, native, rows)
        if a != b:
            mismatches.append(i)
            continue
        kind = "verify" if a else "plain"
        ms = float(rng.uniform(5, 40)) * (1 + len(rows) / 8)
        if rng.random() < 0.03:
            ms *= 5  # a host stall
        python.measured(kind, len(rows), ms)
        native.measured(kind, len(rows), ms)
        if a:
            # a phase where drafts pay, then one where they do not
            good = (i // 600) % 2 == 0
            accepted = rng.integers(0, K + 1 if good else 1, len(rows))
            own = rng.random(len(rows)) < 0.8
            python.verified(rows, accepted, own)
            native.verified(rows, accepted.tolist(), own.tolist())
    assert not mismatches, f"decisions differ at steps {mismatches[:10]}"
    for s in live:
        assert python.acceptance.expected(s) == pytest.approx(native.expected(s), rel=1e-12)


def policy_as_built(runner):
    """The policy of ``runner``'s executor, as ``native.build_executor`` and ``GlmExecutor`` make it: it shares
    the first rank's times when the executor has a share (``native._share``), and decides with its own otherwise.
    After 200 decode steps of one row each, verify steps measured at 60 ms (three times the fit) and plain steps
    at 10 ms."""
    share = native._share(runner)
    policy = _resin.GlmDraftPolicy(
        runner.K, step_cost_terms(runner.nranks, runner.K, False, 4), max_rows=4, prior=0.4 * runner.K,
        buckets=list(DECODE_BUCKETS), shared=share is not None, mode="auto",
    )  # fmt: skip
    for _ in range(200):
        policy.measured("verify", 1, 60.0)
        policy.measured("plain", 1, 10.0)
    return share, policy


def test_a_native_rank_group_driver_decides_with_its_measured_times() -> None:
    """With a native rank group the driver decides for every rank (the followers take its decision with the step):
    its executor gets no share, and its policy decides with the step times it measures. A row at the prior (1.2
    drafts) then decodes plainly: 2.2 tokens in 60 ms lose to 1 token in 10 ms (the fits, 19.1 ms and 10.33 ms,
    would verify)."""
    group = SimpleNamespace(rank=0, size=4)
    share, policy = policy_as_built(SimpleNamespace(nranks=4, K=3, groups=None, native_ranks=group))
    assert share is None
    assert not policy.sync_due()
    assert (policy.ms("verify", 1), policy.ms("plain", 1)) == (60.0, 10.0)
    assert not policy.speculate([7])


def test_without_a_native_rank_group_the_ranks_decide_with_the_first_ranks_times() -> None:
    """Without a native rank group every rank decides: until the first sync step all of them use the fits, and from
    it the first rank's times, which the share gives every rank."""
    share, policy = policy_as_built(SimpleNamespace(nranks=4, K=3, groups=None, native_ranks=None))
    assert share is not None
    assert policy.ms("verify", 1) == pytest.approx(19.1)
    assert policy.ms("plain", 1) == pytest.approx(10.33)
    decisions = []
    while not policy.sync_due():
        decisions.append(policy.speculate([7]))
    assert len(decisions) == 31 and all(decisions)
    policy.set_shared(share(policy.table()))
    assert (policy.ms("verify", 1), policy.ms("plain", 1)) == (60.0, 10.0)
    assert not policy.speculate([7])
