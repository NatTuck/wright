"""DeepSeek-V4.1's DSpark switch at run time, on CPU: the Engine's ``max_speculative_rows`` changes between steps, so
each decode step leases the write lookahead that the runner expects under its switch (``drafting``), mid-sequence
too; and the runner's ``drafting`` property refuses what its ranks cannot follow.

As in test_deepseek_v41_continuations.py, tokens are positions, and a step returns the next positions, so a correct
result is the next consecutive integers whatever each step returns."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_deepseek_v41_continuations import LOOKAHEAD, SPECULATIVE, Runner

from resin import BatchPhase, Engine, SamplingParams


class Switched(Runner):
    """``Runner`` whose drafts turn off and on: while off, no step returns more than one token a row and none
    expects a lookahead lease (the plan check fails on any other lease)."""

    def __init__(self, extension: int | None = None) -> None:
        super().__init__(SPECULATIVE, extension)
        self.drafting = True
        self.leases: list[
            tuple[bool, int, int]
        ] = []  # (drafting, rows, lookahead) of each decode plan

    def lookahead(self, plan) -> int:
        if not self.drafting:
            return 0
        return super().lookahead(plan)

    def check(self, plan) -> None:
        super().check(plan)
        if plan.phase == BatchPhase.DECODE:
            self.leases.append(
                (self.drafting, plan.num_sequences, self.lookahead(plan))
            )


def switch(engine: Engine, runner: Switched, on: bool) -> None:
    """What ``resin.bench.set_drafting`` does: the Engine's lease rule, then the runner's switch."""
    engine.max_speculative_rows = SPECULATIVE if on else 0
    runner.drafting = on


@pytest.mark.parametrize("rows", [1, 2, 3])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("extension", [None, 2048])
def test_a_switch_between_steps_changes_the_lease_and_keeps_the_output(
    rows, overlap, extension
):
    out = 300
    prompts = [100 + 37 * i for i in range(rows)]
    runner = Switched(extension)
    engine = Engine(
        runner,
        context_length=2048,
        max_running_requests=8,
        enable_overlap=overlap,
    )
    assert engine.max_speculative_rows == SPECULATIVE
    leases = [
        engine.submit(
            list(range(n)), sampling_params=SamplingParams(max_output_tokens=out)
        )
        for n in prompts
    ]
    schedule = iter([False, True, False, True])
    steps = 0
    while engine.has_work():
        engine.step()
        steps += 1
        if steps % 25 == 0:
            on = next(schedule, None)
            if on is not None:
                # drain the pipeline: the next plan is a fresh step under the new rule
                while engine.overlap_pending:
                    engine.step(allow_overlap=False)
                switch(engine, runner, on)
    for lease, n in zip(leases, prompts):
        assert lease.take_result() == list(range(n, n + out))
    engine.close()
    # every decode plan leased what the runner expected under its switch: none while off, the drafts' positions
    # while on for a step of up to SPECULATIVE rows
    assert {drafting for drafting, _, _ in runner.leases} == {False, True}
    for drafting, n, lookahead in runner.leases:
        assert lookahead == (LOOKAHEAD if drafting and n <= SPECULATIVE else 0)


def test_the_lease_rule_changes_only_between_steps():
    runner = Switched()
    engine = Engine(
        runner, context_length=2048, max_running_requests=8, enable_overlap=True
    )
    engine.submit(
        list(range(100)), sampling_params=SamplingParams(max_output_tokens=300)
    )
    while not engine.overlap_pending:
        engine.step()
    with pytest.raises(RuntimeError, match="in-flight"):
        engine.max_speculative_rows = 0
    assert engine.max_speculative_rows == SPECULATIVE
    engine.step(allow_overlap=False)
    assert not engine.overlap_pending
    switch(engine, runner, False)
    assert engine.max_speculative_rows == 0
    while engine.has_work():
        engine.step()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.max_speculative_rows = SPECULATIVE


class Executor:
    def __init__(self) -> None:
        self.drafting = True


def runner(*, draft_tokens=3, groups=None, native_ranks=None, rank=0, nranks=8):
    from resin_models_gaudi.deepseek_v41.runner import DeepSeekV41GaudiRunner

    r = object.__new__(DeepSeekV41GaudiRunner)  # with the fields the switch reads
    r.speculative_tokens = draft_tokens
    r.groups, r.native_ranks = groups, native_ranks
    r.rank, r.nranks = rank, nranks
    r.executor = Executor()
    return r


def test_the_runner_switches_its_executor():
    r = runner(native_ranks=SimpleNamespace())
    assert r.drafting
    r.drafting = False
    assert r.executor.drafting is False and not r.drafting
    r.drafting = True
    assert r.drafting
    one = runner(
        groups=SimpleNamespace(), nranks=1
    )  # a group of one rank has no other rank to tell
    one.drafting = False
    assert not one.drafting


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        (dict(draft_tokens=0), ValueError),  # no DSpark to switch on
        (
            dict(groups=SimpleNamespace()),
            NotImplementedError,
        ),  # Gloo ranks cannot learn the switch
        (
            dict(native_ranks=SimpleNamespace(), rank=3),
            ValueError,
        ),  # a follower takes rank 0's decision
    ],
)
def test_the_runner_refuses_a_switch_its_ranks_cannot_follow(fields, error):
    r = runner(**fields)
    with pytest.raises(error):
        r.drafting = True
    assert r.executor.drafting is True
