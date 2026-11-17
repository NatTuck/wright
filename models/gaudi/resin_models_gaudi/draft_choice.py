"""Whether a decode step verifies drafts, from each row's recent acceptance and measured step times: for runners on
several cards.

A step verifies drafts when its expected tokens per ms beat a plain step's: each row is expected to commit one token
and its average accepted drafts (``RowAcceptance``), over the verify step's time, against one token a row over the
plain step's time. A row with no verified step yet counts at the average of all rows. Once a bucket decodes plainly,
its rows' averages do not change, so after a run of plain steps a few steps verify drafts again (a probe). A probe
costs the tokens that plain steps would commit in its time, less the tokens it commits: its first step verifies zeros
(a plain step made no drafts), and the others accept the rows' averages. The run before a probe is long enough that
probes cost at most PROBE_SHARE of the run's tokens. So a bucket whose rows are near the break-even point probes
often (a probe costs little there, and the text can change to text that the drafter predicts well), and a bucket far
below it probes seldom. A bucket probes only where drafts could pay: where a verify step that accepts every draft
beats a plain step.

Each rank of a tensor-parallel model decides on its own, and the ranks must decide the same, or one rank verifies
while another decodes plainly and their collectives do not line up. The drafts each row accepted are the same on
every rank. Step times are not: each rank measures its own (``StepTimes``), and the ranks decide with the first
rank's times, which ``share`` sends to every rank at the same steps (SYNC_FIRST, then every SYNC_STEPS decode
steps). Until the first share, and for a kind and bucket not measured yet, the times come from fits (``StepCosts``),
scaled by what the nearest measured bucket of the kind showed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


class RowAcceptance:
    """Each row's drafts accepted a verify step: an average that starts at the average of all rows and weights the
    latest step by BETA (about the last 5 steps), so that one step does not decide a row. ``prior`` starts the
    average of all rows, which weights each row's step by PRIOR_BETA."""

    BETA = 0.2
    PRIOR_BETA = 0.02
    LIMIT = 4096  # rows kept (the oldest goes first): a row that ends without a release is dropped in time

    def __init__(self, prior: float) -> None:
        self.prior = float(prior)
        self.rows: dict[int, float] = {}

    def expected(self, sequence: int) -> float:
        return self.rows.get(sequence, self.prior)

    def record(self, sequence: int, accepted: int) -> None:
        """A verify step of ``sequence`` that verified drafts of its own (not zeros) accepted ``accepted``."""
        old = self.rows.pop(sequence, self.prior)
        self.rows[sequence] = old + self.BETA * (accepted - old)
        self.prior += self.PRIOR_BETA * (accepted - self.prior)
        if len(self.rows) > self.LIMIT:
            del self.rows[next(iter(self.rows))]

    def forget(self, sequences: Sequence[int]) -> None:
        for sequence in sequences:
            self.rows.pop(int(sequence), None)


@dataclass(frozen=True)
class StepCosts:
    """Fitted decode step times (ms) by rows, ``plain(rows)`` and ``verify(rows)``: the start, before steps are
    measured."""

    plain: Callable[[int], float]
    verify: Callable[[int], float]


class StepTimes:
    """Measured decode step times (ms) by kind ("plain" or "verify") and bucket: averages that take the first steps
    equally and then weight the latest step by BETA (about the last 20 steps). A step SPIKE times longer than its
    average is not counted (a host stall), unless SPIKE_RUN such steps come in a row: then the time has changed
    (longer contexts, or other work on the cards)."""

    BETA = 0.05
    SPIKE = 3.0
    SPIKE_RUN = 3

    def __init__(self) -> None:
        self._stats: dict[tuple[str, int], list[float]] = {}  # [steps, average ms, spikes in a row]

    def record(self, kind: str, bucket: int, ms: float) -> None:
        if not ms > 0:
            return
        stats = self._stats.setdefault((kind, bucket), [0, 0.0, 0])
        if stats[0] and ms > self.SPIKE * stats[1]:
            stats[2] += 1
            if stats[2] < self.SPIKE_RUN:
                return
        stats[2] = 0
        stats[0] += 1
        stats[1] += max(self.BETA, 1.0 / stats[0]) * (ms - stats[1])

    def table(self) -> dict[tuple[str, int], float]:
        return {key: stats[1] for key, stats in self._stats.items()}


class DraftChoice:
    """Whether each decode step verifies drafts (see the module docstring). ``max_rows``: steps of more rows decode
    plainly. ``share``: the first rank's table on every rank (None on one card: each step's choice uses the times
    measured so far). ``mode``: auto (by the rates), always (every step of up to max_rows rows verifies drafts) or
    never. ``speculate`` must run once a decode step, on every rank in the same order, after the last step's
    ``verified``."""

    # a bucket that verifies drafts keeps them until plain steps are this many times faster, so that a row near the
    # break-even point switches less on the noise of its average (a switch back costs a step that verifies zeros). A
    # wider margin keeps rows that are a little slower with drafts on drafts.
    MARGIN = 1.1
    PROBE_MAX = 512  # the most plain steps in a bucket before a probe
    PROBE_STEPS = 4  # verify steps of a probe; the first verifies zeros (a plain step made no drafts)
    # the most of a plain run's tokens that its probe costs. The probes do not back off after probes that find
    # drafts slower: text changes
    PROBE_SHARE = 0.05
    SYNC_FIRST = (32, 64, 128)  # decode steps at which the ranks first share the measured times
    SYNC_STEPS = 256  # then every this many decode steps
    MODES = ("auto", "always", "never")

    def __init__(
        self,
        K: int,
        costs: StepCosts,
        *,
        max_rows: int,
        prior: float,
        buckets: Sequence[int],
        share: Callable[[dict], dict] | None = None,
        mode: str = "auto",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"draft policy {mode!r}: one of {', '.join(self.MODES)}")
        self.K, self.costs, self.max_rows, self.mode = K, costs, max_rows, mode
        self.buckets = tuple(sorted(buckets))
        self.acceptance = RowAcceptance(prior)
        self.times = StepTimes()
        self._share = share
        self.shared: dict[tuple[str, int], float] = {}  # the times every rank decides with
        self._steps = 0
        self._drafting: dict[int, bool] = {}  # each bucket's last decision
        self._plain_run: dict[int, int] = {}  # plain decisions in a row, by bucket
        self._probe = 0  # verify steps left in a probe

    def bucket(self, rows: int) -> int:
        return next((b for b in self.buckets if b >= rows), self.buckets[-1])

    def ms(self, kind: str, rows: int) -> float:
        """A step's time: the shared measurement of its bucket, else the fit scaled by the nearest measured bucket
        of the kind (the fit when the kind has none)."""
        table = self.times.table() if self._share is None else self.shared
        b = self.bucket(rows)
        if (measured := table.get((kind, b))) is not None:
            return measured
        fit = self.costs.verify if kind == "verify" else self.costs.plain
        near = [k[1] for k in table if k[0] == kind]
        if not near:
            return fit(rows)
        n = min(near, key=lambda k: (abs(k.bit_length() - b.bit_length()), k))
        return fit(rows) * table[(kind, n)] / fit(n)

    def rates(self, sequences: Sequence[int]) -> tuple[float, float]:
        """The expected tokens per ms of a verify step and of a plain step of these rows."""
        rows = len(sequences)
        tokens = sum(1.0 + min(self.acceptance.expected(int(s)), self.K) for s in sequences)
        return tokens / self.ms("verify", rows), rows / self.ms("plain", rows)

    def speculate(self, sequences: Sequence[int]) -> bool:
        rows = len(sequences)
        if not self.K or self.mode != "auto":
            return bool(self.K) and self.mode == "always" and 0 < rows <= self.max_rows
        self._steps += 1
        if self._share is not None and (
            self._steps in self.SYNC_FIRST or self._steps % self.SYNC_STEPS == 0
        ):
            self.shared = self._share(self.times.table())
        if not rows or rows > self.max_rows:
            return False
        if self._probe:
            self._probe -= 1
            return True
        b = self.bucket(rows)
        spec, plain = self.rates(sequences)
        drafting = self._drafting.get(b, True)
        decide = spec * self.MARGIN >= plain if drafting else spec > plain
        self._drafting[b] = decide
        if decide:
            self._plain_run[b] = 0
            return True
        run = self._plain_run.get(b, 0) + 1
        if run >= self.probe_after(sequences):
            self._plain_run[b] = 0
            self._probe = self.PROBE_STEPS - 1
            return True
        self._plain_run[b] = run
        return False

    def probe_after(self, sequences: Sequence[int]) -> float:
        """The plain steps of these rows before a probe (inf: drafts cannot pay, even when the rows accept every
        draft). A verify step takes ``ratio`` plain steps' time: a probe's first step costs ratio - 1 tokens a row
        (it verifies zeros), and each other step ratio - 1 - the rows' average accepted drafts, at least 0."""
        rows = len(sequences)
        ratio = self.ms("verify", rows) / self.ms("plain", rows)
        if 1 + self.K <= ratio:
            return math.inf
        accepted = sum(min(self.acceptance.expected(int(s)), self.K) for s in sequences) / rows
        cost = ratio - 1 + (self.PROBE_STEPS - 1) * max(0.0, ratio - 1 - accepted)
        return min(math.ceil(cost / self.PROBE_SHARE), self.PROBE_MAX)

    def measured(self, kind: str, rows: int, ms: float) -> None:
        """A decode step of ``rows`` rows took ``ms`` (on this rank)."""
        self.times.record(kind, self.bucket(rows), ms)

    def verified(
        self, sequences: Sequence[int], accepted: np.ndarray, own: np.ndarray | None = None
    ) -> None:
        """A finished verify step: each row's accepted drafts. ``own``: the rows that verified drafts of their own
        (None: all); a row that verified zeros (its drafts did not carry over) says nothing of its acceptance."""
        for r, sequence in enumerate(sequences):
            if own is None or own[r]:
                self.acceptance.record(int(sequence), int(accepted[r]))

    def forget(self, sequences: Sequence[int]) -> None:
        self.acceptance.forget(sequences)
