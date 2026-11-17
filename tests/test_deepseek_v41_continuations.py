"""DeepSeek-V4.1's decode continuations under the Engine's overlap, on CPU: a decode step of more than
DSPARK_ROWS rows writes one token a row and leases no write lookahead (RunnerConfig.max_speculative_rows), so the
Engine keeps continuing it until the last token of a block; with RunnerConfig.continuation_extension it continues
across blocks too, leasing the next block in place. Every plan matches its exact active ranges as the Gaudi runner
checks them (PagedCachePlan.from_plan with the step's lookahead).

Tokens are positions (prompt 0 .. n - 1), and a step returns the next positions, so a correct result is the next
consecutive integers whatever each step returns."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest
from output_helpers import forward_output

from resin import BatchPhase, Engine, RunnerConfig, SamplingParams
from resin.backend.plans import PagedCachePlan
from resin_models_gaudi.deepseek_v41.cache import CacheGeometry

LOOKAHEAD = 3  # DSpark's draft tokens
SPECULATIVE = 2  # DSPARK_ROWS
CHUNK = 256
# a WINDOW ring and HISTORY blocks, as the model's
GEOMETRY = CacheGeometry(2, (0, 1), (1, 1), 128, CHUNK)


@dataclass
class Feedback:
    tokens: list[int]
    positions: list[int]


class Completion:
    def __init__(self, output, feedback: Feedback) -> None:
        self.output, self.feedback = output, feedback

    def result(self):
        return self.output


class Runner:
    """Steps of up to ``limit`` rows (all steps when None) may return 1 + lookahead tokens a row, as a verify does;
    wider steps return one."""

    def __init__(self, limit: int | None, extension: int | None = None) -> None:
        self.limit = limit
        self.cache_layout = GEOMETRY.layout(256)
        self.runner_config = RunnerConfig(
            prefill_chunk_size=CHUNK,
            continuation_prefill=True,
            cache_write_lookahead_tokens=LOOKAHEAD,
            cache_layout=self.cache_layout,
            max_speculative_rows=limit,
            continuation_extension=extension,
        )
        self.steps: Counter[str] = Counter()

    def bind_resources(self, binding) -> None:
        pass

    def lookahead(self, plan) -> int:
        return (
            LOOKAHEAD if self.limit is None or plan.num_sequences <= self.limit else 0
        )

    def check(self, plan) -> None:
        PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.lookahead(plan)
        )

    def rows(self, plan, tokens, positions) -> list[list[int]]:
        most = self.lookahead(plan) + 1
        out = []
        for token, position in zip(tokens, positions):
            assert token == position
            count = 1 + position % most
            out.append(list(range(position + 1, position + 1 + count)))
        return out

    def forward(self, plan):
        self.check(plan)
        if plan.phase == BatchPhase.PREFILL:
            return forward_output(
                [
                    [int(plan.token_ids[int(end) - 1]) + 1]
                    for end in plan.token_indptr[1:]
                ]
            )
        self.steps["forward"] += 1
        return forward_output(
            self.rows(plan, plan.token_ids.tolist(), plan.context_lengths.tolist())
        )

    def enqueue(self, plan, *, feedback: Feedback | None = None):
        self.check(plan)
        if feedback is None:
            self.steps["enqueue"] += 1
            tokens, positions = plan.token_ids.tolist(), plan.context_lengths.tolist()
        else:
            self.steps["continuation"] += 1
            tokens, positions = feedback.tokens, feedback.positions
        rows = self.rows(plan, tokens, positions)
        return Completion(
            forward_output(rows),
            Feedback(
                [row[-1] for row in rows],
                [p + len(row) for p, row in zip(positions, rows)],
            ),
        )


@pytest.mark.parametrize("rows", [1, 2, 3, 5])
def test_continuations_follow_the_leased_range_and_keep_the_output(rows):
    out = 300
    prompts = [100 + 37 * i for i in range(rows)]
    steps = {}
    variants = {
        "sync": (None, False, None, False),
        "overlap": (None, True, None, False),
        "rows": (SPECULATIVE, True, None, False),
        "extended": (SPECULATIVE, True, 2048, False),
        "extended, prefix cache": (SPECULATIVE, True, 2048, True),
    }
    for name, (limit, overlap, extension, prefix_cache) in variants.items():
        runner = Runner(limit, extension)
        engine = Engine(
            runner,
            context_length=2048,
            max_running_requests=8,
            prefix_cache=prefix_cache,
            enable_overlap=overlap,
        )
        leases = [
            engine.submit(
                list(range(n)), sampling_params=SamplingParams(max_output_tokens=out)
            )
            for n in prompts
        ]
        while engine.has_work():
            engine.step()
        for lease, n in zip(leases, prompts):
            assert lease.take_result() == list(range(n, n + out))
        engine.close()
        steps[name] = runner.steps
    if rows > SPECULATIVE:
        # one token a row and no lookahead: a fallback only at a block's last token, not within 2 x 4 tokens of it
        assert steps["rows"]["forward"] * 3 < steps["overlap"]["forward"]
        assert steps["rows"]["continuation"] > steps["overlap"]["continuation"]
    else:
        assert steps["rows"] == steps["overlap"]
    # across blocks: one chain from the first decode step to the output budget (a dropped continuation restarts
    # with a fresh enqueue after the previous step finishes)
    assert steps["rows"]["enqueue"] > 1
    for name in ("extended", "extended, prefix cache"):
        assert steps[name]["enqueue"] == 1, (name, steps)
        assert steps[name]["continuation"] > steps["rows"]["continuation"], (name, steps)
