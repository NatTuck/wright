"""A request with ``return_progress`` gets prompt progress from the committed prefill steps."""

import logging

import pytest
from http_packets import decoded_events, ingress
from output_helpers import forward_output

import resin
from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SchedulerConfig,
)
from resin._resin import _HttpSession

PROMPT = list(range(1, 8))  # 7 tokens: prefill chunks of 2 give 4 prefill steps


class Runner:
    def __init__(self, cached: bool = False):
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
            if cached
            else None,
        )

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        pass

    def forward(self, plan):
        rows = len(plan.token_indptr) - 1
        return forward_output(
            [5] * rows if plan.phase == BatchPhase.PREFILL else [[5]] * rows
        )


class Recipe:
    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt(PROMPT)

    def create_parser(self, prompt):
        class Parser:
            def feed(self, ids):
                return [resin.recipe.TextDelta("x") for _ in ids]

        return Parser()


def request(request_id: str, return_progress: bool | None) -> dict:
    body = {
        "id": request_id,
        "model": "test",
        "messages": [],
        "tools": [],
        "max_output_tokens": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "seed": None,
        "logprobs": False,
        "top_logprobs": 0,
        "prompt_logprobs": False,
        "routed_dp_rank": None,
        "stop": [],
        "reasoning_effort": None,
        "extra": {},
    }
    if return_progress is not None:
        body["return_progress"] = return_progress
    return body


def drain(engine: Engine, session: _HttpSession) -> list[tuple[str, dict]]:
    events = []
    for _ in range(32):
        events += decoded_events(
            session.process(engine.step() if engine.has_work() else [])
        )
        if session.active_count == 0:
            break
    return events


def run(*requests: dict) -> list[tuple[str, dict]]:
    engine = Engine(Runner(), scheduler_config=SchedulerConfig(prefill_chunk_size=2))
    session = _HttpSession(engine, Recipe())
    session.admit(ingress(list(requests)))
    events = drain(engine, session)
    session.close()
    engine.close()
    return events


def test_progress_runs_from_admission_to_the_whole_prompt_before_the_first_text():
    events = [
        event for request_id, event in run(request("a", True)) if request_id == "a"
    ]
    kinds = [event["type"] for event in events]
    assert kinds[0] == "started"
    progress = [event["data"] for event in events if event["type"] == "prompt_progress"]
    assert [p["processed"] for p in progress] == [0, 2, 4, 6, 7]
    assert all(p["total"] == len(PROMPT) and p["cache"] == 0 for p in progress)
    times = [p["time_ms"] for p in progress]
    assert times == sorted(times)
    assert kinds.index("text_delta") > max(
        i for i, kind in enumerate(kinds) if kind == "prompt_progress"
    )
    assert kinds[-1] == "finished"


@pytest.mark.parametrize("flag", [None, False])
def test_no_progress_without_the_flag(flag):
    events = run(request("b", flag))
    assert not [event for _, event in events if event["type"] == "prompt_progress"]
    assert events[-1][1]["type"] == "finished"


def test_progress_goes_only_to_the_requests_that_ask(caplog):
    # Logging off: the progress (and the prefill counters) do not depend on the step log.
    caplog.set_level(logging.WARNING)
    events = run(request("asks", True), request("quiet", None))
    by_request = {
        request_id: [
            event
            for rid, event in events
            if rid == request_id and event["type"] == "prompt_progress"
        ]
        for request_id in ("asks", "quiet")
    }
    assert by_request["asks"] and by_request["asks"][-1]["data"]["processed"] == len(
        PROMPT
    )
    assert not by_request["quiet"]


def test_a_cached_prefix_starts_the_progress_at_the_cache():
    engine = Engine(
        Runner(cached=True),
        scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        prefix_cache=True,
    )
    session = _HttpSession(engine, Recipe())
    session.admit(ingress([request("first", True)]))
    drain(engine, session)
    session.admit(ingress([request("second", True)]))
    progress = [
        event["data"]
        for _, event in drain(engine, session)
        if event["type"] == "prompt_progress"
    ]
    session.close()
    engine.close()
    cache = progress[0]["cache"]
    assert cache > 0
    assert progress[0]["processed"] == cache
    assert progress[-1]["processed"] == len(PROMPT)
    processed = [p["processed"] for p in progress]
    assert processed == sorted(processed)
