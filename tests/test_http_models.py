"""One HTTP server for two models: each request reaches its model's Engine, and the Engines
step in turn on one thread."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
from output_helpers import forward_output

import resin
from resin import Engine
from resin.recipe import RenderedPrompt
from resin.server import HttpServer, ModelCapabilities


class Recipe:
    tokenizer = None
    stop_token_ids = []

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def render(self, messages, *, tools=(), options=None):
        return RenderedPrompt([1, 2, 3])

    def create_parser(self, prompt):
        reply = self.reply

        class Parser:
            def feed(self, token_ids):
                return [resin.recipe.TextDelta(reply)]

        return Parser()


class Runner:
    def __init__(self) -> None:
        self.steps = 0

    def forward(self, plan):
        self.steps += 1
        return forward_output([4] * plan.num_sequences)


@contextmanager
def two_models():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    runners = [Runner(), Runner()]
    engines = [
        Engine(runners[0], context_length=4096),
        Engine(runners[1], context_length=1024),
    ]
    server = HttpServer(engines[0], recipe=Recipe("from big"), model="big")
    server.register(
        engines[1],
        recipe=Recipe("from small"),
        model="small",
        capabilities=ModelCapabilities(reasoning=False),
    )
    errors: list[BaseException] = []

    def serve():
        try:
            server.serve(port=port)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while True:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.2):
                break
        except (OSError, urllib.error.URLError):
            if errors or time.monotonic() >= deadline:
                server.shutdown()
                raise AssertionError(f"server did not become ready: {errors}")
            time.sleep(0.01)
    try:
        yield server, runners, base
    finally:
        server.shutdown()
        thread.join(timeout=7)
        assert not thread.is_alive()
        assert not errors, errors


def chat(base: str, model: str) -> tuple[int, dict]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_each_request_reaches_its_model() -> None:
    with two_models() as (server, runners, base):
        assert server.models == ["big", "small"]
        for _ in range(3):
            assert chat(base, "big")[1]["choices"][0]["message"]["content"] == (
                "from bigfrom big"
            )
            assert chat(base, "small")[1]["choices"][0]["message"]["content"] == (
                "from smallfrom small"
            )
        assert runners[0].steps > 0 and runners[1].steps > 0
        status, body = chat(base, "other")
        assert status == 404
        assert '"big", "small"' in body["error"]["message"]


def test_the_model_list_has_each_engine_s_limits() -> None:
    with two_models() as (_, _, base):
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
            data = json.load(response)["data"]
        assert [entry["id"] for entry in data] == ["big", "small"]
        assert data[0]["max_context_tokens"] == 4096
        assert data[1]["max_context_tokens"] == 1024
        assert data[1]["supports_reasoning"] is False


def test_concurrent_requests_to_both_models_finish() -> None:
    with two_models() as (_, _, base):
        results: dict[str, list[int]] = {"big": [], "small": []}

        def worker(model: str) -> None:
            for _ in range(5):
                results[model].append(chat(base, model)[0])

        threads = [threading.Thread(target=worker, args=(m,)) for m in results]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert results == {"big": [200] * 5, "small": [200] * 5}


def test_register_checks_its_model() -> None:
    engine = Engine(Runner())
    server = HttpServer(engine, recipe=Recipe("x"), model="big")
    with pytest.raises(ValueError, match="already served"):
        server.register(Engine(Runner()), recipe=Recipe("y"), model="big")
    with pytest.raises(ValueError, match="its own Engine"):
        server.register(engine, recipe=Recipe("y"), model="small")
    with pytest.raises(ValueError, match="share"):
        server.register(Engine(Runner()), recipe=Recipe("y"), model="small", share=0)
