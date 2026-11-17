"""A model served by one-card replicas in other processes, beside a local model, behind one
HTTP listener: routing, conversation affinity, a failed replica, shutdown."""

from __future__ import annotations

import json
import re
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch.multiprocessing as mp
from output_helpers import forward_output

import resin
from resin import Engine, RunnerConfig
from resin.server import HttpServer

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="replicas need Linux")


class DigitRecipe:
    """Renders a prompt as three tokens; shows each generated token t as the text t - 10."""

    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([1, 2, 3])

    def create_parser(self, prompt):
        class Parser:
            def feed(self, ids):
                return [resin.recipe.TextDelta("".join(str(t - 10) for t in ids))]

        return Parser()


@dataclass
class TokenRunner:
    """Generates the token `10 + digit` for every sequence; `explode` in a prompt raises."""

    digit: int
    fail_on: int | None = None
    runner_config: RunnerConfig = field(default_factory=RunnerConfig)
    calls: int = 0

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        pass

    def forward(self, plan):
        self.calls += 1
        if self.fail_on is not None and self.calls >= self.fail_on:
            raise RuntimeError(f"injected failure in replica {self.digit}")
        return forward_output([10 + self.digit] * plan.num_sequences)


class SlowGate:
    """A card gate that refuses the first `refusals` turns (each after its whole timeout), then
    gives every turn. It saves its counts in `directory`."""

    def __init__(self, refusals: int, directory: str, index: int) -> None:
        self.refusals = refusals
        self.path = Path(directory, f"gate-{index}.json")
        self.counts = dict(acquired=0, refused=0, released=0)
        self.timeouts: set[float] = set()

    def acquire(self, timeout: float) -> bool:
        self.timeouts.add(timeout)
        if self.counts["refused"] < self.refusals:
            self.counts["refused"] += 1
            time.sleep(timeout)
            self._save()
            return False
        self.counts["acquired"] += 1
        return True

    def release(self) -> None:
        self.counts["released"] += 1
        self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.counts | {"timeouts": sorted(self.timeouts)}))


def _replica(index, path, token, directory, fail_on, refusals=None):
    from resin._resin import _ReplicaServer

    engine = Engine(TokenRunner(index, fail_on), max_running_requests=8)
    gate = None if refusals is None else SlowGate(refusals, directory, index)
    record = {"index": index}
    try:
        server = _ReplicaServer(
            path, token, index, engine, DigitRecipe(), model="rep", timeout=30, gate=gate
        )
        server.serve()
        record["result"] = "closed"
    except BaseException as error:
        record["result"] = f"{type(error).__name__}: {error}"
    Path(directory, f"replica-{index}.json").write_text(json.dumps(record))


def _post(base, model, first, *later, system=True):
    messages = [{"role": "system", "content": "be brief"}] if system else []
    messages.append({"role": "user", "content": first})
    for text in later:
        messages += [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": text},
        ]
    body = {"model": model, "messages": messages, "max_tokens": 2}
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code}: {error.read().decode()}"


def _progress(base, model):
    """The `prompt_progress` objects of a streamed request that asks for them."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "progress"}],
        "max_tokens": 2,
        "stream": True,
        "return_progress": True,
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        lines = response.read().decode().splitlines()
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in lines
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    return [chunk["prompt_progress"] for chunk in chunks if "prompt_progress" in chunk]


def _serve(tmp_path, fail_on=None, refusals=None):
    from resin._resin import _ReplicaRouter

    path, token = str(tmp_path / "replicas"), secrets.token_bytes(16)
    router = _ReplicaRouter(path, token, 2)
    context = mp.start_processes(
        _replica,
        args=(path, token, str(tmp_path), fail_on, refusals),
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    metadata = json.loads(router.accept(60))
    server = HttpServer(
        Engine(TokenRunner(7), max_running_requests=8),
        recipe=DigitRecipe(),
        model="local",
    )
    server.register_replicas(router, model="rep")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def serve():
        try:
            server.serve(port=port)
        except RuntimeError as error:
            Path(tmp_path, "server.error").write_text(str(error))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.2):
                break
        except OSError:
            time.sleep(0.02)
    return server, thread, context, base, metadata


def _metric_rows(base, name):
    with urllib.request.urlopen(f"{base}/metrics", timeout=2) as response:
        text = response.read().decode()
    return [
        (dict(re.findall(r'(\w+)="([^"]*)"', line)), float(line.rsplit(" ", 1)[1]))
        for line in text.splitlines()
        if line.startswith(name + "{")
    ]


def _wait_for(check, seconds=10):
    deadline = time.monotonic() + seconds
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def _stop(server, thread, context, tmp_path):
    server.shutdown()
    thread.join(timeout=30)
    assert not thread.is_alive()
    for child in context.processes:
        child.join(timeout=30)
        assert child.exitcode == 0
    return [
        json.loads((tmp_path / f"replica-{index}.json").read_text())["result"]
        for index in range(2)
    ]


def test_replicas_serve_a_model_beside_a_local_one(tmp_path):
    server, thread, context, base, metadata = _serve(tmp_path)
    try:
        # The replicas report their Engine: this one has no prefix cache.
        assert metadata["supports_prompt_caching"] is False
        assert server.models == ["local", "rep"]
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
            listed = [entry["id"] for entry in json.load(response)["data"]]
        assert listed == ["local", "rep"]
        assert _post(base, "local", "hi") == "77"
        # Prompt progress reaches the client from the local model and through the router.
        for model in ("local", "rep"):
            progress = _progress(base, model)
            assert progress[0] | {"time_ms": 0} == {"total": 3, "cache": 0, "processed": 0, "time_ms": 0}
            assert progress[-1]["processed"] == 3, (model, progress)
        # A tokenize request has events without an Engine step.
        request = urllib.request.Request(
            f"{base}/tokenize",
            data=json.dumps(
                {"model": "rep", "messages": [{"role": "user", "content": "x"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            assert json.load(response)["tokens"] == [1, 2, 3]
        # Each answer names its replica: the later turns of a conversation stay on its replica,
        # with or without a system message, and different conversations use both.
        for system in (True, False):
            homes = {}
            for conversation in range(12):
                first = _post(base, "rep", f"topic {conversation}", system=system)
                assert first in ("00", "11"), first
                homes[conversation] = first
                for turn in range(2):
                    later = [f"more {turn}"] * (turn + 1)
                    answer = _post(
                        base, "rep", f"topic {conversation}", *later, system=system
                    )
                    assert answer == first, (system, conversation, turn)
            assert set(homes.values()) == {"00", "11"}
        # Concurrent requests all finish.
        results = [None] * 16
        threads = [
            threading.Thread(
                target=lambda i=i: results.__setitem__(i, _post(base, "rep", f"c{i}"))
            )
            for i in range(16)
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert all(result in ("00", "11") for result in results), results

        # /metrics holds the local Engine's snapshot and each replica's, by replica index.
        def published():
            rows = [row for row, _ in _metric_rows(base, "resin:num_requests_total")]
            replicas = {
                row["attn_dp_rank"] for row in rows if row["model_name"] == "rep"
            }
            local = any(row["model_name"] == "local" for row in rows)
            return replicas == {"0", "1"} and local

        assert _wait_for(published), "replica metrics did not publish"

        # The router records the time to the first token of the replicas' requests. The
        # /metrics text changes once a second.
        def first_tokens():
            rows = _metric_rows(base, "resin:http_time_to_first_token_seconds_count")
            return {row["model_name"]: value for row, value in rows}

        assert _wait_for(lambda: first_tokens().get("rep", 0) >= 12 * 3 * 2 + 16), (
            first_tokens()
        )
    finally:
        results = _stop(server, thread, context, tmp_path)
    assert results == ["closed", "closed"]


def test_a_gated_replica_steps_only_on_its_turns_and_keeps_reading(tmp_path):
    """A replica on a card that a tensor-parallel model also uses: while its gate gives no turn,
    it reads the router's frames, and it steps once the gate gives it turns."""
    server, thread, context, base, _ = _serve(tmp_path, refusals=25)
    try:
        started = time.monotonic()
        answers = [_post(base, "rep", f"conversation {i}") for i in range(4)]
        elapsed = time.monotonic() - started
        assert all(answer in ("00", "11") for answer in answers), answers
        assert elapsed > 0.4  # one replica at least refused 25 turns of 20 ms
    finally:
        _stop(server, thread, context, tmp_path)
    gates = [
        json.loads(path.read_text()) for path in sorted(tmp_path.glob("gate-*.json"))
    ]
    assert gates and all(gate["acquired"] == gate["released"] for gate in gates)
    assert any(gate["refused"] == 25 and gate["acquired"] for gate in gates)
    assert {timeout for gate in gates for timeout in gate["timeouts"]} == {0.02}


def test_a_failed_replica_ends_its_requests_and_the_other_serves(tmp_path, capfd):
    # Every replica fails at its third step; the first to fail takes its requests down.
    server, thread, context, base, _ = _serve(tmp_path, fail_on=3)

    def replica_rows():
        rows = _metric_rows(base, "resin:num_requests_total")
        return {row["attn_dp_rank"] for row, _ in rows if row["model_name"] == "rep"}

    try:
        # Before any request, both replicas publish their metrics.
        assert _wait_for(lambda: replica_rows() == {"0", "1"})
        answers = [_post(base, "rep", f"conversation {i}") for i in range(8)]
        failed = [answer for answer in answers if answer not in ("00", "11")]
        assert failed, answers
        assert any("injected failure in replica" in answer for answer in failed), failed
        # With both replicas failed, new requests get an error at once, and their metrics
        # leave /metrics.
        assert "no replica" in _post(base, "rep", "late")
        assert _post(base, "local", "still") == "77"
        assert _wait_for(lambda: replica_rows() == set())
    finally:
        server.shutdown()
        thread.join(timeout=30)
        for child in context.processes:
            child.join(timeout=30)
    results = [
        json.loads((tmp_path / f"replica-{index}.json").read_text())["result"]
        for index in range(2)
    ]
    assert all("injected failure" in result for result in results), results
    # The server logged each failure when it happened, and still stops cleanly.
    assert not (tmp_path / "server.error").exists()
    logged = capfd.readouterr().err
    for index in range(2):
        message = f"[resin-http] replica {index}: RuntimeError: injected failure"
        assert message in logged, logged


class TensorRunner:
    """Two tensor-parallel ranks that all-reduce each step: every token is 10 + 2 + 3 = 15."""

    runner_config = RunnerConfig()

    def __init__(self, groups):
        self.groups = groups

    def forward_distributed(self, plan, context, **_):
        import torch
        import torch.distributed as dist

        rows = context.max_num_sequences
        values = torch.full(
            (rows,), 2 + self.groups.attn_tp.local_rank, dtype=torch.int64
        )
        dist.all_reduce(values, group=self.groups.attn_tp.handle)
        if plan is None:
            return None
        return forward_output([10 + int(values[0])] * plan.num_sequences)


def _layout_process(index, directory, path, token, port):
    """Process 0 drives the tensor-parallel model and serves HTTP, 1 follows it, 2 is a
    replica of another model."""
    from datetime import timedelta

    import torch.distributed as dist

    from resin import Parallelism
    from resin.backend.distributed import DistributedEngine, DistributedWorker
    from resin.backend.step_transport import create_step_transport
    from resin.backend.topology import ParallelGroups

    if index == 2:
        _replica(0, path, token, directory, None)
        return
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    groups = ParallelGroups.create(
        Parallelism(world_size=2, attn_tp=2), device_backend="gloo"
    )
    transport = create_step_transport(groups, "socket", timeout=30)
    worker = DistributedWorker(TensorRunner(groups), groups, transport=transport)
    if not groups.is_driver:
        worker.serve()
        return
    from resin._resin import _ReplicaRouter

    router = _ReplicaRouter(path, token, 1)
    engine = DistributedEngine(worker, context_length=2048)
    router.accept(30)
    server = HttpServer(engine, recipe=DigitRecipe(), model="big")
    server.register_replicas(router, model="rep")
    stop = Path(directory, "stop")

    def watch():
        while not stop.exists():
            time.sleep(0.05)
        server.shutdown()

    threading.Thread(target=watch, daemon=True).start()
    server.serve(port=port)
    Path(directory, "driver.json").write_text(json.dumps({"result": "closed"}))


def test_replicas_serve_beside_a_distributed_model(tmp_path):
    path, token = str(tmp_path / "replicas"), secrets.token_bytes(16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = mp.start_processes(
        _layout_process,
        args=(str(tmp_path), path, token, port),
        nprocs=3,
        join=False,
        start_method="spawn",
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
            assert [entry["id"] for entry in json.load(response)["data"]] == [
                "big",
                "rep",
            ]
        results = {}
        threads = [
            threading.Thread(
                target=lambda k=k, model=model: results.__setitem__(
                    (model, k), _post(base, model, f"question {k}")
                )
            )
            for k in range(6)
            for model in ("big", "rep")
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert {key: value for key, value in results.items() if key[0] == "big"} == {
            ("big", k): "55" for k in range(6)
        }
        assert all(results[("rep", k)] == "00" for k in range(6)), results
    finally:
        (tmp_path / "stop").touch()
        for child in context.processes:
            child.join(timeout=30)
    assert [child.exitcode for child in context.processes] == [0, 0, 0]
    assert json.loads((tmp_path / "driver.json").read_text())["result"] == "closed"
    assert json.loads((tmp_path / "replica-0.json").read_text())["result"] == "closed"
