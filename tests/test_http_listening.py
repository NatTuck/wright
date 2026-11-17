"""The server is ready only when it listens: ``on_listening`` and the ready line of ``wright``."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from output_helpers import forward_output

import resin
from resin import Engine
from resin.server import HttpServer

ROOT = Path(__file__).resolve().parents[1]


class Recipe:
    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([1, 2])

    def create_parser(self, prompt):
        class Parser:
            def feed(self, ids):
                return [resin.recipe.TextDelta("ok")]

        return Parser()


class Runner:
    def forward(self, plan):
        return forward_output([3] * plan.num_sequences)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_client_connects_as_soon_as_on_listening_is_called():
    port = free_port()
    server = HttpServer(Engine(Runner()), recipe=Recipe(), model="test")
    seen: dict[str, object] = {}

    def health() -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5
        ) as response:
            seen["status"] = response.status

    def listening() -> None:
        # The listener is bound: a connection now is not refused, and it is served.
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            seen["connected"] = True
        client = threading.Thread(target=health)
        client.start()
        seen["client"] = client  # only a started thread: the test joins it as soon as it sees it

    errors = []

    def serve() -> None:
        try:
            server.serve(port=port, on_listening=listening)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while "client" not in seen and time.monotonic() < deadline and not errors:
        time.sleep(0.01)
    client = seen["client"]
    assert isinstance(client, threading.Thread)
    client.join(timeout=10)
    server.shutdown()
    thread.join(timeout=10)
    assert not errors
    assert seen["connected"] is True and seen["status"] == 200


def test_an_on_listening_error_stops_the_server_and_raises():
    server = HttpServer(Engine(Runner()), recipe=Recipe(), model="test")

    def listening() -> None:
        raise ValueError("refused")

    with pytest.raises(ValueError, match="refused"):
        server.serve(port=free_port(), on_listening=listening)
    assert not server.running


CHILD = textwrap.dedent(
    """
    import os, signal, sys
    sys.path.insert(0, {tests!r})
    from test_http_listening import Recipe, Runner
    from resin import Engine
    from resin.server import HttpServer
    from deploy import serving
    from deploy.logs import configure_logging

    configure_logging(3, "off")

    class Server(HttpServer):
        def serve(self, **options):
            if {early}:
                os.kill(os.getpid(), signal.SIGTERM)  # before the listener: the stop waits for it
            super().serve(**options)

    server = Server(Engine(Runner()), recipe=Recipe(), model="test")
    port = int(sys.argv[1])
    serving._serve_http(server, "127.0.0.1", port, "serving test", f"test is ready at http://127.0.0.1:{{port}}")
    print("served", flush=True)
    """
)


@pytest.mark.skipif(sys.platform != "linux", reason="signals of a child process")
@pytest.mark.parametrize("early", [False, True])
def test_the_ready_line_comes_when_the_server_listens_and_sigterm_stops_it_normally(
    tmp_path, early
):
    port = free_port()
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(tests=str(ROOT / "tests"), early=early))
    child = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ | {"PYTHONPATH": str(ROOT)},
    )
    try:
        assert child.stderr is not None
        lines = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = child.stderr.readline()
            if not line:
                break
            lines.append(line)
            if "is ready at" in line:
                break
        if not early:
            # At the ready line, a client connects and is served.
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                assert response.status == 200
            child.send_signal(15)
        out, err = child.communicate(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
    text = "".join(lines) + err
    assert any("is ready at" in line for line in lines), text
    assert child.returncode == 0, text
    assert "served" in out
    assert "stopped during startup" not in text
