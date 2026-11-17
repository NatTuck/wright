"""Native log lines go to the server's handlers without Python: ``setup_logging`` gives the native sink the
handlers' prefix, colors and log file (the line format is tested in resin-py's logging.rs)."""

from __future__ import annotations

import logging

import pytest

import resin
from deploy import logs


@pytest.fixture(autouse=True)
def _restore_wright_logger():
    logger = logging.getLogger("wright")
    state = logger.handlers[:], logger.propagate, logger.level
    yield
    for handler in logger.handlers:
        if handler not in state[0]:
            handler.close()
    logger.handlers[:], logger.propagate = state[0], state[1]
    logger.setLevel(state[2])


def test_setup_logging_gives_native_code_the_handlers_format(
    monkeypatch, tmp_path
) -> None:
    sinks = []
    monkeypatch.setattr(
        resin,
        "set_log_sink",
        lambda prefix=None, colors=False, path=None: sinks.append(
            (prefix, colors, path)
        ),
    )
    monkeypatch.setattr(resin, "set_log_level", lambda level: None)
    monkeypatch.setenv("WRIGHT_LOG_COLORS", "off")
    monkeypatch.setenv("WRIGHT_LOG_FILE", str(tmp_path / "server.log"))
    logs.setup_logging(rank=3)
    logs.setup_logging(name="Qwen3.5-9B 2")
    monkeypatch.delenv("WRIGHT_LOG_FILE")
    monkeypatch.setenv("WRIGHT_LOG_COLORS", "on")
    logs.setup_logging()
    colors = sinks[-1][1]
    assert sinks == [
        ("rank 3: ", False, str(tmp_path / "server.log")),
        ("Qwen3.5-9B 2: ", False, str(tmp_path / "server.log")),
        ("", colors, None),
    ]
    try:
        import coloredlogs  # noqa: F401
    except ImportError:
        assert not colors  # a plain handler
    else:
        assert colors  # coloredlogs' formatter with WRIGHT_LOG_COLORS=on
