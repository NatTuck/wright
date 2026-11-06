"""Gaudi flags and card selection, without a card or Synapse."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "gaudi"))

from deploy.gaudi_cli import (
    add_gaudi_arguments,
    apply_gaudi_arguments,
    check_listen_address,
    decode_share,
    positive_int,
)
from deploy.models import get_model, validate_gaudi_options
from resin.process import ProcessError, spawn
from resin_models_gaudi.synapse.cards import (
    VISIBLE_MODULES,
    all_cards,
    parse_modules,
    visible_cards,
)


@pytest.fixture
def sysfs(tmp_path: Path) -> Path:
    """Modules are not accel indexes: accel0 is module 2, accel1 is module 0."""
    for accel, module, status in (
        (0, 2, "Operational"),
        (1, 0, "Operational"),
        (2, 1, "Malfunction"),
    ):
        device = tmp_path / "devices" / f"0000:{accel:02x}:00.0"
        device.mkdir(parents=True)
        (device / "module_id").write_text(f"{module}\n")
        (device / "status").write_text(f"{status}\n")
        path = tmp_path / f"accel{accel}"
        path.mkdir()
        (path / "device").symlink_to(device)
    (tmp_path / "accel_controlD0").mkdir()
    return tmp_path


def test_cards_sort_by_module(sysfs: Path) -> None:
    assert [(c.module, c.accel) for c in all_cards(sysfs)] == [(0, 1), (1, 2), (2, 0)]


def test_module_list_selects_by_module_in_listed_order(sysfs: Path) -> None:
    cards = visible_cards({VISIBLE_MODULES: "2,0"}, sysfs)
    assert [card.accel for card in cards] == [0, 1]


@pytest.mark.parametrize(
    "listed", [{}, {VISIBLE_MODULES: ""}, {VISIBLE_MODULES: "all"}]
)
def test_no_list_means_every_operational_card(sysfs: Path, listed: dict) -> None:
    assert [card.module for card in visible_cards(listed, sysfs)] == [0, 2]


def test_a_listed_card_that_is_not_operational_is_left_out(sysfs: Path) -> None:
    assert [c.module for c in visible_cards({VISIBLE_MODULES: "1,0"}, sysfs)] == [0]


def test_an_unknown_module_is_an_error(sysfs: Path) -> None:
    with pytest.raises(ValueError, match="module 9 does not exist"):
        visible_cards({VISIBLE_MODULES: "9"}, sysfs)


@pytest.mark.parametrize("text", ["", "a", "0,", "0,,1", "0,0", "-1"])
def test_bad_module_lists(text: str) -> None:
    with pytest.raises(ValueError):
        parse_modules(text)


def test_flags_set_the_variables_that_ranks_inherit(monkeypatch, tmp_path) -> None:
    for name in (VISIBLE_MODULES, "RESIN_GAUDI_CACHE", "LOG_LEVEL_ALL"):
        # setenv first so teardown restores the variable even if it was unset.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    cache = str(tmp_path / "cache")
    parser = argparse.ArgumentParser()
    add_gaudi_arguments(parser)
    args = parser.parse_args(["--gpu", "0,1", "--cache-dir", cache, "--log-level", "2"])
    monkeypatch.setitem(os.environ, VISIBLE_MODULES, "7")
    apply_gaudi_arguments(args)
    assert os.environ[VISIBLE_MODULES] == "0,1"
    assert os.environ["RESIN_GAUDI_CACHE"] == cache
    assert os.environ["LOG_LEVEL_ALL"] == "2"


def test_no_flag_keeps_the_variable(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    add_gaudi_arguments(parser)
    monkeypatch.setitem(os.environ, VISIBLE_MODULES, "7")
    apply_gaudi_arguments(parser.parse_args([]))
    assert os.environ[VISIBLE_MODULES] == "7"


@pytest.mark.parametrize("text", ["0", "-1"])
def test_positive_int_rejects(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(text)


def test_listen_address_checks() -> None:
    assert "IP address" in (check_listen_address("localhost", 8000) or "")
    assert "65535" in (check_listen_address("127.0.0.1", 70000) or "")
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        assert "cannot listen" in (
            check_listen_address("127.0.0.1", taken.getsockname()[1]) or ""
        )
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    assert check_listen_address("127.0.0.1", port) is None


@pytest.mark.parametrize(
    "model, options, match",
    [
        ("qwen3_5-gaudi2", {"max_batch_size": 64}, "max-batch-size of 32"),
        ("qwen3_5-gaudi2", {"draft_tokens": 9}, "draft-tokens"),
        (
            "qwen3_5-gaudi2",
            {"prefill_chunk_size": 256},
            "prefill-chunk-size 512, 1024, 2048, 4096",
        ),
        (
            "glm5_3_flash-gaudi2",
            {"prefill_chunk_size": 512},
            "prefill-chunk-size 1024, 2048, 4096",
        ),
        ("mistral_large4-gaudi2", {"draft_tokens": 2}, "draft-tokens 0, 3"),
        ("mistral_large4-gaudi2", {"prefill_chunk_size": 1500}, "power of two"),
        ("glm5_3_flash-gaudi2", {"draft_tokens": 4}, "draft-tokens"),
        ("glm5_3_flash-gaudi2", {"max_batch_size": 65}, "max-batch-size of 64"),
    ],
)
def test_gaudi_options_rejected_before_the_card_opens(model, options, match) -> None:
    spec = get_model(model)
    settings = {
        "max_batch_size": spec.max_batch_size,
        "prefill_chunk_size": spec.prefill_chunk_size,
        "draft_tokens": spec.draft_tokens,
    }
    with pytest.raises(ValueError, match=match):
        validate_gaudi_options(spec, **{**settings, **options})


@pytest.mark.parametrize(
    "model, chunk",
    [
        ("qwen3_5-gaudi2", 512),
        ("qwen3_5-gaudi2", 2048),
        ("glm5_3_flash-gaudi2", 1024),
        ("mistral_large4-gaudi2", 1024),
    ],
)
def test_smaller_prefill_chunks_are_accepted(model, chunk) -> None:
    spec = get_model(model)
    validate_gaudi_options(
        spec,
        max_batch_size=spec.max_batch_size,
        prefill_chunk_size=chunk,
        draft_tokens=spec.draft_tokens,
    )


@pytest.mark.parametrize(
    "text, share",
    [("50:90", (0.5, 0.9)), ("80", (0.8, 0.8)), ("0", None), ("0:0", None)],
)
def test_decode_share_parses_percent(text, share) -> None:
    assert decode_share(text) == share


@pytest.mark.parametrize("text", ["90:50", "100", "-1", "50:100", "a", "1:2:3", ""])
def test_decode_share_rejects_bad_values(text) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        decode_share(text)


@pytest.mark.parametrize(
    "model", ["qwen3_5-gaudi2", "mistral_large4-gaudi2", "glm5_3_flash-gaudi2"]
)
def test_gaudi_presets_pass_their_own_checks(model) -> None:
    spec = get_model(model)
    validate_gaudi_options(
        spec,
        max_batch_size=spec.max_batch_size,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=spec.draft_tokens,
    )


def _rank_stops_on_sigterm(rank: int, marker: str) -> None:
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
    Path(marker, f"{rank}.ready").write_text("")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if rank == 0 and stopped:
            Path(marker, "0.stopped").write_text("")
            return
        if rank != 0 and Path(marker, "0.stopped").exists():
            return
        time.sleep(0.01)
    raise SystemExit(3)


def test_a_signal_asks_rank_0_to_stop_the_group(tmp_path: Path) -> None:
    context = spawn(
        _rank_stops_on_sigterm,
        args=(str(tmp_path),),
        devices=["cpu"] * 2,
        numa="off",
        join=False,
    )
    deadline = time.monotonic() + 30
    while len(list(tmp_path.glob("*.ready"))) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM)).start()
    context.join_stopping_on_signals(grace=20.0)
    assert (tmp_path / "0.stopped").exists()


def _rank_fails(rank: int) -> None:
    raise SystemExit(78)


def test_the_exit_code_of_a_reported_failure_reaches_the_parent() -> None:
    context = spawn(_rank_fails, devices=["cpu"], numa="off", join=False)
    with pytest.raises(ProcessError) as failure:
        context.join_stopping_on_signals(grace=1.0)
    assert failure.value.exitcode == 78
