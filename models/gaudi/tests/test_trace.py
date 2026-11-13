"""Synapse profiler events become Chrome trace events (host only)."""

from __future__ import annotations

import gzip
import json
import os
from types import SimpleNamespace

import pytest

from resin.bench import BenchCase
from resin_models_gaudi.synapse.trace import (
    chrome_events,
    enable_profiler,
    perfetto_trace,
)

ROWS = [
    ("M", 3, 0, 0, 0, "process_name", "__metadata", "TPC"),
    ("M", 3, 2, 0, 0, "thread_name", "__metadata", "TPC2"),
    ("B", 3, 2, 10.5, 0, "gdn_decode", "kernel", ""),
    ("E", 3, 2, 14.0, 0, "gdn_decode", "kernel", ""),
    ("X", 5, 1, 11.0, 2.5, "dma", "", ""),
    ("i", 0, 0, 9.0, 0, "launch", "api", ""),
    ("f", 0, 0, 9.0, 0, "flow", "", ""),
]


def write_tsv(path):
    lines = ["ph\tengine_type\tengine_index\tts\tdur\tname\tcategory\tmeta"]
    lines += ["\t".join(str(value) for value in row) for row in ROWS]
    path.write_text("\n".join(lines) + "\n")


def test_events_keep_engines_as_processes_and_threads(tmp_path):
    tsv = tmp_path / "trace.tsv"
    write_tsv(tsv)
    events = chrome_events(tsv)
    assert [e["ph"] for e in events] == ["M", "M", "B", "E", "X", "i"]
    assert events[0] == {
        "ph": "M",
        "pid": 3,
        "tid": 0,
        "ts": 0.0,
        "name": "process_name",
        "cat": "__metadata",
        "args": {"name": "TPC"},
    }
    assert events[2]["pid"] == 3 and events[2]["tid"] == 2 and events[2]["ts"] == 10.5
    assert events[4]["dur"] == 2.5 and "cat" not in events[4]
    assert events[5]["s"] == "t"


def test_trace_writes_named_cases_and_requires_device_activity(tmp_path, monkeypatch):
    calls = []

    def stop(path):
        write_tsv(path)
        calls.append("stop")
        return len(ROWS)

    device = SimpleNamespace(
        sync=lambda: calls.append("sync"),
        prof_start=lambda: calls.append("start"),
        prof_stop=stop,
    )
    trace = perfetto_trace(tmp_path, 0, {"model": "m"}, device=device, decode_passes=3)
    assert trace.decode_passes == 3
    case = BenchCase(phase="decode", context_length=8192, batch_size=4)
    trace(case, lambda: calls.append("pass"))
    assert calls == ["sync", "start", "pass", "pass", "pass", "stop"]
    with gzip.open(tmp_path / "decode-ctx8192-b4-rank000.json.gz", "rt") as source:
        data = json.load(source)
    assert data["resin_rank"] == 0 and data["resin_benchmark"]["trace_passes"] == 3
    assert data["resin_benchmark"]["model"] == "m" and len(data["traceEvents"]) == 6

    monkeypatch.setattr("resin_models_gaudi.synapse.trace.chrome_events", lambda _: [])
    try:
        trace(
            BenchCase(phase="prefill", context_length=4096, batch_size=1), lambda: None
        )
    except RuntimeError as error:
        assert "no device engine activity" in str(error)
    else:
        raise AssertionError("an empty trace must fail")


def test_prefill_defaults_to_two_cohorts_so_truncation_is_detectable(
    tmp_path, monkeypatch
):
    """One pass makes "units detected == passes recorded" true by construction (STATUS.md 13).

    The default was changed to 1, which restored exactly the failure section 13 records: a
    fragment passes every internal check. Two passes is the smallest count under which a
    truncated prefill trace shows one unit where two were recorded.
    """
    monkeypatch.delenv("RESIN_TRACE_PREFILL_PASSES", raising=False)
    passes = []
    device = SimpleNamespace(
        sync=lambda: None, prof_start=lambda: None, prof_stop=write_tsv
    )
    case = BenchCase(phase="prefill", context_length=512000, batch_size=1)

    perfetto_trace(tmp_path, 0, None, device=device)(case, lambda: passes.append(1))
    assert len(passes) == 2
    config = json.loads(
        (tmp_path / "prefill-ctx512000-b1-rank000-config.json").read_text()
    )
    assert config["passes"] == 2 and config["unit"] == "prefill cohorts"

    # and the override still works, because a buffer that cannot hold two cohorts needs one
    monkeypatch.setenv("RESIN_TRACE_PREFILL_PASSES", "1")
    perfetto_trace(tmp_path / "one", 0, None, device=device)(case, lambda: passes.append(1))
    assert len(passes) == 3


def test_profiler_configuration_points_outputs_at_the_directory(tmp_path, monkeypatch):
    # placeholders make teardown restore (or remove) the variables the call sets

    monkeypatch.setenv("HABANA_PROFILE", "0")
    monkeypatch.setenv("HABANA_PROF_CONFIG", "")
    enable_profiler(tmp_path / "prof")
    config = json.loads((tmp_path / "prof" / "profiler.json").read_text())
    assert config["GeneralSettings"]["values"]["outdir"]["value"] == str(
        tmp_path / "prof"
    )
    assert os.environ["HABANA_PROFILE"] == "1"
    assert os.environ["HABANA_PROF_CONFIG"] == str(tmp_path / "prof" / "profiler.json")


def test_the_case_config_survives_a_failed_json_write_and_carries_the_timing(
    tmp_path, monkeypatch
):
    """The config is written BEFORE the Chrome JSON, and carries the harness's own seconds.

    The JSON is the artifact that grows to gigabytes and gets killed; the config is a kilobyte
    and is what `trace_union` gates a trace against. Our 8192 prefill run is the proof that the
    ordering matters: the hook died inside the JSON write and took the config with it, so the
    pass count had to be reconstructed by hand and the completeness argument had to fall back on
    triangulation.
    """

    def stop(path):
        write_tsv(path)
        return len(ROWS)

    device = SimpleNamespace(sync=lambda: None, prof_start=lambda: None, prof_stop=stop)
    trace = perfetto_trace(tmp_path, 0, {"model": "m"}, device=device, decode_passes=2)

    def killed(*args, **kwargs):
        raise OSError("killed during the JSON write")

    monkeypatch.setattr("resin_models_gaudi.synapse.trace.gzip.open", killed)
    case = BenchCase(phase="prefill", context_length=8192, batch_size=1)
    with pytest.raises(OSError):
        trace(case, lambda: {"seconds": 3.5, "steps": 4})

    config = json.loads(
        (tmp_path / "prefill-ctx8192-b1-rank000-config.json").read_text()
    )
    assert config["phase"] == "prefill"
    assert config["passes"] == 2
    assert config["prefill_seconds"] == 3.5
    assert config["prefill_seconds_per_pass"] == [3.5, 3.5]
