"""Synapse profiler events become Chrome trace events (host only)."""

from __future__ import annotations

import gzip
import json
import os
from types import SimpleNamespace

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


def test_trace_keeps_the_raw_tsv_when_asked(tmp_path, monkeypatch):
    """The TSV is the only artifact that carries recipe_id/recipe_name; keep it on request."""

    def stop(path):
        write_tsv(path)
        return len(ROWS)

    device = SimpleNamespace(sync=lambda: None, prof_start=lambda: None, prof_stop=stop)
    trace = perfetto_trace(tmp_path, 0, None, device=device)
    monkeypatch.setenv("RESIN_TRACE_KEEP_TSV", "1")
    trace(BenchCase(phase="prefill", context_length=4096, batch_size=1), lambda: None)
    kept = tmp_path / "prefill-ctx4096-b1-rank000.tsv.gz"
    assert kept.exists()
    with gzip.open(kept, "rt") as source:
        assert source.readline().startswith("ph\tengine_type\tengine_index")


def test_profiler_configuration_points_outputs_at_the_directory(tmp_path, monkeypatch):
    # placeholders make teardown restore (or remove) the variables the call sets
    monkeypatch.setenv("HABANA_PROFILE", "0")
    monkeypatch.setenv("HABANA_PROF_CONFIG", "")
    # the profiler requires a loadable HwTrace plugin and a writable $HOME (post-graphs); both are
    # checked by enable_profiler, so give it a hermetic pair instead of the host's.
    monkeypatch.setattr(
        "resin_models_gaudi.synapse.trace.synapse_loaded", lambda: False
    )
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "libhw_trace.so").write_text("")
    monkeypatch.setenv("HABANA_PLUGINS_LIB_PATH", str(plugins))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    enable_profiler(tmp_path / "prof")
    config = json.loads((tmp_path / "prof" / "profiler.json").read_text())
    assert config["GeneralSettings"]["values"]["outdir"]["value"] == str(
        tmp_path / "prof"
    )
    assert os.environ["HABANA_PROFILE"] == "1"
    assert os.environ["HABANA_PROF_CONFIG"] == str(tmp_path / "prof" / "profiler.json")


def test_profiler_requires_the_hw_trace_plugin(tmp_path, monkeypatch):
    """A missing plugin is a status=18 from synProfilerStart and nothing else; name it up front."""
    monkeypatch.setattr(
        "resin_models_gaudi.synapse.trace.synapse_loaded", lambda: False
    )
    monkeypatch.setenv("HABANA_PLUGINS_LIB_PATH", str(tmp_path / "no-plugins"))
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        enable_profiler(tmp_path / "prof")
    except RuntimeError as error:
        assert "libhw_trace.so" in str(error) and "status=18" in str(error)
    else:
        raise AssertionError("a plugin directory without libhw_trace.so must fail")


def test_profiler_must_be_enabled_before_the_device_opens(tmp_path, monkeypatch):
    """libSynapse's profiler stubs answer status=18 once the library is mapped: catch the order."""
    monkeypatch.setattr("resin_models_gaudi.synapse.trace.synapse_loaded", lambda: True)
    try:
        enable_profiler(tmp_path / "prof")
    except RuntimeError as error:
        assert "before the device is opened" in str(error) and "status=18" in str(error)
    else:
        raise AssertionError("enabling the profiler after the device opened must fail")


def test_profiler_requires_a_writable_post_graph_directory(tmp_path, monkeypatch):
    """With HABANA_PROFILE=1 the compiler writes post-graphs under $HOME; read-only fails status=26."""
    monkeypatch.setattr(
        "resin_models_gaudi.synapse.trace.synapse_loaded", lambda: False
    )
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "libhw_trace.so").write_text("")
    monkeypatch.setenv("HABANA_PLUGINS_LIB_PATH", str(plugins))
    home = tmp_path / "home"
    (home / ".habana").mkdir(parents=True)
    (home / ".habana" / "post_graph").write_text(
        ""
    )  # a file where the directory must be
    monkeypatch.setenv("HOME", str(home))
    try:
        enable_profiler(tmp_path / "prof")
    except RuntimeError as error:
        assert "post_graph" in str(error) and "status=26" in str(error)
    else:
        raise AssertionError("a $HOME without a writable post_graph must fail")
