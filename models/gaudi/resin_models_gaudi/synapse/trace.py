"""Perfetto traces of Synapse device activity, for ``resin.bench.run_sweep(trace_step=...)``.

Device activity is recorded between ``Device.prof_start`` and ``prof_stop``. Call
``enable_profiler`` before the device opens. Events are written as a Chrome trace whose
processes are engine types and whose threads are engine instances.

The config below is the image's own ``profile_api`` template
(``/opt/habanalabs/bin/synprof_configuration --use-template profile_api -c gaudi2``) plus the
advanced per-unit records and an explicit trace buffer. What the device trace then carries, per
launch (measured, one recipe, one card):

* ``V`` / category ``Recipe Start`` — one per launch, with the recipe name and its duration;
* ``B``/``E`` pairs named ``n<idx>_<node>``, one per engine instance the node runs on (category
  ``SP 1``/``SP 2``), plus the kernel window ``SP 8`` → ``SP 3`` and an ``SP 4`` instant when
  ``archProfileUnits...AdvancedProfiling`` is on;
* ``X`` / category ``Host API`` spans, and ``M`` metadata rows naming engines and instances.

Three environment preconditions are load-bearing, and all fail *opaque* when unmet:

* **``enable_profiler`` must run before the device is opened.** Synapse loads its profiler plugin
  when ``HABANA_PROFILE`` is set at the first Synapse call, and only then do the plugin's
  ``synSingleton::profiler*`` definitions interpose over the stubs in ``libSynapse.so``. Those stubs
  (``libSynapse.so`` 0xfd51a0-0xfd51fa) return ``0x12`` = ``synUnsupported`` for ``profilerStart``,
  ``profilerStop``, ``profilerGetTrace`` and ``profilerSetUserBuffer``, so a process that armed the
  profiler *after* opening the device gets ``synProfilerStart ... status=18 (Unsupported operation)``
  and nothing else. Measured both ways (a card claimed before ``enable_profiler`` fails; the same
  card claimed after it traces), and this is why ``scripts/bench.py`` enables the profiler before
  ``claim_card`` rather than before the runner.
* **The HwTrace plugin must be loadable.** ``HABANA_PLUGINS_LIB_PATH`` must contain
  ``libhw_trace.so``; without it the plugin is not loaded and the same stubs answer ``status=18``.
* **``$HOME`` must be writable.** With ``HABANA_PROFILE=1`` the graph compiler serialises post-graphs
  to ``$HOME/.habana/post_graph``; on a read-only ``$HOME`` every ``synGraphCompile`` then fails with
  ``status=26`` (``synapse_runtime.log``: ``open failed after 3 retries ... errno: 30``). This is the
  reason a profiled run needs a writable ``HOME`` (or ``DUMP_POST_GRAPHS=0``, which loses the dumps
  that attribute a region to its recipe).

``enable_profiler`` checks all three and raises with the path or the call order it resolved, so a
trace run cannot get several minutes into a boot only to fail with a bare status code.

``RESIN_TRACE_KEEP_TSV=1`` also keeps each trace's raw event rows (``<case>.tsv.gz`` next to the
Chrome trace), which is what ``scripts/trace_union.py`` reads; the Chrome trace drops the
``recipe_id``/``recipe_name`` columns.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

_PHASES = {"B", "E", "X", "i", "M", "V"}

# Synapse's own plugin directory, used when HABANA_PLUGINS_LIB_PATH is unset.
_DEFAULT_PLUGINS = "/opt/habanalabs/habana_plugins"


def synapse_loaded() -> bool:
    """Whether this process has already mapped ``libSynapse.so``.

    Synapse loads its profiler plugin when ``HABANA_PROFILE`` is set at the first Synapse call, and
    the plugin's ``synSingleton::profiler*`` definitions interpose over the stubs in ``libSynapse.so``
    only then. After ``libSynapse.so`` is mapped the stubs are what a call reaches, and they return
    ``synUnsupported`` (18) — so the profiler must be enabled before the device is opened."""
    try:
        with open("/proc/self/maps") as maps:
            return any("libSynapse.so" in line for line in maps)
    except OSError:
        return False


def _check_profiler_environment() -> None:
    """Fail with the resolved paths when the profiler cannot work in this process.

    Each condition below otherwise surfaces only as a bare ``synStatus`` from a call that is
    several minutes into a run (see the module docstring)."""
    if synapse_loaded():
        raise RuntimeError(
            "the Synapse profiler must be enabled before the device is opened: this process has "
            "already loaded libSynapse.so, whose profiler stubs answer synProfilerStart with "
            "status=18 (synUnsupported) because the HwTrace plugin can no longer be interposed. "
            "Call enable_profiler before open_device/claim_card."
        )
    directory = Path(os.environ.get("HABANA_PLUGINS_LIB_PATH") or _DEFAULT_PLUGINS)
    if not (directory / "libhw_trace.so").exists():
        raise RuntimeError(
            f"the Synapse profiler needs {directory / 'libhw_trace.so'} (HABANA_PLUGINS_LIB_PATH="
            f"{os.environ.get('HABANA_PLUGINS_LIB_PATH') or '<unset>'}); without it synProfilerStart "
            f"returns status=18 with no other trace. Set HABANA_PLUGINS_LIB_PATH to the directory "
            f"holding libhw_trace.so (e.g. {_DEFAULT_PLUGINS})."
        )
    post_graph = Path.home() / ".habana" / "post_graph"
    try:
        post_graph.mkdir(parents=True, exist_ok=True)
        probe = post_graph / ".resin-write-probe"
        probe.write_text("")
        probe.unlink()
    except OSError as error:
        raise RuntimeError(
            f"the profiler makes the graph compiler write post-graphs under {post_graph}, and this "
            f"process cannot ({error}). Every synGraphCompile then fails with status=26. Run with a "
            f"writable HOME (HOME=/tmp/<dir>), or set DUMP_POST_GRAPHS=0 to give up the dumps."
        ) from None


def enable_profiler(directory: str | Path) -> None:
    """Configure this process's Synapse profiler (before ``open_device``); the profiler's
    own output files go to ``directory``."""
    _check_profiler_environment()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_only = {"value": True}
    off = {"value": False}
    config = {
        "GeneralSettings": {
            "values": {
                "outdir": {"value": str(directory)},
                "session": {"value": "resin"},
            }
        },
        "Plugins": [
            {
                "enable": True,
                "lib": "libhost_profiler.so",
                "name": "HostProfiler",
                # the template's shape: the host profiler is armed by the API call, not at device
                # acquire. It supplies the Host API spans the device events are read against.
                "values": {
                    "start_disabled": {"value": True},
                    "api_group": {"SYNAPSE": {"value": True}},
                },
            },
            {
                "enable": True,
                "lib": "libhw_trace.so",
                "name": "HwTrace",
                "values": {
                    "generalOptions": {
                        "arch": {"value": "gaudi2"},
                        # the only phase synProfilerStart/synProfilerStop/synProfilerGetTrace can
                        # drive: activateProfiling hard-gates on it (source read, libhw_trace.so)
                        "profilePhase": {"value": "profileApi"},
                        # 128 MiB (the template's value) in DRAM: the default is 80 MiB and a
                        # truncated buffer looks exactly like coarse events
                        "traceBufferSize": {"value": "0x8000000"},
                        "traceBufferLocation": {"value": "dram"},
                    },
                    # the per-node kernel window (SP 8 -> SP 3) and the named Recipe Start record;
                    # a SIBLING of generalOptions (inside it, these are silently ignored)
                    "archProfileUnits": {
                        "gaudi2": {
                            "TPC": {"Gaudi2TpcAdvancedProfiling": {"value": True}},
                            "MME": {"Gaudi2MmeAdvancedProfiling": {"value": True}},
                        }
                    },
                    "parseOptions": {
                        "addFuserMetadata": {"value": True},
                        "showNullDescs": {"value": False},
                        "mergeWithHost": {"value": True},
                        # parse cost only: the trace analyzer renders derived tables nobody reads
                        "traceAnalyzer": {"traceAnalyzerJson": {"value": False}},
                        "outputPerInvocation": {
                            "binary": off,
                            "csv": off,
                            "dbgInfo": off,
                            "hltv": off,
                            "hltvWithHost": off,
                            "json": json_only,
                            "text": off,
                        }
                    },
                },
            },
        ],
    }
    path = directory / "profiler.json"
    path.write_text(json.dumps(config))
    os.environ["HABANA_PROFILE"] = "1"
    os.environ["HABANA_PROF_CONFIG"] = str(path)


def chrome_events(tsv: str | Path) -> list[dict]:
    """The profiler's TSV events (``rg_prof_stop``) as Chrome trace events."""
    events = []
    with open(tsv, newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            phase = row["ph"]
            if phase not in _PHASES:
                continue
            event = {
                "ph": phase,
                "pid": int(row["engine_type"]),
                "tid": int(row["engine_index"]),
                "ts": float(row["ts"]),
                "name": row["name"],
            }
            if row["category"]:
                event["cat"] = row["category"]
            if phase in ("X", "V"):
                event["dur"] = float(row["dur"])
            elif phase == "M":
                event["args"] = {"name": row["meta"]}
            elif phase == "i":
                event["s"] = "t"
            events.append(event)
    return events


def perfetto_trace(
    directory: str | Path,
    rank: int = 0,
    metadata: dict | None = None,
    *,
    device,
    decode_passes: int = 2,
):
    """A ``run_sweep`` trace hook for a Synapse device: two untimed prefill cohorts or
    ``decode_passes`` completions per case, written as compressed Chrome traces."""
    if type(decode_passes) is not int or decode_passes < 1:
        raise ValueError("decode_passes must be a positive integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def trace(case, run_pass):
        name = f"{case.phase}-ctx{case.context_length}-b{case.batch_size}-rank{rank:03}"
        passes = decode_passes if case.phase == "decode" else 2
        unit = "decode completions" if case.phase == "decode" else "prefill cohorts"
        device.sync()
        device.prof_start()
        try:
            for _ in range(passes):
                run_pass()
        finally:
            with tempfile.TemporaryDirectory(prefix="resin-synapse-trace-") as staging:
                tsv = Path(staging) / f"{name}.tsv"
                device.prof_stop(tsv)
                events = chrome_events(tsv)
                if os.environ.get("RESIN_TRACE_KEEP_TSV") == "1":
                    # the raw event rows (with recipe_id/recipe_name, which the Chrome trace drops):
                    # what scripts/trace_union.py reads, kept only when asked for
                    with gzip.open(directory / f"{name}.tsv.gz", "wb") as target:
                        target.write(tsv.read_bytes())
        busy = sum(event["ph"] in ("B", "X") and event["pid"] != 0 for event in events)
        trace_data = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "resin_rank": rank,
            "resin_timing": f"{passes} untimed {unit} after measurement",
            "resin_benchmark": {
                **(metadata or {}),
                "case": asdict(case),
                "trace_passes": passes,
                "trace_unit": unit,
                "trace_source": "synapse_profiler",
            },
        }
        with gzip.open(directory / f"{name}.json.gz", "wt", encoding="utf-8") as target:
            json.dump(trace_data, target)
        print(
            f"[bench-trace] rank={rank} case={name} Synapse engine events={busy}",
            flush=True,
        )
        if not busy:
            raise RuntimeError(f"{case.phase} trace contains no device engine activity")

    setattr(trace, "decode_passes", decode_passes)
    return trace
