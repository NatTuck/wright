"""Perfetto traces of Synapse device activity, for ``resin.bench.run_sweep(trace_step=...)``.

Device activity is recorded between ``Device.prof_start`` and ``prof_stop``. Call
``enable_profiler`` before the device opens. Events are written as a Chrome trace whose
processes are engine types and whose threads are engine instances.

The configuration below is not free to choose: it is the only combination measured to emit
device execution records (cases/wright/measurements/profile/matrix/README.md).

  * ``profilePhase`` must be ``profileApi``. ``profileDevice`` is refused by the HwTrace
    plugin itself ("Profiler phase is not Profile instrumentation (API mode)"), which
    surfaces as the generic ``synProfilerStart`` status 26. ``profileDevice`` is not even in
    ``libhw_trace.so``'s phase vocabulary.
  * the trace *type* selects the device records: ``RESIN_PROF_TRACE_TYPE=device``
    (``synTraceDevice``). ``synTraceAll`` on this plugin returns host API calls only.
  * ``outputPerInvocation`` does not gate device records; json alone is the smallest.
  * the device emits ``B``/``E`` *pairs* with ``dur=0``; a duration is ``ts(E) - ts(B)``.
    ``recipe_id`` is populated on device rows, ``recipe_name`` almost never.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import asdict
from pathlib import Path

_PHASES = {"B", "E", "X", "i", "M", "V"}
# `V` is the per-launch `Recipe Start` record: one per recipe launch, carrying the
# recipe_id and the launch's duration. It is the only program-order marker in a device
# trace, so a parser needs it to tell one launch of a recipe from the next.


def enable_profiler(
    directory: str | Path, *, phase: str = "profileApi", trace_type: str = "device"
) -> None:
    """Configure this process's Synapse profiler (before ``open_device``); the profiler's
    own output files go to ``directory``.

    ``phase`` defaults to ``profileApi`` because that is the only phase the HwTrace plugin
    accepts (``profileDevice`` is refused, and is not in the plugin's vocabulary at all).
    ``trace_type`` is what ``rg_prof_start`` passes to ``synProfilerStart``; ``device`` is
    the one that emits device execution records.
    """
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
                "lib": "libhost_profiler.so",
                "name": "HostProfiler",
                "values": {"output": {"csv": off, "hltv": off, "json": json_only}},
            },
            {
                "lib": "libhw_trace.so",
                "name": "HwTrace",
                "values": {
                    "generalOptions": {
                        "arch": {"value": "gaudi2"},
                        "profilePhase": {"value": phase},
                    },
                    "parseOptions": {
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
    # The native side reads this to choose synTraceType; without it the type is synTraceAll,
    # which on this plugin returns host API calls and no device records.
    os.environ["RESIN_PROF_TRACE_TYPE"] = trace_type
    (directory / "trace-config.json").write_text(
        json.dumps(
            {
                "profile_phase": phase,
                "trace_type": trace_type,
                "output_per_invocation": ["json"],
                "plugins": ["HostProfiler", "HwTrace"],
                "DUMP_POST_GRAPHS": os.environ.get("DUMP_POST_GRAPHS"),
                "HABANA_PROFILE": os.environ.get("HABANA_PROFILE"),
                "resin_prof_trace_type": os.environ.get("RESIN_PROF_TRACE_TYPE"),
                "config": str(path),
            },
            indent=2,
        )
    )


def chrome_events(tsv: str | Path) -> list[dict]:
    """The profiler's TSV events (``rg_prof_stop``) as Chrome trace events."""
    events = []
    # latin-1, not utf-8: the ``ph=M`` metadata rows carry binary garbage in the union
    # columns (the writer reads that union as recipe fields), and a strict decode aborts
    # the whole trace. See measurements/profile/matrix/README.md.
    with open(tsv, newline="", encoding="latin-1") as source:
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
            if phase != "M":
                # recipe identity is what makes a device row attributable; it is populated
                # on device rows and on the `V`/Recipe Start rows, never on metadata rows.
                args = event.get("args") or {}
                if row.get("recipe_id") and row["recipe_id"] != "0":
                    args["recipe_id"] = int(row["recipe_id"])
                if row.get("recipe_name"):
                    args["recipe_name"] = row["recipe_name"]
                if args:
                    event["args"] = args
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
        # The raw TSV is the attributable artifact (it carries recipe_id and both halves of
        # every B/E pair); the Chrome JSON is a view of it. Keep both.
        tsv = directory / f"{name}.tsv"
        try:
            for _ in range(passes):
                run_pass()
        finally:
            device.prof_stop(tsv)
            events = chrome_events(tsv)
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
        # A self-describing directory: every knob that decides what a device trace contains,
        # resolved, beside the trace it produced.
        (directory / f"{name}-config.json").write_text(
            json.dumps(
                {
                    "phase": case.phase,
                    "context_length": case.context_length,
                    "batch_size": case.batch_size,
                    "shared_prefix_tokens": case.shared_prefix_tokens,
                    "rank": rank,
                    "passes": passes,
                    "unit": unit,
                    "model": (metadata or {}).get("model"),
                    "checkpoint": (metadata or {}).get("checkpoint"),
                    "hardware": (metadata or {}).get("hardware"),
                    "profile_phase": "profileApi",
                    "resin_prof_trace_type": os.environ.get("RESIN_PROF_TRACE_TYPE"),
                    "output_per_invocation": ["json"],
                    "DUMP_POST_GRAPHS": os.environ.get("DUMP_POST_GRAPHS"),
                    "HABANA_PROFILE": os.environ.get("HABANA_PROFILE"),
                    "tsv": str(tsv),
                    "events": len(events),
                    "device_events": busy,
                },
                indent=2,
            )
        )
        print(
            f"[bench-trace] rank={rank} case={name} Synapse engine events={busy}",
            flush=True,
        )
        if not busy:
            raise RuntimeError(f"{case.phase} trace contains no device engine activity")

    setattr(trace, "decode_passes", decode_passes)
    return trace
