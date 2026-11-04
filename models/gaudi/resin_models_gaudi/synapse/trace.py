"""Perfetto traces of Synapse device activity, for ``resin.bench.run_sweep(trace_step=...)``.

Device activity is recorded between ``Device.prof_start`` and ``prof_stop``. Call
``enable_profiler`` before the device opens. Events are written as a Chrome trace whose
processes are engine types and whose threads are engine instances.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

_PHASES = {"B", "E", "X", "i", "M"}


def enable_profiler(directory: str | Path) -> None:
    """Configure this process's Synapse profiler (before ``open_device``); the profiler's
    own output files go to ``directory``."""
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
                        "profilePhase": {"value": "profileApi"},
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
            if phase == "X":
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
