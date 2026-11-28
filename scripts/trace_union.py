"""Per-node, per-engine union table from a Wright/Synapse device-trace TSV.

The TSV is what ``Device.prof_stop`` writes (``rg_prof_stop`` in
``models/gaudi/resin_models_gaudi/synapse/native/runtime.cpp``):
``ph engine_type engine_index ts dur name category meta [recipe_id recipe_name]``. Device events are
``B``/``E`` marker pairs named ``n<idx>_<node>``, one pair per engine instance the node ran on, plus a
``V`` ``Recipe Start`` row per launch. This tool merges each node's marker pairs into intervals and
reports the **union** (merged-interval duration) per node and per engine — never a sum of event
durations (manual ch2 §2.9).

usage: trace_union.py TRACE.tsv [--window preferred|alternate] [--json OUT.json]

Engine instances are normalised by module: ``engine_type`` carries the module as ``+4000 * module``
(notes/trace-event-union.md §4a), so the family is ``engine_type % 4000``. Host rows are
``category == "Host API"`` or ``engine_type >= 16000``.

Each engine numbers its sync points differently, so the marker pair is per engine and was read off a
4k prefill trace (562 launches, 11.8 M rows): TPC ``SP 1``→``SP 2`` command / ``SP 8``→``SP 3`` kernel,
MME ``SP 0``→``SP 1`` and ``SP 2``→``SP 3``, EDMA and PDMA ``SP 8``→``SP 15``. A single global pair
silently drops every engine whose markers are numbered otherwise (the first version of this tool
reported TPC only on a trace that also held 4.4 M MME events).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# gaudi2 device engine ids within a module
FAMILY = {2028: "TPC", 2024: "MME", 2018: "EDMA", 2034: "PDMA"}

# DMA records are named after the copy command, not after a graph node: some carry per-chunk
# counters ("lin_memset stream_index 0 chunk_id 3"), others are per-tensor copies whose names are
# unique per event ("t12_bundle_1201/memcpy/0" — 132,876 distinct names in one 4k prefill trace).
# Group by the command kind so the table stays readable.
_CHUNK = re.compile(r"(?:\s+\w+ \d+)+$")
_DMA = frozenset({"EDMA", "PDMA"})


def _dma_label(name: str) -> str:
    """The copy command a DMA record belongs to, without per-event counters or tensor ids."""
    head, _, rest = name.partition(" ")
    if rest:
        return _CHUNK.sub("", head)
    parts = head.split("/")
    return "/".join(parts[1:]) if len(parts) > 1 else head


# begin category -> end category, per engine: the preferred window, and the second one where the
# engine carries two (TPC's command markers, MME's inner window).
WINDOWS = {
    "TPC": ("SP 8", "SP 3"),
    "MME": ("SP 0", "SP 1"),
    "EDMA": ("SP 8", "SP 15"),
    "PDMA": ("SP 8", "SP 15"),
}
ALTERNATES = {"TPC": ("SP 1", "SP 2"), "MME": ("SP 2", "SP 3")}


def rows(path: str | Path):
    with open(path, newline="") as source:
        yield from csv.DictReader(source, delimiter="\t")


def is_device(row: dict) -> bool:
    """A device-engine row: not a host API span, not a flow row, engine id below the host range."""
    if row.get("category") == "Host API" or not row.get("ph"):
        return False
    try:
        engine = int(row["engine_type"])
    except (TypeError, ValueError):
        return False
    return 0 < engine < 16000


def windows(window: str) -> dict[str, tuple[str, str]]:
    """The begin/end marker pair to use for each engine family."""
    if window == "alternate":
        return {
            family: ALTERNATES.get(family, pair) for family, pair in WINDOWS.items()
        }
    return dict(WINDOWS)


def device_intervals(
    path: str | Path, window: str = "preferred"
) -> list[tuple[str, str, float, float]]:
    """``(engine family, node, start_us, end_us)`` per matched marker pair, per engine family."""
    wanted = windows(window)
    open_pairs: dict[tuple, list[float]] = defaultdict(list)
    found: list[tuple[str, str, float, float]] = []
    for row in rows(path):
        if not is_device(row):
            continue
        name = row["name"]
        if not name:
            continue
        engine = int(row["engine_type"])
        family = FAMILY.get(engine % 4000, f"engine{engine % 4000}")
        pair = wanted.get(family)
        if pair is None:
            continue
        key = (family, engine, row["engine_index"], name)
        if row["ph"] == "B" and row["category"] == pair[0]:
            open_pairs[key].append(float(row["ts"]))
        elif row["ph"] == "E" and row["category"] == pair[1]:
            starts = open_pairs.get(key)
            if starts:
                label = _dma_label(name) if family in _DMA else name
                found.append((family, label, starts.pop(0), float(row["ts"])))
    return found


def merge(spans: list[tuple[float, float]]) -> float:
    """Merged-interval duration of ``spans`` (the union), in the spans' own units."""
    if not spans:
        return 0.0
    ordered = sorted(spans)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + (end - start)


def union_table(path: str | Path, window: str = "preferred") -> dict:
    """The per-node and per-engine union table for one trace."""
    intervals = device_intervals(path, window)
    per_node: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for family, name, start, end in intervals:
        per_node[(family, name)].append((start, end))
    per_engine: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (family, _), spans in per_node.items():
        per_engine[family].extend(spans)
    launches = [
        (float(row["ts"]), float(row["ts"]) + float(row["dur"]))
        for row in rows(path)
        if row["ph"] == "V" and row["category"] == "Recipe Start"
    ]
    node_rows = [
        {
            "engine": family,
            "node": name,
            "intervals": len(spans),
            "union_us": round(merge(spans), 3),
        }
        for (family, name), spans in per_node.items()
    ]
    node_rows.sort(key=lambda row: -float(row["union_us"]))
    engine_rows = [
        {
            "engine": family,
            "union_us": round(merge(spans), 3),
            "nodes": sum(1 for (f, _) in per_node if f == family),
        }
        for family, spans in sorted(
            per_engine.items(), key=lambda item: -merge(item[1])
        )
    ]
    return {
        "trace": str(path),
        "window": window,
        "launches": len(launches),
        "launch_window_us": round(merge(launches), 3),
        "merged_us": round(
            merge([span for spans in per_engine.values() for span in spans]), 3
        ),
        "engines": engine_rows,
        "nodes": node_rows,
        "attributed": sum(float(row["union_us"]) for row in node_rows) > 0,
        "unnamed_device_events": sum(
            1
            for row in rows(path)
            if is_device(row) and not row["name"] and row["ph"] in ("B", "E")
        ),
    }


def report(result: dict) -> None:
    print(f"trace: {result['trace']}")
    print(
        f"window: {result['window']}   launches: {result['launches']}"
        f"   launch window union: {result['launch_window_us']:.1f} us"
    )
    print(f"merged (all device engines): {result['merged_us']:.1f} us")
    print("(engines overlap, so the shares are of the union and do not sum to 100%)")
    print()
    total = result["merged_us"] or 1.0
    print(f"{'engine':6} {'union_us':>10} {'share':>7} {'nodes':>6}")
    for row in result["engines"]:
        print(
            f"{row['engine']:6} {row['union_us']:10.1f} {100 * row['union_us'] / total:6.1f}%"
            f" {row['nodes']:6}"
        )
    print()
    print(f"{'engine':6} {'union_us':>10} {'share':>7} {'intervals':>9}  node")
    for row in result["nodes"][:60]:
        print(
            f"{row['engine']:6} {row['union_us']:10.1f} {100 * row['union_us'] / total:6.1f}%"
            f" {row['intervals']:9}  {row['node']}"
        )
    if len(result["nodes"]) > 60:
        print(f"... {len(result['nodes']) - 60} more nodes")
    if result["unnamed_device_events"]:
        print(
            f"\nWARNING: {result['unnamed_device_events']} device B/E events carry no node name"
        )
    if not result["attributed"]:
        print("\nWARNING: no attributed device intervals - the trace is not analysable")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("trace", type=Path)
    ap.add_argument("--window", default="preferred", choices=("preferred", "alternate"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    result = union_table(args.trace, args.window)
    report(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))
    return 0 if result["attributed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
