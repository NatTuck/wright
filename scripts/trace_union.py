#!/usr/bin/env python3
"""Union accounting for a Wright Synapse *device* trace.

The manual's rules, implemented literally (g2-opt-manual ch 2 §2.9):

  * a duration is a MERGED UNION of intervals, never a sum of events;
  * **union within a recipe, sum across recipes** -- recipes are serial, so per-recipe
    unions partition the step exactly;
  * merge per node per tid first (MME and DMA markers self-overlap heavily);
  * never sum across TP ranks: one trace is one device, and a rank's trace is that device.

Input is the raw profiler TSV (`rg_prof_stop`), which is the attributable artifact, or the
gzipped Chrome JSON written beside it (a lossy view: it keeps recipe_id, not recipe_name).

Device rows come in BEGIN/END pairs with ``dur`` 0 by construction:

    B  2028  31  ...  0.000  n1_rg_swiglu_f32  SP 1  62451
    E  2028  31  ...  0.000  n1_rg_swiglu_f32  SP 2  62451

The two rows of one pair carry DIFFERENT ``SP n`` categories, so they are paired on
``(engine_type, engine_index, name)`` with a stack -- never on category, and not on
``recipe_id`` alone (a recipe launched twice has one id and two instances).

Three facts that silently corrupt a TP4 table if ignored:

  * ``engine_type`` carries the module index as ``+4000`` per module: TPC is 2028 on module 0
    and 6028 on module 1. Normalised with ``% 4000`` before anything is keyed on it.
  * the host pseudo-engine (16425, 16433, ... 17740) is a per-run handle. Host rows are
    identified by ``category == "Host API"``, never by engine id.
  * ``ph=M`` metadata rows carry binary garbage in the recipe columns; they are skipped, and
    the file is decoded latin-1 so one bad byte cannot abort a whole trace.

Usage:
    trace_union.py TRACE [--label TEXT] [--out FILE] [--top N] [--period N] [--rank N]

Exit status 2 means the trace has no device execution events: a pre-flight failure, not a
measurement.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Engine ids as they appear after `% 4000`. The names are only a fallback: a trace's own
# `__metadata` process_name rows are authoritative and are preferred when present.
_ENGINES = {
    2018: "EDMA",
    2022: "EDMA?",
    2024: "MME",
    2028: "TPC",
    2030: "PDMA?",
    2034: "PDMA",
    2157: "ENG ARC",
    2004: "?2004",
    2008: "?2008",
}
_DEVICE_MIN, _DEVICE_MAX = 2000, 16000  # 4 modules * 4000 + 3999 stays below 16000


def engine_of(engine_type: int) -> int:
    """Drop the module tag (``+4000`` per module) so four ranks are four engines, not twelve."""
    if _DEVICE_MIN <= engine_type < _DEVICE_MAX:
        return engine_type % 4000
    return engine_type


def merge(intervals) -> float:
    """Total covered length of [start, end) intervals -- a union, not a sum."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    return total + (cur_e - cur_s)


# Op family from a node name. Node names in a recipe are `<index>_<kernel>`, e.g.
# `n1_rg_swiglu_f32`, so the leading index is stripped first. Ordered: the more specific
# prefixes win. The report prints the name->family mapping it used, so it is auditable.
_FAMILIES = (
    ("mx4_decs", "expert-dequant"),
    ("mx4", "expert-dequant"),
    ("fp8_gemm", "gemm"),
    ("batch_gemm", "gemm"),
    ("gemm", "gemm"),
    ("moe_route", "routing"),
    ("ix_thr", "indexer"),
    ("ix_gather", "indexer"),
    ("ix_slot", "indexer"),
    ("stream_index", "indexer"),
    ("swiglu", "act"),
    ("memset", "copy/dma"),
    ("attn_pre", "attn_pre"),
    ("attn_out", "attn_out"),
    ("attn", "attn"),
    ("ffn", "ffn"),
    ("moe", "ffn"),
    ("expert", "ffn"),
    ("engram", "engram"),
    ("index", "index/select"),
    ("select", "index/select"),
    ("gather", "index/select"),
    ("head", "head"),
    ("embed", "embed"),
    ("reduce", "reduce/collective"),
    ("collective", "reduce/collective"),
    ("allreduce", "reduce/collective"),
    ("hccl", "reduce/collective"),
    ("copy", "copy/dma"),
    ("pdma", "copy/dma"),
    ("h2d", "copy/dma"),
    ("d2h", "copy/dma"),
    ("barrier", "sync/barrier"),
)


def family(name: str) -> str:
    lower = name.lower()
    body = lower.split("_", 1)[1] if lower.startswith("n") and "_" in lower else lower
    for key, value in _FAMILIES:
        if key in body or key in lower:
            return value
    return "other"


class Row:
    __slots__ = ("ph", "engine", "index", "ts", "dur", "name", "cat", "rid", "rname")

    def __init__(self, ph, engine, index, ts, dur, name, cat, rid, rname):
        self.ph, self.engine, self.index = ph, engine, index
        self.ts, self.dur, self.name = ts, dur, name
        self.cat, self.rid, self.rname = cat, rid, rname


def load(path: Path) -> list[Row]:
    """Read a profiler TSV or the Chrome JSON beside it into rows."""
    text_path = path
    if path.suffix == ".gz" or path.suffix == ".json":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="latin-1") as fh:
            data = json.load(fh)
        rows = []
        for event in data["traceEvents"]:
            phase = event.get("ph")
            if phase not in ("B", "E", "X", "i", "V"):
                continue
            args = event.get("args") or {}
            rows.append(
                Row(
                    phase,
                    int(event.get("pid", 0)),
                    int(event.get("tid", 0)),
                    float(event.get("ts", 0.0)),
                    float(event.get("dur", 0.0)),
                    event.get("name") or "",
                    event.get("cat") or "",
                    int(args.get("recipe_id", 0) or 0),
                    args.get("recipe_name") or "",
                )
            )
        return rows

    rows = []
    # latin-1: the ph=M metadata rows carry binary garbage in the recipe columns.
    with open(text_path, newline="", encoding="latin-1") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            phase = row.get("ph") or ""
            if phase not in ("B", "E", "X", "i", "V"):
                continue
            if phase == "M":
                continue
            try:
                ts = float(row["ts"])
                dur = float(row["dur"])
                engine = int(row["engine_type"])
                index = int(row["engine_index"])
            except (TypeError, ValueError):
                continue
            rid = row.get("recipe_id") or "0"
            rows.append(
                Row(
                    phase,
                    engine,
                    index,
                    ts,
                    dur,
                    row.get("name") or "",
                    row.get("category") or "",
                    int(rid) if rid.isdigit() else 0,
                    row.get("recipe_name") or "",
                )
            )
    return rows


def engine_names(path: Path) -> dict[int, str]:
    """Process names from the trace's own `__metadata` rows, keyed by engine_type."""
    names: dict[int, str] = {}
    if path.suffix in (".gz", ".json"):
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt", encoding="latin-1") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return names
        for event in data["traceEvents"]:
            if event.get("ph") == "M" and event.get("name") == "process_name":
                label = (event.get("args") or {}).get("name") or ""
                if label:
                    names[int(event.get("pid", 0))] = label
        return names
    try:
        with open(path, newline="", encoding="latin-1") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("ph") == "M" and row.get("name") == "process_name":
                    label = row.get("meta") or ""
                    if label:
                        names[int(row.get("engine_type") or 0)] = label
    except (OSError, ValueError):
        pass
    return names


def summarize(path: Path) -> dict:
    """Recipe set and node counts of a trace, for comparing a substituted capture with the
    real one. Deliberately independent of the accounting above: a substitution is only valid
    if the recipe set and the node counts match, and that must be checkable, not asserted."""
    rows = load(path)
    stacks: dict[tuple[int, int, str], list] = defaultdict(list)
    intervals = []
    for row in sorted(rows, key=lambda r: r.ts):
        if row.cat == "Host API" or row.ph == "V":
            continue
        key = (engine_of(row.engine), row.index, row.name)
        if row.ph == "B":
            stacks[key].append(row)
        elif row.ph == "E" and stacks[key]:
            begin = stacks[key].pop()
            intervals.append((key[0], begin.rid, begin.ts, row.ts, begin.rname or row.rname))
    launches = [
        (r.ts, r.rid, r.rname) for r in rows if r.ph == "V" and r.cat == "Recipe Start"
    ]
    launches.sort()
    by_launch: dict[int, list[tuple[float, float]]] = defaultdict(list)
    starts: dict[int, list[float]] = defaultdict(list)
    for ts, rid, _name in launches:
        starts[rid].append(ts)
    names: dict[int, str] = {}
    for engine, rid, start, end, rname in intervals:
        if not rid or rid not in starts:
            continue
        seq = starts[rid]
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] <= start:
                lo = mid + 1
            else:
                hi = mid
        which = lo - 1
        if which < 0:
            continue
        key = (rid, seq[which])
        ordinal = next(
            (i for i, (ts, r, _n) in enumerate(launches) if (r, ts) == key), None
        )
        if ordinal is None:
            continue
        by_launch[ordinal].append((start, end))
        if rname and ordinal not in names:
            names[ordinal] = rname
    per_launch = sorted(len(v) for v in by_launch.values())
    engines = {
        engine
        for engine, *_rest in intervals
    }
    return {
        "path": path,
        "recipe_ids": sorted({rid for _ts, rid, _n in launches}),
        "recipe_names": sorted({n for n in names.values() if n}),
        "launches": len(launches),
        "nodes": per_launch,
        "median_nodes": per_launch[len(per_launch) // 2] if per_launch else 0,
        "engines": sorted(engines),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--label", default="", help="heading for the report")
    parser.add_argument("--out", type=Path, help="write the report here as well as stdout")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--period",
        type=int,
        help="launches per step, overriding detection (a trace of k identical passes "
        "otherwise detects the repeating launch sequence itself)",
    )
    parser.add_argument("--rank", type=int, default=None, help="only for the heading")
    parser.add_argument(
        "--compare",
        type=Path,
        help="another trace: print recipe-set and node-count agreement, so a substituted "
        "capture (a smaller context, a different batch) is CHECKED rather than asserted",
    )
    args = parser.parse_args(argv)

    rows = load(args.trace)
    if not rows:
        print(f"NO ROWS: {args.trace}")
        return 2

    names = engine_names(args.trace)
    phases = Counter(r.ph for r in rows)
    categories = Counter(r.cat or "?" for r in rows)

    # ---- device intervals: pair B with E on (engine, index, name) with a stack
    stacks: dict[tuple[int, int, str], list[Row]] = defaultdict(list)
    intervals = []  # (engine, index, name, rid, start, end, recipe_name)
    unmatched_end = 0
    leftover_e: list[Row] = []
    for row in sorted(rows, key=lambda r: r.ts):
        if row.cat == "Host API" or row.ph in ("V",):
            continue
        key = (engine_of(row.engine), row.index, row.name)
        if row.ph == "B":
            stacks[key].append(row)
        elif row.ph == "E":
            if stacks[key]:
                begin = stacks[key].pop()
                intervals.append(
                    (
                        key[0],
                        key[1],
                        key[2],
                        begin.rid,
                        begin.ts,
                        row.ts,
                        begin.rname or row.rname,
                    )
                )
            else:
                unmatched_end += 1
                leftover_e.append(row)
        elif row.ph == "X" and row.dur > 0:
            intervals.append(
                (
                    engine_of(row.engine),
                    row.index,
                    row.name,
                    row.rid,
                    row.ts,
                    row.ts + row.dur,
                    row.rname,
                )
            )
    # A second pass for what the name-keyed pass could not pair. Synapse emits flush and
    # end markers (`END_EVENT 324`, `DBG_DMA_TRC_WR_DATA_LAST 169`) whose begin and end rows
    # do not carry the same name, so a name-keyed stack leaves them unmatched -- 12,112 of
    # 250,318 B rows on a 32k prefill trace, 4.8%. A single engine instance is serial, so on
    # (engine, index) alone a stack is exact; this recovers those pairs without weakening the
    # name-keyed pass, and the count is reported so a reader can see how much came this way.
    leftover: list[Row] = []
    for stack in stacks.values():
        leftover.extend(stack)
    leftover.extend(leftover_e)
    leftover.sort(key=lambda r: r.ts)
    by_engine: dict[tuple[int, int], list[Row]] = defaultdict(list)
    for row in leftover:
        by_engine[(engine_of(row.engine), row.index)].append(row)
    fallback = 0
    for (engine, index), rows_ in by_engine.items():
        open_b: list[Row] = []
        for row in rows_:
            if row.ph == "B":
                open_b.append(row)
            elif open_b:
                begin = open_b.pop()
                intervals.append(
                    (
                        engine,
                        index,
                        begin.name,
                        begin.rid,
                        begin.ts,
                        row.ts,
                        begin.rname or row.rname,
                    )
                )
                fallback += 1
    unmatched_begin = sum(len(stack) for stack in stacks.values()) - fallback

    if not intervals:
        print(f"NO DEVICE EVENTS: {args.trace}")
        print(f"  categories: {dict(categories.most_common(8))}")
        print("  a device trace needs profilePhase=profileApi + RESIN_PROF_TRACE_TYPE=device")
        return 2

    # ---- launches: the `V` / `Recipe Start` rows, one per recipe launch, in program order
    launches = sorted(
        ((r.ts, r.dur, r.rid) for r in rows if r.ph == "V" and r.cat == "Recipe Start"),
        key=lambda item: item[0],
    )
    launch_starts: dict[int, list[float]] = defaultdict(list)
    for ts, _dur, rid in launches:
        launch_starts[rid].append(ts)

    def launch_of(rid: int, ts: float) -> int:
        """Index into `launches` of the instance this interval belongs to (-1 if none)."""
        if not rid or rid not in launch_starts:
            return -1
        starts = launch_starts[rid]
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= ts:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return -1
        return lo - 1

    # Launch ordinal for an interval = position of its (rid, start) in the global launch list.
    launch_index: dict[tuple[int, float], int] = {
        (rid, ts): i for i, (ts, _dur, rid) in enumerate(launches)
    }
    per_launch: dict[int, list[tuple[float, float]]] = defaultdict(list)
    per_launch_engine: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_launch_family: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_launch_name: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    outside: list[tuple[float, float]] = []
    outside_by_engine: dict[int, list[tuple[float, float]]] = defaultdict(list)
    unattributed = 0
    launch_rname: dict[int, str] = {}
    for engine, _index, name, rid, start, end, rname in intervals:
        if end <= start:
            continue
        if not launches:
            # No launch markers: the whole window is one unit and every interval is in it.
            per_launch[0].append((start, end))
            per_launch_engine[0][engine].append((start, end))
            per_launch_family[0][family(name)].append((start, end))
            per_launch_name[0][name].append((start, end))
            continue
        ordinal = launch_of(rid, start)
        if ordinal < 0:
            if rid == 0:
                outside.append((start, end))
                outside_by_engine[engine].append((start, end))
            else:
                unattributed += 1
            continue
        key = (rid, launch_starts[rid][ordinal])
        which = launch_index.get(key)
        if which is None:
            unattributed += 1
            continue
        per_launch[which].append((start, end))
        if rname and which not in launch_rname:
            launch_rname[which] = rname
        per_launch_engine[which][engine].append((start, end))
        per_launch_family[which][family(name)].append((start, end))
        per_launch_name[which][name].append((start, end))

    launch_union = {k: merge(v) for k, v in per_launch.items()}

    # ---- the repeating unit: one decode step, or one prefill chunk-step
    sequence = [rid for _ts, _dur, rid in launches]
    period = args.period
    detected = "given"
    if period is None:
        period = len(sequence)
        detected = "whole window (no repeating launch sequence found)"
        n = len(sequence)
        for candidate in range(1, n // 2 + 1):
            if n % candidate:
                continue
            if all(sequence[i] == sequence[i + candidate] for i in range(n - candidate)):
                period = candidate
                detected = "detected from the repeating launch sequence"
                break
    if period <= 0 or not launches:
        period = max(len(launch_union), 1)
        detected = "whole window"
    units = [range(i, min(i + period, len(launches))) for i in range(0, len(launches), period)]

    unit_union = [
        sum(launch_union.get(i, 0.0) for i in unit) for unit in units
    ]
    unit_span = []
    for unit in units:
        spans = [
            (start, end)
            for i in unit
            for (start, end) in per_launch.get(i, ())
        ]
        unit_span.append(
            0.0 if not spans else max(e for _s, e in spans) - min(s for s, _e in spans)
        )

    # ---- diagnostics
    all_intervals = [(iv[4], iv[5]) for iv in intervals]
    total_span = max(e for _s, e in all_intervals) - min(s for s, _e in all_intervals)
    step_union = unit_union[0] if unit_union else 0.0
    step_span = unit_span[0] if unit_span else total_span
    merged_all = merge(all_intervals)
    outside_union = merge(outside)
    rid_rows = sum(1 for r in rows if r.rid and r.cat != "Host API")

    # Three numbers, on ONE scope (the first repeating unit), because they only compare if
    # they cover the same time window:
    #   (a) merged union over every interval in the unit      -- device busy
    #   (b) sum of per-recipe unions                          -- the manual's accounting number
    #   (c) the difference, labelled: recipe-less device work (host copies, barriers, and any
    #       device activity with no recipe_id). If (c) dominates, the per-recipe table is not
    #       the step and saying so is the finding.
    boundaries = [launches[unit[0]][0] for unit in units] if launches else [0.0]

    def unit_of(ts: float) -> int:
        lo, hi = 0, len(boundaries)
        while lo < hi:
            mid = (lo + hi) // 2
            if boundaries[mid] <= ts:
                lo = mid + 1
            else:
                hi = mid
        return max(0, lo - 1)

    outside_by_unit: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for start, end in outside:
        outside_by_unit[unit_of(start)].append((start, end))

    def unit_merged(k: int) -> float:
        """Every device interval that starts inside unit k, merged: device busy for that step."""
        spans = [iv for i in units[k] for iv in per_launch.get(i, ())]
        spans.extend(outside_by_unit.get(k, ()))
        return merge(spans)

    merged_unit = unit_merged(0) if units else merged_all
    recipe_less_unit = merge(outside_by_unit.get(0, ()))
    # All units are nominally the same step measured k times, so the mean is the better
    # estimator than unit 0; unit 0 is printed beside it for comparison.
    n_units = max(len(units), 1)
    merged_unit_mean = sum(unit_merged(k) for k in range(len(units))) / n_units
    recipe_less_mean = (
        sum(merge(outside_by_unit.get(k, ())) for k in range(len(units))) / n_units
    )
    step_union_mean = sum(unit_union) / n_units if unit_union else 0.0
    # A trace that lost events to buffer overflow shows up as fewer launches than the model
    # issued. Two independent checks: the units must partition the launches exactly, and every
    # interval that carries a recipe_id must have found its launch.
    expected_units = None
    config_path = args.trace.with_name(
        args.trace.name.replace(".tsv", "-config.json").replace(".json.gz", "-config.json")
    )
    if config_path.exists():
        try:
            expected_units = json.loads(config_path.read_text()).get("passes")
        except (OSError, ValueError):
            expected_units = None
    launches_cover = len(units) * period == len(launches) if launches else False

    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(f"# device trace union — {args.label or args.trace.name}")
    emit(f"source: {args.trace}")
    if args.rank is not None:
        emit(f"rank: {args.rank} (one trace = one device; never summed across ranks)")
    emit("")
    emit("## pre-flight (a trace that fails this is not a measurement)")
    emit(f"rows read: {len(rows)}  phases: {dict(phases.most_common())}")
    emit(f"categories: {dict(categories.most_common(12))}")
    labels = {}
    label_of: dict[int, str] = {}
    for pid, label in names.items():
        label_of.setdefault(engine_of(pid), label)
    for engine, count in Counter(i[0] for i in intervals).most_common():
        label = label_of.get(engine) or _ENGINES.get(engine, "?")
        labels[engine] = label
        emit(f"  engine {engine:<5} {label:<24} intervals={count}")
    emit(
        f"B/E pairs: {len(intervals)}   unmatched B: {unmatched_begin}   unmatched E: {unmatched_end}"
        f"   (of which paired on (engine, index) alone, because the two halves carry different"
        f" names: {fallback})"
    )
    emit(f"rows carrying a recipe_id: {rid_rows}   Recipe Start rows: {len(launches)}")
    emit(f"device events present: {'YES' if intervals else 'NO'}")
    emit("")
    emit("## truncation check (the only protection against reading a buffer overflow)")
    emit(
        f"  Recipe Start rows = {len(launches)};  units {len(units)} x period {period}"
        f" = {len(units) * period}  -> {'PASS' if launches_cover else 'FAIL'}"
    )
    if expected_units is not None:
        emit(
            f"  passes recorded in the trace config = {expected_units},"
            f" units detected = {len(units)}  ->"
            f" {'PASS' if len(units) == expected_units else 'FAIL (truncated or mis-split)'}"
        )
    else:
        emit("  passes in the trace config: no <case>-config.json beside this trace")
    emit(
        f"  intervals carrying a recipe_id that found no launch = {unattributed}"
        f"  -> {'PASS' if unattributed == 0 else 'FAIL (lost Recipe Start rows)'}"
    )
    # The buffer size is a config value, so read it rather than assume it: the device refused
    # 1 GiB outright (scal_allocate_buffer rc -5) once the KV pool was reserved, so the
    # working size is a property of this run.
    buffer_bytes = 0x8000000
    buffer_note = "default assumption"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            raw = cfg.get("trace_buffer_size") or ""
            if isinstance(raw, str) and raw:
                buffer_bytes = int(raw, 0)
                buffer_note = f"from {config_path.name}"
        except (OSError, ValueError):
            pass
    capacity_events = max(buffer_bytes // 64, 1)
    emit(
        f"  events observed = {len(rows)}; buffer {buffer_bytes / 2**20:.0f} MiB at ~64 B/event"
        f" holds ~{capacity_events} ({buffer_note})"
        f"  -> {100 * len(rows) / capacity_events:.1f}% full"
    )
    if len(rows) > capacity_events:
        emit("  WARNING: more events than the buffer holds -> this trace may be TRUNCATED")
    emit("")
    emit("## the step, three numbers on the same scope (mean over the repeating units)")
    emit(
        f"  (a) merged union over every device interval = {merged_unit_mean:10.1f} us"
        "   <- device busy"
    )
    emit(
        f"  (b) sum of per-recipe unions               = {step_union_mean:10.1f} us"
        "   <- union within a recipe, sum across recipes"
    )
    emit(
        f"  (c) recipe-less device work (a) minus (b)  = {merged_unit_mean - step_union_mean:10.1f} us"
        f"   = {100 * (merged_unit_mean - step_union_mean) / merged_unit_mean:.1f}% of (a)"
    )
    emit(
        f"      of which intervals with recipe_id 0 (host copies, barriers)"
        f" = {recipe_less_mean:10.1f} us"
        f"   = {100 * recipe_less_mean / merged_unit_mean:.1f}% of (a)"
    )
    if len(units) > 1:
        emit(
            f"      unit 0 alone: (a) {merged_unit:.1f} us, (b) {step_union:.1f} us,"
            f" recipe-less {recipe_less_unit:.1f} us"
        )
    span_mean = sum(unit_span) / n_units if unit_span else step_span
    emit(f"  step span (unit 0, first start -> last end) = {step_span:10.1f} us")
    emit(f"  step span (mean over the units)             = {span_mean:10.1f} us")
    # The ratio must divide like by like: both numbers are means over the units. Quoting a
    # mean union over unit 0's span (or the reverse) gives a fraction that no pair of printed
    # numbers reproduces, which is exactly the kind of figure a reader cannot re-check.
    emit(
        f"  union/span (mean union / mean span) = {step_union_mean / span_mean:.3f}"
        "   <- device busy fraction of the step's own span"
        if span_mean
        else "  union/span = n/a"
    )
    emit(
        f"  union/span (unit 0 union / unit 0 span) = {step_union / step_span:.3f}"
        if step_span
        else "  union/span = n/a"
    )
    emit(
        "  (b) + (c) can exceed (a): recipes overlap each other across engines, and (c) is a"
        " merge, so the two are not disjoint."
    )
    if len(units) > 1:
        emit(
            f"  per-unit (b): mean {sum(unit_union) / len(unit_union):.1f} us"
            f"  min {min(unit_union):.1f}  max {max(unit_union):.1f}  ({len(units)} units)"
        )
        emit(
            f"  per-unit (a): mean {sum(unit_merged(k) for k in range(len(units))) / len(units):.1f} us"
        )
    emit(f"  whole window: span {total_span:.1f} us, merged union {merged_all:.1f} us"
         f" ({merged_all / total_span:.3f} of window span)")
    emit("")
    emit("## nodes per recipe (per unit) — a recipe is several device nodes, not one")
    nodes_per_launch = sorted(len(per_launch.get(i, ())) for unit in units for i in unit)
    if nodes_per_launch:
        emit(
            f"  device intervals per launch: median {nodes_per_launch[len(nodes_per_launch) // 2]}"
            f"  min {nodes_per_launch[0]}  max {nodes_per_launch[-1]}"
            f"  over {len(nodes_per_launch)} launches"
        )
        emit(
            f"  launches per unit {period}, so one unit is ~{period} recipes x ~"
            f"{sum(nodes_per_launch) // len(nodes_per_launch)} device nodes"
            f" = ~{sum(nodes_per_launch) // len(units)} node intervals"
        )
    per_launch_engines = [
        len(per_launch_engine.get(i, {})) for unit in units for i in unit
    ]
    if per_launch_engines:
        emit(
            f"  engine streams per launch: median {sorted(per_launch_engines)[len(per_launch_engines) // 2]}"
            f"  max {max(per_launch_engines)}"
        )
    emit("")
    emit("## window")
    emit(f"span (all device intervals): {total_span:.1f} us")
    emit(
        f"launches: {len(launches)}   repeating unit: {period} launches ({detected})"
        f"   units in trace: {len(units)}"
    )
    if outside_union:
        emit(
            f"outside any recipe over the whole window (recipe_id 0)"
            f" = {outside_union:10.1f} us"
        )
    emit("")
    emit("## by engine type (sum over recipes of that recipe's per-engine union)")
    engine_totals: dict[int, float] = defaultdict(float)
    for unit in units:
        for i in unit:
            for engine, iv in per_launch_engine.get(i, {}).items():
                engine_totals[engine] += merge(iv)
    per_unit_engine = {
        engine: total / len(units) for engine, total in engine_totals.items()
    }
    for engine, total in sorted(per_unit_engine.items(), key=lambda kv: -kv[1]):
        share = 100 * total / step_union if step_union else 0.0
        emit(
            f"  {labels.get(engine, _ENGINES.get(engine, '?')):<12}"
            f" {total:10.1f} us  {share:5.1f}% of union"
        )
    emit("")
    emit("## by op family (sum over recipes; a family can span engines, so families overlap)")
    family_totals: dict[str, float] = defaultdict(float)
    for unit in units:
        for i in unit:
            for fam, iv in per_launch_family.get(i, {}).items():
                family_totals[fam] += merge(iv)
    per_unit_family = {
        fam: total / len(units) for fam, total in family_totals.items()
    }
    for fam, total in sorted(per_unit_family.items(), key=lambda kv: -kv[1]):
        share = 100 * total / step_union if step_union else 0.0
        emit(f"  {fam:<18} {total:10.1f} us  {share:5.1f}% of union")
    emit("")
    emit(f"## top {args.top} recipes by union (per unit; one launch = one recipe)")
    recipe_totals: dict[tuple[int, str], float] = defaultdict(float)
    for unit in units:
        for i in unit:
            key = (launches[i][2], launch_rname.get(i, ""))
            recipe_totals[key] += launch_union.get(i, 0.0)
    for (rid, rname), total in sorted(
        recipe_totals.items(), key=lambda kv: -kv[1]
    )[: args.top]:
        per = total / n_units
        share = 100 * per / step_union if step_union else 0.0
        label = rname or f"recipe_id {rid}"
        emit(f"  {per:9.1f} us  {share:5.1f}%  [{family(rname)}] {label}  (id {rid})")
    emit("")
    emit(
        "## per recipe id (one step): multiplicity, union, dominant node"
        "  <- how many launches each recipe contributes"
    )
    unit0 = list(units[0]) if units else list(range(len(launches)))
    by_rid: dict[int, list[int]] = defaultdict(list)
    for i in unit0:
        by_rid[launches[i][2]].append(i)
    emit(
        f"  one step = {len(unit0)} launches of {len(by_rid)} distinct recipes"
        f" (a recipe is launched many times: once per layer)"
    )
    for rid, idxs in sorted(
        by_rid.items(),
        key=lambda kv: -sum(launch_union.get(i, 0.0) for i in kv[1]),
    ):
        total = sum(launch_union.get(i, 0.0) for i in idxs)
        names: dict[str, float] = defaultdict(float)
        for i in idxs:
            for node, iv in per_launch_name.get(i, {}).items():
                names[node] += merge(iv)
        top = sorted(names.items(), key=lambda kv: -kv[1])[:2]
        dom = top[0][0] if top else "?"
        emit(
            f"  id {rid:<6} launches {len(idxs):3d}  union/step {total:9.1f} us"
            f"  per-launch {total / len(idxs):7.1f} us  dominant {dom}"
            f"  [{family(dom)}]"
        )
    emit("")
    emit(
        f"## top {args.top} node names by union (per unit; the union is per node, not per launch)"
    )
    name_totals: dict[str, float] = defaultdict(float)
    for unit in units:
        for i in unit:
            for name, iv in per_launch_name.get(i, {}).items():
                name_totals[name] += merge(iv)
    per_unit_name = {name: total / len(units) for name, total in name_totals.items()}
    for name, total in sorted(per_unit_name.items(), key=lambda kv: -kv[1])[: args.top]:
        share = 100 * total / step_union if step_union else 0.0
        emit(f"  {total:9.1f} us  {share:5.1f}%  [{family(name)}] {name}")
    emit("")
    emit("## name -> family mapping used")
    mapping: dict[str, set[str]] = defaultdict(set)
    for iv in intervals:
        name = iv[2]
        mapping[family(name)].add(name)
    for fam in sorted(mapping, key=lambda f: -len(mapping[f])):
        sample = sorted(mapping[fam])[:3]
        emit(f"  {fam:<18} {len(mapping[fam]):3d} names  e.g. {', '.join(sample)}")

    if args.compare is not None:
        emit("")
        emit(f"## substitution check against {args.compare}")
        try:
            other = summarize(args.compare)
        except (OSError, ValueError) as error:
            emit(f"  could not read it: {error}")
        else:
            here = {
                "recipe_ids": sorted({ts_rid for _ts, _dur, ts_rid in launches}),
                "recipe_names": sorted(
                    {n for n in launch_rname.values() if n}
                ),
                "median_nodes": nodes_per_launch[len(nodes_per_launch) // 2]
                if nodes_per_launch
                else 0,
                "engines": sorted(labels),
                "launches": len(launches),
                "path": args.trace,
            }
            emit(
                f"  launches: this {here['launches']} (period {period}),"
                f" other {other['launches']}"
            )
            emit(
                f"  recipe ids: this {len(here['recipe_ids'])},"
                f" other {len(other['recipe_ids'])}"
                f"  -> identical set:"
                f" {'YES' if here['recipe_ids'] == other['recipe_ids'] else 'NO'}"
            )
            if here["recipe_ids"] != other["recipe_ids"]:
                only_here = sorted(set(here["recipe_ids"]) - set(other["recipe_ids"]))
                only_other = sorted(set(other["recipe_ids"]) - set(here["recipe_ids"]))
                emit(f"    only here: {only_here[:20]}")
                emit(f"    only there: {only_other[:20]}")
            emit(
                f"  recipe names: this {len(here['recipe_names'])},"
                f" other {len(other['recipe_names'])}"
            )
            emit(
                f"  device intervals per launch (median): this {here['median_nodes']},"
                f" other {other['median_nodes']}"
                f"  -> {'MATCH' if here['median_nodes'] == other['median_nodes'] else 'DIFFER'}"
            )
            emit(
                f"  engine types: this {here['engines']}, other {other['engines']}"
                f"  -> {'MATCH' if here['engines'] == other['engines'] else 'DIFFER'}"
            )
            emit(
                "  a substitution is valid only if the recipe set and the node counts agree;"
                " the union itself is then read at the substituted context, not at 512k."
            )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
