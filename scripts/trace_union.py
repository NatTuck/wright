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
    intervals = []  # (engine, index, name, rid, start, end)
    unmatched_end = 0
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
                    (key[0], key[1], key[2], begin.rid, begin.ts, row.ts)
                )
            else:
                unmatched_end += 1
        elif row.ph == "X" and row.dur > 0:
            intervals.append(
                (engine_of(row.engine), row.index, row.name, row.rid, row.ts, row.ts + row.dur)
            )
    unmatched_begin = sum(len(stack) for stack in stacks.values())

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
    for engine, _index, name, rid, start, end in intervals:
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
    all_intervals = [(start, end) for *_rest, start, end in intervals]
    total_span = max(e for _s, e in all_intervals) - min(s for s, _e in all_intervals)
    step_union = unit_union[0] if unit_union else 0.0
    step_span = unit_span[0] if unit_span else total_span
    merged_all = merge(all_intervals)
    outside_union = merge(outside)
    rid_rows = sum(1 for r in rows if r.rid and r.cat != "Host API")

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
    )
    emit(f"rows carrying a recipe_id: {rid_rows}   Recipe Start rows: {len(launches)}")
    emit(f"device events present: {'YES' if intervals else 'NO'}")
    emit("")
    emit("## window")
    emit(f"span (all device intervals): {total_span:.1f} us")
    emit(
        f"launches: {len(launches)}   repeating unit: {period} launches ({detected})"
        f"   units in trace: {len(units)}"
    )
    emit(
        "accounting: union within a recipe, sum across recipes (recipes are serial)"
    )
    emit(f"  step union   (sum of per-recipe unions)  = {step_union:10.1f} us")
    emit(f"  step span    (first start -> last end)   = {step_span:10.1f} us")
    emit(
        f"  union/span   = {step_union / step_span:.3f}"
        if step_span
        else "  union/span   = n/a"
    )
    emit(
        f"  merged union across every interval       = {merged_all:10.1f} us"
        f"   ({merged_all / total_span:.3f} of window span; diagnostic, cross-recipe overlap"
        " counted once)"
    )
    if outside_union:
        emit(
            f"  outside any recipe (recipe_id 0: host copies, barriers)"
            f" = {outside_union:10.1f} us"
        )
    if unattributed:
        emit(f"  intervals with a recipe_id but no launch marker = {unattributed}")
    if len(units) > 1:
        emit(
            f"  per-unit union: mean {sum(unit_union) / len(unit_union):.1f} us"
            f"  min {min(unit_union):.1f}  max {max(unit_union):.1f}"
            f"  ({len(units)} units)"
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
    emit(f"## top {args.top} recipes by union (per unit)")
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
    for _engine, _index, name, *_rest in intervals:
        mapping[family(name)].add(name)
    for fam in sorted(mapping, key=lambda f: -len(mapping[f])):
        sample = sorted(mapping[fam])[:3]
        emit(f"  {fam:<18} {len(mapping[fam]):3d} names  e.g. {', '.join(sample)}")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
