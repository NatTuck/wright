"""Grade a device trace's launches against their declared floors.

The launch log gives host cadence; this reads the *device* timeline (``V``/``Recipe Start`` rows carry a
device ``ts`` and ``dur`` per launch) and labels each launch by the nodes inside its window, because the trace
has no recipe name. Then, for each family with a declared floor:

    ratio = device time of the family's launches / (launches x bytes per launch / envelope)

That is the ranking the plan needs: measured device time against the bytes the work must move at the full
envelope, with no small-tile allowance.

**Read the ratios as upper bounds.** A traced run is slower than a clean one (per-node device markers cost
time): the 4k prefill measures ~0.78 s per step traced against ~0.39 s clean, so every ratio here is inflated
by roughly 2x. The ranking is what is trustworthy, not the absolute number.

usage: span_grade.py TRACE.tsv [--top N] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.span_floors import TPC_BYTES_PER_S, derivation, floor  # noqa: E402
from scripts.trace_union import device_intervals, rows  # noqa: E402

# What a launch's nodes say about which recipe it was. Order matters: the first marker that matches names it.
MARKERS = (
    ("ds41_attn_out_tile", ("rg_ds_attn_online", "rg_ds_attn_acc")),
    ("ds41_ix_chain", ("rg_ds_ix_merge", "rg_ds_ix_thr", "rg_ds_ix_bcnt")),
    ("ds41_ffn_paged", ("rg_moe_asg", "rg_moe_pages", "rg_ds_xact_lin", "rg_ds_xcomb")),
    ("ds41_attn_out_wo", ("rg_ds_xact_lin",)),
)


def classify(nodes: set[str]) -> str | None:
    for family, needles in MARKERS:
        if any(needle in node for node in nodes for needle in needles):
            return family
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    launches = [
        (float(row["ts"]), float(row["dur"]))
        for row in rows(args.trace)
        if row["ph"] == "V" and row["category"] == "Recipe Start"
    ]
    launches.sort()
    intervals = device_intervals(args.trace, "preferred")
    print(f"trace {args.trace}: {len(launches)} launches, {len(intervals)} device intervals")

    # Which nodes ran inside each launch window.
    labels: list[set[str]] = [set() for _ in launches]
    starts = [ts for ts, _ in launches]
    import bisect

    for _, name, start, _end in intervals:
        at = bisect.bisect_right(starts, start) - 1
        if at >= 0:
            labels[at].add(name)

    per: dict[str, dict] = defaultdict(lambda: {"launches": 0, "us": 0.0, "names": set()})
    unknown = 0
    unknown_us = 0.0
    for (ts, dur), nodes in zip(launches, labels):
        family = classify(nodes)
        if family is None:
            unknown += 1
            unknown_us += dur
            continue
        entry = per[family]
        entry["launches"] += 1
        entry["us"] += dur
        entry["names"].update(nodes)

    table = []
    for family, entry in per.items():
        declared = floor(f"{family}_T2048")
        if declared is None:
            declared = floor(f"{family}_T512")
        if declared is None:
            continue
        per_launch, regime, rate = declared
        moved = per_launch * entry["launches"]
        floor_us = moved / rate * 1e6
        table.append(
            {
                "family": family,
                "launches": entry["launches"],
                "device_ms": entry["us"] / 1e3,
                "bytes": moved,
                "floor_ms": floor_us / 1e3,
                "ratio": entry["us"] / floor_us if floor_us else None,
                "gbs": moved / (entry["us"] * 1e-6) / 1e9 if entry["us"] else None,
                "derivation": derivation(f"{family}_T2048"),
            }
        )
    table.sort(key=lambda r: -(r["ratio"] or 0))

    print(f"\n{'ratio':>8} {'device ms':>10} {'floor ms':>9} {'GB/s':>8} {'launches':>9}  family")
    for r in table[: args.top]:
        print(
            f"{r['ratio']:8.1f} {r['device_ms']:10.1f} {r['floor_ms']:9.2f} {r['gbs']:8.1f} "
            f"{r['launches']:9d}  {r['family']}"
        )
    graded_us = sum(r["device_ms"] * 1e3 for r in table)
    total_us = sum(dur for _, dur in launches)
    print(
        f"\nclassified {len(launches) - unknown}/{len(launches)} launches "
        f"({100 * graded_us / total_us:.0f}% of launch time graded, "
        f"{100 * unknown_us / total_us:.0f}% unclassified)"
    )
    if args.json:
        args.json.write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
