"""The launch timeline: every recipe launch with its name, from ``RESIN_LAUNCH_LOG``.

``runtime.cpp`` appends ``(ticks, recipe index)`` per launch to a preallocated buffer — a TSC read and one
relaxed store, no sync and no device query — and dumps ``<dir>/launches.<pid>.tsv`` at exit. This reads those
files: the host-side launch timeline, with identities, at the run's own speed.

What it answers: per recipe, when it launched and how often; and where the *host* stopped launching. A gap
between consecutive launches is host time the device was not being fed (a fill, a D2H wait, an allocator),
which the bench's ``step_seconds`` folds into the step and cannot separate.

It does not give device durations — a launch's cost on the device is the profiler's business
(``scripts/trace_union.py``). Two instruments, two questions.

usage: launch_timeline.py DIR [--rank PID] [--top N] [--gap-us US] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> list[tuple[float, int, str]]:
    """``(ns, recipe, name)`` per launch, in order."""
    out = []
    with path.open() as source:
        for line in source:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            out.append((float(parts[0]), int(parts[2]), parts[3]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rank", type=int, help="the pid to report (default: the most launches)")
    parser.add_argument("--top", type=int, default=25, help="recipes to list")
    parser.add_argument("--gap-us", type=float, default=200.0, help="report host gaps at least this long")
    parser.add_argument("--json", type=Path, help="also write the table here")
    args = parser.parse_args()

    files = sorted(args.directory.glob("launches.*.tsv"))
    if not files:
        raise SystemExit(f"no launches.*.tsv in {args.directory}")
    per_file = {int(p.stem.split(".")[1]): read(p) for p in files}
    pid = args.rank if args.rank is not None else max(per_file, key=lambda k: len(per_file[k]))
    rows = per_file[pid]
    if not rows:
        raise SystemExit(f"{pid}: no launches")
    t0 = rows[0][0]
    end = rows[-1][0]
    span = (end - t0) / 1e9
    print(f"pid {pid}: {len(rows)} launches over {span:.3f} s  ({len(rows) / span:.0f} launches/s)")
    if len(per_file) > 1:
        print(f"  ({len(per_file)} rank files: " + ", ".join(f"{k}:{len(v)}" for k, v in sorted(per_file.items())) + ")")

    per_recipe: dict[str, list[float]] = defaultdict(list)
    for ns, _, name in rows:
        per_recipe[name].append((ns - t0) / 1e6)
    table = sorted(per_recipe.items(), key=lambda item: -len(item[1]))
    print(f"\n{'launches':>9} {'share':>7} {'ms/launch':>10}  recipe")
    for name, times in table[: args.top]:
        gap = (times[-1] - times[0]) / max(len(times) - 1, 1)
        print(f"{len(times):9d} {100 * len(times) / len(rows):6.1f}% {gap:10.3f}  {name}")
    if len(table) > args.top:
        rest = sum(len(t) for _, t in table[args.top :])
        print(f"{rest:9d} {100 * rest / len(rows):6.1f}%   (remaining {len(table) - args.top} recipes)")

    gaps = []
    for (a_ns, _, a_name), (b_ns, _, b_name) in zip(rows, rows[1:]):
        gap = (b_ns - a_ns) / 1000.0
        if gap >= args.gap_us:
            gaps.append((gap, (a_ns - t0) / 1e6, a_name, b_name))
    gaps.sort(reverse=True)
    print(f"\nhost gaps >= {args.gap_us:.0f} us: {len(gaps)}")
    for gap, at, a_name, b_name in gaps[:20]:
        print(f"  {gap:9.1f} us at {at:9.3f} ms   after {a_name[:44]:44} before {b_name[:44]}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "launches": len(rows),
                    "span_s": span,
                    "recipes": [
                        {
                            "name": name,
                            "launches": len(times),
                            "first_ms": times[0],
                            "last_ms": times[-1],
                        }
                        for name, times in table
                    ],
                    "gaps": [
                        {"us": gap, "at_ms": at, "after": a, "before": b}
                        for gap, at, a, b in gaps
                    ],
                },
                indent=1,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
