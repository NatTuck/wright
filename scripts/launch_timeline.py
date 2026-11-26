"""The launch timeline: every recipe launch with its name, from ``RESIN_LAUNCH_LOG``.

``runtime.cpp`` writes one 16-byte record per launch straight into a memory-mapped file —
``<dir>/launches.<pid>.bin``: a 48-byte header (magic, version, record size, ns-per-tick, origin, capacity)
then ``capacity`` records of ``(ticks u64, recipe u32, pad u32)``. Writing into the mapping rather than
buffering for a dump is deliberate: a bench worker leaves through ``os._exit``, which skips ``atexit``, and a
crash would lose the run — mapped pages are already the file.

Per launch the cost is a TSC read and one 16-byte store: no sync, no device query, no allocation, no lock, no
syscall. That is the difference from ``RESIN_STEP_PROF``, which syncs after every launch and so destroys the
overlap a timeline exists to show.

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
import struct
from collections import defaultdict
from pathlib import Path

HEADER = struct.Struct("<8sIIdQQ")
RECORD = struct.Struct("<QII")
MAGIC = b"RGLAUNCH"


def read(path: Path) -> tuple[list[tuple[float, int, str]], dict]:
    """``(ns, recipe, name)`` per launch in order, plus the header, from one ``launches.<pid>.bin``."""
    blob = path.read_bytes()
    if len(blob) < HEADER.size:
        raise SystemExit(f"{path}: not a launch log ({len(blob)} bytes, header is {HEADER.size})")
    magic, version, entry, ns_per_tick, _origin, capacity = HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: not a launch log (magic {magic!r})")
    if version != 1 or entry != RECORD.size:
        raise SystemExit(f"{path}: version {version} / record {entry} bytes, expected 1 / {RECORD.size}")
    names = names_for(path)
    out = []
    at = HEADER.size
    for _ in range(capacity):
        if at + RECORD.size > len(blob):
            break
        ticks, ri, _pad = RECORD.unpack_from(blob, at)
        at += RECORD.size
        if ticks == 0:
            break  # records are written in order and never cleared, so the first zero ends the log
        out.append((ticks * ns_per_tick, ri, names.get(ri, f"recipe-{ri}")))
    return out, {
        "version": version,
        "ns_per_tick": ns_per_tick,
        "capacity": capacity,
        "written": len(out),
    }


def names_for(path: Path) -> dict[int, str]:
    """``recipe -> name`` from the ``names.<pid>.tsv`` the device writes beside the log, if present."""
    side = path.with_name(path.name.replace("launches.", "names.").replace(".bin", ".tsv"))
    names: dict[int, str] = {}
    if not side.exists():
        return names
    for line in side.read_text().splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            names[int(parts[0])] = parts[1]
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rank", type=int, help="the pid to report (default: the most launches)")
    parser.add_argument("--top", type=int, default=25, help="recipes to list")
    parser.add_argument("--gap-us", type=float, default=200.0, help="report host gaps at least this long")
    parser.add_argument("--json", type=Path, help="also write the table here")
    args = parser.parse_args()

    files = sorted(args.directory.glob("launches.*.bin"))
    if not files:
        raise SystemExit(f"no launches.*.bin in {args.directory}")
    per_file = {}
    headers = {}
    for path in files:
        rows, header = read(path)
        per_file[int(path.stem.split(".")[1])] = rows
        headers[int(path.stem.split(".")[1])] = header
    pid = args.rank if args.rank is not None else max(per_file, key=lambda k: len(per_file[k]))
    rows = per_file[pid]
    if not rows:
        raise SystemExit(f"{pid}: no launches")
    t0 = rows[0][0]
    end = rows[-1][0]
    span = (end - t0) / 1e9
    rate = f"{len(rows) / span:.0f} launches/s" if span > 0 else "a single launch"
    print(
        f"pid {pid}: {len(rows)} launches over {span:.3f} s  ({rate})"
        f"   [{headers[pid]['ns_per_tick']:.4f} ns/tick, capacity {headers[pid]['capacity']}]"
    )
    if len(per_file) > 1:
        print(
            "  "
            + f"{len(per_file)} rank files: "
            + ", ".join(f"{k}:{len(v)}" for k, v in sorted(per_file.items()))
        )

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
    gaps.sort(key=lambda item: -item[0])
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
