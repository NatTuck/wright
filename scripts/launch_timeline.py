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
import statistics
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.span_floors import TPC_BYTES_PER_S, derivation, family, floor


@dataclass
class SpanRow:
    """One span's line in the cadence table: measured host cadence, and the declared floor for reference."""

    group: str
    launches: int
    period_ms: float
    bytes: float | None = None
    regime: str | None = None
    floor_ms: float | None = None
    ratio: float | None = None
    gbs: float | None = None


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


def spans(rows: list[tuple[float, int, str]], t0: float, args) -> None:
    """Every span against its declared floor, ranked by how far above it the span runs.

    Two span kinds, because a launch log holds host timestamps and nothing else:

    * **per family**: the *median period* between consecutive launches of that family, against the bytes one
      launch must move. A family launched once per layer then reports the layer period, which is its cost.
      The median is what makes this robust: the handful of ~780 ms between-sample holes in a bench run cannot
      move it, and they are reported separately as gaps.
    * **window** (``--between START END``): an explicit multi-launch span, from the first launch whose name
      contains START to the first after it whose name contains END. Its floor is the sum of every launch's
      own floor inside the window, so a 100-launch chunk is judged as a whole.

    **What these numbers are, and are not.** The log holds *host* timestamps, so a period is host cadence and
    a window is host elapsed time. Within a step the host races ahead of the device (it issues a chunk's ~100
    launches in ~130 ms while the device takes ~390 ms to run them) and then blocks at the next step's first
    launch, which is the ~780 ms holes. So a host window is **not** a device span and must not be divided into
    a device byte floor: the ratio would flatter the span by however far the host ran ahead. The floors are
    printed as a reference, and the ratio is printed only for the *gap* spans (host work with a host floor),
    never for a device span.
    """
    per: dict[str, list[float]] = defaultdict(list)
    for ns, _, name in rows:
        per[family(name) if args.spans == "family" else name].append(ns / 1e6)

    table: list[SpanRow] = []
    for group, times in per.items():
        times.sort()
        periods = [b - a for a, b in zip(times, times[1:])]
        period = statistics.median(periods) if periods else 0.0
        row = SpanRow(group=group, launches=len(times), period_ms=period)
        declared = floor(group)
        if declared is not None:
            per_launch, regime, rate = declared
            row.bytes = per_launch
            row.regime = regime
            row.floor_ms = per_launch / rate * 1e3
            row.ratio = period / row.floor_ms if row.floor_ms > 0 else None
            row.gbs = per_launch / (period / 1e3) / 1e9 if period > 0 else None
        table.append(row)

    table.sort(key=lambda r: -r.launches)

    print("\nhost cadence per span (host timestamps: NOT device time, so no ratio against a device floor):")
    print(f"{'period ms':>10} {'launches':>9} {'MB/launch':>10} {'floor ms':>9}  span")
    for r in table[: args.top_spans]:
        mb = "-" if r.bytes is None else f"{r.bytes / 1e6:.3f}"
        fl = "-" if r.floor_ms is None else f"{r.floor_ms:.4f}"
        note = "" if r.floor_ms is not None else f"   [{derivation(r.group)}]"
        print(f"{r.period_ms:10.3f} {r.launches:9d} {mb:>10} {fl:>9}  {r.group}{note}")

    for start, end in args.between:
        window(rows, t0, start, end)


def window(rows: list[tuple[float, int, str]], t0: float, start: str, end: str) -> None:
    """One multi-launch span: first launch containing ``start`` to the first following one containing ``end``."""
    first = next((i for i, (_, _, n) in enumerate(rows) if start in n), None)
    if first is None:
        print(f"\nwindow {start!r}..{end!r}: no launch matches {start!r}")
        return
    last = next((i for i in range(first, len(rows)) if end in rows[i][2]), None)
    if last is None:
        print(f"\nwindow {start!r}..{end!r}: no launch matches {end!r} after the start")
        return
    launches = rows[first : last + 1]
    measured_ms = (rows[last][0] - rows[first][0]) / 1e6
    moved = 0.0
    unfloored = 0
    for _, _, name in launches:
        declared = floor(name)
        if declared is None:
            unfloored += 1
        else:
            moved += declared[0]
    floor_ms = moved / TPC_BYTES_PER_S * 1e3
    print(
        f"\nwindow {start!r}..{end!r}: {len(launches)} launches, {measured_ms:.1f} ms of HOST elapsed "
        f"({unfloored} launches with no floor declared)"
    )
    print(
        f"  its launches declare {moved / 1e6:.1f} MB, a {floor_ms:.2f} ms device floor at the full envelope."
        f"  Host elapsed is not device time, so this is not a ratio: it says how far the host ran ahead."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rank", type=int, help="the pid to report (default: the most launches)")
    parser.add_argument("--top", type=int, default=25, help="recipes to list")
    parser.add_argument("--gap-us", type=float, default=200.0, help="report host gaps at least this long")
    parser.add_argument(
        "--spans",
        choices=("family", "recipe"),
        help="also compare spans against their declared floors (scripts/span_floors.py)",
    )
    parser.add_argument("--top-spans", type=int, default=20, help="spans to list")
    parser.add_argument(
        "--between",
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="also measure one multi-launch span, from the first launch whose name contains START to the "
        "first following one containing END (repeatable)",
    )
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

    if args.spans:
        spans(rows, t0, args)

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
