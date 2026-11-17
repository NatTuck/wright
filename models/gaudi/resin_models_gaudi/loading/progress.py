"""Startup progress of a Wright server: every rank of every model, every phase.

Each process reports through ``progress`` (this module's ``Progress``): ``phase`` starts a
phase with its total, ``advance`` adds done units and ``note`` sets a short label. The device
reports uploaded weight bytes, the kernel build reports compiled kernels, and the recipe cache
reports the graphs that it built, compiled or loaded (``graph_built``, ``recipes_planned``,
``recipe_done``).

Records live on a ``Board``: a small shared file with one fixed record per (model, rank), so
the ranks of a multi-card model report to one place without a collective. Each rank writes only
its own record. One process (rank 0, or the only process) runs a ``Renderer``: a live block on a
terminal, or otherwise a log line per phase change and every 30 seconds.

Weight totals are estimates from file sizes on a first load, so the bar stops at 99% until the
phase ends. Each rank then saves the bytes it uploaded under a key for the checkpoint and the
options, and the next load of the same checkpoint uses that exact total.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("wright")
info = logger.info

PHASES = ("kernels", "weights", "recipes", "warm-up", "ready")
_UNITS = {"recipes": "graphs", "kernels": "kernels"}
_LOG_SECONDS = 30.0

RECORD = np.dtype(
    [
        ("seq", "<u8"),
        ("phase", "<u4"),
        ("exact", "<u4"),
        ("done", "<u8"),
        ("total", "<u8"),
        ("note", "S64"),
        ("graphs", "<u4"),
        ("to_compile", "<u4"),
        ("compiled", "<u4"),
        ("shared", "<u4"),
        ("cached", "<u4"),
        ("compile_ms", "<u8"),
        ("loaded", "<u8"),
    ]
)
"""One rank's record. ``phase`` is 1 + an index of ``PHASES`` (0: not started). ``seq`` is odd
while the writer changes the record. The recipe counts: ``graphs`` built before the compile,
``to_compile`` of them not in the cache when the compile began, then ``compiled`` here,
``shared`` (compiled by another process while this one waited) and ``cached``; ``compile_ms``
is the time of this rank's own compiles. ``loaded``: the weight bytes, kept when the weights
phase ends."""


def weight_file_bytes(directory: str | Path, pattern: str) -> int:
    return sum(path.stat().st_size for path in Path(directory).glob(pattern))


class Board:
    """The records of ``models`` x ``ranks`` processes, in memory or in a file (``path``) that
    several processes map."""

    def __init__(
        self, ranks: int = 1, models: int = 1, path: str | Path | None = None
    ) -> None:
        self.ranks, self.models = ranks, models
        nbytes = RECORD.itemsize * ranks * models
        self.path = None if path is None else Path(path)
        if self.path is None:
            self._buffer: bytearray | mmap.mmap = bytearray(nbytes)
        else:
            if not self.path.exists() or self.path.stat().st_size != nbytes:
                with open(self.path, "wb") as f:
                    f.write(bytes(nbytes))
            with open(self.path, "r+b") as f:
                self._buffer = mmap.mmap(f.fileno(), nbytes)
        self.records = np.frombuffer(self._buffer, RECORD).reshape(models, ranks)

    def write(self, model: int, rank: int, **fields) -> None:
        record = self.records[model, rank]
        record["seq"] += 1
        for name, value in fields.items():
            record[name] = value
        record["seq"] += 1

    def read(self, model: int, rank: int) -> np.void:
        """A consistent copy of one record."""
        record = self.records[model, rank]
        copy = record.copy()
        for _ in range(100):
            before = int(record["seq"])
            copy = record.copy()
            if before % 2 == 0 and int(record["seq"]) == before:
                break
        return copy


class Progress:
    """This process's reports: the record of one (model, rank) on a board."""

    def __init__(self) -> None:
        self.board = Board()
        self.model = self.rank = 0
        self.key: str | None = None
        self.cache_root: Path | None = None
        self._lock = threading.Lock()

    def attach(self, board: Board, rank: int = 0, model: int = 0) -> None:
        self.board, self.rank, self.model = board, rank, model

    def remember(self, root: str | Path, **identity) -> None:
        """Save and reuse exact weight totals under ``root`` for this checkpoint and options."""
        text = json.dumps(identity, sort_keys=True, default=str)
        self.key = hashlib.sha256(text.encode()).hexdigest()[:24]
        self.cache_root = Path(root)

    def _totals_file(self) -> Path | None:
        if self.key is None or self.cache_root is None:
            return None
        return self.cache_root / "load-totals" / f"{self.key}-r{self.rank}.json"

    def phase(self, name: str, total: int = 0, note: str = "") -> None:
        """Start phase ``name`` with ``total`` units to do (0: unknown)."""
        exact = 0
        if name == "weights":
            saved = self._saved_total()
            if saved:
                total, exact = saved, 1
        elif total:
            exact = 1
        with self._lock:
            record = self.board.records[self.model, self.rank]
            if self.current() == "weights":
                self.board.write(self.model, self.rank, loaded=int(record["done"]))
            self.board.write(
                self.model,
                self.rank,
                phase=PHASES.index(name) + 1,
                exact=exact,
                done=0,
                total=max(total, 0),
                note=note.encode()[:64],
            )

    def extend(self, units: int) -> None:
        """More units in the current phase, which are known exactly (not weight estimates)."""
        with self._lock:
            record = self.board.records[self.model, self.rank]
            exact = int(record["exact"]) or int(self.current() != "weights")
            self.board.write(
                self.model, self.rank, total=int(record["total"]) + units, exact=exact
            )

    def advance(self, units: int) -> None:
        with self._lock:
            record = self.board.records[self.model, self.rank]
            self.board.write(self.model, self.rank, done=int(record["done"]) + units)

    def note(self, text: str) -> None:
        with self._lock:
            self.board.write(self.model, self.rank, note=text.encode()[:64])

    def _add(self, **counts: int) -> None:
        with self._lock:
            record = self.board.records[self.model, self.rank]
            self.board.write(
                self.model, self.rank, **{k: int(record[k]) + v for k, v in counts.items()}
            )

    def building_graphs(self) -> None:
        """A model starts to build the graphs of its recipes: the recipes phase starts (or goes on)."""
        if self.current() == "recipes":
            self.note("")
        else:
            self.phase("recipes")

    def graph_built(self) -> None:
        """A graph for the recipes phase is built (counted only in that phase)."""
        if self.current() == "recipes":
            self._add(graphs=1)

    def recipes_planned(self, to_compile: int) -> None:
        """``to_compile`` more graphs of the recipes phase were not in the cache."""
        self._add(to_compile=to_compile)

    def recipe_done(self, outcome: str, seconds: float = 0.0) -> None:
        """One graph of the recipes phase is ready: ``outcome`` is ``compiled`` (here, in
        ``seconds``), ``shared`` (compiled by another process) or ``cached``."""
        counts = {outcome: 1, "done": 1}
        if outcome == "compiled":
            counts["compile_ms"] = int(seconds * 1000)
        self._add(**counts)

    def current(self) -> str | None:
        index = int(self.board.records[self.model, self.rank]["phase"])
        return PHASES[index - 1] if index else None

    def done(self) -> int:
        return int(self.board.records[self.model, self.rank]["done"])

    def finish_weights(self) -> None:
        """The weights phase ended: save its exact total for the next load."""
        done, path = self.done(), self._totals_file()
        if path is None or done == 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"bytes": done}))
            temporary.replace(path)
        except OSError as error:
            logger.debug(f"load totals not saved: {error}")

    def _saved_total(self) -> int:
        path = self._totals_file()
        try:
            return int(json.loads(path.read_text())["bytes"]) if path else 0
        except (OSError, ValueError, KeyError):
            return 0


progress = Progress()


# ------------------------------------------------------------------ rendering
def _gib(n: float) -> str:
    return f"{n / 2**30:.1f}"


def _rate(bytes_per_second: float) -> str:
    """A load rate: GiB/s, or MiB/s below 1 GiB/s."""
    if bytes_per_second >= 2**30:
        return f"{bytes_per_second / 2**30:.1f} GiB/s"
    return f"{bytes_per_second / 2**20:.0f} MiB/s"


def _duration(seconds: float) -> str:
    if seconds < 1:
        return "<1 s"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def _bar(fraction: float, width: int) -> str:
    filled = int(width * min(max(fraction, 0.0), 1.0))
    return "#" * filled + "-" * (width - filled)


def _sum(rows, field: str) -> int:
    return sum(int(r[field]) for r in rows)


def _recipes_status(rows, elapsed: float) -> list[str]:
    """The recipes phase: graphs built, then graphs ready, compiles done and left, the cache,
    and the time left of the slowest rank (from each rank's own compile rate). With several
    ranks the counts are per card."""
    ranks = len(rows)
    per = " per card" if ranks > 1 else ""
    total = _sum(rows, "total")
    if not total:
        return [f"building graphs: {_sum(rows, 'graphs') // ranks}{per}"]
    parts = [f"{_sum(rows, 'done') // ranks}/{total // ranks} graphs{per}"]
    to_compile = _sum(rows, "to_compile")
    resolved = _sum(rows, "compiled") + _sum(rows, "shared")
    if to_compile:
        compiles = f"{min(resolved, to_compile) // ranks}/{to_compile // ranks} compiled"
        if elapsed >= 1 and resolved:
            compiles += f" ({resolved / ranks / elapsed:.1f}/s)"
        parts.append(compiles)
    parts.append(f"{_sum(rows, 'cached') // ranks} from the cache")
    lefts = []
    for r in rows:
        planned, ready = int(r["to_compile"]), int(r["compiled"]) + int(r["shared"])
        if planned > ready:
            if not ready:
                lefts = []
                break
            lefts.append(elapsed * (planned - ready) / ready)
    if lefts and elapsed >= 1:
        parts.append(f"about {_duration(max(lefts))} left")
    return parts


def _summary(rows, took: dict[str, float]) -> str:
    """What each phase did and how long it took, for the ready line."""
    ranks = len(rows)
    parts = []
    for name in PHASES[:-1]:
        if name not in took:
            continue
        text = f"{name} {_duration(took[name])}"
        if name == "weights" and took[name] >= 1 and _sum(rows, "loaded"):
            per_card = _sum(rows, "loaded") / ranks
            text += f" ({_gib(per_card)} GiB per card, {_rate(per_card / took[name])})"
        elif name == "recipes":
            compiled = (_sum(rows, "compiled") + _sum(rows, "shared")) // ranks
            cached = _sum(rows, "cached") // ranks
            if compiled or cached:
                text += f" ({compiled} compiled, {cached} from the cache)"
        parts.append(text)
    return ", ".join(parts)


class _Model:
    """What the renderer saw of one model: when each phase started and ended."""

    def __init__(self) -> None:
        self.phase = 0
        self.started: dict[int, float] = {}
        self.ended: dict[int, float] = {}

    def see(self, phase: int, now: float) -> int:
        """Record a move to ``phase``; phases it skipped between two looks took no time."""
        before = self.phase
        if phase > before:
            for skipped in range(before, phase):
                if skipped:
                    self.started.setdefault(skipped, now)
                    self.ended[skipped] = now
            self.started[phase] = now
            self.phase = phase
        return before

    def took(self, phase: int) -> float:
        return self.ended.get(phase, 0.0) - self.started.get(phase, 0.0)


class Renderer:
    """Draws a board: ``names[m]`` names model ``m``. ``live``: redraw a block on ``stream``
    (a terminal); otherwise log a line per phase change and every 30 seconds. ``status``, if
    given, gets a one-line state of the load at most once a second (systemd's STATUS)."""

    def __init__(
        self,
        board: Board,
        names: list[str],
        *,
        stream=None,
        live: bool | None = None,
        clock=time.monotonic,
        log=info,
        status=None,
    ) -> None:
        self.board, self.names = board, names
        self.stream = stream if stream is not None else sys.stderr
        if live is None:
            live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.live = live
        self.clock, self.log = clock, log
        self.status = status
        self._status_at = float("-inf")
        self.started = clock()
        self._models = [_Model() for _ in names]
        self._logged = self.started
        self._drawn = 0
        self._paused_until = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._hooks: tuple | None = None

    # ---- the state of a model
    def _records(self, model: int) -> list[np.void]:
        return [self.board.read(model, rank) for rank in range(self.board.ranks)]

    @staticmethod
    def _fraction(rows) -> tuple[float, int, int, bool]:
        """(fraction, done, total, exact) over the ranks; an estimate stops at 99%."""
        total = sum(int(r["total"]) for r in rows)
        done = sum(min(int(r["done"]), int(r["total"])) for r in rows)
        exact = all(bool(r["exact"]) for r in rows)
        if not total:
            return 0.0, done, total, exact
        fraction = done / total
        return (fraction if exact else min(fraction, 0.99)), done, total, exact

    def _status(self, model: int, rows, phase: int) -> str:
        """The current phase in one line: amount, rate, time left, the slowest rank, notes."""
        name = PHASES[phase - 1]
        fraction, done, total, exact = self._fraction(rows)
        elapsed = self.clock() - self._models[model].started.get(phase, self.clock())
        ranks = len(rows)
        parts = []
        if total and name == "weights":
            parts.append(f"{_gib(done / ranks)}/{_gib(total / ranks)} GiB per card")
            if elapsed >= 1 and done:
                parts.append(_rate(done / ranks / elapsed))
                if fraction > 0.01:
                    left = _duration(elapsed * (1 / fraction - 1))
                    parts.append(
                        f"about {left} left" + ("" if exact else ", estimated")
                    )
        elif name == "recipes":
            parts += _recipes_status(rows, elapsed)
        elif total:
            parts.append(f"{done}/{total} {_UNITS.get(name, '')}".rstrip())
            if name == "kernels" and elapsed >= 1 and done and done < total - 1:
                left = elapsed * (total - done) / done
                parts.append(f"about {_duration(left)} left")
        if ranks > 1 and total:

            def share(rank: int) -> float:
                t = int(rows[rank]["total"])
                return int(rows[rank]["done"]) / t if t else 0.0

            slowest = min(range(ranks), key=share)
            parts.append(f"slowest rank {slowest} at {share(slowest):.0%}")
        notes = sorted({r["note"].decode(errors="replace") for r in rows} - {""})
        if len(notes) == 1:
            parts.append(notes[0])
        if not parts:
            parts.append(_duration(elapsed))
        # an unfinished phase never shows 100%
        shown = min(fraction, 0.99) if done < total else fraction
        head = f"{shown:4.0%}  " if total else ""
        return head + ", ".join(parts)

    # ---- output
    def lines(self) -> list[str]:
        """The live block."""
        out: list[str] = []
        now = self.clock()
        for model, name in enumerate(self.names):
            rows = self._records(model)
            seen = self._models[model]
            phase = min(int(r["phase"]) for r in rows)
            seen.see(phase, now)
            ranks = self.board.ranks
            out.append(name + (f" on {ranks} cards" if ranks > 1 else ""))
            for index, label in enumerate(PHASES[:-1], start=1):
                if index < phase:
                    out.append(f"  done  {label:<9} {_duration(seen.took(index))}")
                elif index == phase:
                    fraction, *_ = self._fraction(rows)
                    out.append(
                        f"  >     {label:<9} [{_bar(fraction, 24)}] "
                        + self._status(model, rows, phase)
                    )
                    if ranks > 1 and int(rows[0]["total"]):
                        out.append(
                            "                  "
                            + "  ".join(
                                f"{r} {_bar(int(row['done']) / int(row['total']) if int(row['total']) else 0, 5)}"
                                for r, row in enumerate(rows)
                            )
                        )
                else:
                    out.append(f"        {label}")
            if phase == len(PHASES):
                out.append(f"  {self._ready_text(model, rows, now)}")
        return out

    def _ready_text(self, model: int, rows, now: float) -> str:
        seen = self._models[model]
        took = {
            PHASES[index - 1]: seen.took(index)
            for index in range(1, len(PHASES))
            if index in seen.ended
        }
        summary = _summary(rows, took)
        return f"ready after {_duration(now - self.started)}" + (f": {summary}" if summary else "")

    def status_line(self) -> str:
        """The current phase of each model, in one line."""
        parts = []
        for model, name in enumerate(self.names):
            rows = self._records(model)
            phase = min(int(r["phase"]) for r in rows)
            prefix = f"{name}: " if len(self.names) > 1 else ""
            if phase == 0:
                parts.append(f"{prefix}starting")
            elif phase == len(PHASES):
                parts.append(f"{prefix}ready")
            else:
                parts.append(f"{prefix}{PHASES[phase - 1]}: {self._status(model, rows, phase)}")
        return " ".join("; ".join(parts).split())

    def _send_status(self) -> None:
        if self.status is None or self.clock() - self._status_at < 1.0:
            return
        self._status_at = self.clock()
        self.status(self.status_line())

    def clear(self) -> None:
        """Remove the live block from the terminal."""
        with self._lock:
            if self.live and self._drawn:
                self.stream.write(f"\x1b[{self._drawn}F\x1b[J")
                self.stream.flush()
                self._drawn = 0

    def draw(self) -> None:
        with self._lock:
            self._send_status()
            if not self.live:
                self._log_changes()
                return
            if self.clock() < self._paused_until:
                return
            self.clear()
            block = self.lines()
            self.stream.write("\n".join(block) + "\n")
            self.stream.flush()
            self._drawn = len(block)

    def _log_changes(self) -> None:
        now = self.clock()
        periodic = now - self._logged >= _LOG_SECONDS
        for model, name in enumerate(self.names):
            rows = self._records(model)
            seen = self._models[model]
            phase = min(int(r["phase"]) for r in rows)
            before = seen.see(phase, now)
            if phase == 0:
                continue
            prefix = f"{name}: " if len(self.names) > 1 else ""
            if phase != before:
                for ended in range(max(before, 1), phase):
                    self.log(
                        f"{prefix}{PHASES[ended - 1]} done in {_duration(seen.took(ended))}"
                    )
                if phase < len(PHASES):
                    self.log(
                        f"{prefix}{PHASES[phase - 1]}: {self._status(model, rows, phase)}"
                    )
                else:
                    self.log(f"{prefix}{self._ready_text(model, rows, now)}")
            elif periodic and phase < len(PHASES):
                self.log(
                    f"{prefix}{PHASES[phase - 1]}: {self._status(model, rows, phase)}"
                )
        if periodic:
            self._logged = now

    # ---- the drawing thread, and log lines around the live block
    def start(self, interval: float | None = None) -> Renderer:
        if interval is None:
            interval = 0.25 if self.live else 1.0
        if self.live:
            self._install_log_hooks()

        def run() -> None:
            while not self._stop.wait(interval):
                try:
                    self.draw()
                except Exception as error:  # a display error never stops a load
                    logger.debug(f"progress display: {error}")

        self._thread = threading.Thread(target=run, name="wright-progress", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._paused_until = 0.0
        self.draw()
        self._remove_log_hooks()

    def before_log(self) -> None:
        """A log line follows: remove the block, and do not draw it again until ``after_log``
        (or a second passes)."""
        with self._lock:
            self._paused_until = self.clock() + 1.0
            self.clear()

    def after_log(self) -> None:
        with self._lock:
            self._paused_until = 0.0
        if not self._stop.is_set():
            self.draw()

    def _install_log_hooks(self) -> None:
        handlers = list(logger.handlers)
        if not handlers:
            return
        renderer = self

        class Before(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                renderer.before_log()
                return True

        class After(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                renderer.after_log()

        before, after = Before(), After(logging.NOTSET)
        handlers[0].addFilter(before)
        logger.addHandler(after)
        self._hooks = (handlers[0], before, after)

    def _remove_log_hooks(self) -> None:
        if self._hooks is None:
            return
        first, before, after = self._hooks
        first.removeFilter(before)
        logger.removeHandler(after)
        self._hooks = None
