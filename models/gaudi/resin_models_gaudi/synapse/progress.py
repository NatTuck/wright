"""Weight loading progress: a live bar on a terminal, and periodic log lines otherwise."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

info = logging.getLogger("wright").info
_LOG_SECONDS = 30.0
_BAR = 30


def weight_file_bytes(directory: str | Path, pattern: str) -> int:
    return sum(path.stat().st_size for path in Path(directory).glob(pattern))


class LoadProgress:
    """``total`` is an estimate from the file sizes, so the percentage stops short of 100
    until the load finishes."""

    def __init__(self, total: int, *, log=info) -> None:
        self.total, self.log, self.done = max(total, 1), log, 0
        self.started = self.reported = time.monotonic()
        self.live = sys.stderr.isatty() and log is info
        self.log(f"loading weights, about {self.total / 2**30:.1f} GiB per card")

    def add(self, nbytes: int) -> None:
        self.done += nbytes
        now = time.monotonic()
        if now - self.reported < (0.5 if self.live else _LOG_SECONDS):
            return
        self.reported = now
        fraction = min(self.done / self.total, 0.99)
        elapsed = now - self.started
        status = (
            f"{self.done / 2**30:.1f}/{self.total / 2**30:.1f} GiB, {fraction:3.0%}, "
            f"{elapsed:.0f} s elapsed, about {elapsed * (1 / max(fraction, 1e-3) - 1):.0f} s left"
        )
        if self.live:
            filled = int(_BAR * fraction)
            bar = "#" * filled + "-" * (_BAR - filled)
            print(f"\r[{bar}] {status}", end="", file=sys.stderr, flush=True)
        else:
            self.log(f"weights: {status}")

    def finish(self) -> None:
        if self.live:
            print(file=sys.stderr, flush=True)
        self.log(
            f"weights loaded: {self.done / 2**30:.1f} GiB in "
            f"{time.monotonic() - self.started:.0f} s"
        )
