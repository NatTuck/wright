"""Log setup for the server and its ranks. The settings pass to spawned ranks in the environment."""

from __future__ import annotations

import logging
import os
import sys

LEVELS = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG,
}
_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE = "%H:%M:%S"


def configure_logging(
    verbosity: int = 3, colors: str = "auto", path: str | None = None
) -> None:
    """Keep the settings for this process and its ranks, and apply them here."""
    os.environ["WRIGHT_LOG_VERBOSITY"] = str(verbosity)
    os.environ["WRIGHT_LOG_COLORS"] = colors
    if path:
        os.environ["WRIGHT_LOG_FILE"] = path
    setup_logging()


def setup_logging(rank: int = 0, name: str | None = None) -> logging.Logger:
    """Install the handlers from the settings. Other ranks repeat rank 0, so they show only
    warnings unless the verbosity asks for debug output. A process with a ``name`` (a replica,
    which serves requests of its own) logs as rank 0 does, with its name before each line."""
    verbosity = int(os.environ.get("WRIGHT_LOG_VERBOSITY", "3"))
    level = LEVELS[min(max(verbosity, 0), 4)]
    if rank != 0 and name is None and level > logging.DEBUG:
        level = max(level, logging.WARNING)
    colors = os.environ.get("WRIGHT_LOG_COLORS", "auto")
    logger = logging.getLogger("wright")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    prefix = f"{name}: " if name else ("" if rank == 0 else f"rank {rank}: ")
    template = f"%(asctime)s %(levelname)s {prefix}%(message)s"
    try:
        import coloredlogs
    except (
        ImportError
    ):  # CUDA and Modal installs do not have the Gaudi command's packages
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(template, _DATE))
        logger.addHandler(handler)
    else:
        coloredlogs.install(
            logger=logger,
            level=level,
            fmt=template,
            datefmt=_DATE,
            stream=sys.stderr,
            isatty={"on": True, "off": False}.get(colors),
            reconfigure=True,
        )
    path = os.environ.get("WRIGHT_LOG_FILE")
    if path:
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(template, _DATE))
        logger.addHandler(handler)
    # The HTTP frontend's request lifecycle lines and each step's line come from native code, which writes
    # them as these handlers do (no Python call for a line).
    from resin import set_log_level, set_log_sink

    colored = any(
        type(handler.formatter).__name__ == "ColoredFormatter"
        for handler in logger.handlers
    )
    set_log_sink(prefix, colors=colored, path=path or None)
    set_log_level(level)
    return logger
