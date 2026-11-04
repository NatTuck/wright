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


def setup_logging(rank: int = 0) -> logging.Logger:
    """Install the handlers from the settings. Other ranks repeat rank 0, so they show only
    warnings unless the verbosity asks for debug output."""
    verbosity = int(os.environ.get("WRIGHT_LOG_VERBOSITY", "3"))
    level = LEVELS[min(max(verbosity, 0), 4)]
    if rank != 0 and level > logging.DEBUG:
        level = max(level, logging.WARNING)
    colors = os.environ.get("WRIGHT_LOG_COLORS", "auto")
    logger = logging.getLogger("wright")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    prefix = "" if rank == 0 else f"rank {rank}: "
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
    return logger
