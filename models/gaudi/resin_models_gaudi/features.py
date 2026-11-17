"""Optional features of the Gaudi runtime: ``wright -f``, and ``--features`` of deploy/serving.py and scripts/bench.py.

A feature is off unless the command line names it. The command line writes the features to RESIN_GAUDI_FEATURES
(names separated by commas), so that the ranks that the server starts get them too. ``-f help`` shows the list.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Iterable

VARIABLE = "RESIN_GAUDI_FEATURES"

FEATURES = {
    "oldqwen": "Qwen3.5/3.8 steps run in the Python executor, not in the native executor (Rust). The Python "
    "executor runs on one card only: a Qwen model on more than one card (--ranks 2 or more, or a Qwen replica over "
    "several cards) and --step-transport native do not work with it. It does not have the wide attention launches "
    "of the native executor.",
    "skipvalidation": "Launches skip the check of their tensor bindings. Use it only with a build that you trust, "
    "not while you change graph bindings.",
}


def parse(values: Iterable[str]) -> tuple[str, ...]:
    """The features that ``values`` name (each value is a name or a list of names separated by commas), sorted
    and each once. A ValueError names an unknown feature."""
    names: set[str] = set()
    for value in values:
        for name in value.split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name not in FEATURES:
                raise ValueError(
                    f"unknown feature {name!r}: the features are {', '.join(FEATURES)} (-f help)"
                )
            names.add(name)
    return tuple(sorted(names))


def describe() -> str:
    """The text that ``-f help`` shows."""
    width = max(map(len, FEATURES))
    lines = [
        "Features: give -f NAME[,NAME...]. A feature is off unless you name it.",
        "",
    ]
    for name, text in FEATURES.items():
        body = textwrap.wrap(text, 100 - width - 4)
        lines.append(f"  {name:<{width}}  {body[0]}")
        lines += [" " * (width + 4) + line for line in body[1:]]
    return "\n".join(lines)


def set_features(names: Iterable[str]) -> None:
    """Make ``names`` the features of this process and of the ranks it starts after this."""
    os.environ[VARIABLE] = ",".join(parse(names))


def enabled_features() -> tuple[str, ...]:
    """The features of this process (RESIN_GAUDI_FEATURES)."""
    return parse([os.environ.get(VARIABLE, "")])


def enabled(name: str) -> bool:
    """Whether feature ``name`` is on in this process."""
    if name not in FEATURES:
        raise KeyError(f"no feature {name!r}")
    return name in enabled_features()
