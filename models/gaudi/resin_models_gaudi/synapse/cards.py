"""Gaudi cards, named by module ID as ``hl-smi`` shows them. No Synapse is loaded here.

``HABANA_VISIBLE_MODULES`` lists the modules a process may use. Unset means every operational card.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

VISIBLE_MODULES = "HABANA_VISIBLE_MODULES"
SYSFS = Path("/sys/class/accel")


@dataclass(frozen=True, slots=True)
class Card:
    accel: int
    """The ``/sys/class/accel`` index, which is not the module ID."""
    module: int
    pci: str
    operational: bool


def all_cards(root: Path = SYSFS) -> list[Card]:
    """Every card on this host, by module ID."""
    cards = []
    for path in root.glob("accel[0-9]*"):
        device = path / "device"
        cards.append(
            Card(
                accel=int(path.name.removeprefix("accel")),
                module=int((device / "module_id").read_text()),
                pci=device.resolve().name,
                operational=(device / "status").read_text().strip() == "Operational",
            )
        )
    return sorted(cards, key=lambda card: card.module)


def parse_modules(text: str) -> tuple[int, ...]:
    """A comma-separated module list, such as ``0,1,2,3``."""
    if not re.fullmatch(r"\d+(,\d+)*", text):
        raise ValueError(f"expected module IDs such as 0,1,2,3, got {text!r}")
    modules = tuple(int(module) for module in text.split(","))
    if len(set(modules)) != len(modules):
        raise ValueError(f"module list {text!r} has a duplicate")
    return modules


def visible_cards(
    environ: Mapping[str, str] = os.environ, root: Path = SYSFS
) -> list[Card]:
    """The operational cards this process may use, in the listed order."""
    cards = all_cards(root)
    listed = environ.get(VISIBLE_MODULES, "")
    if not listed or listed == "all":
        return [card for card in cards if card.operational]
    by_module = {card.module: card for card in cards}
    selected = []
    for module in parse_modules(listed):
        if module not in by_module:
            raise ValueError(
                f"module {module} does not exist; this host has "
                f"{','.join(str(card.module) for card in cards) or 'no cards'}"
            )
        if by_module[module].operational:
            selected.append(by_module[module])
    return selected
