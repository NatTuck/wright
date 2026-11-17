"""Every Gaudi model takes its stream event slots from the device (``Device.event_slots``), so several models of
one process never share a slot; no model module fixes a slot number."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from resin_models_gaudi.synapse.runtime import Device, SynapseError

PACKAGE = Path(__file__).resolve().parents[1] / "models/gaudi/resin_models_gaudi"


def device() -> Device:
    lib: Any = SimpleNamespace(rg_module_id=lambda: 0)
    libraries: Any = SimpleNamespace(runtime="librt.so")
    return Device(lib, 0, libraries)


def test_slots_are_handed_out_once_up_to_the_runtimes_64():
    dev = device()
    taken = [
        dev.event_slots(n) for n in (4, 8, 4, 4)
    ]  # e.g. Qwen, GLM, DeepSeek, Mistral
    assert taken == [0, 4, 12, 16]
    dev.event_slots(64 - 20)
    with pytest.raises(SynapseError, match="do not fit in 64"):
        dev.event_slots(1)


def test_no_model_fixes_an_event_slot():
    fixed = [
        f"{path.relative_to(PACKAGE)}:{n}: {line.strip()}"
        for path in PACKAGE.rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.match(r"_[A-Z_]*EVENT\s*=\s*\d", line)
    ]
    assert fixed == []
