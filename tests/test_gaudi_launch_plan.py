"""``LaunchPlan`` as a native program (``resin._resin.GaudiProgram``) leaves each recipe with the bindings the Python
loop leaves, launch by launch, with patched bindings read from registers; recorded without a card."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import resin._resin as native

pytestmark = pytest.mark.skipif(
    not hasattr(native, "GaudiProgram"), reason="resin built without the gaudi feature"
)


class Library:
    """bind / launch calls of the Python loop, as (recipe, {slot: address} after the bindings)"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[int, int]]] = []
        self.bound: dict[int, dict[int, int]] = {}

    def rg_bind_slots(self, recipe: int, slots: int, addresses: int, n: int) -> int:
        s = (ctypes.c_int32 * n).from_address(slots) if n else []
        a = (ctypes.c_uint64 * n).from_address(addresses) if n else []
        bound = self.bound.setdefault(recipe, {})
        for slot, address in zip(list(s), list(a)):
            bound[slot] = address
        return 0

    def rg_launch(self, recipe: int) -> int:
        self.calls.append((recipe, dict(self.bound.get(recipe, {}))))
        return 0


def replay(calls) -> list[tuple[int, dict[int, int]]]:
    """the program's recorded calls as the same (recipe, bindings so far) per launch"""
    bound: dict[int, dict[int, int]] = {}
    out = []
    for kind, values in calls:
        assert kind == "launch"
        recipe, binds = values[0], values[1:]
        b = bound.setdefault(recipe, {})
        for slot, address in zip(binds[::2], binds[1::2]):
            b[slot] = address
        out.append((recipe, dict(b)))
    return out


def plan():
    from resin_models_gaudi.synapse.graph import LaunchPlan

    device: Any = SimpleNamespace(lib=Library(), glue=None)
    recipe = lambda handle: SimpleNamespace(handle=handle, name=f"r{handle}")  # noqa: E731
    steps: Any = [  # recipes stand in by their handle and name
        (recipe(1), np.array([0, 1, 2]), np.array([10, 11, 12])),
        # slot 1 twice: the later binding wins
        (recipe(2), np.array([1, 0, 1]), np.array([20, 21, 22])),
        (recipe(1), np.array([2]), np.array([13])),  # recipe 1 keeps its other bindings
        (recipe(3), np.array([], np.int32), np.array([], np.uint64)),
        (recipe(4), np.array([5, 6]), np.array([40, 41])),
    ]
    return device, LaunchPlan(device, steps)


def python_loop(device, p) -> list:
    device.lib.calls.clear()
    device.lib.bound.clear()
    p.run()
    return list(device.lib.calls)


def test_the_program_leaves_the_python_loops_bindings():
    device, p = plan()
    expected = python_loop(device, p)
    assert expected[1] == (2, {1: 22, 0: 21})
    program = p._build()
    assert replay(program.calls(p._registers.tolist(), [0])) == expected


def test_patched_bindings_come_from_registers():
    device, p = plan()
    n = int(p.starts[-1])
    p.patch(n - 1, 99)  # recipe 4's slot 6
    p.patch(4, 77)  # recipe 2's last slot-1 binding
    program = p._build()
    p._program = program
    assert replay(program.calls(p._registers.tolist(), [0])) == python_loop(device, p)
    # later patches of the same bindings change only the registers
    p.patch(n - 1, 123)
    p.patch(4, 456)
    assert p._program is program
    assert replay(program.calls(p._registers.tolist(), [0])) == python_loop(device, p)
    # patching a binding the program has as fixed (recipe 1's slot 2 in launch 0) rebuilds it
    p.patch(2, 555)
    assert p._program is None
    program = p._build()
    assert replay(program.calls(p._registers.tolist(), [0])) == python_loop(device, p)
