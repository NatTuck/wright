"""Qwen3.8-Flash-Next's step program (``qwen4_exp.model._Program``) runs as a ``resin._resin.GaudiProgram`` that
makes the calls the runtime library's ``rg_run_program`` (since removed) made for the same ops: each launch after
its bindings (none when it has no bindings), then its collective. Recorded without a card."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import resin._resin as native

pytestmark = pytest.mark.skipif(
    not hasattr(native, "GaudiProgram"), reason="resin built without the gaudi feature"
)


def recipe(handle: int, *names: str) -> Any:
    slots = {name: 10 * handle + i for i, name in enumerate(names)}
    return SimpleNamespace(handle=handle, slot=slots.__getitem__, name=f"r{handle}")


def rg_run_program(ops) -> list:
    """what runtime.cpp's rg_run_program (since removed) issued for these ops"""
    calls = []
    for r, names, collective in ops:
        binds = [value for n, a in names.items() for value in (r.slot(n), a)]
        calls.append(("launch", [r.handle, *binds]))
        if collective is not None:
            kind, src, dst, count = collective
            bf16 = kind >> 2 & 1
            if kind & 3 == 1:
                calls.append(("allreduce", [src, count, bf16]))
            elif kind & 3 == 2:
                calls.append(("allgather", [src, dst, count, bf16]))
    return calls


def test_a_frozen_program_makes_the_c_programs_calls():
    from resin_models_gaudi.qwen4_exp.model import _Program

    prog = _Program()
    prog.add(recipe(1, "x", "w"), {"x": 0x100, "w": 0x200}, None)
    prog.add(recipe(2, "p"), {"p": 0x300}, (1, 0x300, 0x300, 64))
    prog.add(recipe(3), {}, None)  # bound before, so no bindings now
    prog.add(recipe(4, "p", "q"), {"p": 0x300, "q": 0x400}, (2, 0x300, 0x500, 32))
    prog.add(recipe(5, "y"), {"y": 0x600}, (2 | 4, 0x600, 0x700, 8))
    prog.freeze()
    assert prog.n == 5
    assert prog.native.calls([], [0]) == rg_run_program(prog.ops)
    assert prog.native.collective_free(0) is False


def test_an_unknown_collective_is_refused():
    from resin_models_gaudi.qwen4_exp.model import _Program

    prog = _Program()
    prog.add(recipe(1, "x"), {"x": 1}, (3, 1, 1, 1))
    with pytest.raises(ValueError, match="collective kind 3"):
        prog.freeze()
