"""GLM-5.3-Flash's step as one native program (``Glm53Gaudi._native_program``, a ``resin._resin.GaudiProgram``) makes
the launches, with their bindings, and the collectives of the Python launch loop (``glm_reference``) in the same order,
for every token source and every index slot and slot launch count; on 1 and 4 ranks, in gathered and reduced
shapes. Both run on a model object with fake recipes and a recording runtime library, no card."""

from __future__ import annotations

import ctypes
import itertools
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from glm_reference import Reference

import resin._resin as native

pytestmark = pytest.mark.skipif(
    not hasattr(native, "GaudiProgram"), reason="resin built without the gaudi feature"
)


class Library:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []
        self.binds: list[int] = []

    def rg_bind_slots(self, recipe: int, slots: int, addresses: int, n: int) -> int:
        s = (ctypes.c_int32 * n).from_address(slots) if n else []
        a = (ctypes.c_uint64 * n).from_address(addresses) if n else []
        self.binds = [value for pair in zip(list(s), list(a)) for value in pair]
        return 0

    def rg_launch(self, recipe: int) -> int:
        self.calls.append(("launch", [recipe, *self.binds]))
        return 0

    def rg_ccl_allgather(self, send: int, recv: int, count: int, dtype: int) -> int:
        self.calls.append(("allgather", [send, recv, count, dtype]))
        return 0

    def rg_ccl_allreduce(self, send: int, recv: int, count: int, dtype: int) -> int:
        assert send == recv, "the residual all-reduce is in place"
        self.calls.append(("allreduce", [send, count, dtype]))
        return 0


class Device:
    glue = None

    def __init__(self) -> None:
        self.lib = Library()

    def allgather(
        self, send: int, recv: int, count: int, *, bf16: bool = False
    ) -> None:
        self.lib.calls.append(("allgather", [send, recv, count, int(bf16)]))

    def check(self, status: int, what: str) -> None:
        raise AssertionError(what)


def model(nranks: int, sparse: bool):
    from resin_models_gaudi.glm5_3_flash import model as M

    device = Device()
    handles = itertools.count(1)
    addresses = itertools.count(0x1000, 0x100)

    def op(*names: str):
        slots = {name: i for i, name in enumerate(names)}
        recipe: Any = SimpleNamespace(
            handle=next(handles), slot=slots.__getitem__, device=device, name="r"
        )
        return M._Op(recipe, {name: next(addresses) for name in names})

    m: Any = object.__new__(M.Glm53Gaudi)  # with the fields a step's launches read
    m.device, m.nranks, m.probe = device, nranks, None
    m.d = SimpleNamespace(hidden=16, vocab=64, gathered=lambda T: nranks > 1 and T <= 4)
    m.pa, m.pb, m.pa_all, m.pb_all = 0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000
    m.logits, m.logits_all = 0x50_0000, 0x60_0000
    m._registers = np.zeros(M._REGISTERS, np.uint64)
    prog = M._Program()
    prog.embed, prog.embed_device, prog.head = (
        op("tok", "emb"),
        op("tok", "emb"),
        op("x"),
    )

    def slots() -> list:  # a sparse step's one launch, or the slot launches
        return [op("sparse")] if sparse else [op("s") for _ in range(4)]

    # a linear-attention layer (no index, no slots), and attention layers with and without the indexer
    prog.layers = [
        (op("kda"), [], None, [], None, op("ffn")),
        (op("pre"), [op("ix") for _ in range(3)], op("ixe"), slots(), op("post"), op("ffn")),
        (op("pre"), [], None, slots(), op("post"), op("ffn")),
        (op("pre"), [op("ix") for _ in range(3)], op("ixe"), slots(), op("post"), op("ffn")),
    ]  # fmt: skip
    prog.taps = {2: op("tap")}
    return M, m, prog


def legacy(m, prog, T, device_tokens, nix, nsl) -> list:
    m.device.lib.calls.clear()
    Reference(m)._launches(prog, T, device_tokens, nix, nsl)
    return list(m.device.lib.calls)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("nranks", [1, 4])
@pytest.mark.parametrize("T", [1, 16])
def test_the_step_program_makes_the_launch_loops_calls(
    nranks: int, T: int, sparse: bool
):
    M, m, prog = model(nranks, sparse)
    program = m._native_program(prog, T)
    assert prog.native is program
    compared = 0
    counts = (
        (1,) if sparse else range(5)
    )  # a sparse step makes its one launch per layer
    for device_tokens, nix, nsl in itertools.product((False, True), range(4), counts):
        expected = legacy(m, prog, T, device_tokens, nix, nsl)
        registers = [0] * M._REGISTERS
        registers[M._R_DEVICE_TOKENS], registers[M._R_INDEX], registers[M._R_SLOTS] = (
            device_tokens,
            nix,
            nsl,
        )
        assert program.calls(registers, [0]) == expected, (device_tokens, nix, nsl)
        compared += 1
    assert compared == 2 * 4 * len(counts)


def test_more_launches_than_a_layer_has_fail():
    M, m, prog = model(4, sparse=False)
    program = m._native_program(prog, 1)
    registers = [0] * M._REGISTERS
    registers[M._R_SLOTS] = 5  # a layer has four slot launches
    with pytest.raises(RuntimeError, match="takes 5 ops of 4"):
        program.calls(registers, [0])


def bound(calls: list) -> list:
    """the calls with each launch's bindings as a set: a launch with unique slots binds in any order"""
    return [
        (kind, values[0], frozenset(zip(values[1::2], values[2::2])))
        if kind == "launch"
        else (kind, tuple(values))
        for kind, values in calls
    ]


def mtp_program(M, m, op, keys_only: bool):
    prog = M._MtpProgram()
    prog.mtp_in = op("mtp_prev", "x", "hf", "mtp_tok", "cols")
    prog.slot_prev = prog.mtp_in.index("mtp_prev")
    prog.slot_hf = prog.mtp_in.index("hf")
    prog.slot_tok = prog.mtp_in.index("mtp_tok")
    prog.pre = op("pre")
    if keys_only:  # a prefill chunk's program
        return prog
    prog.ix = [op("ix") for _ in range(3)]
    prog.ix_end = op("ixe")
    prog.slots = [op("s") for _ in range(4)]
    prog.post, prog.ffn = op("post"), op("ffn")
    prog.head = op("x", "hf_out", "logits")
    prog.slot_hf_out = prog.head.index("hf_out")
    return prog


@pytest.mark.parametrize("nranks", [1, 4])
@pytest.mark.parametrize("T", [1, 16])
def test_the_mtp_program_makes_the_mtp_loops_calls(nranks: int, T: int):
    M, m, _ = model(nranks, sparse=False)
    m._mregisters = np.zeros(M._M_REGISTERS, np.uint64)
    handles = itertools.count(1000)
    addresses = itertools.count(0x9000, 0x100)

    def op(*names: str):
        slots = {name: i for i, name in enumerate(names)}
        recipe: Any = SimpleNamespace(
            handle=next(handles), slot=slots.__getitem__, device=m.device, name="r"
        )
        return M._Op(recipe, {name: next(addresses) for name in names})

    def legacy_mtp(prog, *args) -> list:
        m.device.lib.calls.clear()
        Reference(m)._mtp_launches(prog, T, *args)
        return list(m.device.lib.calls)

    full, keys = mtp_program(M, m, op, False), mtp_program(M, m, op, True)
    for prog in (full, keys):
        program = m._mtp_native(prog, T)
        layers = (False, True) if prog.head is not None else (False,)
        for step, (layer, nix, nsl) in enumerate(
            itertools.product(layers, range(4), range(5))
        ):
            prev, hf, tok, hf_out = (
                0xA000 + step,
                0xB000 + step,
                0xC000 + step,
                0xD000 + step,
            )
            expected = legacy_mtp(prog, layer, prev, hf, tok, hf_out, nix, nsl)
            registers = [0] * M._M_REGISTERS
            registers[M._M_PREV], registers[M._M_HF], registers[M._M_TOK] = (
                prev,
                hf,
                tok,
            )
            registers[M._M_HF_OUT], registers[M._M_INDEX], registers[M._M_SLOTS] = (
                hf_out,
                nix,
                nsl,
            )
            got = program.calls(registers, [0, 1] if layer else [0])
            assert bound(got) == bound(expected), (layer, nix, nsl)
    assert keys.native.sections == ["keys"]
    assert full.native.collective_free("keys")
