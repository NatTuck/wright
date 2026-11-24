"""DeepSeek-V4.1's decode step as one native program (``_decode_program``, a ``resin._resin.GaudiProgram``) makes
the same launches, with the same bindings, and the same collectives in the same order as the Python launch loop
(``_run_layer``), for every count a decode step's fill can give. Both run on a model object with fake recipes and
a recording runtime library, no card."""

from __future__ import annotations

import ctypes
import itertools
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import resin._resin as native

pytestmark = pytest.mark.skipif(
    not hasattr(native, "GaudiProgram"), reason="resin built without the gaudi feature"
)


class Library:
    """The runtime library's launch and bind calls, recorded as the program's calls are."""

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


class Device:
    def __init__(self) -> None:
        self.lib = Library()

    def allreduce(self, address: int, count: int, *, bf16: bool = False) -> None:
        self.lib.calls.append(("allreduce", [address, count, int(bf16)]))

    def allgather(
        self, send: int, recv: int, count: int, *, bf16: bool = False
    ) -> None:
        self.lib.calls.append(("allgather", [send, recv, count, int(bf16)]))

    def check(self, status: int, what: str) -> None:
        assert status >= 0, what


def model(nranks: int):
    from resin_models_gaudi.deepseek_v41 import model as M
    from resin_models_gaudi.glm5_3_flash.model import _Op

    device = Device()
    handles = itertools.count(1)
    addresses = itertools.count(0x1000, 0x100)

    def op(*names: str):
        """a launch of its own recipe with one binding per name, at distinct addresses"""
        slots = {name: i for i, name in enumerate(names)}
        recipe: Any = SimpleNamespace(
            handle=next(handles), slot=slots.__getitem__, device=device, name="r"
        )
        return _Op(recipe, {name: next(addresses) for name in names})

    def ops(n: int) -> list:
        return [op("q", "k") for _ in range(n)]

    from ds41_reference import ReferenceModel

    m: Any = object.__new__(ReferenceModel)  # with the fields a decode program reads
    m.device, m.nranks = device, nranks
    # layer 0 has no index; 1 and 3 are ratio 1, 2 and 4 ratio 2; the selections differ per layer
    m.cfg = SimpleNamespace(compress_ratios=[0, 1, 2, 1, 2])
    m.d = SimpleNamespace(
        hidden=16, vocab=64, ranks=nranks, gathered=lambda T: nranks > 1 and T <= 4
    )
    m.logits, m.logits_all = 0x80_0000, 0x90_0000
    m.pbuf, m.pall = [0x10_0000, 0x20_0000], [0x30_0000, 0x40_0000]
    m.io = {"eg_emb": 0x50_0000, "eg_kv": 0x60_0000, "eg_kv_all": 0x70_0000}
    m._native_programs, m._registers = {}, np.zeros(M._REGISTERS, np.uint64)

    def layer(
        ix: int = 0,
        ixc: int = 0,
        sel=(),
        engram: bool = False,
        out: bool = True,
        dense: bool = False,
    ):
        layer = {
            "attn": op("x"),
            "ffn": op("x", "y"),
            "p_attn": m.pbuf[0],
            "p_ffn": m.pbuf[1],
        }
        if engram:
            layer |= {"eg_gather": op("e"), "eg": op("e", "f"), "eg_gate": op("g")}
        if out:
            layer["out"] = op("o")
        if dense:  # a prefill step's dense attention and output
            layer["attn_dense"] = op("x", "d")
            if out:
                layer["out_dense"] = op("o", "d")
        if ix:  # the tiled chain: each group's score launches, its block maxima or mask, then its merge
            layer["ixp"] = ops(ix)
            layer["ixc2"] = ops(ix + 1)
            layer["ixc_tail"] = [op("s")]
        if ixc:  # the sparse reindex path
            layer["ixsp"] = ops(ixc)
            layer["ixsp_tail"] = [op("s")]
        return layer

    prog = SimpleNamespace(
        layers=[
            layer(engram=True),
            layer(
                ix=3,
                ixc=2,
                sel=("sparse", "all", False, True, "pf", "pf_all"),
                dense=True,
            ),
            layer(ix=2, sel=(False, True, "pf"), out=False, dense=True),
            layer(ix=3, sel=("all", False, True)),
            layer(ix=2, ixc=2, sel=("sparse", False, True)),
        ],
        inject=op("i", "j"),
        embed=op("tok", "emb"),
        head=op("x", "logits"),
        store=op("st", "kv"),
        ixg=[],
    )
    m.n_layers = len(prog.layers)
    return M, m, prog


def legacy(m, prog, s, counts: dict, logits=None):
    calls = m.device.lib.calls
    calls.clear()
    try:
        m._step_launches(prog, s, counts, logits)
    except KeyError:  # a selection the layer does not have
        return None
    return list(calls)


FLAGS = (
    "sparse",
    "cand",
    ("radix", 1),
    ("radix", 2),
    ("pf", 1),
    ("pf", 2),
    ("dense", 1),
    ("dense", 2),
)
LAUNCHES = list(
    itertools.product(range(3), range(4), range(3))
)  # ixc, and the index launches per ratio


def steps():
    """every combination of the step's flags, each with other launch counts and logits"""
    for i, flags in enumerate(itertools.product((False, True), repeat=len(FLAGS))):
        ixc, ix1, ix2 = LAUNCHES[i % len(LAUNCHES)]
        yield (
            {**dict(zip(FLAGS, flags)), "ixc": ixc, 1: ix1, 2: ix2},
            (None, "all", 3)[i % 3],
        )


def registers(M, m, counts, logits) -> list[int]:
    r = m._step_registers(counts)
    r[M._R_COLUMN] = 3 if m._one_column(logits) else 0
    return r.tolist()


@pytest.mark.parametrize("nranks", [1, 2])
@pytest.mark.parametrize("T", [1, 16])
def test_the_step_program_makes_the_launch_loops_calls(nranks: int, T: int):
    M, m, prog = model(nranks)
    s = M.Shape(T, T > 1)
    program = m._step_program(s, prog)
    assert m._step_program(s, prog) is program
    compared = failed = 0
    for counts, logits in steps():
        expected = legacy(m, prog, s, counts, logits)
        got = m._sections(0, m.n_layers, logits)
        if expected is None:
            with pytest.raises(RuntimeError, match="no arm"):
                program.calls(registers(M, m, counts, logits), got)
            failed += 1
        else:
            assert program.calls(registers(M, m, counts, logits), got) == expected, (
                counts,
                logits,
            )
        compared += 1
    assert compared == 2 ** len(FLAGS)
    # a prefiltered step of layer 3 (no pf selection) takes its all or radix selection: every step runs
    assert failed == 0


def test_the_sections_are_the_parts_of_a_step():
    M, m, prog = model(2)
    program = m._step_program(M.Shape(1, False), prog)
    names = program.sections
    n = m.n_layers
    assert [names[i] for i in m._sections(0, n, "all")] == [
        "embed",
        *(f"layer {i}" for i in range(n)),
        "inject",
        "head",
    ]
    assert [names[i] for i in m._sections(0, n, 3)][-1] == "head_column"
    assert [names[i] for i in m._sections(0, 2, None)] == [
        "embed",
        "layer 0",
        "layer 1",
    ]
    assert [names[i] for i in m._sections(2, n, None)] == [
        *(f"layer {i}" for i in range(2, n)),
        "inject",
    ]
    assert names[n + 4] == "store"


def test_bounded_replay_makes_the_launch_loops_calls():
    """the layers before the cut and the owner store, then the layers from the cut and the head"""
    M, m, prog = model(2)
    s = M.Shape(16, True)
    program = m._step_program(s, prog)
    counts, logits = next(steps())
    lib, cut, n = m.device.lib, 2, m.n_layers
    lib.calls.clear()
    prog.embed.run(lib)
    for layer in range(cut):
        m._run_layer(layer, prog.layers[layer], counts, s, True)
    prog.store.run(lib)
    first = list(lib.calls)
    lib.calls.clear()
    for layer in range(cut, n):
        m._run_layer(layer, prog.layers[layer], counts, s, True)
    prog.inject.run(lib)
    m._head(prog, "all", s.T, True)
    second = list(lib.calls)
    r = registers(M, m, counts, None)
    assert program.calls(r, [*m._sections(0, cut, None), n + 4]) == first
    assert program.calls(r, m._sections(cut, n, "all")) == second


def test_a_missing_selection_fails_like_the_launch_loop():
    M, m, prog = model(2)
    del prog.layers[1]["ixc2"]  # a candidate step then has no selection for layer 1
    s = M.Shape(1, False)
    program = m._step_program(s, prog)
    counts = {flag: False for flag in FLAGS} | {
        "sparse": False,
        "cand": True,
        "ixc": 1,
        1: 1,
        2: 1,
    }
    assert legacy(m, prog, s, counts) is None
    with pytest.raises(RuntimeError, match="no arm") as raised:
        program.calls(registers(M, m, counts, None), m._sections(0, m.n_layers, None))
    # the embedding, layer 0 and layer 1's attention launch and partial collectives came first
    assert getattr(raised.value, "issued") > 0
    counts["cand"] = False
    expected = legacy(m, prog, s, counts)
    assert (
        program.calls(registers(M, m, counts, None), m._sections(0, m.n_layers, None))
        == expected
    )


@pytest.mark.parametrize("nranks", [1, 2])
def test_the_draft_program_makes_the_draft_loops_calls(nranks: int):
    M, m, prog = model(nranks)
    from resin_models_gaudi.glm5_3_flash.model import _Op

    handles = itertools.count(900)

    def op(*names: str):
        slots = {name: i for i, name in enumerate(names)}
        recipe: Any = SimpleNamespace(
            handle=next(handles), slot=slots.__getitem__, device=m.device, name="d"
        )
        return _Op(recipe, {name: 0xF000 + i for i, name in enumerate(names)})

    m._draft_natives = {}
    draft = {
        "embed": op("tok"),
        "stages": [
            {
                "attn": op("a"),
                "ffn": op("f", "g"),
                "p_attn": m.pbuf[0],
                "p_ffn": m.pbuf[1],
            }
            for _ in range(2)
        ],
        "head": op("x", "logits"),
        "markov": {1: op("m1"), 2: op("m1", "m2")},
    }
    s = M.Shape(4, False)
    program = m._draft_native(s, draft)
    for B in (1, 2):
        m.device.lib.calls.clear()
        m._draft_launches(draft, s, B)
        assert program.calls([B], [0]) == m.device.lib.calls, B
    for B in (0, 3):  # no Markov launch for these
        with pytest.raises(RuntimeError):
            program.calls([B], [0])
