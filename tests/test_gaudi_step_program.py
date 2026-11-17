"""The native step program (``resin._resin.GaudiProgram``, the ``gaudi`` feature): a model step's stream calls,
built once and recorded per step without a card. ``calls`` records the same walk that ``run`` makes on a card."""

from __future__ import annotations

import pytest

import resin._resin as native

pytestmark = pytest.mark.skipif(
    not hasattr(native, "GaudiProgram"), reason="resin built without the gaudi feature"
)

# registers of the example step: 0 the number of index launches, 1 sparse (0 or 1), 2 the token count
INDEX, SPARSE, TOKENS = 0, 1, 2


def example():
    p = native.GaudiProgram(3)
    assert p.section("layers") == 0
    # the token count decides an address: base + registers[TOKENS] * 4096
    p.launch(7, [0, 1], [0x1000, 0x2000], dynamic=[(1, (0x9000, TOKENS, 4096))])
    p.allreduce(0x1000, (0, TOKENS, 1024), bf16=True)
    p.take(INDEX)
    for recipe in (10, 11, 12):
        p.launch(recipe, [], [])
    p.end()
    p.select([(SPARSE, 2)], [0, 1])
    assert p.arm() == 0
    p.launch(20, [], [])
    assert p.arm() == 1
    p.launch(21, [], [])
    p.end()
    assert p.section("harvest") == 1
    p.download(0x7000, 0x3000, (0, TOKENS, 4))
    p.record(5)
    p.finish()
    return p


def test_a_step_makes_the_calls_its_registers_choose():
    p = example()
    assert p.registers == 3
    assert p.sections == ["layers", "harvest"]
    calls = p.calls([2, 1, 3], ["layers", 1])
    assert calls == [
        ("launch", [7, 0, 0x1000, 1, 0x2000, 1, 0x9000 + 3 * 4096]),
        ("allreduce", [0x1000, 3 * 1024, 1]),
        ("launch", [10]),
        ("launch", [11]),
        ("launch", [21]),
        ("download", [0x7000, 0x3000, 12]),
        ("record", [5]),
    ]
    assert [c[1][0] for c in p.calls([0, 0, 1], [0]) if c[0] == "launch"] == [7, 20]


def test_collectives_are_refused_where_ranks_could_disagree():
    p = example()
    assert not p.collective_free("layers")
    assert p.collective_free("harvest")
    q = native.GaudiProgram(1)
    q.section("step")
    q.take(0)
    with pytest.raises(ValueError, match="collective inside a take"):
        q.allgather(1, 2, 3)


def test_a_failed_step_says_how_many_calls_were_made():
    p = example()
    with pytest.raises(RuntimeError, match="takes 4 ops of 3") as raised:
        p.calls([4, 0, 1], ["layers"])
    # the first launch and the all-reduce were made before the take failed (an attribute the binding adds)
    assert getattr(raised.value, "issued") == 2
    with pytest.raises(RuntimeError, match="not below 2"):
        p.calls([0, 2, 1], ["layers"])
    with pytest.raises(RuntimeError, match="2 registers for a program of 3"):
        p.calls([0, 0], ["layers"])
    with pytest.raises(KeyError):
        p.calls([0, 0, 0], ["missing"])
    with pytest.raises(ValueError, match="section 2 of 2"):
        p.calls([0, 0, 0], [2])


def test_the_builder_checks_each_op_and_a_finished_program_is_fixed():
    p = native.GaudiProgram(1)
    with pytest.raises(ValueError, match="before the first section"):
        p.record(0)
    p.section("step")
    with pytest.raises(ValueError, match="register 1 of 1"):
        p.copy(1, 2, (0, 1, 1))
    with pytest.raises(ValueError, match="2 slots and 1 addresses"):
        p.launch(1, [0, 1], [5])
    with pytest.raises(ValueError, match="unequal lists"):
        p.copy_multi([1, 2], [3], [4, 5])
    with pytest.raises(ValueError, match="not finished"):
        p.calls([0], [0])
    p.take(0)
    with pytest.raises(ValueError, match="not ended"):
        p.finish()
    p.end()  # a failed finish leaves the builder as it was
    p.finish()
    with pytest.raises(ValueError, match="is finished"):
        p.record(0)


def test_several_programs_run_as_one_step():
    a = example()
    b = native.GaudiProgram(0)
    b.section("draft")
    b.copy_multi([1, 2], [8, 16], [3, 4])
    b.upload(0x10, 0x20, 64)
    b.upload_ring(0x30, 0x40, 8)
    b.memset(0x50, 7, 32)
    b.finish()
    calls = native.GaudiProgram.sequence_calls(
        [(b, "draft"), (a, "harvest")], [0, 0, 1]
    )
    assert calls == [
        ("copy_multi", [2, 1, 2, 8, 16, 3, 4]),
        ("upload", [0x10, 0x20, 64]),
        ("upload_ring", [0x30, 0x40, 8]),
        ("memset", [0x50, 7, 32]),
        ("download", [0x7000, 0x3000, 4]),
        ("record", [5]),
    ]


def test_the_glue_needs_the_loaded_runtime():
    with pytest.raises(ValueError, match="is not loaded"):
        native.GaudiGlue("/nonexistent/libresin_gaudi_rt.so")
