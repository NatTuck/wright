"""The device zeroing of a host copy's not-written-whole ranges (``native._device_zero``).

The fill used to zero those ranges on the host — 40-49 ms of every prefill chunk, with the device idle.
``native.py``'s io program now clears them on the device instead, in the upload section, and ``fill`` skips
its own zeroing. These gates pin the contract that makes that a pure move:

  * one memset per range, at the range's offset in the device table, for its length in 32-bit words, value 0;
  * the memsets come AFTER the upload (so the upload cannot overwrite them) and BEFORE the step's sections;
  * an empty range is skipped;
  * RED: a range the function fails to emit shows up as a missing call.

They need no card: ``GaudiProgram.calls`` records the stream calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import resin._resin as native

pytestmark = pytest.mark.skipif(
    not hasattr(native, "GaudiProgram"), reason="resin built without the gaudi feature"
)

R_BYTES = 13


class Registers:
    R_BYTES = R_BYTES


def _program(zero, table=0x1_0000_0000, nbytes=1 << 20):
    from resin_models_gaudi.deepseek_v41.native import _device_zero

    io = native.GaudiProgram(64)
    io.section("upload")
    io.upload(table, (0x7f00_0000_0000, 12, 0), (0, R_BYTES, 1))
    m = SimpleNamespace(table=table, tables={0: SimpleNamespace(zero=zero)})
    _device_zero(io, Registers, m, 0)
    io.finish()
    registers = [0] * 64
    registers[R_BYTES] = nbytes
    return io.calls(registers, ["upload"])


def test_one_memset_per_range_after_the_upload():
    zero = [(0x100, 0x300), (0x400, 0x480)]
    calls = _program(zero)
    kinds = [c[0] for c in calls]
    assert kinds[0] == "upload", "the upload must come first"
    memsets = [c for c in calls if c[0] == "memset"]
    assert [m[1][0] for m in memsets] == [0x1_0000_0100, 0x1_0000_0400]
    assert [m[1][1] for m in memsets] == [0, 0]
    assert [m[1][2] for m in memsets] == [(0x300 - 0x100) // 4, (0x480 - 0x400) // 4]
    # in the real program a select/record pair follows; what matters here is upload -> memsets, in order
    assert kinds == ["upload", "memset", "memset"]


def test_an_empty_range_is_skipped():
    calls = _program([(0x100, 0x100), (0x200, 0x240)])
    memsets = [c for c in calls if c[0] == "memset"]
    assert len(memsets) == 1 and memsets[0][1][0] == 0x1_0000_0200


def test_no_ranges_no_memsets():
    calls = _program([])
    assert not [c for c in calls if c[0] == "memset"]


def test_red_a_dropped_range_is_a_missing_call():
    """The gate must be able to fail: a zero list with one range missing emits one fewer memset."""
    both = [c for c in _program([(0x100, 0x300), (0x400, 0x480)]) if c[0] == "memset"]
    one = [c for c in _program([(0x100, 0x300)]) if c[0] == "memset"]
    assert len(both) == 2 and len(one) == 1
