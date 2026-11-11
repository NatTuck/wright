"""The per-recipe device column's arithmetic, host side only (no device, no server).

The documented formula (cases/wright/measurements/profile/SUMMARY-recipe-device.txt) is

    raw           = sum over recipes of (t1 - t0)
    host          = calls x 44 us
    sync_overhead = sum of the second syncs
    corrected     = raw - host - sync_overhead

so the second-sync control is subtracted ONCE, in the corrected column. The code used to take it
out of the bracket as well (`elapsed = ... - second`), which subtracted it twice and left the
corrected column ~3.5% low. These tests pin the bracket's meaning with a scripted clock.
"""

from __future__ import annotations

import numpy as np
import pytest

from resin_models_gaudi.glm5_3_flash import model as M


class _Recipe:
    """Enough of a `Recipe` for `_Op.run`: a handle, a name, a device to sync, one slot."""

    name = "ds41_test_recipe"
    workspace = 0
    handle = 1

    class _Device:
        @staticmethod
        def sync() -> None:
            pass

    device = _Device()

    def slot(self, name: str) -> int:
        return 0


@pytest.fixture
def recipe_device(monkeypatch):
    """`_Op.run` in recipe-device mode, with the host clock scripted and the tables cleared."""
    monkeypatch.setattr(M, "_PROFILE_STEP", True)
    monkeypatch.setattr(M, "_RECIPE_DEVICE", True)
    monkeypatch.setattr(M, "_LAUNCH_MODE", "full")
    monkeypatch.setattr(M, "_do_bind", lambda lib, args: 0)
    monkeypatch.setattr(M, "_do_launch", lambda lib, handle: 0)
    M.LAUNCH_BY_RECIPE.clear()
    M.SECOND_SYNC_BY_RECIPE.clear()
    M.LAST_CALLS.clear()
    M._SYNC_OVERHEAD_TOTAL[0] = 0.0
    M._SYNC_OVERHEAD_COUNT[0] = 0
    yield
    M.LAUNCH_BY_RECIPE.clear()
    M.SECOND_SYNC_BY_RECIPE.clear()
    M.LAST_CALLS.clear()
    M._SYNC_OVERHEAD_TOTAL[0] = 0.0
    M._SYNC_OVERHEAD_COUNT[0] = 0


def test_the_bracket_keeps_the_second_sync(recipe_device, monkeypatch):
    """raw is `t1 - t0` as measured: the control is inside it, and subtracted only in corrected.

    Scripted clock: start 0.0, first sync done 1.0, second sync 1.0 -> 1.5, bracket ends 2.0.
    So the bracket is 2.0 ms of which 0.5 ms is the control, and
        corrected = raw - host - sync_overhead = 2.0 - 44us - 0.5
    A double subtraction shows up as raw == 1.5 here.
    """
    clock = iter([0.0, 1.0, 1.5, 2.0])
    monkeypatch.setattr(M.time, "perf_counter", lambda: next(clock))

    M._Op(_Recipe(), {"x": 0}).run(object())

    calls, raw, _workspace = M.LAUNCH_BY_RECIPE[_Recipe.name]
    overhead = M.SECOND_SYNC_BY_RECIPE[_Recipe.name][1]
    assert calls == 1
    assert overhead == pytest.approx(0.5)
    assert raw == pytest.approx(2.0)  # the bracket as measured, second sync included
    assert M._SYNC_OVERHEAD_TOTAL[0] == pytest.approx(0.5)
    assert M._SYNC_OVERHEAD_COUNT[0] == 1
    assert M.LAST_CALLS == [pytest.approx(2.0)]

    # the printed corrected column, exactly as runner._report_recipe_device computes it
    from resin_models_gaudi.deepseek_v41.runner import HOST_CALL_S

    corrected = raw - calls * HOST_CALL_S - overhead
    assert corrected == pytest.approx(1.5 - HOST_CALL_S)
    # and it is NOT the doubly-subtracted 1.0 - host
    assert corrected != pytest.approx(1.0 - HOST_CALL_S)


def test_many_calls_stay_additive(recipe_device, monkeypatch):
    """Both corrections are per call, so a family's corrected union is still additive."""
    ticks, base = [], 0.0
    for _ in range(3):
        ticks += [base, base + 1.0, base + 1.5, base + 2.0]
        base += 2.0
    clock = iter(ticks)
    monkeypatch.setattr(M.time, "perf_counter", lambda: next(clock))

    for _ in range(3):
        M._Op(_Recipe(), {"x": 0}).run(object())

    calls, raw, _workspace = M.LAUNCH_BY_RECIPE[_Recipe.name]
    overhead = M.SECOND_SYNC_BY_RECIPE[_Recipe.name][1]
    assert calls == 3
    assert raw == pytest.approx(6.0)
    assert overhead == pytest.approx(1.5)
    assert M._SYNC_OVERHEAD_TOTAL[0] == pytest.approx(1.5)
    np.testing.assert_allclose(
        raw - calls * 44e-6 - overhead, 3 * (2.0 - 44e-6 - 0.5)
    )
