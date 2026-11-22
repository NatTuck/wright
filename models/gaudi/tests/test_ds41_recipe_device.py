"""The branch's per-recipe device-column arithmetic tests — SKIPPED on main (Python-executor artifact).

origin/four-card-ds's test_ds41_recipe_device.py pins the Python executor's per-recipe device table:
the `raw = sum(t1 - t0)` bracket, the `HOST_CALL_S = 44us` bind+launch correction, and the second-sync
control subtracted once in `corrected = raw - host - sync_overhead`, through `_Op.run` in
recipe-device profiling mode and `DeepSeekV41GaudiRunner._report_recipe_device`.

Every symbol it drives is gone from main 0.3.0, which deleted the Python per-step executor for the
Rust `Ds41Executor` (see models/gaudi/ARCHITECTURE.md and tests/test_ds41_native_only.py):

  - runner.HOST_CALL_S / runner.DeepSeekV41GaudiRunner._report_recipe_device  -> absent
  - glm5_3_flash.model._RECIPE_DEVICE / LAUNCH_BY_RECIPE / SECOND_SYNC_BY_RECIPE
    / _LAUNCH_MODE / _do_bind / _do_launch                                  -> absent

So the test cannot run on the native path, and the accounting it guards (HOST_CALL_S, the Python
per-recipe launch bracket) is explicitly NOT carried over — it is a measurement artifact of the
deleted Python launch path. Porting it would reintroduce the very symbols the hard-rule gate forbids.

DEFERRED, not dropped: the per-recipe device attribution is re-established for the NATIVE executor in
the profiling phase (trace_union.py / steptimer.rs / the graph.py + runtime.cpp recipe_id attribution),
 coordinated separately. When that lands, re-add a native-path regression here that pins the native
per-recipe union arithmetic (price in `merged`, never summed work — manual ch2 §2.9), with its own
RED-first negative control. Until then this module skips with this reason rather than carrying a
Python-executor test that could only pass by resurrecting the Python executor.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "test_ds41_recipe_device guards the deleted Python executor's HOST_CALL_S / _report_recipe_device "
    "accounting; main runs steps in the Rust Ds41Executor. Deferred to the native profiling phase "
    "(see models/gaudi/ARCHITECTURE.md).",
    allow_module_level=True,
)