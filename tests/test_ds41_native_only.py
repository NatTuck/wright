"""HARD RULE gate: DeepSeek-V4.1 steps run ONLY in the Rust native executor.

Python loads weights, builds/compiles graphs, and RECORDS native programs once per shape at startup.
It must not run a per-step loop: no Python recipe launches, no per-step table fill, no per-step table
upload, no Python forward/enqueue step entry. The branch origin/four-card-ds ran steps in a Python
executor (model.launch / runner.enqueue / _fill / .run(lib) / h2d_async(self.table) / HOST_CALL_S);
main 0.3.0 deleted it for resin._resin.Ds41Executor. This gate fails if that Python executor is
reintroduced. See models/gaudi/ARCHITECTURE.md.

The gate is CPU-only (no device, no server). Its negative control runs the SAME detector against the
branch's Python model.py/runner.py (git show origin/four-card-ds:...) and requires every forbidden
pattern to fire there - proving the gate can fail, not just that it passes today (manual ch3/ch9:
a gate that cannot fail on the unfixed code is decoration).
"""

from __future__ import annotations

import inspect
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "models/gaudi/resin_models_gaudi/deepseek_v41/model.py"
_RUNNER = "models/gaudi/resin_models_gaudi/deepseek_v41/runner.py"
_NATIVE = "models/gaudi/resin_models_gaudi/deepseek_v41/native.py"

# (regex, description). Each MUST match the branch's Python executor and MUST NOT match main's
# native recording path. `def launch(op)` (nested recording helper) and `native.launch(...)` are
# allowed; `def launch(self` (a per-step method) and `.run(lib)` (a per-step launch) are not.
_MODEL_PATTERNS = [
    (r"\.run\(lib\)", "per-step Python recipe launch (.run(lib))"),
    (r"def launch\(\s*self", "per-step Python model.launch method"),
    (r"def _fill\(\s*self", "per-step Python table fill (_fill)"),
    (r"h2d_async\(self\.table", "per-step Python table-image upload"),
    (r"def forward\(\s*self", "per-step Python model.forward"),
    (r"def sample\(\s*self", "per-step Python model.sample"),
]
_RUNNER_PATTERNS = [
    (r"def forward\(self", "per-step Python runner.forward"),
    (r"def enqueue\(self", "per-step Python runner.enqueue"),
    (r"HOST_CALL_S", "Python per-recipe launch-cost accounting (HOST_CALL_S)"),
    (r"def _report_recipe_device", "Python per-recipe device table"),
    (r"def _harvest\(self", "Python in-flight sample harvest"),
    (r"self\._inflight", "Python in-flight step list"),
]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text()


def _hits(source: str, patterns) -> list[tuple[str, int, str]]:
    """(description, lineno, line) for every forbidden pattern present in source.

    Scans the whole text (not line by line) so a multi-line signature like ``def forward(\\n self``
    is caught by ``def forward\\(\\s*self``; the reported line is the match's start line.
    """
    out = []
    lines = source.splitlines()
    for regex, desc in patterns:
        for m in re.finditer(regex, source):
            n = source.count("\n", 0, m.start()) + 1
            out.append((desc, n, lines[n - 1].strip()))
    return out


def _branch_source(rel: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"origin/four-card-ds:{rel}"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"origin/four-card-ds:{rel} unavailable: {proc.stderr.strip()}")
    return proc.stdout


# ------------------------------------------------------------------ (a) the step entry is native
def test_step_entry_is_the_rust_executor():
    """Ds41Executor.forward/enqueue are Rust method descriptors, not Python functions."""
    from resin._resin import Ds41Executor

    for name in ("forward", "enqueue", "release"):
        member = getattr(Ds41Executor, name)
        assert not inspect.isfunction(member), f"Ds41Executor.{name} is a Python function"
        assert isinstance(member, type(Ds41Executor.forward)), (
            f"Ds41Executor.{name} is not a native method descriptor"
        )


def test_runner_binds_the_executor_not_a_python_step():
    """The runner class defines no Python forward/enqueue step; attach binds the executor's, and the
    native executor is built from deepseek_v41.native.build_executor."""
    from resin_models_gaudi.deepseek_v41.runner import DeepSeekV41GaudiRunner

    assert "forward" not in vars(DeepSeekV41GaudiRunner), "runner defines a Python forward step"
    assert "enqueue" not in vars(DeepSeekV41GaudiRunner), "runner defines a Python enqueue step"

    runner = _read(_RUNNER)
    for binding in (
        "self.forward = self.executor.forward",
        "self.enqueue = self.executor.enqueue",
        "self.release = self.executor.release",
        "self.executor = build_executor(self)",
        "from resin_models_gaudi.deepseek_v41.native import build_executor",
    ):
        assert binding in runner, f"runner.py missing native binding: {binding!r}"


def test_native_executor_is_built_from_the_rust_type():
    """native.py builds resin._resin.Ds41Executor (the Rust step runner), not a Python loop."""
    native = _read(_NATIVE)
    assert "from resin._resin import Ds41Executor" in native
    assert "return Ds41Executor(" in native


# ------------------------------------------------------------------ (b) no per-step Python orchestration
def test_model_has_no_per_step_python_orchestration():
    hits = _hits(_read(_MODEL), _MODEL_PATTERNS)
    assert not hits, "per-step Python orchestration in model.py:\n" + "\n".join(
        f"  {desc} @ {n}: {line}" for desc, n, line in hits
    )


def test_runner_has_no_per_step_python_orchestration():
    hits = _hits(_read(_RUNNER), _RUNNER_PATTERNS)
    assert not hits, "per-step Python orchestration in runner.py:\n" + "\n".join(
        f"  {desc} @ {n}: {line}" for desc, n, line in hits
    )


# ------------------------------------------------------------------ negative control: the gate CAN fail
def test_gate_fires_on_the_branch_python_model():
    """Every forbidden model pattern must match the branch's Python model.py. If any pattern stops
    matching the branch it is dead (decoration) and the gate could not catch a reintroduction."""
    source = _branch_source(_MODEL)
    hits = _hits(source, _MODEL_PATTERNS)
    fired = {desc for desc, _n, _line in hits}
    expected = {desc for _rx, desc in _MODEL_PATTERNS}
    assert fired == expected, (
        "gate patterns that did NOT fire on the branch Python model (dead detectors): "
        f"{sorted(expected - fired)}"
    )


def test_gate_fires_on_the_branch_python_runner():
    source = _branch_source(_RUNNER)
    hits = _hits(source, _RUNNER_PATTERNS)
    fired = {desc for desc, _n, _line in hits}
    expected = {desc for _rx, desc in _RUNNER_PATTERNS}
    assert fired == expected, (
        "gate patterns that did NOT fire on the branch Python runner (dead detectors): "
        f"{sorted(expected - fired)}"
    )


def test_branch_runner_would_fail_the_native_binding_check():
    """The branch runner has a Python forward/enqueue and no executor binding: the (a) check fails
    on it, proving the gate rejects the Python executor rather than merely passing on main."""
    source = _branch_source(_RUNNER)
    assert "def forward(self" in source and "def enqueue(self" in source
    assert "self.forward = self.executor.forward" not in source