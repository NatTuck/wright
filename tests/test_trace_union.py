"""`trace_union.py` refuses a trace that cannot be shown to cover the cohort it claims.

The tool's truncation checks were all derived from the trace itself, and a 16-chunk trace covering
102 ms of a 16.38 s cohort passed every one of them (cases/wright/STATUS.md section 13). The
`--expect-seconds` gate is the check that can see that, so the assertions fail CLOSED: a failed or
unperformed check refuses the trace and suppresses every accounting number. These are synthetic
traces - no device, no profiler.
"""

from __future__ import annotations

import json

from scripts.trace_union import main

PERIOD = 2  # launches per unit, and distinct recipe ids per unit


def write_trace(path, *, units=2, phase="prefill", passes=2, config=True):
    """A minimal device trace: `units` units of PERIOD (Recipe Start, one X interval) launches."""
    events, ts = [], 0.0
    for _unit in range(units):
        for k in range(PERIOD):
            rid = 10 + k
            events.append(
                {
                    "ph": "V",
                    "pid": 2028,
                    "tid": 31,
                    "ts": ts,
                    "dur": 1000.0,
                    "name": "launch",
                    "cat": "Recipe Start",
                    "args": {"recipe_id": rid},
                }
            )
            events.append(
                {
                    "ph": "X",
                    "pid": 2028,
                    "tid": 31,
                    "ts": ts + 1.0,
                    "dur": 1000.0,
                    "name": "n1_rg_swiglu_f32",
                    "cat": "kernel",
                    "args": {"recipe_id": rid},
                }
            )
            ts += 1000.0
    path.write_text(json.dumps({"traceEvents": events}))
    if config:
        # one unit is PERIOD launches of 1000 us: the window of a full 2-unit trace is 4000 us
        beside = path.with_name(path.name.replace(".json", "-config.json"))
        beside.write_text(
            json.dumps(
                {
                    "phase": phase,
                    "passes": passes,
                    "unit": "prefill cohorts" if phase == "prefill" else "decode completions",
                    "trace_buffer_size": "0x8000000",
                }
            )
        )
    return path


def run(capsys, path, *argv):
    status = main([str(path), *argv])
    return status, capsys.readouterr().out


def test_a_complete_trace_is_quoted(tmp_path, capsys):
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json")
    status, out = run(capsys, trace, "--expect-seconds", "0.004")
    assert status == 0
    assert "REFUSED" not in out
    assert "## the step, three numbers" in out


def test_a_truncated_trace_is_refused(tmp_path, capsys):
    """One unit where the config recorded two: the old code printed FAIL and the numbers anyway."""
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json", units=1, passes=2)
    status, out = run(capsys, trace, "--expect-seconds", "0.004")
    assert status == 1
    assert "REFUSED - this trace cannot be quoted" in out
    assert "truncated or mis-split" in out
    assert "## the step, three numbers" not in out  # no number may be quoted
    assert "merged union over every device interval" not in out


def test_one_pass_cannot_be_checked_and_is_refused(tmp_path, capsys):
    """The B4 regression: with one recorded pass the unit check passes by construction."""
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json", units=1, passes=1)
    status, out = run(capsys, trace, "--expect-seconds", "0.002")
    assert status == 1
    assert "records 1 prefill pass" in out
    assert "## the step, three numbers" not in out


def test_a_window_that_does_not_match_the_cohort_is_refused(tmp_path, capsys):
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json")
    status, out = run(capsys, trace, "--expect-seconds", "16.38")
    assert status == 1
    assert "fragment of the cohort" in out
    assert "## the step, three numbers" not in out


def test_a_prefill_trace_without_the_cohort_claim_is_refused(tmp_path, capsys):
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json")
    status, out = run(capsys, trace)
    assert status == 1
    assert "without --expect-seconds" in out
    assert "## the step, three numbers" not in out


def test_a_trace_without_its_config_is_refused(tmp_path, capsys):
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json", config=False)
    status, out = run(capsys, trace, "--expect-seconds", "0.004")
    assert status == 1
    assert "no <case>-config.json" in out
    assert "## the step, three numbers" not in out


def test_the_refusal_is_written_to_out_so_a_stale_report_cannot_be_quoted(tmp_path, capsys):
    trace = write_trace(tmp_path / "prefill-ctx512000-b1-rank000.json", units=1)
    report = tmp_path / "report.md"
    status, out = run(
        capsys, trace, "--expect-seconds", "0.004", "--out", str(report)
    )
    assert status == 1
    assert report.read_text() == out
    assert "REFUSED" in report.read_text()
    assert "three numbers" not in report.read_text()
