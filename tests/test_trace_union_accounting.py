"""The accounting half of `trace_union.py`: the union arithmetic itself.

The refusal tests above pin when a trace may be quoted. These pin what the quoted numbers are, on a
synthetic trace whose every interval was chosen by hand, so the three-number identity is checked
against known values rather than against the tool's own output:

  * the same trace parsed twice prints byte-identical output (an instrument, not a lottery);
  * per-recipe sum <= merged union <= span, and recipe-less = merged - per-recipe exactly;
  * a recipe's figure is a UNION of its nodes, never a sum of their durations;
  * the recipe_id-0 share is reported as a component of (a) and is NOT additive: with work that
    overlaps a recipe, (b) + (recipe_id 0) exceeds (a), and (a) is still the merged union.

Two real errors of this campaign would have failed here: a 14.3 ms figure that exceeded its own
merged union, and a prefill fragment read as a cohort. Synthetic traces - no device, no profiler.
"""

from __future__ import annotations

import json
import re

from scripts.trace_union import main

PERIOD = 3  # launches per unit, and distinct recipe ids per unit (see LAUNCHES)

# The accounting fixture, in microseconds from the start of a unit. Chosen so that every number
# below is hand-computable and so that each hazard is present:
#   * two nodes of ONE recipe overlap  -> its union (60) is less than the sum of its nodes (80);
#   * one recipe's nodes sit on two modules (2028 and 6028) -> the +4000 tag must be normalised;
#   * a Host API row exists and must not enter any union;
#   * recipe_id-0 work both overlaps a recipe (30-50 inside 0-95) and stands alone (160-180).
UNIT = 1000.0
NODES = (
    # (recipe_id, engine_type, engine_index, name, start, end)
    (10, 2028, 0, "n1_rg_swiglu_f32", 0.0, 40.0),
    (10, 6028, 0, "n2_fp8_gemm", 45.0, 95.0),  # module 1: normalises to engine 2028
    (11, 2024, 0, "n3_attn_f32", 100.0, 150.0),
    (12, 2028, 0, "n4_rg_mx4_decs_f32", 200.0, 240.0),
    (12, 2028, 0, "n5_rg_rope_f32", 220.0, 260.0),  # overlaps n4: union 60, sum 80
    (0, 2034, 0, "copy_h2d", 30.0, 50.0),  # recipe-less, overlapping recipe 10
    (0, 2157, 0, "barrier_sync", 160.0, 180.0),  # recipe-less, free-standing
)
LAUNCHES = ((10, 0.0), (11, 100.0), (12, 200.0))

# Hand-computed expectations for one unit.
EXPECTED_PER_RECIPE = (
    90.0 + 50.0 + 60.0
)  # 200: union within a recipe, sum across recipes
EXPECTED_PER_RECIPE_SUMMED = (
    90.0 + 50.0 + 80.0
)  # 220: the same, forgetting the within-recipe union
EXPECTED_EVENT_SUM = 40 + 50 + 50 + 40 + 40 + 20 + 20  # 260: every duration added up
EXPECTED_RECIPE_LESS = 20.0 + 20.0  # 40: the two recipe_id-0 intervals, merged
EXPECTED_MERGED = 95.0 + 50.0 + 20.0 + 60.0  # 225: device busy for the unit
EXPECTED_SPAN = 260.0  # first start to last end


def write_accounting_trace(path, *, units=2, passes=2):
    """A device trace whose intervals are known by construction (see NODES above)."""
    events = []
    for unit in range(units):
        base = unit * UNIT
        for rid, start in LAUNCHES:
            events.append(
                {
                    "ph": "V",
                    "pid": 2028,
                    "tid": 31,
                    "ts": base + start,
                    "dur": 0.0,
                    "name": "launch",
                    "cat": "Recipe Start",
                    "args": {"recipe_id": rid},
                }
            )
        for rid, engine, index, name, start, end in NODES:
            for phase, ts in (("B", base + start), ("E", base + end)):
                events.append(
                    {
                        "ph": phase,
                        "pid": engine,
                        "tid": index,
                        "ts": ts,
                        "dur": 0.0,
                        "name": name,
                        "cat": "kernel",
                        "args": {"recipe_id": rid},
                    }
                )
        # A host-side row: it carries a duration but belongs to no device union.
        events.append(
            {
                "ph": "X",
                "pid": 16425,
                "tid": 0,
                "ts": base + 10.0,
                "dur": 5.0,
                "name": "h2d_copy",
                "cat": "Host API",
                "args": {"recipe_id": 0},
            }
        )
    path.write_text(json.dumps({"traceEvents": events}))
    path.with_name(path.name.replace(".json", "-config.json")).write_text(
        json.dumps(
            {
                "phase": "decode",
                "passes": passes,
                "unit": "decode steps",
                "trace_buffer_size": "0x8000000",
            }
        )
    )
    return path


def three_numbers(out: str) -> dict[str, float]:
    """(a), (b), (c), the recipe_id-0 share and the span, as printed.

    Every field is required: a rename in the report fails here rather than silently returning a
    plausible table, which is the whole point of testing an instrument.
    """
    patterns = {
        "a": r"\(a\) merged union over every device interval\s*=\s*([\d.]+) us",
        "b": r"\(b\) sum of per-recipe unions\s*=\s*([\d.]+) us",
        "c": r"\(c\) recipe-less device work \(a\) minus \(b\)\s*=\s*([\d.]+) us",
        "rid0": r"of which intervals with recipe_id 0 \(host copies, barriers\)\s*=\s*([\d.]+) us",
        "span": r"step span \(mean over the units\)\s*=\s*([\d.]+) us",
    }
    found = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, out)
        assert match, (
            f"the report no longer prints `{key}` in a form the test can read:\n{out}"
        )
        found[key] = float(match.group(1))
    return found


def run_trace(capsys, path, *argv):
    status = main([str(path), *argv])
    return status, capsys.readouterr().out


def test_the_same_trace_parsed_twice_prints_the_same_report(tmp_path, capsys):
    trace = write_accounting_trace(tmp_path / "decode-ctx4096-b8-rank000.json")
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    status, out_first = run_trace(capsys, trace, "--out", str(first))
    assert status == 0
    status, out_second = run_trace(capsys, trace, "--out", str(second))
    assert status == 0

    assert out_first == out_second
    assert first.read_text() == second.read_text() == out_first


def test_the_three_numbers_hold_the_identity(tmp_path, capsys):
    """per-recipe sum <= merged union <= span, on one scope, with (c) = (a) - (b)."""
    trace = write_accounting_trace(tmp_path / "decode-ctx4096-b8-rank000.json")
    status, out = run_trace(capsys, trace)
    assert status == 0
    numbers = three_numbers(out)

    assert numbers["b"] <= numbers["a"] <= numbers["span"], numbers
    # (c) is (a) - (b) by definition, so asserting that identity would only re-check a
    # subtraction. What has content is the OTHER number two lines below it: the recipe_id-0
    # union is larger than (c), because the copy overlaps recipe work. If the two were ever
    # conflated, a reader would price the recipe-less term at 20 us instead of 40 us.
    assert numbers["rid0"] == EXPECTED_RECIPE_LESS, numbers
    assert numbers["rid0"] > numbers["c"], (
        "the recipe_id-0 union must exceed (a)-(b) when recipe-less work overlaps a recipe; "
        f"got rid0={numbers['rid0']} and c={numbers['c']}"
    )
    assert numbers["b"] == EXPECTED_PER_RECIPE
    assert numbers["a"] == EXPECTED_MERGED
    assert numbers["c"] == EXPECTED_MERGED - EXPECTED_PER_RECIPE
    assert numbers["span"] == EXPECTED_SPAN


def test_a_recipe_is_a_union_of_its_nodes_not_a_sum(tmp_path, capsys):
    """Recipe 12's two nodes overlap: quoting the sum of durations would inflate every figure."""
    trace = write_accounting_trace(tmp_path / "decode-ctx4096-b8-rank000.json")
    status, out = run_trace(capsys, trace)
    assert status == 0
    numbers = three_numbers(out)

    assert numbers["b"] == EXPECTED_PER_RECIPE
    assert numbers["b"] != EXPECTED_PER_RECIPE_SUMMED, (
        "the within-recipe union was dropped"
    )
    assert numbers["b"] != EXPECTED_EVENT_SUM, "event durations were summed"
    assert "union within a recipe, sum across recipes" in out


def test_recipe_less_is_merged_minus_per_recipe_and_rid0_is_not_additive(
    tmp_path, capsys
):
    """The recipe_id-0 share is a component of (a), and it overlaps the recipes: not an addend."""
    trace = write_accounting_trace(tmp_path / "decode-ctx4096-b8-rank000.json")
    status, out = run_trace(capsys, trace)
    assert status == 0
    numbers = three_numbers(out)

    # recipe-less work is exactly what the per-recipe table cannot see
    assert numbers["c"] == numbers["a"] - numbers["b"]
    # ... and the recipe_id-0 share is reported beside it, as a sub-part of (a)
    assert numbers["rid0"] == EXPECTED_RECIPE_LESS
    assert "of which" in out
    # it is NOT additive: half of the recipe_id-0 work overlaps recipe 10, so adding it to the
    # per-recipe sum overshoots the merged union, which is still the device-busy figure.
    assert numbers["b"] + numbers["rid0"] > numbers["a"]
    assert numbers["a"] == EXPECTED_MERGED
    assert "not disjoint" in out


def test_the_host_api_row_enters_no_union(tmp_path, capsys):
    """A Host API row carries a duration; counting it would put host time into device busy."""
    trace = write_accounting_trace(tmp_path / "decode-ctx4096-b8-rank000.json")
    status, out = run_trace(capsys, trace)
    assert status == 0
    numbers = three_numbers(out)
    # the 5 us host row sits inside 0-95 us, so a device union that counted it would be
    # unchanged only by luck; the real check is that (a) is exactly the device intervals.
    assert numbers["a"] == EXPECTED_MERGED


def test_expect_seconds_falls_back_to_the_trace_config(tmp_path, capsys):
    """The gate is only as external as its number: the trace's own config must be the source.

    The hook writes `prefill_seconds` into <case>-config.json from the run that produced the
    trace, so the completeness gate can use the harness's own timing without an operator typing
    it. The report must say which source was used, because a number from another boot (or from
    the command line) is a weaker claim than one from this run.
    """
    trace = write_accounting_trace(tmp_path / "prefill-ctx8192-b1-rank000.json")
    config = json.loads(
        trace.with_name("prefill-ctx8192-b1-rank000-config.json").read_text()
    )
    config["phase"] = "prefill"
    config["prefill_seconds"] = EXPECTED_SPAN / 1e6  # one unit's window, in seconds
    trace.with_name("prefill-ctx8192-b1-rank000-config.json").write_text(
        json.dumps(config)
    )

    status, out = run_trace(capsys, trace)
    assert status == 0, out
    assert "from the trace's config" in out, out
    assert "ratio 1.000" in out, out


def test_a_prefill_trace_with_no_timing_anywhere_is_refused(tmp_path, capsys):
    """No prefill_seconds in the config and no --expect-seconds: the gate cannot be performed.

    A prefill trace whose completeness cannot be checked is not quotable, and the refusal must
    print no accounting numbers.
    """
    trace = write_accounting_trace(tmp_path / "prefill-ctx8192-b1-rank000.json")
    config = json.loads(
        trace.with_name("prefill-ctx8192-b1-rank000-config.json").read_text()
    )
    config["phase"] = "prefill"
    config.pop("prefill_seconds", None)
    trace.with_name("prefill-ctx8192-b1-rank000-config.json").write_text(
        json.dumps(config)
    )

    status, out = run_trace(capsys, trace)
    assert status == 1, out
    assert "REFUSED" in out, out
    assert "(a) merged union" not in out, out


def test_a_tiny_configured_buffer_does_not_refuse_a_complete_trace(tmp_path, capsys):
    """The buffer ratio is information, not a check -- measured, not assumed.

    A real prefill trace configured for 256 MiB recorded 49,128,810 events and is complete, so
    `traceBufferSize` does not bound the recorded count. A capacity division therefore says
    nothing about truncation, and it must not be able to refuse a trace (nor be "excusable": a
    span is invariant to losing interior events, which is exactly what an event-count check was
    supposed to catch). Set the configured buffer absurdly small and the trace must still pass.
    """
    trace = write_accounting_trace(tmp_path / "prefill-ctx8192-b1-rank000.json")
    config_path = trace.with_name("prefill-ctx8192-b1-rank000-config.json")
    config = json.loads(config_path.read_text())
    config["phase"] = "prefill"
    config["prefill_seconds"] = EXPECTED_SPAN / 1e6
    config["trace_buffer_size"] = "0x1000"  # 4 KiB: ~64 events, against hundreds recorded
    config_path.write_text(json.dumps(config))

    status, out = run_trace(capsys, trace)
    assert status == 0, out
    assert "NOT a check" in out, out
    assert "REFUSED" not in out, out
    # the interior-loss detectors are the stated basis for completeness
    assert "B/E pairing" in out, out
    assert "recipe attribution" in out, out
