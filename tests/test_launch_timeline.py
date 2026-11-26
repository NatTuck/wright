"""``scripts/launch_timeline.py`` over a synthetic launch log.

The log is what ``runtime.cpp`` writes from ``RESIN_LAUNCH_LOG``: one ``<ns> <ticks> <recipe> <name>`` line
per launch. These gates check the tool reads it, ranks recipes, and finds a host gap — the thing the
timeline exists for. The gap gate is RED-provable: widen the hole and the reported gap must widen with it.
"""

from __future__ import annotations

import json

import pytest

from scripts.launch_timeline import main


def write_log(directory, pid: int, rows: list[tuple[float, int, str]]) -> None:
    with (directory / f"launches.{pid}.tsv").open("w") as stream:
        stream.write("# ns\tticks\trecipe\tname\n")
        for ns, ri, name in rows:
            stream.write(f"{ns:.0f}\t{int(ns)}\t{ri}\t{name}\n")


def run(directory, *extra: str) -> None:
    main.__globals__["__name__"] = "__main__"
    import sys

    argv = sys.argv
    sys.argv = ["launch_timeline.py", str(directory), *extra]
    try:
        main()
    finally:
        sys.argv = argv


def test_reads_ranks_and_finds_the_host_gap(tmp_path, capsys):
    rows = [
        (0.0, 0, "ffn_T2048"),
        (1_000_000.0, 1, "ixchain_T2048"),
        (2_000_000.0, 0, "ffn_T2048"),
        (2_000_500.0, 1, "ixchain_T2048"),
    ]
    write_log(tmp_path, 4242, rows)
    run(tmp_path, "--json", str(tmp_path / "out.json"))
    out = capsys.readouterr().out
    assert "pid 4242: 4 launches" in out
    # two ffn launches 2 ms apart, two ixchain launches 1 ms apart
    assert "ffn_T2048" in out and "ixchain_T2048" in out
    # the two 1 ms holes are the only gaps at the default threshold (the 0.5 us step is below it)
    assert "host gaps >= 200 us: 2" in out
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["launches"] == 4
    # both 1 ms holes, each named by the launches on either side of it
    assert [(g["us"], g["after"], g["before"]) for g in report["gaps"]] == [
        (1000.0, "ixchain_T2048", "ffn_T2048"),
        (1000.0, "ffn_T2048", "ixchain_T2048"),
    ]


def test_the_gap_is_red_provable(tmp_path, capsys):
    """Widen the hole; the reported gap must widen. A tool that reported a constant would pass the first
    gate and fail this one."""
    narrow = [(0.0, 0, "a"), (1_000_000.0, 1, "b")]
    wide = [(0.0, 0, "a"), (5_000_000.0, 1, "b")]
    for rows, expect in ((narrow, 1000.0), (wide, 5000.0)):
        write_log(tmp_path, 7, rows)
        run(tmp_path, "--json", str(tmp_path / "o.json"))
        capsys.readouterr()
        report = json.loads((tmp_path / "o.json").read_text())
        assert report["gaps"][0]["us"] == pytest.approx(expect, rel=1e-6)


def test_picks_the_rank_with_most_launches(tmp_path, capsys):
    write_log(tmp_path, 1, [(0.0, 0, "a")])
    write_log(tmp_path, 2, [(0.0, 0, "a"), (1.0, 1, "b"), (2.0, 2, "c")])
    run(tmp_path)
    assert "pid 2: 3 launches" in capsys.readouterr().out
