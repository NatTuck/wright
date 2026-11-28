"""``scripts/launch_timeline.py`` over a synthetic launch log.

The log is what ``runtime.cpp`` maps under ``RESIN_LAUNCH_LOG``: a 48-byte header, then one 16-byte record
``(ticks u64, recipe u32, pad u32)`` per launch, with a ``names.<pid>.tsv`` sidecar for the identities. These
gates check the tool reads that, ranks recipes, and finds a host gap — the thing the timeline exists for. The
gap gate is RED-provable: widen the hole and the reported gap must widen with it.
"""

from __future__ import annotations

import json
import struct

import pytest

from scripts.launch_timeline import HEADER, MAGIC, RECORD, main


def write_log(directory, pid: int, rows: list[tuple[float, int, str]], *, capacity: int = 4096) -> None:
    """``rows`` are ``(ns, recipe, name)``; the ticks are nanoseconds (ns_per_tick = 1) offset by a realistic
    non-zero TSC, because a zero tick count is what ends the log."""
    base = 1 << 40
    names = {}
    blob = bytearray(HEADER.pack(MAGIC, 1, RECORD.size, 1.0, base, capacity))
    for ns, ri, name in rows:
        names[ri] = name
        blob += RECORD.pack(base + int(ns), ri, 0)
    blob += b"\0" * (HEADER.size + capacity * RECORD.size - len(blob))
    (directory / f"launches.{pid}.bin").write_bytes(bytes(blob))
    (directory / f"names.{pid}.tsv").write_text(
        "# recipe\tname\n" + "".join(f"{ri}\t{name}\n" for ri, name in sorted(names.items()))
    )


def run(directory, *extra: str) -> None:
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
    assert "ffn_T2048" in out and "ixchain_T2048" in out
    # the two 1 ms holes are the only gaps at the default threshold (the 0.5 us step is below it)
    assert "host gaps >= 200 us: 2" in out
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["launches"] == 4
    assert [(g["us"], g["after"], g["before"]) for g in report["gaps"]] == [
        (1000.0, "ffn_T2048", "ixchain_T2048"),
        (1000.0, "ixchain_T2048", "ffn_T2048"),
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


def test_a_truncated_or_foreign_file_is_refused(tmp_path):
    (tmp_path / "launches.9.bin").write_bytes(b"not a log at all")
    with pytest.raises(SystemExit, match="not a launch log"):
        run(tmp_path)


def test_an_unnamed_recipe_is_visible(tmp_path, capsys):
    """A record whose recipe has no sidecar entry must not silently become an empty name."""
    write_log(tmp_path, 5, [(0.0, 0, "ffn_T2048")])
    (tmp_path / "names.5.tsv").write_text("# recipe\tname\n")
    run(tmp_path)
    assert "recipe-0" in capsys.readouterr().out


def test_a_window_reports_host_elapsed_and_its_device_floor(tmp_path, capsys):
    """A window's measured time is host elapsed; its floor is the device bytes its launches declare. The tool
    must print both and must not present their quotient as a ratio — host time is not device time."""
    rows = [(0.0, 0, "glm_embed_T2048"), (5_000_000.0, 1, "ds41_ffn_paged_T2048"),
            (9_000_000.0, 2, "ds41_owner_store_T2048_r1_G2048")]
    write_log(tmp_path, 11, rows)
    run(tmp_path, "--spans", "family", "--between", "glm_embed", "ds41_owner_store")
    out = capsys.readouterr().out
    assert "window 'glm_embed'..'ds41_owner_store': 3 launches" in out
    assert "9.0 ms of HOST elapsed" in out
    # the ffn_paged launch declares the whole per-card expert set: 384*3*5120*2304*0.5/4 bytes
    assert "MB, a" in out and "device floor at the full envelope" in out
    assert "this is not a ratio" in out


def test_span_floors_declare_a_regime_and_a_derivation():
    """Every floor entry must say which resource binds it and where its number comes from; a family with no
    formula must say so instead of silently reporting zero."""
    from scripts.span_floors import FLOORS, derivation, floor, family

    for name, entry in FLOORS.items():
        assert entry.regime in ("tpc", "dma", "flops"), name
        assert entry.derivation, name
    moved, regime, rate = floor("ds41_ffn_paged_T2048")
    assert regime == "dma" and rate > 0
    assert moved == pytest.approx(384 * 3 * 5120 * 2304 * 0.5 / 4)
    # the MoE floor does not depend on T above one row per expert
    assert floor("ds41_ffn_paged_T512")[0] == moved
    assert floor("ds41_attn_win_T2048_S16_C384_Q128") is None
    assert "UNKNOWN" in derivation("ds41_attn_win_T2048")
    assert family("ds41_attn_out_tile_T512_Q128") == "ds41_attn_out_tile"


def test_span_grade_labels_a_launch_by_its_nodes():
    """The trace carries no recipe name, so a launch is named by the nodes inside its window. The markers
    must separate the families, and an unmatched window must stay unclassified rather than be guessed."""
    from scripts.span_grade import classify

    assert classify({"n10_rg_ds_attn_online_f32", "n10_rg_ds_attn_acc_f32"}) == "ds41_attn_out_tile"
    assert classify({"n16_rg_ds_ix_merge_f32"}) == "ds41_ix_chain"
    assert classify({"n10_rg_moe_asg_f32", "n19_rg_mx4_decs_f32"}) == "ds41_ffn_paged"
    assert classify({"n22_rg_ds_gather_list_blk_f4_f32"}) is None


def test_the_attention_tile_floor_is_dominated_by_its_f32_state():
    """The 10-block online softmax moves a (D, nh, Q) f32 accumulator twice per block plus the PV product;
    those two terms must dominate, because that is the traffic worth removing."""
    from scripts.span_floors import ATTN_BLOCKS, floor

    moved, regime, rate = floor("ds41_attn_out_tile_T512_Q128_B64_A27588")
    assert regime == "tpc"
    state = ATTN_BLOCKS * (2 * 512 * 64 * 128 * 4 + 2 * 512 * 64 * 128 * 4)  # pv + acc_o, per block
    assert state / moved > 0.6
    # and it is not a small-tile case: one launch moves hundreds of MB (526 MB at B=128; it was 1019 MB at
    # B=64, before ATTN_KEY_BLK doubled and the block count halved)
    assert moved > 5e8
