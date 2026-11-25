"""Device-trace union arithmetic on synthetic TSVs (host only, no card)."""

from scripts.trace_union import (
    device_intervals,
    is_device,
    merge,
    union_table,
)

HEADER = "ph\tengine_type\tengine_index\tts\tdur\tname\tcategory\tmeta"


def row(ph, engine, index, ts, name, category, dur=0.0):
    return f"{ph}\t{engine}\t{index}\t{ts}\t{dur}\t{name}\t{category}\t"


def write(path, lines):
    path.write_text("\n".join([HEADER, *lines]) + "\n")
    return path


def test_merge_is_a_union_not_a_sum():
    assert merge([(0.0, 10.0), (5.0, 20.0)]) == 20.0  # overlapping -> merged
    assert merge([(0.0, 10.0), (30.0, 40.0)]) == 20.0  # disjoint -> both
    assert merge([]) == 0.0


def test_host_rows_and_flow_rows_are_not_device_events():
    assert not is_device({"ph": "X", "engine_type": "16389", "category": "Host API"})
    assert not is_device({"ph": "f", "engine_type": "16389", "category": ""})
    assert is_device({"ph": "B", "engine_type": "2028", "category": "SP 1"})


def test_tpc_kernel_window_merges_instances_into_a_union(tmp_path):
    # two TPC instances of one node, each with a command window and a kernel window
    lines = [
        row("B", 2028, 21, 0.0, "n1_rg_k", "SP 1"),
        row("B", 2028, 21, 1.0, "n1_rg_k", "SP 8"),
        row("E", 2028, 21, 5.0, "n1_rg_k", "SP 3"),
        row("E", 2028, 21, 9.0, "n1_rg_k", "SP 2"),
        row("B", 2028, 22, 2.0, "n1_rg_k", "SP 8"),
        row("E", 2028, 22, 4.0, "n1_rg_k", "SP 3"),
        row("V", 0, 0, 0.0, "", "Recipe Start", dur=10.0),
    ]
    trace = write(tmp_path / "t.tsv", lines)
    assert device_intervals(trace) == [
        ("TPC", "n1_rg_k", 1.0, 5.0),
        ("TPC", "n1_rg_k", 2.0, 4.0),
    ]
    table = union_table(trace)
    assert table["window"] == "preferred"
    assert table["merged_us"] == 4.0  # [1,5] union [2,4] = [1,5]
    assert table["nodes"] == [
        {"engine": "TPC", "node": "n1_rg_k", "intervals": 2, "union_us": 4.0}
    ]
    assert table["launches"] == 1 and table["launch_window_us"] == 10.0
    assert table["attributed"] is True


def test_every_engine_pairs_its_own_marker_ids(tmp_path):
    """TPC, MME, EDMA and PDMA number their sync points differently; all must appear."""
    lines = [
        row("B", 2028, 21, 0.0, "n1_tpc", "SP 8"),
        row("E", 2028, 21, 5.0, "n1_tpc", "SP 3"),
        row("B", 2024, 3, 0.0, "n2_gemm", "SP 0"),
        row("E", 2024, 3, 6.0, "n2_gemm", "SP 1"),
        row("B", 2018, 7, 0.0, "n3_edma", "SP 8"),
        row("E", 2018, 7, 2.0, "n3_edma", "SP 15"),
        row("B", 2034, 91, 0.0, "n4_pdma", "SP 8"),
        row("E", 2034, 91, 1.0, "n4_pdma", "SP 15"),
    ]
    trace = write(tmp_path / "t.tsv", lines)
    table = union_table(trace)
    assert {row["engine"] for row in table["nodes"]} == {"TPC", "MME", "EDMA", "PDMA"}
    # the four windows all start at 0 and overlap, so the union is the longest one
    assert table["merged_us"] == 6.0
    assert [row["union_us"] for row in table["nodes"]] == [6.0, 5.0, 2.0, 1.0]


def test_module_offset_is_normalised(tmp_path):
    # module 1 carries +4000 on the engine id
    lines = [
        row("B", 6028, 3, 0.0, "n2_rg_other", "SP 8"),
        row("E", 6028, 3, 7.5, "n2_rg_other", "SP 3"),
    ]
    trace = write(tmp_path / "t.tsv", lines)
    table = union_table(trace)
    assert table["nodes"] == [
        {"engine": "TPC", "node": "n2_rg_other", "intervals": 1, "union_us": 7.5}
    ]


def test_an_unattributed_trace_is_reported(tmp_path):
    trace = write(
        tmp_path / "t.tsv", [row("M", 2000, 0, 0.0, "process_name", "__metadata")]
    )
    table = union_table(trace)
    assert table["attributed"] is False
    assert table["nodes"] == [] and table["merged_us"] == 0.0
