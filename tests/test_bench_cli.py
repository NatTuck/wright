"""The benchmark CLI builds and validates the workload before it loads a model."""

import json
from pathlib import Path

import pytest

from scripts import bench


@pytest.mark.parametrize("name", bench.MODEL_SPECS)
def test_model_defaults_build_valid_natural_sweeps(name):
    args = bench.argument_parser().parse_args(["--model", name])
    spec, cases, topology, options = bench._configuration(args)
    assert {case["phase"] for case in cases} == {"prefill", "decode"}
    assert topology.world_size == spec.world_size
    assert options["draft_tokens"] == spec.draft_tokens
    assert options["accepted_drafts"] is None
    assert options["prefill_chunk_size"] == spec.prefill_chunk_size
    assert [
        (case["context_length"], case["batch_size"])
        for case in cases
        if case["phase"] == "prefill"
    ] == [(length, 1) for length in (4096, 8192, 16384, 32768, 65536)]
    assert [
        (case["context_length"], case["batch_size"])
        for case in cases
        if case["phase"] == "decode"
    ] == [(8192, batch) for batch in (1, 2, 4, 8, 16, 32, 64)]


def test_phase_specific_sweeps_override_shared_values():
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "glm5_3_flash-gaudi2",
            "--context-lengths",
            "2048",
            "--batch-sizes",
            "2",
            "--prefill-context-lengths",
            "4096,8192",
            "--prefill-batch-sizes",
            "1",
            "--decode-context-lengths",
            "8192",
            "--decode-batch-sizes",
            "1,4",
        ]
    )
    _, cases, _, options = bench._configuration(args)
    assert [
        (case["phase"], case["context_length"], case["batch_size"]) for case in cases
    ] == [
        ("prefill", 4096, 1),
        ("prefill", 8192, 1),
        ("decode", 8192, 1),
        ("decode", 8192, 4),
    ]
    assert options["prefill_chunk_size"] == 4096


def test_shared_sweep_options_remain_compatible():
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "mistral_large4-gaudi2",
            "--context-lengths",
            "1024,2048",
            "--batch-sizes",
            "1,2",
            "--prefill-chunk-size",
            "1024",
        ]
    )
    _, cases, _, options = bench._configuration(args)
    assert [
        (case["phase"], case["context_length"], case["batch_size"]) for case in cases
    ] == [
        (phase, length, batch)
        for phase in ("prefill", "decode")
        for length in (1024, 2048)
        for batch in (1, 2)
    ]
    assert options["prefill_chunk_size"] == 1024


def test_card_count_and_chunk_follow_the_preset():
    parser = bench.argument_parser()
    with pytest.raises(ValueError, match="runs on 8 Gaudi cards"):
        bench._configuration(
            parser.parse_args(["--model", "glm5_3_flash-gaudi2", "--world-size", "4"])
        )
    with pytest.raises(ValueError, match="4096-token chunks"):
        bench._configuration(
            parser.parse_args(
                ["--model", "qwen3_5-gaudi2", "--prefill-chunk-size", "2048"]
            )
        )


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        (["--accepted-drafts", "4", "--draft-tokens", "3"], "accepted drafts"),
        (["--measure-steps", "0"], "measure_steps"),
        (["--warmup-steps", "-1"], "warmup_steps"),
        (["--prefill-chunk-size", "0"], "prefill_chunk_size"),
        (["--phases", "decode,invalid"], "phases"),
        (["--gpu-memory-utilization", "1.5"], "gpu_memory_utilization"),
    ],
)
def test_invalid_options_fail_before_the_load(arguments, fragment, capsys):
    with pytest.raises(SystemExit) as error:
        bench.main(["--model", "qwen3_5-gaudi2", "--checkpoint", "unused", *arguments])
    assert error.value.code == 2
    assert fragment in capsys.readouterr().err


def test_requires_explicit_checkpoint(capsys):
    with pytest.raises(SystemExit):
        bench.main(["--model", "qwen3_5-gaudi2"])
    assert "--checkpoint is required" in capsys.readouterr().err


def test_prefill_trace_uses_fresh_run_directory_and_appends_results(
    tmp_path, monkeypatch
):
    from scripts import merge_bench_traces

    output = tmp_path / "results.jsonl"
    output.write_text('{"prior":true}\n')
    checkpoint = tmp_path / "checkpoint"
    traces = tmp_path / "traces"
    seen = {}
    monkeypatch.setattr(bench, "source_metadata", lambda: {"git_commit": "fixture"})
    monkeypatch.setattr(bench, "checkpoint_revision", lambda spec, path: "fixture")

    def require(spec, **options):
        assert spec.name == "qwen3_5-gaudi2" and options["checkpoint"] == checkpoint
        assert options["draft_tokens"] == spec.draft_tokens
        return checkpoint

    def run(model, path, cases, *, parallelism, **options):
        assert model == "qwen3_5-gaudi2" and path == checkpoint
        assert parallelism.world_size == 1
        assert {case["phase"] for case in cases} == {"prefill"}
        assert options["metadata"]["git_commit"] == "fixture"
        seen.update(options)
        return [{"phase": "prefill", "input_tokens": 8192}]

    monkeypatch.setattr(bench, "require_checkpoint", require)
    monkeypatch.setattr(bench, "run_benchmark", run)
    monkeypatch.setattr(
        merge_bench_traces, "main", lambda argv: seen.update(merged=argv)
    )
    bench.main(
        [
            "--model",
            "qwen3_5-gaudi2",
            "--checkpoint",
            str(checkpoint),
            "--phases",
            "prefill",
            "--context-lengths",
            "4096,8192",
            "--batch-sizes",
            "1",
            "--trace-dir",
            str(traces),
            "--output",
            str(output),
        ]
    )
    assert Path(seen["trace_dir"]).parent == traces
    assert seen["merged"] == [seen["trace_dir"]]
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"prior": True},
        {"phase": "prefill", "input_tokens": 8192},
    ]
