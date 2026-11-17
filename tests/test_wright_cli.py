from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deploy import models, serving
from resin_models_gaudi import cli
from resin_models_gaudi.detect import DetectionError, choose_preset, detect


def checkpoint(path, **config):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    return path


FP8 = {"quantization_config": {"quant_method": "fp8"}}


def test_detects_glm_from_config(tmp_path):
    glm = checkpoint(tmp_path / "b", model_type="glm5_next", **FP8)
    assert detect(glm).preset == "glm5_3_flash-gaudi2"


def qwen(path, hidden, layers):
    text = {"model_type": "qwen3_5", "hidden_size": hidden, "num_hidden_layers": layers}
    return checkpoint(path, text_config=text, **FP8)


@pytest.mark.parametrize(
    ("hidden", "layers", "preset", "name"),
    [
        (1024, 24, "qwen3_5-0.8b-gaudi2", "Qwen3.5-0.8B"),
        (2048, 24, "qwen3_5-2b-gaudi2", "Qwen3.5-2B"),
        (2560, 32, "qwen3_5-4b-gaudi2", "Qwen3.5-4B"),
        (4096, 32, "qwen3_5-9b-gaudi2", "Qwen3.5-9B"),
        (5120, 64, "qwen3_5-gaudi2", "Qwen3.8-27B"),
    ],
)
def test_qwen_size_chooses_the_preset(tmp_path, hidden, layers, preset, name):
    found = detect(qwen(tmp_path / "a", hidden, layers))
    assert (found.preset, found.name) == (preset, name)
    assert preset in serving.MODEL_SPECS


def test_refuses_a_qwen_size_with_no_runner(tmp_path):
    with pytest.raises(DetectionError, match="hidden size 8192 and 80 layers"):
        detect(qwen(tmp_path / "a", 8192, 80))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"model_type": "glm5_next"}, "needs FP8"),
        ({"model_type": "llama", **FP8}, "has no Gaudi runner"),
    ],
)
def test_refuses_what_it_cannot_run(tmp_path, config, message):
    with pytest.raises(DetectionError, match=message):
        detect(checkpoint(tmp_path / "a", **config))


def test_refuses_a_path_with_no_model(tmp_path):
    with pytest.raises(DetectionError, match="does not exist"):
        detect(tmp_path / "missing")
    with pytest.raises(DetectionError, match="cannot find the model"):
        detect(tmp_path)
    other = tmp_path / "weights.bin"
    other.write_bytes(b"x")
    with pytest.raises(DetectionError, match=".gguf file"):
        detect(other)


def fake_gguf(monkeypatch, **metadata):
    class Model:
        def __init__(self, path):
            self.metadata = metadata

    monkeypatch.setattr("resin_models_gaudi.gguf.reader.GgufModel", Model)


def test_detects_gguf_as_a_file_and_in_a_subdirectory(tmp_path, monkeypatch):
    fake_gguf(
        monkeypatch,
        **{"general.architecture": "mistral-large4", "general.file_type": 15},
    )
    weights = tmp_path / "UD-Q4_K_XL"
    weights.mkdir()
    part = weights / "model-00001-of-00014.gguf"
    part.write_bytes(b"x")
    for path in (tmp_path, weights, part):
        found = detect(path)
        assert found.preset == "mistral_large4-gaudi2"
        assert found.checkpoint == tmp_path
        assert not found.tokenizer_included


def test_refuses_other_gguf(tmp_path, monkeypatch):
    (tmp_path / "model.gguf").write_bytes(b"x")
    fake_gguf(monkeypatch, **{"general.architecture": "llama", "general.file_type": 15})
    with pytest.raises(DetectionError, match="has no Gaudi runner"):
        detect(tmp_path)
    fake_gguf(
        monkeypatch,
        **{"general.architecture": "mistral-large4", "general.file_type": 7},
    )
    with pytest.raises(DetectionError, match="UD-Q4_K_XL"):
        detect(tmp_path)


def test_glm_preset_follows_cards_and_context(tmp_path):
    found = detect(checkpoint(tmp_path / "a", model_type="glm5_next", **FP8))
    assert choose_preset(found, None, None) == "glm5_3_flash-gaudi2"
    assert choose_preset(found, 8, 65536) == "glm5_3_flash-gaudi2"
    assert choose_preset(found, 4, None) == "glm5_3_flash-gaudi2-tp4"
    assert choose_preset(found, None, 262144) == "glm5_3_flash-gaudi2-256k"


@pytest.mark.parametrize(
    ("text", "modules"),
    [
        ("HPU0,HPU1", "0,1"),
        ("0,1", "0,1"),
        ("hpu3", "3"),
        (" 4, 5 ", "4,5"),
        ("0-3", "0,1,2,3"),
        ("HPU4-HPU7,1", "4,5,6,7,1"),
    ],
)
def test_device_names(text, modules):
    assert cli.device_modules(text) == modules


def run(monkeypatch, tmp_path, *flags):
    """The arguments that reach the server for these llama-server flags."""
    seen = []
    monkeypatch.setattr(serving, "main", lambda argv, on_error: seen.append(argv))
    path = qwen(tmp_path / "model", 5120, 64)
    result = CliRunner().invoke(cli.app, ["-m", str(path), *flags])
    return result, seen[0] if seen else None


def value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_llama_flags_reach_the_server(monkeypatch, tmp_path):
    result, argv = run(
        monkeypatch,
        tmp_path,
        *("-c", "16384", "-np", "8", "-b", "2048", "--spec-draft-n-max", "1"),
        *("-a", "name", "-dev", "HPU2", "--host", "0.0.0.0", "--port", "9000"),
        "--no-cache-prompt",
    )
    assert result.exit_code == 0, result.output
    assert value(argv, "--model") == "qwen3_5-gaudi2"
    assert value(argv, "--context-length") == "16384"
    assert value(argv, "--max-batch-size") == "8"
    assert value(argv, "--prefill-chunk-size") == "2048"
    assert value(argv, "--draft-tokens") == "1"
    assert value(argv, "--alias") == "name"
    assert value(argv, "--gpu") == "2"
    assert value(argv, "--host") == "0.0.0.0"
    assert value(argv, "--port") == "9000"
    assert "--no-prefix-cache" in argv


def test_decode_share_reaches_the_server(monkeypatch, tmp_path):
    result, argv = run(monkeypatch, tmp_path, "--decode-share", "60:95")
    assert result.exit_code == 0, result.output
    assert value(argv, "--decode-share") == "60:95"
    args = serving.argument_parser().parse_args(argv)
    assert args.decode_share == (0.6, 0.95)


def test_defaults_leave_the_preset_in_charge(monkeypatch, tmp_path):
    result, argv = run(monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    for flag in (
        "--context-length",
        "--max-batch-size",
        "--draft-tokens",
        "--gpu",
        "--decode-share",
    ):
        assert flag not in argv
    assert serving.argument_parser().parse_args(argv).decode_share == (0.5, 0.9)
    assert value(argv, "--port") == "8080"
    assert "--prefix-cache" in argv


def test_one_device_for_each_model_puts_them_on_cards_of_their_own(
    monkeypatch, tmp_path
):
    seen = []
    monkeypatch.setattr(
        serving,
        "main_layout",
        lambda argvs, shares, on_error: seen.append((argvs, shares)),
    )
    glm = checkpoint(tmp_path / "glm", model_type="glm5_next", **FP8)
    small = qwen(tmp_path / "small", 4096, 32)
    first = ["-m", str(glm), "-a", "big", "-dev", "HPU0,HPU1,HPU2,HPU3", "--ranks", "4"]
    second = ["-m", str(small), "-a", "fast", "-dev", "4-7", "--ranks", "1"]

    def invoke(*extra, head=()):
        return CliRunner().invoke(cli.app, [*head, *first, *second, *extra])

    result = invoke()
    assert result.exit_code == 0, result.output
    (one, two), shares = seen[0]
    assert shares is None
    assert value(one, "--model") == "glm5_3_flash-gaudi2-tp4"
    assert value(one, "--gpu") == "0,1,2,3" and value(one, "--alias") == "big"
    assert value(two, "--model") == "qwen3_5-9b-gaudi2"
    assert value(two, "--gpu") == "4,5,6,7" and value(two, "--ranks") == "1"
    for argv in (one, two):
        serving.argument_parser().parse_args(argv)
    # A replica model's memory share and --share follow its -m. Before the first -m, a memory share is for
    # the replica models: the first model takes the memory that is left.
    result = invoke("--gpu-memory-utilization", "0.2", "--share", "0.5")
    assert result.exit_code == 0, result.output
    (one, two), shares = seen[-1]
    assert "--gpu-memory-utilization" not in one
    assert value(two, "--gpu-memory-utilization") == "0.2"
    assert shares == [1.0, 0.5]
    result = invoke(head=["--gpu-memory-utilization", "0.2"])
    assert result.exit_code == 0, result.output
    (one, two), _ = seen[-1]
    assert "--gpu-memory-utilization" not in one
    assert value(two, "--gpu-memory-utilization") == "0.2"
    result = CliRunner().invoke(
        cli.app, [*first, "--gpu-memory-utilization", "0.2", *second]
    )
    assert result.exit_code == 2 and "after a replica model's -m" in result.output
    # -o is the server's: it may come anywhere
    result = invoke("-o", "throughput")
    assert result.exit_code == 0, result.output
    (one, two), _ = seen[-1]
    assert value(one, "--optimize") == "throughput" == value(two, "--optimize")
    # draft tokens for each model, or before the first -m for every model
    result = CliRunner().invoke(
        cli.app, [*first, "--spec-draft-n-max", "3", *second, "--spec-draft-n-max", "0"]
    )
    assert result.exit_code == 0, result.output
    (one, two), _ = seen[-1]
    assert value(one, "--draft-tokens") == "3" and value(two, "--draft-tokens") == "0"
    result = invoke(head=["--spec-draft-n-max", "3"])
    assert result.exit_code == 0, result.output
    (one, two), _ = seen[-1]
    assert value(one, "--draft-tokens") == "3" == value(two, "--draft-tokens")
    result = invoke("--spec-draft-n-max", "3", "--spec-draft-n-max", "0")
    assert result.exit_code == 2
    assert f"--spec-draft-n-max is given 2 times after -m {small}" in result.output
    result = CliRunner().invoke(cli.app, ["-m", str(small), "--optimize", "throughput"])
    assert result.exit_code == 2 and "-o throughput is for replicas" in result.output
    result = invoke("--kv-floor", "1")
    assert result.exit_code == 2 and "--kv-floor is for several models on one card" in result.output
    # Each model of a layout gives its cards.
    result = CliRunner().invoke(cli.app, [*first, "-m", str(small)])
    assert result.exit_code == 2 and f"{small} has none" in result.output
    # Without a -dev for each model, the models share one card.
    result = CliRunner().invoke(cli.app, ["-m", str(glm), "--ranks", "4", "-m", str(small)])
    assert result.exit_code == 2 and "cards of their own" in result.output


def test_each_model_takes_its_own_flags(monkeypatch, tmp_path):
    """The flags after an -m are that model's; before the first -m, they are every model's."""
    seen = []
    monkeypatch.setattr(
        serving,
        "main_layout",
        lambda argvs, shares, on_error: seen.append(argvs),
    )
    glm = checkpoint(tmp_path / "glm", model_type="glm5_next", **FP8)
    small = qwen(tmp_path / "small", 4096, 32)
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    result = CliRunner().invoke(
        cli.app,
        ["-c", "32768", "-np", "8", "--decode-share", "25:99", "--port", "9000"]
        + ["-m", str(glm), "-dev", "0-7", "-c", "65536", "-md", str(drafter), "-b", "4096"]
        + ["-m", str(small), "-dev", "0-7", "-np", "16", "--no-cache-prompt", "--no-overlap"]
        + ["--host", "0.0.0.0"],
    )
    assert result.exit_code == 0, result.output
    ((one, two),) = seen
    assert value(one, "--context-length") == "65536" and value(two, "--context-length") == "32768"
    assert value(one, "--max-batch-size") == "8" and value(two, "--max-batch-size") == "16"
    assert value(one, "--prefill-chunk-size") == "4096" and "--prefill-chunk-size" not in two
    assert value(one, "--assistant-checkpoint") == str(drafter)
    assert "--assistant-checkpoint" not in two
    assert "--prefix-cache" in one and "--no-prefix-cache" in two
    assert "--overlap" in one and "--no-overlap" in two
    assert value(one, "--decode-share") == "25:99" == value(two, "--decode-share")
    # the server's flags reach every model, wherever they are
    for argv in (one, two):
        assert value(argv, "--port") == "9000" and value(argv, "--host") == "0.0.0.0"
        serving.argument_parser().parse_args(argv)
    # a server flag is given once
    result = CliRunner().invoke(
        cli.app,
        ["--port", "9000", "-m", str(glm), "-dev", "0-7", "-m", str(small), "-dev", "0-7"]
        + ["--port", "9001"],
    )
    assert result.exit_code == 2
    assert f"--port is given before the first -m and after -m {small}" in result.output
    # with one model, the flags may come before -m
    monkeypatch.setattr(serving, "main", lambda argv, on_error: seen.append([argv]))
    result = CliRunner().invoke(cli.app, ["-c", "4096", "-m", str(small), "-np", "4"])
    assert result.exit_code == 0, result.output
    ((argv,),) = seen[-1:]
    assert value(argv, "--context-length") == "4096" and value(argv, "--max-batch-size") == "4"


def test_step_transport_reaches_the_server(monkeypatch, tmp_path):
    _, argv = run(monkeypatch, tmp_path)
    assert serving.argument_parser().parse_args(argv).step_transport == "auto"
    result, argv = run(monkeypatch, tmp_path, "--step-transport", "gloo")
    assert result.exit_code == 0, result.output
    assert serving.argument_parser().parse_args(argv).step_transport == "gloo"
    result, argv = run(monkeypatch, tmp_path, "--step-transport", "native")
    assert result.exit_code == 0, result.output
    assert serving.argument_parser().parse_args(argv).step_transport == "native"
    result, _ = run(monkeypatch, tmp_path, "--step-transport", "shm")
    assert result.exit_code == 2


def test_multi_card_qwen_checks_its_step_transport(monkeypatch):
    """Only the Qwen3.5/3.8, DeepSeek-V4.1 and GLM-5.3-Flash families take --step-transport native, and in layouts
    of several models DeepSeek-V4.1 and GLM-5.3-Flash only."""
    from deploy.models import get_model

    parser = serving.argument_parser()

    def error(message):
        raise SystemExit(message)

    monkeypatch.setattr(parser, "error", error)
    argv = ["--model", "mistral_large4-gaudi2", "--step-transport", "native"]
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit, match="has no native rank dispatch"):
        serving.serve_gaudi_ranks(parser, args, get_model(args.model))
    argv = ["--model", "qwen3_5-gaudi2", "--ranks", "2", "--draft-tokens", "0", "--step-transport", "native"]
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit, match="DeepSeek-V4.1 and GLM-5.3-Flash only"):
        serving.serve_gaudi_ranks(parser, args, get_model(args.model), [dict(name="x", modules=[4])])


def test_a_multi_card_qwen_needs_the_native_executor(monkeypatch):
    """A Qwen3.5/3.8 model on several cards stops before the load with the oldqwen feature (the Python executor
    runs on one card). (A layout's Qwen replicas over several cards: test_a_layout_checks_its_models.)"""
    from deploy.models import get_model

    monkeypatch.delenv("RESIN_GAUDI_FEATURES", raising=False)
    parser = serving.argument_parser()

    def error(message):
        raise SystemExit(message)

    monkeypatch.setattr(parser, "error", error)
    argv = ["--model", "qwen3_5-gaudi2", "--ranks", "2", "--draft-tokens", "0"]
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as stopped:
        serving.serve_gaudi_ranks(parser, args, get_model(args.model))
    assert "oldqwen" not in str(stopped.value)  # the native executor (a later check: no checkpoint here)
    with pytest.raises(SystemExit, match="do not give -f oldqwen"):
        serving.serve_gaudi_ranks(parser, parser.parse_args([*argv, "-f", "oldqwen"]), get_model(args.model))
    # the variable alone also turns it on
    monkeypatch.setenv("RESIN_GAUDI_FEATURES", "oldqwen")
    with pytest.raises(SystemExit, match="do not give -f oldqwen"):
        serving.serve_gaudi_ranks(parser, args, get_model(args.model))


def test_features_reach_the_server(monkeypatch, tmp_path):
    result, argv = run(
        monkeypatch,
        tmp_path,
        *("-f", "skipvalidation", "--features", "oldqwen,skipvalidation"),
    )
    assert result.exit_code == 0, result.output
    assert value(argv, "--features") == "oldqwen,skipvalidation"
    parsed = serving.argument_parser().parse_args(argv)
    assert parsed.features == ("oldqwen", "skipvalidation")
    _, argv = run(monkeypatch, tmp_path)
    assert "--features" not in argv
    assert serving.argument_parser().parse_args(argv).features is None
    result, argv = run(monkeypatch, tmp_path, "-f", "turbo")
    assert result.exit_code == 2 and argv is None
    assert "unknown feature 'turbo'" in result.output
    shown = CliRunner().invoke(cli.app, ["--help"]).output
    assert "--skip-launch-validation" not in shown


def test_cache_types_reach_the_server_for_each_model(monkeypatch, tmp_path):
    """-ctk and -ctv are each model's; the server takes them as --cache-type-k and --cache-type-v."""
    result, argv = run(monkeypatch, tmp_path, "-ctk", "fp8", "--cache-type-v", "bf16")
    assert result.exit_code == 0, result.output
    assert value(argv, "--cache-type-k") == "fp8" and value(argv, "--cache-type-v") == "bf16"
    parsed = serving.argument_parser().parse_args(argv)
    assert serving.resolve_cache_types(models.get_model("qwen3_5-gaudi2"), parsed) == ("fp8", "bf16")
    _, argv = run(monkeypatch, tmp_path)
    assert "--cache-type-k" not in argv and "--cache-type-v" not in argv
    parsed = serving.argument_parser().parse_args(argv)
    assert serving.resolve_cache_types(models.get_model("qwen3_5-gaudi2"), parsed) == ("bf16", "bf16")
    result, argv = run(monkeypatch, tmp_path, "-ctk", "q8_0")
    assert result.exit_code == 2 and argv is None
    seen = []
    monkeypatch.setattr(
        serving, "main_layout", lambda argvs, shares, on_error: seen.append(argvs)
    )
    big, small = qwen(tmp_path / "big", 5120, 64), qwen(tmp_path / "small", 4096, 32)
    result = CliRunner().invoke(
        cli.app,
        ["-ctv", "fp8", "-m", str(big), "-dev", "0-3", "--ranks", "4", "-ctk", "fp8"]
        + ["-m", str(small), "-dev", "0-3"],
    )
    assert result.exit_code == 0, result.output
    ((one, two),) = seen
    assert value(one, "--cache-type-k") == "fp8" and "--cache-type-k" not in two
    assert value(one, "--cache-type-v") == "fp8" == value(two, "--cache-type-v")


@pytest.mark.parametrize(
    ("preset", "types", "message"),
    [
        ("deepseek_v41-gaudi2", ("bf16", None), "keeps its KV cache as fp8: --cache-type-k bf16"),
        ("qwen4_exp-gaudi2", (None, "fp8"), "keeps its KV cache as bf16: --cache-type-v fp8"),
    ],
)
def test_a_model_takes_only_its_cache_types(preset, types, message):
    spec = models.get_model(preset)
    with pytest.raises(ValueError, match=message):
        models.check_cache_types(spec, types)
    assert models.check_cache_types(spec, None) == (models.kv_cache_types(spec)[0],) * 2


def test_features_help_lists_them(monkeypatch, tmp_path):
    """-f help shows every feature and stops, with or without a model."""
    from resin_models_gaudi import features

    path = qwen(tmp_path / "model", 5120, 64)
    for flags in (["-f", "help"], ["-m", str(path), "-f", "oldqwen,help"]):
        result = CliRunner().invoke(cli.app, flags)
        assert result.exit_code == 0, result.output
        assert all(name in result.output for name in features.FEATURES), result.output


def test_every_generated_argument_parses(monkeypatch, tmp_path):
    _, argv = run(monkeypatch, tmp_path, "-c", "4096", "--ranks", "1", "-lv", "4")
    args = serving.argument_parser().parse_args(argv)
    assert args.context_length == 4096 and args.ranks == 1


def test_a_missing_model_and_unsupported_flags_are_usage_errors(tmp_path):
    runner = CliRunner()
    assert runner.invoke(cli.app, []).exit_code == 2
    assert runner.invoke(cli.app, ["-m", str(tmp_path / "missing")]).exit_code == 2
    # The server has no authentication, so it must not accept this flag and ignore it.
    assert runner.invoke(cli.app, ["-m", "x", "--api-key", "secret"]).exit_code == 2


def test_server_errors_name_the_flags_of_this_command(capsys):
    with pytest.raises(cli.typer.Exit):
        cli.server_error(
            "x takes --draft-tokens 0, 3, --gpu 0,1 and --gpu-memory-utilization 0.5"
        )
    assert capsys.readouterr().err.strip() == (
        "error: x takes --spec-draft-n-max 0, 3, --device 0,1 and "
        "--gpu-memory-utilization 0.5"
    )


def test_list_devices(monkeypatch):
    from resin_models_gaudi.synapse import cards

    card = cards.Card(accel=4, module=0, pci="0000:01:00.0", operational=True)
    monkeypatch.setattr(cards, "all_cards", lambda: [card])
    result = CliRunner().invoke(cli.app, ["--list-devices"])
    assert result.exit_code == 0, result.output
    assert "HPU0: /dev/accel/accel4, 0000:01:00.0" in result.output


@pytest.mark.parametrize(
    "module",
    [
        "resin_models_gaudi.qwen3_5.runner:Qwen3_5GaudiRunner",
        "resin_models_gaudi.glm5_3_flash.runner:Glm53FlashGaudiRunner",
        "resin_models_gaudi.mistral_large4.runner:MistralLarge4GaudiRunner",
        "resin_models_gaudi.deepseek_v41.runner:DeepSeekV41GaudiRunner",
    ],
)
def test_runners_report_their_kv_pool_and_stats_as_properties(module):
    import importlib

    name, cls = module.split(":")
    runner = getattr(importlib.import_module(name), cls)
    assert isinstance(runner.__dict__["kv_pool_bytes"], property)
    assert isinstance(runner.__dict__["speculative_stats"], property)


def test_kv_pool_report_gives_free_slabs_and_tokens(monkeypatch):
    from resin import CacheGroupLayout, CacheLayout

    lines = []
    log = type("Log", (), {"info": lines.append, "warning": lines.append})
    monkeypatch.setattr(serving, "logger", log)

    class Runner:
        cache_layout = CacheLayout(
            parent_capacity=2975, groups=[CacheGroupLayout(128, blocks_per_slab=1)]
        )
        kv_pool_bytes = 2**30

    class Engine:
        cache_free_parent_count = 2975

    class Distributed:
        local = Engine()

    for engine in (Engine(), Distributed()):
        lines.clear()
        serving.report_kv_pool(Runner(), engine, 8, 32768, 16)
        text = "\n".join(lines)
        assert "2,975 of 2,975 slabs free" in text
        assert "holds 380,800 tokens in one sequence" in text
        assert "holds 11 of 16 sequences at 32,768 tokens" in text


def test_drafts_need_mtp_weights(tmp_path):
    from deploy.models import check_gaudi_drafts, get_model

    spec = get_model("qwen3_5-9b-gaudi2")
    assert spec.draft_tokens == 0 and spec.recipe == "qwen3_5"
    check_gaudi_drafts(spec, tmp_path, 0)
    with pytest.raises(ValueError, match="no MTP weights"):
        check_gaudi_drafts(spec, tmp_path, 2)
    (tmp_path / "mtp.safetensors").write_bytes(b"x")
    check_gaudi_drafts(spec, tmp_path, 2)


def test_arena_shift_keeps_every_slab_in_each_attention_layer_s_view():
    from resin_models_gaudi.arena import ARENA_PAD_BLOCKS
    from resin_models_gaudi.qwen3_5.cache import ARENA_BLOCK_ROWS, CacheGeometry

    for groups, layers in ((2, 6), (2, 8), (4, 8), (4, 16)):
        geometry = CacheGeometry(layers, 0, groups, 0, 0, 0)
        shift = geometry.arena_blocks(4) - geometry.attention_blocks(4)
        last = (layers - 1) * geometry.kv_layer_rows
        # the views are short by at most the last layer's offset, so each holds its layer's rows of every slab,
        # and the last layer's view ends inside the arena's padding
        assert shift * ARENA_BLOCK_ROWS <= last
        assert last - shift * ARENA_BLOCK_ROWS <= ARENA_PAD_BLOCKS * ARENA_BLOCK_ROWS


@pytest.mark.parametrize("groups", [2, 4])
def test_kv_rows_keep_k_and_v_in_their_columns(groups):
    import numpy as np

    from resin_models_gaudi.common.attention import SlotRow, kv_rows

    k = np.arange(groups)[:, None] * 128 + 1000
    row = SlotRow(column=0, pos0=0, real=1, k_pages=k, v_pages=k + 5000)
    out = kv_rows(2, [row])
    assert out.shape == (2, 8)
    assert (out[0, :groups] == k[:, 0]).all()
    assert (out[0, groups:4] == -1).all()
    assert (out[0, 4 : 4 + groups] == k[:, 0] + 5000).all()
    assert (out[0, 4 + groups :] == -1).all()
    assert (out[1] == -1).all()


@pytest.mark.parametrize(
    ("preset", "thinking"),
    [
        ("qwen3_5-0.8b-gaudi2", False),
        ("qwen3_5-2b-gaudi2", False),
        ("qwen3_5-4b-gaudi2", True),
        ("qwen3_5-9b-gaudi2", True),
    ],
)
def test_dense_presets_start_thinking_like_their_templates(preset, thinking):
    from deploy.models import get_model

    options = dict(get_model(preset).recipe_options)
    assert options.get("enable_thinking", True) is thinking


def test_glm_drafter_flags_reach_the_ranks(monkeypatch, tmp_path):
    """-md and --spec-draft-n-max 7 name GLM-5.3-Flash's DFlash 2 drafter for its ranks."""
    seen = []
    monkeypatch.setattr(serving, "main", lambda argv, on_error: seen.append(argv))
    path = checkpoint(tmp_path / "glm", model_type="glm5_next", **FP8)
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    result = CliRunner().invoke(
        cli.app, ["-m", str(path), "-md", str(drafter), "--spec-draft-n-max", "7"]
    )
    assert result.exit_code == 0, result.output
    argv = seen[0]
    assert value(argv, "--model") == "glm5_3_flash-gaudi2"
    assert value(argv, "--assistant-checkpoint") == str(drafter)
    assert value(argv, "--draft-tokens") == "7"


def test_one_card_qwen_takes_a_drafter_directory(monkeypatch, tmp_path):
    """-md names the DFlash 2 drafter of Qwen3.8-27B; the other one-card flags stay refused."""
    import resin_models_gaudi.synapse.runtime as runtime

    class Claimed(Exception):
        pass

    def claim_card(*args, **kwargs):
        raise Claimed

    monkeypatch.setattr(runtime, "claim_card", claim_card)
    monkeypatch.setattr(serving, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(serving, "_validate", lambda parser, args, spec: None)

    def usage(message):
        raise SystemExit(message)

    drafter = tmp_path / "drafter"
    drafter.mkdir()
    base = [
        "--model",
        "qwen3_5-gaudi2",
        "--checkpoint",
        str(tmp_path),
        "--port",
        "18099",
    ]
    with pytest.raises(Claimed):
        serving.main(
            base + ["--assistant-checkpoint", str(drafter), "--draft-tokens", "7"],
            on_error=usage,
        )
    with pytest.raises(SystemExit, match="does not use --tokenizer"):
        serving.main(base + ["--tokenizer", str(tmp_path)], on_error=usage)


def test_several_models_go_to_one_card_with_their_aliases(monkeypatch, tmp_path):
    big = qwen(tmp_path / "big", 5120, 64)
    small = qwen(tmp_path / "small", 4096, 32)
    seen = []
    monkeypatch.setattr(
        serving,
        "main_models",
        lambda argvs, shares, floors, on_error: seen.append((argvs, shares, floors)),
    )
    result = CliRunner().invoke(
        cli.app,
        ["-np", "8", "--gpu-memory-utilization", "0.8"]
        + ["-m", str(big), "-a", "big", "--share", "2", "--decode-share", "0"]
        + ["-m", str(small), "-a", "small", "--kv-floor", "64", "-c", "8192"],
    )
    assert result.exit_code == 0, result.output
    ((argvs, shares, floors),) = seen
    assert shares == [2.0, 1.0] and floors == [0, 64]
    presets = [value(argv, "--model") for argv in argvs]
    assert presets == ["qwen3_5-gaudi2", "qwen3_5-9b-gaudi2"]
    assert [value(argv, "--alias") for argv in argvs] == ["big", "small"]
    assert all(value(argv, "--max-batch-size") == "8" for argv in argvs)
    assert all(value(argv, "--gpu-memory-utilization") == "0.8" for argv in argvs)
    assert value(argvs[0], "--decode-share") == "0" and "--decode-share" not in argvs[1]
    assert "--context-length" not in argvs[0] and value(argvs[1], "--context-length") == "8192"
    for argv in argvs:  # each is a valid deploy.serving argument list
        serving.argument_parser().parse_args(argv)


def test_one_card_models_take_one_memory_share_and_one_model_takes_no_share(tmp_path):
    big = qwen(tmp_path / "big", 5120, 64)
    small = qwen(tmp_path / "small", 4096, 32)
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["-m", str(big), "--gpu-memory-utilization", "0.5", "-m", str(small)]
    )
    assert result.exit_code == 2 and "give it before the first -m" in result.output
    result = runner.invoke(cli.app, ["-m", str(big), "-m", str(small), "-a", "x", "-a", "y"])
    assert result.exit_code == 2 and "give each model its own --alias after its -m" in result.output
    result = runner.invoke(cli.app, ["-m", str(big), "--share", "2"])
    assert result.exit_code == 2 and "several models" in result.output


def test_a_layout_checks_its_models(monkeypatch):
    seen = []
    monkeypatch.setattr(
        serving,
        "serve_gaudi_ranks",
        lambda parser, args, spec, replicated: seen.append((args, spec, replicated)),
    )
    # The replica models' checkpoints are checked before the first model loads.
    checked = []

    def require_checkpoint(spec, *, checkpoint, draft_tokens):
        if str(checkpoint) == "/missing":
            raise FileNotFoundError("no checkpoint at /missing")
        checked.append(spec.name)
        return Path("/checkpoint")

    monkeypatch.setattr(serving, "require_checkpoint", require_checkpoint)
    monkeypatch.setattr(serving, "check_gaudi_drafts", lambda *args: None)
    # main_layout applies the first model's --gpu to this process.
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "")

    def error(message):
        raise SystemExit(message)

    glm = ["--model", "glm5_3_flash-gaudi2-tp4", "--gpu", "0,1,2,3"]
    small = ["--model", "qwen3_5-9b-gaudi2", "--gpu", "4,5", "--alias", "fast"]
    serving.main_layout([glm, small], on_error=error)
    args, spec, replicated = seen[0]
    assert spec.name == "glm5_3_flash-gaudi2-tp4" and args.gpu == "0,1,2,3"
    assert replicated == [dict(name="fast", modules=[4, 5], argv=small)]
    assert checked == ["qwen3_5-9b-gaudi2"]
    # A replica model takes --ranks 1, which the layout gives each model.
    serving.main_layout([glm, [*small, "--ranks", "1"]], on_error=error)
    for argvs, message in (
        ([small, glm], "first model must span several cards"),
        ([glm, glm], "only the first model can span cards"),
        ([glm, [*small[:3], "3,4", *small[4:]]], "or cards of their own, not both"),
        ([glm, [*small[:3], "1,2", *small[4:]], [*small[:3], "2,3", "--alias", "x"]], r"both use modules \[2\]"),
        ([glm, small[:2]], "give each model its cards with --gpu"),
        ([glm, [*small[:3], "4,4", *small[4:]]], "has a duplicate"),
        ([glm, [*small, "--ranks", "2"]], "give it --ranks 1"),
        ([glm, [*small, "--num-layers", "2"]], "does not use --num-layers"),
        ([glm, [*small, "--checkpoint", "/missing"]], "no checkpoint at /missing"),
    ):
        with pytest.raises(SystemExit, match=message):
            serving.main_layout(argvs, on_error=error)
    # A Qwen replica can span cards of the first model's ranks: two replicas of two cards each, on the native
    # executor (not with the oldqwen feature, a server setting: the first model's)
    monkeypatch.setenv("RESIN_GAUDI_FEATURES", "")
    wide = ["--model", "qwen3_5-9b-gaudi2", "--gpu", "0,1,2,3", "--alias", "wide", "--ranks", "2"]
    with pytest.raises(SystemExit, match="replicas of 2 cards: .*do not give -f oldqwen"):
        serving.main_layout([[*glm, "-f", "oldqwen"], [*wide, "--draft-tokens", "0"]], on_error=error)
    monkeypatch.setenv("RESIN_GAUDI_FEATURES", "")  # (-f wrote the variable)
    serving.main_layout([glm, [*wide, "--draft-tokens", "0"]], on_error=error)
    (served,) = seen[-1][2]
    assert served["replica_ranks"] == 2 and served["ranks"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert not served["shared_kv"]  # its ranks take a fixed share of each card
    assert "--gpu-memory-utilization" in served["argv"]
    with pytest.raises(SystemExit, match="--ranks 3 does not divide its 4 cards"):
        serving.main_layout(
            [glm, [*wide[:-1], "3", "--draft-tokens", "0"]], on_error=error
        )
    # its MTP layer drafts on every rank (each rank keeps the whole layer)
    serving.main_layout([glm, [*wide, "--draft-tokens", "2"]], on_error=error)
    assert seen[-1][2][0]["replica_ranks"] == 2


def test_replicas_keep_a_fixed_share_beside_a_drafter_model(monkeypatch, tmp_path):
    """A drafter model (DFlash 2 or DSpine) does not take a shared cache arena yet: with one on the first model
    or on the replicas, each replica keeps a fixed share of its card's memory, so that the server starts."""
    seen = []
    monkeypatch.setattr(
        serving,
        "serve_gaudi_ranks",
        lambda parser, args, spec, replicated: seen.append(replicated),
    )
    monkeypatch.setattr(
        serving, "require_checkpoint", lambda spec, **_: Path("/checkpoint")
    )
    monkeypatch.setattr(serving, "check_gaudi_drafts", lambda *args: None)
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "")

    def error(message):
        raise SystemExit(message)

    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text('{"block_size": 8}')
    glm = ["--model", "glm5_3_flash-gaudi2-tp4", "--gpu", "4,5,6,7"]
    small = ["--model", "qwen3_5-9b-gaudi2", "--gpu", "7,5", "--alias", "fast"]
    fixed = str(serving.COLOCATED_MEMORY)
    serving.main_layout(
        [[*glm, "--assistant-checkpoint", str(drafter)], small], on_error=error
    )
    (served,) = seen[-1]
    assert not served["shared_kv"]
    assert value(served["argv"], "--gpu-memory-utilization") == fixed
    # with no drafts, no drafter model runs: the models share each card's KV cache
    serving.main_layout(
        [[*glm, "--assistant-checkpoint", str(drafter), "--draft-tokens", "0"], small],
        on_error=error,
    )
    (served,) = seen[-1]
    assert served["shared_kv"] and "--gpu-memory-utilization" not in served["argv"]
    # a replica with a drafter model of its own keeps a fixed share too
    deepseek = ["--model", "deepseek_v41-gaudi2", "--gpu", "0,1,2,3,4,5,6,7"]
    serving.main_layout(
        [deepseek, [*small, "--assistant-checkpoint", str(drafter)]], on_error=error
    )
    (served,) = seen[-1]
    assert not served["shared_kv"]
    assert value(served["argv"], "--gpu-memory-utilization") == fixed
    # DSpark drafts inside DeepSeek-V4.1-Flash, with no drafter model: it shares
    serving.main_layout([[*deepseek, "--draft-tokens", "3"], small], on_error=error)
    assert seen[-1][0]["shared_kv"]


def test_a_layout_puts_replicas_on_the_cards_of_the_first_model(monkeypatch):
    """A replica model on cards of the first model's ranks other than rank 0's: each replica runs
    in the process of the rank of its card, with a small cache, and steps when the first model is
    idle there unless --share gives it a weight."""
    seen = []
    monkeypatch.setattr(
        serving,
        "serve_gaudi_ranks",
        lambda parser, args, spec, replicated: seen.append(replicated),
    )
    monkeypatch.setattr(
        serving, "require_checkpoint", lambda spec, **_: Path("/checkpoint")
    )
    monkeypatch.setattr(serving, "check_gaudi_drafts", lambda *args: None)
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "")

    def error(message):
        raise SystemExit(message)

    glm = ["--model", "glm5_3_flash-gaudi2-tp4", "--gpu", "4,5,6,7"]
    small = ["--model", "qwen3_5-9b-gaudi2", "--gpu", "7,5", "--alias", "fast"]
    serving.main_layout([glm, small], on_error=error)
    (served,) = seen[-1]
    assert served["ranks"] == {3: 0, 1: 1} and served["share"] == 0.0
    # rank 0's card can hold a replica too
    serving.main_layout([glm, [*small[:3], "4,5", *small[4:]]], on_error=error)
    assert seen[-1][0]["ranks"] == {0: 0, 1: 1}
    serving.main_layout([glm, small], on_error=error)
    # GLM loads its weights first: it and the replicas share each card's KV cache
    (served,) = seen[-1]
    assert served["shared_kv"] and "--gpu-memory-utilization" not in served["argv"]
    # Mistral-Large-4 makes its cache when it loads: each replica takes a fixed share of its card's
    # memory
    mistral = ["--model", "mistral_large4-gaudi2", "--gpu", "0,1,2,3,4,5,6,7"]
    serving.main_layout([mistral, small], on_error=error)
    (served,) = seen[-1]
    assert not served["shared_kv"]
    assert value(served["argv"], "--gpu-memory-utilization") == str(
        serving.COLOCATED_MEMORY
    )
    # DeepSeek loads its weights first: it and the replicas share each card's KV cache, unless a
    # replica takes a share of its own
    deepseek = ["--model", "deepseek_v41-gaudi2", "--gpu", "0,1,2,3,4,5,6,7"]
    serving.main_layout([deepseek, small], on_error=error)
    (served,) = seen[-1]
    assert served["shared_kv"] and "--gpu-memory-utilization" not in served["argv"]
    serving.main_layout(
        [deepseek, [*small, "--gpu-memory-utilization", "0.3"]], on_error=error
    )
    (served,) = seen[-1]
    assert not served["shared_kv"]
    # Two replica models on its cards: each card's replica has a table side of its own, and only the ones that
    # share; the one with a fixed share keeps it.
    shared = ["--model", "qwen3_5-9b-gaudi2", "--gpu", "1,2,3", "--alias", "a"]
    fixed = ["--model", "qwen3_5-9b-gaudi2", "--gpu", "5,4", "--alias", "b"]
    serving.main_layout(
        [deepseek, shared, [*fixed, "--gpu-memory-utilization", "0.3"]], on_error=error
    )
    a, b = seen[-1]
    assert a["shared_kv"] and not b["shared_kv"]
    assert serving.card_kv_sides(seen[-1]) == {1: 1, 2: 2, 3: 3}
    serving.main_layout([deepseek, shared, fixed], on_error=error)
    assert serving.card_kv_sides(seen[-1]) == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    serving.main_layout(
        [glm, [*small, "--gpu-memory-utilization", "0.3"]],
        shares=[2.0, 1.0],
        on_error=error,
    )
    (served,) = seen[-1]
    assert served["share"] == 0.5 and served["optimize"] == "latency"
    assert value(served["argv"], "--gpu-memory-utilization") == "0.3"
    # Throughput: both models run at once, so the replicas take no --share.
    serving.main_layout([[*glm, "--optimize", "throughput"], small], on_error=error)
    (served,) = seen[-1]
    assert served["optimize"] == "throughput"
    fast = [*glm, "--optimize", "throughput"]
    for argvs, shares, message in (
        ([fast, small], [1.0, 1.0], "--share weighs the card time of --optimize latency"),
        ([fast, [*small[:3], "8,9", *small[4:]]], None, "these replicas have cards of their own"),
    ):
        with pytest.raises(SystemExit, match=message):
            serving.main_layout(argvs, shares=shares, on_error=error)
    for shares, message in (
        ([1.0], "one --share for each model"),
        ([0.0, 1.0], "positive weight for the first model"),
        ([1.0, -1.0], "0 or more for the others"),
    ):
        with pytest.raises(SystemExit, match=message):
            serving.main_layout([glm, small], shares=shares, on_error=error)


def test_each_model_flag_has_the_default_of_serve():
    """A model with no flag of its own takes the default that `serve` gives the flag."""
    import dataclasses

    command = cli.typer.main.get_command(cli.app)
    defaults = {param.name: param.default for param in command.params}
    for field in dataclasses.fields(cli.ModelFlags):
        assert defaults[field.name] == field.default, field.name


def test_an_attached_model_path_starts_a_model_and_one_model_takes_each_flag_once(monkeypatch, tmp_path):
    """-mPATH starts a model as -m PATH does (-md stays the drafter flag); with one model, a flag given twice
    is an error, as with several."""
    seen = []
    monkeypatch.setattr(
        serving,
        "main_layout",
        lambda argvs, shares, on_error: seen.append(argvs),
    )
    glm = checkpoint(tmp_path / "glm", model_type="glm5_next", **FP8)
    small = qwen(tmp_path / "small", 4096, 32)
    result = CliRunner().invoke(
        cli.app,
        [f"-m{glm}", "-dev", "0-3", "--ranks", "4", "-c", "65536", f"-m{small}", "-dev", "4-7"],
    )
    assert result.exit_code == 0, result.output
    ((one, two),) = seen
    assert value(one, "--context-length") == "65536" and "--context-length" not in two
    assert value(two, "--gpu") == "4,5,6,7"
    assert cli.split_models(["-md", "x", "-m", "a"], frozenset({"-m", "-md"})) == (["-md", "x"], [("a", [])])
    result = CliRunner().invoke(cli.app, ["-m", str(small), "-c", "4096", "-c", "8192"])
    assert result.exit_code == 2 and "--ctx-size is given 2 times: give it once" in result.output


def test_a_one_card_model_gives_its_drafter_to_the_runner(monkeypatch):
    """-md (--assistant-checkpoint) reaches create_runner on the one-card path, as on the multi-card path."""
    seen = {}

    class Stop(Exception):
        pass

    def create_runner(spec, checkpoint, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(serving, "require_checkpoint", lambda spec, **kwargs: Path("/checkpoint"))
    monkeypatch.setattr(serving, "check_gaudi_drafts", lambda *args: None)
    monkeypatch.setattr(serving, "current_device_free", lambda: 1 << 40)
    monkeypatch.setattr(serving, "checkpoint_weight_bytes", lambda spec, checkpoint: 1 << 30)
    monkeypatch.setattr(serving, "_remember_load", lambda *args, **kwargs: None)
    monkeypatch.setattr(serving, "create_runner", create_runner)
    for argv, drafter in (
        (["--draft-tokens", "7", "--assistant-checkpoint", "/drafter"], Path("/drafter")),
        (["--draft-tokens", "2"], None),
    ):
        seen.clear()
        args = serving.argument_parser().parse_args(["--model", "qwen3_5-gaudi2", *argv])
        with pytest.raises(Stop), serving._one_card_model(args, models.get_model("qwen3_5-gaudi2"), "hpu:0"):
            pass
        assert seen["assistant_checkpoint"] == drafter and seen["draft_tokens"] == int(argv[1])
