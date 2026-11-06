from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from deploy import serving
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
    [("HPU0,HPU1", "0,1"), ("0,1", "0,1"), ("hpu3", "3"), (" 4, 5 ", "4,5")],
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


def test_arena_shift_covers_every_layer_offset():
    from resin_models_gaudi.qwen3_5.cache import ARENA_BLOCK_ROWS, CacheGeometry

    for groups, layers in ((2, 6), (2, 8), (4, 8), (4, 16)):
        geometry = CacheGeometry(layers, 0, groups, 0, 0, 0)
        blocks = geometry.arena_blocks(4) - geometry.attention_blocks(4)
        assert blocks * ARENA_BLOCK_ROWS >= (layers - 1) * geometry.kv_layer_rows


@pytest.mark.parametrize("groups", [2, 4])
def test_kv_rows_keep_k_and_v_in_their_columns(groups):
    import numpy as np

    from resin_models_gaudi.qwen3_5.attention import SlotRow, kv_rows

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
