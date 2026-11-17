"""Shared serving/benchmark configuration preserves runner and checkpoint contracts."""

import json
from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest

from deploy.models import (
    MODEL_SPECS,
    create_runner,
    download_checkpoint,
    get_model,
    require_checkpoint,
)


@pytest.mark.parametrize("device, card", [(None, None), ("hpu", None), ("hpu:3", 3)])
def test_gaudi_factory_arguments_match_runner(monkeypatch, tmp_path, device, card):
    module = pytest.importorskip("resin_models_gaudi.qwen3_5.runner")
    spec = get_model("qwen3_5-gaudi2")
    constructor = signature(module.Qwen3_5GaudiRunner.__init__)
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            constructor.bind(self, *args, **kwargs)
            calls.append((args, kwargs))

    monkeypatch.setattr(module, "Qwen3_5GaudiRunner", Runner)
    options = dict(
        groups=None,
        context_length=8192,
        max_batch_size=8,
        prefill_chunk_size=4096,
        draft_tokens=spec.draft_tokens,
    )
    create_runner(spec, device=device, **options)
    args, kwargs = calls[0]
    assert args[0] == spec.path
    assert kwargs["card"] == card and kwargs["mtp"] == spec.draft_tokens
    assert kwargs["max_seq_len"] == 8192 + 2 * spec.draft_tokens - 1
    assert kwargs["max_batch_size"] == 8 and kwargs["drafter"] is None
    # a DFlash 2 drafter directory: its drafts, and a verify step's lookahead of K
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(
        json.dumps({"dflash_config": {"block_size": 8}})
    )
    create_runner(
        spec,
        device=device,
        **{**options, "draft_tokens": 7},
        assistant_checkpoint=drafter,
    )
    args, kwargs = calls[1]
    assert kwargs["mtp"] == 7 and kwargs["drafter"] == drafter
    assert kwargs["max_seq_len"] == 8192 + 7
    create_runner(
        spec,
        device=device,
        **{**options, "draft_tokens": 0},
        assistant_checkpoint=drafter,
    )
    assert calls[2][1]["drafter"] is None  # no drafts: the drafter is not loaded
    with pytest.raises(ValueError, match="hpu"):
        create_runner(spec, device="cuda:0", **options)
    with pytest.raises(ValueError, match="4096"):
        create_runner(spec, device=device, **{**options, "prefill_chunk_size": 8192})


@pytest.mark.parametrize("device, card", [("hpu", None), ("hpu:5", 5)])
def test_glm_factory_arguments_match_runner(monkeypatch, tmp_path, device, card):
    """GLM-5.3-Flash speculates with its MTP layer (a lookahead of 2K - 1) or a DFlash 2 drafter
    directory (a lookahead of K); without drafts the drafter is not loaded."""
    from deploy.models import validate_gaudi_options

    module = pytest.importorskip("resin_models_gaudi.glm5_3_flash.runner")
    spec = get_model("glm5_3_flash-gaudi2")
    constructor = signature(module.Glm53FlashGaudiRunner.__init__)
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            constructor.bind(self, *args, **kwargs)
            calls.append((args, kwargs))

    monkeypatch.setattr(module, "Glm53FlashGaudiRunner", Runner)
    options = dict(
        groups=None,
        context_length=65536,
        max_batch_size=16,
        prefill_chunk_size=4096,
        draft_tokens=3,
    )
    create_runner(spec, device=device, **options)
    args, kwargs = calls[0]
    assert args[0] == str(spec.path)
    assert kwargs["card"] == card and kwargs["mtp"] == 3 and kwargs["drafter"] is None
    assert kwargs["max_seq_len"] == 65536 + 5
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(
        json.dumps({"dflash_config": {"block_size": 8}})
    )
    create_runner(
        spec,
        device=device,
        **{**options, "draft_tokens": 7},
        assistant_checkpoint=drafter,
    )
    args, kwargs = calls[1]
    assert kwargs["mtp"] == 7 and kwargs["drafter"] == drafter
    assert kwargs["max_seq_len"] == 65536 + 7
    create_runner(
        spec,
        device=device,
        **{**options, "draft_tokens": 0},
        assistant_checkpoint=drafter,
    )
    assert calls[2][1]["drafter"] is None
    # the option check: up to 3 MTP drafts, up to the drafter's block with -md
    settings = dict(max_batch_size=16, prefill_chunk_size=4096)
    validate_gaudi_options(spec, draft_tokens=3, **settings)
    with pytest.raises(ValueError, match="--draft-tokens"):
        validate_gaudi_options(spec, draft_tokens=7, **settings)
    validate_gaudi_options(spec, draft_tokens=7, assistant=drafter, **settings)
    with pytest.raises(ValueError, match="--draft-tokens"):
        validate_gaudi_options(spec, draft_tokens=8, assistant=drafter, **settings)


@pytest.mark.parametrize("device, card", [("hpu", None), ("hpu:5", 5)])
def test_mistral_large4_factory_arguments_match_runner(
    monkeypatch, tmp_path, device, card
):
    module = pytest.importorskip("resin_models_gaudi.mistral_large4.runner")
    spec = get_model("mistral_large4-gaudi2")
    constructor = signature(module.MistralLarge4GaudiRunner.__init__)
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            constructor.bind(self, *args, **kwargs)
            calls.append((args, kwargs))

    monkeypatch.setattr(module, "MistralLarge4GaudiRunner", Runner)
    groups = SimpleNamespace(topology=SimpleNamespace(world_size=8))
    options = dict(
        groups=groups,
        context_length=65536,
        max_batch_size=16,
        prefill_chunk_size=module.PREFILL_CHUNK_SIZE,
        draft_tokens=0,
    )
    (tmp_path / "UD-Q4_K_XL").mkdir()
    create_runner(spec, tmp_path, device=device, **options)
    args, kwargs = calls[0]
    assert args[0] == str(tmp_path / "UD-Q4_K_XL")  # the quantization's subdirectory
    assert kwargs["card"] == card and kwargs["groups"] is groups
    assert kwargs["max_seq_len"] == 65536 and kwargs["max_batch_size"] == 16
    assert spec.world_size == spec.attn_tp == 8 and spec.tokenizer is not None
    assert kwargs["draft_checkpoint"] is None
    assert kwargs["prefill_chunk_size"] == module.PREFILL_CHUNK_SIZE
    assert spec.prefill_chunk_size == module.PREFILL_CHUNK_SIZE
    wider = {**options, "prefill_chunk_size": 4096}
    create_runner(spec, tmp_path, device=device, **wider)
    assert calls.pop()[1]["prefill_chunk_size"] == 4096
    with pytest.raises(ValueError, match="blocks of 3"):
        create_runner(spec, tmp_path, device=device, **{**options, "draft_tokens": 2})
    create_runner(
        spec,
        tmp_path,
        device=device,
        assistant_checkpoint=tmp_path / "draft",
        **{**options, "draft_tokens": 3},
    )
    _, kwargs = calls[1]
    assert kwargs["draft_checkpoint"] == str(tmp_path / "draft")
    assert kwargs["max_seq_len"] == 65536 + 3  # a verify step's lookahead
    with pytest.raises(ValueError, match="hpu"):
        create_runner(spec, tmp_path, device="cuda:0", **options)


def test_text_only_recipe_refuses_media():
    from deploy.models import TextOnlyRecipe

    text = SimpleNamespace(media=None)
    image = SimpleNamespace(media=object())
    native = SimpleNamespace(
        stop_token_ids=[1],
        tokenizer="tokenizer",
        render=lambda messages, **_: messages,
        create_parser=lambda prompt: ("parser", prompt),
    )
    recipe = TextOnlyRecipe(native)
    assert recipe.render(text) is text and recipe.stop_token_ids == [1]
    assert recipe.create_parser(text) == ("parser", text)
    with pytest.raises(ValueError, match="text only"):
        recipe.render(image)


THINKING_LEVELS = {
    "qwen3_5": ("low", "medium", "xhigh"),
    "qwen4_exp": ("low", "medium", "xhigh"),
    "glm5_3_flash": ("low", "high", "max"),
    "deepseek_v41": ("low", "high", "max"),
    "mistral_large4": ("low", "medium", "high"),
}


@pytest.mark.parametrize("name", sorted(MODEL_SPECS))
def test_model_capabilities_come_from_the_preset_and_its_recipe(name, tmp_path):
    from test_recipe import _byte_tokenizer

    from deploy.models import create_recipe, model_capabilities

    spec = MODEL_SPECS[name]
    _byte_tokenizer().save(str(tmp_path / "tokenizer.json"))
    recipe = create_recipe(spec, tmp_path, tokenizer_checkpoint=tmp_path)
    capabilities = model_capabilities(spec, recipe)
    assert capabilities.function_calling and capabilities.reasoning
    # The dense Qwen3.5 recipe ignores reasoning_effort, thus it has no levels.
    expected = None if spec.recipe == "qwen3_5" else THINKING_LEVELS[spec.family]
    assert capabilities.thinking_levels == expected
    assert capabilities.input_modalities == ("text",) and not spec.media
    assert capabilities.output_modalities == ("text",)
    media = model_capabilities(replace(spec, media=True), recipe)
    assert media.input_modalities == ("text", "image")


def make_checkpoint(path, required_files):
    path.mkdir(parents=True)
    for name in required_files:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


@pytest.mark.parametrize("name", MODEL_SPECS)
def test_existing_complete_cache_adoption_never_downloads(monkeypatch, tmp_path, name):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    spec = replace(get_model(name), model_mount=str(tmp_path))
    make_checkpoint(spec.path, spec.checkpoint.required_files)
    if spec.assistant:
        make_checkpoint(spec.assistant_path, spec.assistant.required_files)
    if spec.tokenizer:
        make_checkpoint(spec.tokenizer_path, spec.tokenizer.required_files)

    def no_download(*args, **kwargs):
        raise AssertionError("unexpected download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        require_checkpoint(spec)
    report = download_checkpoint(spec, metadata_only=True)
    assert report["cache_hit"] and report["metadata_only"]
    assert require_checkpoint(spec) == spec.path
    weights = (
        spec.checkpoint.required_files[0]
        if spec.checkpoint.weights_listed
        else "model.safetensors"
    )
    (spec.path / weights).write_bytes(b"changed size")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        require_checkpoint(spec)


def test_sampling_defaults_come_from_generation_config(tmp_path):
    from deploy.models import SamplingDefaults

    assert SamplingDefaults.read(tmp_path) == SamplingDefaults()
    (tmp_path / "generation_config.json").write_text(
        '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "do_sample": true}'
    )
    assert SamplingDefaults.read(tmp_path) == SamplingDefaults(1.0, 0.95, 20)
    (tmp_path / "generation_config.json").write_text(
        '{"temperature": "hot", "top_k": true}'
    )
    assert SamplingDefaults.read(tmp_path) == SamplingDefaults()
