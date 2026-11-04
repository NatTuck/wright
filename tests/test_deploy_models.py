"""Shared serving/benchmark configuration preserves runner and checkpoint contracts."""

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
def test_gaudi_factory_arguments_match_runner(monkeypatch, device, card):
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
    assert kwargs["max_batch_size"] == 8
    with pytest.raises(ValueError, match="hpu"):
        create_runner(spec, device="cuda:0", **options)
    with pytest.raises(ValueError, match="4096"):
        create_runner(spec, device=device, **{**options, "prefill_chunk_size": 8192})


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
