"""Pinned model configuration shared by serving and benchmark launchers.

Model-specific constructor differences live here; launchers share checkpoint handling and runtime
lifecycle code.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deploy.checkpoint_cache import (
    checkpoint_inventory,
    checkpoint_is_complete,
    publish_checkpoint,
)

GLM53_ZAI_REVISION = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
ML4_GGUF_QUANT = "UD-Q4_K_XL"  # the GGUF quantization the Gaudi lq GEMVs run


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    repo_id: str
    revision: str
    directory: str
    required_files: tuple[str, ...] = ("config.json", "tokenizer.json")
    patterns: tuple[str, ...] = (
        "*.json",
        "*.jinja",
        "*.model",
        "tokenizer*",
        "model*.safetensors",
        "merges.txt",
        "vocab.json",
    )
    # required_files lists every file, weights included (GGUF shards, tokenizer-only parts)
    weights_listed: bool = False


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    name: str
    family: str
    checkpoint: Checkpoint
    model_mount: str
    model_volume: str
    gpu: str
    label: str
    app_name: str | None = None
    model_volume_version: int = 1
    kernel_mount: str | None = None
    kernel_volume: str | None = None
    kernel_volume_version: int = 1
    assistant: Checkpoint | None = None
    # A tokenizer / chat-template repository, for weights published without one (GGUF).
    tokenizer: Checkpoint | None = None
    world_size: int = 1
    attn_dp: int = 1
    attn_tp: int = 1
    ep: int = 1
    context_length: int = 131_072
    max_batch_size: int = 128
    prefill_chunk_size: int = 4096
    draft_tokens: int = 0
    gpu_memory_utilization: float = 0.9
    expert_backend: str = "flashinfer"
    memory: int = 256 * 1024
    enable_anthropic: bool = True
    image_warmup: bool = False
    # "cuda" or "gaudi2" (Synapse; Torch only for the Gloo control plane). Modal serves CUDA presets only.
    accelerator: str = "cuda"
    # False: the runner has no media encoder, so requests with media are refused.
    media: bool = True
    # The chat recipe, when the family's default does not fit this checkpoint.
    recipe: str | None = None
    # Defaults for the chat recipe, as (name, value) pairs.
    recipe_options: tuple[tuple[str, Any], ...] = ()
    # DeepSeek-V4.1 Engram: "card" shards the lookup tables across the cards; "host" keeps them in
    # host RAM (a shared mmap of the checkpoint shards) and gathers/dequantizes the rows per step in
    # the native fill. "host" is what fits the tables on 4 cards at 512k context.
    engram: str = "card"
    engram_gather: str = "auto"
    engram_prewarm: bool = False

    @property
    def repo_id(self) -> str:
        return self.checkpoint.repo_id

    @property
    def revision(self) -> str:
        return self.checkpoint.revision

    @property
    def path(self) -> Path:
        return Path(self.model_mount) / self.checkpoint.directory / self.revision

    @property
    def assistant_path(self) -> Path | None:
        if self.assistant is None:
            return None
        return (
            Path(self.model_mount) / self.assistant.directory / self.assistant.revision
        )

    @property
    def tokenizer_path(self) -> Path:
        """Where ``tokenizer.json`` lives: the tokenizer checkpoint, else the weights'."""
        if self.tokenizer is None:
            return self.path
        return (
            Path(self.model_mount) / self.tokenizer.directory / self.tokenizer.revision
        )


MODEL_SPECS = {
    "qwen3_5-gaudi2": ModelSpec(
        name="qwen3_5-gaudi2",
        family="qwen3_5",
        accelerator="gaudi2",
        checkpoint=Checkpoint(
            repo_id="Qwen/Qwen3.8-27B-FP8",
            revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            directory="",
            required_files=(
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "mtp.safetensors",
            ),
            patterns=(
                "*.json",
                "*.jinja",
                "*.safetensors",
                "merges.txt",
                "vocab.json",
            ),
        ),
        model_mount="/models/resin-qwen3-5-fp8",
        model_volume="resin-qwen3-5-fp8-models-v1",
        gpu="HL-225",
        label="qwen3-8-27b-gaudi2",
        context_length=32_768,
        max_batch_size=64,
        draft_tokens=2,
        media=False,
    ),
    # The smaller dense Qwen3.5 models, in FP8 from a third party, with no MTP weights.
    **{
        name: ModelSpec(
            name=name,
            family="qwen3_5",
            accelerator="gaudi2",
            checkpoint=Checkpoint(
                repo_id=f"surogate/Qwen3.5-{size}-FP8",
                revision=revision,
                directory="",
                required_files=(
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
                patterns=(
                    "*.json",
                    "*.jinja",
                    "*.safetensors",
                    "merges.txt",
                    "vocab.json",
                ),
            ),
            model_mount="/models/resin-qwen3-5-dense-fp8",
            model_volume="resin-qwen3-5-dense-fp8-models-v1",
            gpu="HL-225",
            label=name,
            context_length=32_768,
            max_batch_size=32,
            draft_tokens=0,
            media=False,
            recipe="qwen3_5",
            # The official template of the two smallest sizes starts without thinking.
            recipe_options=(
                (("enable_thinking", False),) if size in ("0.8B", "2B") else ()
            ),
        )
        for name, size, revision in (
            (
                "qwen3_5-0.8b-gaudi2",
                "0.8B",
                "1423a80c469215a6045bd4cdb9f342fcc293d0ea",
            ),
            ("qwen3_5-2b-gaudi2", "2B", "c11f46d26e90d3ec37b858163849983c45957d73"),
            ("qwen3_5-4b-gaudi2", "4B", "f69d4910303be8136e8c7adbfa2128999029a554"),
            ("qwen3_5-9b-gaudi2", "9B", "0d8b9bd14f1e192f0094a3d97b7900b82b475c12"),
        )
    },
    # Text only: no vision tower. 8 cards serve; 4 for tests, so disjoint runs can coexist.
    **{
        name: ModelSpec(
            name=name,
            family="glm5_3_flash",
            accelerator="gaudi2",
            checkpoint=Checkpoint(
                repo_id="zai-org/GLM-5.3-Flash",
                revision=GLM53_ZAI_REVISION,
                directory="",
                required_files=(
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                ),
                patterns=("*.json", "*.jinja", "*.safetensors"),
            ),
            model_mount="/models/resin-glm5-3-flash-fp8",
            model_volume="resin-glm5-3-flash-fp8-models-v1",
            gpu="HL-225",
            label=name,
            world_size=ranks,
            attn_tp=ranks,
            context_length=context,
            max_batch_size=64,
            draft_tokens=0,
            prefill_chunk_size=4096,
            media=False,
        )
        for name, ranks, context in (
            ("glm5_3_flash-gaudi2", 8, 65_536),
            ("glm5_3_flash-gaudi2-tp4", 4, 32_768),
            ("glm5_3_flash-gaudi2-256k", 8, 262_144),
        )
    },
    # Text only (the vision tower is not supported). The experts and the n-gram table shard over the cards;
    # contexts stay within the QSA indexer's dense budget until the sparse path is wired.
    **{
        name: ModelSpec(
            name=name,
            family="qwen4_exp",
            accelerator="gaudi2",
            checkpoint=Checkpoint(
                repo_id="Qwen/Qwen3.8-Flash-Next-FP8",
                revision="7fb48d708ca6cd8eeed8f4db0433be18af96984f",
                directory="",
                required_files=(
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                ),
                patterns=(
                    "*.json",
                    "*.jinja",
                    "*.safetensors",
                    "merges.txt",
                    "vocab.json",
                ),
            ),
            model_mount="/models/resin-qwen3-8-flash-next-fp8",
            model_volume="resin-qwen3-8-flash-next-fp8-models-v1",
            gpu="HL-225",
            label=name,
            world_size=ranks,
            attn_tp=ranks,
            context_length=2048,
            max_batch_size=16,
            draft_tokens=0,
            prefill_chunk_size=2048,
            media=False,
            recipe="qwen3_8",  # its template renders as Qwen3.8's (it only adds generation markers)
        )
        for name, ranks in (("qwen4_exp-gaudi2", 8), ("qwen4_exp-gaudi2-tp4", 4))
    },
    # Text only: no vision tower. 8 cards; DSpark drafting with --draft-tokens 1..5.
    "deepseek_v41-gaudi2": ModelSpec(
        name="deepseek_v41-gaudi2",
        family="deepseek_v41",
        accelerator="gaudi2",
        checkpoint=Checkpoint(
            repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
            directory="",
            required_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
            patterns=("*.json", "model*.safetensors"),
        ),
        model_mount="/models/deepseek-v41-flash-1017",
        model_volume="resin-deepseek-v41-flash-1017-models-v1",
        gpu="HL-225",
        label="deepseek_v41-gaudi2",
        world_size=8,
        attn_tp=8,
        context_length=131_072,
        max_batch_size=64,
        draft_tokens=0,
        prefill_chunk_size=2048,
        media=False,
    ),
    # 4 cards, 512k context, batch 1-8: the 188.8 GiB Engram tables do not fit on 4 cards (47.2 GiB/card
    # of table crowds out the KV arena), so they stay in host RAM (a shared mmap of the checkpoint
    # shards) and the native fill gathers/dequantizes the rows per step. Host-only; prewarm makes the
    # gather RAM-resident. Text only (no vision tower).
    "deepseek_v41-gaudi2-tp4": ModelSpec(
        name="deepseek_v41-gaudi2-tp4",
        family="deepseek_v41",
        accelerator="gaudi2",
        checkpoint=Checkpoint(
            repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
            directory="",
            required_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
            patterns=("*.json", "model*.safetensors"),
        ),
        model_mount="/models/deepseek-v41-flash-1017",
        model_volume="resin-deepseek-v41-flash-1017-models-v1",
        gpu="HL-225",
        label="deepseek_v41-gaudi2-tp4",
        world_size=4,
        attn_tp=4,
        context_length=512_000,
        max_batch_size=8,
        draft_tokens=0,
        prefill_chunk_size=2048,
        media=False,
        engram="host",
        engram_gather="mmap",
        engram_prewarm=True,
    ),
    # Text only: the GGUF has no Pixtral tower; the tokenizer and chat template come from
    # the official release (its vocabulary equals the GGUF's, id for id).
    "mistral_large4-gaudi2": ModelSpec(
        name="mistral_large4-gaudi2",
        family="mistral_large4",
        accelerator="gaudi2",
        checkpoint=Checkpoint(
            repo_id="unsloth/Mistral-Large-4-1T-A52B-GGUF",
            revision="1eb4373d37e76f6e7b4b8efa20e1b1ca40b3b964",
            directory="",
            required_files=tuple(
                f"{ML4_GGUF_QUANT}/Mistral-Large-4-1T-A52B-{ML4_GGUF_QUANT}-{i:05d}-of-00014.gguf"
                for i in range(1, 15)
            ),
            patterns=(f"{ML4_GGUF_QUANT}/*.gguf",),
            weights_listed=True,
        ),
        tokenizer=Checkpoint(
            repo_id="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
            revision="bb6d00fe249b0e876f0cffc6b85014be02de3e68",
            directory="tokenizer",
            required_files=("tokenizer.json", "chat_template.jinja", "params.json"),
            patterns=("tokenizer*.json", "chat_template.jinja", "params.json"),
            weights_listed=True,
        ),
        # the DSpark drafter: the release's last shard holds only its tensors
        assistant=Checkpoint(
            repo_id="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
            revision="bb6d00fe249b0e876f0cffc6b85014be02de3e68",
            directory="draft",
            required_files=(
                "params.json",
                "consolidated-00032-of-00032.safetensors",
            ),
            patterns=("params.json", "consolidated-00032-of-00032.safetensors"),
            weights_listed=True,
        ),
        model_mount="/models/resin-mistral-large4",
        model_volume="resin-mistral-large4-models-v1",
        gpu="HL-225",
        label="mistral-large-4-gaudi2",
        world_size=8,
        attn_tp=8,
        context_length=65_536,
        max_batch_size=16,
        prefill_chunk_size=2048,
        media=False,
    ),
}


def get_model(model: str | ModelSpec) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    try:
        return MODEL_SPECS[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; choose from {', '.join(MODEL_SPECS)}"
        ) from None


def checkpoint_parts(spec: ModelSpec):
    yield spec.checkpoint, spec.path
    if spec.assistant is not None:
        yield spec.assistant, spec.assistant_path
    if spec.tokenizer is not None:
        yield spec.tokenizer, spec.tokenizer_path


def download_checkpoint(model: str | ModelSpec, *, metadata_only: bool = False) -> dict:
    """Download missing files and publish identical inventories for every launcher.

    Complete existing volumes are adopted without downloading their weight bodies.
    The caller owns Modal volume reload/commit, so this also works on local paths.
    """
    spec = get_model(model)
    started, inventory, cache_hit = time.perf_counter(), {}, True
    for checkpoint, path in checkpoint_parts(spec):
        try:
            checkpoint_inventory(
                path,
                required_files=checkpoint.required_files,
                weights_listed=checkpoint.weights_listed,
            )
        except RuntimeError:
            if metadata_only:
                raise
            from huggingface_hub import snapshot_download

            cache_hit = False
            path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                checkpoint.repo_id,
                revision=checkpoint.revision,
                local_dir=path,
                allow_patterns=list(checkpoint.patterns),
            )
        files = publish_checkpoint(
            path,
            checkpoint.repo_id,
            checkpoint.revision,
            required_files=checkpoint.required_files,
            weights_listed=checkpoint.weights_listed,
        )
        inventory.update(
            {
                str(path.relative_to(spec.model_mount) / name): size
                for name, size in files.items()
            }
        )
    return dict(
        model=spec.name,
        repo_id=spec.repo_id,
        revision=spec.revision,
        path=str(spec.path),
        cache_hit=cache_hit,
        metadata_only=metadata_only,
        files=len(inventory),
        bytes=sum(inventory.values()),
        elapsed_s=time.perf_counter() - started,
    )


def require_checkpoint(
    model: str | ModelSpec,
    *,
    checkpoint: str | Path | None = None,
    assistant_checkpoint: str | Path | None = None,
    tokenizer_checkpoint: str | Path | None = None,
    draft_tokens: int | None = None,
) -> Path:
    """Require a published volume, or validate explicitly supplied local weights."""
    spec = get_model(model)
    for part, path in checkpoint_parts(spec):
        if part is spec.assistant and draft_tokens == 0:
            continue
        override = (
            assistant_checkpoint
            if part is spec.assistant
            else tokenizer_checkpoint
            if part is spec.tokenizer
            else checkpoint
        )
        if override is not None:
            checkpoint_inventory(
                Path(override),
                required_files=part.required_files,
                weights_listed=part.weights_listed,
            )
        elif not checkpoint_is_complete(
            path,
            part.repo_id,
            part.revision,
            required_files=part.required_files,
            weights_listed=part.weights_listed,
        ):
            raise FileNotFoundError(
                f"incomplete {spec.name} checkpoint: run the download command (or --publish-only for existing files)"
            )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return spec.path if checkpoint is None else Path(checkpoint)


def warm_engine(engine, recipe, spec: ModelSpec, rank: int = 0) -> None:
    """Run a short greedy request (and an image one on presets that warm their media
    tower) through a fresh engine before it serves."""
    import numpy as np

    from resin import SamplingParams
    from resin.recipe import Image, Message, Role, Text

    # Exercise several decode steps, rather than ending after a one-word reply.
    content = [Text("Count from 1 to 100.")]
    for image_input in (False, True) if spec.image_warmup else (False,):
        lease = None
        # This function runs only on attention replica drivers. Warm each
        # replica's image tower; round-robin or sticky HTTP routing may use any
        # of them immediately after startup.
        if rank == 0 or image_input:
            parts = content
            if image_input:
                axis = np.linspace(0, 1, 96, dtype=np.float32)
                horizontal = np.broadcast_to(axis[None], (96, 96))
                pixels = np.stack(
                    (horizontal, horizontal.T, 1 - horizontal), axis=-1
                ).copy()
                parts = [Image(source=pixels), Text("Name one visible color.")]
            prompt = recipe.render([Message(Role.USER, parts)])
            inputs = (
                {"media": prompt.media}
                if image_input
                else {"token_ids": prompt.token_ids}
            )
            lease = engine.submit(
                **inputs,
                sampling_params=SamplingParams(
                    max_output_tokens=64,
                    stop_token_ids=recipe.stop_token_ids,
                    temperature=0.0,
                ),
            )
        while engine.has_work():
            engine.step()
        if lease is not None:
            lease.take_result()


# the families whose ranks can exchange their steps natively (``--step-transport native``: no Gloo after the start)
NATIVE_RANK_FAMILIES = ("qwen3_5", "deepseek_v41", "glm5_3_flash")


def kv_cache_types(spec: ModelSpec) -> tuple[str, ...]:
    """The types that a model's KV cache pages can have (``--cache-type-k``, ``--cache-type-v``), its default
    first: bf16 or fp8 for the K and the V pages of Qwen3.5/3.8; fp8 only for the latent caches of
    DeepSeek-V4.1-Flash, GLM-5.3-Flash and Mistral-Large-4; bf16 only for Qwen3.8-Flash-Next."""
    if spec.family == "qwen3_5":
        return ("bf16", "fp8")
    if spec.family == "qwen4_exp":
        return ("bf16",)
    return ("fp8",)


def check_cache_types(
    spec: ModelSpec, cache_types: tuple[str | None, str | None] | None
) -> tuple[str, str]:
    """``cache_types`` (K, V; None: the model's default) as the model's types, or a ValueError."""
    kinds = kv_cache_types(spec)
    k, v = cache_types or (None, None)
    for flag, kind in (("--cache-type-k", k), ("--cache-type-v", v)):
        if kind is not None and kind not in kinds:
            raise ValueError(
                f"{spec.name} keeps its KV cache as {' or '.join(kinds)}: {flag} {kind} is not one of them"
            )
    return (k or kinds[0], v or kinds[0])


def create_runner(
    model: str | ModelSpec,
    checkpoint: str | Path | None = None,
    *,
    device,
    groups=None,
    context_length: int,
    max_batch_size: int,
    prefill_chunk_size: int,
    draft_tokens: int,
    gpu_memory_utilization: float | None = None,
    assistant_checkpoint: str | Path | None = None,
    num_layers: int | None = None,
    defer_cache: bool = False,
    native_ranks=None,
    cache_types: tuple[str | None, str | None] | None = None,
    engram: str | None = None,
    engram_gather: str | None = None,
    engram_prewarm: bool | None = None,
):
    """Construct one runner with model-specific cache geometry and draft support.
    ``defer_cache``: load the weights only, for a card that several models share (the runner's
    ``attach`` gives it the shared cache arena). ``native_ranks``: a Qwen3.5/3.8 rank's
    ``resin._resin.GaudiRanks`` (``--step-transport native``), in place of ``groups``.
    ``cache_types``: the types of the K and V cache pages (``kv_cache_types``; None: the model's).
    ``engram``/``engram_gather``/``engram_prewarm``: DeepSeek-V4.1 Engram overrides (None: the preset's)."""
    spec = get_model(model)
    checkpoint = spec.path if checkpoint is None else Path(checkpoint)
    return _create_gaudi_runner(
        spec,
        checkpoint,
        device=device,
        groups=groups,
        context_length=context_length,
        max_batch_size=max_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        draft_tokens=draft_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        assistant=(
            spec.assistant_path
            if assistant_checkpoint is None
            else Path(assistant_checkpoint)
        ),
        num_layers=num_layers,
        defer_cache=defer_cache,
        native_ranks=native_ranks,
        cache_types=check_cache_types(spec, cache_types),
        engram=engram,
        engram_gather=engram_gather,
        engram_prewarm=engram_prewarm,
    )


def _create_gaudi_runner(
    spec: ModelSpec,
    checkpoint: Path,
    *,
    device,
    groups,
    context_length: int,
    max_batch_size: int,
    prefill_chunk_size: int,
    draft_tokens: int,
    gpu_memory_utilization: float | None,
    assistant: Path | None = None,
    num_layers: int | None = None,
    defer_cache: bool = False,
    native_ranks=None,
    cache_types: tuple[str, str] = ("bf16", "bf16"),
    engram: str | None = None,
    engram_gather: str | None = None,
    engram_prewarm: bool | None = None,
):
    """One Gaudi card per rank, selected by ``hpu:N`` (its accel index) or the first free one.
    ``cache_types``: checked by ``check_cache_types``; only Qwen3.5/3.8 has a choice."""
    if defer_cache and not gaudi_defers_cache(spec):
        raise ValueError(f"{spec.name} does not share a card with other models yet")
    if native_ranks is not None and spec.family not in NATIVE_RANK_FAMILIES:
        raise ValueError(f"{spec.name} has no native rank dispatch")
    # Engram flags are DeepSeek-V4.1-only; resolve overrides against the preset.
    if (engram is not None or engram_gather is not None or engram_prewarm is not None) and (
        spec.family != "deepseek_v41"
    ):
        raise ValueError(f"{spec.name} has no Engram tables (--engram/--engram-gather/--engram-prewarm)")
    engram = spec.engram if engram is None else engram
    engram_gather = spec.engram_gather if engram_gather is None else engram_gather
    engram_prewarm = spec.engram_prewarm if engram_prewarm is None else engram_prewarm
    if engram not in ("card", "host"):
        raise ValueError(f"engram {engram!r}: 'card' or 'host'")
    if engram_gather not in ("pooled", "mmap", "auto"):
        raise ValueError(f"engram_gather {engram_gather!r}: 'pooled', 'mmap' or 'auto'")
    if spec.family == "glm5_3_flash":
        from resin_models_gaudi.glm5_3_flash.runner import (
            PREFILL_CHUNK_SIZES,
            Glm53FlashGaudiRunner,
            lookahead_tokens,
        )

        # The runner keeps its compiled prefill widths. The Engine takes the smaller chunk.
        if prefill_chunk_size not in PREFILL_CHUNK_SIZES:
            raise ValueError(
                f"{spec.name} prefills in chunks of {', '.join(map(str, PREFILL_CHUNK_SIZES))} tokens"
            )
        # a speculative step near the end of a context writes past it: the DFlash drafter's K
        # positions, or an MTP step's 2K - 1
        drafter = None if draft_tokens == 0 else assistant
        lookahead = lookahead_tokens(draft_tokens, drafter is not None)
        return Glm53FlashGaudiRunner(
            str(checkpoint),
            groups=groups,
            native_ranks=native_ranks,
            card=_gaudi_card(spec, device),
            max_batch_size=max_batch_size,
            max_seq_len=context_length + lookahead,
            num_layers=num_layers,
            mtp=draft_tokens,
            drafter=drafter,
            defer_cache=defer_cache,
        )
    if spec.family == "deepseek_v41":
        from resin_models_gaudi.deepseek_v41.runner import (
            PREFILL_CHUNK_SIZE,
            DeepSeekV41GaudiRunner,
        )

        if prefill_chunk_size != PREFILL_CHUNK_SIZE:
            raise ValueError(
                f"{spec.name} prefills in {PREFILL_CHUNK_SIZE}-token chunks"
            )
        return DeepSeekV41GaudiRunner(
            str(checkpoint),
            groups=groups,
            native_ranks=native_ranks,
            card=_gaudi_card(spec, device),
            max_batch_size=max_batch_size,
            max_seq_len=context_length + draft_tokens,
            num_layers=num_layers,
            draft_tokens=draft_tokens,
            defer_cache=defer_cache,
            engram=engram,
            engram_prewarm=engram_prewarm,
        )
    if spec.family == "qwen4_exp":
        from resin_models_gaudi.qwen4_exp.runner import (
            PREFILL_CHUNK_SIZE as FNEXT_CHUNK,
        )
        from resin_models_gaudi.qwen4_exp.runner import Qwen4ExpGaudiRunner

        if prefill_chunk_size != FNEXT_CHUNK:
            raise ValueError(f"{spec.name} prefills in {FNEXT_CHUNK}-token chunks")
        if draft_tokens:
            raise ValueError(f"{spec.name} has no speculative decoding yet")
        return Qwen4ExpGaudiRunner(
            checkpoint,
            groups=groups,
            card=_gaudi_card(spec, device),
            max_batch_size=max_batch_size,
            max_seq_len=context_length,
            memory_utilization=spec.gpu_memory_utilization
            if gpu_memory_utilization is None
            else gpu_memory_utilization,
            defer_cache=defer_cache,
        )
    if spec.family == "mistral_large4":
        return _create_mistral_large4_runner(
            spec,
            checkpoint,
            device=device,
            groups=groups,
            context_length=context_length,
            max_batch_size=max_batch_size,
            prefill_chunk_size=prefill_chunk_size,
            draft_tokens=draft_tokens,
            assistant=assistant,
            num_layers=num_layers,
        )
    from resin_models_gaudi.qwen3_5.runner import (
        PREFILL_CHUNK_SIZES,
        Qwen3_5GaudiRunner,
        lookahead_tokens,
    )

    # The runner keeps its compiled prefill widths. The Engine takes the smaller chunk.
    if prefill_chunk_size not in PREFILL_CHUNK_SIZES:
        raise ValueError(
            f"{spec.name} prefills in chunks of {', '.join(map(str, PREFILL_CHUNK_SIZES))} tokens"
        )
    card = None
    if device not in (None, "hpu"):
        kind, _, index = str(device).partition(":")
        if kind != "hpu" or not index.isdigit():
            raise ValueError(
                f"{spec.name} devices are 'hpu' or 'hpu:N' (got {device!r})"
            )
        card = int(index)
    # a speculative step near the end of a context writes past it: the MTP chain's reach, or
    # the verified columns with the DFlash 2 drafter (``assistant``)
    drafter = assistant if draft_tokens else None
    lookahead = lookahead_tokens(draft_tokens, drafter is not None)
    return Qwen3_5GaudiRunner(
        checkpoint,
        card=card,
        max_batch_size=max_batch_size,
        max_seq_len=context_length + lookahead,
        memory_utilization=spec.gpu_memory_utilization
        if gpu_memory_utilization is None
        else gpu_memory_utilization,
        drafter=drafter,
        mtp=draft_tokens,
        defer_cache=defer_cache,
        groups=groups,
        native_ranks=native_ranks,
        cache_types=cache_types,
    )


def gaudi_defers_cache(spec: ModelSpec) -> bool:
    """Whether this preset's runner can load its weights first and take a cache arena that other
    models share later (``create_runner(defer_cache=True)``, then the runner's ``attach``)."""
    return spec.family in ("qwen3_5", "deepseek_v41", "glm5_3_flash", "qwen4_exp")


def gaudi_supports_ranks(spec: ModelSpec) -> bool:
    """Whether this preset's runner declares tensor parallelism."""
    if spec.family == "glm5_3_flash":
        from resin_models_gaudi.glm5_3_flash.runner import (
            Glm53FlashGaudiRunner as runner,
        )
    elif spec.family == "mistral_large4":
        from resin_models_gaudi.mistral_large4.runner import (
            MistralLarge4GaudiRunner as runner,
        )
    elif spec.family == "deepseek_v41":
        from resin_models_gaudi.deepseek_v41.runner import (
            DeepSeekV41GaudiRunner as runner,
        )
    elif spec.family == "qwen4_exp":
        from resin_models_gaudi.qwen4_exp.runner import Qwen4ExpGaudiRunner as runner
    elif spec.family == "qwen3_5":
        from resin_models_gaudi.qwen3_5.runner import Qwen3_5GaudiRunner as runner
    else:
        return False
    return runner.parallelism_capabilities.attn_tp


def check_gaudi_ranks(spec: ModelSpec, checkpoint: Path, ranks: int) -> None:
    """Raise ValueError unless the checkpoint's widths divide across ``ranks``."""
    if spec.family == "glm5_3_flash":
        from resin.models.glm5_3_flash.config import Glm53Config
        from resin_models_gaudi.glm5_3_flash.weights import RankSlices

        raw = json.loads((checkpoint / "config.json").read_text())
        RankSlices.of(Glm53Config.from_dict(raw), 0, ranks)
    elif spec.family == "deepseek_v41":
        from resin.models.deepseek_v41 import DeepSeekV41Config

        raw = json.loads((checkpoint / "config.json").read_text())
        DeepSeekV41Config.from_hugging_face(raw).validate_tensor_parallel(ranks)
    elif spec.family == "qwen4_exp":
        from resin_models_gaudi.qwen4_exp.checkpoint import Qwen4ExpCheckpoint

        Qwen4ExpCheckpoint(checkpoint).shard(0, ranks)
    elif spec.family == "qwen3_5":
        from resin_models_gaudi.qwen3_5.checkpoint import Qwen35Checkpoint

        Qwen35Checkpoint(checkpoint).shard(0, ranks)
    else:
        from resin.models.mistral_large4.config import MistralLarge4Config
        from resin_models_gaudi.gguf.reader import GgufModel
        from resin_models_gaudi.mistral_large4.weights import RankSlices

        gguf = checkpoint / ML4_GGUF_QUANT
        cfg = MistralLarge4Config.from_gguf(
            GgufModel(str(gguf if gguf.is_dir() else checkpoint)).metadata
        )
        RankSlices.of(cfg, 0, ranks)


DFLASH_FAMILIES = frozenset({"qwen3_5", "glm5_3_flash"})
"""Presets whose ``--assistant-checkpoint`` names a DFlash 2 drafter directory (Qwen3.8-27B: or a
DSpine drafter's)."""


def default_draft_tokens(spec: ModelSpec, assistant: str | Path | None = None) -> int:
    """The preset's draft tokens, or the drafts a block of a DFlash 2 or DSpine drafter makes."""
    if spec.family in DFLASH_FAMILIES and assistant is not None:
        from resin_models_gaudi.common.dflash import drafter_draft_tokens

        return drafter_draft_tokens(assistant)
    return spec.draft_tokens


def check_gaudi_drafts(
    spec: ModelSpec,
    checkpoint: Path,
    draft_tokens: int,
    assistant: str | Path | None = None,
) -> None:
    """Raise ValueError if MTP drafts are asked for and the checkpoint has no MTP weights (a
    DFlash 2 drafter, ``assistant``, needs none)."""
    if spec.family != "qwen3_5" or not draft_tokens or assistant is not None:
        return
    if (checkpoint / "mtp.safetensors").exists():
        return
    index = checkpoint / "model.safetensors.index.json"
    if index.exists() and any(
        name.startswith("mtp.") for name in json.loads(index.read_text())["weight_map"]
    ):
        return
    raise ValueError(
        f"{spec.name} has no MTP weights in {checkpoint}, so it cannot make drafts. To add them, "
        "run scripts/qwen35_mtp_fp8.py, or use 0 draft tokens"
    )


def checkpoint_weight_bytes(spec: ModelSpec, checkpoint: Path) -> int:
    """The size of the weight files, for a fit check before they load."""
    root = checkpoint
    if spec.family == "mistral_large4" and (checkpoint / ML4_GGUF_QUANT).is_dir():
        root = checkpoint / ML4_GGUF_QUANT
    return sum(
        path.stat().st_size
        for path in root.iterdir()
        if path.suffix in (".gguf", ".safetensors")
    )


def engram_table_bytes(spec: ModelSpec, checkpoint: Path) -> int:
    """The Engram lookup tables' file bytes — ``sum(engram_num_embeddings) * 264`` (256 E4M3 codes +
    8 E8M0 scale bytes per row). Host-Engram keeps these in host RAM, not on the cards, so the fit
    check subtracts them from the card weight share. 0 for families without Engram tables."""
    if spec.family != "deepseek_v41":
        return 0
    import json

    cfg = json.loads((checkpoint / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    return sum(text.get("engram_num_embeddings", [])) * 264


def validate_gaudi_options(
    spec: ModelSpec,
    *,
    max_batch_size: int,
    prefill_chunk_size: int,
    draft_tokens: int,
    assistant: str | Path | None = None,
) -> None:
    """Raise ValueError for a setting that the preset's runner rejects after the card opens.
    ``assistant``: a draft model directory (Qwen3.8-27B and GLM-5.3-Flash: a DFlash 2
    drafter)."""
    if spec.family == "glm5_3_flash":
        from resin_models_gaudi.glm5_3_flash.runner import (
            MAX_BATCH,
            MTP_MAX,
            PREFILL_CHUNK_SIZES,
        )

        drafts, chunks = range(MTP_MAX + 1), PREFILL_CHUNK_SIZES
        if assistant is not None:  # up to the drafts the drafter's block makes
            drafts = range(default_draft_tokens(spec, assistant) + 1)
    elif spec.family == "deepseek_v41":
        from resin_models_gaudi.deepseek_v41.runner import MAX_BATCH, PREFILL_CHUNK_SIZE

        drafts, chunks = range(6), (PREFILL_CHUNK_SIZE,)  # DSpark drafts blocks of 5
    elif spec.family == "mistral_large4":
        from resin_models_gaudi.mistral_large4.dspark import DEPTH
        from resin_models_gaudi.mistral_large4.runner import MAX_BATCH

        drafts, chunks = (0, DEPTH), None
        if prefill_chunk_size < 1024 or prefill_chunk_size & (prefill_chunk_size - 1):
            raise ValueError(
                f"{spec.name} takes a --prefill-chunk-size that is a power of two, 1024 or "
                f"more, got {prefill_chunk_size}"
            )
    else:
        from resin_models_gaudi.qwen3_5.model import DECODE_BUCKETS
        from resin_models_gaudi.qwen3_5.runner import PREFILL_CHUNK_SIZES

        MAX_BATCH = DECODE_BUCKETS[-1]
        drafts, chunks = range(4), PREFILL_CHUNK_SIZES
        if assistant is not None:  # up to the drafts the drafter's block makes
            drafts = range(default_draft_tokens(spec, assistant) + 1)
    if max_batch_size > MAX_BATCH:
        raise ValueError(f"{spec.name} takes a --max-batch-size of {MAX_BATCH} or less")
    if draft_tokens not in drafts:
        raise ValueError(
            f"{spec.name} takes --draft-tokens {', '.join(map(str, drafts))}, got {draft_tokens}"
        )
    if chunks is not None and prefill_chunk_size not in chunks:
        raise ValueError(
            f"{spec.name} takes --prefill-chunk-size {', '.join(map(str, chunks))}, "
            f"got {prefill_chunk_size}"
        )


def _gaudi_card(spec: ModelSpec, device) -> int | None:
    if device in (None, "hpu"):
        return None
    kind, _, index = str(device).partition(":")
    if kind != "hpu" or not index.isdigit():
        raise ValueError(f"{spec.name} devices are 'hpu' or 'hpu:N' (got {device!r})")
    return int(index)


def _create_mistral_large4_runner(
    spec: ModelSpec,
    checkpoint: Path,
    *,
    device,
    groups,
    context_length: int,
    max_batch_size: int,
    prefill_chunk_size: int,
    draft_tokens: int,
    assistant: Path | None = None,
    num_layers: int | None = None,
):
    """One tensor-parallel rank of Mistral-Large-4 (the GGUF in ``checkpoint`` or its
    quantization subdirectory), with the DSpark drafter when ``draft_tokens`` asks for it."""
    from resin_models_gaudi.mistral_large4.dspark import DEPTH
    from resin_models_gaudi.mistral_large4.runner import MistralLarge4GaudiRunner

    if draft_tokens not in (0, DEPTH):
        raise ValueError(f"{spec.name} drafts blocks of {DEPTH} tokens (or none)")
    if draft_tokens and assistant is None:
        raise ValueError(f"{spec.name} needs the drafter's checkpoint to speculate")
    gguf = checkpoint / ML4_GGUF_QUANT
    return MistralLarge4GaudiRunner(
        str(gguf if gguf.is_dir() else checkpoint),
        groups=groups,
        card=_gaudi_card(spec, device),
        max_batch_size=max_batch_size,
        max_seq_len=context_length + draft_tokens,
        num_layers=num_layers,
        draft_checkpoint=str(assistant) if draft_tokens else None,
        prefill_chunk_size=prefill_chunk_size,
    )


class TextOnlyRecipe:
    """A recipe for a runner without a media encoder: requests with media fail at
    rendering, before they reach the Engine."""

    def __init__(self, recipe) -> None:
        self._recipe = recipe

    @property
    def stop_token_ids(self):
        return self._recipe.stop_token_ids

    @property
    def tokenizer(self):
        return self._recipe.tokenizer

    def render(self, messages, *, tools=(), options=None):
        prompt = self._recipe.render(messages, tools=tools, options=options)
        if prompt.media is not None:
            raise ValueError(
                "this model serves text only; remove image and video inputs"
            )
        return prompt

    def create_parser(self, prompt):
        return self._recipe.create_parser(prompt)

    @property
    def reasoning_efforts(self):
        return self._recipe.reasoning_efforts


@dataclass(frozen=True)
class SamplingDefaults:
    """The sampling settings a request takes when it gives none, from a checkpoint's
    ``generation_config.json`` (those it has). None: the server's own default (temperature 1.0,
    top_p 1.0, no top_k)."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def read(cls, checkpoint: str | Path) -> SamplingDefaults:
        path = Path(checkpoint) / "generation_config.json"
        if not path.is_file():
            return cls()
        config = json.loads(path.read_text())

        def number(name: str):
            value = config.get(name)
            return value if type(value) in (int, float) else None

        top_k = number("top_k")
        return cls(
            temperature=None if (t := number("temperature")) is None else float(t),
            top_p=None if (p := number("top_p")) is None else float(p),
            top_k=None if top_k is None else int(top_k),
        )


def model_capabilities(
    model: str | ModelSpec, recipe, checkpoint: str | Path | None = None
):
    """What ``GET /v1/models`` tells about a preset. Every chat recipe of MODEL_SPECS renders
    tools and parses reasoning, and the recipe gives its reasoning_effort levels. A preset
    without media serves text only. The sampling defaults come from the checkpoint's
    ``generation_config.json`` (``checkpoint``, default the preset's path)."""
    from resin.server import ModelCapabilities

    spec = get_model(model)
    defaults = SamplingDefaults.read(spec.path if checkpoint is None else checkpoint)
    return ModelCapabilities(
        function_calling=True,
        reasoning=True,
        thinking_levels=tuple(recipe.reasoning_efforts) or None,
        input_modalities=("text", "image") if spec.media else ("text",),
        output_modalities=("text",),
        default_temperature=defaults.temperature,
        default_top_p=defaults.top_p,
        default_top_k=defaults.top_k,
    )


def create_recipe(
    model: str | ModelSpec,
    checkpoint: str | Path | None = None,
    *,
    runner=None,
    tokenizer_checkpoint: str | Path | None = None,
):
    from resin.recipe import get_native_recipe

    spec = get_model(model)
    path = spec.path if checkpoint is None else Path(checkpoint)
    if spec.tokenizer is not None:
        path = (
            spec.tokenizer_path
            if tokenizer_checkpoint is None
            else Path(tokenizer_checkpoint)
        )
    name = spec.recipe or ("qwen3_8" if spec.family == "qwen3_5" else spec.family)
    if not spec.media:
        # No media processor: the bare tokenizer renders text, and media is refused.
        from tokenizers import Tokenizer

        return TextOnlyRecipe(
            get_native_recipe(
                name,
                Tokenizer.from_file(str(path / "tokenizer.json")),
                **dict(spec.recipe_options),
            )
        )
    if spec.family in {"deepseek_v41", "glm5_3_flash"}:
        from tokenizers import Tokenizer

        return get_native_recipe(
            spec.family, Tokenizer.from_file(str(path / "tokenizer.json"))
        )
    if spec.family == "gpt_oss":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        return get_native_recipe(spec.family, tokenizer, reasoning_effort="medium")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    options = (
        {"stop_token_ids": runner.config.eos_token_ids}
        if spec.family == "gemma4" and runner is not None
        else {}
    )
    return get_native_recipe(name, processor, **options)
