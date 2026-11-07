"""Pinned model configuration shared by serving and benchmark launchers.

Model-specific constructor differences live here; launchers share checkpoint handling and runtime
lifecycle code.
"""

from __future__ import annotations

import json
import os
import time
import warnings
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
            # The MTP weights are listed in the safetensor index, as `mtp.safetensors` in the
            # official release or inside a numbered shard in a compressed-tensors export, so the
            # index check below covers them. check_gaudi_drafts decides whether drafts can run.
            required_files=(
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
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
        max_batch_size=32,
        draft_tokens=2,
        # Qwen3.8-27B's vision tower runs on the card and answers the accuracy set 19/19
        # (docs/vision.md). `image_warmup` compiles and loads the image path at startup, so
        # the first image request does not pay a compile. The dense presets below have none.
        media=True,
        image_warmup=True,
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
            max_batch_size=16,
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
        max_batch_size=16,
        draft_tokens=0,
        prefill_chunk_size=2048,
        media=False,
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
    tower) through a fresh engine before it serves.

    The text warmup failing is fatal - something is broken and the server must not come up
    pretending otherwise. The image warmup failing is not: a bad checkpoint, a missing
    preprocessor config or a grid that will not compile should cost the image feature, not
    the whole server, so it warns and continues.
    """
    import numpy as np

    from resin import SamplingParams
    from resin.recipe import Image, Message, Role, Text

    # Exercise several decode steps, rather than ending after a one-word reply.
    content = [Text("Count from 1 to 100.")]

    def warm_one(image_input: bool) -> None:
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

    for image_input in (False, True) if spec.image_warmup else (False,):
        try:
            warm_one(image_input)
        except Exception as error:  # noqa: BLE001 - a warmup must not decide the server's fate
            if not image_input:
                raise
            warnings.warn(
                f"image warmup failed, so this server will serve text only: {error!r}",
                RuntimeWarning,
                stacklevel=2,
            )


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
):
    """Construct one runner with model-specific cache geometry and draft support."""
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
):
    """One Gaudi card per rank, selected by ``hpu:N`` (its accel index) or the first free one."""
    if spec.family == "glm5_3_flash":
        from resin_models_gaudi.glm5_3_flash.runner import (
            PREFILL_CHUNK_SIZES,
            Glm53FlashGaudiRunner,
        )

        # The runner keeps its compiled prefill widths. The Engine takes the smaller chunk.
        if prefill_chunk_size not in PREFILL_CHUNK_SIZES:
            raise ValueError(
                f"{spec.name} prefills in chunks of {', '.join(map(str, PREFILL_CHUNK_SIZES))} tokens"
            )
        # a speculative step near the end of a context writes 2K - 1 positions past it
        lookahead = 2 * draft_tokens - 1 if draft_tokens else 0
        return Glm53FlashGaudiRunner(
            str(checkpoint),
            groups=groups,
            card=_gaudi_card(spec, device),
            max_batch_size=max_batch_size,
            max_seq_len=context_length + lookahead,
            num_layers=num_layers,
            mtp=draft_tokens,
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
            card=_gaudi_card(spec, device),
            max_batch_size=max_batch_size,
            max_seq_len=context_length + draft_tokens,
            num_layers=num_layers,
            draft_tokens=draft_tokens,
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
    )

    if groups is not None and groups.topology.world_size != 1:
        raise ValueError(f"{spec.name} currently runs on one Gaudi card")
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
    # a speculative step near the end of a context writes 2K - 1 positions past it
    lookahead = 2 * draft_tokens - 1 if draft_tokens else 0
    return Qwen3_5GaudiRunner(
        checkpoint,
        card=card,
        max_batch_size=max_batch_size,
        max_seq_len=context_length + lookahead,
        memory_utilization=spec.gpu_memory_utilization
        if gpu_memory_utilization is None
        else gpu_memory_utilization,
        mtp=draft_tokens,
        # the preset decides: a text-only deployment does not compile the media embed graph
        media=spec.media,
    )


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
    else:
        from resin.models.mistral_large4.config import MistralLarge4Config
        from resin_models_gaudi.gguf.reader import GgufModel
        from resin_models_gaudi.mistral_large4.weights import RankSlices

        gguf = checkpoint / ML4_GGUF_QUANT
        cfg = MistralLarge4Config.from_gguf(
            GgufModel(str(gguf if gguf.is_dir() else checkpoint)).metadata
        )
        RankSlices.of(cfg, 0, ranks)


def check_gaudi_drafts(spec: ModelSpec, checkpoint: Path, draft_tokens: int) -> None:
    """Raise ValueError if MTP drafts are asked for and the checkpoint has no MTP weights."""
    if spec.family != "qwen3_5" or not draft_tokens:
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


def validate_gaudi_options(
    spec: ModelSpec,
    *,
    max_batch_size: int,
    prefill_chunk_size: int,
    draft_tokens: int,
) -> None:
    """Raise ValueError for a setting that the preset's runner rejects after the card opens."""
    if spec.family == "glm5_3_flash":
        from resin_models_gaudi.glm5_3_flash.runner import (
            MAX_BATCH,
            PREFILL_CHUNK_SIZES,
        )

        drafts, chunks = range(4), PREFILL_CHUNK_SIZES
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


class StillImageRecipe:
    """A media recipe whose runner serves one still image per prompt.

    Multiple images and video are refused at render time: a media request the
    runner cannot place has no per-request error channel - an exception inside
    ``runner.forward`` kills the inference actor and the server with it, where a
    render error is an ordinary bad request.
    """

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
        media = prompt.media
        if media is not None:
            import numpy as np

            grids = np.asarray(media.image_grid_thw)
            if grids.shape[0] != 1 or int(grids[0, 0]) != 1:
                raise ValueError(
                    "this preset serves one still image per prompt; "
                    "multiple images and video are not supported yet"
                )
        return prompt

    def create_parser(self, prompt):
        return self._recipe.create_parser(prompt)


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
    if spec.family == "qwen3_5":
        # The vision tower needs no AutoProcessor. `media::qwen::Config` on the Rust side is
        # exactly the checkpoint's own preprocessor_config.json - both structs are
        # #[serde(default)], so absent keys take their defaults and the processor_class keys
        # are ignored - so the file goes straight in as media_config. Keeping transformers
        # out of this path is what lets the gaudi group serve images at all.
        from tokenizers import Tokenizer

        preprocessor = path / "preprocessor_config.json"
        if not preprocessor.exists():
            raise FileNotFoundError(
                f"{preprocessor}: the vision tower takes its image processor config from here"
            )
        return StillImageRecipe(
            get_native_recipe(
                name,
                Tokenizer.from_file(str(path / "tokenizer.json")),
                media_config=json.loads(preprocessor.read_text()),
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
