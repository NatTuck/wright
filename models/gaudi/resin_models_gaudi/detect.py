"""Find the Gaudi preset for a checkpoint from its contents: config.json or GGUF metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from resin_models_gaudi.quant import BLOCK, weight_block

# GGUF file type of the Q4_K_XL weights that the Mistral-Large-4 kernels run.
_GGUF_Q4_K = 15
_GGUF_DIRECTORY = "UD-Q4_K_XL"

_ARCHITECTURES = {
    "qwen3_5": ("qwen3_5-gaudi2", "Qwen3.8-27B"),
    "glm5_next": ("glm5_3_flash-gaudi2", "GLM-5.3-Flash"),
    "deepseek_v41": ("deepseek_v41-gaudi2", "DeepSeek-V4.1-Flash"),
}
# (hidden size, layers) of the dense Qwen3.5 sizes.
_QWEN_SIZES = {
    (1024, 24): ("qwen3_5-0.8b-gaudi2", "Qwen3.5-0.8B"),
    (2048, 24): ("qwen3_5-2b-gaudi2", "Qwen3.5-2B"),
    (2560, 32): ("qwen3_5-4b-gaudi2", "Qwen3.5-4B"),
    (4096, 32): ("qwen3_5-9b-gaudi2", "Qwen3.5-9B"),
    (5120, 64): ("qwen3_5-gaudi2", "Qwen3.8-27B"),
}
_GLM_FOUR_CARDS = "glm5_3_flash-gaudi2-tp4"
_GLM_LONG_CONTEXT = "glm5_3_flash-gaudi2-256k"
_GLM_LONG_TOKENS = 262_144


class DetectionError(ValueError):
    """The path holds no checkpoint that Wright can run. The message says why."""


@dataclass(frozen=True)
class Detected:
    preset: str
    name: str
    checkpoint: Path
    tokenizer_included: bool = True


def _safetensors(path: Path, config: dict) -> Detected:
    kind = config.get("model_type") or config.get("text_config", {}).get("model_type")
    if kind not in _ARCHITECTURES:
        supported = "dense Qwen3.5 and Qwen3.8, GLM-5.3-Flash, DeepSeek-V4.1-Flash and Mistral-Large-4"
        raise DetectionError(
            f"{path}: model type {kind!r} has no Gaudi runner. Wright runs {supported}"
        )
    preset, name = _ARCHITECTURES[kind]
    text = config.get("text_config", config)
    if kind == "qwen3_5":
        shape = (text.get("hidden_size"), text.get("num_hidden_layers"))
        if shape not in _QWEN_SIZES:
            sizes = ", ".join(size for _, size in _QWEN_SIZES.values())
            raise DetectionError(
                f"{path}: no Qwen3.5 runner for hidden size {shape[0]} and {shape[1]} layers. "
                f"Wright runs {sizes}"
            )
        preset, name = _QWEN_SIZES[shape]
    quantization = config.get("quantization_config") or text.get("quantization_config")
    block = weight_block(quantization)
    method = (quantization or {}).get("quant_method")
    if block != [BLOCK, BLOCK]:
        detail = f"{method} with block size {block}" if block else (method or "no quantization")
        raise DetectionError(
            f"{path}: {name} needs FP8 weights in 128x128 blocks, and this checkpoint has "
            f"{detail}"
        )
    return Detected(preset, name, path)


def _gguf(path: Path) -> Detected:
    from resin_models_gaudi.gguf.reader import GgufModel

    metadata = GgufModel(str(path)).metadata
    architecture = metadata.get("general.architecture")
    if architecture != "mistral-large4":
        raise DetectionError(
            f"{path}: GGUF architecture {architecture!r} has no Gaudi runner. Wright runs "
            "Mistral-Large-4 in GGUF"
        )
    if metadata.get("general.file_type") != _GGUF_Q4_K:
        raise DetectionError(f"{path}: Mistral-Large-4 needs the UD-Q4_K_XL GGUF")
    checkpoint = path.parent if path.name == _GGUF_DIRECTORY else path
    return Detected(
        "mistral_large4-gaudi2",
        "Mistral-Large-4",
        checkpoint,
        tokenizer_included=(path / "tokenizer.json").exists()
        or (checkpoint / "tokenizer.json").exists(),
    )


def detect(path: Path) -> Detected:
    """``path`` is a checkpoint directory, a GGUF directory, or one file of a GGUF."""
    path = path.expanduser()
    if not path.exists():
        raise DetectionError(f"{path} does not exist")
    if path.is_file():
        if path.suffix != ".gguf":
            raise DetectionError(f"{path}: give a checkpoint directory or a .gguf file")
        return _gguf(path.parent)
    config = path / "config.json"
    if config.exists():
        return _safetensors(path, json.loads(config.read_text()))
    if any(path.glob("*.gguf")):
        return _gguf(path)
    if any((path / _GGUF_DIRECTORY).glob("*.gguf")):
        return _gguf(path / _GGUF_DIRECTORY)
    raise DetectionError(
        f"{path} has no config.json and no .gguf file, so Wright cannot find the model"
    )


def choose_preset(detected: Detected, ranks: int | None, context: int | None) -> str:
    """GLM-5.3-Flash has one preset per card count and context, with the same weights."""
    if detected.preset != "glm5_3_flash-gaudi2":
        return detected.preset
    if ranks == 4:
        return _GLM_FOUR_CARDS
    if context is not None and context >= _GLM_LONG_TOKENS:
        return _GLM_LONG_CONTEXT
    return detected.preset
