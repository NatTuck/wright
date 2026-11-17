"""Backend-neutral Qwen3.8-Flash-Next (``qwen4_exp``) text configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self, cast


@dataclass(frozen=True, slots=True)
class Qwen4ExpConfig:
    """Hyper-connection streams, gated delta net / gated attention with the QSA indexer, a 512-expert MoE in
    every layer, and a hashed n-gram embedding (PLE) at one layer."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    mrope_section: tuple[int, int, int] | None
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    hc_count: int
    hc_lowrank: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    ple_layer_ids: tuple[int, ...]
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ple_embed_dim: int
    ple_conv_kernel_size: int
    indexer_n_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    mtp_num_hidden_layers: int
    max_position_embeddings: int
    eos_token_id: int
    bos_token_id: int | None
    output_gate_type: str = "sigmoid"
    indexer_kv_heads: int = 1
    rope_scaling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        """Parse the multimodal root config (``qwen4_exp``) or its text sub-config."""
        if "text_config" in values:
            if values.get("model_type") != "qwen4_exp":
                raise ValueError(
                    f"expected a qwen4_exp checkpoint, got {values.get('model_type')!r}"
                )
            text = values["text_config"]
        else:
            text = values
        if not isinstance(text, dict) or text.get("model_type") != "qwen4_exp_text":
            raise ValueError(
                "Qwen3.8-Flash-Next checkpoint is missing its qwen4_exp_text config"
            )
        rope = text.get("rope_parameters") or {}
        if not isinstance(rope, dict):
            raise ValueError("rope_parameters must be an object")
        scaling = {
            k: v
            for k, v in rope.items()
            if k
            not in (
                "rope_theta",
                "partial_rotary_factor",
                "mrope_section",
                "mrope_interleaved",
            )
        }
        config = cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            layer_types=tuple(str(kind) for kind in text["layer_types"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            partial_rotary_factor=float(
                rope.get(
                    "partial_rotary_factor", text.get("partial_rotary_factor", 1.0)
                )
            ),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000.0))),
            mrope_section=(
                cast(tuple[int, int, int], tuple(int(s) for s in rope["mrope_section"]))
                if "mrope_section" in rope
                else None
            ),
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            hc_count=int(text["hc_count"]),
            hc_lowrank=int(text["hc_lowrank"]),
            num_experts=int(text["num_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(
                text["shared_expert_intermediate_size"]
            ),
            ple_layer_ids=tuple(int(i) for i in text.get("ple_layer_ids", ())),
            ngram_size=int(text.get("ngram_size", 3)),
            heads_per_ngram=int(text.get("heads_per_ngram", 8)),
            ngram_vocab_size_base=int(text.get("ngram_vocab_size_base", 0)),
            ple_embed_dim=int(text.get("ple_embed_dim", text["hidden_size"])),
            ple_conv_kernel_size=int(text.get("ple_conv_kernel_size", 4)),
            indexer_n_heads=int(text["indexer_n_heads"]),
            indexer_head_dim=int(text["indexer_head_dim"]),
            indexer_budget=int(text["indexer_budget"]),
            indexer_compress_ratio=int(text["indexer_compress_ratio"]),
            mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers", 0)),
            max_position_embeddings=int(text["max_position_embeddings"]),
            eos_token_id=int(text["eos_token_id"]),
            bos_token_id=None
            if text.get("bos_token_id") is None
            else int(text["bos_token_id"]),
            output_gate_type=str(text.get("output_gate_type", "sigmoid")),
            indexer_kv_heads=int(text.get("indexer_kv_heads", 1)),
            rope_scaling=scaling,
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "hc_count",
            "hc_lowrank",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "indexer_n_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
            "max_position_embeddings",
        )
        if invalid := [
            n
            for n in positive
            if type(getattr(self, n)) is not int or getattr(self, n) <= 0
        ]:
            raise ValueError(f"dimensions must be positive integers: {invalid}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("layer_types must be linear_attention or full_attention")
        if self.hc_count != 4:
            raise ValueError("the Gaudi kernels hold 4 hyper-connection streams")
        if self.output_gate_type != "sigmoid":
            raise ValueError("Qwen3.8-Flash-Next gates its delta net with a sigmoid")
        if self.linear_key_head_dim != 128 or self.linear_value_head_dim != 128:
            raise ValueError("the delta-net kernels take 128-wide heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("value heads must be a multiple of key heads")
        if self.head_dim != 256 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "the attention kernels take 256-wide heads in whole KV groups"
            )
        if self.indexer_compress_ratio != 4 or self.indexer_budget % 4:
            raise ValueError("the QSA indexer pools 4 keys per block")
        if self.indexer_head_dim != 128 or self.indexer_kv_heads != 1:
            raise ValueError(
                "the QSA indexer kernels take 128-wide heads over one key head"
            )
        if self.ngram_size != 3:
            raise ValueError("the n-gram hashing kernel takes bigrams and trigrams")
        if self.ple_embed_dim % ((self.ngram_size - 1) * self.heads_per_ngram):
            raise ValueError("ple_embed_dim must split over the n-gram heads")
        if self.hidden_size % 256:
            raise ValueError("hidden_size must be a multiple of 256")
        if self.rope_scaling.get("rope_type", "default") not in ("default", "yarn"):
            raise ValueError(
                f"unsupported rope scaling {self.rope_scaling.get('rope_type')!r}"
            )

    # ---- derived
    @property
    def full_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.layer_types) if k == "full_attention")

    @property
    def linear_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            i for i, k in enumerate(self.layer_types) if k == "linear_attention"
        )

    @property
    def ple_layers(self) -> tuple[int, ...]:
        """0-based decoder layers that take the n-gram embedding (``ple_layer_ids`` are 1-based)."""
        return tuple(i - 1 for i in self.ple_layer_ids)

    @property
    def ple_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ple_head_dim(self) -> int:
        return self.ple_embed_dim // self.ple_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_conv_dim(self) -> int:
        return (
            2 * self.linear_num_key_heads + self.linear_num_value_heads
        ) * self.linear_key_head_dim

    @property
    def heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def indexer_blocks(self) -> int:
        """blocks of 4 keys a query keeps"""
        return self.indexer_budget // self.indexer_compress_ratio

    def is_full(self, layer: int) -> bool:
        return (
            layer == self.num_hidden_layers
            or self.layer_types[layer] == "full_attention"
        )
