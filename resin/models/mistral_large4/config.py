"""Backend-neutral text configuration for Mistral-Large-4 (``mistral-large4`` GGUF, or Mistral's
consolidated ``params.json``), with the per-layer topology both formats describe:

- every 4th layer is NoPE (no rotary dims: queries are 128 nope dims, keys the latent only);
- KV sharing: a layer whose source is an earlier layer reads that layer's latent cache;
- DSA indexer sharing: a "shared" layer reuses the top-k selection of the nearest earlier
  layer that owns an indexer;
- layers before ``first_dense_layers`` have a dense SwiGLU; the rest are MoE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class MistralLarge4Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    """dense SwiGLU width (the first layers)"""
    rms_norm_eps: float
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    final_logit_softcap: float
    rope_theta: float
    yarn_factor: float
    yarn_original_max_position: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    num_shared_experts: int
    first_dense_layers: int
    routed_scale: float
    index_topk: int
    index_heads: int
    index_head_dim: int
    index_rope_dim: int
    index_norm_eps: float
    rope_layers: tuple[bool, ...]
    index_owners: tuple[bool, ...]
    kv_sources: tuple[int, ...]
    max_position_embeddings: int
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        n = self.num_hidden_layers
        if not (
            len(self.rope_layers) == len(self.index_owners) == len(self.kv_sources) == n
        ):
            raise ValueError("per-layer topology must cover every layer")
        for i, s in enumerate(self.kv_sources):
            if not 0 <= s <= i or self.kv_sources[s] != s:
                raise ValueError(f"layer {i} reads KV from {s}, which owns no cache")
            if self.rope_layers[s] != self.rope_layers[i]:
                raise ValueError(f"layer {i} and its KV source {s} differ in rope")
        if not self.index_owners[0]:
            raise ValueError("layer 0 must own a DSA indexer")
        if self.num_shared_experts != 1:
            raise ValueError("Mistral-Large-4 expects one shared expert")
        if self.qk_nope_head_dim + self.qk_rope_head_dim != self.v_head_dim:
            raise ValueError("query head width must equal the value head width")

    # ------------------------------------------------------------------ derived topology
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        """1 / sqrt(qk dim): YaRN's attention factor is 1 (no magnitude scaling)"""
        return self.qk_head_dim**-0.5

    def owns_kv(self, layer: int) -> bool:
        return self.kv_sources[layer] == layer

    def index_source(self, layer: int) -> int:
        """the nearest layer at or before ``layer`` that owns a DSA indexer"""
        return max(j for j in range(layer + 1) if self.index_owners[j])

    def is_moe(self, layer: int) -> bool:
        return layer >= self.first_dense_layers

    def key_dim(self, layer: int) -> int:
        """cached key width: the latent, plus the rotary key in rope layers"""
        return self.kv_lora_rank + (
            self.qk_rope_head_dim if self.rope_layers[layer] else 0
        )

    def nope_dim(self, layer: int) -> int:
        """query dims absorbed through k_b (all of them in NoPE layers)"""
        return self.qk_head_dim - (
            self.qk_rope_head_dim if self.rope_layers[layer] else 0
        )

    @property
    def kv_owner_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.owns_kv(i))

    @property
    def index_owner_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.index_owners[i])

    def truncated(self, num_layers: int) -> Self:
        """the first ``num_layers`` layers (bring-up and oracle tests)"""
        from dataclasses import replace

        n = num_layers
        if not 0 < n <= self.num_hidden_layers:
            raise ValueError(f"cannot truncate {self.num_hidden_layers} layers to {n}")
        return replace(
            self,
            num_hidden_layers=n,
            rope_layers=self.rope_layers[:n],
            index_owners=self.index_owners[:n],
            kv_sources=self.kv_sources[:n],
        )

    # ------------------------------------------------------------------ parsers
    @classmethod
    def from_gguf(cls, metadata: dict[str, Any]) -> Self:
        arch = metadata.get("general.architecture")
        if arch != "mistral-large4":
            raise ValueError(f"expected a mistral-large4 GGUF, got {arch!r}")

        def g(key: str) -> Any:
            try:
                return metadata[f"mistral-large4.{key}"]
            except KeyError:
                raise ValueError(f"GGUF metadata lacks mistral-large4.{key}") from None

        if g("expert_group_count") != 1 or not g("expert_weights_norm"):
            raise ValueError("expected one expert group with normalized top-k weights")
        if g("expert_gating_func") != 1:
            raise ValueError("expected softmax expert gating")
        if g("rope.scaling.type") != "yarn":
            raise ValueError("expected YaRN rope scaling")
        rope = g("rope.dimension_count")
        qk = g("attention.key_length_mla")
        return cls(
            vocab_size=g("vocab_size"),
            hidden_size=g("embedding_length"),
            num_hidden_layers=g("block_count"),
            num_attention_heads=g("attention.head_count"),
            intermediate_size=g("feed_forward_length"),
            rms_norm_eps=g("attention.layer_norm_rms_epsilon"),
            kv_lora_rank=g("attention.kv_lora_rank"),
            qk_rope_head_dim=rope,
            qk_nope_head_dim=qk - rope,
            v_head_dim=g("attention.value_length_mla"),
            final_logit_softcap=g("final_logit_softcapping"),
            rope_theta=g("rope.freq_base"),
            yarn_factor=g("rope.scaling.factor"),
            yarn_original_max_position=g("rope.scaling.original_context_length"),
            yarn_beta_fast=g("rope.scaling.yarn_beta_fast"),
            yarn_beta_slow=g("rope.scaling.yarn_beta_slow"),
            num_experts=g("expert_count"),
            num_experts_per_tok=g("expert_used_count"),
            moe_intermediate_size=g("expert_feed_forward_length"),
            num_shared_experts=g("expert_shared_count"),
            first_dense_layers=g("leading_dense_block_count"),
            routed_scale=g("expert_weights_scale"),
            index_topk=g("attention.indexer.top_k"),
            index_heads=g("attention.indexer.head_count"),
            index_head_dim=g("attention.indexer.key_length"),
            index_rope_dim=rope,
            index_norm_eps=g("attention.layer_norm_epsilon"),
            rope_layers=tuple(bool(v) for v in g("attention.rope_pattern")),
            index_owners=tuple(bool(v) for v in g("attention.indexer.types")),
            kv_sources=tuple(int(v) for v in g("attention.kv_source_map")),
            max_position_embeddings=g("context_length"),
            bos_token_id=int(metadata.get("tokenizer.ggml.bos_token_id", 1)),
            eos_token_id=int(metadata.get("tokenizer.ggml.eos_token_id", 2)),
        )

    @classmethod
    def from_params(cls, p: dict[str, Any]) -> Self:
        """Mistral's consolidated ``params.json``"""
        n = p["n_layers"]
        if p.get("q_lora_rank") is not None:
            raise ValueError("Mistral-Large-4 has no query LoRA")
        y, m, d = p["yarn"], p["moe"], p["dsa"]
        if y.get("apply_scale"):
            raise ValueError("YaRN magnitude scaling is not supported")
        if m["num_expert_groups"] != 1:
            raise ValueError("expected one expert group")
        pattern = p["nope"].split(",")
        sharing = (
            d["sharing"].split(",")
            if isinstance(d["sharing"], str)
            else list(d["sharing"])
        )
        sources = [int(s) for s in p["kv_sharing"]["source_layers_str"].split(",")]
        return cls(
            vocab_size=p["vocab_size"],
            hidden_size=p["dim"],
            num_hidden_layers=n,
            num_attention_heads=p["n_heads"],
            intermediate_size=p["hidden_dim"],
            rms_norm_eps=p["norm_eps"],
            kv_lora_rank=p["kv_lora_rank"],
            qk_rope_head_dim=p["qk_rope_head_dim"],
            qk_nope_head_dim=p["qk_nope_head_dim"],
            v_head_dim=p["v_head_dim"],
            final_logit_softcap=p["softmax_tanh_gating"],
            rope_theta=p["rope_theta"],
            yarn_factor=y["factor"],
            yarn_original_max_position=y["original_max_position_embeddings"],
            yarn_beta_fast=y["beta"],
            yarn_beta_slow=y["alpha"],
            num_experts=m["num_experts"],
            num_experts_per_tok=m["num_experts_per_tok"],
            moe_intermediate_size=m["expert_hidden_dim"],
            num_shared_experts=m["num_shared_experts"],
            first_dense_layers=m["first_k_dense_replace"],
            routed_scale=m["routed_scale"],
            index_topk=d["topk"],
            index_heads=d["n_heads"],
            index_head_dim=d["head_dim"],
            index_rope_dim=d["rope_head_dim"],
            index_norm_eps=d["norm_eps"],
            rope_layers=tuple(pattern[i % len(pattern)] == "rope" for i in range(n)),
            index_owners=tuple(sharing[i] == "full" for i in range(n)),
            kv_sources=tuple(sources[:n]),
            max_position_embeddings=p["max_position_embeddings"],
        )
