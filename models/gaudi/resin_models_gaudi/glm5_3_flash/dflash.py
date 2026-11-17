"""DFlash 2 speculative drafting for GLM-5.3-Flash on Gaudi (``incoai/GLM-5.3-Flash-DFlash2``).

The drafter is the same block-diffusion Qwen3 drafter as Qwen3.8-27B's (``common.dflash``), at
GLM's width (hidden 4096, 32 query heads over 8 KV heads of 128, SwiGLU 12288, about 1.2B
parameters), so this module only adapts the target:

- **Taps.** GLM's residual is four mHC streams. The drafter was trained on their mean after
  target layers 5, 14, 24, 33 and 42, so the tap graph sums the
  streams of the layer's completed output and the ``fc`` slices carry the 1 / 4. Layer t's
  output streams are materialized by the residual update at the start of layer t + 1's
  attention launch (``X_out``), so the tap runs right after it.
- **Tensor parallelism.** Every rank runs the whole drafter on the same inputs: the streams are
  identical on every rank after the collectives, and so are the samples that decide
  acceptance, so the drafts agree without a collective. The LM head over the block uses a
  full copy of the target's head on every rank (``dfl.head``, bf16); the embedding is already
  replicated.
- **Cache.** The drafter's K and V pages live in the DRAFT cache group (``cache.DRAFT_GROUP``,
  the MTP group's place) with WINDOW retention; the block's own K and V in scratch pages after
  the arena. A verify step injects all its columns; the next steps rewrite the rejected ones.

Every emitted token is the target's sample at its position (``model.verify_finish``), so the
output equals plain decoding whatever the drafter computes; its precision only moves the
acceptance rate.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from resin_models_gaudi.common.dflash import (
    Drafter,
    DrafterConfig,
    DraftRow,
    prepare_weights,
)
from resin_models_gaudi.common.layers import GEMM_NT, Dims, LayerGraph
from resin_models_gaudi.common.weights import weight_items
from resin_models_gaudi.glm5_3_flash.weights import NS
from resin_models_gaudi.loading.convert import kperm
from resin_models_gaudi.loading.source import Safetensors
from resin_models_gaudi.synapse.graph import Graph, Recipe
from resin_models_gaudi.synapse.runtime import BF16, F32, Device

if TYPE_CHECKING:
    from resin_models_gaudi.glm5_3_flash.model import Glm53Gaudi

HF_REPO = "incoai/GLM-5.3-Flash-DFlash2"
HEAD = "dfl.head"  # the full LM head, kperm'd bf16, on every rank
SMALL = 256


def tap_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """One tap of a target pass: the streams ``x`` (4H, T) f32 summed, cast to bf16, through the
    tap's slice ``fc`` (H, H) bf16 of the drafter's context projection (which carries the mean's
    1 / 4) -> ``feat`` (H, T) f32."""
    lg = LayerGraph(device, f"glm_dfl_tap_T{tokens}", dims, tokens)
    H, T = dims.hidden, tokens
    x = lg.input("x", (NS * H, T), F32)
    streams = [lg.tensor((H, T), F32) for _ in range(NS)]
    lg.node("split", [x], streams, struct.pack("I", 0))
    total = streams[0]
    for part in streams[1:]:
        acc = lg.tensor((H, T), F32)
        lg.node("add_fwd_f32", [total, part], [acc])
        total = acc
    xb = lg.tensor((H, T), BF16)
    lg.node("cast_f32_to_bf16", [total], [xb])
    lg.node(
        "gemm",
        [xb, lg.input("fc", (H, H), BF16)],
        [lg.input("feat", (H, T), F32)],
        GEMM_NT,
    )
    return lg.g


def drafter_items(
    path: str | Path, cfg: DrafterConfig, target: Safetensors
) -> Iterator[tuple[str, object, str]]:
    """The drafter's device tensors for the GLM ``WeightLoader``: ``prepare_weights`` with the
    stream mean folded into ``fc``, FP8 weights as ``inc8`` (already Gaudi FP8 with block
    scales), and the target's whole LM head for the block's logits."""
    yield from weight_items(prepare_weights(path, cfg, fc_scale=1.0 / NS))
    yield HEAD, kperm(target.bf16("lm_head.weight")), "bf16"


def drafter_weight_bytes(cfg: DrafterConfig, path: str | Path) -> int:
    """Bytes the drafter adds to a rank's load: its files and the head copy."""
    files = sum(p.stat().st_size for p in Path(path).glob("*.safetensors"))
    return files + 2 * cfg.vocab * cfg.hidden


def target_dims(hidden: int, vocab: int, eps: float) -> Dims:
    """The target as the drafter's graphs see it: its hidden size and vocabulary (the drafter's
    own widths come from ``DrafterConfig.dims``)."""
    return Dims(
        hidden=hidden,
        ffn=0,
        vocab=vocab,
        heads=0,
        kv_groups=0,
        head_dim=0,
        nk=0,
        nv=0,
        conv_dim=0,
        eps=eps,
    )


@dataclass(frozen=True, slots=True)
class Mode:
    """A target pass shape by its bucket: ``T{columns}``."""

    name: str
    columns: int


class GlmDrafterHost:
    """``common.dflash.DrafterHost`` over a ``Glm53Gaudi``: the drafter's recipes are compiled
    with the target's, its tables and block-step logits live in buffers of its own."""

    def __init__(
        self,
        model: Glm53Gaudi,
        cfg: DrafterConfig,
        *,
        buckets: Sequence[int],
        row_buckets: Sequence[int],
        columns: int,
        max_positions: int,
    ) -> None:
        self.model, self.cfg = model, cfg
        self.device = model.device
        self.dims = target_dims(model.cfg.hidden_size, model.cfg.vocab_size, cfg.eps)
        self.columns = columns
        self.decode_buckets = tuple(row_buckets)
        self.max_positions = max_positions
        self.modes = {f"T{T}": Mode(f"T{T}", T) for T in buckets}
        TD = cfg.block * self.decode_buckets[-1]
        self._logits = model.device.malloc(
            max(SMALL, -(-4 * self.dims.vocab * TD // SMALL) * SMALL)
        )

    @property
    def K(self) -> int:
        return self.model.K

    @property
    def arena(self) -> int:
        return self.model.arena.address

    def weight(self, name: str) -> int:
        if name == "E_gather":
            return self.model.w["emb"]
        if name == "E_head":
            return self.model.w[HEAD]
        return self.model.w[name]

    def buffer(self, name: str) -> int:
        if name == "one":
            return self.model.consts["one"]
        if name == "logits":
            return self._logits
        raise KeyError(name)

    def recipe_of(self, key: tuple) -> Recipe:
        return self.model.r[key]

    def upload(self, address: int, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        if array.nbytes <= 65536:
            self.device.h2d_nb(address, array)
        else:
            self.device.h2d(address, array)

    def tap_source(self, layer: int) -> int:
        """Layer ``layer``'s output streams: ``X_out`` of the next layer's attention launch."""
        return self.model.x[1]

    def tap_graph(self, tokens: int) -> Graph:
        return tap_graph(self.device, self.dims, tokens)


def row_buckets(
    decode_buckets: Sequence[int], draft_tokens: int, max_columns: int
) -> tuple[int, ...]:
    """The block-step row buckets: the decode buckets whose verify step fits ``max_columns``."""
    out = tuple(b for b in decode_buckets if b * (draft_tokens + 1) <= max_columns)
    if not out:
        raise ValueError(
            f"{draft_tokens} drafts: a verify step exceeds {max_columns} columns"
        )
    return out


__all__ = [
    "HEAD",
    "HF_REPO",
    "Drafter",
    "DrafterConfig",
    "DraftRow",
    "GlmDrafterHost",
    "Mode",
    "drafter_items",
    "drafter_weight_bytes",
    "row_buckets",
    "tap_graph",
    "target_dims",
]
