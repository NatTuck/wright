"""DFlash 2 speculative drafting for Qwen3.8-27B on Gaudi (``incoai/Qwen3.8-27B-DFlash2``).

The drafter is five Qwen3 decoder layers at the target's width (hidden 5120, 32 query heads, 8 KV
heads of 128, full NeoX RoPE, SwiGLU 17408), about 1.9B parameters, run in FP8 like the target.
It shares the target's token embedding and LM head. A decode step verifies [anchor, 7 drafts] of
every row in one target pass (``Qwen35Gaudi.verify_launch``), and the drafter then makes the next
step's drafts from the step's results:

- **Taps.** During every target pass, the residual outputs of target layers 5, 19, 33, 47 and 61
  (``cfg.taps``) go through their slice of the drafter's ``fc`` (``tap_graph``, five launches
  inside the pass) into ``feat[k]``.
- **Injection.** After the pass, ``inject_graph`` sums the five parts, applies ``hidden_norm`` and
  stores, for every drafter layer, k_norm(k_proj) with RoPE and v_proj of every column into the
  drafter's cache (the DRAFT cache group: 128-token blocks of 5 layers, retained for the last
  ``window`` = 2048 positions plus the step's writes). A verify step injects all its columns at
  once; the rejected ones are rewritten by the next block and the next step before anything reads
  them. The target's own cache is unchanged.
- **The block.** ``block`` runs [anchor, mask x 7] at positions p .. p + 7 through the five layers:
  the target's embeddings, each sublayer wrapped in a two-tap grouped dynamic convolution along the
  block (``dfl_conv``), attention over the block (bidirectional) and the cache entries within 2048
  positions before each query (``attn_slot_pd``: slot field 6 = window start, 7 = window, all keys
  below ``klen`` visible). The block's own K and V live in scratch pages (no cache lease), so a
  step's cache writes stay within its verified positions. Finally ``norm``, the LM head over the
  block, the 16 largest logits per mask position and the greedy candidate walk (``dfl_select``)
  write the next step's tokens [anchor, drafts] where the next verify pass reads them.

Layout: the 128-wide KV heads are paired, so the target's 256-wide page kernels (``kv_store``,
``kv_gather``, ``attn_slot_*``) serve the drafter unchanged. Pair pg = KV heads (2 pg, 2 pg + 1)
holds [k_2pg | k_2pg+1] in one 512-byte row; head hd is pseudo-head hd % 8 of pair hd / 8, its
query zero-extended to the half it reads. A pair's attention over 256 dims is then exactly each
head's attention over its 128 (``dfl_prep``, ``dfl_oq``).

Numerics follow ``models/gaudi/tools/qwen_dflash2_ref.py`` (the CPU reference) in FP8: the drafter
only proposes, and the target's samples decide, so the output equals plain decoding whatever the
drafter computes. Only the acceptance rate depends on its precision.

DSpine (``tencent/Qwen3.8-27B-DSpine-b16``, ``DrafterConfig.kind == "dspine"``) runs in the same
runtime: the same taps (layers 1, 16, 31, 46, 61), injection and cache, a block of 16 (15 drafts),
with these differences (``models/gaudi/tools/qwen_dspine_ref.py`` is its CPU reference):

- no convolutions: each sublayer is a plain FP8 GEMM of its normed input;
- causal attention over the block (``attn_slot_pc``), the cache within 8192 positions;
- the code bridge after every layer (``post_graph``): ``dsp_read`` (the residual, its RMS ``ws`` and
  the unit-RMS read operand), the layer's read GEMM (bf16), ``dsp_codes`` (each column's code, the
  anchor's from the codebook), the packed GEMM [gate | predecessor_gate | prev], ``dsp_msg`` (the
  message from the column before) and the layer's write GEMM, added at ``ws`` by ``dfl_norm`` with
  the next layer's input norm;
- the serial refinement (``head_graph``): the LM head, the 16 largest logits per mask column, and
  ``dsp_select``, a greedy walk that redoes each column's last write from the exact code of the
  token chosen before it, through the tables ``tg`` = (LM head * norm) . write_last and ``tsv`` =
  [predecessor_gate | prev] . code (built at load from the target's LM head);
- a verify step takes 4, 8 or 16 columns per row (``Qwen35Gaudi.verify_widths``, chosen from
  the last step's acceptance by ``model.choose_width``), and the block step runs only the
  first ``width`` positions of the block (``Drafter.block_cfg``: none of them sees a later one).

Every drafter's head takes the k largest logits in two kernels (``dfl_tk1``: the k chunks of 16
with the largest maxima; ``dfl_tk2``: the k largest of their logits).
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from resin_models_gaudi.common.layers import (
    ARENA_BLOCK_ROWS,
    GEMM_NT,
    SLOT_KEYS,
    Dims,
    LayerGraph,
    attn_slots_graph,
)
from resin_models_gaudi.common.weights import DenseWeight, Fp8Weight
from resin_models_gaudi.loading.convert import (
    BLOCK,
    E4M3,
    FP8_MAX,
    bf16_rne,
    bf16_to_f32,
    e4m3_encode,
    kperm,
    sigma_inv,
    tperm,
)
from resin_models_gaudi.loading.source import Safetensors
from resin_models_gaudi.synapse.graph import Graph, LaunchPlan, Recipe
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, I32, Device

HF_REPO = "incoai/Qwen3.8-27B-DFlash2"
HF_REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"
PAGE_TOKENS = 128
HEAD_DIM = 128  # what the kernels implement
PAIR_HEADS = 8  # query heads per KV pair (two KV heads of four)
KV_HEADS = 8
CONV_GROUP = 16
SELECTOR_RANK = 256
CANDIDATES_MAX = 16
BRIDGE_RANK = 512  # what ``dsp_select`` implements
DFLASH2, DSPINE = "dflash2", "dspine"
SMALL = 256
DYNAMIC = ("slots", "k_pages", "v_pages")  # bound per slot launch


# ------------------------------------------------------------------ configuration
class TargetConfig(Protocol):
    """What a drafter's configuration is checked against: its target's shape."""

    @property
    def hidden_size(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def num_hidden_layers(self) -> int: ...


@dataclass(frozen=True, slots=True)
class DrafterConfig:
    """``config.json`` of a DFlash 2 Qwen3 drafter, checked against what this port implements."""

    hidden: int
    ffn: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    eps: float
    rope_theta: float
    window: int
    block: int
    conv_kernel: int
    conv_group: int
    mask_token: int
    rank: int
    top_k: int
    taps: tuple[int, ...]
    num_target_layers: int
    kind: str = DFLASH2
    """``dflash2`` (convolutions and a candidate selector, bidirectional block) or ``dspine`` (code
    bridge and serial refinement, causal block); ``rank`` and ``top_k`` are then the bridge's rank
    and the refinement's candidates."""
    code_size: int = 0
    bridge_eps: float = 0.0

    @staticmethod
    def load(path: str | Path, target: TargetConfig) -> DrafterConfig:
        c = json.loads((Path(path) / "config.json").read_text())
        d = c.get("dflash_config", {})
        if d.get("selected_prefix_code_bridge"):
            return DrafterConfig._load_dspine(path, c, target)
        if (
            c.get("architectures") != ["DFlash2DraftModel"]
            or c.get("model_type") != "qwen3"
        ):
            raise ValueError(f"{path} is not a DFlash 2 Qwen3 drafter")
        if c.get("is_causal", False) or set(c.get("layer_types", [])) != {
            "sliding_attention"
        }:
            raise ValueError(
                "the drafter must be non-causal with sliding-window layers"
            )
        if c["rope_parameters"].get("rope_type", "default") != "default":
            raise ValueError("unsupported RoPE scaling in the drafter")
        cfg = DrafterConfig(
            hidden=int(c["hidden_size"]),
            ffn=int(c["intermediate_size"]),
            vocab=int(c["vocab_size"]),
            heads=int(c["num_attention_heads"]),
            kv_heads=int(c["num_key_value_heads"]),
            head_dim=int(c["head_dim"]),
            layers=int(c["num_hidden_layers"]),
            eps=float(c["rms_norm_eps"]),
            rope_theta=float(c["rope_parameters"]["rope_theta"]),
            window=int(c["sliding_window"]),
            block=int(d["block_size"]),
            conv_kernel=int(d["conv_kernel_size"]),
            conv_group=int(d["conv_group_size"]),
            mask_token=int(d["mask_token_id"]),
            rank=int(d["selector_rank"]),
            top_k=int(d["selector_top_k"]),
            taps=tuple(int(i) for i in d["target_layer_ids"]),
            num_target_layers=int(c["num_target_layers"]),
        )
        checks = {
            "two-tap convolutions of 16 channels": cfg.conv_kernel == 2
            and cfg.conv_group == CONV_GROUP,
            "selector rank 256": cfg.rank == SELECTOR_RANK,
            f"at most {CANDIDATES_MAX} candidates": 1 <= cfg.top_k <= CANDIDATES_MAX,
        }
        return cfg._checked(path, target, checks)

    def _checked(
        self, path: str | Path, target: TargetConfig, checks: dict[str, bool]
    ) -> DrafterConfig:
        checks = {
            "hidden size": self.hidden == target.hidden_size,
            "vocabulary": self.vocab == target.vocab_size,
            "target layer count": self.num_target_layers == target.num_hidden_layers,
            "tap layers": all(0 <= t < target.num_hidden_layers for t in self.taps),
            "head_dim 128": self.head_dim == HEAD_DIM,
            "32 query heads over 8 KV heads": self.heads == 32
            and self.kv_heads == KV_HEADS,
            "a block of at most 16": 2 <= self.block <= 16,
            "mask token": 0 <= self.mask_token < self.vocab,
            "hidden a multiple of 1024": self.hidden % 1024 == 0,
            "window a multiple of the page": self.window % PAGE_TOKENS == 0,
            **checks,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"{path}: unsupported drafter ({', '.join(failed)})")
        return self

    @staticmethod
    def _load_dspine(path: str | Path, c: dict, target: TargetConfig) -> DrafterConfig:
        """A DSpine drafter (``DFlashDraftModel`` with a code bridge): the serial last-write
        refinement over independent, terminal-free writes after every layer."""
        d = c["dflash_config"]
        bridge = d["selected_prefix_code_bridge"]
        refine = d.get("selected_prefix_last_write_refine") or {}
        second = c.get("code_bridge_second_pass") or {}
        layers = int(c["num_hidden_layers"])
        if (
            c.get("architectures") != ["DFlashDraftModel"]
            or c.get("model_type") != "qwen3"
        ):
            raise ValueError(f"{path} is not a DSpine Qwen3 drafter")
        if not (c.get("is_causal") and d.get("draft_block_causal")):
            raise ValueError("a DSpine drafter's block must be causal")
        if set(c.get("layer_types", [])) != {"sliding_attention"}:
            raise ValueError("the drafter must have sliding-window layers")
        if c["rope_parameters"].get("rope_type", "default") != "default":
            raise ValueError("unsupported RoPE scaling in the drafter")
        unsupported = [
            key
            for key in (
                "conv_kernel_size",
                "selector_rank",
                "selector_top_k",
                "selected_prefix_terminal",
                "selected_prefix_terminal_residual_rank",
                "selected_prefix_feature_next_token_rows",
                "selected_prefix_feature_reliability",
            )
            if d.get(key)
        ]
        if unsupported:
            raise ValueError(f"{path}: unsupported DSpine settings {unsupported}")
        top_k = int(refine.get("head_topk", 0))
        cfg = DrafterConfig(
            hidden=int(c["hidden_size"]),
            ffn=int(c["intermediate_size"]),
            vocab=int(c["vocab_size"]),
            heads=int(c["num_attention_heads"]),
            kv_heads=int(c["num_key_value_heads"]),
            head_dim=int(c["head_dim"]),
            layers=layers,
            eps=float(c["rms_norm_eps"]),
            rope_theta=float(c["rope_parameters"]["rope_theta"]),
            window=int(c["sliding_window"]),
            block=int(c["block_size"]),
            conv_kernel=0,
            conv_group=0,
            mask_token=int(d["mask_token_id"]),
            rank=int(bridge["rank"]),
            top_k=top_k,
            taps=tuple(int(i) for i in d["target_layer_ids"]),
            num_target_layers=int(c["num_target_layers"]),
            kind=DSPINE,
            code_size=int(bridge["code_size"]),
            bridge_eps=float(bridge["eps"]),
        )
        return cfg._checked(
            path,
            target,
            {
                "independent writes": d.get("selected_prefix_feature_write_mode")
                == "independent"
                and d.get("selected_prefix_feature_writes") is True,
                "a write after every layer": sorted(
                    int(i) for i in d.get("selected_prefix_feature_layers", [])
                )
                == list(range(1, layers + 1)),
                "the exact-sender last-write refinement": refine.get("kind")
                == "exact_sender"
                and int(refine.get("version", 1)) == 1,
                "the serial table redoes the last write": not second
                or (
                    second.get("mode") == "serial_table"
                    and int(second.get("first_exact", -1)) == layers - 1
                    and int(second.get("head_topk", top_k)) == top_k
                ),
                f"bridge rank {BRIDGE_RANK}": cfg.rank == BRIDGE_RANK,
                "codes a multiple of 128": cfg.code_size > 0
                and cfg.code_size % 128 == 0,
                f"at most {CANDIDATES_MAX} candidates": 1 <= top_k <= CANDIDATES_MAX,
                "a block of 16 or less": cfg.block <= 16,
            },
        )

    @property
    def causal(self) -> bool:
        """the block attends causally (DSpine) rather than bidirectionally (DFlash 2)"""
        return self.kind == DSPINE

    @property
    def pairs(self) -> int:
        return self.kv_heads // 2

    @property
    def drafts(self) -> int:
        """draft tokens a block makes (the anchor takes the first column); a step may verify
        fewer (``Qwen35Gaudi.K``)"""
        return self.block - 1

    @property
    def groups(self) -> int:
        return self.hidden // self.conv_group if self.conv_group else 0

    @property
    def kp_rows(self) -> int:
        """rows of a convolution's coefficient projection: 2 sides x taps x groups"""
        return 2 * self.conv_kernel * self.groups

    @property
    def qkv_rows(self) -> int:
        return (self.heads + 2 * self.kv_heads) * self.head_dim

    @property
    def kv_rows(self) -> int:
        return 2 * self.kv_heads * self.head_dim

    @property
    def layer_rows(self) -> int:
        """arena rows of one drafter layer's pages in a cache block: K and V pages per pair"""
        return 2 * self.pairs * PAGE_TOKENS

    @property
    def block_rows(self) -> int:
        """arena rows of one DRAFT cache block (128 tokens of every drafter layer)"""
        return self.layers * self.layer_rows

    def dims(self, target: Dims) -> Dims:
        """The drafter's dimensions for the target's graph builders: KV pairs as 256-wide groups."""
        return replace(
            target,
            ffn=self.ffn,
            heads=self.pairs * PAIR_HEADS,
            kv_groups=self.pairs,
            head_dim=2 * self.head_dim,
            eps=self.eps,
        )


def drafter_draft_tokens(path: str | Path) -> int:
    """The drafts a block of the drafter at ``path`` makes (its block size less the anchor)."""
    config = json.loads((Path(path) / "config.json").read_text())
    try:
        block = int(
            (config.get("dflash_config") or {}).get("block_size")
            or config["block_size"]
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{path} has no DFlash block size in its config.json"
        ) from None
    if block < 2:
        raise ValueError(f"{path}: a block of {block} makes no drafts")
    return block - 1


# ------------------------------------------------------------------ weights
FP8_SUFFIX, SCALE_SUFFIX = "_quant", "_scale"


class DrafterTensors:
    """A drafter checkpoint's tensors, stored bf16 (the release) or fp8_inc: every tensor of
    more than 16 elements as ``<name>_quant`` e4m3 bytes (|x| <= 240, the Gaudi encoding) with
    one f32 ``<name>_scale`` (x = code * scale), smaller ones as f32 or i64. The format is
    detected from the tensor names."""

    def __init__(self, path: str | Path) -> None:
        self.files = Safetensors(path, mode="mmap")
        self.fp8 = any(
            k.endswith(FP8_SUFFIX)
            and k[: -len(FP8_SUFFIX)] + SCALE_SUFFIX in self.files
            for k in self.files.meta
        )

    def __contains__(self, name: str) -> bool:
        return name in self.files or name + FP8_SUFFIX in self.files

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        return self.files.raw(name)

    def codes(self, name: str) -> tuple[np.ndarray, float] | None:
        """(e4m3 bytes, scale) of an fp8_inc tensor, else None."""
        if name + FP8_SUFFIX not in self.files:
            return None
        codes, dtype = self.files.raw(name + FP8_SUFFIX)
        scale, sdtype = self.files.raw(name + SCALE_SUFFIX)
        if dtype not in ("U8", "F8_E4M3") or sdtype != "F32":
            raise ValueError(f"{name}: fp8_inc as {dtype} with a {sdtype} scale")
        return codes, float(np.asarray(scale).reshape(-1)[0])

    def f32(self, name: str) -> np.ndarray:
        quantized = self.codes(name)
        if quantized is not None:
            codes, scale = quantized
            return E4M3[codes] * np.float32(scale)
        array, dtype = self.files.raw(name)
        if dtype == "BF16":
            return bf16_to_f32(array)
        if dtype == "F32":
            return np.array(array)
        raise ValueError(f"{name}: expected bf16, f32 or fp8_inc, got {dtype}")

    def bf16(self, name: str) -> np.ndarray:
        """bf16 bits: as stored, or rounded from fp8_inc / f32."""
        if name + FP8_SUFFIX not in self.files:
            array, dtype = self.files.raw(name)
            if dtype == "BF16":
                return array
        return bf16_rne(self.f32(name))

    def fp8_gemm(self, names: Sequence[str], permute_k: bool = False) -> Fp8Weight:
        """The rows of ``names`` (each (M_i, K)) stacked as one block-FP8 GEMM weight: an
        fp8_inc checkpoint's bytes as stored, each part's scale on its row blocks (``kperm``
        on K first with ``permute_k``); a bf16 one through ``block_fp8``."""
        parts = [self.codes(n) for n in names]
        stored = [q for q in parts if q is not None]
        if len(stored) < len(parts) or any(q[0].shape[0] % BLOCK for q in stored):
            w = np.concatenate([self.f32(n) for n in names])
            return block_fp8(kperm(w) if permute_k else w)
        codes = np.concatenate([q[0] for q in stored])
        if permute_k:
            codes = kperm(codes)
        K = codes.shape[1]
        if K % 256:
            raise ValueError(f"FP8 weight {codes.shape}: K must be a multiple of 256")
        scales = np.concatenate(
            [
                np.full((q[0].shape[0] // BLOCK, K // BLOCK), q[1], np.float32)
                for q in stored
            ]
        )
        return Fp8Weight(tperm(np.ascontiguousarray(codes)), scales)


def drafter_is_fp8(path: str | Path) -> bool:
    """The drafter checkpoint at ``path`` is stored fp8_inc (else bf16)."""
    return DrafterTensors(path).fp8


def block_fp8(w: np.ndarray) -> Fp8Weight:
    """f32 (M, K) -> block-FP8 with 128 x 128 block scales (amax / 240), K in tau order, as the
    checkpoint loader expects (the device requantizes to per-row FP8)."""
    M, K = w.shape
    if M % 128 or K % 256:
        raise ValueError(
            f"FP8 weight {w.shape}: rows must be a multiple of 128, K of 256"
        )
    blocks = w.reshape(M // 128, 128, K // 128, 128)
    amax = np.abs(blocks).max(axis=(1, 3))
    scale = np.where(amax > 0, amax / FP8_MAX, 1.0).astype(np.float32)
    scaled = np.clip(blocks / scale[:, None, :, None], -FP8_MAX, FP8_MAX)
    q = e4m3_encode(scaled.reshape(M, K).astype(np.float32))
    return Fp8Weight(tperm(q), np.ascontiguousarray(scale))


def scale_bf16(bits: np.ndarray, scale: float) -> np.ndarray:
    """bf16 bits times a power of two (exact where the result stays normal)."""
    return bf16_rne(bf16_to_f32(bits) * np.float32(scale))


def prepare_weights(
    path: str | Path,
    cfg: DrafterConfig,
    fc_scale: float = 1.0,
    head: Callable[[], np.ndarray] | None = None,
) -> Iterator[tuple[str, Fp8Weight | DenseWeight]]:
    """The drafter's device tensors, one at a time: ``dfl.*`` shared, ``d{l}.*`` per layer. GEMM
    weights as block-FP8 (``block_fp8``); ``fc`` as five bf16 slices (natural K order: the tap
    graph casts the f32 residual), scaled by ``fc_scale`` (a power of two, exact in bf16: a
    target whose tap is the mean of its residual streams folds the 1 / streams in here); the
    selector's projection bf16 in sigma K order (its operand is the drafter's sigma-packed final
    hidden state); the codebooks bf16 as (vocab, rank) tables. A DSpine drafter needs ``head``,
    the target's LM head (vocab, hidden) bf16 bits, for its refinement tables (``dspine_tables``)."""
    files = DrafterTensors(path)
    H = cfg.hidden
    if fc_scale <= 0 or not math.log2(fc_scale).is_integer():
        raise ValueError(f"fc_scale must be a power of two, not {fc_scale}")
    f32, bf16 = files.f32, files.bf16

    def dense(x: np.ndarray) -> DenseWeight:
        return DenseWeight(np.ascontiguousarray(x))

    yield "dfl.hidden_norm", dense(f32("hidden_norm.weight"))
    fc = bf16("fc.weight")  # (H, taps H)
    if fc.shape != (H, len(cfg.taps) * H):
        raise ValueError(
            f"fc.weight {fc.shape} does not fit {len(cfg.taps)} taps of {H}"
        )
    if fc_scale != 1.0:
        fc = scale_bf16(fc, fc_scale)
    for k in range(len(cfg.taps)):
        yield f"dfl.fc{k}", dense(fc[:, k * H : (k + 1) * H])
    if cfg.kind == DSPINE:
        if head is None:
            raise ValueError("a DSpine drafter needs the target's LM head")
        yield from _dspine_weights(files, cfg, head, dense)
        return
    for layer in range(cfg.layers):
        p = f"layers.{layer}."
        a = p + "self_attn."
        yield f"d{layer}.attn_norm", dense(f32(p + "input_layernorm.weight"))
        yield f"d{layer}.post_norm", dense(f32(p + "post_attention_layernorm.weight"))
        yield (
            f"d{layer}.qk_norm",
            dense(np.concatenate([f32(a + "q_norm.weight"), f32(a + "k_norm.weight")])),
        )
        k, v = f32(a + "k_proj.weight"), f32(a + "v_proj.weight")
        yield (
            f"d{layer}.w_qkv",
            block_fp8(np.concatenate([f32(a + "q_proj.weight"), k, v])),
        )
        yield f"d{layer}.w_kv", block_fp8(np.concatenate([k, v]))
        del k, v
        yield f"d{layer}.w_o", block_fp8(kperm(f32(a + "o_proj.weight")))
        yield (
            f"d{layer}.w_gu",
            block_fp8(
                np.concatenate(
                    [f32(p + "mlp.gate_proj.weight"), f32(p + "mlp.up_proj.weight")]
                )
            ),
        )
        yield f"d{layer}.w_down", block_fp8(f32(p + "mlp.down_proj.weight"))
        for side, conv in (("a", "attention_conv"), ("m", "mlp_conv")):
            yield (
                f"d{layer}.kp_{side}",
                block_fp8(f32(f"{p}{conv}.kernel_projection.weight")),
            )
            base = f32(f"{p}{conv}.base_kernel")  # (2 sides, taps, H)
            if base.shape != (2, cfg.conv_kernel, H):
                raise ValueError(f"{conv}.base_kernel {base.shape}")
            yield f"d{layer}.base_{side}", dense(base.reshape(2 * cfg.conv_kernel, H))
    yield "dfl.norm", dense(f32("norm.weight"))
    yield "dfl.hproj", dense(kperm(bf16("candidate_selector.hidden_projection.weight")))
    for name, source in (
        ("pred", "predecessor_codebook"),
        ("succ", "successor_codebook"),
    ):
        table = bf16(f"candidate_selector.{source}")
        if table.shape != (cfg.vocab, cfg.rank):
            raise ValueError(f"{source} {table.shape}")
        yield f"dfl.{name}", dense(table)


def _attention_mlp(
    files: DrafterTensors, p: str, layer: int, dense
) -> Iterator[tuple[str, Fp8Weight | DenseWeight]]:
    """A Qwen3 drafter layer's norms and FP8 GEMMs (``d{layer}.*``)."""
    f32 = files.f32
    a = p + "self_attn."
    yield f"d{layer}.attn_norm", dense(f32(p + "input_layernorm.weight"))
    yield f"d{layer}.post_norm", dense(f32(p + "post_attention_layernorm.weight"))
    yield (
        f"d{layer}.qk_norm",
        dense(np.concatenate([f32(a + "q_norm.weight"), f32(a + "k_norm.weight")])),
    )
    q, k, v = (a + f"{x}_proj.weight" for x in "qkv")
    yield f"d{layer}.w_qkv", files.fp8_gemm([q, k, v])
    yield f"d{layer}.w_kv", files.fp8_gemm([k, v])
    yield f"d{layer}.w_o", files.fp8_gemm([a + "o_proj.weight"], permute_k=True)
    yield (
        f"d{layer}.w_gu",
        files.fp8_gemm([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
    )
    yield f"d{layer}.w_down", files.fp8_gemm([p + "mlp.down_proj.weight"])


def dspine_tables(
    head: np.ndarray,
    norm: np.ndarray,
    write_last: np.ndarray,
    sender: np.ndarray,
    codebook: np.ndarray,
    chunk: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    """The serial refinement's tables as bf16 bits, rows by token (natural order):
    ``tg`` (V, R) = bf16(head[v] * norm) . write_last and ``tsv`` (V, 2R) = [predecessor_gate |
    prev] . bf16(sqrt(C) * codebook[v]). head (V, H) and codebook (V, C) bf16 bits, norm (H,) f32,
    write_last (H, R) f32 (the last layer's write), sender (2R, C) f32."""
    V, C = codebook.shape
    R = write_last.shape[1]
    tg = np.empty((V, R), np.uint16)
    tsv = np.empty((V, 2 * R), np.uint16)
    scale = np.float32(math.sqrt(C))
    wb = bf16_to_f32(bf16_rne(write_last))
    for s in range(0, V, chunk):
        e = min(V, s + chunk)
        h = bf16_to_f32(bf16_rne(bf16_to_f32(head[s:e]) * norm[None, :]))
        tg[s:e] = bf16_rne(h @ wb)
        code = bf16_to_f32(bf16_rne(bf16_to_f32(codebook[s:e]) * scale))
        tsv[s:e] = bf16_rne(code @ sender.T)
    return tg, tsv


def _dspine_weights(
    files: DrafterTensors,
    cfg: DrafterConfig,
    head: Callable[[], np.ndarray],
    dense,
) -> Iterator[tuple[str, Fp8Weight | DenseWeight]]:
    """A DSpine drafter's layers and code bridge: the layers' FP8 GEMMs as DFlash's (no
    convolutions); per layer the bridge's read (C, H) and write (H, R) bf16 in sigma K order (their
    operands come from ``dsp_read`` / ``dsp_msg``); shared the packed projection [gate |
    predecessor_gate | prev] (3R, C) bf16 in sigma K order, the gate bias, the codebook (V, C)
    bf16 natural, and the refinement tables ``dfl.tg`` / ``dfl.tsv``. An fp8_inc checkpoint's GEMM
    weights load as stored, its other tensors through bf16 / f32."""
    f32, bf16 = files.f32, files.bf16
    b = "selected_prefix_bridge."
    for name in ("independent_write_version", "terminal_disabled_version"):
        array, _ = files.raw(b + name)
        if int(np.asarray(array).reshape(-1)[0]) != 1:
            raise ValueError(f"unsupported DSpine {name}")
    if np.any(bf16_to_f32(bf16(b + "out.weight")) != 0):
        raise ValueError(
            "a terminal-disabled DSpine drafter has a nonzero out projection"
        )
    C, R, H = cfg.code_size, cfg.rank, cfg.hidden
    early, last = bf16(b + "early_read_weights"), bf16(b + "read_weight")
    writes = bf16(b + "feature_write_weights")
    if early.shape != (cfg.layers - 1, C, H) or last.shape != (C, H):
        raise ValueError(f"bridge reads {early.shape} / {last.shape}")
    if writes.shape != (cfg.layers, H, R):
        raise ValueError(f"bridge writes {writes.shape}")
    for layer in range(cfg.layers):
        yield from _attention_mlp(files, f"layers.{layer}.", layer, dense)
        read = early[layer] if layer < cfg.layers - 1 else last
        yield f"d{layer}.w_read", dense(kperm(np.ascontiguousarray(read)))
        yield f"d{layer}.w_write", dense(kperm(np.ascontiguousarray(writes[layer])))
    norm = f32("norm.weight")
    yield "dfl.norm", dense(norm)
    gate, pred, prev = (
        bf16(b + "gate.weight"),
        bf16(b + "predecessor_gate.weight"),
        bf16(b + "prev.weight"),
    )
    yield "dfl.w_pack", dense(kperm(np.concatenate([gate, pred, prev])))
    yield "dfl.gate_b", dense(f32(b + "gate.bias"))
    codebook = bf16(b + "codebook")
    if codebook.shape != (cfg.vocab, C):
        raise ValueError(f"codebook {codebook.shape}")
    yield "dfl.codebook", dense(codebook)
    tg, tsv = dspine_tables(
        head(),
        norm,
        bf16_to_f32(writes[-1]),
        bf16_to_f32(np.concatenate([pred, prev])),
        codebook,
    )
    yield "dfl.tg", dense(tg)
    yield "dfl.tsv", dense(tsv)


def rope_table(positions: int, cfg: DrafterConfig) -> np.ndarray:
    """(positions, 128) f32 rows ``[cos(64) | sin(64)]`` of the drafter's full-head RoPE."""
    half = cfg.head_dim // 2
    inverse = cfg.rope_theta ** (
        -np.arange(half, dtype=np.float64) * 2.0 / cfg.head_dim
    )
    angle = np.arange(positions, dtype=np.float64)[:, None] * inverse[None, :]
    return np.concatenate([np.cos(angle), np.sin(angle)], 1).astype(np.float32)


# ------------------------------------------------------------------ graphs
def tap_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """One tap of a target pass: the residual ``x`` (H, T) f32 through its slice ``fc`` of the
    drafter's context projection (bf16, natural K) -> ``feat`` (H, T) f32."""
    lg = LayerGraph(device, f"dfl_tap_T{tokens}", dims, tokens)
    H, T = dims.hidden, tokens
    xb = lg.tensor((H, T), BF16)
    lg.node("cast_f32_to_bf16", [lg.input("x", (H, T), F32)], [xb])
    lg.node(
        "gemm",
        [xb, lg.input("fc", (H, H), BF16)],
        [lg.input("feat", (H, T), F32)],
        GEMM_NT,
    )
    return lg.g


def _norm(lg: LayerGraph, x, y, ys, norm: str, outs=None, add: bool = True):
    """``dfl_norm``: x (+ y * ys), RMS-normed with ``norm`` -> (residual, hn f32, h16 bf16)."""
    H, T = lg.d.hidden, lg.T
    if outs is None:
        outs = (lg.tensor((H, T), F32), lg.tensor((H, T), F32), lg.tensor((H, T), BF16))
    lg.node(
        "rg_dfl_norm_f32",
        [x, y, ys, lg.weight(norm, (H,), F32)],
        list(outs),
        struct.pack("fi", lg.d.eps, int(add)),
    )
    return outs


def _conv_q8(lg: LayerGraph, cfg: DrafterConfig, hn, dyn, dyn_s, base: str):
    """Side 0 of a sublayer's convolution over ``hn`` (f32) -> its FP8 operand and scale."""
    H, T = lg.d.hidden, lg.T
    q, s = lg.tensor((H, T), F8), lg.tensor((1, T), F32)
    lg.node(
        "rg_dfl_conv_q8_f32",
        [hn, dyn, dyn_s, lg.weight(base, (H, 2 * cfg.conv_kernel), F32)],
        [q, s],
        struct.pack("ii", 0, cfg.block),
    )
    return q, s


def _conv_out(lg: LayerGraph, cfg: DrafterConfig, y, ys, dyn, dyn_s, base: str):
    """Side 1: the sublayer's FP8-GEMM output ``y`` with scale ``ys``, convolved -> f32."""
    H, T = lg.d.hidden, lg.T
    out = lg.tensor((H, T), F32)
    lg.node(
        "rg_dfl_conv_s_f32",
        [y, ys, dyn, dyn_s, lg.weight(base, (H, 2 * cfg.conv_kernel), F32)],
        [out],
        struct.pack("ii", 1, cfg.block),
    )
    return out


def _coefficients(
    lg: LayerGraph, cfg: DrafterConfig, hn, name: str, dyn=None, dyn_s=None
):
    """The convolution coefficients of a sublayer from its normed input: FP8 ``hn`` through the
    kernel projection ``name``; returns (dyn (rows, T) f32 unscaled, its token scale)."""
    H, T = lg.d.hidden, lg.T
    q = lg.tensor((H, T), F8)
    s = dyn_s if dyn_s is not None else lg.tensor((1, T), F32)
    lg.node("rg_quant_fp8_f32", [hn], [q, s])
    w, ws = lg.fp8_weight(name, cfg.kp_rows, H)
    y = dyn if dyn is not None else lg.tensor((cfg.kp_rows, T), F32)
    lg.node("fp8_gemm_f32", [q, w, lg.one(), ws], [y], GEMM_NT)
    return y, s


def _rope_rows(lg: LayerGraph, positions: int):
    """The drafter's RoPE rows (128, T) of each column's position: ``dcs_tab`` (128, positions)
    at ``pos`` (T,)."""
    cs = lg.tensor((128, lg.T), F32)
    lg.node(
        "gather_fwd_f32",
        [lg.input("dcs_tab", (128, positions), F32), lg.input("pos", (lg.T,), I32)],
        [cs],
        struct.pack("i", 1),
    )
    return cs


def inject_graph(
    device: Device,
    dims: Dims,
    cfg: DrafterConfig,
    tokens: int,
    blocks: int,
    positions: int,
) -> Graph:
    """A target step's T columns into the drafter's cache: ``feat0 .. feat4`` summed and normed,
    then per layer l k_proj / v_proj (``w_kv{l}``), k_norm + RoPE (``dcs_tab`` rows at ``pos``),
    ``kv_store`` at ``kv_rows{l}`` (8, T) [K pairs | V pairs] (-1 skips)."""
    lg = LayerGraph(device, f"dfl_inject_T{tokens}_NB{blocks}", dims, tokens)
    H, T = dims.hidden, tokens
    Tv = 128 if 1 < T < 128 else T
    total = lg.input("feat0", (H, T), F32)
    for k in range(1, len(cfg.taps)):
        acc = lg.tensor((H, T), F32)
        lg.node("add_fwd_f32", [total, lg.input(f"feat{k}", (H, T), F32)], [acc])
        total = acc
    ones = lg.input("ones", (1, T), F32)
    _, hn, _ = _norm(lg, total, total, ones, "hidden_norm", add=False)
    q8, hs = lg.tensor((H, T), F8), lg.tensor((1, T), F32)
    lg.node("rg_quant_fp8_f32", [hn], [q8, hs])
    cs = _rope_rows(lg, positions)
    arena = lg.input("arena", (256, ARENA_BLOCK_ROWS, blocks), BF16)
    for layer in range(cfg.layers):
        kv = lg.gemm8(q8, f"w_kv{layer}", cfg.kv_rows)
        kn = [lg.tensor((256, T), BF16) for _ in range(cfg.pairs)]
        vn = [lg.tensor((256, Tv), BF16) for _ in range(cfg.pairs)]
        lg.node(
            "rg_dfl_prep_kv_f32",
            [kv, hs, lg.weight(f"qk_norm{layer}", (2 * cfg.head_dim,), F32), cs],
            kn + vn,
            struct.pack("f", cfg.eps),
        )
        lg.node(
            "rg_kv_store_f32",
            [arena, lg.input(f"kv_rows{layer}", (8, T), I32), *kn, *vn],
            [lg.input(f"sink{layer}", (64, 1), I32)],
        )
    return lg.g


def embed_graph(device: Device, dims: Dims, tokens: int) -> Graph:
    """A block's input: the target's embeddings of ``tok`` (T,), normed with layer 0's
    ``attn_norm`` -> the drafter carry (``x_out``, ``hn_out``, ``h16_out``)."""
    lg = LayerGraph(device, f"dfl_embed_T{tokens}", dims, tokens)
    H, T = dims.hidden, tokens
    emb = lg.tensor((H, T), BF16)
    lg.node(
        "gather_fwd_bf16",
        [lg.weight("E_gather", (H, dims.vocab), BF16), lg.input("tok", (T,), I32)],
        [emb],
        struct.pack("i", 1),
    )
    x0 = lg.tensor((H, T), F32)
    lg.node("cast_bf16_to_f32", [emb], [x0])
    _norm(
        lg,
        x0,
        x0,
        lg.input("ones", (1, T), F32),
        "attn_norm",
        _carry_out(lg),
        add=False,
    )
    return lg.g


def _carry_out(lg: LayerGraph):
    H, T = lg.d.hidden, lg.T
    return (
        lg.input("x_out", (H, T), F32),
        lg.input("hn_out", (H, T), F32),
        lg.input("h16_out", (H, T), BF16),
    )


def pre_graph(
    device: Device,
    dims: Dims,
    cfg: DrafterConfig,
    tokens: int,
    blocks: int,
    positions: int,
) -> Graph:
    """A drafter layer up to its attention: the attention convolution's coefficients from ``hn``
    (``dyn``, ``dyn_s``, kept for the epilogue), side 0 over ``hn``, the QKV projection, q / k
    norm + RoPE in the paired layout (``q_all`` for the slot launches), the block's K / V into the
    scratch pages (``kv_rows``) and the accumulator reset."""
    lg = LayerGraph(device, f"{_prefix(cfg)}_pre_T{tokens}_NB{blocks}", dims, tokens)
    H, T, G = dims.hidden, tokens, cfg.pairs
    Tv = 128 if 1 < T < 128 else T
    W = PAIR_HEADS * T
    hn = lg.input("hn", (H, T), F32)
    if cfg.conv_kernel:
        dyn, dyn_s = _coefficients(
            lg,
            cfg,
            hn,
            "kp_a",
            lg.input("dyn", (cfg.kp_rows, T), F32),
            lg.input("dyn_s", (1, T), F32),
        )
        q8, qs = _conv_q8(lg, cfg, hn, dyn, dyn_s, "base_a")
    else:  # DSpine: no convolution
        q8, qs = lg.quant8(hn)
    p = lg.gemm8(q8, "w_qkv", cfg.qkv_rows)
    cs = _rope_rows(lg, positions)
    kn = [lg.tensor((256, T), BF16) for _ in range(G)]
    vn = [lg.tensor((256, Tv), BF16) for _ in range(G)]
    lg.node(
        "rg_dfl_prep_qkv_f32",
        [p, qs, lg.weight("qk_norm", (2 * cfg.head_dim,), F32), cs],
        [lg.input("q_all", (256, W * G), BF16), *kn, *vn],
        struct.pack("f", cfg.eps),
    )
    arena = lg.input("arena", (256, ARENA_BLOCK_ROWS, blocks), BF16)
    lg.node(
        "rg_kv_store_f32",
        [arena, lg.input("kv_rows", (8, T), I32), *kn, *vn],
        [lg.input("sink_kv", (64, 1), I32)],
    )
    acc = [
        lg.input("acc_o", (256, W, G), F32),
        lg.input("acc_m", (W, G), F32),
        lg.input("acc_l", (W, G), F32),
    ]
    lg.node("rg_attn_acc_init_f32", acc, [lg.input("sink_acc", (64, 1), I32)])
    return lg.g


def slots_graph(
    device: Device, dims: Dims, cfg: DrafterConfig, tokens: int, slots: int, blocks: int
) -> Graph:
    """One launch of the block's attention slots (``layers.attn_slots_graph`` with the drafter's
    softmax: bidirectional (DSpine: causal) block, window start and width per slot, scale of a
    128-wide head)."""
    return attn_slots_graph(
        device,
        dims,
        tokens,
        slots,
        blocks,
        cfg.block,
        kernel="rg_attn_slot_pc_f32" if cfg.causal else "rg_attn_slot_pd_f32",
        per_column=True,
        scale=1.0 / cfg.head_dim**0.5,
    )


def post_graph(device: Device, dims: Dims, cfg: DrafterConfig, tokens: int) -> Graph:
    """A drafter layer from its attention output: ``acc_o`` -> the o_proj operand (``dfl_oq``),
    o_proj, side 1 of the attention convolution, the residual and ``post_norm``, the MLP with its
    convolution (coefficients ``kp_m``, ``base_m``), the residual and ``next_norm`` (the next
    layer's ``attn_norm`` or the final ``norm``) -> the carry."""
    if cfg.kind == DSPINE:
        return _dspine_post_graph(device, dims, cfg, tokens)
    lg = LayerGraph(device, f"dfl_post_T{tokens}", dims, tokens)
    H, T, G, F = dims.hidden, tokens, cfg.pairs, cfg.ffn
    W = PAIR_HEADS * T
    ones = lg.input("ones", (1, T), F32)
    o8, os_ = lg.tensor((cfg.heads * cfg.head_dim, T), F8), lg.tensor((1, T), F32)
    lg.node("rg_dfl_oq_f32", [lg.input("acc_o", (256, W, G), F32)], [o8, os_])
    y = lg.gemm8(o8, "w_o", H)
    dyn, dyn_s = lg.input("dyn", (cfg.kp_rows, T), F32), lg.input("dyn_s", (1, T), F32)
    a = _conv_out(lg, cfg, y, os_, dyn, dyn_s, "base_a")
    x2, hn2, _ = _norm(lg, lg.input("x", (H, T), F32), a, ones, "post_norm")
    dyn2, s2 = _coefficients(lg, cfg, hn2, "kp_m")
    g8, gs = _conv_q8(lg, cfg, hn2, dyn2, s2, "base_m")
    gu = lg.gemm8(g8, "w_gu", 2 * F)
    sg = lg.tensor((F, T), F32)
    lg.node("rg_swiglu_f8_f32", [gu, gs], [sg])
    sg8, sgs = lg.quant8(sg)
    m = lg.gemm8(sg8, "w_down", H)
    mo = _conv_out(lg, cfg, m, sgs, dyn2, s2, "base_m")
    _norm(lg, x2, mo, ones, "next_norm", _carry_out(lg))
    return lg.g


def _dspine_post_graph(
    device: Device, dims: Dims, cfg: DrafterConfig, tokens: int
) -> Graph:
    """A DSpine layer from its attention output: o_proj, the residual and ``post_norm``, the MLP,
    then the code bridge's write (module docstring) and ``next_norm`` -> the carry. The write's
    scale ``dsp_ws`` (1, T) and the packed codes ``dsp_pk`` (3R, T) stay in persistent buffers
    (the last layer's are the refinement's)."""
    lg = LayerGraph(device, f"dsp_post_T{tokens}", dims, tokens)
    H, T, G, F = dims.hidden, tokens, cfg.pairs, cfg.ffn
    C, R, B = cfg.code_size, cfg.rank, tokens // cfg.block
    W = PAIR_HEADS * T
    o8, os_ = lg.tensor((cfg.heads * cfg.head_dim, T), F8), lg.tensor((1, T), F32)
    lg.node("rg_dfl_oq_f32", [lg.input("acc_o", (256, W, G), F32)], [o8, os_])
    y = lg.gemm8(o8, "w_o", H)
    x2, hn2, _ = _norm(lg, lg.input("x", (H, T), F32), y, os_, "post_norm")
    g8, gs = lg.quant8(hn2)
    gu = lg.gemm8(g8, "w_gu", 2 * F)
    sg = lg.tensor((F, T), F32)
    lg.node("rg_swiglu_f8_f32", [gu, gs], [sg])
    sg8, sgs = lg.quant8(sg)
    m = lg.gemm8(sg8, "w_down", H)
    raw, nb = lg.tensor((H, T), F32), lg.tensor((H, T), BF16)
    ws = lg.input("dsp_ws", (1, T), F32)
    lg.node(
        "rg_dsp_read_f32",
        [x2, m, sgs],
        [raw, nb, ws],
        struct.pack("fi", cfg.bridge_eps, cfg.block),
    )
    proj = lg.gemm(nb, lg.weight("w_read", (H, C), BF16), C)
    codes = lg.tensor((C, T), BF16)
    lg.node(
        "rg_dsp_codes_f32",
        [
            proj,
            lg.weight("codebook", (C, cfg.vocab), BF16),
            lg.input("anchors", (B,), I32),
        ],
        [codes],
        struct.pack("fi", cfg.bridge_eps, cfg.block),
    )
    pk = lg.input("dsp_pk", (3 * R, T), F32)
    lg.node("gemm", [codes, lg.weight("w_pack", (C, 3 * R), BF16)], [pk], GEMM_NT)
    msg = lg.tensor((R, T), BF16)
    lg.node(
        "rg_dsp_msg_f32",
        [pk, lg.weight("gate_b", (R,), F32)],
        [msg],
        struct.pack("i", cfg.block),
    )
    delta = lg.gemm(msg, lg.weight("w_write", (R, H), BF16), H)
    _norm(lg, raw, delta, ws, "next_norm", _carry_out(lg))
    return lg.g


def _dspine_head_graph(
    device: Device, dims: Dims, cfg: DrafterConfig, tokens: int, drafts: int
) -> Graph:
    """The block's drafts by the serial refinement: the LM head over the final hidden state
    ``h16``, each column's ``top_k`` logits and ids, and ``dsp_select``'s greedy walk from
    ``anchors`` over the last write (``x_fin`` the final residual, ``dsp_ws``, ``dsp_pk``) ->
    ``next`` (drafts + 1, B) i32."""
    lg = LayerGraph(device, f"dsp_head_T{tokens}_K{drafts}", dims, tokens)
    H, V, T, R = dims.hidden, dims.vocab, tokens, cfg.rank
    B = T // cfg.block
    h = lg.input("h16", (H, T), BF16)
    logits = lg.input("logits", (V, T), F32)
    lg.node("gemm", [h, lg.weight("E_head", (H, V), BF16)], [logits], GEMM_NT)
    top_v, top_i = _top_k(lg, logits, cfg.top_k)
    lg.node(
        "rg_dsp_select_f32",
        [
            top_v,
            top_i,
            lg.input("x_fin", (H, T), F32),
            lg.input("dsp_ws", (1, T), F32),
            lg.input("dsp_pk", (3 * R, T), F32),
            lg.weight("gate_b", (R,), F32),
            lg.input("anchors", (B,), I32),
            lg.weight("tg", (R, V), BF16),
            lg.weight("tsv", (2 * R, V), BF16),
        ],
        [lg.input("next", (drafts + 1, B), I32)],
        struct.pack("if", cfg.block, cfg.eps),
    )
    return lg.g


def _top_k(lg: LayerGraph, logits, k: int):
    """The ``k`` largest logits of each column and their ids, unordered (``dfl_tk1`` picks the
    k chunks of 16 with the largest maxima, ``dfl_tk2`` the k largest of their logits)."""
    T = logits.shape[1]
    keys = lg.tensor((k, T), I32)
    lg.node("rg_dfl_tk1_f32", [logits], [keys])
    top_v, top_i = lg.tensor((k, T), F32), lg.tensor((k, T), I32)
    lg.node("rg_dfl_tk2_f32", [logits, keys], [top_v, top_i])
    return top_v, top_i


def _prefix(cfg: DrafterConfig) -> str:
    """graph name prefix of the drafter kind's layer graphs"""
    return "dsp" if cfg.kind == DSPINE else "dfl"


def head_graph(
    device: Device, dims: Dims, cfg: DrafterConfig, tokens: int, drafts: int
) -> Graph:
    """The block's drafts: the LM head over the final hidden state ``h16`` (T columns, B = T /
    block rows), each column's ``top_k`` logits and ids, the selector projection, and the greedy
    walk from ``anchors`` (B,) over the first ``drafts`` mask positions -> ``next`` (drafts + 1,
    B) i32: [anchor, drafts] per row, as the next verify step reads its tokens."""
    if cfg.kind == DSPINE:
        return _dspine_head_graph(device, dims, cfg, tokens, drafts)
    lg = LayerGraph(device, f"dfl_head_T{tokens}_K{drafts}", dims, tokens)
    H, V, T = dims.hidden, dims.vocab, tokens
    B = T // cfg.block
    h = lg.input("h16", (H, T), BF16)
    logits = lg.input("logits", (V, T), F32)
    lg.node("gemm", [h, lg.weight("E_head", (H, V), BF16)], [logits], GEMM_NT)
    top_v, top_i = _top_k(lg, logits, cfg.top_k)
    hp = lg.gemm(h, lg.weight("hproj", (H, cfg.rank), BF16), cfg.rank)
    lg.node(
        "rg_dfl_select_f32",
        [
            top_v,
            top_i,
            hp,
            lg.input("anchors", (B,), I32),
            lg.weight("pred", (cfg.rank, V), BF16),
            lg.weight("succ", (cfg.rank, V), BF16),
        ],
        [lg.input("next", (drafts + 1, B), I32)],
        struct.pack("i", cfg.block),
    )
    return lg.g


# ------------------------------------------------------------------ tables
@dataclass(frozen=True, slots=True)
class DraftRow:
    """One row of a drafting step: its anchor's position (the block's first), and the DRAFT cache
    group's block rows of the row, ``first`` being the logical block of ``blocks`` [0]."""

    position: int
    blocks: np.ndarray
    first: int


def inject_rows(
    cfg: DrafterConfig, tokens: int, rows: Sequence[tuple[int, int, int, DraftRow]]
) -> np.ndarray:
    """(layers, tokens, 8) i32 ``kv_rows`` of an injection: per (column, position, count, row),
    the K pair rows then the V pair rows (sigma_inv order inside a page) of ``count`` columns from
    ``column`` at positions from ``position``; -1 elsewhere."""
    out = np.full((cfg.layers, tokens, 8), -1, np.int64)
    pair = np.arange(cfg.pairs) * PAGE_TOKENS
    for column, position, count, row in rows:
        if not count:
            continue
        pos = position + np.arange(count)
        page, r = pos // PAGE_TOKENS - row.first, pos % PAGE_TOKENS
        # positions before the retained blocks are older than the window: nothing reads them
        kept = page >= 0
        if page.max() >= len(row.blocks):
            raise ValueError("positions past the row's DRAFT blocks")
        columns = column + np.flatnonzero(kept)
        page, r = page[kept], r[kept]
        base = row.blocks[page]  # (kept,)
        for layer in range(cfg.layers):
            lb = base + layer * cfg.layer_rows
            out[layer, columns, : cfg.pairs] = lb[:, None] + pair[None, :] + r[:, None]
            out[layer, columns, 4 : 4 + cfg.pairs] = (
                lb[:, None]
                + (cfg.pairs + np.arange(cfg.pairs))[None, :] * PAGE_TOKENS
                + sigma_inv(r)[:, None]
            )
    return np.ascontiguousarray(out).astype(np.int32)


def block_rows_table(
    cfg: DrafterConfig, tokens: int, scratch: np.ndarray, count: int
) -> np.ndarray:
    """(layers, tokens, 8) i32 ``kv_rows`` of a block step's own K / V: row b's column j goes to
    its scratch pages ``scratch`` (B, layers, pairs, 2) first rows, K at row j, V at sigma_inv(j);
    rows past ``count`` get -1."""
    out = np.full((cfg.layers, tokens, 8), -1, np.int64)
    j = np.arange(cfg.block)
    for b in range(count):
        cols = slice(b * cfg.block, (b + 1) * cfg.block)
        for layer in range(cfg.layers):
            out[layer, cols, : cfg.pairs] = (
                scratch[b, layer, :, 0][None, :] + j[:, None]
            )
            out[layer, cols, 4 : 4 + cfg.pairs] = (
                scratch[b, layer, :, 1][None, :] + sigma_inv(j)[:, None]
            )
    return np.ascontiguousarray(out).astype(np.int32)


def plan_block_slots(
    cfg: DrafterConfig,
    rows: Sequence[DraftRow],
    scratch: np.ndarray,
    per_launch: int,
    low: Sequence[int] | None = None,
    records: list[tuple[int, int]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The slot launches of a block step's attention, for layer 0 (layer l adds ``l *
    cfg.layer_rows`` to every page): per row and pair, one slot per 2048-key chunk of the context
    the block can see (positions >= p - window + 1, below p) and one slot for the block itself
    (keys p .. p + block - 1 from the row's scratch pages). Slot fields: [pair, qcol0, p, key0,
    klen, 0, kmin, window]; (per_launch, 16) page tables, -1 for pages no key of the chunk needs.
    Returns per launch (slots (S, 8), k_pages (S, 16), v_pages (S, 16)) i32.

    With ``low`` (per row, at most its position): the anchor may sit anywhere in [low, position],
    and the launches hold every chunk and page a block at any of those positions reads (the
    kernels mask by each slot's kmin and klen); ``records`` then receives (row, kind) of each slot
    in order, kind 0 a context chunk and 1 the block, for ``spec_fix`` to set their positions."""
    per_chunk = SLOT_KEYS // PAGE_TOKENS
    slots, kps, vps = [], [], []
    lane = np.arange(per_chunk)
    for b, row in enumerate(rows):
        p = row.position
        lo = p if low is None else int(low[b])
        kmin = max(0, lo - cfg.window + 1)
        qcol0 = PAIR_HEADS * cfg.block * b
        chunks = (
            range(kmin // SLOT_KEYS, (p - 1) // SLOT_KEYS + 1) if p > 0 else range(0)
        )
        for key0 in (c * SLOT_KEYS for c in chunks):
            pages = key0 // PAGE_TOKENS + lane
            first_page = max(kmin, key0) // PAGE_TOKENS
            last_page = (p - 1) // PAGE_TOKENS
            live = (pages >= first_page) & (pages <= last_page)
            index = np.where(live, pages - row.first, 0)
            if (index[live] < 0).any() or (index[live] >= len(row.blocks)).any():
                raise ValueError("the window reaches outside the row's DRAFT blocks")
            base = np.where(live, row.blocks[index], -1)
            for pg in range(cfg.pairs):
                slots.append([pg, qcol0, p, key0, p, 0, kmin, cfg.window])
                kps.append(np.where(live, base + pg * PAGE_TOKENS, -1))
                vps.append(np.where(live, base + (cfg.pairs + pg) * PAGE_TOKENS, -1))
                if records is not None:
                    records.append((b, 0))
        for pg in range(cfg.pairs):
            slots.append([pg, qcol0, p, p, p + cfg.block, 0, p, cfg.window])
            if records is not None:
                records.append((b, 1))
            k = np.full(per_chunk, -1, np.int64)
            v = np.full(per_chunk, -1, np.int64)
            k[0], v[0] = scratch[b, 0, pg, 0], scratch[b, 0, pg, 1]
            kps.append(k)
            vps.append(v)
    slots = np.asarray(slots, np.int64).reshape(-1, 8)
    kps, vps = np.asarray(kps, np.int64), np.asarray(vps, np.int64)
    pad = -(-len(slots) // per_launch) * per_launch - len(slots)
    if pad:
        fill = np.zeros((pad, 8), np.int64)
        fill[:, 0] = -1
        slots = np.concatenate([slots, fill])
        kps = np.concatenate([kps, np.full((pad, per_chunk), -1)])
        vps = np.concatenate([vps, np.full((pad, per_chunk), -1)])
    n = len(slots) // per_launch
    return [
        (
            slots[i * per_launch : (i + 1) * per_launch].astype(np.int32),
            kps[i * per_launch : (i + 1) * per_launch].astype(np.int32),
            vps[i * per_launch : (i + 1) * per_launch].astype(np.int32),
        )
        for i in range(n)
    ]


def slots_per_launch(rows: int, pairs: int) -> int:
    """slots of one launch of a block step of ``rows`` rows: one chunk of every row and pair"""
    return max(16, rows * pairs)


# ------------------------------------------------------------------ the runtime
def _round(nbytes: int) -> int:
    return max(SMALL, -(-nbytes // SMALL) * SMALL)


class _Mode(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def columns(self) -> int: ...


class DrafterHost(Protocol):
    """What a target model gives its ``Drafter``: its device and dimensions, the shapes of its
    passes, its cache arena, and lookups of the tensors the drafter's recipes bind."""

    @property
    def device(self) -> Device: ...

    @property
    def dims(self) -> Dims:
        """the target's hidden size and vocabulary (``E_gather``, ``E_head``, the taps)"""
        ...

    @property
    def columns(self) -> int:
        """columns of the widest target pass (the ``feat`` buffers)"""
        ...

    @property
    def decode_buckets(self) -> Sequence[int]:
        """row buckets of the block steps"""
        ...

    @property
    def max_positions(self) -> int: ...

    @property
    def K(self) -> int:
        """drafts a step verifies (1 .. the drafter's block - 1)"""
        ...

    @property
    def modes(self) -> Mapping[str, _Mode]:
        """the target's passes by name: each needs tap and inject recipes of its columns"""
        ...

    @property
    def arena(self) -> int: ...

    def weight(self, name: str) -> int:
        """Device address of ``E_gather``, ``E_head``, ``dfl.*``, ``d{l}.*`` (and ``.s``)."""
        ...

    def buffer(self, name: str) -> int:
        """Device address of ``one`` (f32 ones) and ``logits`` (vocab x columns f32 scratch)."""
        ...

    def recipe_of(self, key: tuple) -> Recipe:
        """The compiled recipe of a drafter graph (``Drafter.graphs`` keys)."""
        ...

    def upload(self, address: int, array: np.ndarray) -> None:
        """Queue a table upload behind the launches already submitted."""
        ...

    def tap_source(self, layer: int) -> int:
        """Device address of the residual output of target layer ``layer`` in a pass."""
        ...

    def tap_graph(self, tokens: int) -> Graph:
        """The tap graph of a pass of ``tokens`` columns: inputs ``x`` (the ``tap_source``),
        ``fc`` (H, H) bf16 and output ``feat`` (H, T) f32 (``tap_graph`` for Qwen)."""
        ...


class Drafter:
    """The drafter of a target built with a DFlash 2 checkpoint: ``tap_steps`` inside a target
    pass, ``inject`` after it, ``block`` for the next drafts."""

    def __init__(self, m: DrafterHost, cfg: DrafterConfig, scratch_rows: int) -> None:
        self.m, self.cfg, dev = m, cfg, m.device
        d = m.dims
        H = d.hidden
        self.dims = cfg.dims(d)
        T, B = m.columns, m.decode_buckets[-1]
        TD = cfg.block * B  # the widest block step
        self.TD = TD

        def buf(nbytes: int) -> int:
            return dev.malloc(_round(nbytes))

        self.feat = [buf(H * T * 4) for _ in cfg.taps]
        self.ones = buf(T * 4)
        dev.h2d(self.ones, np.ones(T, np.float32))
        self.positions = positions = m.max_positions + cfg.block
        self.dcs_tab = buf(positions * 128 * 4)
        dev.h2d(self.dcs_tab, rope_table(positions, cfg))
        self.pos_in = buf(T * 4)  # the injected columns' positions
        self.kv_rows = buf(cfg.layers * _round(8 * T * 4))
        self.kv_stride = _round(8 * T * 4)
        self.sinks = buf((cfg.layers + 3) * SMALL)  # per inject layer, kv, acc, slots
        # the block step
        self.carry = [
            {"x": buf(H * TD * 4), "hn": buf(H * TD * 4), "h16": buf(H * TD * 2)}
            for _ in range(2)
        ]
        self.dyn, self.dyn_s = buf(cfg.kp_rows * TD * 4), buf(TD * 4)
        # DSpine: the code bridge's write scale and packed codes (the last layer's are read
        # by the refinement)
        self.dsp_ws, self.dsp_pk = buf(TD * 4), buf(3 * cfg.rank * TD * 4)
        W = PAIR_HEADS * TD
        self.q_all = buf(256 * W * cfg.pairs * 2)
        self.acc_o = buf(256 * W * cfg.pairs * 4)
        self.acc_m, self.acc_l = buf(W * cfg.pairs * 4), buf(W * cfg.pairs * 4)
        self.pos, self.tok, self.anchors = buf(TD * 4), buf(TD * 4), buf(B * 4)
        self.next = buf(cfg.block * B * 4)
        self.block_rows = buf(cfg.layers * _round(8 * TD * 4))
        self.block_stride = _round(8 * TD * 4)
        S = slots_per_launch(B, cfg.pairs)
        self.slot_stride = _round(S * 8 * 4) + 2 * _round(S * 16 * 4)
        # per row and pair at most window / SLOT_KEYS + 1 context chunks and the block
        self.max_launches = cfg.window // SLOT_KEYS + 2
        self.tables = buf(cfg.layers * self.max_launches * self.slot_stride)
        self.tables_words = cfg.layers * self.max_launches * self.slot_stride // 4
        # scratch pages: (B, layers, pairs, [K, V]) first rows, after the cache arena
        pages = np.arange(B * cfg.layers * cfg.pairs * 2, dtype=np.int64) * PAGE_TOKENS
        self.scratch = (scratch_rows + pages).reshape(B, cfg.layers, cfg.pairs, 2)
        self._binds: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._plans: dict[tuple, LaunchPlan] = {}
        # the widths the next verify step may take; ``next`` holds the last block step's
        # tokens for ``next_width`` columns per row
        self.widths = tuple(getattr(m, "verify_widths", ()) or (m.K + 1,))
        self.next_width = m.K + 1

    @staticmethod
    def scratch_rows(cfg: DrafterConfig, rows: int) -> int:
        """arena rows of the scratch pages of ``rows`` block-step rows"""
        return rows * cfg.layers * cfg.pairs * 2 * PAGE_TOKENS

    # ---- graphs
    def graphs(self, blocks: int) -> dict[tuple, Graph]:
        m, cfg, dev, d = self.m, self.cfg, self.m.device, self.dims
        out: dict[tuple, Graph] = {}
        P = self.positions
        for mode in m.modes.values():
            out["dfl_tap", mode.name] = m.tap_graph(mode.columns)
            out["dfl_inject", mode.name] = inject_graph(
                dev, m.dims, cfg, mode.columns, blocks, P
            )
        for B in m.decode_buckets:
            for w in self.widths:
                c = self.block_cfg(w)
                T = c.block * B
                out["dfl_embed", B, w] = embed_graph(dev, d, T)
                out["dfl_pre", B, w] = pre_graph(dev, d, c, T, blocks, P)
                out["dfl_slots", B, w] = slots_graph(
                    dev, d, c, T, slots_per_launch(B, c.pairs), blocks
                )
                out["dfl_post", B, w] = post_graph(dev, d, c, T)
                out["dfl_head", B, w] = head_graph(dev, d, c, T, w - 1)
        return out

    def block_cfg(self, width: int) -> DrafterConfig:
        """The configuration of a block step for a verify of ``width`` columns: a causal block
        runs only its first ``width`` positions (none of them sees the later ones), a
        bidirectional block all of them."""
        cfg = self.cfg
        return replace(cfg, block=width) if cfg.causal and width < cfg.block else cfg

    # ---- bindings
    def _names(self, kind: str, layer: int = 0) -> dict[str, int]:
        """Every persistent tensor of a drafter recipe of ``kind`` launched as ``layer``."""
        m, cfg = self.m, self.cfg
        w = m.weight
        names: dict[str, int] = {
            "one": m.buffer("one"),
            "ones": self.ones,
            "dcs_tab": self.dcs_tab,
            "arena": m.arena,
            "E_gather": w("E_gather"),
            "E_head": w("E_head"),
            "logits": m.buffer("logits"),
            "tok": self.tok,
            "anchors": self.anchors,
            "next": self.next,
            "q_all": self.q_all,
            "acc_o": self.acc_o,
            "acc_m": self.acc_m,
            "acc_l": self.acc_l,
            "dyn": self.dyn,
            "dyn_s": self.dyn_s,
            "sink_kv": self.sinks + cfg.layers * SMALL,
            "sink_acc": self.sinks + (cfg.layers + 1) * SMALL,
            "sink_slots": self.sinks + (cfg.layers + 2) * SMALL,
            "norm": w("dfl.norm"),
            "hidden_norm": w("dfl.hidden_norm"),
        }
        if cfg.kind == DSPINE:
            names.update(
                {
                    "dsp_ws": self.dsp_ws,
                    "dsp_pk": self.dsp_pk,
                    "codebook": w("dfl.codebook"),
                    "w_pack": w("dfl.w_pack"),
                    "gate_b": w("dfl.gate_b"),
                    "tg": w("dfl.tg"),
                    "tsv": w("dfl.tsv"),
                }
            )
        else:
            names.update(
                {"hproj": w("dfl.hproj"), "pred": w("dfl.pred"), "succ": w("dfl.succ")}
            )
        for k in range(len(cfg.taps)):
            names[f"feat{k}"] = self.feat[k]
        if kind == "dfl_inject":
            names["pos"] = self.pos_in
            for layer in range(cfg.layers):
                names[f"w_kv{layer}"] = w(f"d{layer}.w_kv")
                names[f"w_kv{layer}.s"] = w(f"d{layer}.w_kv.s")
                names[f"qk_norm{layer}"] = w(f"d{layer}.qk_norm")
                names[f"kv_rows{layer}"] = self.kv_rows + layer * self.kv_stride
                names[f"sink{layer}"] = self.sinks + layer * SMALL
            return names
        src, dst = self.carry[layer % 2], self.carry[1 - layer % 2]
        if kind == "dfl_embed":
            names.update({f"{k}_out": v for k, v in self.carry[0].items()})
            names["attn_norm"] = w("d0.attn_norm")
            return names
        if kind == "dfl_head":
            names["h16"] = self.carry[cfg.layers % 2]["h16"]
            names["x_fin"] = self.carry[cfg.layers % 2]["x"]
            return names
        names["pos"] = self.pos
        names.update(src)
        names.update({f"{k}_out": v for k, v in dst.items()})
        names["kv_rows"] = self.block_rows + layer * self.block_stride
        nxt = f"d{layer + 1}.attn_norm" if layer + 1 < cfg.layers else "dfl.norm"
        names["next_norm"] = w(nxt)
        dspine = cfg.kind == DSPINE
        dense = ("w_read", "w_write") if dspine else ("base_a", "base_m")
        for key in ("attn_norm", "post_norm", "qk_norm", *dense):
            names[key] = w(f"d{layer}.{key}")
        fp8 = () if dspine else ("kp_a", "kp_m")
        for key in ("w_qkv", "w_o", "w_gu", "w_down", *fp8):
            names[key] = w(f"d{layer}.{key}")
            names[key + ".s"] = w(f"d{layer}.{key}.s")
        return names

    def _step(self, key: tuple, layer: int = 0, extra: dict[str, int] | None = None):
        """(recipe, slots, addresses) of one launch of a drafter recipe."""
        recipe: Recipe = self.m.recipe_of(key)
        bound = self._binds.get((key, layer))
        if bound is None:
            if key[0] == "dfl_tap":  # x, fc and feat come per launch
                pairs = []
            else:
                names = self._names(key[0], layer)
                pairs = [
                    (recipe.slot(n), names[n])
                    for n in recipe.persistent
                    if n not in DYNAMIC
                ]
            bound = self._binds[(key, layer)] = (
                np.array([p[0] for p in pairs], np.int32),
                np.array([p[1] for p in pairs], np.uint64),
            )
        slots, addresses = bound
        if extra:
            slots = np.concatenate(
                [slots, np.array([recipe.slot(n) for n in extra], np.int32)]
            )
            addresses = np.concatenate(
                [addresses, np.array(list(extra.values()), np.uint64)]
            )
        return recipe, slots, addresses

    # ---- the target pass
    def tap_steps(self, mode: str) -> dict[int, tuple]:
        """The tap launches of a target pass of ``mode``, by the target layer after which each
        runs: ``x`` is that layer's output (``tap_source``), ``fc`` its slice, ``feat`` its
        part."""
        m = self.m
        out = {}
        for k, layer in enumerate(self.cfg.taps):
            out[layer] = self._step(
                ("dfl_tap", mode),
                extra={
                    "x": m.tap_source(layer),
                    "fc": m.weight(f"dfl.fc{k}"),
                    "feat": self.feat[k],
                },
            )
        return out

    def inject(self, mode: str, rows: Sequence[tuple[int, int, int, DraftRow]]) -> None:
        """The last target pass of ``mode`` into the cache: ``rows`` as ``inject_rows``."""
        m, cfg = self.m, self.cfg
        T = m.modes[mode].columns
        table = inject_rows(cfg, T, rows)
        pos = np.zeros(T, np.int32)
        for column, position, count, _ in rows:
            pos[column : column + count] = position + np.arange(count)
        m.upload(self.pos_in, pos)
        for layer in range(cfg.layers):
            m.upload(self.kv_rows + layer * self.kv_stride, table[layer])
        recipe, slots, addresses = self._step(("dfl_inject", mode))
        recipe.bind_slots(slots, addresses)
        recipe.launch()

    # ---- the block
    def block(
        self,
        rows: Sequence[DraftRow],
        anchors: Sequence[int],
        bucket: int,
        width: int | None = None,
    ) -> None:
        """Draft after each row's anchor (token ``anchors`` [b] at ``rows`` [b].position): the
        block step of ``bucket`` rows; ``self.next`` (width, bucket) i32 then holds [anchor,
        drafts] per row for a verify step of ``width`` columns (default: every draft; rows past
        ``len(rows)`` are padding)."""
        width = width or self.m.K + 1
        launches, _ = self._tables(rows, anchors, bucket, width)
        self.run(bucket, launches, width)

    def prepare_device(
        self, rows: Sequence[DraftRow], low: Sequence[int], bucket: int, width: int
    ) -> tuple[int, np.ndarray]:
        """The tables of a block step whose anchors the device decides (``spec_fix``): row b's
        anchor at a position in [low[b], rows[b].position], its token and positions written by
        ``spec_fix``. Returns the slot launches per layer and ``spec_fix``'s patch entries (P, 4)
        [record offset in ``tables`` (i32 units), row, kind, 0] of every layer's slot records."""
        records: list[tuple[int, int]] = []
        launches, slots = self._tables(rows, None, bucket, width, low, records)
        cfg = self.block_cfg(width)
        per_layer = self.max_launches * self.slot_stride // 4
        entries = []
        for layer in range(cfg.layers):
            for k, (row, kind) in enumerate(records):
                i, s = divmod(k, slots)
                entries.append(
                    (
                        layer * per_layer + i * self.slot_stride // 4 + 8 * s,
                        row,
                        kind,
                        0,
                    )
                )
        return launches, np.asarray(entries, np.int32).reshape(-1, 4)

    def _tables(
        self,
        rows: Sequence[DraftRow],
        anchors: Sequence[int] | None,
        bucket: int,
        width: int,
        low: Sequence[int] | None = None,
        records: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        """Upload a block step's tokens, positions, anchors, scratch rows and slot tables (with
        ``anchors`` None: the anchors' columns and positions are left for ``spec_fix``). Returns
        (slot launches per layer, slots per launch)."""
        m, dev = self.m, self.m.device
        if width not in self.widths:
            raise ValueError(f"no block step for a verify of {width} columns")
        cfg = self.block_cfg(width)
        B, T = len(rows), cfg.block * bucket
        if not 0 < B <= bucket <= m.decode_buckets[-1]:
            raise ValueError(f"{B} rows in a block step of {bucket}")
        tok = np.full(T, cfg.mask_token, np.int32)
        pos = np.zeros(T, np.int32)
        anc = np.zeros(bucket, np.int32)
        for b, row in enumerate(rows):
            if anchors is not None:
                tok[b * cfg.block] = anchors[b]
                anc[b] = anchors[b]
            pos[b * cfg.block : (b + 1) * cfg.block] = row.position + np.arange(
                cfg.block
            )
        if int(pos.max()) + 1 > m.max_positions + cfg.block:
            raise ValueError(f"position exceeds {m.max_positions}")
        dev.h2d_nb(self.tok, tok)
        dev.h2d_nb(self.pos, pos)
        dev.h2d_nb(self.anchors, anc)
        table = block_rows_table(cfg, T, self.scratch, B)
        for layer in range(cfg.layers):
            m.upload(self.block_rows + layer * self.block_stride, table[layer])
        S = slots_per_launch(bucket, cfg.pairs)
        launches = plan_block_slots(cfg, rows, self.scratch, S, low, records)
        if len(launches) > self.max_launches:
            raise ValueError(
                f"{len(launches)} slot launches exceed {self.max_launches}"
            )
        stride, k_off = self.slot_stride, _round(S * 8 * 4)
        v_off = k_off + _round(S * 16 * 4)
        for layer in range(cfg.layers):
            shift = layer * cfg.layer_rows
            table = np.zeros((len(launches), stride // 4), np.int32)
            for i, (slots, kp, vp) in enumerate(launches):
                table[i, : S * 8] = slots.reshape(-1)
                table[i, k_off // 4 : k_off // 4 + S * 16] = np.where(
                    kp >= 0, kp + shift, -1
                ).reshape(-1)
                table[i, v_off // 4 : v_off // 4 + S * 16] = np.where(
                    vp >= 0, vp + shift, -1
                ).reshape(-1)
            m.upload(self.tables + layer * self.max_launches * stride, table)
        return len(launches), S

    def run(self, bucket: int, launches: int, width: int) -> None:
        """Launch a block step whose tables are uploaded (``_tables``, ``spec_fix``)."""
        key = (bucket, launches, width)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = LaunchPlan(
                self.m.device, self.steps(bucket, launches, width)
            )
        plan.run()
        self.next_width = width

    def steps(
        self, bucket: int, launches: int, width: int | None = None
    ) -> list[tuple]:
        """The launches of a block step of ``bucket`` rows with ``launches`` slot launches per
        layer: embed, per layer pre / slots / post, head (for ``width`` verify columns)."""
        m, cfg = self.m, self.cfg
        w = width or m.K + 1
        S = slots_per_launch(bucket, cfg.pairs)
        stride, k_off = self.slot_stride, _round(S * 8 * 4)
        v_off = k_off + _round(S * 16 * 4)
        steps = [self._step(("dfl_embed", bucket, w))]
        recipe = m.recipe_of(("dfl_slots", bucket, w))
        table_slots = np.array(
            [recipe.slot(n) for n in ("slots", "k_pages", "v_pages")], np.int32
        )
        for layer in range(cfg.layers):
            steps.append(self._step(("dfl_pre", bucket, w), layer))
            r, s, a = self._step(("dfl_slots", bucket, w), layer)
            for i in range(launches):
                base = self.tables + (layer * self.max_launches + i) * stride
                tables = np.array([base, base + k_off, base + v_off], np.uint64)
                steps.append(
                    (r, np.concatenate([s, table_slots]), np.concatenate([a, tables]))
                )
            steps.append(self._step(("dfl_post", bucket, w), layer))
        steps.append(self._step(("dfl_head", bucket, w)))
        return steps

    def read_next(self, bucket: int, width: int | None = None) -> np.ndarray:
        """(bucket, width) i32: the last block step's [anchor, drafts] per row (waits)."""
        return self.m.device.d2h(
            self.next, (bucket, width or self.next_width), np.int32
        )


__all__ = [
    "CANDIDATES_MAX",
    "HF_REPO",
    "HF_REVISION",
    "DraftRow",
    "Drafter",
    "DrafterConfig",
    "DrafterHost",
    "block_fp8",
    "block_rows_table",
    "drafter_draft_tokens",
    "inject_rows",
    "plan_block_slots",
    "prepare_weights",
    "rope_table",
    "scale_bf16",
    "slots_per_launch",
    "tap_graph",
]
