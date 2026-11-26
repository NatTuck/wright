"""Declared floors: the bytes a recipe family must move, and the regime that binds it.

Every entry states its derivation in one line, so a number can be re-checked rather than trusted. The floor is
**bytes divided by the full envelope at any tile size** — a span that misses it is above floor, and a small
tile is a finding to fix, not an allowance to make. The envelope figures are the manual's (ch 1, MEASURED at
≥16 MB DMA / ≥134 MB TPC); using them on a 5 MB tile is deliberate: it makes the small-tile penalty visible
as a ratio instead of hiding it as a caveat.

Regime: ``tpc`` (TPC stream: gather / dequant / elementwise), ``dma`` (DMA stream: copies and the MME's
operand fetch) or ``flops`` (MME matmul). A matmul judged
against a byte floor would look infinitely good and a gather against a FLOP floor infinitely bad, so a family
must say which binds it. ``None`` means the formula is not derivable from the recipe name alone — the span is
reported without a ratio rather than with a wrong one.

Usage from the timeline tool: ``family(name)`` groups, ``floor(name)`` gives ``(bytes, regime, derivation)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# DeepSeek-V4.1-Flash text config (config.json), and the split we run.
HIDDEN = 5120
MOE_INTER = 2304
LAYERS = 40
EXPERTS = 384
TOP = 6
ENGRAM_HEADS = 8
ENGRAM_HEAD_DIM = 256
CARDS = 4

# Envelope bandwidth, bytes/s (manual ch 1: TPC stream ≈ 2147 GB/s at ≥134 MB, DMA ≈ 1776 GB/s at 16–64 MB).
TPC_BYTES_PER_S = 2147e9
DMA_BYTES_PER_S = 1776e9


def _t(name: str) -> int:
    """The token count a recipe name carries (``_T2048``), or 0 when it names none."""
    found = re.search(r"_T(\d+)", name)
    return int(found.group(1)) if found else 0


@dataclass(frozen=True)
class Floor:
    """One family's floor: what it must move, which resource binds it, and why."""

    regime: str
    """``tpc`` (TPC stream), ``dma`` (DMA stream) or ``flops`` (MME)."""
    bytes_of: Callable[[str], float] | None
    derivation: str


def _embed_bytes(name: str) -> float:
    # out is (T, hidden) bf16; the table rows it reads are the same size
    t = _t(name)
    return 2.0 * t * HIDDEN * 2


def _engram_bytes(name: str) -> float:
    # out is (T, engram_heads, engram_head_dim) f32; the gather itself is host-side
    t = _t(name)
    return float(t * ENGRAM_HEADS * ENGRAM_HEAD_DIM * 4)


def _ffn_paged_bytes(name: str) -> float:
    # every routed expert is touched once per chunk (T*top slots over 384 experts ≥ 1 row each), so the MoE
    # reads the whole per-card expert set: 384 x 3 x 5120 x 2304 params, fp4 (0.5 B), over 4 cards, one layer
    per_layer = EXPERTS * 3 * HIDDEN * MOE_INTER * 0.5 / CARDS
    return per_layer


def _owner_store_bytes(name: str) -> float:
    # the cut layer's latent KV for the kept span: T x 1 kv head x 512 dim x K+V x 1 B (fp8)
    t = _t(name)
    return float(t * 2 * 512)


FLOORS: dict[str, Floor] = {
    "glm_embed": Floor(
        "tpc",
        _embed_bytes,
        "out (T, 5120) bf16 written + the same bytes of table rows read = 2*T*5120*2",
    ),
    "ds41_engram": Floor(
        "tpc",
        _engram_bytes,
        "out (T, 8 heads, 256) f32 written = T*8*256*4; the gather is host-side and not in this recipe",
    ),
    "ds41_ffn_paged": Floor(
        "dma",
        _ffn_paged_bytes,
        "all 384 routed experts touched per chunk: 384*3*5120*2304 params at 0.5 B (fp4) over 4 cards, "
        "one layer; T does not change it above T*top/384 >= 1 row per expert",
    ),
    "ds41_owner_store": Floor(
        "tpc",
        _owner_store_bytes,
        "kept-span latent KV write: T*1 kv head*512 dim*2 (K and V)*1 B (fp8)",
    ),
    # No formula derivable from the recipe name alone: the attention path's per-pass KV and activation
    # shapes, and the indexer's key counts, are not in the name. Listed so their silence is visible.
    "ds41_attn_pre": Floor("tpc", None, "UNKNOWN: per-pass q/k/v and KV shapes not in the name"),
    "ds41_attn_win": Floor("tpc", None, "UNKNOWN: window slot geometry not in the name"),
    "ds41_attn_out": Floor("tpc", None, "UNKNOWN: tile/block parameters not in the name"),
    "ds41_attn_out_tile": Floor("tpc", None, "UNKNOWN: tile/block parameters not in the name"),
    "ds41_attn_out_wo": Floor("tpc", None, "UNKNOWN: tile/block parameters not in the name"),
    "ds41_ix_chain": Floor("tpc", None, "UNKNOWN: indexer key counts and tile census not in the name"),
    "ds41_ix_gslot": Floor("tpc", None, "UNKNOWN: slot geometry not in the name"),
    "glm_sample": Floor("flops", None, "UNKNOWN: sampler work is not a bandwidth span"),
}

# Longest name first, so ``ds41_attn_out_tile`` is matched before ``ds41_attn_out``.
_FAMILIES = sorted(FLOORS, key=len, reverse=True)


def family(name: str) -> str:
    """The floor-table family of a recipe name (``ds41_ffn_paged_T2048`` -> ``ds41_ffn_paged``)."""
    for candidate in _FAMILIES:
        if name.startswith(candidate):
            return candidate
    return name


def floor(name: str) -> tuple[float, str, float] | None:
    """``(bytes, regime, envelope_bytes_per_s)`` for a recipe, or None when no floor is declared."""
    entry = FLOORS.get(family(name))
    if entry is None or entry.bytes_of is None:
        return None
    rate = {"tpc": TPC_BYTES_PER_S, "dma": DMA_BYTES_PER_S}.get(entry.regime, DMA_BYTES_PER_S)
    return entry.bytes_of(name), entry.regime, rate


def derivation(name: str) -> str:
    entry = FLOORS.get(family(name))
    return entry.derivation if entry else "no floor declared"
