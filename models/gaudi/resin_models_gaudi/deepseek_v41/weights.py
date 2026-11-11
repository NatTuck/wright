"""DeepSeek-V4.1-Flash device weights for one tensor-parallel rank, from the MXFP8 / MXFP4 release.

Formats in the checkpoint:
- dense FP8: OCP E4M3 bytes (M, K) with one E8M0 scale per 32x32 block, ``name.scale`` (M/32, K/32);
- routed experts: E2M1 codes, two per byte with the low nibble first, (M, K/2), with one E8M0 scale per
  32 elements of a row, ``name.scale`` (M, K/32);
- Engram tables: OCP E4M3 bytes (rows, 256) with one E8M0 scale per 32 elements, (rows, 8).

On the card:
- dense FP8 weights are E4M3 with bias 7 (+-240) and one power-of-two scale per output row (``name``,
  ``name.s``), with K in tau order for the FP8 GEMMs (``mx8_rows``);
- routed experts keep their codes and scale bytes. Each rank holds a slot per expert: the w1 / w3 rows
  of its slice of the expert width, ordered [w1 0..256 | w3 0..256 | w1 256.. | w3 256..], and the
  matching K slice of w2 (``expert_slots``). The dequant kernel runs at one exponent k per tensor, and
  ``name.c`` holds 2^k for each output row;
- Engram tables keep their bytes; each rank holds a contiguous range of rows.

Rank r of n holds query heads and attention-sink entries [8r, 8r + 8) (n = 8), output group r of
``wo_a``, the matching input columns of ``wo_b``, a slice of every expert's width (routed and shared),
a range of each Engram table's rows, and a slice of the vocabulary. Replicated: wq_a, wkv, the norms, the
compressor, the indexer, the router and the hyper-connections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from resin.models.deepseek_v41.config import DeepSeekV41Config
from resin_models_gaudi.glm5_3_flash.weights import Safetensors
from resin_models_gaudi.qwen3_5.checkpoint import _POOL, kperm, tperm

# chunks of rows handed to the read pool for one Engram gather: enough to fill 32 workers four times
_ENGRAM_CHUNKS = 128
# Below this many rows the gather reads inline. The pool costs ~1.7 ms per call whatever the row
# count, so it pays for a prefill chunk (49,152 rows: 2.4 s pooled against 138 s inline, cold) and
# does not pay for a decode step (24 rows: 1.85 ms pooled against 0.15 ms inline, warm).
_ENGRAM_POOL_ROWS = 256

DTYPES = {
    "BF16": np.uint16,
    "F32": np.float32,
    "F8_E4M3": np.uint8,
    "F8_E8M0": np.uint8,
    "I8": np.uint8,
    "I64": np.int64,
}
FP8_MAX = 240.0
EXACT_SPREAD = 13  # the MXFP4 dequant is exact for blocks at most this many exponents below k's block


def reader(path) -> Safetensors:
    return Safetensors(path, DTYPES)


# ------------------------------------------------------------------ FP8 encodings
def _ocp_values() -> np.ndarray:
    """OCP E4M3 (bias 7, +-448; 0x7F / 0xFF are NaN, read as 0) for each byte."""
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * np.exp2(e - 10.0), m * np.exp2(-9.0))
    v = np.where((b & 0x7F) == 0x7F, 0.0, v)
    return np.where(b & 0x80, -v, v)


OCP = _ocp_values()


def inc_bytes(v: np.ndarray) -> np.ndarray:
    """Values -> E4M3 bias 7 bytes (+-240, subnormals to 2^-9), nearest with ties to even, saturating."""
    a = np.abs(v)
    e = np.clip(np.floor(np.log2(np.where(a > 0, a, 1.0))), -6, None).astype(np.int64)
    q = np.round(a / np.exp2(e - 3.0)).astype(np.int64)
    up = q == 16
    e, q = np.where(up, e + 1, e), np.where(up, 8, q)
    byte = np.where(q >= 8, ((e + 7) << 3) | (q - 8), q)
    byte = np.where((a == 0) | (byte == 0), 0, byte)
    byte = np.where(e > 7, 0x77, np.minimum(byte, 0x77))
    return (byte | np.where((v < 0) & (byte != 0), 0x80, 0)).astype(np.uint8)


def inc_values(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.int64)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * np.exp2(e - 10.0), m * np.exp2(-9.0))
    return np.where(b & 0x80, -v, v)


_SHIFT_LO, _SHIFT_HI = -48, 8
_SHIFTS = np.arange(_SHIFT_LO, _SHIFT_HI + 1)
# _LUT[d - _SHIFT_LO, b]: the bias-7 byte of OCP byte b times 2^d
_LUT = np.stack([inc_bytes(OCP * np.exp2(float(d))) for d in _SHIFTS]).astype(np.uint8)
_EXACT = np.stack(
    [inc_values(_LUT[i]) == OCP * np.exp2(float(d)) for i, d in enumerate(_SHIFTS)]
)


@dataclass(frozen=True, slots=True)
class Mx8Rows:
    bytes: np.ndarray
    """(M, K) E4M3 bias 7, K in tau order (natural with ``tau=False``); with tau, K padded with zeros to a
    multiple of 256"""
    row_scales: np.ndarray
    """(M,) f32 powers of two"""
    inexact: int
    """elements that the conversion rounded"""


def mx8_rows(
    w: np.ndarray, scales: np.ndarray, *, tau: bool = True, count: bool = True
) -> Mx8Rows:
    """OCP E4M3 (M, K) with E8M0 scales per 32x32 block (M/32, K/32), or per 32 elements of a row (M, K/32), ->
    bias-7 bytes with one power-of-two scale per row.

    The row scale 2^r is the smallest power of two that puts the row's largest value at or below 240, so
    each row's part of a block is a shift d = S - 127 - r of its codes, and a byte converts through one
    table. Elements far below the row's largest value round (counted in ``inexact``, unless not ``count``)."""
    M, K = w.shape
    if M % 32 or K % 32 or scales.shape not in ((M // 32, K // 32), (M, K // 32)):
        raise ValueError(f"MXFP8 shape {w.shape} with scales {scales.shape}")
    # each row's block maxima: E4M3 magnitudes order as their bytes do (0x7F, NaN, reads as 0)
    mag = w & 0x7F
    mag[mag == 0x7F] = 0
    blocks = OCP[mag.reshape(M, K // 32, 32).max(axis=2)]  # (M, K/32)
    exponent = scales.astype(np.int64)
    exponent = (exponent if scales.shape[0] == M else exponent.repeat(32, 0)) - 127
    with np.errstate(divide="ignore"):
        top = np.where(
            blocks > 0, np.log2(np.where(blocks > 0, blocks, 1.0)) + exponent, -np.inf
        )
    row_top = top.max(axis=1)  # (M,): log2 of the row's largest value
    r = np.where(
        np.isfinite(row_top), np.ceil(row_top - np.log2(FP8_MAX)), -100
    ).astype(np.int64)
    d = np.clip(exponent - r[:, None], _SHIFT_LO, _SHIFT_HI)  # (M, K/32)
    out = np.empty((M, K), np.uint8)
    inexact = np.zeros(M // 32, np.int64)

    shift = (d - _SHIFT_LO).astype(
        np.uint16
    ) * 256  # (a table row: under 2^16 with its byte)

    def one(b: int) -> None:
        rows = slice(32 * b, 32 * b + 32)
        idx = shift[rows].repeat(32, axis=1) + w[rows]
        out[rows] = _LUT.reshape(-1)[idx]
        if count:
            inexact[b] = int(np.count_nonzero(~_EXACT.reshape(-1)[idx]))

    list(_POOL.map(one, range(M // 32)))
    row_scales = np.exp2(r.astype(np.float64)).astype(np.float32)
    if (
        tau and K % 256
    ):  # the FP8 activations pad K with zeros to whole tau blocks: so does the weight
        out = np.pad(out, ((0, 0), (0, -K % 256)))
    return Mx8Rows(tperm(out) if tau else out, row_scales, int(inexact.sum()))


def mx8_f64(w: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """The dequantized MXFP8 tensor."""
    s = np.exp2(scales.astype(np.float64) - 127).repeat(32, 0).repeat(32, 1)
    return OCP[w] * s[: w.shape[0], : w.shape[1]]


# ------------------------------------------------------------------ rank slices
@dataclass(frozen=True, slots=True)
class RankSlices:
    heads: slice
    """query heads (rows of wq_b in units of head_dim, entries of attn_sink)"""
    group: slice
    """output groups of wo_a (rows in units of o_lora_rank); columns of wo_b in the same units"""
    expert: slice
    """each expert's intermediate rows (routed and shared)"""
    vocab: slice

    @classmethod
    def of(cls, cfg: DeepSeekV41Config, rank: int, nranks: int) -> RankSlices:
        cfg.validate_tensor_parallel(nranks)
        if not 0 <= rank < nranks:
            raise ValueError(f"rank {rank} of {nranks}")

        def part(n: int) -> slice:
            s = n // nranks
            return slice(rank * s, (rank + 1) * s)

        vocab = -(-cfg.vocab_size // nranks)
        return cls(
            heads=part(cfg.num_attention_heads),
            group=part(cfg.o_groups),
            expert=part(cfg.moe_intermediate_size),
            vocab=slice(rank * vocab, min(cfg.vocab_size, (rank + 1) * vocab)),
        )


def scaled(s: slice, unit: int) -> slice:
    return slice(s.start * unit, s.stop * unit)


# ------------------------------------------------------------------ routed experts
@dataclass(frozen=True, slots=True)
class ExpertSlots:
    codes13: np.ndarray
    """(E, 2w * H / 2) u8: per expert, rows [w1 0..w0 | w3 0..w0 | w1 w0..w | w3 w0..w] of the rank's
    slice (w0 = w rounded down to 128), each row H inputs"""
    scales13: np.ndarray
    """(E, 2w * H / 32) u8 E8M0, the same rows"""
    codes2: np.ndarray
    """(E, H * w / 2) u8: per expert, the H rows of w2, each the rank's w inputs"""
    scales2: np.ndarray
    """(E, H * w / 32) u8"""
    k13: np.ndarray
    """(E,) i32: each expert's dequant exponent"""
    k2: np.ndarray


def slot_rows(width: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of w1 and w3 (indices into the rank's slice) for slot rows [w1 main | w3 main | w1 tail | w3
    tail]: (source matrix 0 = w1 / 1 = w3, row)."""
    w0 = width // 128 * 128
    main, tail = np.arange(w0), np.arange(w0, width)
    which = np.concatenate(
        [np.zeros(w0), np.ones(w0), np.zeros(width - w0), np.ones(width - w0)]
    )
    rows = np.concatenate([main, main, tail, tail])
    return which.astype(np.int64), rows


def kernel_k(scales: np.ndarray) -> int:
    """The dequant exponent for a tensor: k = max S - 127 - 5, checked to keep every block exact."""
    hi, lo = int(scales.max()), int(scales.min())
    if hi == 0xFF:
        raise ValueError("an expert scale is 0xFF (NaN)")
    if hi - lo > EXACT_SPREAD:
        raise ValueError(
            f"expert scales span {hi - lo} exponents, more than {EXACT_SPREAD}"
        )
    return hi - 132


def expert_slots(
    src: Safetensors, prefix: str, experts: int, width: slice, hidden: int
) -> ExpertSlots:
    """Each routed expert's slot for one rank, from ``{prefix}.experts.{e}.w1/w2/w3``."""
    w = width.stop - width.start
    if width.start % 32 or w % 32:
        raise ValueError(f"expert slice {width} is not in 32-element blocks")
    which, rows = slot_rows(w)
    codes13 = np.empty((experts, 2 * w, hidden // 2), np.uint8)
    scales13 = np.empty((experts, 2 * w, hidden // 32), np.uint8)
    codes2 = np.empty((experts, hidden, w // 2), np.uint8)
    scales2 = np.empty((experts, hidden, w // 32), np.uint8)
    c2, s2 = (
        slice(width.start // 2, width.stop // 2),
        slice(width.start // 32, width.stop // 32),
    )

    def one(e: int) -> None:
        name = f"{prefix}.experts.{e}"
        for m, proj in enumerate(("w1", "w3")):
            pick = rows[which == m]
            codes13[e, which == m] = src.read(f"{name}.{proj}.weight", width)[pick]
            scales13[e, which == m] = src.read(f"{name}.{proj}.scale", width)[pick]
        codes2[e] = src.read(f"{name}.w2.weight")[:, c2]
        scales2[e] = src.read(f"{name}.w2.scale")[:, s2]

    list(_POOL.map(one, range(experts)))
    return ExpertSlots(
        codes13.reshape(experts, -1),
        scales13.reshape(experts, -1),
        codes2.reshape(experts, -1),
        scales2.reshape(experts, -1),
        expert_k(scales13),
        expert_k(scales2),
    )


def expert_k(scales: np.ndarray) -> np.ndarray:
    """Each expert's dequant exponent (E,): one for the tensor when its blocks allow it, else each expert's own
    (the DSpark stages' wider spreads)."""
    try:
        return np.full(len(scales), kernel_k(scales), np.int32)
    except ValueError:
        return np.array([kernel_k(x) for x in scales], np.int32)


# ------------------------------------------------------------------ Engram tables
def table_rows(rows: int, rank: int, nranks: int) -> slice:
    """The rank's contiguous range of an Engram table's rows."""
    per = -(-rows // nranks)
    return slice(min(rows, rank * per), min(rows, (rank + 1) * per))


def engram_table(src: Safetensors, layer: int, rows: slice, chunk: int = 1 << 17):
    """A rank's rows of an Engram table converted for ``eg_gather``, in chunks: (first row relative to
    ``rows.start``, bias-7 bytes (n, 256), power-of-two row scales (n,))."""
    name = f"layers.{layer}.engram.embed"
    for a in range(rows.start, rows.stop, chunk):
        part = slice(a, min(rows.stop, a + chunk))
        w, s = src.read(f"{name}.weight", part), src.read(f"{name}.scale", part)
        n = len(w)
        pad = -n % 32
        if pad:
            w = np.concatenate([w, np.zeros((pad, w.shape[1]), np.uint8)])
            s = np.concatenate([s, np.full((pad, s.shape[1]), 127, np.uint8)])
        m = mx8_rows(w, s, tau=False, count=False)
        yield a - rows.start, m.bytes[:n], m.row_scales[:n]


# ------------------------------------------------------------------ one layer
def _f32(src: Safetensors, name: str) -> np.ndarray:
    a = src.read(name)
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(a, np.float32)


def _mx8(
    src: Safetensors, name: str, rows: slice | None = None, cols: slice | None = None
) -> Mx8Rows:
    w, s = src.read(f"{name}.weight"), src.read(f"{name}.scale")
    if rows is not None:
        w, s = w[rows], s[rows.start // 32 : rows.stop // 32]
    if cols is not None:
        w, s = w[:, cols], s[:, cols.start // 32 : cols.stop // 32]
    return mx8_rows(np.ascontiguousarray(w), np.ascontiguousarray(s))


def _put_mx8(out: dict, name: str, parts: list[Mx8Rows]) -> None:
    out[name] = (
        np.concatenate([p.bytes for p in parts]) if len(parts) > 1 else parts[0].bytes
    )
    out[name + ".s"] = np.concatenate([p.row_scales for p in parts])


def _hc(src: Safetensors, prefix: str, site: str, tag: str, out: dict) -> None:
    fn = _f32(src, f"{prefix}.hc_{site}_fn")
    out[f"{tag}_fn"] = np.concatenate(
        [fn, np.zeros((32 - fn.shape[0], fn.shape[1]), np.float32)]
    )
    out[f"{tag}_base"] = _f32(src, f"{prefix}.hc_{site}_base")
    out[f"{tag}_scale"] = _f32(src, f"{prefix}.hc_{site}_scale")


def prepare_layer(
    src: Safetensors,
    cfg: DeepSeekV41Config,
    layer: int,
    sl: RankSlices,
    prefix: str | None = None,
    experts: int | None = None,
) -> dict[str, np.ndarray]:
    """One backbone layer's device tensors for one rank, by the graphs' input names (window attention and
    the MoE; compressor, indexer and Engram tensors are separate); or a DSpark stage's (``prefix`` mtp.k,
    ``experts`` routed experts)."""
    p = prefix or f"layers.{layer}"
    D, H, w = cfg.head_dim, cfg.hidden_size, sl.expert.stop - sl.expert.start
    out: dict[str, np.ndarray] = {}
    _put_mx8(out, "w_a", [_mx8(src, f"{p}.attn.wq_a"), _mx8(src, f"{p}.attn.wkv")])
    _put_mx8(out, "wq_b", [_mx8(src, f"{p}.attn.wq_b", rows=scaled(sl.heads, D))])
    for g in range(sl.group.start, sl.group.stop):
        rows = slice(g * cfg.o_lora_rank, (g + 1) * cfg.o_lora_rank)
        _put_mx8(
            out, f"wo_a.{g - sl.group.start}", [_mx8(src, f"{p}.attn.wo_a", rows=rows)]
        )
    _put_mx8(
        out,
        "wo_b",
        [_mx8(src, f"{p}.attn.wo_b", cols=scaled(sl.group, cfg.o_lora_rank))],
    )
    out["attn_sink"] = _f32(src, f"{p}.attn.attn_sink")[sl.heads]
    for name, key in (("q_norm", "attn.q_norm"), ("kv_norm", "attn.kv_norm"), ("attn_norm", "attn_norm"),
                      ("ffn_norm", "ffn_norm")):  # fmt: skip
        out[name] = _f32(src, f"{p}.{key}.weight")
    _hc(src, p, "attn", "hca", out)
    _hc(src, p, "ffn", "hcf", out)
    out["router"] = kperm(src.read(f"{p}.ffn.gate.weight"))
    out["router_bias"] = _f32(src, f"{p}.ffn.gate.bias")
    sh = f"{p}.ffn.shared_experts"
    _put_mx8(
        out,
        "sh_w13",
        [_mx8(src, f"{sh}.w1", rows=sl.expert), _mx8(src, f"{sh}.w3", rows=sl.expert)],
    )
    _put_mx8(out, "sh_w2", [_mx8(src, f"{sh}.w2", cols=sl.expert)])
    E = experts or cfg.num_routed_experts
    slots = expert_slots(src, f"{p}.ffn", E, sl.expert, H)
    out["x13"], out["x13.s"], out["x2"], out["x2.s"] = (
        slots.codes13,
        slots.scales13,
        slots.codes2,
        slots.scales2,
    )
    out["x13.k"], out["x2.k"] = slots.k13, slots.k2
    out["x13.c"] = np.repeat(np.exp2(slots.k13.astype(np.float32))[:, None], 2 * w, 1)
    out["x2.c"] = np.repeat(np.exp2(slots.k2.astype(np.float32))[:, None], H, 1)
    return out


def prepare_draft_stage(
    src: Safetensors, cfg: DeepSeekV41Config, stage: int, sl: RankSlices
) -> dict[str, np.ndarray]:
    """DSpark stage ``stage``'s device tensors for one rank: the block (window attention, the MoE over the
    stages' routed experts), stage 0's main_proj and main_norm, the last stage's norm and its Markov head
    (replicated: its embedding and head, (vocab, rank) bf16 each)."""
    p = f"mtp.{stage}"
    out = prepare_layer(
        src, cfg, cfg.num_hidden_layers + stage, sl, p, cfg.dspark_n_routed_experts
    )
    _put_mx8(
        out, "wkv", [_mx8(src, f"{p}.attn.wkv")]
    )  # the window keys of the target's positions
    if stage == 0:
        _put_mx8(out, "main_proj", [_mx8(src, f"{p}.main_proj")])
        out["main_norm"] = _f32(src, f"{p}.main_norm.weight")
    if stage == cfg.num_nextn_predict_layers - 1:
        out["norm"] = _f32(src, f"{p}.norm.weight")
        out["mk_embed"] = src.read(f"{p}.markov_head.embed.weight")
        out["mk_head"] = src.read(
            f"{p}.markov_head.head.weight"
        )  # natural order: its input is an embedding row
    return out


def prepare_engram(src: Safetensors, layer: int) -> dict[str, np.ndarray]:
    """An Engram layer's projection and gate weights (the table is separate): ``eg_wkv`` (5H, 6144) FP8 and
    ``eg_w`` = q_weight * k_weight (4H)."""
    p = f"layers.{layer}.engram"
    out: dict[str, np.ndarray] = {}
    _put_mx8(out, "eg_wkv", [_mx8(src, f"{p}.wkv")])
    out["eg_w"] = (_f32(src, f"{p}.q_weight") * _f32(src, f"{p}.k_weight")).reshape(-1)
    return out


def split64(x: np.ndarray) -> np.ndarray:
    """(..., 2) i32: the high and low 32 bits of non-negative int64 values."""
    u = np.asarray(x, np.int64).view(np.uint64)
    return (
        np.stack([u >> np.uint64(32), u & np.uint64(0xFFFFFFFF)], -1)
        .astype(np.uint32)
        .view(np.int32)
    )


def engram_hash_tables(h, k: int) -> dict[str, np.ndarray]:
    """``eg_hash``'s tables of Engram layer ``k`` (its index in engram_layer_ids) from an ``EngramHash``:
    ``eg_a0`` (V, 2) i32 every raw token's product with the first multiplier as (hi, lo), and ``eg_hp`` (4, 64)
    i32 per lane l = 8 g + head: p, row offset, 2^32 mod p, f32 bits of 1 / p (lanes past 24 repeat lane 0)."""
    primes, offsets = h.primes[k].reshape(-1), h.offsets[k].reshape(-1)
    lanes = np.where(np.arange(64) < len(primes), np.arange(64), 0)
    p, off = primes[lanes], offsets[lanes]
    if p.max() >= 1 << 31 or off.max() >= 1 << 31:
        raise ValueError("Engram primes and row offsets must fit in 31 bits")
    inv = (1.0 / p.astype(np.float64)).astype(np.float32).view(np.int32)
    hp = np.stack([p, off, (1 << 32) % p, inv]).astype(np.int32)
    return {"eg_a0": split64(h.newest()[k]), "eg_hp": hp}


def engram_rows(src: Safetensors, layer: int, rows: np.ndarray) -> np.ndarray:
    """Rows of an Engram table, dequantized (n, 256) f32.

    The row ids are hash outputs, so neighbouring rows are not adjacent in the file: a 2,048-token
    chunk's rows measured 1.00 rows per contiguous run, and an iovec array cannot serve them. What
    is left is one random 264-byte read per row, which at queue depth 1 costs the disk's full
    latency - with the page cache dropped, a chunk's gather took 138 s. Reading the rows from the
    thread pool gives the reads queue depth (2.0 s for the same chunk) and the dequant then runs
    over the whole block at once instead of a row at a time."""
    name = f"layers.{layer}.engram.embed"
    rows = np.asarray(rows, np.int64).reshape(-1)
    w = _engram_bytes(src, f"{name}.weight", rows, 256)
    s = _engram_bytes(src, f"{name}.scale", rows, 8)
    out = OCP[w] * np.exp2(s.astype(np.float32) - 127).repeat(32, axis=1)
    return out.astype(np.float32)


def _engram_bytes(src: Safetensors, name: str, rows: np.ndarray, width: int) -> np.ndarray:
    """``rows`` of one Engram tensor as raw bytes: one preadv per row, spread over the thread pool."""
    fd, entry, base = src._entry(name)
    start = base + entry["data_offsets"][0]
    out = np.empty((len(rows), width), np.uint8)
    view = memoryview(out.reshape(-1))

    def read(bounds: tuple[int, int]) -> None:
        for i in range(*bounds):
            os.preadv(fd, [view[i * width : (i + 1) * width]], start + int(rows[i]) * width)

    if len(rows) >= _ENGRAM_POOL_ROWS:
        step = -(-len(rows) // min(len(rows), _ENGRAM_CHUNKS))
        list(
            _POOL.map(
                read, ((i, min(i + step, len(rows))) for i in range(0, len(rows), step))
            )
        )
    elif len(rows):
        read((0, len(rows)))
    return out


def prepare_compressed(
    src: Safetensors, cfg: DeepSeekV41Config, layer: int
) -> dict[str, np.ndarray]:
    """A compressed layer's compressor (KV owners) and indexer (index sources) weights, replicated on every rank.

    - ``cmp_wkv`` / ``cmp_wgate`` (512, H) f32, ``cmp_norm`` (512): the compressor on the attention input (f32
      projections, as the reference promotes them);
    - ``ix_wk`` (128, 512) bf16 in natural order, ``ix_knorm`` (128): index keys from the normed latent;
    - ``ix_wq_b`` (index heads x 128, q_lora) FP8, ``ix_wproj`` (index heads, H) bf16 in sigma order (the
      collapse's bf16 output): the index queries and their head weights."""

    p = f"layers.{layer}.attn"
    out: dict[str, np.ndarray] = {}
    if layer in cfg.kv_source_layers:
        out["cmp_wkv"] = _f32(src, f"{p}.compressor.wkv.weight")
        if cfg.compress_ratios[layer] > 1:
            out["cmp_wgate"] = _f32(src, f"{p}.compressor.wgate.weight")
        out["cmp_norm"] = _f32(src, f"{p}.compressor.norm.weight")
        out["ix_wk"] = src.read(f"{p}.indexer.wk.weight")
        out["ix_knorm"] = _f32(src, f"{p}.indexer.k_norm.weight")
    if layer in cfg.index_source_layers:
        _put_mx8(out, "ix_wq_b", [_mx8(src, f"{p}.indexer.wq_b")])
        out["ix_wproj"] = kperm(src.read(f"{p}.indexer.weights_proj.weight"))
    return out
