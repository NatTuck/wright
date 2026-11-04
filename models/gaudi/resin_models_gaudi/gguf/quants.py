"""GGUF block formats the Gaudi GEMVs consume (Q4_K, Q5_K, Q8_0): geometry and numpy reference
dequantization with ggml-quants.c semantics. The threaded C versions live in
``synapse/native/pack.c`` (``resin_models_gaudi.gguf.pack``); these are their test oracles."""

from __future__ import annotations

import numpy as np

# qtype -> (pack.c type code, block elements, block bytes)
LQ_TYPES = {"Q4_K": (0, 256, 144), "Q5_K": (1, 256, 176), "Q8_0": (2, 32, 34)}


def check_qtype(name: str, qtype: str) -> None:
    if qtype not in LQ_TYPES:
        raise ValueError(
            f"{name}: {qtype} has no Gaudi GEMV (supported: {', '.join(LQ_TYPES)})"
        )


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32)[..., 0]


def _scale_min_k4(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """the 8 (6-bit scale, 6-bit min) pairs of a K-quant superblock: sc [..., 12] -> [..., 8] x 2"""
    sc = sc.astype(np.uint8)
    d = np.empty(sc.shape[:-1] + (8,), np.uint8)
    m = np.empty_like(d)
    d[..., :4] = sc[..., 0:4] & 63
    m[..., :4] = sc[..., 4:8] & 63
    d[..., 4:] = (sc[..., 8:12] & 0xF) | ((sc[..., 0:4] >> 6) << 4)
    m[..., 4:] = (sc[..., 8:12] >> 4) | ((sc[..., 4:8] >> 6) << 4)
    return d, m


def dequantize(blocks: np.ndarray, qtype: str) -> np.ndarray:
    """native rows [N, row_bytes] u8 -> f32 [N, K]"""
    check_qtype("dequantize", qtype)
    size = LQ_TYPES[qtype][2]
    B = np.asarray(blocks, np.uint8)
    N = B.shape[0]
    b = B.reshape(N, -1, size)
    if qtype == "Q8_0":
        d = _f16(b[..., 0:2])
        q = b[..., 2:].view(np.int8).astype(np.float32)
        return (d[..., None] * q).reshape(N, -1)
    d, dmin = _f16(b[..., 0:2]), _f16(b[..., 2:4])
    s, m = _scale_min_k4(b[..., 4:16])
    ql = b[..., 16:144] if qtype == "Q4_K" else b[..., 48:176]
    ql = ql.reshape(N, -1, 4, 32)  # [n, sb, chunk of 64, 32]
    q = np.stack([ql & 0xF, ql >> 4], 3).astype(np.int32)  # [n, sb, chunk, half, 32]
    if qtype == "Q5_K":
        qh = b[..., 16:48]  # bit 2c (+1 for the high half) of byte l
        shift = (2 * np.arange(4)[:, None] + np.arange(2)[None, :]).astype(np.uint8)
        hi = (qh[:, :, None, None, :] >> shift[None, None, :, :, None]) & 1
        q = q + 16 * hi.astype(np.int32)
    sub = np.arange(8).reshape(4, 2)  # sub-block of (chunk, half)
    scale = d[..., None, None] * s[:, :, sub].astype(np.float32)
    offset = dmin[..., None, None] * m[:, :, sub].astype(np.float32)
    return (scale[..., None] * q.astype(np.float32) - offset[..., None]).reshape(N, -1)
