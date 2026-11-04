"""The planar weight layout of the ``lq_qmv8`` GEMVs, its numpy model and the GEMV's K split.

A matrix is ``[row block][superblock][SBB]`` bytes; rows past the matrix end are zero.

The C packer (``pack.c lq_planar_rb``) is the production path; ``gguf_planar`` here is its
test oracle.
"""

from __future__ import annotations

import numpy as np

# layout per qtype: (SB, nf16, nbp, code planes: 4 = 4-bit, 5 = 4-bit + 1-bit, 8 = bytes)
LAYOUT = {"Q4_K": (256, 2, 12, 4), "Q5_K": (256, 2, 12, 5), "Q8_0": (32, 1, 0, 8)}

# GEMV GUID suffix, weight group and the two ``qmv8_split`` weights
QMV8 = {
    "Q4_K": ("q4k", 32, 300, 100),
    "Q5_K": ("q5k", 32, 700, 100),
    "Q8_0": ("q8_0", 32, 385, 0),
}


def superblock_bytes(qtype: str) -> int:
    sb, nf16, nbp, planes = LAYOUT[qtype]
    code = {4: sb // 4 * 256, 5: sb // 4 * 256 + sb // 16 * 256, 8: sb // 2 * 256}[
        planes
    ]
    return nf16 * 256 + nbp * 128 + code


def size(qtype: str, M: int, K: int) -> int:
    """bytes of an M x K planar matrix"""
    return -(-M // 128) * (K // LAYOUT[qtype][0]) * superblock_bytes(qtype)


def _plane(codes: np.ndarray, sb: int, w: int, sbit: int) -> np.ndarray:
    M, K = codes.shape
    n, mask = 8 // w, (1 << w) - 1
    nrb, nsb = M // 128, K // sb
    c = ((codes.astype(np.int32) >> sbit) & mask).reshape(
        nrb, 2, 64, nsb, sb // (2 * n), n, 2
    )
    b = np.zeros(c.shape[:5] + (2,), np.int32)
    for t in range(n):
        b |= c[..., t, :] << (w * t)
    return b.astype(np.uint8).transpose(0, 3, 4, 2, 1, 5).reshape(nrb, nsb, -1)


def _planar(codes, bp, f16, sb, planes):
    M, K = codes.shape
    nrb, nsb = M // 128, K // sb
    codes = np.asarray(codes).view(np.uint8)
    if planes == 8:
        cp = [_plane(codes, sb, 8, 0)]
    else:
        cp = [_plane(codes, sb, 4, 0)] + (
            [_plane(codes, sb, 1, 4)] if planes == 5 else []
        )
    parts = []
    if f16 is not None:
        f = np.asarray(f16, np.uint16).reshape(nrb, 128, nsb, -1).transpose(0, 2, 3, 1)
        parts.append(np.ascontiguousarray(f).view(np.uint8).reshape(nrb, nsb, -1))
    if bp is not None:
        p = np.asarray(bp, np.uint8).reshape(nrb, 128, nsb, -1).transpose(0, 2, 3, 1)
        parts.append(p.reshape(nrb, nsb, -1))
    return np.ascontiguousarray(np.concatenate(parts + cp, 2))


def gguf_planar(blocks: np.ndarray, qtype: str) -> np.ndarray:
    """native rows [M, row_bytes] (Q4_K / Q5_K / Q8_0) -> [ceil(M/128), K/SB, SBB] u8"""
    B = np.asarray(blocks, np.uint8)
    M = B.shape[0]
    pad = -M % 128
    if pad:
        B = np.concatenate([B, np.zeros((pad, B.shape[1]), np.uint8)])
    Mp = B.shape[0]
    if qtype == "Q8_0":
        b = B.reshape(Mp, -1, 34)
        f16 = np.ascontiguousarray(b[:, :, :2]).view(np.uint16)
        return _planar(b[:, :, 2:].reshape(Mp, -1), None, f16, 32, 8)
    size_ = {"Q4_K": 144, "Q5_K": 176}[qtype]
    b = B.reshape(Mp, -1, size_)
    e = np.arange(256)
    chunk, r = e // 64, e % 64
    by = (16 if qtype == "Q4_K" else 48) + chunk * 32 + r % 32
    sh = np.where(r < 32, 0, 4).astype(np.uint8)
    codes = (b[:, :, by] >> sh) & 15
    if qtype == "Q5_K":
        hb, hs = 16 + r % 32, (2 * chunk + (r >= 32)).astype(np.uint8)
        codes |= ((b[:, :, hb] >> hs) & 1) << 4
    f16 = np.ascontiguousarray(b[:, :, :4]).view(np.uint16)
    planes = 5 if qtype == "Q5_K" else 4
    return _planar(codes.reshape(Mp, -1), b[:, :, 4:16], f16, 256, planes)


def nmem_for(work: int, per: int = 16) -> int:
    return max(1, min(24, -(-work // per)))


def qmv8_split(K: int, M: int, ncol: int, nu: int, flat: bool, qtype: str):
    """K splits of an lq_qmv8 node -> (nks, items, nbx, nxg, nrb)"""
    sb, _, _, planes = LAYOUT[qtype]
    _, g, csb, cpar = QMV8[qtype]
    ku, nrb, rb = max(g, 32), -(-M // 128), (2 if planes == 5 else 4)
    nkc = K // ku
    R = nu * nrb if flat else nrb
    nbx, nxg = -(-R // rb), (ncol // nu if flat else ncol)
    items = nxg * nbx
    best, nks = None, 1
    for n in range(1, min(nkc, 64) + 1):
        per = -(-nkc // n)
        if n > 1 and (n - 1) * per >= nkc:
            continue
        nsb = 0
        for k in range(n):
            a, b = k * per * ku, min(nkc, (k + 1) * per) * ku
            nsb = max(nsb, (b - 1) // sb - a // sb + 1)
        t = -(-(items * n) // 24) * (
            rb * (per * ku * csb // 256 + nsb * cpar) + 600
        ) + (2000 if n > 1 else 0)
        if best is None or t < best:
            best, nks = t, n
    return nks, items, nbx, nxg, nrb
