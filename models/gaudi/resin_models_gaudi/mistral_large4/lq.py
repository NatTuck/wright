"""Graph builders for the ``lq_*`` int8 GEMVs (``kernels/lq``) over
planar GGUF weights: activation quantization (``lq_xq8`` / ``lq_xq8c``), the GEMV
(``lq_qmv8_<type>``) and its K-split sum (``lq_ksum``).

All tensors are flat: x ``(K * N,)`` f32 column-major (column n at ``offx + n * sx1``), the
output ``(ncol * M,)`` f32 (column c at ``c * M``), weights ``(bytes per matrix, matrices)``
u8, expert ids ``(64 * T,)`` i32 (token t's slot s at ``64 t + s``). ``dep`` is any small
tensor (the kernels' trailing input).
"""

from __future__ import annotations

import struct

from resin_models_gaudi.gguf.planar import QMV8, nmem_for, qmv8_split
from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import F32, U8

MAX_COLUMNS = 16
MAX_SLOTS = 64
COMPACT = ("Q8_0",)  # x in the compact layout (lq_xq8c): 8 steps per 256 B


def xq8(
    g: Graph,
    x: Tensor,
    K: int,
    N: int,
    dep: Tensor,
    offx: int = 0,
    sx1: int | None = None,
    compact: bool = False,
) -> tuple[Tensor, Tensor]:
    """int8 x per 32 steps -> (X8, XG)"""
    sx1 = K if sx1 is None else sx1
    x8 = g.tensor(((K // 8 if compact else K // 2) * 256 * N,), U8)
    xg = g.tensor((K // 32 * 256 * N,), F32)
    g.node(
        "rg_lq_xq8c_f32" if compact else "rg_lq_xq8_f32",
        [x, dep],
        [x8, xg],
        struct.pack("iiiii", nmem_for(N * -(-K // 64), 2), K, N, offx, sx1),
    )
    return x8, xg


def qmv8(
    g: Graph,
    qtype: str,
    W: Tensor,
    x8: Tensor,
    xg: Tensor,
    dep: Tensor,
    K: int,
    M: int,
    ncol: int,
    ids: Tensor | None = None,
    nu: int = 1,
    bcast: bool = False,
    out: Tensor | None = None,
    exact: bool = False,
) -> Tensor:
    """out[c * M + r] = W_c[r, :] . x_c. Dense (ids None): ncol x columns, one matrix.
    Experts: ncol = T * nu slots (slot s of token t uses matrix ids[64 t + s]); bcast: one x
    column per token, else one per slot. ``exact``: the K split of a one-token launch at any
    width."""
    has_ids = ids is not None
    flat = has_ids and bcast
    nks, items, nbx, nxg, nrb = qmv8_split(K, M, ncol, nu, flat, qtype)
    if exact:
        nks = qmv8_split(K, M, nu, nu, flat, qtype)[0]
    if has_ids:
        base = (K, M, nrb, ncol, nu, int(bcast), 1, 1 if bcast else nu, 0, 1, 64)
    else:
        base = (K, M, nrb, ncol, 1, 0, 0, 1, 0, 0, 0)
    prm = struct.pack(
        "iiiiiiiiiiiiiffff",
        nmem_for(nks * items, 1),
        *base,
        nks,
        1.0 / nbx,
        1.0 / nxg,
        1.0 / nrb,
        1.0 / nu,
    )
    y = out if out is not None else g.tensor((ncol * M,), F32)
    ins = [W, x8, xg, ids if ids is not None else x8, dep]
    guid = f"rg_lq_qmv8_{QMV8[qtype][0]}_f32"
    if nks == 1:
        g.node(guid, ins, [y], prm)
        return y
    part = g.tensor((nks * ncol * M,), F32)
    g.node(guid, ins, [part], prm)
    g.node(
        "rg_lq_ksum_f32",
        [part, dep],
        [y],
        struct.pack("iii", nmem_for(-(-(M * ncol) // 64), 4), M * ncol, nks),
    )
    return y


def gemv(
    g: Graph,
    qtype: str,
    W: Tensor,
    x: Tensor,
    dep: Tensor,
    K: int,
    M: int,
    ncol: int,
    *,
    ids: Tensor | None = None,
    nu: int = 1,
    bcast: bool = False,
    out: Tensor | None = None,
    offx: int = 0,
    sx1: int | None = None,
    exact: bool = False,
) -> Tensor:
    """xq8 + qmv8 (x (K * ncol,) f32 dense, or per token / slot as qmv8's ids / nu / bcast)"""
    N = ncol // nu if bcast else ncol
    if ncol > (MAX_SLOTS if ids is not None else MAX_COLUMNS):
        raise ValueError(f"lq GEMV of {ncol} columns exceeds the kernel limits")
    x8, xg = xq8(g, x, K, N, dep, offx=offx, sx1=sx1, compact=qtype in COMPACT)
    return qmv8(
        g,
        qtype,
        W,
        x8,
        xg,
        dep,
        K,
        M,
        ncol,
        ids=ids,
        nu=nu,
        bcast=bcast,
        out=out,
        exact=exact,
    )
