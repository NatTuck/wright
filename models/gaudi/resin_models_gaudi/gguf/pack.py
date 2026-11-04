"""Threaded host packing of GGUF weights through ``synapse/native/pack.c``: native blocks into
the ``lq_qmv8`` planar layout, and dequantization. ctypes releases the GIL, so one thread pool
packs row blocks (and experts) in parallel; experts are read with pread per expert, never
through a memory map."""

from __future__ import annotations

import ctypes as C
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import numpy as np

from resin_models_gaudi.gguf.planar import LAYOUT, superblock_bytes
from resin_models_gaudi.gguf.quants import LQ_TYPES, check_qtype
from resin_models_gaudi.gguf.reader import GgufModel

_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            threads = int(os.environ.get("RESIN_GAUDI_LOAD_THREADS", "32"))
            _POOL = ThreadPoolExecutor(threads, thread_name_prefix="gguf-pack")
        return _POOL


@cache
def _lib() -> C.CDLL:
    from resin_models_gaudi.synapse.build import build_pack

    lib = C.CDLL(str(build_pack()))
    lib.lq_planar_rb.restype = C.c_long
    lib.lq_planar_rb.argtypes = [
        C.c_int,
        C.c_void_p,
        C.c_long,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_void_p,
    ]
    lib.gguf_dequant_rows.restype = C.c_long
    lib.gguf_dequant_rows.argtypes = [
        C.c_int,
        C.c_void_p,
        C.c_long,
        C.c_int,
        C.c_int,
        C.c_void_p,
    ]
    return lib


def _rows_view(B: np.ndarray) -> np.ndarray:
    if B.dtype != np.uint8 or B.ndim != 2 or B.strides[1] != 1:
        raise ValueError("expected u8 rows [N, row_bytes] with contiguous rows")
    return B


def dequantize(blocks: np.ndarray, qtype: str) -> np.ndarray:
    """native rows [N, row_bytes] -> f32 [N, K] (ggml semantics; bit-exact with
    ``quants.dequantize``)"""
    check_qtype("dequantize", qtype)
    B = _rows_view(blocks)
    code, elements, size = LQ_TYPES[qtype]
    n, K = B.shape[0], B.shape[1] // size * elements
    out = np.empty((n, K), np.float32)
    if n == 0:
        return out
    f, step = _lib().gguf_dequant_rows, max(1, -(-n // 128))

    def one(r0: int) -> None:
        r1 = min(n, r0 + step)
        f(code, B[r0:].ctypes.data, B.strides[0], r1 - r0, K, out[r0].ctypes.data)

    list(pool().map(one, range(0, n, step)))
    return out


def _geometry(qtype: str, M_all: int, nsb_all: int, parts):
    """[(rows, superblocks, M, nsb, byte offset)] of the planar slices ``parts`` and their total
    bytes per matrix"""
    geo, off = [], 0
    for rows, sbs in parts:
        rows, sbs = rows or slice(0, M_all), sbs or slice(0, nsb_all)
        M, nsb = rows.stop - rows.start, sbs.stop - sbs.start
        geo.append((rows, sbs, M, nsb, off))
        off += -(-M // 128) * nsb * superblock_bytes(qtype)
    return geo, off


def planar(
    blocks: np.ndarray,
    qtype: str,
    rows: slice | None = None,
    superblocks: slice | None = None,
) -> np.ndarray:
    """native rows [M_all, row_bytes] -> [ceil(M/128), nsb, SBB] u8 for the row slice and
    superblock (input) slice; rows past the slice end are zero."""
    check_qtype("planar", qtype)
    B = _rows_view(blocks)
    code, _, size = LQ_TYPES[qtype]
    SBB = superblock_bytes(qtype)
    rows = rows or slice(0, B.shape[0])
    sbs = superblocks or slice(0, B.shape[1] // size)
    M, nsb = rows.stop - rows.start, sbs.stop - sbs.start
    nrb = -(-M // 128)
    out = np.empty((nrb, nsb, SBB), np.uint8)
    f, base = _lib().lq_planar_rb, B[rows.start :].ctypes.data

    def one(rb: int) -> None:
        if f(
            code, base, B.strides[0], M, 128 * rb, sbs.start, nsb, out[rb].ctypes.data
        ):
            raise RuntimeError(f"lq_planar_rb failed for {qtype}")

    list(pool().map(one, range(nrb)))
    return out


def planar_tensor(
    model: GgufModel,
    name: str,
    rows: slice | None = None,
    inputs: slice | None = None,
) -> np.ndarray:
    """A 2-D GGUF matrix's planar bytes (flat) for a row slice and an input-column slice
    (whole superblocks), read with pread."""
    t = model.tensor(name)
    check_qtype(name, t.qtype)
    sbs = None
    if inputs is not None:
        sb = LAYOUT[t.qtype][0]
        if inputs.start % sb or inputs.stop % sb:
            raise ValueError(f"{name}: input slice {inputs} is not whole superblocks")
        sbs = slice(inputs.start // sb, inputs.stop // sb)
    B = model.read_rows(name, rows)
    return planar(B, t.qtype, None, sbs).reshape(-1)


def planar_experts(
    model: GgufModel,
    name: str,
    parts: Sequence[tuple[slice | None, slice | None]],
) -> np.ndarray:
    """Every expert of a stacked [E, M_all, K] GGUF tensor -> [E, bytes]: per expert the
    concatenated planar slices ``parts = [(row slice, input slice), ...]`` (input slices in
    whole superblocks). Each expert's needed rows are read with one pread into a
    thread-local buffer."""
    t = model.tensor(name)
    check_qtype(name, t.qtype)
    if len(t.shape) != 3:
        raise ValueError(f"{name}: expected stacked experts, got {t.shape}")
    E, M_all, K = t.shape
    code = LQ_TYPES[t.qtype][0]
    sb, rb_bytes = LAYOUT[t.qtype][0], t.row_bytes
    sb_parts = []
    for rows, inputs in parts:
        if inputs is not None and (inputs.start % sb or inputs.stop % sb):
            raise ValueError(f"{name}: input slice {inputs} is not whole superblocks")
        sb_parts.append(
            (
                rows,
                None
                if inputs is None
                else slice(inputs.start // sb, inputs.stop // sb),
            )
        )
    geo, total = _geometry(t.qtype, M_all, K // sb, sb_parts)
    lo = min(g[0].start for g in geo)
    hi = max(g[0].stop for g in geo)
    out = np.empty((E, total), np.uint8)
    f, SBB = _lib().lq_planar_rb, superblock_bytes(t.qtype)
    local = threading.local()

    def one(e: int) -> None:
        buf = getattr(local, "buf", None)
        if buf is None or buf.shape != (hi - lo, rb_bytes):
            buf = local.buf = np.empty((hi - lo, rb_bytes), np.uint8)
        model.read_rows(name, slice(lo, hi), expert=e, out=buf)
        dst = out[e].ctypes.data
        for rows, sbs, M, nsb, o in geo:
            src = buf.ctypes.data + (rows.start - lo) * rb_bytes
            for rb in range(-(-M // 128)):
                if f(
                    code,
                    src,
                    rb_bytes,
                    M,
                    128 * rb,
                    sbs.start,
                    nsb,
                    dst + o + rb * nsb * SBB,
                ):
                    raise RuntimeError(f"lq_planar_rb failed for {name}")

    list(pool().map(one, range(E)))
    return out
