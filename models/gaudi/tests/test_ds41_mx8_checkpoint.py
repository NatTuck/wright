"""The MXFP8 weight conversion against the checkpoint's OWN block scales (no card).

Every GEMM in the model reads weights the loader produced with ``mx8_rows``: it folds the checkpoint's
per-32x32 E8M0 block scales into one power-of-two scale per row *by shifting each 32-element block's E4M3
codes*, and (by default) permutes the result into the GEMM operand's tau order.  That shift-and-refold is
the least obvious transform in the load path, and nothing checked it against the checkpoint's declared
scales -- the loader's own tests use synthetic tables, the same "oracle beside the code" trap that hid the
gslot and Engram bugs.

The oracle is built from the checkpoint alone:
    W[m, k] = e4m3(code[m, k]) * 2^(scales[m // 32, k // 32] - 127)
and the loader's bytes are dequantized the same way.  With ``tau=False`` they must agree to rounding; with
``tau=True`` they must differ, because that is the operand permutation the kernels read through -- so this
also pins that the permutation is the *only* difference.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from resin_models_gaudi.deepseek_v41.weights import mx8_rows

CHECKPOINT = Path(
    os.environ.get(
        "DS41_CHECKPOINT",
        "/home/nat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
        "/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170",
    )
)
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists(), reason="DS41_CHECKPOINT is not set"
)


def read(name: str) -> np.ndarray:
    for path in sorted(glob.glob(os.path.join(CHECKPOINT, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        if name not in header:
            continue
        info = header[name]
        with open(path, "rb") as f:
            data = f.read()
        base = 8 + n
        off = base + info["data_offsets"][0]
        nb = info["data_offsets"][1] - info["data_offsets"][0]
        raw = np.frombuffer(data, np.uint8, nb, off)
        if info["dtype"] in ("F8_E4M3", "F8_E8M0"):
            return raw.copy().reshape(info["shape"])
        if info["dtype"] == "BF16":
            return ((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(info["shape"])
        if info["dtype"] == "F32":
            return raw.view(np.float32).copy().reshape(info["shape"])
        raise AssertionError(f"{name}: unhandled dtype {info['dtype']}")
    raise AssertionError(f"{name}: not found")


def e4m3(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes).astype(np.int64)
    e, m = (c >> 3) & 15, c & 7
    sign = np.where(c & 0x80, -1.0, 1.0)
    mag = np.where(e == 0, (m / 8.0) * 2.0**-6, (1.0 + m / 8.0) * 2.0 ** (e - 7))
    mag = np.where((c & 0x7F) == 0x7F, 0.0, mag)
    return (sign * mag).astype(np.float64)


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.attn.wq_a",
        "layers.0.attn.wkv",
        "layers.0.attn.wq_b",
        "layers.0.attn.wo_a",
        "layers.0.attn.wo_b",
    ],
)
def test_mx8_conversion_matches_the_checkpoint_block_scales(name):
    """The loader's fold of the 32x32 E8M0 block scales, against the checkpoint's own scales."""
    w = read(f"{name}.weight")[:256]
    s = read(f"{name}.scale")[:8]
    M, K = w.shape
    assert s.shape == (M // 32, K // 32), (s.shape, w.shape)
    want = e4m3(w) * np.repeat(np.repeat(np.exp2(s.astype(np.int64) - 127), 32, 0), 32, 1)

    natural = mx8_rows(np.ascontiguousarray(w), np.ascontiguousarray(s), tau=False)
    got = e4m3(natural.bytes) * np.asarray(natural.row_scales, np.float64)[:, None]
    err = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30)
    assert err < 1e-5, f"natural order: relative error {err:.3e}"

    # the default is the GEMM operand's tau order, so it MUST differ -- that is the permutation the
    # kernels read through, and it is the only difference
    tau = mx8_rows(np.ascontiguousarray(w), np.ascontiguousarray(s))
    other = e4m3(tau.bytes) * np.asarray(tau.row_scales, np.float64)[:, None]
    err_tau = np.linalg.norm(other - want) / (np.linalg.norm(want) + 1e-30)
    assert err_tau > 1e-2, (
        f"tau order should be a permutation of the natural order, not a match "
        f"(relative error {err_tau:.3e}); if it matches, tau is not being applied"
    )
