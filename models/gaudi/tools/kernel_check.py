"""Numeric check for a single custom kernel, on a real card.

Builds a one-node graph, binds host arrays, launches, and compares against the numpy
reference. Small and fast, so a kernel can be iterated without compiling the model.

    HABANA_VISIBLE_MODULES=6 .venv/bin/python models/gaudi/tools/kernel_check.py
"""

from __future__ import annotations

import struct
import sys

import numpy as np

from resin_models_gaudi.qwen3_5.layers import F32
from resin_models_gaudi.synapse.graph import Graph, RecipeCache
from resin_models_gaudi.synapse.runtime import SynapseError, open_device


def main() -> int:
    try:
        device = open_device(None)
    except SynapseError as error:
        print(f"no card: {error}", file=sys.stderr)
        return 1
    print(f"card: accel {device.card}")

    rng = np.random.default_rng(0)
    C, T = 1152, 64
    x = rng.normal(size=(C, T)).astype(np.float32)
    w = rng.normal(size=(C,)).astype(np.float32)
    b = rng.normal(size=(C,)).astype(np.float32)

    which = sys.argv[1] if len(sys.argv) > 1 else "ln"
    g = Graph(device, f"{which}_{C}x{T}")
    if which == "softmax":
        g.node(
            "rg_softmax_rows_f32",
            [g.input("x", (C, T), F32)],
            [g.input("y", (C, T), F32)],
        )
    else:
        g.node(
            "rg_ln_affine_f32",  # LayerNorm with weight and bias, our kernel
            [g.input("x", (C, T), F32), g.input("w", (C,), F32), g.input("b", (C,), F32)],
            [g.input("y", (C, T), F32)],
            struct.pack("f", 1e-6),
        )
    recipe = RecipeCache(device).compile(g)

    addr = {
        "x": device.malloc(x.nbytes),
        "w": device.malloc(w.nbytes),
        "b": device.malloc(b.nbytes),
        "y": device.malloc(x.nbytes),
    }
    device.h2d(addr["y"], np.full((C, T), -7.0, np.float32))  # prove the kernel writes
    # Synapse tensors keep dim 0 contiguous, so the host array must be feature-major.
    for name, array in (("x", x.T), ("w", w), ("b", b)):
        device.h2d(addr[name], np.ascontiguousarray(array))
    recipe.bind_all({k: v for k, v in addr.items() if which != "softmax" or k in ("x", "y")})
    recipe.launch()
    device.sync()
    got = device.d2h(addr["y"], (T, C), np.float32).T

    if which == "softmax":
        e = np.exp(x - x.max(0))
        want = e / e.sum(0)
        error = np.abs(got - want).max()
        print(f"softmax max abs error {error:.3e}")
        return 0 if error < 1e-5 else 2
    mean = x.mean(0)
    var = x.var(0)
    want = (x - mean) / np.sqrt(var + 1e-6) * w[:, None] + b[:, None]

    np.set_printoptions(precision=4, suppress=True)
    print("got  row0:", got[0], " want row0:", want[0])
    print("got  col0[:3]:", got[:3, 0], " want col0[:3]:", want[:3, 0])

    error = np.abs(got - want).max()
    scale = np.abs(want).max()
    print(f"max abs error {error:.3e} (relative {error / scale:.3e})")
    if not np.isfinite(got).all():
        print("NON-FINITE OUTPUT")
        return 2
    return 0 if error / scale < 3e-2 else 2  # nl_rsqrt is a fast approximation


if __name__ == "__main__":
    raise SystemExit(main())
