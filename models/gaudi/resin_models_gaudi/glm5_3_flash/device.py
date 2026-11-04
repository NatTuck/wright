"""GLM-5.3-Flash weights resident on one card.

Plain tensors upload as prepared. FP8 weights convert with ``fp8_inc``, then requantize to per-row scales:
``requant_fp8`` for dense weights (``name`` and ``name.s``), ``x8_requant`` for routed experts (``name`` and ``name.c``).
"""

from __future__ import annotations

import numpy as np

from resin_models_gaudi.glm5_3_flash.weights import Fp8
from resin_models_gaudi.qwen3_5.weights import _prefetched
from resin_models_gaudi.synapse.graph import Graph, Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import F8, F32, U8, Device


class WeightLoader:
    def __init__(self, device: Device, recipes: RecipeCache) -> None:
        self.device = device
        self.recipes = recipes
        self.w: dict[str, int] = {}
        self.nbytes = 0
        self._recipes: dict[tuple, Recipe] = {}
        self._staging = [(0, 0), (0, 0), (0, 0)]

    def load(self, items) -> None:
        try:
            for name, payload, kind in _prefetched(items):
                if kind == "fp8":
                    self._fp8(name, payload)
                elif kind == "x8":
                    self._x8(name, payload)
                else:
                    self._plain(name, payload)
        finally:
            for address, _ in self._staging:
                if address:
                    self.device.free(address)
            self._staging = [(0, 0), (0, 0), (0, 0)]

    def _plain(self, name: str, array: np.ndarray) -> None:
        a = np.ascontiguousarray(array)
        address = self.device.malloc(max(a.nbytes, 256))
        self.device.h2d(address, a)
        self.w[name] = address
        self.nbytes += a.nbytes

    def _stage(self, slot: int, nbytes: int) -> int:
        address, size = self._staging[slot]
        if size < nbytes:
            if address:
                self.device.free(address)
            address, size = self.device.malloc(nbytes), nbytes
            self._staging[slot] = (address, size)
        return address

    def _recipe(self, key: tuple, build) -> Recipe:
        recipe = self._recipes.get(key)
        if recipe is None:
            graph = build()
            recipe = self.recipes.compile(graph)
            graph.destroy()
            self._recipes[key] = recipe
        return recipe

    def _inc(self, raw: np.ndarray, address: int) -> None:
        """the raw bytes in staging -> fp8_inc bytes at ``address``"""
        n = raw.size // 256

        def build() -> Graph:
            g = Graph(self.device, f"fp8_inc_{n}")
            g.node("rg_fp8_inc_f32", [g.input("w", (256, n), U8)], [g.input("o", (256, n), U8)])
            return g

        recipe = self._recipe(("inc", n), build)
        raw_address = self._stage(0, raw.nbytes)
        self.device.h2d(raw_address, np.ascontiguousarray(raw))
        recipe.bind_all({"w": raw_address, "o": address})
        recipe.launch()

    def _fp8(self, name: str, weight: Fp8) -> None:
        M, K = weight.bytes.shape
        scales = np.ascontiguousarray(weight.block_scales * 2, np.float32)
        inc, s_in = self._stage(1, M * K), self._stage(2, scales.nbytes)
        self._inc(weight.bytes, inc)
        self.device.h2d(s_in, scales)

        def build() -> Graph:
            g = Graph(self.device, f"requant_fp8_{K}x{M}")
            g.node(
                "rg_requant_fp8_f32",
                [g.input("w", (K, M), F8), g.input("bs", (K // 128, M // 128), F32)],
                [g.input("o", (K, M), F8), g.input("rs", (1, M), F32)],
            )
            return g

        recipe = self._recipe(("requant", K, M), build)
        out, rows = self.device.malloc(M * K), self.device.malloc(M * 4)
        recipe.bind_all({"w": inc, "bs": s_in, "o": out, "rs": rows})
        recipe.launch()
        self.device.sync()
        self.w[name], self.w[name + ".s"] = out, rows
        self.nbytes += M * K + M * 4

    def _x8(self, name: str, weight: Fp8) -> None:
        E, K, M = weight.bytes.shape
        scales = np.ascontiguousarray(weight.block_scales * 2, np.float32)
        inc, s_in = self._stage(1, E * K * M), self._stage(2, scales.nbytes)
        self._inc(weight.bytes, inc)
        self.device.h2d(s_in, scales)

        def build() -> Graph:
            g = Graph(self.device, f"x8_requant_{M}x{K}x{E}")
            g.node(
                "rg_x8_requant_f32",
                [g.input("w", (M, K, E), F8), g.input("bs", (M // 128, K // 128, E), F32)],
                [g.input("o", (M, K, E), F8), g.input("c", (M, E), F32)],
            )
            return g

        recipe = self._recipe(("x8", M, K, E), build)
        out, rows = self.device.malloc(E * K * M), self.device.malloc(4 * M * E)
        recipe.bind_all({"w": inc, "bs": s_in, "o": out, "c": rows})
        recipe.launch()
        self.device.sync()
        self.w[name], self.w[name + ".c"] = out, rows
        self.nbytes += E * K * M + 4 * M * E
