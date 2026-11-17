"""The device uploader that every Gaudi model uses for its weights.

A model's weight preparation yields ``(name, payload, kind)`` items; ``WeightLoader.load``
prepares the next items on a thread while it uploads. Kinds:

- ``fp8``: ``Fp8`` with OCP-range e4m3 bytes (M, K). ``fp8_inc`` halves them on the card, then
  ``requant_fp8`` gives per-row scales: ``name`` (FP8) and ``name.s`` (f32, M).
- ``inc8``: ``Fp8`` already in the Gaudi range; ``requant_fp8`` only.
- ``x8``: ``Fp8`` routed experts (E, K, M), OCP range: ``fp8_inc``, then ``x8_requant`` gives
  ``name`` and ``name.c`` (f32, (E, M)).
- ``stream``: an object with ``nbytes`` and ``upload(address, h2d, memset)`` that uploads itself
  in parts (tables too large to prepare whole in host memory).
- any other kind: an array, uploaded as it is. The kind is recorded (``kinds``) for models
  whose graphs depend on it.

Allocations are at least ``min_alloc`` bytes. The cache arena is sized from the card memory that
the weights leave free, and its size is part of every recipe's identity, so each model keeps the
allocation sizes it always had.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from resin_models_gaudi.synapse.graph import Graph, Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import F8, F32, U8, Device


@dataclass(frozen=True, slots=True)
class Fp8:
    """Block-FP8 bytes and their f32 scales, one per 128 x 128 block."""

    bytes: np.ndarray
    block_scales: np.ndarray


def prefetched(items: Iterable, depth: int = 2) -> Iterator:
    """Prepare the next items on a thread while the caller uploads; at most ``depth`` wait."""
    done = object()
    pending: queue.Queue = queue.Queue(maxsize=depth)

    def produce() -> None:
        try:
            for item in items:
                pending.put(item)
        except BaseException as error:  # surfaced in the consumer
            pending.put(error)
        pending.put(done)

    threading.Thread(target=produce, daemon=True).start()
    while (item := pending.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


class WeightLoader:
    """Weights on the card by name. ``w`` maps each name to its device address (or, for names
    a model adds itself, to any value it needs), ``kinds`` to the item's kind."""

    def __init__(
        self, device: Device, recipes: RecipeCache, *, min_alloc: int = 256
    ) -> None:
        self.device = device
        self.recipes = recipes
        self.min_alloc = min_alloc
        self.w: dict[str, int] = {}
        self.kinds: dict[str, str] = {}
        self.nbytes = 0
        self._owned: list[int] = []
        self._recipes: dict[tuple, Recipe] = {}
        self._staging = [(0, 0), (0, 0), (0, 0)]

    def load(self, items: Iterable[tuple[str, object, str]]) -> None:
        """Upload ``items``; the staging buffers are freed at the end."""
        try:
            for name, payload, kind in prefetched(items):
                if name in self.w:
                    raise ValueError(f"weight {name} uploaded twice")
                if kind in ("fp8", "inc8"):
                    assert isinstance(payload, Fp8), name
                    self._fp8(name, payload, inc=kind == "fp8")
                elif kind == "x8":
                    assert isinstance(payload, Fp8), name
                    self._x8(name, payload)
                elif kind == "stream":
                    self._stream(name, payload)
                else:
                    self._plain(name, np.asarray(payload))
                self.kinds[name] = kind
        finally:
            for address, _ in self._staging:
                if address:
                    self.device.free(address)
            self._staging = [(0, 0), (0, 0), (0, 0)]

    def malloc(self, nbytes: int) -> int:
        """A device allocation that ``close`` frees."""
        address = self.device.malloc(nbytes)
        self._owned.append(address)
        return address

    def close(self) -> None:
        """Free this loader's allocations (not the device: other models may use it)."""
        for address in self._owned:
            self.device.free(address)
        self._owned.clear()
        self.w.clear()
        self.kinds.clear()

    # ---- kinds
    def _plain(self, name: str, array: np.ndarray) -> None:
        a = np.ascontiguousarray(array)
        address = self.malloc(max(a.nbytes, self.min_alloc))
        self.device.h2d(address, a)
        self.w[name] = address
        self.nbytes += a.nbytes

    def _stream(self, name: str, payload) -> None:
        address = self.malloc(payload.nbytes)
        payload.upload(
            address,
            self.device.h2d,
            lambda a, n: self.device.memset32(a, 0, n // 4),
        )
        self.w[name] = address
        self.nbytes += payload.nbytes

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
            g.node(
                "rg_fp8_inc_f32",
                [g.input("w", (256, n), U8)],
                [g.input("o", (256, n), U8)],
            )
            return g

        recipe = self._recipe(("inc", n), build)
        raw_address = self._stage(0, raw.nbytes)
        self.device.h2d(raw_address, np.ascontiguousarray(raw))
        recipe.bind_all({"w": raw_address, "o": address})
        recipe.launch()

    def _fp8(self, name: str, weight: Fp8, inc: bool = True) -> None:
        """``inc``: the bytes are OCP e4m3 (halved to Gaudi FP8 here, their scales doubled); else
        Gaudi FP8."""
        M, K = weight.bytes.shape
        scales = np.ascontiguousarray(
            weight.block_scales * (2 if inc else 1), np.float32
        )
        staged, s_in = self._stage(1, M * K), self._stage(2, scales.nbytes)
        if inc:
            self._inc(weight.bytes, staged)
        else:
            self.device.h2d(staged, np.ascontiguousarray(weight.bytes))
        self.device.h2d(s_in, scales)

        def build() -> Graph:
            g = Graph(self.device, f"requant_fp8_{K}x{M}")
            g.node(
                "rg_requant_fp8_f32",
                [
                    g.input("w", (K, M), F8),
                    g.input("bs", (K // 128, M // 128), F32),
                ],
                [g.input("o", (K, M), F8), g.input("rs", (1, M), F32)],
            )
            return g

        recipe = self._recipe(("requant", K, M), build)
        out, rows = self.malloc(M * K), self.malloc(M * 4)
        recipe.bind_all({"w": staged, "bs": s_in, "o": out, "rs": rows})
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
                [
                    g.input("w", (M, K, E), F8),
                    g.input("bs", (M // 128, K // 128, E), F32),
                ],
                [g.input("o", (M, K, E), F8), g.input("c", (M, E), F32)],
            )
            return g

        recipe = self._recipe(("x8", M, K, E), build)
        out, rows = self.malloc(E * K * M), self.malloc(4 * M * E)
        recipe.bind_all({"w": inc, "bs": s_in, "o": out, "c": rows})
        recipe.launch()
        self.device.sync()
        self.w[name], self.w[name + ".c"] = out, rows
        self.nbytes += E * K * M + 4 * M * E
