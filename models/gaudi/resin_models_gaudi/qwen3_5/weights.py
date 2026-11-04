"""Checkpoint weights resident on the Gaudi card.

Dense tensors are uploaded as prepared. Block-FP8 weights are uploaded with their 128x128 block
scales and requantized on the device to per-row FP8 (``requant_fp8``). ``name`` then holds the
FP8 bytes and ``name + ".s"`` the row scales.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from resin_models_gaudi.qwen3_5.checkpoint import (
    DenseWeight,
    Fp8Weight,
    Qwen35Checkpoint,
)
from resin_models_gaudi.synapse.graph import Graph, Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import BF16, F8, F32, Device

_DTYPES = {np.dtype(np.uint16): BF16, np.dtype(np.float32): F32}


@dataclass(frozen=True, slots=True)
class DeviceTensor:
    """A device allocation with its graph shape (fastest dimension first)."""

    address: int
    dtype: int
    shape: tuple[int, ...]


class DeviceWeights:
    """Weights by checkpoint name (``l3.w_qkv``, ``l3.w_qkv.s``, ``E_head``, ...)."""

    def __init__(
        self,
        device: Device,
        recipes: RecipeCache,
        checkpoint: Qwen35Checkpoint,
        *,
        mtp: bool = False,
        layers: Collection[int] | None = None,
    ) -> None:
        self.device = device
        self.recipes = recipes
        self.tensors: dict[str, DeviceTensor] = {}
        self.nbytes = 0
        self._requant: dict[tuple[int, int], Recipe] = {}
        self._staging = [(0, 0), (0, 0)]
        try:
            for name, weight in _prefetched(checkpoint.weights(mtp=mtp, layers=layers)):
                if isinstance(weight, Fp8Weight):
                    self._upload_fp8(name, weight)
                else:
                    self._upload_dense(name, weight)
        finally:
            for address, _ in self._staging:
                if address:
                    device.free(address)
            self._staging = [(0, 0), (0, 0)]

    def __getitem__(self, name: str) -> DeviceTensor:
        return self.tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def close(self) -> None:
        for tensor in self.tensors.values():
            self.device.free(tensor.address)
        self.tensors.clear()

    def _upload_dense(self, name: str, weight: DenseWeight) -> None:
        array = np.ascontiguousarray(weight.array)
        address = self.device.malloc(array.nbytes)
        self.device.h2d(address, array)
        self.tensors[name] = DeviceTensor(
            address, _DTYPES[array.dtype], tuple(reversed(array.shape))
        )
        self.nbytes += array.nbytes

    def _stage(self, slot: int, nbytes: int) -> int:
        address, size = self._staging[slot]
        if size < nbytes:
            if address:
                self.device.free(address)
            address, size = self.device.malloc(nbytes), nbytes
            self._staging[slot] = (address, size)
        return address

    def _requant_recipe(self, K: int, M: int) -> Recipe:
        recipe = self._requant.get((K, M))
        if recipe is None:
            graph = Graph(self.device, f"requant_fp8_{K}x{M}")
            w = graph.input("w", (K, M), F8)
            bs = graph.input("bs", (K // 128, M // 128), F32)
            out = graph.input("o", (K, M), F8)
            rs = graph.input("rs", (1, M), F32)
            graph.node("rg_requant_fp8_f32", [w, bs], [out, rs])
            recipe = self.recipes.compile(graph)
            graph.destroy()
            self._requant[(K, M)] = recipe
        return recipe

    def _upload_fp8(self, name: str, weight: Fp8Weight) -> None:
        M, K = weight.bytes.shape
        scales = np.ascontiguousarray(weight.block_scales, np.float32)
        w_in, s_in = self._stage(0, weight.bytes.nbytes), self._stage(1, scales.nbytes)
        self.device.h2d(w_in, np.ascontiguousarray(weight.bytes))
        self.device.h2d(s_in, scales)
        out, row_scales = self.device.malloc(M * K), self.device.malloc(M * 4)
        recipe = self._requant_recipe(K, M)
        recipe.bind_all({"w": w_in, "bs": s_in, "o": out, "rs": row_scales})
        recipe.launch()
        self.device.sync()
        self.tensors[name] = DeviceTensor(out, F8, (K, M))
        self.tensors[name + ".s"] = DeviceTensor(row_scales, F32, (M,))
        self.nbytes += M * K + M * 4


def _prefetched(items: Iterator, depth: int = 2) -> Iterator:
    """Prepare the next checkpoint items on a thread while the caller uploads."""
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
