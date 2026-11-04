"""Synapse graphs, and compiled recipes cached by the content of their definition.

A recipe's identity is the hash of its graph definition (tensors, sections, nodes, node
parameters and dependencies), the digests of the custom kernels it uses, the kernel glue,
the runtime that builds the graph, the Synapse build and compiler-affecting environment
variables. Every persistent tensor is bound at launch.
"""

from __future__ import annotations

import ctypes as C
import mmap
import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from blake3 import blake3

from resin.kernel_cache import ArtifactCache
from resin_models_gaudi.synapse.build import default_cache_root
from resin_models_gaudi.synapse.runtime import ITEM_SIZE, Device, SynapseError

CUSTOM_PREFIX = "rg_"
CUSTOM_SUFFIX = "_f32"

# Environment variables in a recipe's identity.
_COMPILER_ENV_PREFIXES = (
    "ENABLE_",
    "GC_",
    "SYN_",
    "RUN_TPC",
    "SRAM_",
    "NON_BUNDLE_",
    "DCORE_",
    "DEFAULT_",
    "ARC_",
    "DISABLE_",
    "MAX_TPC",
    "HABANA_PROFILE",
)
_COMPILER_ENV_EXCLUDED = frozenset({"GC_KERNEL_PATH", "ENABLE_CONSOLE"})
_COMPILER_LIBRARY = "/usr/lib/habanalabs/libSynapse.so"


@cache
def _compiler_reads(key: str) -> bool:
    """Whether the installed Synapse knows this name. Unknown means yes."""
    try:
        with open(_COMPILER_LIBRARY, "rb") as library:
            with mmap.mmap(library.fileno(), 0, prot=mmap.PROT_READ) as data:
                return data.find(key.encode() + b"\0") >= 0
    except OSError:
        return True


def compiler_environment() -> str:
    return "".join(
        f"{key}={os.environ[key]};"
        for key in sorted(os.environ)
        if key.startswith(_COMPILER_ENV_PREFIXES)
        and key not in _COMPILER_ENV_EXCLUDED
        and _compiler_reads(key)
    )


@dataclass(frozen=True, slots=True)
class Tensor:
    """A graph tensor; ``shape`` is fastest-dimension first."""

    graph: Graph
    index: int
    name: str
    dtype: int
    shape: tuple[int, ...]
    persistent: bool

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * ITEM_SIZE[self.dtype]


class Graph:
    """A Synapse graph under construction."""

    def __init__(self, device: Device, name: str) -> None:
        self.device = device
        self.name = name
        self.handle = device.check(device.lib.rg_graph_new(), "graph_new")
        self.tensors: dict[str, Tensor] = {}
        self.kernels: set[str] = set()
        self._definition = blake3(name.encode())
        self._nodes = 0
        self._anonymous = 0
        self.destroyed = False

    def _record(self, text: str, params: bytes = b"") -> None:
        self._definition.update(f"{len(text)}:{text}{len(params)}:".encode())
        self._definition.update(params)

    def tensor(
        self,
        shape: Sequence[int],
        dtype: int,
        name: str | None = None,
        *,
        persistent: bool = False,
    ) -> Tensor:
        if name is None:
            self._anonymous += 1
            name = f"t{self._anonymous}"
        if name in self.tensors:
            raise ValueError(f"graph {self.name} already has tensor {name}")
        shape = tuple(int(s) for s in shape)
        sizes = (C.c_uint64 * len(shape))(*shape)
        index = self.device.check(
            self.device.lib.rg_tensor(
                self.handle, name.encode(), dtype, len(shape), sizes, int(persistent)
            ),
            f"tensor {name}",
        )
        self._record(f"T|{name}|{dtype}|{shape}|{int(persistent)}")
        tensor = Tensor(self, index, name, dtype, shape, persistent)
        self.tensors[name] = tensor
        return tensor

    def input(self, name: str, shape: Sequence[int], dtype: int) -> Tensor:
        """A persistent tensor bound at launch (input or output)."""
        return self.tensor(shape, dtype, name, persistent=True)

    def tensor_in_section(
        self, name: str, shape: Sequence[int], dtype: int, other: Tensor, offset: int
    ) -> Tensor:
        """A persistent tensor at byte ``offset`` inside ``other``'s section."""
        shape = tuple(int(s) for s in shape)
        sizes = (C.c_uint64 * len(shape))(*shape)
        index = self.device.check(
            self.device.lib.rg_tensor_in_section(
                self.handle,
                name.encode(),
                dtype,
                len(shape),
                sizes,
                other.index,
                offset,
            ),
            f"tensor_in_section {name}",
        )
        self._record(f"S|{name}|{dtype}|{shape}|{other.name}|{offset}")
        tensor = Tensor(self, index, name, dtype, shape, True)
        self.tensors[name] = tensor
        return tensor

    def section_rmw(self, tensor: Tensor) -> None:
        self.device.check(
            self.device.lib.rg_section_rmw(self.handle, tensor.index), "section_rmw"
        )
        self._record(f"R|{tensor.name}")

    def node(
        self,
        guid: str,
        inputs: Sequence[Tensor | None],
        outputs: Sequence[Tensor],
        params: bytes = b"",
        name: str | None = None,
    ) -> int:
        """Add a node; returns its index for ``dep``."""
        self._nodes += 1
        name = name or f"n{self._nodes}_{guid}"
        ins = (C.c_int * max(1, len(inputs)))(
            *[-1 if t is None else t.index for t in inputs]
        )
        outs = (C.c_int * max(1, len(outputs)))(*[t.index for t in outputs])
        buffer = C.create_string_buffer(params, len(params)) if params else None
        if guid.startswith(CUSTOM_PREFIX) and guid.endswith(CUSTOM_SUFFIX):
            kernel = guid[len(CUSTOM_PREFIX) : -len(CUSTOM_SUFFIX)]
            if kernel not in self.device.libraries.kernel_digests:
                raise SynapseError(f"unknown custom kernel {guid}")
            self.kernels.add(kernel)
        input_names = [None if t is None else t.name for t in inputs]
        self._record(
            f"N|{guid}|{name}|{input_names}|{[t.name for t in outputs]}", bytes(params)
        )
        return self.device.check(
            self.device.lib.rg_node(
                self.handle,
                guid.encode(),
                name.encode(),
                ins,
                len(inputs),
                outs,
                len(outputs),
                buffer,
                len(params),
            ),
            f"node {guid}",
        )

    def dep(self, blocking: int, blocked: int) -> None:
        """Run node ``blocked`` after node ``blocking``."""
        self.device.check(self.device.lib.rg_dep(self.handle, blocking, blocked), "dep")
        self._record(f"D|{blocking}|{blocked}")

    def identity(self) -> str:
        libraries = self.device.libraries
        digest = blake3(self._definition.digest())
        for kernel in sorted(self.kernels):
            digest.update(f"{kernel}={libraries.kernel_digests[kernel]};".encode())
        digest.update(f"glue={libraries.glue_digest};".encode())
        digest.update(f"runtime={libraries.runtime_digest};".encode())
        digest.update(f"synapse={libraries.synapse_build};".encode())
        digest.update(compiler_environment().encode())
        return digest.hexdigest()

    def destroy(self) -> None:
        if not self.destroyed:
            self.device.lib.rg_graph_destroy(self.handle)
            self.destroyed = True


@dataclass(slots=True)
class Recipe:
    """A compiled graph; persistent tensors are bound by name or slot before launch."""

    device: Device
    handle: int
    name: str
    identity: str
    from_cache: bool
    persistent: dict[str, Tensor]
    _slots: dict[str, int] = field(default_factory=dict)

    def slot(self, name: str) -> int:
        slot = self._slots.get(name)
        if slot is None:
            slot = self.device.check(
                self.device.lib.rg_slot(self.handle, name.encode()), f"slot {name}"
            )
            self._slots[name] = slot
        return slot

    def bind(self, name: str, address: int) -> None:
        self.bind_slots(
            np.array([self.slot(name)], np.int32), np.array([address], np.uint64)
        )

    def bind_slots(self, slots: np.ndarray, addresses: np.ndarray) -> None:
        slots = np.ascontiguousarray(slots, np.int32)
        addresses = np.ascontiguousarray(addresses, np.uint64)
        self.device.check(
            self.device.lib.rg_bind_slots(
                self.handle, slots.ctypes.data, addresses.ctypes.data, len(slots)
            ),
            "bind_slots",
        )

    def bind_all(self, addresses: dict[str, int]) -> None:
        slots = np.array([self.slot(name) for name in addresses], np.int32)
        self.bind_slots(slots, np.array(list(addresses.values()), np.uint64))

    @property
    def workspace(self) -> int:
        return int(self.device.lib.rg_recipe_workspace(self.handle))

    def launch(self) -> None:
        self.device.check(self.device.lib.rg_launch(self.handle), f"launch {self.name}")


class LaunchPlan:
    """Launches of compiled recipes, each after rebinding some of its persistent tensors.
    ``steps`` holds (recipe, slots, addresses) per launch; later
    bindings of a slot win. ``patch`` changes one binding's address between runs."""

    def __init__(
        self, device, steps: Sequence[tuple[Recipe, np.ndarray, np.ndarray]]
    ) -> None:
        self.device = device
        self.starts = np.zeros(len(steps) + 1, np.int64)
        self.starts[1:] = np.cumsum([len(slots) for _, slots, _ in steps])
        self.slots = np.ascontiguousarray(
            np.concatenate([np.asarray(s, np.int32) for _, s, _ in steps]), np.int32
        )
        self.addresses = np.ascontiguousarray(
            np.concatenate([np.asarray(a, np.uint64) for _, _, a in steps]), np.uint64
        )
        slots, addresses = self.slots.ctypes.data, self.addresses.ctypes.data
        self._steps = [
            (
                recipe.handle,
                slots + 4 * int(a),
                addresses + 8 * int(a),
                int(b - a),
                recipe.name,
            )
            for (recipe, _, _), a, b in zip(
                steps, self.starts[:-1], self.starts[1:], strict=True
            )
        ]

    def patch(self, index: int, address: int) -> None:
        self.addresses[index] = address

    def run(self) -> None:
        lib = self.device.lib
        bind, launch = lib.rg_bind_slots, lib.rg_launch
        for handle, slots, addresses, n, name in self._steps:
            bind(handle, slots, addresses, n)
            if launch(handle) < 0:
                self.device.check(-1, f"launch {name}")


class RecipeCache:
    """Compiled recipes in a content-validated ``ArtifactCache`` namespace."""

    def __init__(
        self, device: Device, root: str | os.PathLike[str] | None = None
    ) -> None:
        self.device = device
        abi = f"recipes-{device.libraries.synapse_build}"
        self.store = ArtifactCache(Path(root or default_cache_root()), abi)

    @property
    def hits(self) -> int:
        return self.store.hits

    @property
    def misses(self) -> int:
        return self.store.misses

    def compile(self, graph: Graph) -> Recipe:
        """Load ``graph``'s recipe from the cache, or compile and publish it."""
        identity = graph.identity()
        artifact = Path(f"{graph.name}-{identity[:32]}.recipe")
        device = self.device
        loaded = {"from_cache": True}
        with tempfile.TemporaryDirectory(prefix="resin-gaudi-recipe-") as scratch:
            source = Path(scratch) / artifact.name

            def load(path: Path) -> int:
                return device.check(
                    device.lib.rg_recipe_load(graph.handle, str(path).encode()),
                    f"load recipe {graph.name}",
                )

            def build() -> int:
                loaded["from_cache"] = False
                return device.check(
                    device.lib.rg_compile(
                        graph.handle, graph.name.encode(), str(source).encode()
                    ),
                    f"compile {graph.name}",
                )

            handle = self.store.load_or_build_from(
                artifact, source=source, identity=identity, load=load, build=build
            )
        persistent = {n: t for n, t in graph.tensors.items() if t.persistent}
        return Recipe(
            device, handle, graph.name, identity, loaded["from_cache"], persistent
        )

    def compile_many(
        self, graphs: Sequence[Graph], workers: int | None = None
    ) -> list[Recipe]:
        """Compile or load ``graphs`` on ``workers`` threads; no launches may run meanwhile."""
        with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
            return list(pool.map(self.compile, graphs))
