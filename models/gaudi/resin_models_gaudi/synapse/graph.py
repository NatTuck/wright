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
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from blake3 import blake3

from resin.kernel_cache import ArtifactCache
from resin_models_gaudi.loading.progress import info, logger, progress
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
        progress.graph_built()
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
        digest.update(f"graph={libraries.graph_digest};".encode())
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
    compile_seconds: float = 0.0
    """The time of the compile that made this recipe here (0 for a recipe from the cache)."""

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
    bindings of a slot win. ``patch`` changes one binding's address between runs.

    ``run`` makes the launches in one native call (``resin._resin.GaudiProgram``, built at the first run) when the
    device has the glue for it (``Device.glue``), else in a Python loop. The native program keeps each launch's last
    binding of each slot; a binding patched before the first run takes its address from a register, so ``patch``
    costs no rebuild (patching another binding later rebuilds the program once)."""

    def __init__(
        self, device, steps: Sequence[tuple[Recipe, np.ndarray, np.ndarray]]
    ) -> None:
        self.device = device
        self._recipes = [(recipe.handle, recipe.name) for recipe, _, _ in steps]
        self._patched: dict[int, int] = {}  # binding index -> register
        self._program: Any = None
        self._registers = np.zeros(0, np.uint64)
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
        register = self._patched.get(index)
        if register is not None:
            self._registers[register] = address
            return
        self._patched[index] = len(self._patched)
        self._program = None  # the program has this binding as fixed: rebuild it

    def _build(self) -> Any:
        """The native program of these launches: per launch its last binding of each slot, fixed or (patched)
        from a register."""
        from resin._resin import GaudiProgram

        program = GaudiProgram(len(self._patched))
        program.section("launches")
        slots, addresses = self.slots.tolist(), self.addresses.tolist()
        bounds = zip(self.starts[:-1].tolist(), self.starts[1:].tolist(), strict=True)
        for (handle, _), (a, b) in zip(self._recipes, bounds, strict=True):
            last = {slots[i]: i for i in range(a, b)}  # a later binding of a slot wins
            fixed = [i for i in last.values() if i not in self._patched]
            dynamic = [
                (slots[i], (0, self._patched[i], 1))
                for i in last.values()
                if i in self._patched
            ]
            program.launch(
                handle,
                [slots[i] for i in fixed],
                [addresses[i] for i in fixed],
                dynamic=dynamic,
            )
        program.finish()
        self._registers = np.zeros(len(self._patched), np.uint64)
        for index, register in self._patched.items():
            self._registers[register] = self.addresses[index]
        return program

    def run(self) -> None:
        glue = self.device.glue
        if glue is not None:
            if self._program is None:
                self._program = self._build()
            try:
                self._program.run(glue, self._registers, (0,))
            except RuntimeError as error:  # a failed call, as the device reports one
                raise SynapseError(f"launch plan: {error}") from error
            return
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

    def _artifact(self, graph: Graph, identity: str) -> Path:
        return Path(f"{graph.name}-{identity[:32]}.recipe")

    def cached(self, graph: Graph) -> bool:
        """Whether the cache has a recipe for ``graph`` (a file check; loading validates it)."""
        return self.store.record_path(self._artifact(graph, graph.identity())).is_file()

    def compile(self, graph: Graph) -> Recipe:
        """Load ``graph``'s recipe from the cache, or compile and publish it."""
        identity = graph.identity()
        artifact = self._artifact(graph, identity)
        device = self.device
        loaded = {"from_cache": True, "seconds": 0.0}
        with tempfile.TemporaryDirectory(prefix="resin-gaudi-recipe-") as scratch:
            source = Path(scratch) / artifact.name

            def load(path: Path) -> int:
                return device.check(
                    device.lib.rg_recipe_load(graph.handle, str(path).encode()),
                    f"load recipe {graph.name}",
                )

            def build() -> int:
                loaded["from_cache"] = False
                start = time.monotonic()
                handle = device.check(
                    device.lib.rg_compile(
                        graph.handle, graph.name.encode(), str(source).encode()
                    ),
                    f"compile {graph.name}",
                )
                loaded["seconds"] = time.monotonic() - start
                device.count_compile()
                return handle

            handle = self.store.load_or_build_from(
                artifact, source=source, identity=identity, load=load, build=build
            )
        persistent = {n: t for n, t in graph.tensors.items() if t.persistent}
        return Recipe(
            device,
            handle,
            graph.name,
            identity,
            loaded["from_cache"],
            persistent,
            compile_seconds=loaded["seconds"],
        )

    def compile_many(
        self, graphs: Sequence[Graph], workers: int | None = None
    ) -> list[Recipe]:
        """Compile or load ``graphs`` on ``workers`` threads; no launches may run on the calling thread's context
        meanwhile (it grows to the recipes' workspace). Startup progress counts them in its recipes phase: the
        graphs not in the cache, then each graph as it is compiled here, compiled by another process (which
        this one waited for) or loaded. The log gets one summary line, with the slowest compiles."""
        if progress.current() == "recipes":
            progress.extend(len(graphs))
        else:
            progress.phase("recipes", total=len(graphs))
        progress.note("")
        start = time.monotonic()
        missing = [not self.cached(graph) for graph in graphs]
        progress.recipes_planned(sum(missing))

        context = self.device.current_context()

        def one(item: tuple[Graph, bool]) -> Recipe:
            graph, was_missing = item
            with context.use():
                recipe = self.compile(graph)
            if not recipe.from_cache:
                progress.recipe_done("compiled", recipe.compile_seconds)
                logger.debug(f"recipe {graph.name}: compiled in {recipe.compile_seconds:.1f} s")
            else:
                progress.recipe_done("shared" if was_missing else "cached")
            return recipe

        threads = workers or os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=threads) as pool:
            recipes = list(pool.map(one, zip(graphs, missing, strict=True)))
        progress.note("binding recipes")
        info(_compile_summary(recipes, missing, time.monotonic() - start, min(threads, len(graphs))))
        return recipes


def _compile_summary(
    recipes: Sequence[Recipe], missing: Sequence[bool], seconds: float, threads: int = 1
) -> str:
    """``recipes: 148 graphs in 9.1 s: 37 compiled (190 s of compile time on 37 threads), 111 from the
    cache; slowest: ...``"""
    compiled = [r for r in recipes if not r.from_cache]
    shared = sum(m and r.from_cache for r, m in zip(recipes, missing, strict=True))
    cached = len(recipes) - len(compiled) - shared
    parts = []
    if compiled:
        busy = sum(r.compile_seconds for r in compiled)
        on = f" on {min(threads, len(compiled))} threads" if threads > 1 and len(compiled) > 1 else ""
        parts.append(f"{len(compiled)} compiled ({busy:.0f} s of compile time{on})")
    if shared:
        parts.append(f"{shared} compiled by another process")
    parts.append(f"{cached} from the cache")
    line = f"recipes: {len(recipes)} graphs in {seconds:.1f} s: " + ", ".join(parts)
    slowest = sorted(compiled, key=lambda r: r.compile_seconds, reverse=True)[:3]
    if slowest:
        line += "; slowest: " + ", ".join(f"{r.name} {r.compile_seconds:.1f} s" for r in slowest)
    return line
