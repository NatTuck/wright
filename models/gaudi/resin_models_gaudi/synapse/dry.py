"""A Synapse device without a card, for tests: model code builds graphs, uploads weights and
issues launches as it does on a card, and the calls are recorded.

``dry_device()`` returns a ``runtime.Device`` over ``DryLib``, a stand-in for the runtime
library's ``rg_*`` functions:

- memory is a bump allocator over a range of fake addresses with the runtime's rounding
  (``max(bytes, 256) + 128``, 128-aligned). With ``keep_data``, uploads land in host memory
  and ``read`` returns them; without it, copies only record their sizes.
- graphs record their persistent tensors in creation order, which is the order of their
  slots in a compiled recipe.
- ``DryRecipeCache.compile`` "compiles" without an artifact cache: a recipe's slots are its
  graph's persistent tensors.
- launches, bindings, copies, events and collectives append to ``calls``, and the context that
  each issued on to ``issued_on``.
"""

from __future__ import annotations

import ctypes as C
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from resin_models_gaudi.synapse.build import NativeLibraries, kernel_sources
from resin_models_gaudi.synapse.graph import Graph, Recipe, RecipeCache
from resin_models_gaudi.synapse.runtime import Device

BASE = 0x10_0000_0000


@dataclass
class _Graph:
    persistent: list[str] = field(default_factory=list)
    destroyed: bool = False


@dataclass
class _Recipe:
    name: str
    persistent: list[str]
    binds: dict[int, int] = field(default_factory=dict)


class DryLib:
    """The runtime library's functions, recorded. ``calls`` holds ``(function, args...)``;
    ``issued_on[i]`` is the context of ``calls[i]``."""

    def __init__(self, total_bytes: int = 96 << 30, *, keep_data: bool = False) -> None:
        self.total = total_bytes
        self.keep_data = keep_data
        self.calls: list[tuple] = []
        self.issued_on: list[int] = []
        self.compiled_on: list[int] = []  # the context of each compile, in order
        self.recorded_at: dict[int, int] = {}  # event slot -> host ns of its last record
        self.affinities = [0x1]  # context id -> queue slot mask
        self._current = threading.local()
        self.allocations: dict[int, int] = {}  # address -> requested bytes
        self._next = BASE
        self._used = 0
        self._data: dict[int, bytearray] = {}
        self._host: list = []
        self._graphs: dict[int, _Graph] = {}
        self._recipes: dict[int, _Recipe] = {}
        self._lock = threading.Lock()

    def _call(self, entry: tuple) -> None:
        self.calls.append(entry)
        self.issued_on.append(self.rg_context_current())

    # ---- device
    def rg_last_error(self) -> bytes:
        return b"dry device"

    def rg_module_id(self) -> int:
        return 0

    def rg_sync(self) -> int:
        self._call(("sync",))
        return 0

    def rg_sync_all(self) -> int:
        self._call(("sync_all",))
        return 0

    # ---- contexts
    def rg_context_create(self, affinity: int) -> int:
        with self._lock:
            if not affinity:
                used = 0
                for mask in self.affinities:
                    used |= mask
                free = 0xF & ~used
                if not free:
                    return -1
                affinity = free & -free
            self.affinities.append(affinity)
            return len(self.affinities) - 1

    def rg_context_use(self, context: int) -> int:
        if not 0 <= context < len(self.affinities):
            return -1
        previous = self.rg_context_current()
        self._current.id = context
        return previous

    def rg_context_current(self) -> int:
        return getattr(self._current, "id", 0)

    def rg_context_affinity(self, context: int) -> int:
        return self.affinities[context] if 0 <= context < len(self.affinities) else 0

    def rg_compile_count(self) -> int:
        return 0

    def rg_now_ns(self) -> int:
        return 0

    # ---- memory
    def rg_malloc(self, nbytes: int) -> int:
        raw = (nbytes if nbytes else 256) + 128
        if self._used + raw > self.total:
            return 0
        address = self._next
        self._next += -(-raw // 128) * 128
        self._used += raw
        self.allocations[address] = nbytes
        if self.keep_data:
            self._data[address] = bytearray(nbytes)
        return address

    def rg_free(self, address: int) -> int:
        nbytes = self.allocations.pop(address, None)
        if nbytes is None:
            return -1
        self._used -= (nbytes if nbytes else 256) + 128
        self._data.pop(address, None)
        return 0

    def rg_mem_info(self, free, total) -> int:
        free._obj.value = self.total - self._used
        total._obj.value = self.total
        return 0

    def rg_host_malloc(self, nbytes: int) -> int:
        buffer = (C.c_uint8 * max(1, nbytes))()
        self._host.append(buffer)  # pinned memory lives until the device closes
        return C.addressof(buffer)

    def _write(self, address: int, pointer: int, nbytes: int) -> None:
        if not self.keep_data or nbytes == 0:
            return
        for base, data in self._data.items():
            if base <= address and address + nbytes <= base + len(data):
                offset = address - base
                C.memmove(
                    (C.c_uint8 * nbytes).from_buffer(data, offset), pointer, nbytes
                )
                return
        raise ValueError(f"upload to {address:#x} is outside every allocation")

    def rg_h2d(self, address: int, pointer: int, nbytes: int) -> int:
        self._call(("h2d", address, nbytes))
        self._write(address, pointer, nbytes)
        return 0

    def rg_h2d_nb(self, address: int, pointer: int, nbytes: int) -> int:
        self._call(("h2d_nb", address, nbytes))
        self._write(address, pointer, nbytes)
        return 0

    def rg_h2d_async(self, address: int, pointer: int, nbytes: int) -> int:
        self._call(("h2d_async", address, nbytes))
        self._write(address, pointer, nbytes)
        return 0

    def rg_d2h(self, pointer: int, address: int, nbytes: int) -> int:
        self._call(("d2h", address, nbytes))
        return 0

    def rg_d2h_async(self, pointer: int, address: int, nbytes: int) -> int:
        self._call(("d2h_async", address, nbytes))
        return 0

    def rg_d2d_async(self, dst: int, src: int, nbytes: int) -> int:
        self._call(("d2d", dst, src, nbytes))
        return 0

    def rg_d2d_multi(self, src, size, dst, n: int) -> int:
        self._call(("d2d_multi", int(n)))
        return 0

    def rg_memset32(self, address: int, value: int, count: int) -> int:
        self._call(("memset32", address, value, count))
        if self.keep_data and count:
            fill = np.full(count, value, np.uint32)
            self._write(address, fill.ctypes.data, fill.nbytes)
        return 0

    # ---- events
    def rg_event_record(self, slot: int) -> int:
        self._call(("record", slot))
        self.recorded_at[slot] = time.monotonic_ns()
        return 0

    def rg_event_timed(self, slot: int) -> int:
        return 0

    def rg_event_elapsed(self, start: int, end: int) -> int:
        """The host time between the two records: work is done when it is issued."""
        if start not in self.recorded_at or end not in self.recorded_at:
            return -1
        return max(0, self.recorded_at[end] - self.recorded_at[start])

    def rg_event_wait(self, slot: int) -> int:
        self._call(("wait", slot))
        return 0

    def rg_event_query(self, slot: int) -> int:
        return 1

    def rg_event_stream_wait(self, slot: int) -> int:
        self._call(("stream_wait", slot))
        return 0

    # ---- graphs and recipes
    def rg_graph_new(self) -> int:
        handle = len(self._graphs)
        self._graphs[handle] = _Graph()
        return handle

    def rg_tensor(self, g: int, name: bytes, dtype, ndims, sizes, persistent) -> int:
        graph = self._graphs[g]
        if persistent:
            graph.persistent.append(name.decode())
        return len(graph.persistent) + 10_000 * (not persistent)

    def rg_tensor_in_section(self, g, name: bytes, dtype, ndims, sizes, other, offset):
        self._graphs[g].persistent.append(name.decode())
        return len(self._graphs[g].persistent)

    def rg_section_rmw(self, g: int, tensor: int) -> int:
        return 0

    def rg_node(self, g, guid, name, ins, n_in, outs, n_out, params, n_params) -> int:
        return 0

    def rg_dep(self, g: int, blocking: int, blocked: int) -> int:
        return 0

    def rg_graph_destroy(self, g: int) -> int:
        self._graphs[g].destroyed = True
        return 0

    def compile(self, graph: Graph) -> int:
        with self._lock:  # RecipeCache.compile_many compiles on threads
            self.compiled_on.append(self.rg_context_current())
            handle = len(self._recipes)
            self._recipes[handle] = _Recipe(
                graph.name, list(self._graphs[graph.handle].persistent)
            )
        return handle

    def rg_recipe_workspace(self, r: int) -> int:
        return 0

    def rg_recipe_name(self, r: int, name: bytes) -> int:
        with self._lock:
            if r not in self._recipes:
                return -1
            self._recipes[r].name = name.decode()
        return 0

    def rg_launch_log_start(self) -> int:
        return 1  # the dry device writes no launch log

    def rg_slot(self, r: int, name: bytes) -> int:
        try:
            return self._recipes[r].persistent.index(name.decode())
        except ValueError:
            return -1

    def rg_bind_slots(self, r: int, slots: int, addresses: int, n: int) -> int:
        recipe = self._recipes[r]
        s = np.ctypeslib.as_array((C.c_int * n).from_address(slots)) if n else []
        a = np.ctypeslib.as_array((C.c_uint64 * n).from_address(addresses)) if n else []
        for slot, address in zip(s, a, strict=True):
            recipe.binds[int(slot)] = int(address)
        return 0

    def rg_launch(self, r: int) -> int:
        recipe = self._recipes[r]
        bound = {recipe.persistent[s]: a for s, a in recipe.binds.items()}
        missing = [n for n in recipe.persistent if not bound.get(n)]
        if missing:
            return -1
        self._call(("launch", recipe.name, bound))
        return 0

    # ---- collectives
    def rg_ccl_allreduce(self, send: int, recv: int, count: int, bf16: int) -> int:
        self._call(("allreduce", send, recv, count, bf16))
        return 0

    def rg_ccl_allgather(self, send: int, recv: int, count: int, bf16: int) -> int:
        self._call(("allgather", send, recv, count, bf16))
        return 0

    def rg_ccl_destroy(self) -> int:
        return 0

    def rg_shutdown(self) -> int:
        return 0

    # ---- reading back (tests)
    def read(self, address: int, nbytes: int) -> bytes:
        """``nbytes`` uploaded at ``address`` (``keep_data`` only)."""
        for base, data in self._data.items():
            if base <= address and address + nbytes <= base + len(data):
                return bytes(data[address - base : address - base + nbytes])
        raise ValueError(f"{address:#x} is outside every allocation")


class DryRecipeCache(RecipeCache):
    """Recipes "compiled" without a card or an artifact cache."""

    def __init__(self, device: Device) -> None:
        self.device = device
        self.compiled: list[str] = []

    @property
    def hits(self) -> int:
        return 0

    @property
    def misses(self) -> int:
        return len(self.compiled)

    def cached(self, graph: Graph) -> bool:
        return False

    def compile(self, graph: Graph) -> Recipe:
        self.compiled.append(graph.name)
        handle = self.device.lib.compile(graph)
        persistent = {n: t for n, t in graph.tensors.items() if t.persistent}
        return Recipe(
            self.device, handle, graph.name, graph.identity(), False, persistent
        )


def dry_libraries() -> NativeLibraries:
    """Library digests for recipe identities; every kernel source is a known kernel."""
    return NativeLibraries(
        runtime=Path("/dry/libresin_gaudi_rt.so"),
        kernels=Path("/dry/kernels"),
        kernel_digests={name: "dry" for name in kernel_sources()},
        glue_digest="dry",
        graph_digest="dry",
        synapse_build="dry",
    )


def dry_device(total_bytes: int = 96 << 30, *, keep_data: bool = False) -> Device:
    """A ``Device`` over ``DryLib``. It is not the process's device (``open_device`` is not
    involved), so tests may make several."""
    return Device(DryLib(total_bytes, keep_data=keep_data), 0, dry_libraries())  # type: ignore[arg-type]
