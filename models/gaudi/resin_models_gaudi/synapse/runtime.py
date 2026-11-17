"""The process's Synapse device: memory, copies, events and launches over ctypes.

This module keeps a single ``Device``. Multi-card execution runs one process per card. Work issues on the calling
thread's current context (``DeviceContext``): a stream on its own hardware queue slot, with its own workspace and
staging buffers. A thread starts on the device's default context; a process that serves several models gives the
others contexts of their own.
"""

from __future__ import annotations

import atexit
import ctypes as C
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from resin_models_gaudi import features
from resin_models_gaudi.loading.progress import progress
from resin_models_gaudi.synapse.build import (
    TESTED_SYNAPSE,
    NativeLibraries,
    build_native,
    default_cache_root,
    synapse_build,
)
from resin_models_gaudi.synapse.cards import Card, visible_cards

logger = logging.getLogger("wright")

STOCK_KERNELS = "/usr/lib/habanalabs/libtpc_kernels.so"
_SPIN_WAIT = float(os.environ.get("RESIN_GAUDI_SPIN_WAIT", "0.2"))

# synDataType values (F8 is syn_type_fp8_143: e4m3)
I8, BF16, F32, I16, I32, U8, F8, I64 = 1, 2, 4, 8, 16, 32, 8192, 32768
ITEM_SIZE = {I8: 1, BF16: 2, F32: 4, I16: 2, I32: 4, U8: 1, F8: 1, I64: 8}
NUMPY_DTYPE = {
    I8: np.int8,
    BF16: np.uint16,
    F32: np.float32,
    I16: np.int16,
    I32: np.int32,
    U8: np.uint8,
    F8: np.uint8,
    I64: np.int64,
}

_U64P = C.POINTER(C.c_uint64)
_I32P = C.POINTER(C.c_int)

_SIGNATURES: dict[str, tuple[object, list[object]]] = {
    "rg_last_error": (C.c_char_p, []),
    "rg_init": (C.c_int, [C.c_int]),
    "rg_module_id": (C.c_int, []),
    "rg_driver_version": (C.c_int, [C.c_char_p, C.c_int]),
    "rg_sync": (C.c_int, []),
    "rg_sync_all": (C.c_int, []),
    "rg_context_create": (C.c_int, [C.c_uint64]),
    "rg_context_use": (C.c_int, [C.c_int]),
    "rg_context_current": (C.c_int, []),
    "rg_context_affinity": (C.c_uint64, [C.c_int]),
    "rg_shutdown": (C.c_int, []),
    "rg_malloc": (C.c_uint64, [C.c_uint64]),
    "rg_free": (C.c_int, [C.c_uint64]),
    "rg_mem_info": (C.c_int, [C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]),
    "rg_host_malloc": (C.c_void_p, [C.c_uint64]),
    "rg_host_free": (C.c_int, [C.c_void_p]),
    "rg_h2d": (C.c_int, [C.c_uint64, C.c_void_p, C.c_uint64]),
    "rg_h2d_nb": (C.c_int, [C.c_uint64, C.c_void_p, C.c_uint64]),
    "rg_d2h": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64]),
    "rg_h2d_async": (C.c_int, [C.c_uint64, C.c_void_p, C.c_uint64]),
    "rg_d2h_async": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64]),
    "rg_d2d_async": (C.c_int, [C.c_uint64, C.c_uint64, C.c_uint64]),
    "rg_d2d_multi": (C.c_int, [_U64P, _U64P, _U64P, C.c_uint64]),
    "rg_memset32": (C.c_int, [C.c_uint64, C.c_uint32, C.c_uint64]),
    "rg_event_record": (C.c_int, [C.c_int]),
    "rg_event_wait": (C.c_int, [C.c_int]),
    "rg_event_query": (C.c_int, [C.c_int]),
    "rg_event_spin": (C.c_int, [C.c_int, C.c_int64]),
    "rg_event_stream_wait": (C.c_int, [C.c_int]),
    "rg_event_timed": (C.c_int, [C.c_int]),
    "rg_event_elapsed": (C.c_int64, [C.c_int, C.c_int]),
    "rg_graph_new": (C.c_int, []),
    "rg_tensor": (C.c_int, [C.c_int, C.c_char_p, C.c_int, C.c_int, _U64P, C.c_int]),
    "rg_tensor_in_section": (
        C.c_int,
        [C.c_int, C.c_char_p, C.c_int, C.c_int, _U64P, C.c_int, C.c_uint64],
    ),
    "rg_section_rmw": (C.c_int, [C.c_int, C.c_int]),
    "rg_node": (
        C.c_int,
        [C.c_int, C.c_char_p, C.c_char_p, _I32P, C.c_int, _I32P, C.c_int]
        + [C.c_void_p, C.c_int],
    ),
    "rg_dep": (C.c_int, [C.c_int, C.c_int, C.c_int]),
    "rg_compile": (C.c_int, [C.c_int, C.c_char_p, C.c_char_p]),
    "rg_recipe_load": (C.c_int, [C.c_int, C.c_char_p]),
    "rg_compile_count": (C.c_uint64, []),
    "rg_graph_destroy": (C.c_int, [C.c_int]),
    "rg_recipe_workspace": (C.c_uint64, [C.c_int]),
    "rg_slot": (C.c_int, [C.c_int, C.c_char_p]),
    "rg_bind_slots": (C.c_int, [C.c_int, C.c_void_p, C.c_void_p, C.c_int]),
    "rg_launch": (C.c_int, [C.c_int]),
    "rg_prof_start": (C.c_int, []),
    "rg_prof_stop": (C.c_int, [C.c_char_p]),
    "rg_now_ns": (C.c_int64, []),
    "rg_ccl_id_size": (C.c_int, []),
    "rg_ccl_unique_id": (C.c_int, [C.c_void_p]),
    "rg_ccl_init": (C.c_int, [C.c_void_p, C.c_int, C.c_int]),
    "rg_ccl_allreduce": (C.c_int, [C.c_uint64, C.c_uint64, C.c_uint64, C.c_int]),
    "rg_ccl_allgather": (C.c_int, [C.c_uint64, C.c_uint64, C.c_uint64, C.c_int]),
    "rg_ccl_destroy": (C.c_int, []),
}


class SynapseError(RuntimeError):
    """A failed call of the Synapse runtime. Raised from a step, it may have left device state changed (an
    earlier call of the step may already be on the stream), so the Engine treats it as fatal after accept: it
    stops instead of retrying the step on the same device state."""

    _resin_post_accept = True


def _load(path: Path) -> C.CDLL:
    lib = C.CDLL(str(path))
    for name, (restype, argtypes) in _SIGNATURES.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


def _u64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint64)


class Device:
    """The acquired Gaudi card of this process."""

    def __init__(self, lib: C.CDLL, card: int, libraries: NativeLibraries) -> None:
        self.lib = lib
        self.card = card
        self.libraries = libraries
        self.module_id = lib.rg_module_id()
        self.closed = False
        self._loading = False
        self._events = 0
        self._glue: Any = None
        self._compiles: dict[int, int] = {}
        self._compiles_lock = threading.Lock()

    @property
    def glue(self) -> Any:
        """The glue that native step programs (``resin._resin.GaudiProgram``) run through on this device, or
        None: resin built without the ``gaudi`` feature, or ``RESIN_GAUDI_PROGRAM=0`` (the Python launch loops)."""
        if self._glue is None:
            if os.environ.get("RESIN_GAUDI_PROGRAM", "1") == "0":
                self._glue = False
            else:
                try:
                    from resin._resin import GaudiGlue
                except ImportError:
                    self._glue = False
                else:
                    self._glue = GaudiGlue(str(self.libraries.runtime))
        return self._glue or None

    def check(self, rc: int, what: str) -> int:
        if rc < 0:
            raise SynapseError(f"{what}: {self.lib.rg_last_error().decode()}")
        return rc

    @property
    def driver_version(self) -> str:
        buf = C.create_string_buffer(256)
        self.check(self.lib.rg_driver_version(buf, len(buf)), "driver_version")
        return buf.value.decode()

    @property
    def compile_count(self) -> int:
        """Graph compilations on the calling thread's current context, so that each model of a process counts
        its own; deserialized recipes do not count."""
        return self._compiles.get(int(self.lib.rg_context_current()), 0)

    def count_compile(self) -> None:
        """One more graph compiled on the current context (``RecipeCache``)."""
        with self._compiles_lock:
            context = int(self.lib.rg_context_current())
            self._compiles[context] = self._compiles.get(context, 0) + 1

    def sync(self) -> None:
        """Wait for the work of the current context."""
        self.check(self.lib.rg_sync(), "sync")

    def sync_all(self) -> None:
        """Wait for the work of every context."""
        self.check(self.lib.rg_sync_all(), "sync_all")

    # ---- contexts
    def context(self, affinity: int = 0) -> DeviceContext:
        """A new context: a stream on hardware queue slot ``affinity`` (a mask; 0: the lowest slot that no
        context uses) with its own workspace and staging buffers. Work issues on it inside
        :meth:`DeviceContext.use`. A recipe's workspace comes from the context that loads it."""
        return DeviceContext(
            self, self.check(self.lib.rg_context_create(int(affinity)), "context")
        )

    @property
    def default_context(self) -> DeviceContext:
        """The context that every thread starts on."""
        return DeviceContext(self, 0)

    def current_context(self) -> DeviceContext:
        """The calling thread's current context."""
        return DeviceContext(self, int(self.lib.rg_context_current()))

    # ---- memory
    def malloc(self, nbytes: int) -> int:
        address = self.lib.rg_malloc(int(nbytes))
        if not address:
            raise SynapseError(f"malloc({nbytes}): {self.lib.rg_last_error().decode()}")
        return int(address)

    def free(self, address: int) -> None:
        self.check(self.lib.rg_free(address), "free")

    def memory_info(self) -> tuple[int, int]:
        """(free, total) device memory in bytes."""
        free, total = C.c_uint64(), C.c_uint64()
        self.check(self.lib.rg_mem_info(C.byref(free), C.byref(total)), "mem_info")
        return int(free.value), int(total.value)

    def host_buffer(self, shape: int | tuple[int, ...], dtype) -> np.ndarray:
        """Pinned host memory as a NumPy array; it lives until the device closes."""
        dtype = np.dtype(dtype)
        count = int(np.prod(shape))
        pointer = self.lib.rg_host_malloc(max(1, count * dtype.itemsize))
        if not pointer:
            raise SynapseError(f"host_malloc: {self.lib.rg_last_error().decode()}")
        raw = (C.c_uint8 * (count * dtype.itemsize)).from_address(pointer)
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)

    # ---- copies
    def begin_load(self, total_bytes: int) -> None:
        """Report blocking uploads as the weights phase of startup progress until
        :meth:`end_load`; ``total_bytes`` is an estimate."""
        self._loading = True
        progress.phase("weights", total=total_bytes)

    def end_load(self) -> None:
        if self._loading:
            self._loading = False
            progress.finish_weights()

    def h2d(self, address: int, array: np.ndarray) -> None:
        """Blocking upload."""
        array = np.ascontiguousarray(array)
        self.check(self.lib.rg_h2d(address, array.ctypes.data, array.nbytes), "h2d")
        if self._loading:
            progress.advance(array.nbytes)

    def h2d_nb(self, address: int, array: np.ndarray) -> None:
        """Non-blocking upload of at most 64 KiB."""
        array = np.ascontiguousarray(array)
        self.check(
            self.lib.rg_h2d_nb(address, array.ctypes.data, array.nbytes), "h2d_nb"
        )

    def h2d_async(self, address: int, pinned: np.ndarray) -> None:
        """Non-blocking upload from pinned memory the caller keeps unchanged until it lands."""
        self.check(
            self.lib.rg_h2d_async(address, pinned.ctypes.data, pinned.nbytes),
            "h2d_async",
        )

    def d2h(self, address: int, shape: int | tuple[int, ...], dtype) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        self.check(self.lib.rg_d2h(out.ctypes.data, address, out.nbytes), "d2h")
        return out

    def d2h_async(self, pinned: np.ndarray, address: int) -> None:
        self.check(
            self.lib.rg_d2h_async(pinned.ctypes.data, address, pinned.nbytes),
            "d2h_async",
        )

    def d2d(self, dst: int, src: int, nbytes: int) -> None:
        self.check(self.lib.rg_d2d_async(dst, src, nbytes), "d2d")

    def d2d_multi(self, src: np.ndarray, size: np.ndarray, dst: np.ndarray) -> None:
        """``len(src)`` device-to-device copies."""
        src, size, dst = _u64(src), _u64(size), _u64(dst)
        if not len(src):
            return
        if not (len(src) == len(size) == len(dst)):
            raise ValueError("d2d_multi needs equal-length src, size and dst")
        self.check(
            self.lib.rg_d2d_multi(
                src.ctypes.data_as(_U64P),
                size.ctypes.data_as(_U64P),
                dst.ctypes.data_as(_U64P),
                len(src),
            ),
            "d2d_multi",
        )

    def memset32(self, address: int, value: int, count: int) -> None:
        self.check(self.lib.rg_memset32(address, value, count), "memset32")

    # ---- events
    EVENT_SLOTS = 64
    """The runtime's event slots (``kEvents`` in runtime.cpp), all handed out by ``event_slots``."""

    def event_slots(self, count: int) -> int:
        """The first of ``count`` event slots that no other model of this process uses."""
        first = self._events
        if first + count > self.EVENT_SLOTS:
            raise SynapseError(
                f"event slots: {count} more do not fit in {self.EVENT_SLOTS}"
            )
        self._events += count
        return first

    def record(self, slot: int) -> None:
        self.check(self.lib.rg_event_record(slot), "event_record")

    def wait(self, slot: int, spin: float | None = None) -> None:
        """Wait for event ``slot``: poll it for up to ``spin`` seconds, then block."""
        deadline = time.perf_counter() + (_SPIN_WAIT if spin is None else spin)
        while time.perf_counter() < deadline:
            if self.lib.rg_event_query(slot) == 1:
                return
        self.check(self.lib.rg_event_wait(slot), "event_wait")

    def done(self, slot: int) -> bool:
        return bool(self.check(self.lib.rg_event_query(slot), "event_query"))

    def timed_event(self, slot: int) -> None:
        """Event ``slot`` collects device time for :meth:`elapsed`; call it before the slot's first record."""
        self.check(self.lib.rg_event_timed(slot), "event_timed")

    def elapsed(self, start: int, end: int) -> float | None:
        """Device seconds from what timed slot ``start`` recorded to what timed slot ``end`` recorded, or None
        while either is pending (or its time is gone)."""
        ns = int(self.lib.rg_event_elapsed(start, end))
        if ns == -2:
            return None
        return self.check(ns, "event_elapsed") / 1e9

    def stream_wait(self, slot: int) -> None:
        """The current context's later work waits on the device, not on the host, for what event ``slot``
        recorded (on any context)."""
        self.check(self.lib.rg_event_stream_wait(slot), "stream_wait")

    def now_ns(self) -> int:
        return int(self.lib.rg_now_ns())

    # ---- profiling (trace.enable_profiler must configure the process first)
    def prof_start(self) -> None:
        self.check(self.lib.rg_prof_start(), "prof_start")

    def prof_stop(self, tsv_path: str | Path) -> int:
        """Stop after every context drains; write the events as TSV and return their count."""
        return self.check(self.lib.rg_prof_stop(str(tsv_path).encode()), "prof_stop")

    # ---- collectives (HCCL): one communicator per process
    def ccl_unique_id(self) -> bytes:
        """A new communicator id; rank 0 makes it and the ranks exchange it out of band."""
        buf = C.create_string_buffer(int(self.lib.rg_ccl_id_size()))
        self.check(self.lib.rg_ccl_unique_id(buf), "ccl_unique_id")
        return buf.raw

    def ccl_init(self, unique_id: bytes, nranks: int, rank: int) -> None:
        buf = C.create_string_buffer(unique_id, len(unique_id))
        self.check(self.lib.rg_ccl_init(buf, nranks, rank), "ccl_init")

    # Collectives issue on the current context.
    def allreduce(self, address: int, count: int, *, bf16: bool = False) -> None:
        """In-place sum over the ranks of ``count`` f32 (or bf16) values."""
        self.check(
            self.lib.rg_ccl_allreduce(address, address, count, int(bf16)), "allreduce"
        )

    def allgather(
        self, send: int, recv: int, count: int, *, bf16: bool = False
    ) -> None:
        """``recv`` <- every rank's ``count`` values, rank-major."""
        self.check(self.lib.rg_ccl_allgather(send, recv, count, int(bf16)), "allgather")

    def close(self) -> None:
        global _DEVICE
        if self.closed:
            return
        self.lib.rg_ccl_destroy()
        self.lib.rg_shutdown()
        self.closed = True
        atexit.unregister(self.close)
        if _DEVICE is self:
            _DEVICE = None


@dataclass(frozen=True, slots=True)
class DeviceContext:
    """A context of the device: where the calling thread's work issues inside :meth:`use`."""

    device: Device
    id: int

    @property
    def affinity(self) -> int:
        """The hardware queue slot of the context's stream, as a mask."""
        return int(self.device.lib.rg_context_affinity(self.id))

    @contextmanager
    def use(self):
        """Issue the calling thread's work on this context for the body, then on the context before it."""
        previous = self.device.check(self.device.lib.rg_context_use(self.id), "context")
        try:
            yield self
        finally:
            self.device.lib.rg_context_use(previous)


_DEVICE: Device | None = None


def _log_directory(cache: Path) -> str:
    """The Synapse log directory."""
    current = Path(os.environ.get("HABANA_LOGS") or "/var/log/habana_logs")
    try:
        if os.access(current, os.W_OK) and all(
            os.access(path, os.W_OK) for path in current.iterdir()
        ):
            return str(current)
    except OSError:
        pass
    fallback = cache / "logs"
    fallback.mkdir(parents=True, exist_ok=True)
    logger.info(f"{current} is not writable; Synapse logs go to {fallback}")
    return str(fallback)


def open_device(
    card: int | None = None,
    *,
    libraries: NativeLibraries | None = None,
    wait: float = 60.0,
) -> Device:
    """Acquire accel ``card`` (its ``/sys/class/accel`` index), or the first free visible card.

    Acquisition is retried for up to ``wait`` seconds. An interrupt raises KeyboardInterrupt.
    """
    global _DEVICE
    if _DEVICE is not None:
        if card is not None and card != _DEVICE.card:
            raise SynapseError(
                f"this process already holds card {_DEVICE.card}; Synapse allows one device"
            )
        return _DEVICE
    os.environ["HABANA_LOGS"] = _log_directory(default_cache_root())
    if progress.current() is None:
        progress.phase("kernels")
    libraries = libraries or build_native()
    if not synapse_build().startswith(("synapse-", f"habanalabs-graph-{TESTED_SYNAPSE}")):
        logger.warning(f"{synapse_build()} is not the tested Synapse {TESTED_SYNAPSE}")
    os.environ["GC_KERNEL_PATH"] = f"{libraries.kernels}:{STOCK_KERNELS}"
    os.environ.setdefault("ENABLE_CONSOLE", "false")
    os.environ.setdefault("LOG_LEVEL_ALL", "4")
    candidates = (
        [card.accel for card in visible_cards()] if card is None else [int(card)]
    )
    if not candidates:
        raise SynapseError("init: no operational Gaudi card is visible")
    handlers = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    lib = _load(libraries.runtime)
    deadline = time.monotonic() + wait
    chosen = -1
    while chosen < 0:
        for candidate in candidates:
            chosen = lib.rg_init(candidate)
            if chosen >= 0:
                break
        if threading.current_thread() is threading.main_thread():
            for number, handler in handlers.items():
                if handler is not None:
                    signal.signal(number, handler)
        if chosen < 0:
            if time.monotonic() >= deadline:
                raise SynapseError(f"init: {lib.rg_last_error().decode()}")
            time.sleep(1.0)
    _DEVICE = Device(lib, chosen, libraries)
    atexit.register(_DEVICE.close)
    if features.enabled("skipvalidation") and skipped_launch_validations() is None:
        logger.warning(
            "the skipvalidation feature is on but its library is not preloaded (give -f skipvalidation on the "
            "command line, which preloads it into the ranks): Synapse still checks every launch"
        )
    return _DEVICE


def skipped_launch_validations() -> int | None:
    """Launches whose binding check the ``skipvalidation`` feature skipped in this process, or None when its library
    is not preloaded."""
    try:
        count = C.CDLL(None).resin_gaudi_skipped_validations
    except AttributeError:
        return None
    count.restype = C.c_long
    return int(count())


@dataclass(frozen=True, slots=True)
class Claim:
    card: Card
    binding: Any
    """The ``resin.numa.Binding`` this rank runs under, or None with ``numa="off"``."""


@contextmanager
def claim_card(
    rank: int = 0, *, wait: float = 60.0, numa: str = "auto", module: int | None = None
):
    """Open a visible card for this rank, and stay on its NUMA node for the body.

    The rank starts at its own position in the list and skips busy cards; with ``module``, it
    takes only the card of that module. Each candidate is tried under its own binding. Yields a
    :class:`Claim`, or None when no card came free within ``wait`` seconds.
    """
    from resin import numa as numa_module

    cards = visible_cards()
    order = cards[rank % len(cards) :] + cards[: rank % len(cards)] if cards else []
    if module is not None:
        order = [card for card in cards if card.module == module]
    deadline = time.monotonic() + wait
    while True:
        for card in order:
            placement = (
                nullcontext()
                if numa == "off"
                else numa_module.bind(device=f"pci:{card.pci}", strict=numa == "required")
            )
            with placement as binding:
                try:
                    open_device(card.accel, wait=0.0)
                except SynapseError:
                    continue
                yield Claim(card, binding)
                return
        if time.monotonic() >= deadline:
            yield None
            return
        time.sleep(1.0)


def current_device() -> Device:
    if _DEVICE is None:
        raise SynapseError("no Gaudi device is open in this process")
    return _DEVICE
