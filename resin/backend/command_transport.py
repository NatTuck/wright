"""Typed CPU broadcasts for replica commands and prepared execution plans."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import torch

from resin.backend.plan_transport import PlanPacket, RequestRegistration

_OPERATIONS = (
    "bind",
    "release",
    "forward",
    "finish",
    "close",
    "metrics",
    "metrics_snapshot",
)
_VERSION = 3
_HEADER_SIZE = 13


def encode_command(command) -> list[np.ndarray]:
    """A replica command as arrays in send order: a fixed header, the packed body, then each
    request registration. Empty arrays are left out; decode_command takes the same order."""

    packet = command.get("payload")
    sizes = np.asarray(command.get("sizes", ()), dtype=np.uint64).reshape(-1)
    ids = np.asarray(command.get("ids", ()), dtype=np.uint64)
    error = (command.get("error") or "").encode()
    header = np.array(
        [
            _VERSION,
            _OPERATIONS.index(command["operation"]),
            command.get("step_id", 0),
            int(command.get("phase") == "decode"),
            int(command.get("enqueue", False)),
            int(command.get("feedback", False)),
            command.get("cache_pool_id", 0),
            sizes.size // 2,
            ids.size,
            len(error),
            0 if packet is None else packet.data.size,
            0 if packet is None else len(packet.registrations),
            int(command.get("metrics", False)),
        ],
        dtype=np.uint64,
    )
    pieces = [
        sizes.view(np.uint8),
        ids.view(np.uint8),
        np.frombuffer(error, np.uint8),
    ]
    if packet is not None:
        pieces.append(packet.data)
    body = np.concatenate(pieces)
    arrays = [header]
    if body.size:
        arrays.append(body)
    if packet is not None:
        for registration in packet.registrations:
            arrays.extend(_registration_arrays(registration))
    return arrays


def _registration_arrays(registration) -> list[np.ndarray]:
    descriptor = json.dumps(
        {
            "metadata": registration.metadata.decode(),
            "buffers": [
                (array.dtype.str, array.shape) for array in registration.buffers
            ],
        },
        separators=(",", ":"),
    ).encode()
    arrays = [
        np.array([len(descriptor)], dtype=np.uint64),
        np.frombuffer(descriptor, np.uint8).copy(),
    ]
    arrays.extend(array for array in registration.buffers if array.size)
    return arrays


def decode_command(take: Callable[[np.dtype, tuple[int, ...]], np.ndarray]):
    """Rebuild a command from the arrays of encode_command. ``take(dtype, shape)`` returns the
    next array; for an empty shape it must return an empty array without reading."""

    (
        version,
        operation,
        step,
        phase,
        enqueue,
        feedback,
        cache_pool_id,
        size_count,
        id_count,
        error_size,
        plan_size,
        registrations,
        metrics,
    ) = map(int, take(np.dtype(np.uint64), (_HEADER_SIZE,)))
    if version != _VERSION or operation >= len(_OPERATIONS):
        raise ValueError("unsupported replica command format")
    details_size = (size_count * 2 + id_count) * 8
    body = take(np.dtype(np.uint8), (details_size + error_size + plan_size,))
    details = body[:details_size].view(np.uint64)
    offset = details_size + error_size
    admitted = tuple(_take_registration(take) for _ in range(registrations))
    return {
        "operation": _OPERATIONS[operation],
        "step_id": step,
        "phase": "decode" if phase else "prefill",
        "enqueue": bool(enqueue),
        "feedback": bool(feedback),
        "metrics": bool(metrics),
        "cache_pool_id": cache_pool_id,
        "sizes": tuple(map(tuple, details[: size_count * 2].reshape(-1, 2).tolist())),
        "ids": details[size_count * 2 :].tolist(),
        "error": body[details_size:offset].tobytes().decode() or None,
        "payload": PlanPacket(body[offset:], admitted) if plan_size else None,
    }


def _take_registration(take) -> RequestRegistration:
    size = take(np.dtype(np.uint64), (1,))
    descriptor = take(np.dtype(np.uint8), (int(size[0]),))
    fields = json.loads(descriptor.tobytes())
    buffers = []
    for dtype, shape in fields["buffers"]:
        dtype = np.dtype(dtype)
        if dtype.kind not in "biuf" or dtype.hasobject:
            raise ValueError("request buffers must contain numeric data")
        buffers.append(take(dtype, tuple(shape)))
    return RequestRegistration(fields["metadata"].encode(), tuple(buffers))


class CommandTransport:
    """Typed CPU broadcasts of encode_command's arrays: a fixed header and packed plan
    buffers per forward.

    Request registrations follow only on admission. Their descriptors use JSON
    and their arrays travel as raw buffers, never as serialized Python objects.
    Received packets own their storage through plan reconstruction, including
    while an earlier asynchronous forward is still using its own plan.
    """

    def __init__(self, distributed, group, *, source=0):
        self.distributed, self.group, self.source = distributed, group, source

    def _broadcast(self, array):
        # All traffic uses bytes: Gloo need not support every NumPy scalar dtype.
        # Native prompt-token views are immutable; PyTorch requires writable
        # storage even on the source rank. Pixel buffers already satisfy this.
        if not array.flags.writeable or not array.flags.c_contiguous:
            array = array.copy(order="C")
        # Gloo writes on its own thread, outside the caller's inference scope.
        with torch.inference_mode(False):
            tensor = torch.from_numpy(array.reshape(-1).view(np.uint8))
        self.distributed.broadcast(tensor, src=self.source, group=self.group)

    def _take(self, dtype, shape):
        array = np.empty(shape, dtype)
        if array.size:
            self._broadcast(array)
        return array

    def send(self, command):
        for array in encode_command(command):
            self._broadcast(array)

    def receive(self):
        return decode_command(self._take)
