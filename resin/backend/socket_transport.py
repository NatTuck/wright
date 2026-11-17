"""The Unix-socket StepTransport: replica commands and step gates without Gloo.

The driver of a request replica holds one socket per follower (resin._resin.StepChannelDriver).
Each step is a few small frames in a fixed order:

- CMD(seq): a command in the layout of encode_command, one frame per follower.
- For a forward: PREPARED(seq) from every follower, then GO(seq) from the driver. GO carries
  the first failure, if any, so that no rank enters a device collective alone. After the
  step, ACCEPTED(seq) from every follower. The driver waits for every ACCEPTED before its
  enqueue returns; followers do not wait for the driver.
- For bind, release and metrics: DONE(seq) from every follower.
- For a metrics snapshot: BYTES(seq) with each follower's record.

A successful engine step sends nothing more; a failed one sends a finish command with the
error. Each side checks the kind and sequence of every frame. A follower or driver that exits
closes its sockets, and the other side's next read or write fails.
"""

from __future__ import annotations

import math

import numpy as np

from resin.backend.command_transport import decode_command, encode_command
from resin.backend.step_transport import first_error
from resin.backend.topology import ParallelGroups

_CMD = 2
_PREPARED = 3
_GO = 4
_ACCEPTED = 5
_DONE = 6
_BYTES = 7


def _bytes(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).reshape(-1).view(np.dtype(np.uint8))


def _text(text: str | None) -> list[np.ndarray]:
    return [] if not text else [np.frombuffer(text.encode(), np.uint8)]


class _Frame:
    """Reads encode_command's arrays back from one received payload."""

    def __init__(self, payload: np.ndarray):
        self.payload, self.offset = payload, 0

    def take(self, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        size = math.prod(shape) * dtype.itemsize
        if size == 0:
            return np.empty(shape, dtype)
        end = self.offset + size
        if end > self.payload.size:
            raise ValueError("a replica command frame ended early")
        piece = self.payload[self.offset : end]
        self.offset = end
        if dtype == np.uint8:
            # A view: each received frame owns new storage.
            return piece.reshape(shape)
        return piece.view(dtype).reshape(shape).copy()

    def finish(self) -> None:
        if self.offset != self.payload.size:
            raise ValueError("a replica command frame has trailing bytes")


class SocketStepTransport:
    """StepTransport over a connected step channel; see the module docstring."""

    def __init__(self, groups: ParallelGroups, channel, *, timeout: float):
        if groups.domain.ranks != groups.replica.ranks:
            raise ValueError("the socket step transport serves one replica per domain")
        self.groups, self.timeout = groups, timeout
        self._channel = channel
        self._driver = groups.is_driver
        self._local = groups.replica.local_rank
        self._seq = 0
        # A command that arrived while a follower waited for GO: the driver stopped the step.
        self._held = None

    def _send(self, kind, seq, status, pieces, what) -> None:
        try:
            self._channel.send(kind, seq, status, pieces, self.timeout, what)
        except BaseException:
            # A failed write can leave part of a frame in a socket: nothing may follow it.
            self._channel.close()
            raise

    def send(self, command: dict) -> None:
        self._seq += 1
        arrays = [_bytes(array) for array in encode_command(command)]
        self._send(_CMD, self._seq, 0, arrays, f"the {command['operation']} command")

    def receive(self) -> dict:
        if self._held is not None:
            (kind, seq, _, payload), self._held = self._held, None
        else:
            kind, seq, _, payload = self._channel.receive(None, "the next command")
        if kind != _CMD or seq != self._seq + 1:
            raise RuntimeError(
                f"step channel protocol error: wanted command {self._seq + 1}, "
                f"got kind {kind} with sequence {seq}"
            )
        self._seq = seq
        frame = _Frame(payload)
        command = decode_command(frame.take)
        frame.finish()
        return command

    def _collect(self, kind: int, error: str | None, what: str) -> str | None:
        replies = self._channel.gather(kind, self._seq, self.timeout, what)
        errors = [error]
        errors.extend(
            (payload.tobytes().decode() or "failed") if status else None
            for _, status, payload in replies
        )
        return first_error(self.groups.replica, errors)

    def _reply(self, kind: int, error: str | None, what: str) -> str | None:
        self._send(kind, self._seq, int(error is not None), _text(error), what)
        return None if error is None else f"rank {self.groups.rank}: {error}"

    def prepared(self, step_id: int, error: str | None) -> str | None:
        what = f"GO of step {step_id}"
        if self._driver:
            try:
                failure = self._collect(_PREPARED, error, f"PREPARED of step {step_id}")
            except BaseException as exc:
                # Stop the followers that are still there before raising.
                text = f"rank {self.groups.rank}: {type(exc).__name__}: {exc}"
                try:
                    self._send(_GO, self._seq, 1, _text(text), what)
                except BaseException:
                    pass
                raise
            self._send(_GO, self._seq, int(failure is not None), _text(failure), what)
            return failure
        self._reply(_PREPARED, error, f"PREPARED of step {step_id}")
        frame = self._channel.receive(self.timeout, what)
        kind, seq, status, payload = frame
        if kind == _CMD and seq == self._seq + 1:
            # The driver ended the step without GO; its command (a finish with the error) is next.
            self._held = frame
            return f"the driver stopped step {step_id} before it began"
        if kind != _GO or seq != self._seq:
            raise RuntimeError(
                f"step channel protocol error: wanted {what} (sequence {self._seq}), "
                f"got kind {kind} with sequence {seq}"
            )
        return (payload.tobytes().decode() or "failed") if status else None

    def finished(self, step_id: int, error: str | None) -> str | None:
        what = f"ACCEPTED of step {step_id}"
        if self._driver:
            return self._collect(_ACCEPTED, error, what)
        return self._reply(_ACCEPTED, error, what)

    def acknowledge(self, error: str | None) -> str | None:
        if self._driver:
            return self._collect(_DONE, error, "DONE")
        return self._reply(_DONE, error, "DONE")

    def gather_bytes(self, payload: bytes) -> list[bytes] | None:
        if self._driver:
            replies = self._channel.gather(_BYTES, self._seq, self.timeout, "BYTES")
            return [payload, *(record.tobytes() for _, _, record in replies)]
        self._send(_BYTES, self._seq, 0, [np.frombuffer(payload, np.uint8)], "BYTES")
        return None

    def finish(self, error: str | None) -> None:
        if error:
            self.send({"operation": "finish", "error": error})

    def close(self) -> None:
        self._channel.close()


__all__ = ["SocketStepTransport"]
