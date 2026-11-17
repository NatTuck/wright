"""One card shared by a rank of a tensor-parallel model and a one-card replica in the same process.

The TP model goes first. A replica step starts when the TP model has had no step on this card for ``holdoff``
seconds and its last step is done on the device. With a ``share`` above 0, a replica step can also start while
the TP model is busy, as long as the replica has used less than its part of the card's time. A replica step in
the TP model's busy time is charged its device time, which timed events measure.

The two models alternate whole steps, and events order them on the device: a replica step that starts while
the TP model is busy waits on the device for the TP step before it, and the next TP step waits on the device for
that replica step.

Each side calls the gate on its own thread, with its own context current: the TP side through its native
executor (``attach_gate``) or :class:`GatedTransport`, the replica side through ``_ReplicaServer`` (``acquire``
before each Engine step, ``release`` after it). The gate is native (``resin_gaudi::card_gate``): neither side
runs Python to pass it.
"""

from __future__ import annotations

from resin.backend.step_transport import StepTransport

HOLDOFF = 0.005
"""Seconds with no TP step on the card before a replica step starts in the TP model's idle time."""


def CardGate(device, *, share: float = 0.0, holdoff: float = HOLDOFF):
    """The order of the TP model's and the replica's steps on ``device``'s card (see the module docstring): a
    ``resin._resin.CardGate`` over the runtime's events (``resin_gaudi::card_gate``), which the native executors
    and the replica server call without Python."""
    from resin._resin import CardGate as NativeGate

    if share < 0:
        raise ValueError("share takes 0 or more")
    first = device.event_slots(3)  # the TP step, the replica step, a shared replica step's start
    for slot in (first + 2, first + 1):
        device.timed_event(slot)
    return NativeGate(device.glue, first, share, holdoff)


class GatedTransport(StepTransport):
    """A rank's step transport that tells the card's gate when a TP step starts and when its work is on the
    device: a follower's step starts when it receives the forward command, the driver's when it sends it.
    Everything else goes to the transport inside."""

    def __init__(self, inner: StepTransport, gate: CardGate) -> None:
        self._inner = inner
        self._gate = gate
        self._open = False

    def receive(self) -> dict:
        command = self._inner.receive()
        if command.get("operation") == "forward":
            self._gate.tp_begin()
            self._open = True
        return command

    def prepared(self, step_id: int, error: str | None) -> str | None:
        try:
            return self._inner.prepared(step_id, error)
        except BaseException:
            self._end()  # the step stops here: no finished() follows
            raise

    def finished(self, step_id: int, error: str | None) -> str | None:
        try:
            return self._inner.finished(step_id, error)
        finally:
            self._end()

    def _end(self) -> None:
        if self._open:
            self._open = False
            self._gate.tp_end()

    def send(self, command: dict) -> None:
        self._inner.send(command)
        if command.get("operation") == "forward":
            self._gate.tp_begin()
            self._open = True

    def acknowledge(self, error: str | None) -> str | None:
        return self._inner.acknowledge(error)

    def gather_bytes(self, payload: bytes) -> list[bytes] | None:
        return self._inner.gather_bytes(payload)

    def finish(self, error: str | None) -> None:
        self._inner.finish(error)

    def close(self) -> None:
        self._inner.close()
