"""The card gate's TP side in a Python step transport (``GatedTransport``), without a card. The gate's own policy
(the TP model goes first, a replica steps in its idle time or in its share, the two models' steps wait for each
other on the device) is native and tested in resin_gaudi::card_gate."""

from __future__ import annotations

from typing import Any, cast

import pytest

from resin_models_gaudi.synapse.card_gate import CardGate, GatedTransport
from resin_models_gaudi.synapse.dry import dry_device


class _Gate:
    """A card gate's TP side, recorded (the gate itself is native: resin_gaudi::card_gate's tests)."""

    def __init__(self) -> None:
        self.tp_busy = False
        self.calls: list[str] = []

    def tp_begin(self) -> None:
        self.tp_busy = True
        self.calls.append("begin")

    def tp_end(self) -> None:
        self.tp_busy = False
        self.calls.append("end")


def test_the_transport_marks_forward_steps_only():
    class Inner:
        def __init__(self) -> None:
            self.commands = [{"operation": "bind"}, {"operation": "forward"}]

        def receive(self) -> dict:
            return self.commands.pop(0)

        def finished(self, step_id, error):
            return error

        def acknowledge(self, error):
            return "acknowledged"

    gate = _Gate()
    transport = GatedTransport(cast(Any, Inner()), cast(Any, gate))
    assert transport.receive() == {"operation": "bind"}
    assert not gate.tp_busy
    assert transport.acknowledge(None) == "acknowledged"
    transport.receive()
    assert gate.tp_busy
    assert transport.finished(1, "failed") == "failed"
    assert not gate.tp_busy and gate.calls == ["begin", "end"]


def test_the_drivers_transport_marks_the_forward_steps_that_it_sends():
    class Inner:
        def send(self, command) -> None:
            pass

        def finished(self, step_id, error):
            return None

    gate = _Gate()
    transport = GatedTransport(cast(Any, Inner()), cast(Any, gate))
    transport.send({"operation": "release"})
    assert not gate.tp_busy
    transport.send({"operation": "forward"})
    assert gate.tp_busy
    transport.finished(1, None)
    assert not gate.tp_busy


def test_a_step_that_stops_at_its_first_gate_ends_on_the_card():
    class Inner:
        def receive(self) -> dict:
            return {"operation": "forward"}

        def prepared(self, step_id, error):
            raise RuntimeError("another rank failed")

    gate = _Gate()
    transport = GatedTransport(cast(Any, Inner()), cast(Any, gate))
    transport.receive()
    with pytest.raises(RuntimeError, match="another rank"):
        transport.prepared(1, None)
    assert not gate.tp_busy


def test_share_must_not_be_negative():
    with pytest.raises(ValueError):
        CardGate(dry_device(), share=-1)
