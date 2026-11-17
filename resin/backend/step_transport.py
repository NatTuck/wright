"""Rank traffic of serving steps inside one request replica, behind one interface.

A DistributedWorker sends replica commands, gates execution on every rank's
readiness, and checks execution status through a StepTransport. Startup
configuration and driver votes do not use it.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import socket
import sys
import tempfile
from typing import Literal, Protocol

import torch

from resin.backend.command_transport import CommandTransport
from resin.backend.control_plane import ControlOperation, ControlPlane
from resin.backend.topology import ParallelGroups, RankGroup

logger = logging.getLogger(__name__)


def first_error(group: RankGroup, errors) -> str | None:
    """The first rank's failure as "rank N: error", or None if all ranks agree."""

    return next(
        (
            f"rank {rank}: {error}"
            for rank, error in zip(group.ranks, errors, strict=True)
            if error
        ),
        None,
    )


class StepTransport(Protocol):
    """Commands from a replica driver to its followers, and the step gates.

    Every method that returns ``str | None`` returns the failure that stops the
    step, as "rank N: error", or None. A method raises only when the transport
    itself fails; the worker then treats the step as failed on every rank.
    Commands keep the dictionary layout of CommandTransport.
    """

    def send(self, command: dict) -> None:
        """Driver: send one command to every follower in the replica."""
        ...

    def receive(self) -> dict:
        """Follower: block until the next command from the driver."""
        ...

    def prepared(self, step_id: int, error: str | None) -> str | None:
        """Every rank of the domain: the gate before the first device collective."""
        ...

    def finished(self, step_id: int, error: str | None) -> str | None:
        """Every rank of the domain: the step's execution status.

        For an enqueued step, the status covers the enqueue and the harvest of
        the previous completion.
        """
        ...

    def acknowledge(self, error: str | None) -> str | None:
        """Every rank of the replica: the status of bind, release or metrics."""
        ...

    def gather_bytes(self, payload: bytes) -> list[bytes] | None:
        """Every rank of the replica: gather one bounded byte record per rank.

        Returns every rank's record in rank order, or None on a rank that does not
        receive them (a follower of the socket transport).
        """
        ...

    def finish(self, error: str | None) -> None:
        """Driver: end the engine step; an error poisons every follower."""
        ...

    def close(self) -> None:
        """Release the transport's resources after the close command."""
        ...


class GlooStepTransport:
    """Typed Gloo broadcasts for commands and VALIDATE votes for the gates."""

    def __init__(self, groups: ParallelGroups):
        self.groups = groups
        self._control = ControlPlane(groups.distributed)
        self._wire = CommandTransport(
            groups.distributed, groups.replica.handle, source=groups.replica.source
        )

    def send(self, command: dict) -> None:
        self._wire.send(command)

    def receive(self) -> dict:
        return self._wire.receive()

    def _validate(self, group: RankGroup, error: str | None) -> str | None:
        _, errors = self._control.gather(group, ControlOperation.VALIDATE, error=error)
        return first_error(group, errors)

    def prepared(self, step_id: int, error: str | None) -> str | None:
        return self._validate(self.groups.domain, error)

    def finished(self, step_id: int, error: str | None) -> str | None:
        return self._validate(self.groups.domain, error)

    def acknowledge(self, error: str | None) -> str | None:
        return self._validate(self.groups.replica, error)

    def gather_bytes(self, payload: bytes) -> list[bytes]:
        group = self.groups.replica
        distributed = self.groups.distributed
        # Snapshot transport is intentionally separate from step packets:
        # bounded, infrequent records without Python object pickling.
        with torch.inference_mode(False):
            size = torch.tensor([len(payload)], dtype=torch.int64)
            sizes = [torch.empty_like(size) for _ in group.ranks]
            distributed.all_gather(sizes, size, group=group.handle)
            capacity = max(int(item.item()) for item in sizes)
            source = torch.zeros(capacity, dtype=torch.uint8)
            source[: len(payload)] = torch.frombuffer(
                bytearray(payload), dtype=torch.uint8
            )
            records = [torch.empty_like(source) for _ in group.ranks]
            distributed.all_gather(records, source, group=group.handle)
        return [
            record[: int(size.item())].numpy().tobytes()
            for record, size in zip(records, sizes, strict=True)
        ]

    def finish(self, error: str | None) -> None:
        self._wire.send({"operation": "finish", "error": error})

    def close(self) -> None:
        pass


StepTransportMode = Literal["auto", "gloo", "socket"]
STEP_TRANSPORTS = ("auto", "gloo", "socket")


def _message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def create_step_transport(
    groups: ParallelGroups, mode: StepTransportMode = "auto", *, timeout: float
) -> StepTransport:
    """The step transport of this rank's replica; a collective over the replica.

    ``socket`` needs every rank of the replica on one Linux host and one replica per
    communication domain (attention DP 1); ``auto`` uses it when those hold, else Gloo.
    ``timeout`` bounds each wait for another rank, in seconds. Every rank returns the
    same kind of transport, or every rank raises.
    """

    if mode not in STEP_TRANSPORTS:
        raise ValueError(f"step transport must be one of {', '.join(STEP_TRANSPORTS)}")
    replica = groups.replica
    if mode == "gloo" or replica.size == 1:
        return GlooStepTransport(groups)
    facts: list = [None] * replica.size
    groups.distributed.all_gather_object(
        facts,
        (
            socket.gethostname(),
            sys.platform == "linux",
            groups.domain.ranks == replica.ranks,
        ),
        group=replica.handle,
    )
    if not (
        len({host for host, _, _ in facts}) == 1
        and all(linux and alone for _, linux, alone in facts)
    ):
        if mode == "socket":
            raise ValueError(
                "the socket step transport needs every rank of a replica on one Linux "
                "host and attention DP 1"
            )
        return GlooStepTransport(groups)
    try:
        return _connect(groups, timeout)
    except _SetupError as error:
        if mode == "socket":
            raise
        # Every rank saw the same failure, so every rank falls back together.
        logger.warning(f"{error}; the ranks use Gloo")
        return GlooStepTransport(groups)


class _SetupError(RuntimeError):
    pass


def socket_directory(prefix: str = "wright-step-", name: str = "channel") -> str:
    """A new private directory in which a socket path for ``name`` fits the 108-byte limit of
    Unix sockets: in the temporary directory, else in /tmp. The caller removes it."""
    directory = tempfile.mkdtemp(prefix=prefix)
    if len(os.path.join(directory, name).encode()) < 100:
        return directory
    os.rmdir(directory)
    return tempfile.mkdtemp(prefix=prefix, dir="/tmp")


def _connect(groups: ParallelGroups, timeout: float) -> StepTransport:
    from resin._resin import (
        StepChannelDriver,
        StepChannelFollower,
    )
    from resin.backend.socket_transport import SocketStepTransport

    replica = groups.replica
    distributed = groups.distributed
    channel, directory, error = None, None, None
    offer: list = [None]
    if groups.is_driver:
        try:
            directory = socket_directory()
            path = os.path.join(directory, "channel")
            token = secrets.token_bytes(16)
            channel = StepChannelDriver(path, token, list(range(1, replica.size)))
            offer = [(path, token, None)]
        except BaseException as exc:
            error = _message(exc)
            offer = [(None, None, error)]
    try:
        distributed.broadcast_object_list(
            offer, src=replica.source, group=replica.handle
        )
        path, token, failure = offer[0]
        if not groups.is_driver and failure is None:
            try:
                channel = StepChannelFollower(path, token, replica.local_rank, timeout)
            except BaseException as exc:
                error = _message(exc)
        # Every rank learns every setup failure, so that all of them stop together.
        for stage in ("connect", "accept"):
            if stage == "accept" and isinstance(channel, StepChannelDriver):
                try:
                    channel.accept(timeout)
                except BaseException as exc:
                    error = _message(exc)
            errors = [None] * replica.size
            distributed.all_gather_object(errors, error, group=replica.handle)
            failure = first_error(replica, errors)
            if failure:
                raise _SetupError(f"the step channel did not start: {failure}")
    except BaseException:
        if channel is not None:
            channel.close()
        raise
    finally:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
    return SocketStepTransport(groups, channel, timeout=timeout)


__all__ = [
    "STEP_TRANSPORTS",
    "GlooStepTransport",
    "StepTransport",
    "StepTransportMode",
    "create_step_transport",
    "first_error",
]
