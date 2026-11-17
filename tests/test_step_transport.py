"""The socket step transport against Gloo: the same steps, results and failures, plus the
channel's own checks and what happens when a rank dies."""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from output_helpers import forward_output
from test_distributed import _run_case, _tensor_only_case
from test_distributed_overlap import ParallelRunner

from resin import Parallelism, RunnerConfig, SamplingParams
from resin.backend.distributed import (
    DistributedEngine,
    DistributedWorker,
)
from resin.backend.socket_transport import SocketStepTransport
from resin.backend.step_transport import GlooStepTransport, create_step_transport
from resin.backend.topology import ParallelGroups

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux", reason="the step channel is Linux-only"
    ),
    pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="Gloo is required",
    ),
]

CASES = (
    "normal",
    "preflight_failure",
    "forward_failure",
    "malformed_output",
    "release_failure",
    "prepare_failure",
    "initialize_failure",
    "serialize_failure",
)


def _spawn(function, args, nprocs, seconds):
    """Run `function(rank, *args)` on `nprocs` processes; return their exit codes."""
    context = mp.start_processes(
        function, args=args, nprocs=nprocs, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + seconds
    try:
        for child in context.processes:
            child.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(child.is_alive() for child in context.processes):
            raise AssertionError(f"ranks did not finish within {seconds} seconds")
        return [child.exitcode for child in context.processes]
    finally:
        for child in context.processes:
            if child.is_alive():
                child.kill()
            child.join(timeout=5)


def _init(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )


def _equivalence_process(rank, rendezvous, directory):
    _init(rank, 8, rendezvous)
    records = {}
    collective = ParallelGroups.create(
        Parallelism(world_size=8, cp=2, attn_tp=4, ep=4),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    tensor_only = ParallelGroups.create(
        Parallelism(world_size=8, attn_tp=8),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    try:
        for mode in ("gloo", "socket"):
            for case in CASES:
                transport = create_step_transport(collective, mode, timeout=20)
                expected = (
                    SocketStepTransport if mode == "socket" else GlooStepTransport
                )
                assert type(transport) is expected
                records[f"{mode} {case}"] = _run_case(collective, case, transport)
                dist.barrier()
            transport = create_step_transport(tensor_only, mode, timeout=20)
            records[f"{mode} tensor_only"] = _tensor_only_case(tensor_only, transport)
            dist.barrier()
        # auto picks the socket on one host with one replica per domain.
        transport = create_step_transport(collective, "auto", timeout=20)
        assert type(transport) is SocketStepTransport
        transport.close()
        # Attention DP keeps Gloo, and socket refuses it on every rank.
        replicated = ParallelGroups.create(
            Parallelism(world_size=8, cp=2, attn_dp=2, attn_tp=2, ep=4),
            device_backend="gloo",
            timeout=timedelta(seconds=20),
        )
        try:
            transport = create_step_transport(replicated, "auto", timeout=20)
            assert type(transport) is GlooStepTransport
            with pytest.raises(ValueError, match="attention DP 1"):
                create_step_transport(replicated, "socket", timeout=20)
        finally:
            replicated.close()
    finally:
        collective.close()
        tensor_only.close()
        dist.destroy_process_group()
    Path(directory, f"{rank}.json").write_text(json.dumps(records))


def test_socket_steps_match_gloo_in_every_case(tmp_path):
    codes = _spawn(
        _equivalence_process,
        (f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        8,
        150,
    )
    assert codes == [0] * 8
    ranks = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(8)]
    for rank, records in enumerate(ranks):
        for case in (*CASES, "tensor_only"):
            socket, gloo = records[f"socket {case}"], records[f"gloo {case}"]
            if case != "tensor_only":
                # Pool ids count the process's engines, so only their agreement matters.
                for mode, record in (("socket", socket), ("gloo", gloo)):
                    assert record["bindings"] == ranks[0][f"{mode} {case}"]["bindings"]
                socket = {
                    key: value for key, value in socket.items() if key != "bindings"
                }
                gloo = {key: value for key, value in gloo.items() if key != "bindings"}
            assert socket == gloo, (rank, case)
    normal = json.loads((tmp_path / "0.json").read_text())["socket normal"]
    assert not normal["failed"] and list(map(len, normal["results"])) == [3, 2]


def _overlap_process(rank, rendezvous):
    _init(rank, 4, rendezvous)
    groups = ParallelGroups.create(
        Parallelism(world_size=4, cp=2, ep=2), device_backend="gloo"
    )
    try:
        for overlap in (False, True):
            runner = ParallelRunner(groups)
            transport = create_step_transport(groups, "socket", timeout=20)
            worker = DistributedWorker(runner, groups, transport=transport)
            if groups.is_driver:
                engine = DistributedEngine(worker, enable_overlap=overlap)
                request = engine.submit(
                    [1], sampling_params=SamplingParams(max_output_tokens=34)
                )
                events, steps = [], 0
                while engine.has_work():
                    events.extend(engine.step())
                    steps += 1
                    assert steps < 50
                assert request.take_result() == list(range(2, 36))
                assert [event.token_id for event in events] == list(range(2, 36))
                assert (engine.local.overlap_stats[1] > 0) == overlap
                engine.close()
                assert engine.local.cache_free_parent_count == 32
            else:
                worker.serve()
            assert worker._completion is None
            dist.barrier()
    finally:
        groups.close()
        dist.destroy_process_group()


def test_socket_transport_overlaps_decode_with_followers(tmp_path):
    codes = _spawn(_overlap_process, (f"file://{tmp_path / 'rendezvous'}",), 4, 60)
    assert codes == [0] * 4


class _Dying:
    """Tensor-parallel stand-in that kills one rank's process at step 2."""

    runner_config = RunnerConfig()

    def __init__(self, rank, victim, where):
        self.rank, self.victim, self.where = rank, victim, where

    def validate_distributed(self, plan, context):
        if (
            self.where == "validate"
            and self.rank == self.victim
            and context.step_id == 2
        ):
            os._exit(17)

    def forward_distributed(self, plan, context, **_):
        if (
            self.where == "forward"
            and self.rank == self.victim
            and context.step_id == 2
        ):
            os._exit(17)
        if plan is None:
            return None
        return forward_output([7] * plan.num_sequences)


def _kill_process(rank, rendezvous, directory, victim, where):
    _init(rank, 4, rendezvous)
    groups = ParallelGroups.create(
        Parallelism(world_size=4, attn_tp=4), device_backend="gloo"
    )
    transport = create_step_transport(groups, "socket", timeout=20)
    worker = DistributedWorker(_Dying(rank, victim, where), groups, transport=transport)
    record: dict = {}
    started = time.monotonic()
    try:
        if groups.is_driver:
            engine = DistributedEngine(worker)
            engine.submit([1], sampling_params=SamplingParams(max_output_tokens=5))
            while engine.has_work():
                engine.step()
            record["finished"] = True
        else:
            worker.serve()
            record["finished"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = time.monotonic() - started
    Path(directory, f"{rank}.json").write_text(json.dumps(record))
    # Gloo teardown can wait for the dead rank; the record is what the test reads.
    os._exit(0)


@pytest.mark.parametrize(
    ("victim", "where"), [(2, "forward"), (0, "validate"), (0, "forward")]
)
def test_a_dead_rank_stops_the_others_quickly(tmp_path, victim, where):
    codes = _spawn(
        _kill_process,
        (f"file://{tmp_path / 'rendezvous'}", str(tmp_path), victim, where),
        4,
        60,
    )
    assert codes[victim] == 17
    assert all(code == 0 for rank, code in enumerate(codes) if rank != victim)
    for rank in range(4):
        if rank == victim:
            continue
        record = json.loads((tmp_path / f"{rank}.json").read_text())
        assert "error" in record, (rank, record)
        assert record["seconds"] < 10, (rank, record)
        if rank == 0:
            # The driver names the dead follower and the frame it waited for.
            assert record["error"].startswith("DistributedExecutionError")
            assert f"rank {victim}" in record["error"]
            assert "ACCEPTED of step 2" in record["error"]
        else:
            assert "ConnectionError" in record["error"] or "closed" in record["error"]


def _channel(tmp_path, ranks=(1,)):
    from resin._resin import StepChannelDriver

    token = secrets.token_bytes(16)
    path = tmp_path / "channel"
    return StepChannelDriver(str(path), token, list(ranks)), path, token


def _follower(path, token, rank=1, timeout=5.0):
    from resin._resin import StepChannelFollower

    return StepChannelFollower(str(path), token, rank, timeout)


def test_channel_frames_carry_kind_sequence_status_and_payload(tmp_path):
    driver, path, token = _channel(tmp_path, ranks=(1, 2))
    followers = [_follower(path, token, rank) for rank in (2, 1)]
    driver.accept(5.0)
    assert not path.exists()
    assert [rank for rank, _ in driver.followers] == [1, 2]
    assert {pid for _, pid in driver.followers} == {os.getpid()}
    large = np.arange(3 << 20, dtype=np.uint8)
    # A frame larger than the socket buffers needs a reader while it is written.
    sender = threading.Thread(
        target=driver.send,
        args=(2, 9, 0, [np.frombuffer(b"head", np.uint8), large], 5.0, "a command"),
    )
    sender.start()
    for follower in reversed(followers):
        kind, seq, status, payload = follower.receive(5.0, "a command")
        assert (kind, seq, status) == (2, 9, 0)
        assert payload.tobytes() == b"head" + large.tobytes()
        assert payload.flags.writeable
    sender.join()
    followers[1].send(3, 9, 1, [np.frombuffer(b"broken", np.uint8)], 5.0, "a reply")
    followers[0].send(3, 9, 0, [], 5.0, "a reply")
    replies = driver.gather(3, 9, 5.0, "a reply")
    assert [(rank, status, payload.tobytes()) for rank, status, payload in replies] == [
        (1, 1, b"broken"),
        (2, 0, b""),
    ]


def test_channel_rejects_wrong_frames_tokens_and_ranks(tmp_path):
    driver, path, token = _channel(tmp_path)
    follower = _follower(path, token)
    driver.accept(5.0)
    follower.send(3, 4, 0, [], 5.0, "a reply")
    with pytest.raises(RuntimeError, match=r"wanted a reply \(kind 3, sequence 5\)"):
        driver.gather(3, 5, 5.0, "a reply")
    follower.send(4, 5, 0, [], 5.0, "a reply")
    with pytest.raises(RuntimeError, match="got kind 4 with sequence 5 from rank 1"):
        driver.gather(3, 5, 5.0, "a reply")
    with pytest.raises(TimeoutError, match="rank 1 .* sent no a reply"):
        driver.gather(3, 6, 0.2, "a reply")

    (tmp_path / "wrong").mkdir()
    wrong, wrong_path, _ = _channel(tmp_path / "wrong")
    _follower(wrong_path, secrets.token_bytes(16))
    with pytest.raises(RuntimeError, match="wrong HELLO"):
        wrong.accept(5.0)

    (tmp_path / "rank").mkdir()
    unexpected, unexpected_path, unexpected_token = _channel(tmp_path / "rank")
    _follower(unexpected_path, unexpected_token, rank=3)
    with pytest.raises(RuntimeError, match="unexpected rank 3"):
        unexpected.accept(5.0)

    (tmp_path / "late").mkdir()
    late, _, _ = _channel(tmp_path / "late", ranks=(1, 2))
    with pytest.raises(TimeoutError, match="ranks 1, 2 did not join"):
        late.accept(0.1)
    with pytest.raises(RuntimeError, match="listening on .*late/channel"):
        _channel(tmp_path / "late")


def test_channel_reports_a_closed_peer_and_interrupts_waits(tmp_path):
    driver, path, token = _channel(tmp_path)
    follower = _follower(path, token)
    driver.accept(5.0)
    driver.close()
    with pytest.raises(ConnectionError, match="the driver .* closed the step channel"):
        follower.receive(None, "the next command")
    with pytest.raises(ConnectionError):
        follower.send(3, 1, 0, [np.zeros(8, np.uint8)], 5.0, "a reply")

    (tmp_path / "wait").mkdir()
    driver, path, token = _channel(tmp_path / "wait")
    follower = _follower(path, token)
    driver.accept(5.0)
    # A blocked receive releases the GIL: another thread runs and sends meanwhile.
    timer = threading.Timer(0.3, lambda: driver.send(2, 1, 0, [], 5.0, "a command"))
    timer.start()
    started = time.monotonic()
    assert follower.receive(5.0, "a command")[:3] == (2, 1, 0)
    assert 0.25 < time.monotonic() - started < 3
    timer.join()


def _pair(tmp_path, timeout=5.0):
    """A driver and a follower SocketStepTransport over a real channel, in one process."""
    from types import SimpleNamespace

    from resin.backend.topology import RankGroup

    driver, path, token = _channel(tmp_path)
    follower = _follower(path, token)
    driver.accept(5.0)

    def groups(rank):
        group = RankGroup((0, 1), None, rank)
        return SimpleNamespace(
            domain=group, replica=group, is_driver=rank == 0, rank=rank
        )

    return (
        SocketStepTransport(groups(0), driver, timeout=timeout),
        SocketStepTransport(groups(1), follower, timeout=timeout),
    )


def test_a_command_in_place_of_go_fails_the_step_and_comes_next(tmp_path):
    driver, follower = _pair(tmp_path)
    driver.send(
        {"operation": "forward", "step_id": 1, "phase": "decode", "sizes": ((1, 1),)}
    )
    assert follower.receive()["operation"] == "forward"
    # The driver ends the step before GO, as DistributedEngine does after a failure.
    driver.finish("rank 0: injected")
    assert "stopped step 1 before it began" in follower.prepared(1, None)
    command = follower.receive()
    assert (command["operation"], command["error"]) == ("finish", "rank 0: injected")
    driver.send({"operation": "close", "error": "rank 0: injected"})
    assert follower.receive()["operation"] == "close"


def test_a_failed_send_closes_the_channel(tmp_path):
    driver, follower = _pair(tmp_path)
    follower.close()
    with pytest.raises(ConnectionError):
        for _ in range(100):  # the first writes can still land in the socket buffer
            driver.send({"operation": "release", "ids": list(range(4096))})
    with pytest.raises(ConnectionError, match="closed"):
        driver.send({"operation": "close", "error": None})


def test_the_socket_directory_fits_the_path_limit(tmp_path, monkeypatch):
    import tempfile

    from resin.backend import step_transport

    long = tmp_path / ("x" * 90)
    long.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long))
    directory = step_transport.socket_directory()
    try:
        assert directory.startswith("/tmp/wright-step-")
        assert len(os.path.join(directory, "channel").encode()) < 108
    finally:
        os.rmdir(directory)
    # The replica routers of a layout use it with their own names.
    directory = step_transport.socket_directory("wright-replicas-", "replicas-12")
    try:
        assert directory.startswith("/tmp/wright-replicas-")
    finally:
        os.rmdir(directory)
    # A short temporary directory is used as it is.
    short = tempfile.mkdtemp(prefix="s", dir="/tmp")
    monkeypatch.setattr(tempfile, "tempdir", short)
    directory = step_transport.socket_directory()
    try:
        assert directory.startswith(short)
    finally:
        os.rmdir(directory)
        os.rmdir(short)
