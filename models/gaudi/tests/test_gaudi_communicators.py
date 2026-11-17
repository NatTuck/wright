"""Two HCCL communicators in one process, one on each runtime context, over two cards.

A process that serves a model on a group of cards and also a replica that spans cards needs one communicator for
each: the replica's collectives issue on its own context with its own communicator. This test makes the first
communicator on the default context and a second one on a second context (each from its own unique id), runs
all-reduces on both at the same time from two threads, and checks every sum. A third context without a
communicator issues on the process default (the first one), as every context did before.

The cards come from ``VT_HPU_CARDS`` (the grant of the card lock script): the test never takes cards by itself."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

CARDS = re.split(r"[ ,]+", os.environ.get("VT_HPU_CARDS", "").strip())
ROUNDS = 100  # in-place sums double each round: 2^99 keeps every value finite and exact in f32
COUNT = 1 << 16  # f32 elements per all-reduce

pytestmark = pytest.mark.skipif(
    len([c for c in CARDS if c]) < 2
    or not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires two cards granted in VT_HPU_CARDS, the Synapse SDK and tpc-clang",
)


def _worker(rank: int, port: int, cards: list[str], out: str) -> None:
    import torch.distributed as dist

    from resin_models_gaudi import ranks as par
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    world = len(cards)
    par.init_gloo(rank, world, port)
    dev = open_device(par.accel_index(cards[rank]))
    par.hccl_rendezvous(
        dev, rank, world
    )  # the default context's (and the process default) communicator
    second = dev.context()
    with second.use():
        box = [dev.ccl_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        assert isinstance(box[0], bytes)
        dev.ccl_init(box[0], world, rank)
        try:  # one communicator per context
            dev.ccl_init(box[0], world, rank)
            refused = False
        except SynapseError:
            refused = True
    third = dev.context()  # no communicator of its own

    def buffer(value: float) -> int:
        address = dev.malloc(4 * COUNT)
        dev.h2d(address, np.full(COUNT, value, np.float32))
        return address

    a, b, c = buffer(rank + 1.0), buffer(10.0 * (rank + 1)), buffer(100.0 * (rank + 1))
    errors: list[str] = []

    def rounds(context, address: int) -> None:
        try:
            with context.use():
                for _ in range(ROUNDS):
                    dev.allreduce(address, COUNT)
                dev.sync()
        except BaseException as error:  # reported by the parent
            errors.append(repr(error))

    threads = [
        threading.Thread(target=rounds, args=(dev.default_context, a)),
        threading.Thread(target=rounds, args=(second, b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    hung = [t.is_alive() for t in threads]
    rounds(third, c)  # the process default communicator, after the others are done
    with dev.default_context.use():
        dev.sync_all()
    total = sum(range(1, world + 1))
    results = {
        "errors": errors,
        "refused": refused,
        "hung": hung,
        "a": float(dev.d2h(a, COUNT, np.float32)[:4].max()),
        "b": float(dev.d2h(b, COUNT, np.float32)[:4].max()),
        "c": float(dev.d2h(c, COUNT, np.float32)[:4].max()),
        "want": [
            float(total * world ** (ROUNDS - 1)),
            float(10 * total * world ** (ROUNDS - 1)),
            float(100 * total * world ** (ROUNDS - 1)),
        ],
    }
    Path(out).with_suffix(f".{rank}.json").write_text(json.dumps(results))
    par.barrier()
    dev.close()


def test_two_communicators_in_one_process(tmp_path):
    from resin.process import spawn

    entries = [Path(f"/sys/class/accel/accel{i}") for i in CARDS if i][:2]
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "results.json"
    spawn(_worker, args=(port, cards, str(out)), devices=[f"pci:{c}" for c in cards])
    for rank in range(2):
        r = json.loads(out.with_suffix(f".{rank}.json").read_text())
        assert not r["errors"], r["errors"]
        assert r["refused"], "a second communicator on one context was not refused"
        assert not any(r["hung"]), "a context's all-reduces did not finish"
        for got, want in zip((r["a"], r["b"], r["c"]), r["want"], strict=True):
            assert got == pytest.approx(want, rel=1e-6), (rank, got, want)


LATE = 3.0  # seconds that one rank joins a collective late


def _late_peer_worker(
    rank: int, port: int, cards: list[str], out: str, cache: str
) -> None:
    import time

    import torch.distributed as dist

    from resin_models_gaudi import ranks as par
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, open_device

    world = len(cards)
    par.init_gloo(rank, world, port)
    dev = open_device(par.accel_index(cards[rank]))
    par.hccl_rendezvous(dev, rank, world)  # the default context's communicator
    second = dev.context()
    with second.use():
        box = [dev.ccl_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        assert isinstance(box[0], bytes)
        dev.ccl_init(box[0], world, rank)

    def buffer(value: float) -> int:
        address = dev.malloc(4 * COUNT)
        dev.h2d(address, np.full(COUNT, value, np.float32))
        return address

    a, b = buffer(rank + 1.0), buffer(10.0 * (rank + 1))
    graph = Graph(dev, "test_late_peer_mult")
    x, y = graph.input("x", (COUNT,), F32), graph.input("y", (COUNT,), F32)
    graph.node("mult_fwd_f32", [x, y], [graph.input("z", (COUNT,), F32)])
    recipe = RecipeCache(dev, cache).compile(graph)
    xs, ys, zs = buffer(3.0), buffer(3.0), buffer(0.0)
    recipe.bind_all({"x": xs, "y": ys, "z": zs})
    par.barrier()
    result: dict = {}
    if rank == 1:
        dev.allreduce(a, COUNT)  # the default context's collective, at once
        recipe.launch()
        dev.sync()
        time.sleep(LATE)
        with second.use():
            dev.allreduce(b, COUNT)
            dev.sync()
    else:
        start = time.monotonic()
        with second.use():
            dev.allreduce(b, COUNT)  # waits on the device for rank 1
        dev.allreduce(a, COUNT)
        for _ in range(10):
            recipe.launch()
        dev.sync()
        result["other"] = time.monotonic() - start
        with second.use():
            dev.sync()
        result["blocked"] = time.monotonic() - start
    with dev.default_context.use():
        dev.sync_all()
    result["a"] = float(dev.d2h(a, COUNT, np.float32)[:4].max())
    result["b"] = float(dev.d2h(b, COUNT, np.float32)[:4].max())
    result["z"] = float(dev.d2h(zs, COUNT, np.float32)[:4].max())
    Path(out).with_suffix(f".{rank}.json").write_text(json.dumps(result))
    par.barrier()
    dev.close()


def test_a_collective_that_waits_does_not_hold_the_other_context(tmp_path):
    """A model that spans cards beside a tensor-parallel replica on the same cards: when one context's collective
    waits on the device for a late peer, the card still runs the other context's graphs and collectives."""
    from resin.process import spawn

    entries = [Path(f"/sys/class/accel/accel{i}") for i in CARDS if i][:2]
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "late.json"
    spawn(
        _late_peer_worker,
        args=(port, cards, str(out), str(tmp_path / "recipes")),
        devices=[f"pci:{c}" for c in cards],
    )
    first = json.loads(out.with_suffix(".0.json").read_text())
    print(
        f"rank 0: the other context's work {first['other']:.3f} s, "
        f"the late collective {first['blocked']:.3f} s"
    )
    # the second context's collective did wait, and the default context's work did not wait for it
    assert first["blocked"] >= 0.8 * LATE, first
    assert first["other"] < 0.5 * LATE, first
    for rank in range(2):
        r = json.loads(out.with_suffix(f".{rank}.json").read_text())
        assert (r["a"], r["b"], r["z"]) == (3.0, 30.0, 9.0), (rank, r)


ITERATIONS = 64
SLOW = 1 << 24  # f32 elements of the slow rank's extra graph


def _overwrite_worker(
    rank: int, port: int, cards: list[str], out: str, cache: str
) -> None:
    from resin_models_gaudi import ranks as par
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, open_device

    world = len(cards)
    par.init_gloo(rank, world, port)
    dev = open_device(par.accel_index(cards[rank]))
    par.hccl_rendezvous(dev, rank, world)

    def buffer(count: int, value: float) -> int:
        address = dev.malloc(4 * count)
        dev.h2d(address, np.full(count, value, np.float32))
        return address

    def mult(name: str, count: int):
        graph = Graph(dev, name)
        x, y = graph.input("x", (count,), F32), graph.input("y", (count,), F32)
        graph.node("mult_fwd_f32", [x, y], [graph.input("z", (count,), F32)])
        return RecipeCache(dev, cache).compile(graph)

    write = mult("test_overwrite_write", COUNT)
    slow = mult("test_overwrite_slow", SLOW)
    one = buffer(COUNT, 1.0)
    # iteration i writes (rank + 1) * (i + 1) into the partial, all-reduces it in place and copies the sum out;
    # the next iteration's graph writes the partial again at once
    values = [buffer(COUNT, (rank + 1.0) * (i + 1)) for i in range(ITERATIONS)]
    partial = buffer(COUNT, 0.0)
    sums = buffer(COUNT * ITERATIONS, 0.0)
    big = [buffer(SLOW, 1.0) for _ in range(3)]
    slow.bind_all({"x": big[0], "y": big[1], "z": big[2]})
    par.barrier()
    for i in range(ITERATIONS):
        write.bind_all({"x": values[i], "y": one, "z": partial})
        write.launch()
        if rank == 1:
            slow.launch()  # rank 1 reaches each collective later
        dev.allreduce(partial, COUNT)
        dev.d2d(sums + 4 * COUNT * i, partial, 4 * COUNT)
    dev.sync()
    got = dev.d2h(sums, (ITERATIONS, COUNT), np.float32)
    total = np.float32(sum(range(1, world + 1)))
    want = total * np.arange(1, ITERATIONS + 1, dtype=np.float32)
    wrong = [
        (i, float(got[i].min()), float(got[i].max()))
        for i in range(ITERATIONS)
        if not (got[i] == want[i]).all()
    ]
    Path(out).with_suffix(f".{rank}.json").write_text(json.dumps({"wrong": wrong}))
    par.barrier()
    dev.close()


def test_a_partial_written_again_after_its_all_reduce_keeps_every_sum(tmp_path):
    """A graph writes a partial, an in-place all-reduce sums it over the ranks, a copy takes the sum, and the next
    graph writes the partial again at once, while the other rank is late. Every sum must be exact on both ranks."""
    from resin.process import spawn

    entries = [Path(f"/sys/class/accel/accel{i}") for i in CARDS if i][:2]
    cards = [os.path.basename(os.path.realpath(e / "device")) for e in entries]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "overwrite.json"
    spawn(
        _overwrite_worker,
        args=(port, cards, str(out), str(tmp_path / "recipes")),
        devices=[f"pci:{c}" for c in cards],
    )
    for rank in range(2):
        r = json.loads(out.with_suffix(f".{rank}.json").read_text())
        wrong = r["wrong"]
        print(f"rank {rank}: {len(wrong)} of {ITERATIONS} sums wrong: {wrong[:4]}")
    for rank in range(2):
        r = json.loads(out.with_suffix(f".{rank}.json").read_text())
        assert not r["wrong"], (rank, r["wrong"][:8])
