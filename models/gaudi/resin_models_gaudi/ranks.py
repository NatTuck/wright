"""Tensor-parallel bring-up for Gaudi ranks: one spawned process per card, a torch CPU Gloo
control plane, and one HCCL communicator whose id rank 0 makes and broadcasts over Gloo.

Loading the weights takes minutes per rank and the ranks finish at different times, so the
Gloo timeout must outlast the load; ``barrier`` after loading lines the ranks up before the
first collective.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from resin_models_gaudi.synapse.runtime import Device, open_device

LOAD_TIMEOUT = timedelta(minutes=60)


def accel_index(pci: str) -> int:
    """the ``/sys/class/accel`` index of the card at PCI address ``pci``"""
    for entry in Path("/sys/class/accel").iterdir():
        if Path(os.path.realpath(entry / "device")).name == pci:
            return int(entry.name.removeprefix("accel"))
    raise ValueError(f"no accel device at {pci}")


def init_gloo(rank: int, world: int, port: int, timeout: timedelta = LOAD_TIMEOUT):
    """The default process group over Gloo (CPU only)."""
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
            timeout=timeout,
        )
    return dist


def hccl_rendezvous(device: Device, rank: int, world: int, group=None) -> None:
    """One HCCL communicator over the ranks of ``group`` (default: the world), its id
    broadcast from the group's first rank over Gloo."""
    import torch.distributed as dist

    first = 0 if group is None else dist.get_global_rank(group, 0)
    box = [device.ccl_unique_id() if rank == first else None]
    dist.broadcast_object_list(box, src=first, group=group)
    unique_id = box[0]
    assert isinstance(unique_id, bytes)
    device.ccl_init(
        unique_id, world, rank if group is None else dist.get_group_rank(group, rank)
    )


def barrier(group=None) -> None:
    import torch.distributed as dist

    dist.barrier(group=group)


def tensor_parallel(groups, model: str) -> tuple[int, int]:
    """(rank, ranks) of this process in its tensor-parallel replica (0, 1 without ``groups``).
    The Gaudi runners split models by tensor parallelism only."""
    if groups is None:
        return 0, 1
    topology = groups.topology
    if topology.cp > 1 or topology.ep > 1 or topology.attn_dp > 1:
        raise ValueError(f"{model} on Gaudi supports tensor parallelism only")
    return groups.replica.local_rank, groups.replica.size


def open_rank(groups, card: int | None, model: str) -> tuple[Device, int, int]:
    """This rank's card, and (rank, ranks); several ranks share one HCCL communicator."""
    rank, ranks = tensor_parallel(groups, model)
    device = open_device(card)
    if groups is not None and ranks > 1:
        hccl_rendezvous(device, groups.rank, ranks, groups.replica.handle)
    return device, rank, ranks


def min_over_ranks(value: int, groups) -> int:
    """The smallest ``value`` over the ranks of the replica (free card memory: the arena that
    every rank can hold)."""
    if groups is None or groups.replica.size == 1:
        return value
    import torch
    import torch.distributed as dist

    box = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(box, op=dist.ReduceOp.MIN, group=groups.replica.handle)
    return int(box.item())


def from_first_rank(value, groups):
    """The first rank's ``value`` (any picklable object) on every rank of the replica. Every rank must call it at
    the same point of its work."""
    if groups is None or groups.replica.size == 1:
        return value
    import torch.distributed as dist

    group = groups.replica.handle
    box = [value]
    dist.broadcast_object_list(box, src=dist.get_global_rank(group, 0), group=group)
    return box[0]


def requested_capacity(fit: int, parent_capacity: int | None) -> int:
    """``parent_capacity`` when it fits in ``fit`` slabs, else ``fit``."""
    if parent_capacity is None:
        return fit
    if parent_capacity > fit:
        raise ValueError(f"parent_capacity {parent_capacity} exceeds {fit}")
    return parent_capacity
