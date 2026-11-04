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

from resin_models_gaudi.synapse.runtime import Device

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
