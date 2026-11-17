"""The shared bring-up helpers of the Gaudi tensor-parallel runners, without cards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from resin_models_gaudi.ranks import (
    min_over_ranks,
    requested_capacity,
    tensor_parallel,
)


def _groups(*, cp=1, ep=1, attn_dp=1, rank=3, size=8):
    return SimpleNamespace(
        topology=SimpleNamespace(cp=cp, ep=ep, attn_dp=attn_dp),
        replica=SimpleNamespace(local_rank=rank, size=size, handle=None),
        rank=rank,
    )


def test_tensor_parallel_ranks_and_the_meshes_it_refuses():
    assert tensor_parallel(None, "m") == (0, 1)
    assert tensor_parallel(_groups(), "m") == (3, 8)
    for mesh in ({"cp": 2}, {"ep": 2}, {"attn_dp": 2}):
        with pytest.raises(
            ValueError, match="m on Gaudi supports tensor parallelism only"
        ):
            tensor_parallel(_groups(**mesh), "m")


def test_one_rank_needs_no_collective():
    assert min_over_ranks(123, None) == 123
    assert min_over_ranks(123, _groups(size=1)) == 123


def test_a_requested_capacity_must_fit():
    assert requested_capacity(100, None) == 100
    assert requested_capacity(100, 40) == 40
    with pytest.raises(ValueError, match="parent_capacity 120 exceeds 100"):
        requested_capacity(100, 120)
