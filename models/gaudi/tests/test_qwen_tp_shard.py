"""Tensor-parallel shards of Qwen3.8-27B (no card): ``loading.shard.DenseShard`` partitions the heads and channels,
and ``Qwen35Checkpoint.weights(shard=...)`` gives each rank its part of a layer in the kernel layouts (the
checkpoint from ``RESIN_GAUDI_QWEN35``; layers 0 (GDN) and 3 (attention) only).

A rank's GDN value heads are in its own kernel order (local head j reads local key head j % nk_local); mapped
back to the whole model's kernel order, the ranks' per-head tensors and MLP rows rebuild the unsharded ones."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from resin_models_gaudi.loading.shard import DenseShard

CHECKPOINT = os.environ.get("RESIN_GAUDI_QWEN35", "")
DIMS = dict(heads=24, groups=4, nk=16, nv=48, ffn=17408)  # Qwen3.8-27B


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_shards_partition_the_heads_and_channels(n):
    shards = [DenseShard(r, n, **DIMS) for r in range(n)]
    np.testing.assert_array_equal(
        np.concatenate([s.attn_heads for s in shards]), np.arange(24)
    )
    np.testing.assert_array_equal(
        np.concatenate([s.key_heads for s in shards]), np.arange(16)
    )
    np.testing.assert_array_equal(
        np.concatenate([s.value_heads for s in shards]), np.arange(48)
    )
    # each value head reads a key head of its own rank (checkpoint order: o reads o // 3)
    for s in shards:
        assert set(s.value_heads // 3) == set(s.key_heads)
        assert set(s.attn_heads // 6) == set(
            s.attn_groups
        )  # each query head's KV group is on its rank
    ranges = [s.ffn_range for s in shards]
    assert ranges[0][0] == 0 and ranges[-1][1] == 17408
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert all((b - a) % 256 == 0 for a, b in ranges)
    if (
        n == 8
    ):  # 68 blocks: 9 on the first four ranks, 8 on the others; KV groups on pairs of ranks
        assert [b - a for a, b in ranges] == [2304] * 4 + [2048] * 4
        assert [s.attn_groups.tolist() for s in shards] == [
            [g] for g in (0, 0, 1, 1, 2, 2, 3, 3)
        ]


def test_shards_refuse_what_does_not_divide():
    with pytest.raises(ValueError):
        DenseShard(0, 3, **DIMS)
    with pytest.raises(ValueError):
        DenseShard(0, 16, **DIMS)


@pytest.fixture(scope="module")
def checkpoint():
    if not CHECKPOINT or not (Path(CHECKPOINT) / "config.json").exists():
        pytest.skip(f"no checkpoint at {CHECKPOINT!r} (set RESIN_GAUDI_QWEN35)")
    from resin_models_gaudi.qwen3_5.checkpoint import Qwen35Checkpoint

    return Qwen35Checkpoint(CHECKPOINT)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_every_attention_layer_s_view_holds_the_last_slab(checkpoint, n):
    """Full-attention layer a binds the arena ``a * kv_layer_rows`` rows in. Its view must hold that layer's rows of
    the last KV block of the last slab (a rank's fewer KV groups make kv_layer_rows part of a block), and end inside
    the arena with its padding."""
    from resin_models_gaudi.arena import ARENA_BLOCK_ROWS, ARENA_PAD_BLOCKS
    from resin_models_gaudi.qwen3_5.cache import PAGE_TOKENS, CacheGeometry

    shard = None if n == 1 else checkpoint.shard(0, n)
    g = CacheGeometry.from_config(checkpoint.config, shard=shard)
    capacity = 24
    view = g.attention_blocks(capacity) * ARENA_BLOCK_ROWS
    arena = (g.arena_blocks(capacity) + ARENA_PAD_BLOCKS) * ARENA_BLOCK_ROWS
    # the last KV block, as layer 0 counts rows; each layer's view counts from its own base the same way
    last = capacity * g.slab_rows + (g.kv_blocks_per_slab - 1) * g.kv_block_rows
    assert last + 2 * g.kv_groups * PAGE_TOKENS <= view, (n, last, view)
    assert (g.full_layers - 1) * g.kv_layer_rows + view <= arena, (n, view, arena)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_the_mtp_layer_s_view_holds_every_mtp_block(checkpoint, n):
    """The MTP layer binds the arena at its start. On one card it takes the target's attention recipes, whose view
    ends the last layer's offset before the last slab's end: no MTP block of the last slab may reach past it. A
    split model's MTP layer has recipes of its own over the whole arena."""
    from resin_models_gaudi.arena import ARENA_BLOCK_ROWS
    from resin_models_gaudi.qwen3_5.cache import CacheGeometry

    shard = None if n == 1 else checkpoint.shard(0, n)
    g = CacheGeometry.from_config(
        checkpoint.config, mtp=True, shard=shard, mtp_short_view=n == 1
    )
    capacity = 24
    view = (g.attention_blocks(capacity) if n == 1 else g.arena_blocks(capacity)) * ARENA_BLOCK_ROWS
    end = capacity * g.slab_rows + g.mtp_blocks_per_slab * g.mtp_layer_rows
    assert end <= view, (n, end, view)
    # a block more would not fit: the reserve takes no more than it must
    assert end + g.mtp_layer_rows > view or n > 1, (n, end, view)


def _weights(checkpoint, shard=None) -> dict:
    out = {}
    for name, weight in checkpoint.weights(layers={0, 3}, shard=shard):
        if name.startswith("l"):
            out[name] = weight
    return out


def _array(weight):
    return weight.array if hasattr(weight, "array") else weight.bytes


def test_one_rank_is_the_unsharded_model(checkpoint):
    whole, one = _weights(checkpoint), _weights(checkpoint, checkpoint.shard(0, 1))
    assert whole.keys() == one.keys()
    for name in whole:
        np.testing.assert_array_equal(
            _array(one[name]), _array(whole[name]), err_msg=name
        )
        if hasattr(whole[name], "block_scales"):
            np.testing.assert_array_equal(
                one[name].block_scales, whole[name].block_scales
            )


@pytest.mark.parametrize("n", [2, 4, 8])
def test_rank_parts_rebuild_the_model(checkpoint, n):
    whole = _weights(checkpoint)
    parts = [_weights(checkpoint, checkpoint.shard(r, n)) for r in range(n)]
    nk, nv, r = 16, 48, 3
    nk_l = nk // n
    # per value head tensors: local kernel head j of rank q is the model's kernel head (j // nk_l) * nk + q nk_l
    # + j % nk_l
    for key, rows_of in (("l0.ssm_a", 1), ("l0.dt_bias", 1), ("l0.w_ba", 2)):
        full = _array(whole[key])
        rebuilt = np.zeros_like(full)
        for q, part in enumerate(parts):
            local = _array(part[key])
            nv_l = nv // n
            for j in range(nv_l):
                g = (j // nk_l) * nk + q * nk_l + j % nk_l
                if rows_of == 1:
                    rebuilt[g] = local[j]
                else:  # w_ba: b rows then a rows
                    rebuilt[g] = local[j]
                    rebuilt[nv + g] = local[nv_l + j]
        np.testing.assert_array_equal(rebuilt, full, err_msg=key)
    # MLP rows: [gate | up] of each rank's run of the intermediate channels
    for layer in (0, 3):
        full = _array(whole[f"l{layer}.w_gu"])
        F = full.shape[0] // 2
        gate = np.concatenate(
            [
                _array(p[f"l{layer}.w_gu"])[: p[f"l{layer}.w_gu"].bytes.shape[0] // 2]
                for p in parts
            ]
        )
        up = np.concatenate(
            [
                _array(p[f"l{layer}.w_gu"])[p[f"l{layer}.w_gu"].bytes.shape[0] // 2 :]
                for p in parts
            ]
        )
        np.testing.assert_array_equal(gate, full[:F])
        np.testing.assert_array_equal(up, full[F:])
    # attention: each rank's QKV rows are its heads' [q | gate] then its groups' k and v
    for q, part in enumerate(parts):
        s = checkpoint.shard(q, n)
        rows = _array(part["l3.w_qkv"]).shape[0]
        assert rows == 2 * 256 * s.local_heads + 2 * 256 * s.local_groups
        assert _array(part["l0.w_out"]).shape[1] == 128 * s.local_nv
        assert _array(part["l3.w_down"]).shape[1] == s.local_ffn
    del r
