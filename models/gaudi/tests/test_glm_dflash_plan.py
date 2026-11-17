"""Host-side pieces of the DFlash 2 drafter of GLM-5.3-Flash (no card): the drafter configuration
against the GLM target, the cache geometry's tails and DRAFT group, the rows of a speculative step,
the block-step row buckets, and the weight items for the GLM loader (the stream mean folded into
``fc``, the head copy)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))  # glm_reference
import glm_reference  # noqa: E402

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi.common.dflash import FP8_MAX, DrafterConfig, DraftRow
from resin_models_gaudi.glm5_3_flash import dflash
from resin_models_gaudi.glm5_3_flash import model as M
from resin_models_gaudi.glm5_3_flash.cache import (
    DRAFT_GROUP,
    MTP_GROUP,
    PAGE_TOKENS,
    CacheGeometry,
    tail_pools_for,
    tail_rows,
)
from resin_models_gaudi.loading.convert import bf16_to_f32, e4m3_encode

DRAFTER = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "is_causal": False,
    "use_sliding_window": True,
    "sliding_window": 2048,
    "layer_types": ["sliding_attention"] * 5,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "vocab_size": 154880,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
    "num_target_layers": 45,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 154856,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 14, 24, 33, 42],
    },
}


def write_drafter(path: Path, config: dict = DRAFTER, **changes) -> Path:
    config = json.loads(json.dumps(config))
    for key, value in changes.items():
        if key in config["dflash_config"]:
            config["dflash_config"][key] = value
        else:
            config[key] = value
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture
def target() -> Glm53Config:
    return Glm53Config()  # GLM-5.3-Flash's own values


@pytest.fixture
def cfg(tmp_path, target) -> DrafterConfig:
    return DrafterConfig.load(write_drafter(tmp_path), target)


def heads(target: Glm53Config, ranks: int) -> int:
    return target.linear_num_heads // ranks


def test_config_fits_the_glm_target(cfg, target):
    assert cfg.hidden == target.hidden_size == 4096
    assert cfg.vocab == target.vocab_size
    assert cfg.taps == (5, 14, 24, 33, 42) and cfg.num_target_layers == 45
    assert cfg.drafts == 7 and cfg.pairs == 4
    assert cfg.layer_rows == 2 * 4 * PAGE_TOKENS and cfg.block_rows == 5 * 1024


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": 5120},
        {"vocab_size": 248320},
        {"num_target_layers": 64},
        {"target_layer_ids": [5, 14, 24, 33, 45]},
    ],
)
def test_config_rejects_another_target(tmp_path, target, changes):
    with pytest.raises(ValueError, match="unsupported drafter"):
        DrafterConfig.load(write_drafter(tmp_path, **changes), target)


def test_tail_pools_follow_the_draft_count():
    assert [tail_pools_for(k) for k in range(8)] == [2, 2, 2, 2, 2, 2, 3, 3]
    assert [int(tail_rows(p)) for p in range(8)] == [0, 2, 4, 6, 6, 8, 10, 12]
    assert [int(tail_rows(p, 3)) for p in range(12)] == [
        0,
        2,
        4,
        6,
        6,
        8,
        10,
        12,
        12,
        14,
        16,
        18,
    ]


def _tail_corruptions(K: int, pools: int) -> list[int]:
    """Anchors (inside one block) whose rejected drafts' tail rows are among the rows a restart at the
    next position reads."""
    blocks = np.array([0, 2000], np.int64)
    bad = []
    for anchor in range(8, 32):
        launch = M.Row(0, anchor, K + 1, blocks, -1, 0)
        _, written = glm_reference._store_rows(K + 1, [launch], keep_tails=True, tail_pools=pools)
        rejected = {int(t) for t in written[1:, 4] if t >= 0}
        restart = M.Row(0, anchor + 1, K + 1, blocks, -1, 0)
        _, read = glm_reference._store_rows(K + 1, [restart], keep_tails=True, tail_pools=pools)
        for sources in read[:, 1:4]:
            for s in sources:
                if s < 0 and -int(s) - 1 in rejected and -int(s) - 1 != written[0, 4]:
                    bad.append(anchor)
    return sorted(set(bad))


@pytest.mark.parametrize("K", range(1, 8))
def test_rejected_drafts_keep_the_tails_their_restart_reads(K):
    assert _tail_corruptions(K, tail_pools_for(K)) == []


def test_two_pool_tails_fail_past_five_drafts():
    assert _tail_corruptions(5, 2) == [] and _tail_corruptions(6, 2) != []


@pytest.mark.parametrize("ranks", [4, 8])
def test_geometry_adds_a_window_retained_draft_group(cfg, target, ranks):
    plain = CacheGeometry.from_config(target, heads(target, ranks))
    g = CacheGeometry.from_config(
        target,
        heads(target, ranks),
        tail_pools=tail_pools_for(7),
        draft_layers=cfg.layers,
        draft_layer_rows=cfg.layer_rows,
        draft_window=cfg.window,
        draft_write=M.PREFILL_T,
    )
    assert plain.layer_rows == 157 and g.layer_rows == 163
    assert g.kv_block_rows % 128 == 0 and g.kv_block_rows > plain.kv_block_rows
    assert g.layer_offset(g.dsa_layers[1]) == 163
    layout = g.layout(8)
    assert len(layout.groups) == 3 and DRAFT_GROUP == MTP_GROUP == 2
    group = layout.groups[DRAFT_GROUP]
    assert group.block_tokens == PAGE_TOKENS
    assert group.blocks_per_slab == g.slab_rows // 5120 >= 1
    assert group.read_window_tokens == 2048
    assert group.window_tokens == 2048 + M.PREFILL_T - 1
    assert g.draft_block_row(2, 3) == 2 * g.slab_rows + 3 * 5120
    with pytest.raises(ValueError, match="alternatives"):
        CacheGeometry.from_config(
            target,
            heads(target, ranks),
            mtp=True,
            draft_layers=5,
            draft_layer_rows=1024,
        )


def test_row_views_its_draft_blocks():
    blocks = np.array([100, 200, 300])
    row = M.Row(
        0, 300, 1, np.zeros(3, np.int64), -1, 0, draft_blocks=blocks, draft_first=1
    )
    assert row.draft_row(305) == DraftRow(305, blocks, 1)
    with pytest.raises(ValueError, match="DRAFT"):
        M.Row(0, 300, 1, np.zeros(3, np.int64), -1, 0).draft_row(300)


def test_row_buckets_fit_the_verify_step():
    assert dflash.row_buckets(M.DECODE_BUCKETS, 7, M.MTP_T) == (1, 2)
    assert dflash.row_buckets(M.DECODE_BUCKETS, 3, M.MTP_T) == (1, 2, 4)
    assert dflash.row_buckets(M.DECODE_BUCKETS, 1, M.MTP_T) == (1, 2, 4, 8)
    # the step's group of rows is the largest bucket, never MTP_T // (K + 1) itself: with 2 drafts
    # that would be 5 rows, and no block-step graph of 5 rows exists
    for K in range(1, 8):
        largest = dflash.row_buckets(M.DECODE_BUCKETS, K, M.MTP_T)[-1]
        assert largest <= M.MTP_T // (K + 1) < 2 * largest
    assert dflash.row_buckets(M.DECODE_BUCKETS, 2, M.MTP_T) == (1, 2, 4)
    assert dflash.row_buckets(M.DECODE_BUCKETS, 4, M.MTP_T) == (1, 2)
    with pytest.raises(ValueError, match="exceeds"):
        dflash.row_buckets(M.DECODE_BUCKETS, 16, M.MTP_T)


def test_lookahead_is_the_draft_count_with_the_drafter():
    from resin_models_gaudi.glm5_3_flash.runner import lookahead_tokens

    assert lookahead_tokens(7, True) == 7 and lookahead_tokens(3, True) == 3
    assert lookahead_tokens(3, False) == 5 and lookahead_tokens(0, False) == 0


# ---- weights
SMALL_DRAFTER = {
    **DRAFTER,
    "hidden_size": 1024,
    "intermediate_size": 256,
    "vocab_size": 512,
    "dflash_config": {**DRAFTER["dflash_config"], "mask_token_id": 500},
}


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """bf16 (uint16 bits) tensors as a safetensors file tagged BF16 (the numpy writer has no bf16)."""
    header, offset = {}, 0
    for name, array in tensors.items():
        assert array.dtype == np.uint16
        header[name] = {
            "dtype": "BF16",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    meta = json.dumps(header).encode()
    meta += b" " * (-len(meta) % 8)
    with open(path, "wb") as f:
        f.write(len(meta).to_bytes(8, "little"))
        f.write(meta)
        for array in tensors.values():
            f.write(np.ascontiguousarray(array).tobytes())


def _bf16(rng, shape) -> np.ndarray:
    return (
        rng.standard_normal(shape)
        .astype(np.float32)
        .view(np.uint32)
        .__rshift__(16)
        .astype(np.uint16)
    )


def write_small_drafter(path: Path, rng) -> DrafterConfig:
    """A drafter of hidden 1024, SwiGLU 256 and 512 tokens, bf16, with random weights."""
    write_drafter(path, SMALL_DRAFTER)
    H, F, V, R = 1024, 256, 512, 256
    tensors = {
        "fc.weight": _bf16(rng, (H, 5 * H)),
        "hidden_norm.weight": _bf16(rng, (H,)),
        "norm.weight": _bf16(rng, (H,)),
        "candidate_selector.hidden_projection.weight": _bf16(rng, (R, H)),
        "candidate_selector.predecessor_codebook": _bf16(rng, (V, R)),
        "candidate_selector.successor_codebook": _bf16(rng, (V, R)),
    }
    for layer in range(5):
        p = f"layers.{layer}."
        tensors.update(
            {
                p + "input_layernorm.weight": _bf16(rng, (H,)),
                p + "post_attention_layernorm.weight": _bf16(rng, (H,)),
                p + "self_attn.q_norm.weight": _bf16(rng, (128,)),
                p + "self_attn.k_norm.weight": _bf16(rng, (128,)),
                p + "self_attn.q_proj.weight": _bf16(rng, (32 * 128, H)),
                p + "self_attn.k_proj.weight": _bf16(rng, (8 * 128, H)),
                p + "self_attn.v_proj.weight": _bf16(rng, (8 * 128, H)),
                p + "self_attn.o_proj.weight": _bf16(rng, (H, 32 * 128)),
                p + "mlp.gate_proj.weight": _bf16(rng, (F, H)),
                p + "mlp.up_proj.weight": _bf16(rng, (F, H)),
                p + "mlp.down_proj.weight": _bf16(rng, (H, F)),
                p + "attention_conv.base_kernel": _bf16(rng, (2, 2, H)),
                p + "attention_conv.kernel_projection.weight": _bf16(
                    rng, (2 * 2 * H // 16, H)
                ),
                p + "mlp_conv.base_kernel": _bf16(rng, (2, 2, H)),
                p + "mlp_conv.kernel_projection.weight": _bf16(
                    rng, (2 * 2 * H // 16, H)
                ),
            }
        )
    _write_safetensors(path / "model.safetensors", tensors)
    target = SimpleNamespace(hidden_size=H, vocab_size=V, num_hidden_layers=45)
    return DrafterConfig.load(path, target)


def write_small_target(path: Path, rng) -> None:
    """A GLM-shaped release with only the LM head (512 tokens of 1024)."""
    tensors = {"lm_head.weight": _bf16(rng, (512, 1024))}
    _write_safetensors(path / "model-00001-of-00001.safetensors", tensors)
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in tensors
                }
            }
        )
    )


def test_drafter_items_fold_the_stream_mean_and_copy_the_head(tmp_path):
    from resin_models_gaudi.glm5_3_flash.weights import NS, Fp8
    from resin_models_gaudi.loading.convert import kperm
    from resin_models_gaudi.loading.source import Safetensors

    rng = np.random.default_rng(0)
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    cfg = write_small_drafter(drafter, rng)
    target = tmp_path / "target"
    target.mkdir()
    write_small_target(target, rng)
    src = Safetensors(target)
    items = {
        name: (payload, kind)
        for name, payload, kind in dflash.drafter_items(drafter, cfg, src)
    }
    fc_raw = Safetensors(drafter, mode="mmap").raw("fc.weight")[0]
    for k in range(5):
        fc, kind = items[f"dfl.fc{k}"]
        assert kind == "plain" and isinstance(fc, np.ndarray)
        expected = bf16_to_f32(fc_raw[:, k * 1024 : (k + 1) * 1024]) / NS
        np.testing.assert_array_equal(bf16_to_f32(fc), expected)
    for name in (
        "d0.w_qkv",
        "d0.w_kv",
        "d0.w_o",
        "d0.w_gu",
        "d0.w_down",
        "d0.kp_a",
        "d0.kp_m",
        "d4.w_down",
    ):
        payload, kind = items[name]
        assert kind == "inc8"
        assert isinstance(payload, Fp8)
        M_, K_ = payload.bytes.shape
        assert (
            M_ % 128 == 0
            and K_ % 256 == 0
            and payload.block_scales.shape == (M_ // 128, K_ // 128)
        )
    # Gaudi FP8 already, as block_fp8 makes it: no byte reaches FP8_MAX's encoding twice over
    w, _ = items["d0.w_down"]
    assert isinstance(w, Fp8)
    assert (
        np.abs(w.block_scales).max() > 0
        and w.bytes.max() <= e4m3_encode(np.array([FP8_MAX], np.float32))[0] | 0x80
    )
    head, kind = items[dflash.HEAD]
    assert kind == "bf16" and isinstance(head, np.ndarray)
    assert head.shape == (512, 1024)
    np.testing.assert_array_equal(head, kperm(raw_head := src.bf16("lm_head.weight")))
    assert raw_head.shape == (512, 1024)
    assert (
        dflash.drafter_weight_bytes(cfg, drafter)
        == (drafter / "model.safetensors").stat().st_size + 2 * 512 * 1024
    )


def test_target_dims_carry_the_hidden_size_and_vocabulary(cfg):
    d = dflash.target_dims(4096, 154880, cfg.eps)
    assert (d.hidden, d.vocab, d.eps) == (4096, 154880, cfg.eps)
    drafter = cfg.dims(d)
    assert (
        drafter.hidden,
        drafter.ffn,
        drafter.heads,
        drafter.kv_groups,
        drafter.head_dim,
    ) == (
        4096,
        12288,
        32,
        4,
        256,
    )
