"""Host-side pieces of the DFlash 2 drafter of Qwen3.8 (no card): the drafter configuration
checks, block-FP8 weight preparation, the RoPE table, the cache geometry's DRAFT group, and the
tables of an injection and of a block step (cache rows, slot launches with the window and the
scratch pages)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.common import dflash
from resin_models_gaudi.common.dflash import DrafterConfig, DraftRow
from resin_models_gaudi.loading.convert import E4M3, sigma_inv, tau_perm
from resin_models_gaudi.qwen3_5.cache import PAGE_TOKENS, CacheGeometry

TOOLS = Path(__file__).resolve().parents[1] / "tools"

TARGET = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "attention_bias": False,
        "attn_output_gate": True,
        "bos_token_id": 248044,
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25},
        "tie_word_embeddings": False,
        "vocab_size": 248320,
    },
}  # fmt: skip

DRAFTER = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "is_causal": False,
    "use_sliding_window": True,
    "sliding_window": 2048,
    "layer_types": ["sliding_attention"] * 5,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
}


@pytest.fixture
def target() -> Qwen35Config:
    return Qwen35Config.from_hugging_face(TARGET, allow_tied_embeddings=True)


def write_drafter(path: Path, **changes) -> Path:
    config = json.loads(json.dumps(DRAFTER))
    for key, value in changes.items():
        if key in config["dflash_config"]:
            config["dflash_config"][key] = value
        else:
            config[key] = value
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture
def cfg(tmp_path, target) -> DrafterConfig:
    return DrafterConfig.load(write_drafter(tmp_path), target)


def test_config_reads_the_release(cfg):
    assert (cfg.block, cfg.drafts, cfg.taps) == (8, 7, (5, 19, 33, 47, 61))
    assert (cfg.pairs, cfg.layer_rows, cfg.block_rows) == (4, 1024, 5120)
    assert (cfg.qkv_rows, cfg.kv_rows, cfg.kp_rows) == (6144, 2048, 1280)


@pytest.mark.parametrize(
    "changes",
    [
        {"architectures": ["DFlashDraftModel"]},
        {"is_causal": True},
        {"head_dim": 256},
        {"num_key_value_heads": 4},
        {"conv_group_size": 32},
        {"selector_rank": 128},
        {"selector_top_k": 32},
        {"hidden_size": 4096},
        {"num_target_layers": 48},
        {"target_layer_ids": [5, 19, 33, 47, 64]},
    ],
)
def test_config_rejects_other_drafters(tmp_path, target, changes):
    with pytest.raises(ValueError):
        DrafterConfig.load(write_drafter(tmp_path, **changes), target)


def test_config_dims_pair_the_kv_heads(cfg, target):
    from resin_models_gaudi.common.layers import Dims

    d = cfg.dims(Dims.from_config(target))
    assert (d.heads, d.kv_groups, d.head_dim, d.hpg, d.ffn) == (32, 4, 256, 8, 17408)


def test_block_fp8_round_trips_within_a_step():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 512)).astype(np.float32) * np.float32(0.05)
    w[0, 0] = 3.0  # one block with a large scale
    prepared = dflash.block_fp8(w)
    assert prepared.bytes.shape == w.shape and prepared.block_scales.shape == (2, 4)
    # undo the tau order, decode the bytes with their block scales
    inverse = np.argsort(tau_perm(w.shape[1]))
    q = prepared.bytes[:, inverse]
    scale = np.repeat(np.repeat(prepared.block_scales, 128, 0), 128, 1)
    back = E4M3[q] * scale
    step = scale / 8  # half an e4m3 mantissa step at the block's maximum
    assert np.all(np.abs(back - w) <= np.maximum(np.abs(w) / 8, step) + 1e-9)
    assert back[0, 0] == pytest.approx(3.0, rel=1e-6)


def test_rope_table_matches_the_reference_rope(cfg):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(TOOLS))
    from qwen_dflash2_ref import rope_cos_sin

    table = dflash.rope_table(70, cfg)
    cos, sin = rope_cos_sin(torch.arange(70), cfg.head_dim, cfg.rope_theta)
    assert table.shape == (70, 128)
    np.testing.assert_allclose(table[:, :64], cos[:, :64].numpy(), atol=2e-5)
    np.testing.assert_allclose(table[:, 64:], sin[:, :64].numpy(), atol=2e-5)


def test_geometry_adds_a_window_retained_draft_group(cfg, target):
    g = CacheGeometry.from_config(
        target,
        draft_layers=cfg.layers,
        draft_layer_rows=cfg.layer_rows,
        draft_window=cfg.window,
        draft_write=4096,
    )
    layout = g.layout(8)
    assert len(layout.groups) == 3
    group = layout.groups[2]
    assert group.block_tokens == PAGE_TOKENS
    assert group.blocks_per_slab == g.slab_rows // 5120
    assert group.read_window_tokens == 2048
    assert group.window_tokens == 2048 + 4096 - 1
    assert g.draft_block_row(2, 3) == 2 * g.slab_rows + 3 * 5120
    with pytest.raises(ValueError):
        CacheGeometry.from_config(
            target, mtp=True, draft_layers=5, draft_layer_rows=1024
        )


def test_inject_rows_address_each_layer_pair_and_order(cfg):
    blocks = np.array([100000, 200000, 300000])  # logical blocks 3, 4, 5
    row = DraftRow(0, blocks, 3)
    # 4 columns from column 2 at positions 510 .. 513 (crossing a page), column 0 .. 1 unused
    table = dflash.inject_rows(cfg, 8, [(2, 510, 4, row)])
    assert table.shape == (5, 8, 8) and table.dtype == np.int32
    assert (table[:, :2] == -1).all() and (table[:, 6:] == -1).all()
    for layer in range(5):
        for c, pos in enumerate(range(510, 514)):
            base = blocks[pos // 128 - 3] + layer * 1024
            r = pos % 128
            for pg in range(4):
                assert table[layer, 2 + c, pg] == base + pg * 128 + r
                assert table[layer, 2 + c, 4 + pg] == base + (4 + pg) * 128 + sigma_inv(
                    r
                )
    # positions before the retained blocks are older than the window: skipped, not stored
    table = dflash.inject_rows(cfg, 8, [(0, 382, 4, row)])
    assert (table[:, :2] == -1).all() and (table[:, 2:4] != -1).all()
    with pytest.raises(ValueError):
        dflash.inject_rows(cfg, 8, [(0, 760, 10, row)])  # past the retained blocks


def test_block_rows_use_the_scratch_pages(cfg):
    scratch = np.arange(2 * 5 * 4 * 2).reshape(2, 5, 4, 2) * 128 + 5000
    table = dflash.block_rows_table(cfg, 16, scratch, 1)
    assert table.shape == (5, 16, 8)
    assert (table[:, 8:] == -1).all()  # the second row is padding
    for layer in range(5):
        for j in range(8):
            for pg in range(4):
                assert table[layer, j, pg] == scratch[0, layer, pg, 0] + j
                assert table[layer, j, 4 + pg] == scratch[0, layer, pg, 1] + sigma_inv(
                    j
                )


def test_block_slots_cover_the_window_and_the_block(cfg):
    # a row whose block starts at 2100: the window is [53, 2100), two key chunks
    first = 0
    blocks = 10_000 + np.arange(17) * 5120  # logical blocks 0 .. 16
    row = DraftRow(2100, blocks, first)
    scratch = np.arange(1 * 5 * 4 * 2).reshape(1, 5, 4, 2) * 128 + 900_000
    launches = dflash.plan_block_slots(cfg, [row], scratch, 16)
    assert (
        len(launches) == 1
    )  # 4 pairs x (2 chunks + the block) = 12 slots, padded to 16
    slots, kp, vp = launches[0]
    assert slots.shape == (16, 8) and kp.shape == (16, 16) and vp.shape == (16, 16)
    assert (slots[12:, 0] == -1).all()
    chunk0 = slots[:4]
    assert (chunk0[:, 0] == np.arange(4)).all()
    assert (
        chunk0[:, 1:] == [0, 2100, 0, 2100, 0, 53, 2048]
    ).all()  # qcol0, p, key0, klen, 0, kmin, win
    # pages of chunk 0: none below position 53 (page 0), pages 0 .. 15 of the context
    assert kp[0, 0] == blocks[0] and kp[0, 15] == blocks[15]
    assert vp[1, 3] == blocks[3] + (4 + 1) * 128
    chunk1 = slots[4:8]
    assert (chunk1[:, 3:5] == [2048, 2100]).all()
    assert kp[4, 0] == blocks[16] and (kp[4, 1:] == -1).all()
    block = slots[8:12]
    assert (block[:, 1:] == [0, 2100, 2100, 2108, 0, 2100, 2048]).all()
    assert kp[8, 0] == scratch[0, 0, 0, 0] and vp[11, 0] == scratch[0, 0, 3, 1]
    assert (kp[8, 1:] == -1).all()


def test_block_slots_of_a_fresh_sequence_and_a_batch(cfg):
    scratch = np.arange(3 * 5 * 4 * 2).reshape(3, 5, 4, 2) * 128 + 900_000
    rows = [
        DraftRow(0, np.array([100]), 0),  # nothing before the block
        DraftRow(5, np.array([200]), 0),
        DraftRow(
            4096 + 1000, np.array([300 + i * 5120 for i in range(20)]), 23
        ),  # pages 23 .. 42
    ]
    launches = dflash.plan_block_slots(
        cfg, rows, scratch, dflash.slots_per_launch(3, 4)
    )
    slots = np.concatenate([lnch[0] for lnch in launches])
    real = slots[slots[:, 0] >= 0]
    # row 0: the block only; row 1: one chunk and the block; row 2: two chunks and the block
    assert len(real) == 4 * (1 + 2 + 3)
    assert (real[:4, 1] == 0).all() and (real[4:12, 1] == 64).all()
    row2 = real[12:]
    assert (row2[:, 1] == 128).all()
    assert sorted({int(v) for v in row2[:, 3]}) == [2048, 4096, 5096]
    with pytest.raises(
        ValueError
    ):  # the window reaches before the row's retained blocks
        dflash.plan_block_slots(
            cfg, [DraftRow(5000, np.array([1, 2]), 38)], scratch, 16
        )
