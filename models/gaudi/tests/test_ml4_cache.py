"""Mistral-Large-4 cache geometry (host only): page offsets of owning and sharing layers, store
destinations, arena sizing, and the config's per-layer topology."""

from __future__ import annotations

import numpy as np
import pytest

from resin.models.mistral_large4.config import MistralLarge4Config


def config(layers: int = 72) -> MistralLarge4Config:
    pattern = "full," * 17 + "shared,shared,full,full," * 13 + "shared,shared,full"
    sources = list(range(39)) + [35 if i % 4 == 3 else i for i in range(39, 72)]
    params = {
        "dim": 6144, "n_layers": 72, "hidden_dim": 24576, "n_heads": 64, "rope_theta": 1e4,
        "norm_eps": 1e-5, "vocab_size": 204800, "max_position_embeddings": 1 << 20,
        "nope": "rope,rope,rope,nope", "softmax_tanh_gating": 30.0, "q_lora_rank": None,
        "qk_rope_head_dim": 64, "qk_nope_head_dim": 64, "kv_lora_rank": 512, "v_head_dim": 128,
        "dsa": {"topk": 2048, "n_heads": 32, "head_dim": 128, "rope_head_dim": 64,
                "norm_eps": 1e-6, "sharing": pattern.rstrip(",")},
        "yarn": {"original_max_position_embeddings": 512, "factor": 2048, "apply_scale": False,
                 "beta": 32, "alpha": 1},
        "kv_sharing": {"source_layers_str": ",".join(map(str, sources))},
        "moe": {"first_k_dense_replace": 1, "num_experts": 128, "num_experts_per_tok": 4,
                "num_expert_groups": 1, "routed_scale": 1.0, "expert_hidden_dim": 6144,
                "num_shared_experts": 1},
    }  # fmt: skip
    cfg = MistralLarge4Config.from_params(params)
    return cfg if layers == 72 else cfg.truncated(layers)


def test_topology():
    cfg = config()
    assert len(cfg.kv_owner_layers) == 63 and len(cfg.index_owner_layers) == 44
    assert cfg.kv_sources[39] == 35 and not cfg.owns_kv(39) and cfg.index_owners[39]
    assert cfg.index_source(17) == 16 and cfg.index_source(19) == 19
    assert cfg.key_dim(0) == 576 and cfg.key_dim(3) == 512 and cfg.nope_dim(3) == 128
    with pytest.raises(ValueError, match="query LoRA"):
        MistralLarge4Config.from_params({"n_layers": 1, "q_lora_rank": 64})


def test_geometry_offsets_and_rows():
    from resin_models_gaudi.mistral_large4.cache import (
        INDEX_PAGE_ROWS,
        LATENT_PAGE_ROWS,
        ROPE_PAGE_ROWS,
        CacheGeometry,
    )

    cfg = config()
    g = CacheGeometry.from_config(cfg)
    assert g.block_rows == 12672 and g.block_bytes * 8 % (1024 * 512) == 0
    assert g.block_bytes / 128 < 51 * 1024  # bytes per token and card
    latent = {g.latent_offset(i) for i in cfg.kv_owner_layers}
    assert len(latent) == 63 and max(latent) == 62 * LATENT_PAGE_ROWS
    assert g.latent_offset(39) == g.latent_offset(35) and g.rope_offset(39) == -1
    assert g.rope_offset(0) >= 63 * LATENT_PAGE_ROWS + 44 * INDEX_PAGE_ROWS
    pages = sorted(
        [(g.latent_offset(i), LATENT_PAGE_ROWS) for i in cfg.kv_owner_layers]
        + [(g.index_offset(i), INDEX_PAGE_ROWS) for i in cfg.index_owner_layers]
        + [(g.rope_offset(i), ROPE_PAGE_ROWS) for i in g.rope_owners]
    )
    assert all(a + n <= b for (a, n), (b, _) in zip(pages, pages[1:], strict=False))
    assert pages[-1][0] + pages[-1][1] <= g.block_rows
    blocks = g.block_row(np.array([3, 7]))
    rows = g.store_rows(0, blocks, np.array([5, 130]))
    assert rows[0, 0] == blocks[0] + 5 and rows[1, 0] == blocks[1] + 2
    assert (
        rows[0, 3] == blocks[0] + 128 and rows[0, 4] == 5
    )  # the scale row after the token rows
    assert rows[0, 1] == 4 * (blocks[0] + g.rope_offset(0)) + 5
    assert rows[1, 2] == 2 * (blocks[1] + g.index_offset(0)) + 2
    shared = g.store_rows(39, blocks, np.array([5, 130]))  # stores only its index keys
    assert (shared[:, [0, 1, 3]] == -1).all() and (shared[:, 2] >= 0).all()
    padding = g.store_rows(0, np.array([-1]), np.array([0]))
    assert (padding == -1).all()


def test_prefill_widths():
    model = pytest.importorskip("resin_models_gaudi.mistral_large4.model")

    assert model.prefill_buckets(1024) == (128, 1024)
    assert model.prefill_buckets(4096) == (128, 1024, 4096)
    for columns in (512, 3000):
        with pytest.raises(ValueError, match="power of two"):
            model.prefill_buckets(columns)
    for n, columns, step in (
        (48, 4096, 16),
        (49, 4096, 1024),
        (2048, 4096, 1024),
        (2049, 4096, 4096),
        (4096, 4096, 4096),
        (1024, 2048, 1024),
        (1025, 2048, 2048),
        (49, 1024, 1024),
        (1024, 1024, 1024),
    ):
        assert model.prefill_chunks(n, columns).step == step, (n, columns)


def test_capacity_rounds_slabs():
    from resin_models_gaudi.mistral_large4.cache import SLAB_MULTIPLE, CacheGeometry

    for layers in (4, 72):
        g = CacheGeometry.from_config(config(layers))
        cap = g.capacity_for(14 << 30, g.block_bytes)
        assert (cap + 1) % SLAB_MULTIPLE == 0
        assert (cap + 1) * g.block_bytes <= 14 << 30
        assert g.arena_blocks(cap) * 1024 == (cap + 1) * g.block_rows
