"""Qwen3.8-Flash-Next's native decode-step fill (``resin._resin.FnextFill``, the ``gaudi`` feature) writes the same
table image, byte for byte, as the numpy fill (``Qwen4ExpGaudi._fill``), on CPU: a model object with the
checkpoint's geometry (``RESIN_GAUDI_FNEXT``'s config.json; no weights, no card), one rank of four, every decode
bucket, rows at random positions, fresh and continuing, with and without n-gram history."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import resin._resin as native

CHECKPOINT = Path(os.environ.get("RESIN_GAUDI_FNEXT", ""))
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RESIN_GAUDI_FNEXT") or not (CHECKPOINT / "config.json").exists(),
        reason="RESIN_GAUDI_FNEXT is not set",
    ),
    pytest.mark.skipif(
        not hasattr(native, "FnextFill"), reason="resin built without the gaudi feature"
    ),
]
MAX_POSITIONS = 2048


def model(rank: int = 1, nranks: int = 4, table: bool = True):
    from resin.models.qwen4_exp.config import Qwen4ExpConfig
    from resin_models_gaudi.qwen4_exp import model as M
    from resin_models_gaudi.qwen4_exp.cache import CacheGeometry
    from resin_models_gaudi.qwen4_exp.checkpoint import Shard
    from resin_models_gaudi.qwen4_exp.layers import Dims
    from resin_models_gaudi.qwen4_exp.ngram import NGram

    c = Qwen4ExpConfig.from_hugging_face(
        json.loads((CHECKPOINT / "config.json").read_text())
    )
    m = object.__new__(M.Qwen4ExpGaudi)
    m.cfg = c
    m.n_layer = c.num_hidden_layers
    m.layer_ids = list(range(m.n_layer))
    m.full = [i for i in m.layer_ids if c.is_full(i)]
    m.linear = [i for i in m.layer_ids if not c.is_full(i)]
    m.ple_layer = c.ple_layers[0]
    m.has_table = table
    m.ngram = NGram(
        c.vocab_size,
        c.ngram_size,
        c.heads_per_ngram,
        c.ngram_vocab_size_base,
        c.eos_token_id,
    )
    m.max_positions = MAX_POSITIONS
    m.decode_buckets, m.prefill_buckets = M.DECODE_BUCKETS, (128,)
    m.dense_ranks = nranks
    shard = Shard(
        rank, nranks, c.linear_num_key_heads, c.linear_num_value_heads, c.num_attention_heads,
        c.num_key_value_heads, c.num_experts, c.vocab_size, dense_ranks=nranks,
    )  # fmt: skip
    # this rank's slice of the table (its rows and the zero row): a quarter of the rows, somewhere inside
    total = m.ngram.rows_total
    m.table_first = rank * total // nranks
    table_rows = (total // nranks + 1) if table else 0
    m.d = Dims.from_config(c, shard, table_rows=table_rows, gather_tokens=16)
    m.geo = CacheGeometry.from_dims(
        m.d, full_layers=len(m.full), linear_layers=len(m.linear), ple=True
    )
    size = m._layouts()
    m._native = m._native_fill()
    assert m._native is not None
    return m, size


def step(m, T: int, rng):
    """A random decode step of up to ``T`` rows: (columns, params)."""
    from resin_models_gaudi.qwen4_exp.cache import PAGE_TOKENS
    from resin_models_gaudi.qwen4_exp.model import RowCache

    n = int(rng.integers(1, T + 1))
    columns = []
    for b in range(n):
        pos0 = int(rng.integers(0, MAX_POSITIONS - 1))
        pages = pos0 // PAGE_TOKENS + 1 + int(rng.integers(0, 3))
        kv = (rng.integers(1, 1 << 14, pages) * m.geo.kv_block_rows).astype(np.int64)
        src = (
            -1 if pos0 == 0 else int(rng.integers(1, 1 << 10)) * m.geo.state_block_rows
        )
        dst = int(rng.integers(1, 1 << 10)) * m.geo.state_block_rows
        history = (-1, -1) if pos0 == 0 else (
            int(rng.choice([-1, m.cfg.eos_token_id, int(rng.integers(0, m.cfg.vocab_size))])),
            int(rng.choice([m.cfg.eos_token_id, int(rng.integers(0, m.cfg.vocab_size))])),
        )  # fmt: skip
        row = RowCache(pos0, kv, src, dst, 0, (dst,), history)
        token = int(
            rng.choice([m.cfg.eos_token_id, int(rng.integers(0, m.cfg.vocab_size))])
        )
        columns.append((row, b, [token]))
    params = rng.random((n, 8)).astype(np.float32)
    return columns, params


@pytest.mark.parametrize(
    ("ranks", "table"),
    [(4, True), (4, False), (1, True)],
    ids=["tp4-table", "tp4-no-table", "tp1-table"],
)
def test_native_fill_matches_numpy(ranks, table):
    """(one rank holds both KV groups: the raw-key and block-key pages sit after two K and two V pages)"""
    from resin_models_gaudi.qwen4_exp.model import DECODE_BUCKETS

    m, size = model(rank=min(1, ranks - 1), nranks=ranks, table=table)
    rng = np.random.default_rng(3)
    for T in DECODE_BUCKETS:
        lay = m.layouts[T, True]
        for _ in range(12):
            columns, params = step(m, T, rng)
            n = len(columns)
            want = np.full(size // 4, 0x5A5A5A5A, np.int32)
            got = want.copy()
            rows = [row for row, _, _ in columns]
            tokens = [ids for _, _, ids in columns]
            m._fill(lay, want, columns, T, n, True, False, params, rows, tokens)
            m._fill_native(lay, got, columns, T, params)
            if not np.array_equal(want[: lay.size // 4], got[: lay.size // 4]):
                bad = int(np.flatnonzero(want != got)[0]) * 4
                name = max((o, k) for k, o in lay.at.items() if o <= bad)[1]
                pytest.fail(
                    f"T={T} n={n}: native differs at byte {bad} ({name}): {want[bad // 4]} vs {got[bad // 4]}"
                )
