"""GLM-5.3-Flash's native table fill (``resin._resin.GlmFill``) against the numpy fill it replaced
(``glm_reference``), byte for byte, on the CPU:
decode, verify and prefill steps and MTP passes over the cases the tables branch on (open pools, the indexer,
sparse attention, every KDA record kind)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from glm_reference import Reference  # noqa: E402

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_gaudi.glm5_3_flash.cache import POOL, CacheGeometry, tail_pools_for
from resin_models_gaudi.glm5_3_flash.layers import SLOT_POOLS, Dims
from resin_models_gaudi.glm5_3_flash.model import (
    DECODE_BUCKETS,
    MAX_T,
    PAGE_TOKENS,
    PREFILL_BUCKETS,
    TMAX,
    Filled,
    Glm53Gaudi,
    Row,
    bucket,
)
from resin_models_gaudi.glm5_3_flash.weights import RankSlices

_resin = pytest.importorskip("resin._resin", reason="the native extension")
pytestmark = pytest.mark.skipif(
    not hasattr(_resin, "GlmFill"), reason="the native extension was built without the gaudi feature"
)

MAX_SEQ = 20000


class HostModel(Glm53Gaudi):
    """A Glm53Gaudi with only what the table fills read (no device), in check mode: ``_fill`` runs both fills
    and counts its checks."""

    fill_checks: int
    _fill: Callable[..., Filled]


def host_model(nranks: int = 4, drafts: int = 0, mtp: bool = False) -> HostModel:
    """A HostModel of ``nranks`` ranks"""
    cfg = Glm53Config()
    m = object.__new__(HostModel)
    sl = RankSlices.of(cfg, 0, nranks)
    heads = sl.kda_heads.stop - sl.kda_heads.start
    m.geo = CacheGeometry.from_config(
        cfg, heads, mtp=mtp, tail_pools=tail_pools_for(drafts)
    )
    m.d = Dims.of(cfg, sl, m.geo.conv_rows)
    m.max_seq_len = MAX_SEQ
    positions = MAX_SEQ + TMAX
    unit = max(SLOT_POOLS, 4096)
    m.index_pools = -(-(positions // POOL + 1) // unit) * unit
    m.pages = -(-positions // PAGE_TOKENS)
    m.dsa, m.kda = m.geo.dsa_layers, m.geo.kda_layers
    m._dsa_off = np.array([m.geo.layer_offset(i) for i in m.dsa], np.int64)
    m._mtp_off = np.zeros(1, np.int64)
    m.tables = m._step_tables()
    m._mtables = m._mtp_tables()
    m._native = m._native_fill()
    assert m._native is not None
    m.fill_checks = 0
    m._fill = lambda *args, **kwargs: check(m, *args, **kwargs)  # noqa: E731
    return m


def check(m, h, tb, mtp, rows, n, first, tokens, *, head=False, verify=0, keep_tails=False, meta_columns=None):
    """Both fills from the same bytes (words neither writes compare equal): the same bytes and counts."""
    copy = h.copy()
    want = Reference(m)._fill_numpy(h, tb, mtp, rows, n, first, tokens, head, verify, keep_tails, meta_columns)
    got = native_fill(m, copy, tb, mtp, rows, n, first, tokens, head, verify, keep_tails, meta_columns)
    diff = np.flatnonzero(copy[: tb.nbytes].view(np.int32) != h[: tb.nbytes].view(np.int32))
    assert got == want and not len(diff), (got, want, diff[:16].tolist())
    m.fill_checks += 1
    return want


def native_fill(m, h, tb, mtp, rows, n, first, tokens, head, verify, keep_tails, meta_columns):
    blocks = [np.asarray(r.blocks, np.int64) for r in rows]  # an MTP pass's rows hold the MTP layer's
    i64 = lambda values: np.asarray(values, np.int64)  # noqa: E731
    counts = m._native.fill(
        tb.T, mtp, h,
        i64([r.column for r in rows]), i64([r.pos0 for r in rows]), i64([r.real for r in rows]),
        i64([r.state_src for r in rows]), i64([r.state_dst for r in rows]),
        i64(np.cumsum([0, *(len(b) for b in blocks)])), np.concatenate(blocks), n, first,
        None if tokens is None else i64(tokens), head=head, verify=verify, keep_tails=keep_tails,
        meta_columns=meta_columns,
    )  # fmt: skip
    return Filled(*counts[:4], tuple(tuple(u) for u in counts[4]))


def row(
    rng, m: HostModel, column: int, pos0: int, real: int, fresh: bool = False
) -> Row:
    g = m.geo
    n = -(-(pos0 + real) // PAGE_TOKENS)
    blocks = (
        rng.choice(4096, size=n, replace=False).astype(np.int64) * g.kv_block_rows
        + g.slab_rows
    )
    state = lambda: int(rng.integers(1, 64)) * g.slab_rows  # noqa: E731
    return Row(
        column, pos0, real, blocks, -1 if fresh or pos0 == 0 else state(), state()
    )


def host(m: HostModel, rng) -> np.ndarray:
    nbytes = max(tb.nbytes for tb in (*m.tables.values(), *m._mtables.values()))
    return rng.integers(
        0, 256, nbytes, np.uint8
    )  # junk: words neither fill writes must stay equal


@pytest.mark.parametrize("nranks", [4, 8])
@pytest.mark.parametrize("T", DECODE_BUCKETS)
def test_decode_steps_fill_as_numpy(nranks: int, T: int) -> None:
    rng = np.random.default_rng(T + nranks)
    m = host_model(nranks)
    for case in range(6):
        B = int(rng.integers(1, T + 1)) if case else T
        # short rows, rows across a pool or a page, and rows past the indexer's top pools (case 3 on: sparse)
        top = 2048 if case < 3 else 4 * m.d.top_pools + 500
        positions = rng.integers(0, top, B)
        if case >= 3:
            positions[0] = (
                MAX_SEQ - 2
                if case == 4
                else 4 * m.d.top_pools + int(rng.integers(3, 12))
            )
        rows = [
            row(rng, m, b, int(p), 1, fresh=case == 5) for b, p in enumerate(positions)
        ]
        tokens = None if case % 2 else rng.integers(0, 150000, B)
        h = host(m, rng)
        filled = m._fill(
            h, m.tables[bucket(B, False)], False, rows, B, int(positions[0]), tokens
        )
        assert filled.sparse == (case >= 3)
    assert m.fill_checks == 6


@pytest.mark.parametrize("drafts", [1, 3, 5, 6, 7])
def test_verify_steps_keep_every_tail(drafts: int) -> None:
    rng = np.random.default_rng(drafts)
    m = host_model(4, drafts)
    Tq = drafts + 1
    for B in range(1, 16 // Tq + 1):
        for top in (300, 4 * m.d.top_pools + 900):
            positions = rng.integers(0, top, B)
            rows = [row(rng, m, b * Tq, int(p), Tq) for b, p in enumerate(positions)]
            tokens = rng.integers(0, 150000, B * Tq)
            h = host(m, rng)
            tb = m.tables[bucket(B * Tq, False)]
            m._fill(
                h,
                tb,
                False,
                rows,
                B * Tq,
                int(positions[0]),
                tokens,
                verify=Tq,
                keep_tails=True,
            )
    assert m.fill_checks == 2 * (16 // Tq)


@pytest.mark.parametrize("T", PREFILL_BUCKETS)
def test_prefill_chunks_fill_as_numpy(T: int) -> None:
    rng = np.random.default_rng(T)
    m = host_model(4)
    for pos0, real in (
        (0, T // 2 + 3),
        (4096, T),
        (int(rng.integers(1, 9000)), int(rng.integers(1, T + 1))),
        (15000, T),
    ):
        real = min(real, T)
        if pos0 + real > MAX_SEQ:
            continue
        rows = [row(rng, m, 0, pos0, real)]
        tokens = rng.integers(0, 150000, real)
        h = host(m, rng)
        filled = m._fill(
            h, m.tables[bucket(real, True)], False, rows, real, pos0, tokens, head=True
        )
        assert not filled.sparse
    assert m.fill_checks >= 3


@pytest.mark.parametrize("T", [1, 4, 16, 64, *PREFILL_BUCKETS])
def test_mtp_passes_fill_as_numpy(T: int) -> None:
    rng = np.random.default_rng(100 + T)
    m = host_model(4, 3, mtp=True)
    for top in (1000, 4 * m.d.top_pools + 700):
        if T > MAX_T:  # a prefill chunk's keys: one row from its first column
            real = int(rng.integers(1, T + 1))
            rows = [
                Row(
                    0,
                    int(rng.integers(0, top)),
                    real,
                    row(rng, m, 0, top, T).blocks,
                    -1,
                    -1,
                )
            ]
        else:
            B = int(rng.integers(1, T + 1))
            rows = []
            for b in range(B):
                r = row(rng, m, b, int(rng.integers(0, top)), 1)
                rows.append(Row(b, r.pos0, 1, r.blocks, -1, -1))
            real = B
        h = host(m, rng)
        m._fill(
            h,
            m._mtables[T],
            True,
            rows,
            T,
            rows[0].pos0,
            None,
            keep_tails=True,
            meta_columns=real,
        )
    assert m.fill_checks == 2


def test_a_fill_past_the_tables_is_refused() -> None:
    m = host_model(4)
    rng = np.random.default_rng(0)
    rows = [row(rng, m, 0, 5, 1)]
    with pytest.raises(ValueError, match="past the step"):
        native_fill(m, host(m, rng), m.tables[1], False, [Row(3, 5, 1, rows[0].blocks, -1, 0)], 1, 5, None,
                    False, 0, False, None)  # fmt: skip
