"""DeepSeek-V4.1's native step fill (``resin._resin.Ds41Fill``, the ``gaudi`` feature) writes the same host tables,
byte for byte, and returns the same op counts as the numpy fill it replaced (``ds41_reference``:
``_fill_positions``, ``_fill_tokens``, ``_fill_draft``), on CPU: a model object with the checkpoint's geometry and every bucket's tables, no card. Decode
and verify steps, prefill chunks (with bounded replay's floor), the tokens of a prepared step, and DSpark drafts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import resin._resin as native
from resin_models_gaudi.deepseek_v41.model import _native_rows

CHECKPOINT = Path(os.environ.get("DS41_CHECKPOINT", ""))
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("DS41_CHECKPOINT") or not (CHECKPOINT / "config.json").exists(),
        reason="DS41_CHECKPOINT is not set",
    ),
    pytest.mark.skipif(
        not hasattr(native, "Ds41Fill"), reason="resin built without the gaudi feature"
    ),
]


def model(max_seq_len: int = 40_000):
    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin.models.deepseek_v41.engram import EngramHash
    from resin_models_gaudi.deepseek_v41 import model as M
    from resin_models_gaudi.deepseek_v41.cache import PAGE_TOKENS, CacheGeometry
    from resin_models_gaudi.deepseek_v41.rope import cs_table, layer_frequencies

    cfg = DeepSeekV41Config.from_hugging_face(
        json.loads((CHECKPOINT / "config.json").read_text())
    )
    from ds41_reference import ReferenceModel

    m = object.__new__(ReferenceModel)
    m.cfg, m.engram = cfg, "card"
    m.n_layers = cfg.num_hidden_layers
    m.stages, m.block = cfg.num_nextn_predict_layers, cfg.dspark_block_size
    m.geo = CacheGeometry.from_config(
        cfg, layers=m.n_layers, max_write=M.TMAX, draft_layers=m.stages
    )
    # the fill reads only the head count
    setattr(m, "d", SimpleNamespace(heads=cfg.num_attention_heads // 8))
    m.engram_layers = tuple(e for e in cfg.engram_layer_ids if e < m.n_layers)
    m.freqs = {r: layer_frequencies(cfg, r) for r in (0, 1)}
    m.max_seq_len = max_seq_len
    m.kmax = -(-(max_seq_len + M.TMAX) // 4096) * 4096
    m._cs_all = {r: cs_table(np.arange(m.kmax), m.freqs[r]) for r in (0, 1)}
    m.pages = m.kmax // PAGE_TOKENS
    m.kb = -(-m.kmax // 8 // 4096) * 4096
    m.kc = cfg.candidate_topk_blocks * 8 if cfg.candidate_source_layer >= 0 else 0
    rng = np.random.default_rng(0)
    n = cfg.engram_max_ngram_size
    layers = len(cfg.engram_layer_ids)
    m._hash = EngramHash(
        token_map=rng.integers(0, 100_000, cfg.vocab_size, dtype=np.int64),
        pad_id=7,
        multipliers=rng.integers(1, 1 << 62, (layers, n), dtype=np.int64),
        primes=rng.integers(1000, 1 << 20, (layers, n - 1, cfg.engram_n_heads), dtype=np.int64),
        offsets=rng.integers(0, 1 << 20, (layers, (n - 1) * cfg.engram_n_heads), dtype=np.int64),
    )
    m.tables = {s: M._Tables(s, m) for s in m.shapes}
    size = max(tb.nbytes for tb in m.tables.values())
    m.host = [np.zeros(size, np.uint8) for _ in range(2)]
    for tb in m.tables.values():
        for h in m.host:
            tb.bind(h)
    m._layer_rows = np.array(
        [m.geo.layer_offset(k) for k in range(m.n_layers + m.stages)], np.int64
    )
    m._native = m._native_fill()
    assert m._native is not None
    return m


def plan(m, t: int, rng, far: bool):
    """A random decode step of ``t`` columns: rows of one column (plain decode) or of several (verify)."""
    from resin_models_gaudi.deepseek_v41.cache import HISTORY_TOKENS, PAGE_TOKENS
    from resin_models_gaudi.deepseek_v41.model import Row

    reals = []
    while sum(reals) < t:
        reals.append(int(rng.choice([1, 1, 1, 2, 4])))
    reals[-1] -= sum(reals) - t
    if reals[-1] == 0:
        reals.pop()
    if rng.random() < 0.5:  # a step need not fill its bucket
        reals = reals[: max(1, len(reals) - int(rng.integers(0, 2)))]
    rows, positions, column = [], [], 0
    span = (17_000, m.max_seq_len - 64) if far else (0, 9_000)
    for real in reals:
        pos0 = int(rng.integers(*span))
        last = (pos0 + real - 1) // PAGE_TOKENS
        first = max(0, last - int(rng.integers(1, 18)))
        blocks = last - first + 1
        history = (pos0 + real - 1) // HISTORY_TOKENS + 1
        before = rng.integers(0, m.cfg.vocab_size, min(3, pos0))
        rows.append(
            Row(
                column,
                pos0,
                real,
                rng.integers(0, 1 << 20, blocks) * 2 + 1,
                rng.integers(0, 1 << 16, blocks),
                first,
                rng.integers(0, 1 << 22, history),
                before,
            )
        )
        positions.extend(range(pos0, pos0 + real))
        column += real
    tokens = rng.integers(0, m.cfg.vocab_size, len(positions))
    return rows, np.array(positions, np.int64), tokens


@pytest.mark.parametrize("far", [False, True], ids=["near", "far"])
def test_native_fill_matches_numpy(far):
    from resin_models_gaudi.deepseek_v41.model import DECODE_BUCKETS, Shape

    m = model()
    rng = np.random.default_rng(1 + far)
    for t in DECODE_BUCKETS:
        tb = m.tables[Shape(t, False)]
        for case in range(12):
            rows, positions, tokens = plan(m, t, rng, far)
            single = all(r.real == 1 for r in rows)
            for with_tokens in (True, False):
                # the device's tokens (None) only for rows of one column
                for given in (tokens, None) if single else (tokens,):
                    noise = rng.integers(0, 256, m.host[0].size, dtype=np.uint8)
                    for h in m.host:
                        h[:] = noise
                    want = m._fill_positions(0, tb, positions, rows)
                    if with_tokens:
                        m._fill_tokens(0, tb, given, len(positions), rows)
                    got = native_fill(m, 1, tb, given, positions, rows, with_tokens)
                    same(m, tb, f"T={t} case {case}")
                    same_counts(want, got)


def native_fill(m, k: int, tb, tokens, positions, rows, with_tokens: bool, floor=None) -> dict:
    """``Ds41Fill.fill`` of host copy ``k``, its counts as the reference fill returns them."""
    out = m._native.fill(
        tb.shape.T, m.host[k], *_native_rows(rows), np.asarray(positions, np.int64),
        None if tokens is None else np.asarray(tokens, np.int64), with_tokens, tb.shape.prefill, floor,
    )  # fmt: skip
    sparse, cand, ixc, radix1, radix2, ix1, ix2, pf1, pf2, dense1, dense2 = out
    counts = {"sparse": sparse, "cand": cand, 1: ix1, 2: ix2}
    if sparse:
        counts["ixc"] = ixc
    for ratio, radix, pf, dense in ((1, radix1, pf1, dense1), (2, radix2, pf2, dense2)):
        counts["radix", ratio], counts["pf", ratio], counts["dense", ratio] = radix, pf, dense
    return counts


def same(m, tb, case: str) -> None:
    a, b = m.host[0][: tb.nbytes], m.host[1][: tb.nbytes]
    if not np.array_equal(a, b):
        bad = int(np.flatnonzero(a != b)[0])
        name = max((o, n) for n, o in tb.off.items() if o <= bad)[1]
        raise AssertionError(f"{case}: first byte {bad} differs, in table {name}")


def same_counts(want: dict, got: dict) -> None:
    keys = {"sparse", "cand", 1, 2} | {
        (kind, r) for kind in ("radix", "pf", "dense") for r in (1, 2)
    }
    if want["sparse"]:
        keys.add("ixc")
    assert {key: want[key] for key in keys} == {key: got[key] for key in keys}


def prefill_row(m, rng, n: int, pos0: int):
    """One prefill chunk of ``n`` tokens from ``pos0``: its WINDOW blocks from the oldest a window can reach."""
    from resin_models_gaudi.deepseek_v41.cache import HISTORY_TOKENS, PAGE_TOKENS
    from resin_models_gaudi.deepseek_v41.model import Row

    last = (pos0 + n - 1) // PAGE_TOKENS
    first = max(0, pos0 // PAGE_TOKENS - int(rng.integers(1, 3)))
    blocks = last - first + 1
    history = (pos0 + n - 1) // HISTORY_TOKENS + 1
    return Row(
        0, pos0, n,
        rng.integers(0, 1 << 20, blocks) * 2 + 1, rng.integers(0, 1 << 16, blocks), first,
        rng.integers(0, 1 << 22, history), rng.integers(0, m.cfg.vocab_size, min(3, pos0)),
    )  # fmt: skip


@pytest.mark.parametrize("span", ["start", "dense", "listed", "prefiltered"])
def test_native_prefill_fill_matches_numpy(span):
    """Prefill chunks of every bucket: from the sequence's start, while the compressed entries fit the dense
    attention, past it, and past the prefilter's threshold; with and without a replay floor."""
    from resin_models_gaudi.deepseek_v41.model import PREFILL_BUCKETS, Shape

    m = model()
    rng = np.random.default_rng(["start", "dense", "listed", "prefiltered"].index(span))
    lo, hi = {"start": (0, 1), "dense": (1, 1500), "listed": (4500, 9000),
              "prefiltered": (24_000, 36_000)}[span]  # fmt: skip
    seen = set()
    for t in PREFILL_BUCKETS:
        tb = m.tables[Shape(t, True)]
        for case in range(6):
            n = int(rng.integers(max(1, t // 4 + 1), t + 1)) if t > 32 else int(rng.integers(1, 33))
            pos0 = int(rng.integers(lo, hi))
            row = prefill_row(m, rng, n, pos0)
            positions = pos0 + np.arange(n)
            tokens = rng.integers(0, m.cfg.vocab_size, n)
            floor = int(rng.integers(max(0, pos0 - 300), pos0 + 1)) if case % 2 else None
            noise = rng.integers(0, 256, m.host[0].size, dtype=np.uint8)
            for h in m.host:
                h[:] = noise
            want = m._fill_positions(0, tb, positions, [row])
            m._fill_tokens(0, tb, tokens, n, [row])
            if floor is not None:
                slots = tb.v[0]["slots"]
                slots[slots[:, 0] >= 0, 6] = floor
            got = native_fill(m, 1, tb, tokens, positions, [row], True, floor)
            same(m, tb, f"{span} T={t} case {case}")
            same_counts(want, got)
            # the tiled chain's groups this step needs (its index arm takes its groups' launches and merges)
            used = want[1] // (m.ix_lpg(Shape(t, True)) + 1)
            seen.add((used > 1, want["dense", 1], want["dense", 2]))
    # each span reaches the variant it names
    if span == "prefiltered":
        assert (True, False, False) in seen
    if span == "dense":
        assert any(dense for _, dense, _ in seen)


def test_native_tokens_of_a_prepared_step_match_numpy():
    """A verify step's tables come in two parts: positions while the device drafts, then the tokens."""
    from resin_models_gaudi.deepseek_v41.model import Shape

    m = model()
    rng = np.random.default_rng(5)
    for t in (4, 8, 16):
        tb = m.tables[Shape(t, False)]
        for case in range(6):
            rows, positions, tokens = plan(m, t, rng, far=bool(case % 2))
            noise = rng.integers(0, 256, m.host[0].size, dtype=np.uint8)
            for h in m.host:
                h[:] = noise
            m._fill_positions(0, tb, positions, rows)
            m.host[1][:] = m.host[0]
            m._fill_tokens(0, tb, tokens, len(positions), rows)
            m._native.fill_tokens(
                tb.shape.T, m.host[1], *_native_rows(rows), len(positions),
                np.asarray(tokens, np.int64), tb.shape.prefill,
            )  # fmt: skip
            same(m, tb, f"T={t} case {case}")


def test_native_draft_fill_matches_numpy():
    from resin_models_gaudi.deepseek_v41.model import Shape, bucket

    m = model()
    rng = np.random.default_rng(6)
    K = m.block
    for B in (1, 2):
        tb = m.tables[Shape(bucket(B * K, False), False)]
        for case in range(8):
            rows = [prefill_row(m, rng, 1, int(rng.integers(0, 30_000) if case else 0)) for _ in range(B)]
            anchors = rng.integers(0, m.cfg.vocab_size, B)
            noise = rng.integers(0, 256, m.host[0].size, dtype=np.uint8)
            for h in m.host:
                h[:] = noise
            m._fill_draft(0, tb, anchors, rows)
            from resin_models_gaudi.deepseek_v41.model import _native_rows

            _, pos0, _, first, window_indptr, window, *_ = _native_rows(rows)
            m._native.fill_draft(tb.shape.T, m.host[1], pos0, first, window_indptr, window,
                                 anchors.astype(np.int64), K, m.cfg.dspark_noise_token_id, m.stages)  # fmt: skip
            same(m, tb, f"B={B} case {case}")


def test_verify_rows_store_every_even_pending_state():
    """A verify row (several columns) stores the pending state of each even position in its slot, and a row
    starting at an odd position reads the slot of the position before it."""
    from resin_models_gaudi.deepseek_v41.cache import PAGE_TOKENS, PENDING_SLOTS
    from resin_models_gaudi.deepseek_v41.model import Row, Shape

    m = model()
    tb = m.tables[Shape(8, False)]
    geo = m.geo
    # row 0: positions 1000 .. 1003 (even start); row 1: positions 1537 .. 1540 (odd start, after block 11)
    rows = []
    for column, pos0 in ((0, 1000), (4, 1537)):
        first = pos0 // PAGE_TOKENS - 2
        rows.append(Row(column, pos0, 4, np.arange(3) + 7, np.arange(3) + 100 * (column + 1), first,
                        np.arange(8) + 3, np.zeros(3, np.int64)))  # fmt: skip
    positions = np.concatenate([np.arange(p, p + 4) for p in (1000, 1537)])
    m._fill_positions(0, tb, positions, rows)
    npend = len(geo.pending_owners)
    for i, o in enumerate(geo.pending_owners):
        pend = tb.v[0][f"cmp_rows.{o}"]
        groups = tb.v[0][f"cmp_groups.{o}"]
        for c, p in enumerate(positions):
            block = rows[c // 4].window_id(p)
            want = (
                (block * npend + i) * PENDING_SLOTS + p // 2 % PENDING_SLOTS
                if p % 2 == 0
                else -1
            )
            assert pend[c] == want, (c, p)
        # row 1's first group (1536, 1537): token a from the slot of 1536 in the block of 1536
        g = [k for k in range(groups.shape[0]) if groups[k, 1] == 4][0]
        assert groups[g, 0] == -1
        prev = rows[1].window_id(1536)
        assert (
            groups[g, 2]
            == (prev * npend + i) * PENDING_SLOTS + 1536 // 2 % PENDING_SLOTS
        )
        # row 0's groups are inside the step
        for k in range(groups.shape[0]):
            if groups[k, 1] in (1, 3):
                assert groups[k, 0] == groups[k, 1] - 1 and groups[k, 2] == 0
