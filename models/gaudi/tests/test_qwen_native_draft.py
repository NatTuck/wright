"""The native drafter tables of a Qwen3.5/3.8 verify step (``resin._resin.QwenDraftTables``, the ``gaudi``
feature) hold the same values as the Python builders they replace, for the DFlash 2 and DSpine drafters (no
card): ``common.dflash.inject_rows``; a block step's tokens, positions, anchors, ``block_rows_table`` and
``plan_block_slots`` per layer (``Drafter._tables``), with host and with device anchors and ``spec_fix``'s
patch entries (``Drafter.prepare_device``); the Python executor's ``choose_width`` (kept here); and
``resin.backend.speculative.accept_speculative_tokens``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import test_qwen_dflash_plan as dflash_plan
import test_qwen_dspine_plan as dspine_plan

from resin.backend.speculative import accept_speculative_tokens
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_gaudi.common import dflash
from resin_models_gaudi.common.dflash import DrafterConfig, DraftRow

native = pytest.importorskip("resin._resin")
pytestmark = pytest.mark.skipif(
    not hasattr(native, "QwenDraftTables"),
    reason="resin built without the gaudi feature",
)

BMAX = 32  # rows of the widest block step (the scratch pages)
# The Python executor's verify width choice (``qwen3_5.model.choose_width`` before the native executor took it),
# the reference of the native one: P(a step accepts at least j of its drafts | the row's last step), j = 0 .. 15,
# by class of the last step, and the DSpine step-time fit.
# fmt: off
SURVIVAL = np.array(
    (
        (1.000, 0.695, 0.480, 0.329, 0.222, 0.151, 0.103, 0.065, 0.048, 0.036, 0.032, 0.028, 0.022, 0.017, 0.012, 0.012),
        (1.000, 0.710, 0.487, 0.339, 0.216, 0.144, 0.090, 0.063, 0.041, 0.031, 0.023, 0.018, 0.012, 0.007, 0.006, 0.003),
        (0.999, 0.747, 0.535, 0.346, 0.219, 0.151, 0.093, 0.069, 0.048, 0.035, 0.025, 0.015, 0.015, 0.010, 0.009, 0.008),
        (0.999, 0.735, 0.511, 0.373, 0.262, 0.197, 0.130, 0.105, 0.082, 0.064, 0.049, 0.031, 0.029, 0.024, 0.024, 0.014),
        (0.999, 0.735, 0.535, 0.387, 0.293, 0.226, 0.157, 0.114, 0.090, 0.060, 0.042, 0.023, 0.018, 0.013, 0.013, 0.013),
        (0.999, 0.802, 0.618, 0.490, 0.344, 0.242, 0.173, 0.132, 0.121, 0.109, 0.104, 0.083, 0.070, 0.055, 0.052, 0.050),
        (0.998, 0.852, 0.731, 0.593, 0.520, 0.439, 0.407, 0.346, 0.302, 0.277, 0.253, 0.229, 0.217, 0.168, 0.144, 0.128),
        (0.998, 0.983, 0.938, 0.879, 0.865, 0.820, 0.736, 0.682, 0.667, 0.653, 0.643, 0.628, 0.589, 0.549, 0.534, 0.515),
        (1.000, 0.785, 0.604, 0.472, 0.373, 0.301, 0.235, 0.195, 0.172, 0.150, 0.135, 0.116, 0.107, 0.092, 0.087, 0.080),
        (0.999, 0.876, 0.756, 0.643, 0.571, 0.497, 0.440, 0.388, 0.363, 0.347, 0.334, 0.315, 0.297, 0.257, 0.241, 0.227),
    )
)
# fmt: on
SURVIVAL_CLASS = (0, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7)


def _step_ms(width: int, bucket: int) -> float:
    per_column = max(0.4, 0.71 - 0.026 * (bucket - 1))
    return 22.3 + 1.44 * (bucket - 1) + per_column * bucket * width


def choose_width(widths, accepted: np.ndarray, width: int, bucket: int = 1) -> int:
    """The width of the most expected tokens per ms, each row's tokens expected from its accepted drafts."""
    if len(widths) == 1:
        return widths[0]
    classes = []
    for a in np.asarray(accepted).tolist():
        if a >= width - 1 and width < 16:  # all accepted: at least a
            classes.append(8 if width <= 4 else 9)
        else:
            classes.append(SURVIVAL_CLASS[min(int(a), 15)])
    sums = np.cumsum(SURVIVAL, axis=1)
    best, choice = -1.0, widths[-1]
    for w in widths:
        tokens = sum(1.0 + sums[c, min(w - 1, 15)] - SURVIVAL[c, 0] for c in classes)
        rate = tokens / _step_ms(w, bucket)
        if rate > best:
            best, choice = rate, w
    return choice
SCRATCH_ROWS = 7 << 20
MAX_POSITIONS = 32768


def _config(kind, tmp_path) -> DrafterConfig:
    target = Qwen35Config.from_hugging_face(
        dflash_plan.TARGET, allow_tied_embeddings=True
    )
    plan = dspine_plan if kind == "dspine" else dflash_plan
    return DrafterConfig.load(plan.write_drafter(tmp_path), target)


def _native(cfg: DrafterConfig):
    return native.QwenDraftTables(
        layers=cfg.layers,
        pairs=cfg.pairs,
        layer_rows=cfg.layer_rows,
        block=cfg.block,
        window=cfg.window,
        causal=cfg.causal,
        mask_token=cfg.mask_token,
        scratch_rows=SCRATCH_ROWS,
        max_rows=BMAX,
    )


def _scratch(cfg: DrafterConfig) -> np.ndarray:
    """``Drafter.scratch``: (B, layers, pairs, [K, V]) first rows of the scratch pages."""
    pages = np.arange(BMAX * cfg.layers * cfg.pairs * 2, dtype=np.int64) * 128
    return (SCRATCH_ROWS + pages).reshape(BMAX, cfg.layers, cfg.pairs, 2)


def _row(cfg, rng, position: int) -> DraftRow:
    """A row whose retained DRAFT blocks reach back the window from ``position``."""
    first = max(0, (position - cfg.window - 16) // 128)
    count = (position + 16) // 128 - first + 1
    blocks = rng.permutation(4000)[:count].astype(np.int64) * cfg.block_rows
    return DraftRow(position, blocks, first)


@pytest.mark.parametrize("kind", ["dflash", "dspine"])
def test_injection_equals_inject_rows(kind, tmp_path):
    cfg, rng = _config(kind, tmp_path), np.random.default_rng(1)
    fill = _native(cfg)
    for tokens, per_row in ((8, 8), (128, 16), (64, 1)):
        runs = []
        for b in range(tokens // per_row):
            position = int(rng.integers(0, 20000))
            row = _row(cfg, rng, position + per_row)
            runs.append((b * per_row, position, per_row, row))
        want = dflash.inject_rows(cfg, tokens, runs)
        pos, kv = fill.inject(
            tokens,
            [
                (c, p, n, (r.position, r.blocks.tolist(), r.first))
                for c, p, n, r in runs
            ],
        )
        want_pos = np.zeros(tokens, np.int64)
        for c, p, n, _ in runs:
            want_pos[c : c + n] = p + np.arange(n)
        np.testing.assert_array_equal(pos, want_pos)
        np.testing.assert_array_equal(np.reshape(kv, (cfg.layers, tokens, 8)), want)


def _reference_block(cfg, rows, anchors, low, bucket, width):
    """``Drafter._tables`` (and ``prepare_device``'s records) without a device."""
    c = replace(cfg, block=width) if cfg.causal and width < cfg.block else cfg
    T, B = c.block * bucket, len(rows)
    tok = np.full(T, c.mask_token, np.int64)
    pos = np.zeros(T, np.int64)
    anc = np.zeros(bucket, np.int64)
    for b, row in enumerate(rows):
        if anchors is not None:
            tok[b * c.block] = anchors[b]
            anc[b] = anchors[b]
        pos[b * c.block : (b + 1) * c.block] = row.position + np.arange(c.block)
    scratch = _scratch(cfg)
    block_rows = dflash.block_rows_table(c, T, scratch, B)
    S = dflash.slots_per_launch(bucket, c.pairs)
    records: list = []
    launches = dflash.plan_block_slots(
        c, rows, scratch, S, low, records if low is not None else None
    )
    return tok, pos, anc, block_rows, S, launches, records


@pytest.mark.parametrize("kind", ["dflash", "dspine"])
@pytest.mark.parametrize("device_anchors", [False, True])
def test_block_tables_equal_the_drafter(kind, device_anchors, tmp_path):
    cfg, rng = _config(kind, tmp_path), np.random.default_rng(2)
    fill = _native(cfg)
    widths = (4, 8, 16) if cfg.causal else (cfg.block,)
    for bucket in (1, 2, 8, 32):
        for width in widths:
            n = int(rng.integers(1, bucket + 1))
            positions = rng.integers(1, 20000, n)
            rows = [_row(cfg, rng, int(p)) for p in positions]
            anchors = None if device_anchors else rng.integers(0, 248000, n).tolist()
            low = (
                (positions - rng.integers(0, 16, n)).clip(0).tolist()
                if device_anchors
                else None
            )
            got = fill.block(
                bucket,
                [(r.position, r.blocks.tolist(), r.first) for r in rows],
                anchors,
                low,
                width,
                MAX_POSITIONS,
            )
            tok, pos, anc, block_rows, S, launches, records = _reference_block(
                cfg, rows, anchors, low, bucket, width
            )
            np.testing.assert_array_equal(got["tok"], tok)
            np.testing.assert_array_equal(got["pos"], pos)
            np.testing.assert_array_equal(got["anchors"], anc)
            np.testing.assert_array_equal(
                np.reshape(got["block_rows"], block_rows.shape), block_rows
            )
            assert got["slots_per_launch"] == S and got["launches"] == len(launches)
            L = cfg.layers
            slots = np.reshape(got["slots"], (len(launches), L, S, 8))
            kp = np.reshape(got["k_pages"], (len(launches), L, S, 16))
            vp = np.reshape(got["v_pages"], (len(launches), L, S, 16))
            for i, (s, k, v) in enumerate(launches):
                for layer in range(L):
                    shift = layer * cfg.layer_rows
                    np.testing.assert_array_equal(slots[i, layer], s)
                    np.testing.assert_array_equal(
                        kp[i, layer], np.where(k >= 0, k + shift, -1)
                    )
                    np.testing.assert_array_equal(
                        vp[i, layer], np.where(v >= 0, v + shift, -1)
                    )
            if device_anchors:
                assert [tuple(r) for r in got["records"]] == [tuple(r) for r in records]
                # each patch entry addresses its record: launch k // S, slot k % S of every layer, in the
                # launch-major tables (launch i of layer l at (i * layers + l) * stride bytes)
                entries = np.asarray(got["patch"], np.int64).reshape(L, len(records), 4)
                for layer in range(L):
                    for k, (row, kind_) in enumerate(records):
                        offset, r, kd, zero = entries[layer, k]
                        table, within = divmod(int(offset) * 4, got["stride"])
                        assert (r, kd, zero) == (row, kind_, 0)
                        assert divmod(table, L) == (k // S, layer)
                        assert within == 32 * (k % S)
            else:
                assert got["records"] == [] and got["patch"] == []


def test_widths_equal_choose_width():
    rng = np.random.default_rng(3)
    for bucket in (1, 2, 8, 32):
        for width in (4, 8, 16):
            for _ in range(40):
                accepted = rng.integers(0, width, int(rng.integers(1, bucket + 1)))
                want = choose_width((4, 8, 16), accepted, width, bucket)
                got = native.QwenDraftTables.choose_width(
                    [4, 8, 16], accepted.tolist(), width, bucket
                )
                assert got == want, (bucket, width, accepted)
    assert native.QwenDraftTables.choose_width([8], [3], 8, 1) == 8


def test_acceptance_equals_accept_speculative_tokens():
    rng = np.random.default_rng(4)
    for width in (2, 4, 8, 16):
        for _ in range(50):
            rows = int(rng.integers(1, 9))
            drafts = rng.integers(0, 3, (rows, width - 1))
            samples = rng.integers(0, 3, (rows, width))
            want = accept_speculative_tokens(drafts.tolist(), samples.tolist())
            accepted, emitted = native.QwenDraftTables.accept(
                drafts.reshape(-1).tolist(), samples.reshape(-1).tolist(), rows, width
            )
            assert accepted == want.accepted_counts
            assert emitted == want.token_rows
