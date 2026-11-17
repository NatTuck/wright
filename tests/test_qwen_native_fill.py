"""The native Qwen3.5/3.8 step tables (``resin._resin.QwenFill``, the ``gaudi`` feature) hold the same values as
the numpy builders of the Python executor that they replaced: ``common.attention.kv_rows`` and ``plan_slots``,
the GDN state tables of ``Qwen35Gaudi._write_state_tables``, and the columns, metadata and sampler inputs of
``Qwen35Gaudi._target``, over random prefill, decode and verify steps, with bf16 and FP8 K and V pages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from resin_models_gaudi.common.attention import SlotRow, kv_rows, plan_slots
from resin_models_gaudi.common.sampling import PARAMS, philox_uniform

native = pytest.importorskip("resin._resin")
pytestmark = pytest.mark.skipif(
    not hasattr(native, "QwenFill"), reason="resin built without the gaudi feature"
)

G, HPG, LAYERS = 4, 6, 48
STATE_LAYER_ROWS, SSM_ROWS = 6400, 6144
MAX_POSITIONS = 40960
BLOCK_ROWS = 8192  # rows between KV blocks (any multiple of the page rows)


def _state_fields(srcs, dsts) -> np.ndarray:
    """The Python executor's state fields (``Qwen35Gaudi._state_fields``): (layers, rows, 4) [conv_src, ssm_src,
    conv_dst, ssm_dst] of STATE block rows, -1 for a fresh source."""
    layer = np.arange(LAYERS, dtype=np.int64)[:, None] * STATE_LAYER_ROWS
    fields = []
    for bases in (srcs, dsts):
        base = np.asarray(bases, np.int64)[None, :]
        ssm = np.where(base >= 0, base + layer, -1)
        fields += [np.where(base >= 0, ssm + SSM_ROWS, -1), ssm]
    return np.stack(fields, axis=-1).astype(np.int32)


def _mode(fill, name, columns, per_row, query_block, slots, prefill):
    chunks = -(-MAX_POSITIONS // 2048)
    launches = -(
        -(columns // per_row) * -(-per_row // query_block) * G * chunks // slots
    )
    layout = fill.add_mode(
        name,
        columns=columns,
        per_row=per_row,
        query_block=query_block,
        slots=slots,
        prefill=prefill,
        max_launches=launches,
    )
    return SimpleNamespace(
        name=name,
        columns=columns,
        per_row=per_row,
        query_block=query_block,
        slots=slots,
        prefill=prefill,
        layout=layout,
    )


def _words(host: np.ndarray, offset: int, count: int) -> np.ndarray:
    return np.frombuffer(host, np.int32, count, offset)


def _check(fill, mode, rng, kv8):
    T, q, S, lay = mode.columns, mode.per_row, mode.slots, mode.layout
    if mode.prefill:
        n, real = 1, int(rng.integers(1, T + 1))
        pos0 = np.array([int(rng.integers(0, 20000))])
        lengths = [real]
    else:
        n = int(rng.integers(1, T // q + 1))
        real = n
        pos0 = rng.integers(0, 30000, n)
        lengths = [q] * n
    blocks = [
        rng.permutation(400)[: (int(p) + length) // 128 + 1].astype(np.int64)
        * BLOCK_ROWS
        for p, length in zip(pos0, lengths, strict=True)
    ]
    state_src = np.where(rng.random(n) < 0.2, -1, rng.integers(0, 50, n) * 160000)
    state_dst = rng.integers(0, 50, n) * 160000
    tokens = rng.integers(0, 248320, T)
    sampled = 1 if mode.prefill else T
    params = rng.random((sampled if mode.prefill else n, PARAMS)).astype(np.float32)
    host = np.zeros(lay["nbytes"], np.uint8)
    host[:] = 0xA5  # stale bytes everywhere: the fill must write every table it uploads
    indptr = np.concatenate(([0], np.cumsum([len(b) for b in blocks]))).astype(np.int64)
    launches, nbytes = fill.fill(
        mode.name,
        host,
        pos0.astype(np.int64),
        state_src.astype(np.int64),
        state_dst.astype(np.int64),
        indptr,
        np.concatenate(blocks),
        real,
        tokens.astype(np.int64),
        params.reshape(-1),
    )
    # ---- the reference (the numpy builders of the Python executor)
    if mode.prefill:
        pos = np.minimum(pos0[0] + np.arange(T), MAX_POSITIONS - 1)
        starts = ([int(pos0[0])], real)
    else:
        pos = np.zeros(T, np.int64)
        for i in range(n):
            pos[i * q : (i + 1) * q] = pos0[i] + np.arange(q)
        starts = (list(pos0), q)
    slot_rows = []
    k_rows, v_rows = (65 if kv8 & bit else 128 for bit in (1, 2))  # page rows (qwen3_5/cache.py)
    for i, b in enumerate(blocks):
        k = b[None, :] + np.arange(G)[:, None] * k_rows
        v = b[None, :] + G * k_rows + np.arange(G)[:, None] * v_rows
        length = starts[1] if mode.prefill else q
        slot_rows.append(SlotRow(i * q, int(starts[0][i]), length, k, v))
    want_kv = kv_rows(T, slot_rows, kv8)
    want_launches = plan_slots(slot_rows, mode.query_block, HPG, S)
    table4 = _state_fields(state_src, state_dst)
    # ---- compare
    assert launches == len(want_launches)
    assert nbytes == lay["slots"] + launches * lay["launch_stride"]
    np.testing.assert_array_equal(_words(host, lay["pos"], T), pos)
    np.testing.assert_array_equal(_words(host, lay["tok"], T), tokens)
    meta = [int(pos0[0]) if mode.prefill else 0, real, 0, 0]
    np.testing.assert_array_equal(_words(host, lay["meta"], 4), meta)
    if mode.prefill:
        assert _words(host, lay["last"], 1)[0] == real - 1
    samp = host[lay["samp_params"] : lay["samp_params"] + 4 * PARAMS * sampled].view(
        np.float32
    )
    want_samp = np.zeros((sampled, PARAMS), np.float32)
    want_samp[: len(params)] = params
    np.testing.assert_array_equal(samp, want_samp.reshape(-1))
    np.testing.assert_array_equal(
        _words(host, lay["kv_rows"], 8 * T), want_kv.reshape(-1)
    )
    for layer in range(LAYERS):
        at = lay["state"] + layer * lay["state_stride"]
        if mode.prefill:
            for k in range(4):
                assert _words(host, at + 256 * k, 1)[0] == table4[layer, 0, k]
        else:
            rows = T // q
            got = _words(host, at, 4 * rows).reshape(rows, 4)
            np.testing.assert_array_equal(got[:n], table4[layer])
            assert (got[n:] == -1).all()
            conv = _words(host, at + lay["conv_dst"], rows)
            np.testing.assert_array_equal(conv[:n], table4[layer, :, 2])
            assert (conv[n:] == -1).all()
    for i, launch in enumerate(want_launches):
        base = lay["slots"] + i * lay["launch_stride"]
        np.testing.assert_array_equal(
            _words(host, base, 8 * S), launch.slots.reshape(-1)
        )
        np.testing.assert_array_equal(
            _words(host, base + lay["k_pages"], 16 * S), launch.k_pages.reshape(-1)
        )
        np.testing.assert_array_equal(
            _words(host, base + lay["v_pages"], 16 * S), launch.v_pages.reshape(-1)
        )


@pytest.mark.parametrize("kv8", [0, 1, 2, 3], ids=["bf16", "k8", "v8", "kv8"])
def test_the_native_tables_equal_the_numpy_builders(kv8):
    fill = native.QwenFill(
        gdn_layers=LAYERS,
        kv_groups=G,
        hpg=HPG,
        state_layer_rows=STATE_LAYER_ROWS,
        ssm_rows=SSM_ROWS,
        max_positions=MAX_POSITIONS,
        kv8=kv8,
    )
    modes = [_mode(fill, f"p{T}", T, T, min(T, 512), 16, True) for T in (128, 4096)]
    modes += [_mode(fill, f"d{B}", B, 1, 1, max(16, G * B), False) for B in (1, 8, 32)]
    modes += [
        _mode(fill, f"v{B}w{W}", B * W, W, W, max(16, G * B), False)
        for B, W in ((1, 16), (8, 8), (4, 4))
    ]
    rng = np.random.default_rng(7)
    for mode in modes:
        for _ in range(6):
            _check(fill, mode, rng, kv8)


def test_philox_equals_the_numpy_reference():
    rng = np.random.default_rng(3)
    seeds = rng.integers(0, 2**63, 64, dtype=np.uint64)
    offsets = rng.integers(0, 2**40, 64, dtype=np.uint64)
    want = philox_uniform(seeds, offsets)
    got = [native.qwen_philox_uniform(int(s), int(o)) for s, o in zip(seeds, offsets)]
    np.testing.assert_array_equal(got, want)
