"""The native tables of the Qwen3.5/3.8 MTP layer (``resin._resin.QwenFill``, the ``gaudi`` feature) hold the
values that the Python executor uploaded: an MTP pass's columns, tokens, positions, K and V rows, sampler inputs
and slot tables (``Qwen35Gaudi._mtp_pass`` over ``common.attention.kv_rows`` and ``plan_slots``), and a verify
step's commit tables (``_commit``), the next step's first tokens and the draft chain's links (``_chain``), over
random prefill, decode and verify steps."""

from __future__ import annotations

import numpy as np
import pytest

from resin_models_gaudi.common.attention import SlotRow, kv_rows, plan_slots
from resin_models_gaudi.common.sampling import PARAMS

native = pytest.importorskip("resin._resin")
pytestmark = pytest.mark.skipif(
    not hasattr(native, "QwenFill") or not hasattr(native.QwenFill, "mtp_chain"),
    reason="resin built without the gaudi feature",
)

G, HPG, LAYERS = 4, 6, 48
STATE_LAYER_ROWS, SSM_ROWS, HIDDEN_OFFSET = 6400, 6144, LAYERS * 6400
STATE_TOKENS = 4096
MAX_POSITIONS = 40960
BLOCK_ROWS = 1024  # rows of an MTP KV block (one layer of a KV block)


def _launches(columns, per_row, query_block, slots) -> int:
    chunks = -(-MAX_POSITIONS // 2048)
    return -(-(columns // per_row) * -(-per_row // query_block) * G * chunks // slots)


class _Mode:
    def __init__(
        self, fill, name, columns, per_row, query_block, slots, prefill, extras=()
    ):
        self.name, self.columns, self.per_row = name, columns, per_row
        self.query_block, self.slots, self.prefill = query_block, slots, prefill
        self.max_launches = _launches(columns, per_row, query_block, slots)
        self.layout = fill.add_mode(
            name,
            columns=columns,
            per_row=per_row,
            query_block=query_block,
            slots=slots,
            prefill=prefill,
            max_launches=self.max_launches,
            extras=list(extras),
        )


def _fill(kv8: int = 0):
    return native.QwenFill(
        gdn_layers=LAYERS,
        kv_groups=G,
        hpg=HPG,
        state_layer_rows=STATE_LAYER_ROWS,
        ssm_rows=SSM_ROWS,
        max_positions=MAX_POSITIONS,
        kv8=kv8,
    )


def _words(host: np.ndarray, offset: int, count: int) -> np.ndarray:
    return np.frombuffer(host, np.int32, count, offset)


def _floats(host: np.ndarray, offset: int, count: int) -> np.ndarray:
    return np.frombuffer(host, np.float32, count, offset)


def _blocks(rng, last: int) -> np.ndarray:
    """MTP KV block rows covering positions 0 .. last."""
    return rng.permutation(400)[: last // 128 + 1].astype(np.int64) * BLOCK_ROWS


def _pass_reference(mode, starts, real, blocks, kv8=0):
    """``_mtp_pass``'s positions, K / V rows and slot launches (``_pages``, ``kv_rows``, ``_slot_tables``)."""
    T, q = mode.columns, mode.per_row
    count = real if mode.prefill else q
    pos = np.zeros(T, np.int64)
    rows = []
    k_rows, v_rows = (65 if kv8 & bit else 128 for bit in (1, 2))  # page rows (qwen3_5/cache.py)
    for i, (start, b) in enumerate(zip(starts, blocks, strict=True)):
        pos[i * q : i * q + count] = start + np.arange(count)
        k = b[None, :] + np.arange(G)[:, None] * k_rows
        v = b[None, :] + G * k_rows + np.arange(G)[:, None] * v_rows
        rows.append(SlotRow(i * q, int(start), count, k, v))
    pos = np.minimum(pos, MAX_POSITIONS - 1)
    want_kv = kv_rows(T, rows, kv8)
    return pos, want_kv, plan_slots(rows, mode.query_block, HPG, mode.slots)


def _check_launches(host, at: dict, want) -> None:
    S = len(want[0].slots) if want else 0
    for i, launch in enumerate(want):
        base = at["slots"] + i * at["stride"]
        np.testing.assert_array_equal(
            _words(host, base, 8 * S), launch.slots.reshape(-1)
        )
        np.testing.assert_array_equal(
            _words(host, base + at["k_pages"], 16 * S), launch.k_pages.reshape(-1)
        )
        np.testing.assert_array_equal(
            _words(host, base + at["v_pages"], 16 * S), launch.v_pages.reshape(-1)
        )


def _pass_tables(offsets, slots) -> dict:
    at = dict(
        zip(("cols", "tok", "pos", "kv_rows", "params", "slots"), offsets, strict=True)
    )
    at["k_pages"], at["v_pages"], at["stride"] = native.QwenFill.launch_tables(slots)
    return at


def _check_pass(fill, mode, rng, kind, kv8=0) -> None:
    """One random pass of ``kind``: a decode step's keys, a prefill chunk's keys, a verify step's full pass or a
    prompt's first draft (a full pass of one row's decode mode)."""
    T, q = mode.columns, mode.per_row
    full = kind in ("verify", "draft")
    if mode.prefill:
        n, real = 1, int(rng.integers(1, T + 1))
        pos0 = int(rng.integers(0, 3)) * int(rng.integers(0, 20000))
        shift = 0 if pos0 > 0 else 1
        real = max(real, shift + 1)
        starts, count = [pos0 - 1 + shift], real - shift
        cols = (np.arange(count) + shift).astype(np.int64)
        tokens = rng.integers(0, 248320, count).astype(np.int64)
    else:
        n = int(rng.integers(1, T // q + 1))
        # a keys pass also past the rope table (its positions clamp)
        top = MAX_POSITIONS - q if full else MAX_POSITIONS + 4
        starts = [int(p) for p in rng.integers(0, top, n)]
        count = q
        cols = (np.arange(T) + 1).astype(np.int64)
        tokens = None
        if kind == "draft":
            cols = np.array([int(rng.integers(1, 4097))], np.int64)
    blocks = [_blocks(rng, s + count) for s in starts]
    params = rng.random((n * q if full else 0, PARAMS)).astype(np.float32)
    sizes = native.QwenFill.mtp_pass_sizes(T, mode.slots, mode.max_launches, full)
    # anywhere in the image (multiples of 256), over stale bytes: the fill writes every table
    offsets = list(np.cumsum([0, *sizes[:-1]]) + 1024)
    host = np.full(offsets[-1] + sizes[-1], 0xA5, np.uint8)
    indptr = np.concatenate(([0], np.cumsum([len(b) for b in blocks]))).astype(np.int64)
    launches = fill.fill_mtp(
        mode.name,
        host,
        [int(o) for o in offsets],
        starts,
        count if mode.prefill else 0,
        indptr.tolist(),
        np.concatenate(blocks).tolist(),
        cols.tolist(),
        None if tokens is None else tokens.tolist(),
        params.reshape(-1).tolist() if full else None,
        G,
        HPG,
    )
    pos, kv, want = _pass_reference(mode, starts, count, blocks, kv8)
    at = _pass_tables(offsets, mode.slots)
    want_cols = np.zeros(T, np.int64)
    want_cols[: len(cols)] = cols
    np.testing.assert_array_equal(_words(host, at["cols"], T), want_cols)
    if tokens is not None:
        want_tok = np.zeros(T, np.int64)
        want_tok[: len(tokens)] = tokens
        np.testing.assert_array_equal(_words(host, at["tok"], T), want_tok)
    np.testing.assert_array_equal(_words(host, at["pos"], T), pos)
    np.testing.assert_array_equal(_words(host, at["kv_rows"], 8 * T), kv.reshape(-1))
    if not full:
        assert launches == 0
        return
    want_params = np.zeros((T, PARAMS), np.float32)
    want_params[: len(params)] = params
    np.testing.assert_array_equal(
        _floats(host, at["params"], PARAMS * T), want_params.reshape(-1)
    )
    assert launches == len(want)
    _check_launches(host, at, want)


@pytest.mark.parametrize("kv8", [0, 3], ids=["bf16", "kv8"])
def test_mtp_passes_equal_the_python_tables(kv8):
    fill = _fill(kv8)
    rng = np.random.default_rng(11)
    cases = [
        (_Mode(fill, f"p{T}", T, T, min(T, 512), 16, True), "keys") for T in (128, 1024)
    ]
    cases += [
        (_Mode(fill, f"d{B}", B, 1, 1, max(16, G * B), False), "keys")
        for B in (1, 8, 32)
    ]
    cases += [(_Mode(fill, "d1", 1, 1, 1, 16, False), "draft")]
    cases += [
        (_Mode(fill, f"v{B}", B * W, W, W, max(16, G * B), False), "verify")
        for B, W in ((1, 2), (8, 3), (32, 4))
    ]
    for mode, kind in cases:
        for _ in range(6):
            _check_pass(fill, mode, rng, kind, kv8)


def _state_fields(srcs, dsts) -> np.ndarray:
    """``Qwen35Gaudi._state_fields``: (layers, rows, 4) [conv_src, ssm_src, conv_dst, ssm_dst]."""
    layer = np.arange(LAYERS, dtype=np.int64)[:, None] * STATE_LAYER_ROWS
    fields = []
    for bases in (srcs, dsts):
        base = np.asarray(bases, np.int64)[None, :]
        ssm = np.where(base >= 0, base + layer, -1)
        fields += [np.where(base >= 0, ssm + SSM_ROWS, -1), ssm]
    return np.stack(fields, axis=-1).astype(np.int32)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_the_commit_and_chain_equal_the_python_tables(width):
    """``_commit``'s tables, the next step's [anchor, first draft] tokens of ``verify_finish`` and each link of
    ``_chain`` (a decode-mode MTP pass at position anchor + j - 1, the first link reading the verify pass's
    accepted column and the first draft)."""
    fill = _fill()
    rng = np.random.default_rng(width)
    K = width - 1
    for B in (1, 4, 32):
        decode = _Mode(fill, f"d{B}", B, 1, 1, max(16, G * B), False)
        layout = native.QwenFill.mtp_chain(
            B, decode.slots, decode.max_launches, width, LAYERS
        )
        assert len(layout["links"]) == K - 1
        for _ in range(4):
            n = int(rng.integers(1, B + 1))
            pos0 = rng.integers(0, 30000, n)
            if n > 1:  # a row whose verify columns cross into the next STATE block
                pos0[0] = STATE_TOKENS - 1 - int(rng.integers(0, width))
            accepted = rng.integers(0, width, n)
            anchors = rng.integers(0, 248320, n)
            first = rng.integers(0, 248320, n)
            src = np.where(rng.random(n) < 0.2, -1, rng.integers(1, 50, n) * 160000)
            state_first = pos0 // STATE_TOKENS - rng.integers(0, 2, n).clip(
                0, pos0 // STATE_TOKENS
            )
            states = [
                (rng.integers(1, 50, int(p) // STATE_TOKENS - f + 2) * 160000).astype(
                    np.int64
                )
                for p, f in zip(pos0, state_first, strict=True)
            ]
            blocks = [_blocks(rng, int(p) + 2 * width) for p in pos0]
            params = [rng.random((n, PARAMS)).astype(np.float32) for _ in range(K - 1)]
            host = np.full(layout["nbytes"], 0xA5, np.uint8)
            indptr = np.concatenate(([0], np.cumsum([len(b) for b in blocks]))).astype(
                np.int64
            )
            launches, nbytes = fill.fill_chain(
                decode.name,
                width,
                host,
                pos0.tolist(),
                src.tolist(),
                [s.tolist() for s in states],
                state_first.tolist(),
                accepted.tolist(),
                anchors.tolist(),
                first.tolist(),
                indptr.tolist(),
                np.concatenate(blocks).tolist(),
                [p.reshape(-1).tolist() for p in params],
                G,
                HPG,
                (STATE_LAYER_ROWS, SSM_ROWS, STATE_TOKENS, HIDDEN_OFFSET),
            )

            def state_at(r, position):
                return int(states[r][position // STATE_TOKENS - state_first[r]])

            # ---- _commit
            dsts = [state_at(r, int(pos0[r] + accepted[r])) for r in range(n)]
            bases = np.full((LAYERS, B, 4), -1, np.int32)
            bases[:, :n] = _state_fields(src, dsts)
            counts = np.zeros((LAYERS, B), np.int32)
            counts[:, :n] = accepted + 1
            hidden = np.full((B, 2), -1, np.int32)
            for r in range(n):
                hidden[r] = (r * width + accepted[r], dsts[r] + HIDDEN_OFFSET)
            np.testing.assert_array_equal(
                _words(host, layout["commit_bases"], 4 * B * LAYERS), bases.reshape(-1)
            )
            np.testing.assert_array_equal(
                _words(host, layout["commit_counts"], B * LAYERS), counts.reshape(-1)
            )
            np.testing.assert_array_equal(
                _words(host, layout["hidden_table"], 2 * B), hidden.reshape(-1)
            )
            # ---- verify_finish's next tokens and the chain's metadata
            following = np.zeros((B, width), np.int32)
            following[:n, 0], following[:n, 1] = anchors, first
            np.testing.assert_array_equal(
                _words(host, layout["following"], B * width), following.reshape(-1)
            )
            np.testing.assert_array_equal(_words(host, layout["meta"], 4), [0, n, 0, 0])
            # ---- _chain: link j at anchor + j - 1 (the anchor at pos0 + accepted + 1)
            assert len(launches) == K - 1
            for j, link in enumerate(layout["links"], start=1):
                starts = pos0 + accepted + j
                pos, kv, want = _pass_reference(decode, starts.tolist(), 1, blocks)
                cols = np.arange(B) + 1
                if j == 1:
                    cols = np.zeros(B, np.int64)
                    cols[:n] = np.arange(n) * width + accepted + 1
                    tok = np.zeros(B, np.int64)
                    tok[:n] = first
                    np.testing.assert_array_equal(_words(host, link["tok"], B), tok)
                np.testing.assert_array_equal(_words(host, link["cols"], B), cols)
                np.testing.assert_array_equal(_words(host, link["pos"], B), pos)
                np.testing.assert_array_equal(
                    _words(host, link["kv_rows"], 8 * B), kv.reshape(-1)
                )
                want_params = np.zeros((B, PARAMS), np.float32)
                want_params[:n] = params[j - 1]
                np.testing.assert_array_equal(
                    _floats(host, link["params"], PARAMS * B), want_params.reshape(-1)
                )
                assert launches[j - 1] == len(want)
                _check_launches(host, link, want)
            last = layout["links"][-1] if K > 1 else None
            if last is not None:
                assert nbytes == last["slots"] + launches[-1] * last["stride"]
            else:
                assert nbytes == layout["meta"] + 256
