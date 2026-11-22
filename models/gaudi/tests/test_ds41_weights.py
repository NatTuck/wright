"""DeepSeek-V4.1 weight preparation (host only): FP8 conversion, expert slots, rank slices. With
DS41_CHECKPOINT set, also on real tensors of layer 0."""

from __future__ import annotations

import json
import logging
import os
import struct
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from resin_models_gaudi.deepseek_v41.weights import (
    OCP,
    RankSlices,
    expert_slots,
    inc_bytes,
    inc_values,
    kernel_k,
    mx8_f64,
    mx8_rows,
    reader,
    slot_rows,
    table_rows,
)
from resin_models_gaudi.loading.convert import tperm

CHECKPOINT = os.environ.get("DS41_CHECKPOINT") or None


def test_every_bias7_byte_round_trips():
    b = np.arange(256)
    valid = ((b & 0x78) != 0x78) & (b != 0x80)  # no exponent 15; -0 encodes as 0x00
    np.testing.assert_array_equal(inc_bytes(inc_values(b[valid])), b[valid])
    assert inc_bytes(np.array([1e9, -1e9, 245.0, -244.0])).tolist() == [
        0x77,
        0xF7,
        0x77,
        0xF7,
    ]
    assert inc_bytes(np.array([2.0**-11, -(2.0**-11)])).tolist() == [0, 0]


def ocp_bytes(rng, shape) -> np.ndarray:
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    return np.where((b & 0x7F) == 0x7F, b - 1, b).astype(np.uint8)


@pytest.mark.parametrize("spread", [0, 3, 12])
def test_mx8_rows_shift_blocks_exactly(spread):
    rng = np.random.default_rng(spread)
    w = ocp_bytes(rng, (64, 512))
    scales = rng.integers(120 - spread, 121, (2, 16)).astype(np.uint8)
    got = mx8_rows(w, scales, tau=False)
    value = inc_values(got.bytes) * got.row_scales[:, None]
    want = mx8_f64(w, scales)
    assert int(np.count_nonzero(value != want)) == got.inexact
    assert np.max(np.abs(value)) <= np.max(np.abs(want))
    assert np.all(np.abs(inc_values(got.bytes)) <= 240)
    # each row's largest value lands in the top binade: the scale is the smallest that fits
    top = np.abs(inc_values(got.bytes)).max(axis=1)
    assert np.all(top > 120)
    np.testing.assert_array_equal(mx8_rows(w, scales).bytes, tperm(got.bytes))


def test_mx8_rows_counts_rounded_elements():
    w = np.full((32, 64), 0x08, np.uint8)  # OCP 2^-6 everywhere
    w[0, 0] = 0x7E  # 448 in the first block, whose scale is the largest
    scales = np.array(
        [[127, 108]], np.uint8
    )  # the second block sits 19 exponents lower
    got = mx8_rows(w, scales, tau=False)
    assert got.row_scales[0] == 2.0  # 448 needs a scale of 2 to fit 240
    assert (
        got.inexact == 32 * 32
    )  # 2^-6 * 2^-19 / 2 is below the smallest subnormal: rounded to 0


def test_slot_rows_order():
    which, rows = slot_rows(288)
    assert which.tolist() == [0] * 256 + [1] * 256 + [0] * 32 + [1] * 32
    assert rows[:256].tolist() == list(range(256)) and rows[512:544].tolist() == list(
        range(256, 288)
    )


def test_kernel_k_checks_the_spread():
    assert kernel_k(np.array([118, 125], np.uint8)) == 125 - 132
    with pytest.raises(ValueError, match="span"):
        kernel_k(np.array([100, 125], np.uint8))
    with pytest.raises(ValueError, match="NaN"):
        kernel_k(np.array([120, 0xFF], np.uint8))


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    (path.parent / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: path.name for name in tensors}})
    )


def test_expert_slots_layout(tmp_path):
    rng = np.random.default_rng(1)
    E, H, W = 3, 256, 2 * 288  # two ranks of 288
    tensors, full = {}, {}
    for e in range(E):
        for proj, shape in (("w1", (W, H)), ("w3", (W, H)), ("w2", (H, W))):
            codes = rng.integers(0, 256, (shape[0], shape[1] // 2), dtype=np.uint8)
            scales = rng.integers(118, 125, (shape[0], shape[1] // 32), dtype=np.uint8)
            tensors[f"layers.0.ffn.experts.{e}.{proj}.weight"] = ("I8", codes)
            tensors[f"layers.0.ffn.experts.{e}.{proj}.scale"] = ("F8_E8M0", scales)
            full[(e, proj)] = (codes, scales)
    write_safetensors(tmp_path / "model-00001-of-00001.safetensors", tensors)
    src = reader(tmp_path)
    width = slice(288, 576)
    got = expert_slots(src, "layers.0.ffn", E, width, H)
    which, rows = slot_rows(288)
    for e in range(E):
        c13 = got.codes13[e].reshape(576, H // 2)
        for i, (m, r) in enumerate(zip(which, rows)):
            np.testing.assert_array_equal(
                c13[i], full[(e, ("w1", "w3")[m])][0][288 + r]
            )
        c2 = got.codes2[e].reshape(H, 144)
        np.testing.assert_array_equal(c2, full[(e, "w2")][0][:, 144:288])
        s2 = got.scales2[e].reshape(H, 9)
        np.testing.assert_array_equal(s2, full[(e, "w2")][1][:, 9:18])
    assert np.all(got.k13 == int(got.scales13.max()) - 132)


def test_rank_slices_and_table_rows():
    from resin.models.deepseek_v41 import DeepSeekV41Config

    if CHECKPOINT is None:
        pytest.skip("DS41_CHECKPOINT is not set")
    config = Path(CHECKPOINT) / "config.json"
    cfg = DeepSeekV41Config.from_hugging_face(json.loads(config.read_text()))
    s = RankSlices.of(cfg, 3, 8)
    assert (s.heads, s.group, s.expert) == (
        slice(24, 32),
        slice(3, 4),
        slice(864, 1152),
    )
    assert s.vocab == slice(3 * 16160, 4 * 16160)
    parts = [table_rows(384006168, r, 8) for r in range(8)]
    assert parts[0].start == 0 and parts[-1].stop == 384006168
    assert all(a.stop == b.start for a, b in zip(parts, parts[1:]))


@pytest.mark.skipif(CHECKPOINT is None, reason="DS41_CHECKPOINT is not set")
def test_real_layer0_dense_and_experts():
    src = reader(CHECKPOINT)
    w, s = src.read("layers.0.attn.wq_a.weight"), src.read("layers.0.attn.wq_a.scale")
    got = mx8_rows(w, s, tau=False)
    value = inc_values(got.bytes) * got.row_scales[:, None]
    want = mx8_f64(w, s)
    assert int(np.count_nonzero(value != want)) == got.inexact
    assert got.inexact / w.size < 1e-3, f"{got.inexact} of {w.size} rounded"
    slots = expert_slots(src, "layers.0.ffn", 4, slice(0, 288), 5120)
    one = src.read("layers.0.ffn.experts.2.w3.weight", slice(0, 288))
    np.testing.assert_array_equal(
        slots.codes13[2].reshape(576, 2560)[256:512], one[:256]
    )
    assert np.all(OCP[w] == OCP[w])  # no NaN bytes in the dense weight


def test_mx8_rows_pads_k_to_tau_blocks():
    rng = np.random.default_rng(3)
    w = ocp_bytes(rng, (64, 288))
    scales = rng.integers(118, 122, (2, 9)).astype(np.uint8)
    natural = mx8_rows(w, scales, tau=False)
    tau = mx8_rows(w, scales)
    assert tau.bytes.shape == (64, 512)
    padded = np.pad(natural.bytes, ((0, 0), (0, 224)))
    np.testing.assert_array_equal(tau.bytes, tperm(padded))
    np.testing.assert_array_equal(tau.row_scales, natural.row_scales)


def test_expert_k_per_expert_when_the_tensor_spreads_too_far():
    """One exponent for the tensor when every block stays exact under it, else each expert's own."""
    from resin_models_gaudi.deepseek_v41.weights import expert_k

    narrow = np.array([[120, 125], [118, 124]], np.uint8)
    assert expert_k(narrow).tolist() == [125 - 132, 125 - 132]
    wide = np.array([[100, 110], [120, 125]], np.uint8)  # 25 exponents over the tensor, <= 13 per expert
    assert expert_k(wide).tolist() == [110 - 132, 125 - 132]


# ------------------------------------------------------------------ the host Engram gather oracle
# weights.py::engram_rows is the NUMPY ORACLE the native (Rust) host-Engram gather is gated against
# (production gather is Rust; this is the reference, not the production path). These tests pin the
# oracle's dequant and its two read paths. RED on the dev-only per-row stub: it has no _ENGRAM_LUT,
# no mmap/pooled machinery, and does not reject the e8m0 NaN scale.


def test_engram_lut_is_bit_identical_to_the_expression_it_replaced():
    """All 65,536 (scale, code) pairs against the per-element f32 expression the table replaced.

    The scope matters: the replaced expression formed its exponent in f32 (`s.astype(np.float32) -
    127`) and cast once, and the table does the same, so they agree at every byte including 254/255
    (overflow to inf where the f32 form did). A form that formed the exponent in float64 would NOT
    agree - the last assertions pin where and how, so nobody "fixes" the table toward float64 without
    seeing that it would change the output dtype. RED on the stub: `_ENGRAM_LUT` does not exist.
    """
    from resin_models_gaudi.deepseek_v41.weights import _ENGRAM_LUT

    codes = np.arange(256, dtype=np.uint8).reshape(1, 256)
    for scale in range(256):
        s = np.full((1, 8), scale, np.uint8)
        with np.errstate(over="ignore", invalid="ignore"):
            old = OCP[codes] * np.exp2(s.astype(np.float32) - 127).repeat(32, axis=1)
            old = old.astype(np.float32)
        new = _ENGRAM_LUT[s.reshape(1, 8)[:, :, None], codes.reshape(1, 8, 32)].reshape(1, 256)
        np.testing.assert_array_equal(new, old)

    # f64 vs f32 exponent: equal for scales 0..246, diverge from 247 up (448 * 2^120 leaves f32 range).
    f64 = OCP[None, :] * np.exp2(np.arange(256, dtype=np.float64) - 127.0)[:, None]
    differing = [
        s for s in range(256) if not np.array_equal(_ENGRAM_LUT[s].astype(np.float64), f64[s])
    ]
    assert differing == list(range(247, 256))
    assert int(np.count_nonzero(np.isfinite(f64[254]) & ~np.isfinite(_ENGRAM_LUT[254]))) == 126
    assert not np.any(np.isfinite(_ENGRAM_LUT[255]))
    assert np.all(np.isfinite(f64[255]))  # finite in float64, not in f32


def _engram_shard(path: Path, codes: np.ndarray, scales: np.ndarray) -> None:
    write_safetensors(
        path,
        {
            "layers.0.engram.embed.weight": ("F8_E4M3", codes),
            "layers.0.engram.embed.scale": ("F8_E8M0", scales),
        },
    )


def _engram_reference(codes: np.ndarray, scales: np.ndarray, ids) -> np.ndarray:
    """The independent f32 dequant the oracle must match (f32 exponent scope, like the LUT)."""
    ids = np.asarray(ids, np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        return (
            OCP[codes[ids]] * np.exp2(scales[ids].astype(np.float32) - 127).repeat(32, axis=1)
        ).astype(np.float32)


def test_engram_rows_rejects_e8m0_nan_scale(tmp_path):
    """e8m0 byte 255 is NaN by the format (manual ch3): reject at load, never substitute a value.

    RED on the stub: it dequantizes 255 to inf/0 silently instead of raising.
    """
    from resin_models_gaudi.deepseek_v41.weights import engram_rows

    name = "model-00001-of-00001.safetensors"
    codes = np.zeros((2, 256), np.uint8)
    scales = np.full((2, 8), 127, np.uint8)
    scales[1, 3] = 0xFF  # the NaN scale
    _engram_shard(tmp_path / name, codes, scales)
    src = reader(tmp_path)
    with pytest.raises(ValueError, match="0xFF|NaN"):
        engram_rows(src, 0, np.array([1], np.int64))
    # a clean row still gathers
    assert engram_rows(src, 0, np.array([0], np.int64)).shape == (1, 256)


@pytest.fixture
def engram_gather(tmp_path, monkeypatch):
    """One Engram layer's shard mapped through the gather's own mmap path.

    Yields (module, reader, codes, scales) with ``_ENGRAM_MODE == "mmap"`` and the maps live.
    Restores every module global it touches and unmaps what is left, so no test leaks state.
    """
    from resin_models_gaudi.deepseek_v41 import weights as W

    rows, name = 512, "model-00001-of-00001.safetensors"
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 0x78, (rows, 256), dtype=np.uint8)  # finite E4M3 codes only
    scales = rng.integers(120, 129, (rows, 8), dtype=np.uint8)
    _engram_shard(tmp_path / name, codes, scales)
    # the freshly written shard's residency is not deterministic; pin it so mmap stays chosen
    monkeypatch.setattr(W, "_engram_residency", lambda *a, **k: 1.0)

    names = (
        "_ENGRAM_MODE",
        "_ENGRAM_RESIDENT",
        "_ENGRAM_MAPS",
        "_ENGRAM_CHECKPOINT",
        "_ENGRAM_SHARDS",
        "_ENGRAM_LAST_CHECK",
    )
    saved = {field: getattr(W, field) for field in names}
    W._ENGRAM_MAPS = {}
    W._ENGRAM_MODE, W._ENGRAM_RESIDENT = "mmap", 1.0
    W._ENGRAM_CHECKPOINT, W._ENGRAM_SHARDS = tmp_path, {0: name}
    W._ENGRAM_LAST_CHECK = time.monotonic() - 1000.0  # past the re-check interval
    try:
        W._engram_maps()
        yield W, reader(tmp_path), codes, scales
    finally:
        for _weight, _scale, mapping, _raw in W._ENGRAM_MAPS.values():
            try:
                mapping.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the test's failure
                pass
        for field, value in saved.items():
            setattr(W, field, value)


def test_engram_gather_oracle_same_shape_different_data(engram_gather):
    """The oracle reads the data: two same-shape row sets with DIFFERENT rows give different results,
    each bit-identical to the independent f32 reference, and pooled == mmap (the required case class
    for gating the Rust gather - a baked-constant cache would return the first set for both).

    RED on the stub: no `_ENGRAM_MAPS`/`_engram_maps` (the fixture cannot even map).
    """
    W, src, codes, scales = engram_gather
    ids_a = np.array([3, 0, 63, 17, 42, 8], np.int64)
    ids_b = np.array([5, 1, 62, 16, 41, 9], np.int64)  # same shape, different rows
    assert W._ENGRAM_MODE == "mmap" and set(W._ENGRAM_MAPS) == {0}

    mmap_a = W.engram_rows(src, 0, ids_a)
    mmap_b = W.engram_rows(src, 0, ids_b)
    np.testing.assert_array_equal(mmap_a, _engram_reference(codes, scales, ids_a))
    np.testing.assert_array_equal(mmap_b, _engram_reference(codes, scales, ids_b))
    assert not np.array_equal(mmap_a, mmap_b)  # different data, not a baked constant

    W._ENGRAM_MODE = "pooled"
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids_a), mmap_a)
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids_b), mmap_b)


def test_engram_gather_pool_path_bit_identical(engram_gather):
    """At/above _ENGRAM_POOL_ROWS the gather spreads reads over the thread pool; the result must still
    be bit-identical to mmap and to the reference (the pool changes concurrency, not numerics)."""
    W, src, codes, scales = engram_gather
    rng = np.random.default_rng(11)
    ids = rng.integers(0, 512, 300, dtype=np.int64)  # >= 256 -> the pooled path
    assert len(ids) >= W._ENGRAM_POOL_ROWS
    mmap = W.engram_rows(src, 0, ids)
    np.testing.assert_array_equal(mmap, _engram_reference(codes, scales, ids))
    W._ENGRAM_MODE = "pooled"
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), mmap)


def test_configure_engram_gather_auto_by_residency(tmp_path, monkeypatch):
    """auto picks mmap only at/above the residency floor, else pooled; an explicit mmap below the
    floor is downgraded. RED on the stub: configure_engram_gather does not exist."""
    from resin_models_gaudi.deepseek_v41 import weights as W

    name = "model-00001-of-00001.safetensors"
    rng = np.random.default_rng(5)
    _engram_shard(
        tmp_path / name,
        rng.integers(0, 0x78, (32, 256), dtype=np.uint8),
        rng.integers(120, 129, (32, 8), dtype=np.uint8),
    )
    names = (
        "_ENGRAM_MODE", "_ENGRAM_RESIDENT", "_ENGRAM_MAPS",
        "_ENGRAM_CHECKPOINT", "_ENGRAM_SHARDS", "_ENGRAM_LAST_CHECK",
    )
    saved = {f: getattr(W, f) for f in names}
    try:
        monkeypatch.setattr(W, "_engram_residency", lambda *a, **k: 0.99)
        W.configure_engram_gather(tmp_path, [0], mode="auto", log=lambda *_: None)
        assert W._ENGRAM_MODE == "mmap"
        W._drop_engram_maps()
        monkeypatch.setattr(W, "_engram_residency", lambda *a, **k: 0.50)
        W.configure_engram_gather(tmp_path, [0], mode="auto", log=lambda *_: None)
        assert W._ENGRAM_MODE == "pooled"
        # explicit mmap below the floor is downgraded to pooled, not honored
        W.configure_engram_gather(tmp_path, [0], mode="mmap", log=lambda *_: None)
        assert W._ENGRAM_MODE == "pooled"
    finally:
        for _w, _s, mapping, _r in W._ENGRAM_MAPS.values():
            try:
                mapping.close()
            except Exception:  # noqa: BLE001
                pass
        for f, v in saved.items():
            setattr(W, f, v)


# ------------------------------------------------------------------ the residency monitor's fail-safe
# The monitor runs inside `engram_rows`, i.e. inside a request. Its fallback must switch to pooled
# without killing the request: mode and registry change FIRST, the numpy buffer exports are dropped
# before close(), and a close that still fails is logged and skipped. Both tests are RED on code that
# closed before clearing the registry or let BufferError escape `mapping.close()`.


def _below_the_floor(monkeypatch, W):
    monkeypatch.setattr(W, "_engram_residency", lambda *a, **k: 0.0)
    W._ENGRAM_LAST_CHECK = time.monotonic() - 1000.0


@contextmanager
def _wright_warnings():
    """Collect this module's logger WARNING records directly (not caplog, which a session-wide
    handler/propagation change can silently empty)."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect()
    logger = logging.getLogger("wright")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_engram_monitor_falls_back_without_raising(engram_gather, monkeypatch):
    W, src, codes, scales = engram_gather
    ids = np.array([3, 0, 63, 17], np.int64)
    expected = _engram_reference(codes, scales, ids)
    assert W._ENGRAM_MODE == "mmap" and set(W._ENGRAM_MAPS) == {0}
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), expected)

    layer = next(iter(W._ENGRAM_MAPS))
    closed: list = []

    class _Counting:
        def __init__(self, mapping):
            self.mapping = mapping

        def close(self):
            closed.append(self.mapping)
            self.mapping.close()

    weight, scale, mapping, raw = W._ENGRAM_MAPS[layer]
    W._ENGRAM_MAPS[layer] = (weight, scale, _Counting(mapping), raw)
    del weight, scale, mapping, raw  # a view held here would keep the export alive

    _below_the_floor(monkeypatch, W)
    with _wright_warnings() as records:
        W._engram_monitor()  # must not raise

    assert W._ENGRAM_MODE == "pooled"
    assert W._ENGRAM_MAPS == {}
    assert len(closed) == 1 and closed[0].closed
    assert not any("unmapping" in record.getMessage() for record in records)
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), expected)


def test_engram_monitor_falls_back_even_when_close_fails(engram_gather, monkeypatch):
    class Unclosable:
        def __init__(self, mapping):
            self.mapping = mapping  # keep the real map alive so the views stay valid

        def close(self):
            raise OSError(5, "Input/output error")

    W, src, codes, scales = engram_gather
    ids = np.array([3, 0, 63, 17], np.int64)
    expected = _engram_reference(codes, scales, ids)
    layer, (weight, scale, mapping, raw) = next(iter(W._ENGRAM_MAPS.items()))
    W._ENGRAM_MAPS[layer] = (weight, scale, Unclosable(mapping), raw)

    _below_the_floor(monkeypatch, W)
    with _wright_warnings() as records:
        W._engram_monitor()  # must not raise even though close() does

    assert W._ENGRAM_MODE == "pooled"
    assert W._ENGRAM_MAPS == {}
    assert any("unmap" in record.getMessage() for record in records)
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), expected)
