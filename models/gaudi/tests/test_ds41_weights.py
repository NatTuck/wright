"""DeepSeek-V4.1 weight preparation (host only): FP8 conversion, expert slots, rank slices. With
DS41_CHECKPOINT set, also on real tensors of layer 0."""

from __future__ import annotations

import json
import logging
import os
import struct
import time
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
from resin_models_gaudi.qwen3_5.checkpoint import tperm

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


# ------------------------------------------------------------------ the host Engram gather
# The residency monitor runs inside `engram_rows`, i.e. inside a request. Its fallback path used to
# raise BufferError out of `mapping.close()` (numpy still held an exported buffer on the mmap) and
# it closed before clearing the registry, so a residency drop mid-request killed the request and
# left live numpy arrays over unmapped memory. These two tests drive the fallback directly; both
# fail on that code and neither needs a device.
@pytest.fixture
def engram_gather(tmp_path, monkeypatch):
    """One Engram layer's shard mapped through the gather's own mmap path.

    Yields (module, reader, row ids, expected rows) with ``_ENGRAM_MODE == "mmap"`` and the maps
    live. Restores every module global it touches and unmaps what is left, so no test leaks state.
    """
    from resin_models_gaudi.deepseek_v41 import weights as W

    rows, name = 64, "model-00001-of-00001.safetensors"
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 0x78, (rows, 256), dtype=np.uint8)  # finite E4M3 codes only
    scales = rng.integers(120, 129, (rows, 8), dtype=np.uint8)
    write_safetensors(
        tmp_path / name,
        {
            "layers.0.engram.embed.weight": ("F8_E4M3", codes),
            "layers.0.engram.embed.scale": ("F8_E8M0", scales),
        },
    )
    expected = (
        OCP[codes] * np.exp2(scales.astype(np.float64) - 127.0).repeat(32, axis=1)
    ).astype(np.float32)

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
        ids = np.array([3, 0, 63, 17], np.int64)
        yield W, reader(tmp_path), ids, expected
    finally:
        for _weight, _scale, mapping, _raw in W._ENGRAM_MAPS.values():
            try:
                mapping.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the test's failure
                pass
        for field, value in saved.items():
            setattr(W, field, value)


def _below_the_floor(monkeypatch, W):
    """Make the periodic residency check read a table that fell out of page cache."""
    monkeypatch.setattr(W, "_engram_residency", lambda *a, **k: 0.0)
    # the gather itself calls the monitor, which stamps the interval: force the next call past it
    W._ENGRAM_LAST_CHECK = time.monotonic() - 1000.0


def test_engram_monitor_falls_back_without_raising(engram_gather, monkeypatch, caplog):
    """A residency drop mid-request: the gather must switch to pooled, not kill the request.

    RED on the old code: `mapping.close()` raised BufferError (numpy held the mmap's exported
    buffer) and `except OSError` did not catch it, so this call escaped into the request.
    """
    W, src, ids, expected = engram_gather
    assert W._ENGRAM_MODE == "mmap" and set(W._ENGRAM_MAPS) == {0}
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), expected[ids])

    _below_the_floor(monkeypatch, W)
    with caplog.at_level(logging.WARNING, logger="wright"):
        W._engram_monitor()  # must not raise

    assert W._ENGRAM_MODE == "pooled"
    assert W._ENGRAM_MAPS == {}
    # the views were dropped before the close: no unmap failure, not merely a caught one
    assert not any("unmapping" in record.getMessage() for record in caplog.records)
    # and the request it was called from still gets its rows, now from the pooled gather
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), expected[ids])


def test_engram_monitor_falls_back_even_when_close_fails(
    engram_gather, monkeypatch, caplog
):
    """The contract is "a monitor failure never takes a request down", not "close() works".

    The unmap is best-effort: the mode and the registry are what the gather reads, so they must
    change first and the fallback must happen even when closing the mapping itself fails.
    """

    class Unclosable:
        """Stands in for the mmap: `close()` fails the way a racing gather would make it fail."""

        def __init__(self, mapping):
            self.mapping = mapping  # keep the real map alive so the views stay valid

        def close(self):
            raise OSError(5, "Input/output error")

    W, src, ids, expected = engram_gather
    layer, (weight, scale, mapping, raw) = next(iter(W._ENGRAM_MAPS.items()))
    W._ENGRAM_MAPS[layer] = (weight, scale, Unclosable(mapping), raw)

    _below_the_floor(monkeypatch, W)
    with caplog.at_level(logging.WARNING, logger="wright"):
        W._engram_monitor()  # must not raise even though close() does

    assert W._ENGRAM_MODE == "pooled"
    assert W._ENGRAM_MAPS == {}
    assert any("unmap" in record.getMessage() for record in caplog.records)
    np.testing.assert_array_equal(W.engram_rows(src, 0, ids), expected[ids])

