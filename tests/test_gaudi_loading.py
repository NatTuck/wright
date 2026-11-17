"""The Gaudi weight loader's checkpoint reader and shared conversions, without a card."""

from __future__ import annotations

import numpy as np
import pytest
from safetensors_files import write_checkpoint, write_safetensors

from resin_models_gaudi.loading.convert import (
    E4M3,
    bf16_rne,
    bf16_to_f32,
    block_f32,
    e4m3_encode,
    needs_halving,
)
from resin_models_gaudi.loading.source import Safetensors

MODES = ("pread", "mmap")


def _checkpoint(tmp_path, rng):
    bf = bf16_rne(rng.standard_normal((6, 8)).astype(np.float32))
    f32 = rng.standard_normal((5,)).astype(np.float32)
    w8 = e4m3_encode(rng.uniform(-200, 200, (256, 256)).astype(np.float32))
    scales = rng.uniform(0.5, 2.0, (2, 2)).astype(np.float32)
    ids = np.arange(7, dtype=np.int64)
    write_checkpoint(
        tmp_path,
        {
            "model-00001-of-00002.safetensors": {
                "a.bf16": ("BF16", bf),
                "a.f32": ("F32", f32),
            },
            "model-00002-of-00002.safetensors": {
                "b.w": ("F8_E4M3", w8),
                "b.w_scale_inv": ("F32", scales),
                "b.ids": ("I64", ids),
            },
        },
    )
    return bf, f32, w8, scales, ids


@pytest.mark.parametrize("mode", MODES)
def test_reads_every_shard_in_both_modes(tmp_path, mode):
    bf, f32, w8, scales, ids = _checkpoint(tmp_path, np.random.default_rng(0))
    src = Safetensors(tmp_path, mode=mode)
    assert set(src.meta) == {"a.bf16", "a.f32", "b.w", "b.w_scale_inv", "b.ids"}
    assert src.weight_map["b.w"] == "model-00002-of-00002.safetensors"
    assert src.info("a.bf16") == ("BF16", (6, 8))
    assert src.nbytes("b.w") == 256 * 256

    raw, dtype = src.raw("a.bf16")
    assert dtype == "BF16" and not raw.flags.writeable
    np.testing.assert_array_equal(raw, bf)
    np.testing.assert_array_equal(src.raw("b.ids")[0], ids)

    rows = src.read("a.bf16", slice(2, 5))
    assert rows.flags.writeable
    np.testing.assert_array_equal(rows, bf[2:5])
    np.testing.assert_array_equal(src.f32("a.bf16"), bf16_to_f32(bf))
    np.testing.assert_array_equal(src.f32("a.f32"), f32)
    np.testing.assert_array_equal(src.bf16("a.bf16", slice(0, 1)), bf[:1])

    w, s = src.fp8("b.w", slice(128, 256))
    np.testing.assert_array_equal(w, w8[128:])
    np.testing.assert_array_equal(s, scales[1:])
    np.testing.assert_array_equal(src.fp8_f32("b.w"), block_f32(w8, scales))
    src.close()


@pytest.mark.parametrize("mode", MODES)
def test_counts_the_bytes_it_reads(tmp_path, mode):
    bf, *_ = _checkpoint(tmp_path, np.random.default_rng(1))
    src = Safetensors(tmp_path, mode=mode)
    src.raw("a.bf16")
    src.read("a.bf16", slice(0, 2))
    assert src.bytes_read == bf.nbytes + 2 * bf[0].nbytes


def test_typed_reads_check_the_dtype(tmp_path):
    _checkpoint(tmp_path, np.random.default_rng(2))
    src = Safetensors(tmp_path)
    with pytest.raises(ValueError, match="expected bf16"):
        src.bf16("a.f32")
    with pytest.raises(ValueError, match="expected block FP8"):
        src.fp8("a.bf16")
    with pytest.raises(ValueError, match="use fp8_f32"):
        src.f32("b.w")
    with pytest.raises(KeyError, match="not in the checkpoint"):
        src.raw("c")


def test_a_missing_shard_fails_unless_allowed(tmp_path):
    _checkpoint(tmp_path, np.random.default_rng(3))
    (tmp_path / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        Safetensors(tmp_path)
    src = Safetensors(tmp_path, missing_ok=True)
    assert src.missing == ["model-00002-of-00002.safetensors"]
    assert "a.bf16" in src and "b.w" not in src
    with pytest.raises(KeyError, match="files missing"):
        src.raw("b.w")


def test_without_an_index_every_file_is_a_shard_and_one_file_is_a_checkpoint(tmp_path):
    rng = np.random.default_rng(4)
    a = bf16_rne(rng.standard_normal((3, 4)).astype(np.float32))
    b = rng.standard_normal((2,)).astype(np.float32)
    write_safetensors(tmp_path / "one.safetensors", {"a": ("BF16", a)})
    write_safetensors(tmp_path / "two.safetensors", {"b": ("F32", b)})
    assert set(Safetensors(tmp_path).meta) == {"a", "b"}
    single = Safetensors(tmp_path / "two.safetensors")
    assert set(single.meta) == {"b"}
    np.testing.assert_array_equal(single.f32("b"), b)


def test_e4m3_encode_keeps_every_value_and_rounds_ties_to_even():
    codes = np.concatenate([np.arange(0x78), 0x80 + np.arange(0x78)]).astype(np.uint8)
    values = E4M3[codes].astype(np.float32)  # |x| <= 240, the encoder's range
    np.testing.assert_array_equal(e4m3_encode(values), codes)
    np.testing.assert_array_equal(e4m3_encode(values.reshape(16, 15)), codes.reshape(16, 15))
    for sign in (0, 0x80):
        low = np.arange(0x77)  # each code and the next one up
        mid = ((E4M3[sign + low] + E4M3[sign + low + 1]) / 2).astype(np.float32)
        np.testing.assert_array_equal(e4m3_encode(mid), sign + low + (low & 1))
        away = np.nextafter(mid, np.copysign(np.float32(np.inf), mid))
        np.testing.assert_array_equal(e4m3_encode(away), sign + low + 1)


def test_needs_halving_reads_the_first_256_rows():
    gaudi = e4m3_encode(np.full((512, 4), 200.0, np.float32))
    assert not needs_halving(gaudi)
    ocp = gaudi.copy()
    ocp[255, 0] = 0x7E  # 448 in e4m3, above the Gaudi maximum of 240
    assert np.nanmax(E4M3[ocp]) > 240
    assert needs_halving(ocp)
    late = gaudi.copy()
    late[256, 0] = 0x7E  # past the rows the check reads
    assert not needs_halving(late)


def test_block_f32_scales_each_128_block():
    w = e4m3_encode(np.ones((256, 128), np.float32))
    s = np.array([[2.0], [3.0]], np.float32)
    out = block_f32(w, s)
    assert out.shape == (256, 128)
    np.testing.assert_array_equal(out[:128], 2.0)
    np.testing.assert_array_equal(out[128:], 3.0)
