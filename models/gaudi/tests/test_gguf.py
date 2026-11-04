"""GGUF weight source: header parsing, split shards, pread row access, and the C packer and
dequantizer against their numpy oracles (bit-exact). Real-checkpoint checks run when the
Mistral-Large-4 GGUF (and optionally gguf-py) are present."""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from resin_models_gaudi.gguf import planar as P
from resin_models_gaudi.gguf import quants as Q
from resin_models_gaudi.gguf.reader import GgufError, GgufModel

ML4_GGUF = Path(os.environ.get("RESIN_ML4_GGUF", "/nonexistent"))
ML4_TOKENIZER = Path(os.environ.get("RESIN_ML4_TOKENIZER", "/nonexistent"))
GGUF_PY = Path(os.environ.get("RESIN_GGUF_PY", "/nonexistent"))

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="requires gcc")
needs_ml4 = pytest.mark.skipif(not ML4_GGUF.exists(), reason="no Mistral-Large-4 GGUF")

TYPE_IDS = {"F32": 0, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13}


# ------------------------------------------------------------------ a minimal GGUF writer
def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<Ii", 5, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _string(v)
    if isinstance(v, list) and v and isinstance(v[0], str):
        return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_string(s) for s in v)
    if isinstance(v, list):
        return struct.pack("<IIQ", 9, 5, len(v)) + np.asarray(v, np.int32).tobytes()
    raise TypeError(v)


def write_gguf(path: Path, metadata: dict, tensors: dict, alignment: int = 32) -> None:
    """tensors: name -> (qtype, numpy shape, raw bytes)"""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        head += _string(k) + _value(v)
    data, infos = b"", b""
    for name, (qtype, shape, raw) in tensors.items():
        data += b"\0" * (-len(data) % alignment)
        infos += _string(name) + struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<Q", d) for d in reversed(shape))
        infos += struct.pack("<IQ", TYPE_IDS[qtype], len(data))
        data += raw
    head += infos
    head += b"\0" * (-len(head) % alignment)
    path.write_bytes(head + data)


def random_blocks(rng, qtype: str, rows: int, inputs: int) -> np.ndarray:
    """valid native blocks: random payload, finite f16 scales"""
    _, elements, size = Q.LQ_TYPES[qtype]
    nb = inputs // elements
    b = rng.integers(0, 256, (rows, nb, size), dtype=np.uint8)
    nf = 1 if qtype == "Q8_0" else 2
    scales = rng.uniform(-0.02, 0.02, (rows, nb, nf)).astype(np.float16)
    b[:, :, : 2 * nf] = scales.view(np.uint8).reshape(rows, nb, 2 * nf)
    return b.reshape(rows, nb * size)


# ------------------------------------------------------------------ reader
def test_reader_parses_metadata_and_split_shards(tmp_path):
    rng = np.random.default_rng(0)
    q8 = random_blocks(rng, "Q8_0", 6, 64)
    q4 = random_blocks(rng, "Q4_K", 2 * 3, 512)  # 2 experts x 3 rows
    f = rng.standard_normal(5).astype(np.float32)
    meta = {
        "general.architecture": "toy",
        "toy.count": 7,
        "toy.eps": 0.5,
        "toy.flag": True,
        "toy.pattern": [1, 0, 1],
        "toy.names": ["a", "bc"],
        "split.count": 2,
        "split.tensors.count": 3,
    }
    write_gguf(
        tmp_path / "m-00001-of-00002.gguf",
        meta,
        {
            "a.weight": ("Q8_0", (6, 64), q8.tobytes()),
            "n.weight": ("F32", (5,), f.tobytes()),
        },
    )
    write_gguf(
        tmp_path / "m-00002-of-00002.gguf",
        {"split.no": 1},
        {"x.weight": ("Q4_K", (2, 3, 512), q4.tobytes())},
        alignment=32,
    )
    m = GgufModel(tmp_path)
    assert m.architecture == "toy"
    assert m.field("toy.count") == 7 and m.field("toy.flag") is True
    assert m.field("toy.pattern") == [1, 0, 1] and m.field("toy.names") == ["a", "bc"]
    assert m.field("toy.eps") == pytest.approx(0.5)
    a = m.tensor("a.weight")
    assert (a.qtype, a.shape, a.row_bytes) == ("Q8_0", (6, 64), 68)
    np.testing.assert_array_equal(m.read_rows("a.weight", slice(2, 5)), q8[2:5])
    np.testing.assert_array_equal(m.read("n.weight"), f)
    x = m.tensor("x.weight")
    assert x.shape == (2, 3, 512) and x.rows == 3
    np.testing.assert_array_equal(m.read_rows("x.weight", expert=1), q4[3:6])
    np.testing.assert_array_equal(
        m.read_rows("x.weight", slice(1, 3), expert=0), q4[1:3]
    )
    with pytest.raises(ValueError):
        m.read_rows("x.weight")  # a stacked tensor needs an expert
    # a shard path finds its split; an incomplete split fails
    assert len(GgufModel(tmp_path / "m-00002-of-00002.gguf").tensors) == 3
    (tmp_path / "m-00002-of-00002.gguf").unlink()
    with pytest.raises(GgufError, match="incomplete"):
        GgufModel(tmp_path)


def test_reader_rejects_truncated_files(tmp_path):
    q8 = random_blocks(np.random.default_rng(1), "Q8_0", 4, 32)
    path = tmp_path / "t.gguf"
    write_gguf(
        path, {"general.architecture": "toy"}, {"w": ("Q8_0", (4, 32), q8.tobytes())}
    )
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(GgufError, match="past the end"):
        GgufModel(path)


# ------------------------------------------------------------------ packing
@needs_cc
@pytest.mark.parametrize("qtype", ["Q4_K", "Q5_K", "Q8_0"])
def test_c_dequantize_matches_numpy(qtype):
    from resin_models_gaudi.gguf import pack

    B = random_blocks(np.random.default_rng(2), qtype, 300, 512)
    want = Q.dequantize(B, qtype)
    got = pack.dequantize(B, qtype)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@needs_cc
@pytest.mark.parametrize("qtype", ["Q4_K", "Q5_K", "Q8_0"])
def test_c_planar_matches_numpy(qtype):
    from resin_models_gaudi.gguf import pack

    rng = np.random.default_rng(3)
    B = random_blocks(rng, qtype, 200, 1024)  # 200 rows: a padded second row block
    got = pack.planar(B, qtype)
    np.testing.assert_array_equal(got, P.gguf_planar(B, qtype))
    assert got.nbytes == P.size(qtype, 200, 1024)
    _, elements, size = Q.LQ_TYPES[qtype]
    sb = P.LAYOUT[qtype][0]
    # a row slice and an input slice (whole superblocks)
    nsb = 1024 // sb
    k0, k1 = nsb // 4, nsb // 2
    got = pack.planar(B, qtype, rows=slice(64, 192), superblocks=slice(k0, k1))
    per_sb = sb // elements * size
    want = P.gguf_planar(B[64:192, k0 * per_sb : k1 * per_sb], qtype)
    np.testing.assert_array_equal(got, want)


@needs_cc
def test_planar_experts_reads_rank_slices(tmp_path):
    from resin_models_gaudi.gguf import pack

    rng = np.random.default_rng(4)
    E, M, K = 3, 256, 1024
    B = random_blocks(rng, "Q4_K", E * M, K)
    write_gguf(
        tmp_path / "e.gguf",
        {"general.architecture": "toy"},
        {"x": ("Q4_K", (E, M, K), B.tobytes())},
    )
    m = GgufModel(tmp_path / "e.gguf")
    rows, inputs = slice(128, 256), slice(256, 768)
    got = pack.planar_experts(m, "x", [(rows, None), (None, inputs)])
    per_sb = 144
    for e in range(E):
        Be = B[e * M : (e + 1) * M]
        want = np.concatenate(
            [
                P.gguf_planar(Be[rows], "Q4_K").reshape(-1),
                P.gguf_planar(Be[:, 1 * per_sb : 3 * per_sb], "Q4_K").reshape(-1),
            ]
        )
        np.testing.assert_array_equal(got[e], want)


# ------------------------------------------------------------------ the real checkpoint
@needs_ml4
def test_ml4_gguf_matches_gguf_py_and_params():
    from resin.models.mistral_large4.config import MistralLarge4Config

    m = GgufModel(ML4_GGUF)
    cfg = MistralLarge4Config.from_gguf(m.metadata)
    assert cfg.num_hidden_layers == 72 and len(cfg.kv_owner_layers) == 63
    assert len(cfg.index_owner_layers) == 44
    assert {t.qtype for t in m} == {"F32", "Q8_0", "Q4_K", "Q5_K"}
    params = ML4_TOKENIZER / "params.json"
    if params.exists():
        other = MistralLarge4Config.from_params(json.loads(params.read_text()))
        for field in cfg.__dataclass_fields__:
            a, b = getattr(cfg, field), getattr(other, field)
            assert (
                a == pytest.approx(b, rel=1e-6) if isinstance(a, float) else a == b
            ), field
    if not GGUF_PY.exists():
        return
    sys.path.insert(0, str(GGUF_PY))
    try:
        import gguf
    finally:
        sys.path.remove(str(GGUF_PY))
    first = gguf.GGUFReader(m.paths[0])
    for t in first.tensors:
        ours = m.tensor(t.name)
        assert ours.qtype == t.tensor_type.name
        assert ours.shape == tuple(reversed([int(x) for x in t.shape]))
        assert ours.offset == int(t.data_offset)
    name = "blk.1.ffn_down_exps.weight"
    t = m.tensor(name)
    rows = m.read_rows(name, slice(0, 16), expert=2)
    want = gguf.quants.dequantize(rows, gguf.GGMLQuantizationType[t.qtype])
    np.testing.assert_array_equal(Q.dequantize(rows, t.qtype), want.astype(np.float32))


@needs_ml4
def test_ml4_tokenizer_matches_gguf_vocabulary():
    tokenizers = pytest.importorskip("tokenizers")
    path = ML4_TOKENIZER / "tokenizer.json"
    if not path.exists():
        pytest.skip("no Mistral-Large-4 tokenizer.json")
    tokens = GgufModel(ML4_GGUF).field("tokenizer.ggml.tokens")
    vocab = tokenizers.Tokenizer.from_file(str(path)).get_vocab(with_added_tokens=True)
    assert len(vocab) == len(tokens)
    assert all(vocab.get(s) == i for i, s in enumerate(tokens))
