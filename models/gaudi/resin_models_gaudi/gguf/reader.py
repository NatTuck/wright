"""GGUF (v2/v3) checkpoints: metadata, the tensor directory and tensor bytes.

A model is one ``.gguf`` file or a directory of ``*-of-*.gguf`` split shards. Only the headers
are parsed here; tensor bytes are read with ``os.preadv`` into caller buffers and never memory
mapped.

Shapes are in numpy order (ggml's ``ne`` reversed): a 2-D weight is ``(rows, inputs)`` and a
stacked expert tensor ``(experts, rows, inputs)``. A row of a block-quantized tensor is
``inputs / block_elements`` blocks of ``block_bytes``.
"""

from __future__ import annotations

import os
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# ggml_type -> (name, block elements, block bytes)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

# metadata value types
_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING, _ARRAY = 8, 9
_ARRAY_DTYPES = {
    0: np.uint8,
    1: np.int8,
    2: np.uint16,
    3: np.int16,
    4: np.uint32,
    5: np.int32,
    6: np.float32,
    7: np.bool_,
    10: np.uint64,
    11: np.int64,
    12: np.float64,
}

_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class GgufError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    qtype: str
    shape: tuple[int, ...]
    """numpy order"""
    path: Path
    offset: int
    """absolute byte offset of the tensor's data in ``path``"""
    nbytes: int
    block_elements: int
    block_bytes: int

    @property
    def row_bytes(self) -> int:
        return self.shape[-1] // self.block_elements * self.block_bytes

    @property
    def rows(self) -> int:
        """rows of the trailing matrix (``shape[-2]``; 1 for a vector)"""
        return self.shape[-2] if len(self.shape) > 1 else 1


class _Header:
    """Sequential reader over a file header (buffered: headers with vocab arrays are MBs)."""

    def __init__(self, path: Path) -> None:
        self.file = open(path, "rb", buffering=1 << 20)
        self.path = path

    def read(self, n: int) -> bytes:
        data = self.file.read(n)
        if len(data) != n:
            raise GgufError(f"{self.path}: truncated header")
        return data

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.read(self.unpack("<Q")).decode("utf-8", errors="surrogateescape")

    def value(self, vtype: int) -> Any:
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, count = self.unpack("<I"), self.unpack("<Q")
            if itype in _ARRAY_DTYPES:
                dtype = np.dtype(_ARRAY_DTYPES[itype])
                return np.frombuffer(self.read(count * dtype.itemsize), dtype).tolist()
            return [self.value(itype) for _ in range(count)]
        raise GgufError(f"{self.path}: unknown metadata type {vtype}")

    def close(self) -> None:
        self.file.close()


def _parse(path: Path) -> tuple[dict[str, Any], list[TensorInfo]]:
    h = _Header(path)
    try:
        if h.read(4) != MAGIC:
            raise GgufError(f"{path}: not a GGUF file")
        version = h.unpack("<I")
        if version not in (2, 3):
            raise GgufError(f"{path}: GGUF version {version} is not supported")
        n_tensors, n_kv = h.unpack("<Q"), h.unpack("<Q")
        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = h.string()
            metadata[key] = h.value(h.unpack("<I"))
        raw = []
        for _ in range(n_tensors):
            name = h.string()
            ne = [h.unpack("<Q") for _ in range(h.unpack("<I"))]
            raw.append((name, ne, h.unpack("<I"), h.unpack("<Q")))
        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        start = -(-h.file.tell() // alignment) * alignment
    finally:
        h.close()
    size = path.stat().st_size
    tensors = []
    for name, ne, ty, rel in raw:
        if ty not in GGML_TYPES:
            raise GgufError(f"{path}: tensor {name} has unknown ggml type {ty}")
        qtype, elements, block = GGML_TYPES[ty]
        shape = tuple(int(x) for x in reversed(ne))
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if shape and shape[-1] % elements:
            raise GgufError(f"{path}: {name} rows are not whole {qtype} blocks")
        nbytes = count // elements * block
        offset = start + rel
        if offset + nbytes > size:
            raise GgufError(f"{path}: {name} extends past the end of the file")
        tensors.append(
            TensorInfo(name, qtype, shape, path, offset, nbytes, elements, block)
        )
    return metadata, tensors


def shard_paths(path: str | os.PathLike[str]) -> list[Path]:
    """The ``.gguf`` files of a model: one file, or every split shard of a directory (or of the
    split a shard path names), checked complete."""
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix == ".gguf")
        if not files:
            raise GgufError(f"{p}: no .gguf files")
    else:
        m = _SPLIT.search(p.name)
        files = (
            sorted(p.parent.glob(p.name[: m.start()] + "-*-of-*.gguf")) if m else [p]
        )
    found = [_SPLIT.search(f.name) for f in files]
    if len(files) > 1 or found[0]:
        splits = [s for s in found if s is not None]
        if len(splits) != len(files) or len({s.group(2) for s in splits}) != 1:
            raise GgufError(f"{p}: mixed or unnumbered GGUF splits")
        total = int(splits[0].group(2))
        if sorted(int(s.group(1)) for s in splits) != list(range(1, total + 1)):
            raise GgufError(f"{p}: incomplete split ({len(files)} of {total} shards)")
    return files


class GgufModel:
    """Metadata (the first shard's) and the tensor directory of all shards."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.paths = shard_paths(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        for k, shard in enumerate(self.paths):
            metadata, tensors = _parse(shard)
            if k == 0:
                self.metadata = metadata
            for t in tensors:
                if t.name in self.tensors:
                    raise GgufError(f"tensor {t.name} appears in two shards")
                self.tensors[t.name] = t
        expected = self.metadata.get("split.tensors.count")
        if expected is not None and expected != len(self.tensors):
            raise GgufError(
                f"split lists {expected} tensors, found {len(self.tensors)}"
            )
        self._fds: dict[Path, int] = {}

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __iter__(self) -> Iterator[TensorInfo]:
        return iter(self.tensors.values())

    def tensor(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"GGUF has no tensor {name}") from None

    def field(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def _fd(self, path: Path) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY)
        return fd

    def pread(self, path: Path, offset: int, out: np.ndarray) -> np.ndarray:
        """Fill the C-contiguous buffer ``out`` from ``path`` at ``offset`` (releases the GIL)."""
        view = memoryview(out).cast("B")
        fd, done = self._fd(path), 0
        while done < len(view):
            n = os.preadv(fd, [view[done:]], offset + done)
            if n <= 0:
                raise GgufError(f"{path}: short read at {offset + done}")
            done += n
        return out

    def read_rows(
        self,
        name: str,
        rows: slice | None = None,
        *,
        expert: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Native bytes ``[rows, row_bytes]`` u8 of a matrix, or of one expert of a stacked
        3-D tensor (F32 / F16 / BF16 tensors come back as raw bytes too)."""
        t = self.tensor(name)
        if (len(t.shape) == 3) != (expert is not None):
            raise ValueError(f"{name}: shape {t.shape} with expert={expert}")
        M = t.rows
        rows = rows or slice(0, M)
        start, stop, step = rows.indices(M)
        if step != 1:
            raise ValueError("row slices must be contiguous")
        offset = t.offset + ((expert or 0) * M + start) * t.row_bytes
        if out is None:
            out = np.empty((stop - start, t.row_bytes), np.uint8)
        elif out.shape != (stop - start, t.row_bytes) or out.dtype != np.uint8:
            raise ValueError(f"{name}: out must be u8 {(stop - start, t.row_bytes)}")
        return self.pread(t.path, offset, out)

    def read(self, name: str) -> np.ndarray:
        """A whole tensor's native bytes, viewed as its values for F32 / F16 / BF16 / I32
        (BF16 as u16 bits), else ``[*leading, row_bytes]`` u8."""
        t = self.tensor(name)
        buf = self.pread(t.path, t.offset, np.empty(t.nbytes, np.uint8))
        plain = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,
            "I32": np.int32,
        }
        if t.qtype in plain:
            return buf.view(plain[t.qtype]).reshape(t.shape)
        return buf.reshape(*t.shape[:-1], t.row_bytes)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> GgufModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
