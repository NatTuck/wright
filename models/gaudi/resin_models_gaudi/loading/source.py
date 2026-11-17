"""Checkpoint readers.

``Safetensors`` reads a safetensors checkpoint: the shards that
``model.safetensors.index.json`` lists, or else every ``*.safetensors`` file of the directory,
or one ``.safetensors`` file. It reads the headers of every shard at construction.

Two read modes:

- ``pread``: each read is one large ``os.preadv`` per GiB into a new array. A network file
  system serves this at full speed, where page faults over a memory map are slow.
- ``mmap``: ``raw`` returns zero-copy views of a read-only memory map. ``read`` and the typed
  accessors copy.

Both modes count the bytes they read (``bytes_read``).
"""

from __future__ import annotations

import json
import os
import struct
import threading
from pathlib import Path

import numpy as np

from resin_models_gaudi.loading.convert import BLOCK, bf16_to_f32, block_f32

DTYPES: dict[str, type] = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "F8_E4M3": np.uint8,
    "F8_E8M0": np.uint8,
    "U8": np.uint8,
    "I8": np.uint8,
    "I64": np.int64,
}
"""NumPy types of the safetensors dtypes. BF16 is its bits (uint16); FP8 and I8 are their bytes."""

_PIECE = 1 << 30


class Safetensors:
    """The tensors of a safetensors checkpoint, by name.

    ``meta[name]`` is ``(file, data offset, dtype, shape)``. With ``missing_ok``, shards that the
    index lists but the directory lacks (a partial download) only lose their tensors; their names
    are in ``missing``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        mode: str = "pread",
        dtypes: dict[str, type] = DTYPES,
        missing_ok: bool = False,
    ) -> None:
        if mode not in ("pread", "mmap"):
            raise ValueError(f"read mode {mode!r} is not pread or mmap")
        self.path = Path(path)
        self.mode = mode
        self.dtypes = dtypes
        index = self.path / "model.safetensors.index.json"
        if self.path.is_file():
            files = [self.path.name]
            self.path = self.path.parent
        elif index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.path.glob("*.safetensors"))
        self.meta: dict[str, tuple[Path, int, str, tuple[int, ...]]] = {}
        self.missing: list[str] = []
        for name in files:
            file = self.path / name
            if not file.exists():
                if not missing_ok:
                    raise FileNotFoundError(
                        f"{file}: a shard of the checkpoint is missing"
                    )
                self.missing.append(name)
                continue
            with open(file, "rb") as handle:
                (size,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(size))
            header.pop("__metadata__", None)
            for tensor, entry in header.items():
                self.meta[tensor] = (
                    file,
                    8 + size + entry["data_offsets"][0],
                    entry["dtype"],
                    tuple(entry["shape"]),
                )
        self.bytes_read = 0
        self._fds: dict[Path, int] = {}
        self._maps: dict[Path, np.memmap] = {}
        self._lock = threading.Lock()

    # ---- headers
    def __contains__(self, name: str) -> bool:
        return name in self.meta

    @property
    def weight_map(self) -> dict[str, str]:
        """Each tensor's shard file name."""
        return {tensor: entry[0].name for tensor, entry in self.meta.items()}

    def _entry(self, name: str) -> tuple[Path, int, str, tuple[int, ...]]:
        entry = self.meta.get(name)
        if entry is None:
            raise KeyError(
                f"{name}: not in the checkpoint at {self.path}"
                + (" (files missing)" if self.missing else "")
            )
        return entry

    def info(self, name: str) -> tuple[str, tuple[int, ...]]:
        """(safetensors dtype, shape)"""
        _, _, dtype, shape = self._entry(name)
        return dtype, shape

    def nbytes(self, name: str) -> int:
        _, _, dtype, shape = self._entry(name)
        return int(np.prod(shape)) * np.dtype(self.dtypes[dtype]).itemsize

    # ---- reads
    def _fd(self, file: Path) -> int:
        with self._lock:
            fd = self._fds.get(file)
            if fd is None:
                fd = self._fds[file] = os.open(file, os.O_RDONLY)
            return fd

    def _count(self, nbytes: int) -> None:
        with self._lock:
            self.bytes_read += nbytes

    def _pread(self, name: str, file: Path, start: int, out: np.ndarray) -> None:
        fd = self._fd(file)
        view = memoryview(out.reshape(-1).view(np.uint8))
        offset = 0
        while offset < len(view):
            n = os.preadv(fd, [view[offset : offset + _PIECE]], start + offset)
            if n <= 0:
                raise OSError(f"short read of {name} from {file}")
            offset += n
        self._count(len(view))

    def _view(self, file: Path, start: int, count: int) -> np.ndarray:
        with self._lock:
            data = self._maps.get(file)
            if data is None:
                data = self._maps[file] = np.memmap(file, np.uint8, "r")
        return data[start : start + count]

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        """A whole tensor as a read-only array in its NumPy type, and its safetensors dtype."""
        file, start, dtype, shape = self._entry(name)
        kind = np.dtype(self.dtypes[dtype])
        count = int(np.prod(shape)) * kind.itemsize
        if self.mode == "mmap":
            data = self._view(file, start, count)
            self._count(count)
        else:
            data = np.empty(count, np.uint8)
            self._pread(name, file, start, data)
            data.flags.writeable = False
        return data.view(kind).reshape(shape), dtype

    def read(self, name: str, rows: slice | None = None) -> np.ndarray:
        """A tensor, or a range of its first dimension, as a new array in its NumPy type."""
        file, start, dtype, shape = self._entry(name)
        kind = np.dtype(self.dtypes[dtype])
        if rows is not None:
            start += rows.start * int(np.prod(shape[1:])) * kind.itemsize
            shape = (rows.stop - rows.start, *shape[1:])
        out = np.empty(shape, kind)
        if self.mode == "mmap":
            out.reshape(-1).view(np.uint8)[:] = self._view(file, start, out.nbytes)
            self._count(out.nbytes)
        else:
            self._pread(name, file, start, out)
        return out

    # ---- typed reads (new arrays)
    def f32(self, name: str, rows: slice | None = None) -> np.ndarray:
        """A BF16 or F32 tensor as float32."""
        a = self.read(name, rows)
        if a.dtype == np.uint16:
            return bf16_to_f32(a)
        if a.dtype == np.uint8:
            raise ValueError(f"{name}: block FP8, use fp8_f32")
        return np.asarray(a, np.float32)

    def bf16(self, name: str, rows: slice | None = None) -> np.ndarray:
        """A BF16 tensor as its bits (uint16)."""
        a = self.read(name, rows)
        if a.dtype != np.uint16:
            raise ValueError(f"{name}: expected bf16, got {a.dtype}")
        return a

    def fp8(
        self, name: str, rows: slice | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(raw e4m3 bytes, f32 block scales) of a block-FP8 tensor; ``rows`` aligned to 128."""
        dtype, _ = self.info(name)
        if dtype != "F8_E4M3":
            raise ValueError(f"{name}: expected block FP8, got {dtype}")
        scale_rows = (
            None if rows is None else slice(rows.start // BLOCK, rows.stop // BLOCK)
        )
        return self.read(name, rows), self.f32(name + "_scale_inv", scale_rows)

    def fp8_f32(self, name: str, rows: slice | None = None) -> np.ndarray:
        """A block-FP8 tensor dequantized to float32, for load-time folds."""
        w, s = self.fp8(name, rows)
        return block_f32(w, s)

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            self._maps.clear()
