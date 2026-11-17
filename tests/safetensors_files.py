"""Small safetensors checkpoints for tests, written without the safetensors package."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """One file of ``name: (safetensors dtype, array)``. BF16 arrays are their bits (uint16)."""
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
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def write_checkpoint(
    directory: Path,
    shards: dict[str, dict[str, tuple[str, np.ndarray]]],
    index: bool = True,
) -> None:
    """Shards by file name, with a ``model.safetensors.index.json`` that lists them."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, tensors in shards.items():
        write_safetensors(directory / name, tensors)
    if index:
        weight_map = {t: name for name, tensors in shards.items() for t in tensors}
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
