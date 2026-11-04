"""Add the MTP weights of a dense Qwen3.5 checkpoint to its FP8 directory, as mtp.safetensors.

FP8 releases of these models leave out the MTP layer. This script reads only the `mtp.*` tensors of
the official BF16 checkpoint, over HTTP or from a local directory, and writes them in the layout that
the Gaudi loader takes: block FP8 for the projections, and BF16 for the norms and the input layer.

uv run --no-default-groups --group gaudi scripts/qwen35_mtp_fp8.py \
    --source Qwen/Qwen3.5-9B --revision c20223623576 --output /path/to/Qwen3.5-9B-FP8
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from resin_models_gaudi.qwen3_5.checkpoint import (
    BLOCK,
    FP8_MAX,
    bf16_to_f32,
    e4m3_encode,
)

# The projections that the loader takes as block FP8. The norms and mtp.fc stay in BF16.
_FP8 = ("_proj.weight",)

Reader = Callable[[str, int, int], bytes]


def http_reader(repo: str, revision: str) -> Reader:
    def read(name: str, start: int, end: int) -> bytes:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end - 1}"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()

    return read


def local_reader(root: Path) -> Reader:
    def read(name: str, start: int, end: int) -> bytes:
        with open(root / name, "rb") as stream:
            stream.seek(start)
            return stream.read(end - start)

    return read


def mtp_tensors(read: Reader, files: list[str]) -> dict[str, np.ndarray]:
    """The `mtp.*` tensors of the given safetensors files, as BF16 bits."""
    tensors = {}
    for name in files:
        (size,) = struct.unpack("<Q", read(name, 0, 8))
        header = json.loads(read(name, 8, 8 + size))
        for tensor, entry in header.items():
            if not tensor.startswith("mtp."):
                continue
            if entry["dtype"] != "BF16":
                raise ValueError(f"{tensor}: expected bf16, got {entry['dtype']}")
            start, end = entry["data_offsets"]
            data = read(name, 8 + size + start, 8 + size + end)
            tensors[tensor] = np.frombuffer(data, np.uint16).reshape(entry["shape"])
    if not tensors:
        raise ValueError("the source has no mtp.* tensors")
    return tensors


def quantize(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(FP8 bytes, per-block scale) so that value = byte value * scale, block by block."""
    weight = bf16_to_f32(bits)
    rows, columns = weight.shape
    if rows % BLOCK or columns % BLOCK:
        raise ValueError(f"shape {weight.shape} is not a multiple of {BLOCK}")
    scales = np.zeros((rows // BLOCK, columns // BLOCK), np.float32)
    out = np.empty(weight.shape, np.uint8)
    for r in range(scales.shape[0]):
        for c in range(scales.shape[1]):
            block = (r * BLOCK, (r + 1) * BLOCK), (c * BLOCK, (c + 1) * BLOCK)
            part = weight[slice(*block[0]), slice(*block[1])]
            peak = float(np.abs(part).max())
            scale = peak / FP8_MAX if peak else 1.0
            scales[r, c] = scale
            out[slice(*block[0]), slice(*block[1])] = e4m3_encode(part / scale)
    return out, scales


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, offset = {}, 0
    for name, (dtype, array) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    encoded = json.dumps(header).encode()
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        for _, array in tensors.values():
            stream.write(array.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="repository or local directory of the BF16 model",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output", type=Path, required=True, help="the FP8 checkpoint directory"
    )
    args = parser.parse_args()
    local = Path(args.source)
    read = (
        local_reader(local)
        if local.is_dir()
        else http_reader(args.source, args.revision)
    )
    index = json.loads(read("model.safetensors.index.json", 0, 1 << 28))
    files = sorted(
        {file for name, file in index["weight_map"].items() if name.startswith("mtp.")}
    )
    out: dict[str, tuple[str, np.ndarray]] = {}
    for name, bits in sorted(mtp_tensors(read, files).items()):
        if name.endswith(_FP8):
            weight, scales = quantize(bits)
            out[name] = ("F8_E4M3", weight)
            out[name + "_scale_inv"] = ("F32", scales)
        else:
            out[name] = ("BF16", bits)
    write_safetensors(args.output / "mtp.safetensors", out)
    print(f"wrote {args.output / 'mtp.safetensors'}: {len(out)} tensors")


if __name__ == "__main__":
    main()
