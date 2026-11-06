"""Scan the E8M0 scales of the DeepSeek-V4.1 routed experts.

For each layer and each expert, the scan reports the spread of the scale exponents (largest minus
smallest) and the count of 0xFF bytes. The MXFP4 dequant kernel uses one exponent k per launch, and
its output is exact while every block is at most 13 exponents below that k's block. So a spread of
13 or less in a layer allows one k per layer; otherwise each expert needs its own k.

With --dense, it instead converts every dense FP8 weight (not the Engram tables) to E4M3 bias 7 with one
power-of-two scale per row, as the loader does, and reports the elements that the conversion rounds.

usage: ds41_scan.py CHECKPOINT [--layers 0-39] [--json OUT] [--dense]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

EXACT_SPREAD = 13
_NAME = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.scale$")


def headers(path: Path) -> dict[str, tuple[Path, int, int, str, list[int]]]:
    """Tensor name -> (file, offset, size, dtype, shape) for every shard."""
    out = {}
    for file in sorted(path.glob("*.safetensors")):
        with open(file, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        base = 8 + size
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            out[name] = (file, base + start, end - start, info["dtype"], info["shape"])
    return out


def parse_layers(text: str) -> set[int]:
    first, _, last = text.partition("-")
    return set(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", default="0-39")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    if args.dense:
        return dense(args)
    wanted = parse_layers(args.layers)
    tensors = headers(args.checkpoint)
    names = defaultdict(list)
    for name in tensors:
        match = _NAME.match(name)
        if match and (match[1] == "mtp" or int(match[2]) in wanted):
            names[(match[1], int(match[2]))].append((int(match[3]), match[4], name))
    report = {}
    worst_layer = 0
    for (kind, layer), items in sorted(names.items()):
        lo, hi = 255, 0
        experts = {}
        ff = 0
        fds: dict[Path, int] = {}
        for expert, _, name in sorted(items):
            file, offset, size, dtype, _ = tensors[name]
            if dtype != "F8_E8M0":
                raise SystemExit(f"{name}: expected F8_E8M0, got {dtype}")
            fd = fds.get(file)
            if fd is None:
                fd = fds[file] = os.open(file, os.O_RDONLY)
            scales = np.frombuffer(os.pread(fd, size, offset), np.uint8)
            ff += int(np.count_nonzero(scales == 0xFF))
            finite = scales[scales != 0xFF]
            e_lo, e_hi = int(finite.min()), int(finite.max())
            old = experts.get(expert, (255, 0))
            experts[expert] = (min(old[0], e_lo), max(old[1], e_hi))
            lo, hi = min(lo, e_lo), max(hi, e_hi)
        for fd in fds.values():
            os.close(fd)
        spreads = [h - low for low, h in experts.values()]
        layer_spread = hi - lo
        report[f"{kind}.{layer}"] = {
            "experts": len(experts),
            "min": lo,
            "max": hi,
            "layer_spread": layer_spread,
            "worst_expert_spread": max(spreads),
            "experts_over_exact": sum(s > EXACT_SPREAD for s in spreads),
            "ff_bytes": ff,
        }
        worst_layer = max(worst_layer, layer_spread)
        r = report[f"{kind}.{layer}"]
        print(
            f"{kind}.{layer:<3} experts {r['experts']:3d}  exponents {lo:3d}..{hi:3d}  "
            f"layer spread {layer_spread:2d}  worst expert {r['worst_expert_spread']:2d}  "
            f"experts over {EXACT_SPREAD}: {r['experts_over_exact']}  0xFF: {ff}",
            flush=True,
        )
    verdict = "one k per layer" if worst_layer <= EXACT_SPREAD else "one k per expert"
    print(f"worst layer spread {worst_layer}: {verdict} (exact up to {EXACT_SPREAD})")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


def dense(args) -> None:
    from resin_models_gaudi.deepseek_v41.weights import mx8_rows, reader

    src = reader(args.checkpoint)
    wanted = parse_layers(args.layers)
    total = rounded = 0
    worst = (0.0, "")
    report = {}
    for name in sorted(src.weight_map):
        if (
            not name.endswith(".weight")
            or ".experts." in name
            or "engram.embed" in name
        ):
            continue
        match = re.match(r"^layers\.(\d+)\.", name)
        if match and int(match[1]) not in wanted:
            continue
        dtype, _ = src.info(name)
        if dtype != "F8_E4M3":
            continue
        got = mx8_rows(
            src.read(name), src.read(name[: -len("weight")] + "scale"), tau=False
        )
        n = got.bytes.size
        total, rounded = total + n, rounded + got.inexact
        worst = max(worst, (got.inexact / n, name))
        report[name] = {"elements": n, "rounded": got.inexact}
        print(
            f"{name:52s} {n:12d} rounded {got.inexact:9d} ({got.inexact / n:.2e})",
            flush=True,
        )
    print(
        f"total {total} elements, {rounded} rounded ({rounded / max(total, 1):.2e}); worst {worst[1]} {worst[0]:.2e}"
    )
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
