"""The vision tower on the card, with the same `forward` contract as the CPU tower.

`VisionTower` (`vision.py`) stays the oracle; this is the path that serves. One Synapse
graph per image shape, one launch per image, and the checkpoint loaded and uploaded once.

Shape cost, which is the whole reason this file has a cache in it: the graph's tensors are
static, so a recipe is compiled per (height, width) patch grid. `RecipeCache` keeps compiled
recipes on disk keyed by graph name, so a given size is paid for once per host and then
loaded.

There is deliberately no "pad the grid up to the next bucket" scheme, for two independent
reasons, either of which is disqualifying:

1. The attention is bidirectional over every patch with no mask, so a padded patch is a real
   token that every real patch attends to. The real patches' scores change.
2. The position table is bilinearly interpolated to the *actual* grid, so padding the grid
   changes the interpolated positions of the real patches too.

So a new image size is a new compile, not an approximation. If the compile cost ever needs
bounding, the honest route is a masked or length-aware `rg_softmax_rows_f32`, after which
padding becomes safe. See docs/vision.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from resin_models_gaudi.qwen3_5.checkpoint import Safetensors
from resin_models_gaudi.qwen3_5.vision import (
    VisionConfig,
    _f32,
    interpolate_positions_np,
    merge_order,
    vision_rope,
)
from resin_models_gaudi.qwen3_5.vision_graph import HEAD_DIM, PADDED, vision_graph
from resin_models_gaudi.synapse.graph import RecipeCache
from resin_models_gaudi.synapse.runtime import SynapseError, open_device

VISION_PREFIX = "model.visual."


def host(array: np.ndarray) -> np.ndarray:
    """A host array in the layout the card wants: dim 0 contiguous, so (out, in) transposed."""
    array = np.asarray(array, np.float32)
    return np.ascontiguousarray(array.T if array.ndim == 2 else array)


def read(device, address: int, shape: tuple[int, ...]) -> np.ndarray:
    """A logical (d0, d1) device tensor as numpy in the same (d0, d1) order."""
    if len(shape) == 1:
        return device.d2h(address, shape, np.float32)
    return device.d2h(address, (shape[1], shape[0]), np.float32).T


def pad_head(rows: np.ndarray, padded: int = PADDED, head_dim: int = HEAD_DIM) -> np.ndarray:
    """Zero-pad a q or k slice into the head layout `rg_rope_vision_f32` wants.

    The real head_dim is split: its first half goes in lanes [0, head_dim//2) and its second
    half in lanes [64, 64 + head_dim - head_dim//2), so the rope's half-swap is a 64-lane
    aligned block swap. q and k share the permutation, so q.k is unchanged, and a zero lane
    pairs with a zero lane, so the padding stays zero through the rotation.
    """
    z = np.zeros((padded,) + rows.shape[1:], np.float32)
    half = head_dim // 2
    z[:half] = rows[:half]
    z[64 : 64 + head_dim - half] = rows[half:]
    return z


def vision_rope_tables(config: VisionConfig, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-token cos/sin for the vision rope, (padded, patches), in `pad_head`'s layout.

    Built from `vision.vision_rope`, the same function the CPU tower uses, so the two cannot
    drift apart. numpy, not torch: this is the serving path and torch is not a dependency
    of the gaudi package.
    """
    cos, sin = vision_rope(np, HEAD_DIM, height, width, config.merge)
    half = HEAD_DIM // 2
    out = []
    for values in (cos, sin):
        table = np.zeros((PADDED, values.shape[0]), np.float32)
        table[:half] = values[:, :half].T
        table[64 : 64 + HEAD_DIM - half] = values[:, half:].T
        out.append(table)
    return out[0], out[1]


def tower_weights(config: VisionConfig, path: str | Path) -> dict[str, np.ndarray]:
    """Every weight the graph wants, keyed by graph input name, in (out, in) order.

    Shape-independent: the same buffers serve every image size. The fused qkv weight and
    its bias are split per head on the host, and the attention scale is folded into q.
    """
    files = Safetensors(str(path))
    p = VISION_PREFIX
    out = {}

    out["patch_embed.proj.weight"] = _f32(files, p + "patch_embed.proj.weight").reshape(
        config.hidden, -1
    )
    for i in range(config.depth):
        b = f"{p}blocks.{i}."
        for n in ("norm1", "norm2"):
            out[f"blocks.{i}.{n}.weight"] = _f32(files, b + f"{n}.weight")
            out[f"blocks.{i}.{n}.bias"] = _f32(files, b + f"{n}.bias")
        qkv = _f32(files, b + "attn.qkv.weight")  # (3*hidden, hidden)
        qkvb = _f32(files, b + "attn.qkv.bias")  # (3*hidden,)
        proj = _f32(files, b + "attn.proj.weight")
        # divide by sqrt(head_dim) rather than multiply by its f32 reciprocal: one rounding
        # instead of two, and it tracks the oracle's own scale more closely
        scale = np.sqrt(HEAD_DIM)

        def pad(rows):
            """Zero-pad a (HEAD_DIM, hidden) slice to PADDED rows."""
            z = np.zeros((PADDED, rows.shape[1]), np.float32)
            z[:HEAD_DIM] = rows
            return z

        def pad1(vec):
            z = np.zeros(PADDED, np.float32)
            z[:HEAD_DIM] = vec
            return z

        for h in range(config.heads):
            lo = h * HEAD_DIM  # one head's slice: never let this leak between heads
            # the scale goes into q, which feeds only the scores
            out[f"blocks.{i}.attn.q{h}"] = (
                pad_head(qkv[lo : lo + HEAD_DIM]) / scale
            ).astype(np.float32)
            out[f"blocks.{i}.attn.k{h}"] = pad_head(
                qkv[config.hidden + lo : config.hidden + lo + HEAD_DIM]
            )
            out[f"blocks.{i}.attn.v{h}"] = pad(
                qkv[2 * config.hidden + lo : 2 * config.hidden + lo + HEAD_DIM]
            )
            out[f"blocks.{i}.attn.qb{h}"] = (
                pad_head(qkvb[lo : lo + HEAD_DIM]) / scale
            ).astype(np.float32)
            out[f"blocks.{i}.attn.kb{h}"] = pad_head(
                qkvb[config.hidden + lo : config.hidden + lo + HEAD_DIM]
            )
            out[f"blocks.{i}.attn.vb{h}"] = pad1(
                qkvb[2 * config.hidden + lo : 2 * config.hidden + lo + HEAD_DIM]
            )
            pz = np.zeros((config.hidden, PADDED), np.float32)
            pz[:, :HEAD_DIM] = proj[:, lo : lo + HEAD_DIM]
            out[f"blocks.{i}.attn.proj{h}"] = pz
        out[f"blocks.{i}.attn.proj.bias"] = _f32(files, b + "attn.proj.bias")
        out[f"blocks.{i}.mlp.fc1"] = _f32(files, b + "mlp.linear_fc1.weight")
        out[f"blocks.{i}.mlp.fc2"] = _f32(files, b + "mlp.linear_fc2.weight")
        out[f"blocks.{i}.mlp.b1"] = _f32(files, b + "mlp.linear_fc1.bias")
        out[f"blocks.{i}.mlp.b2"] = _f32(files, b + "mlp.linear_fc2.bias")

    out["merger.norm.weight"] = _f32(files, p + "merger.norm.weight")
    out["merger.norm.bias"] = _f32(files, p + "merger.norm.bias")
    out["merger.fc1"] = _f32(files, p + "merger.linear_fc1.weight")
    out["merger.fc2"] = _f32(files, p + "merger.linear_fc2.weight")
    out["merger.b1"] = _f32(files, p + "merger.linear_fc1.bias")
    out["merger.b2"] = _f32(files, p + "merger.linear_fc2.bias")
    return out


def position_bias(config: VisionConfig, path: str | Path, height: int, width: int) -> np.ndarray:
    """The embed bias plus the interpolated position table, (hidden, patches).

    This is the one graph input that depends on the image's patch grid. numpy, not
    torch: this is the serving path and torch is not a dependency of the gaudi package.
    """
    files = Safetensors(str(path))
    table = _f32(files, VISION_PREFIX + "pos_embed.weight")
    pos = interpolate_positions_np(table, config.grid, height, width, config.hidden)
    # patches are consumed in spatial-merge-block order, so the table has to be gathered into
    # that order exactly as `VisionTower._pos_embed` does. Without this the graph's position
    # input is in raster order while everything else is in block order - which is invisible to
    # the embed check (its reference used the same wrong array) and corrupts every stage after
    # it.
    order = merge_order(height, width, config.merge)
    pos = pos[order]
    bias = _f32(files, VISION_PREFIX + "patch_embed.proj.bias")[:, None]
    return (pos.T + bias).astype(np.float32)


class DeviceTower:
    """The tower on the card. Same `forward` contract as `VisionTower`.

    Weights are loaded and uploaded in `__init__`, so `forward` is one h2d of the patches,
    one launch and one d2h of the merged features.
    """

    def __init__(self, config: VisionConfig, path: str | Path, device=None) -> None:
        self.config = config
        self.path = str(path)
        if device is None:
            try:
                device = open_device(None)
            except SynapseError as error:
                raise RuntimeError(f"no card for the vision tower: {error}") from error
        self.device = device
        weights = tower_weights(config, self.path)
        self._buf = {name: device.malloc(a.nbytes) for name, a in weights.items()}
        for name, array in weights.items():
            device.h2d(self._buf[name], host(array))
        del weights  # the card has them; no reason to hold 1.8 GB of host copies
        self._recipes: dict[tuple[int, int], object] = {}
        self._scratch: dict[tuple[int, int], dict[str, int]] = {}
        self._cache = RecipeCache(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, config: dict, device=None) -> DeviceTower:
        """Load ``model.visual.*`` from a checkpoint directory, as `VisionTower` does."""
        vision = config.get("vision_config")
        if vision is None:
            raise ValueError(f"{path}: config.json has no vision_config")
        return cls(VisionConfig.from_hugging_face(vision), path, device=device)

    @property
    def cache(self) -> tuple[int, int]:
        """Recipe cache (hits, misses), so a caller can tell a cold compile from a load."""
        return self._cache.hits, self._cache.misses

    # ---- shapes
    def warm(self, height: int, width: int) -> None:
        """Compile (or load) the recipe for this patch grid, so `forward` does not."""
        self._prepare(height, width)

    def _prepare(self, height: int, width: int):
        key = (height, width)
        if key in self._recipes:
            return self._recipes[key], self._scratch[key]
        # one scratch set per image shape, held for the tower's lifetime: a shape seen
        # once is likely seen again, and the set is bounded by the distinct image sizes
        # actually served - the warmup at startup pins the common ones. `close` frees all
        # of it.
        c = self.config
        patches, merged = height * width, (height // c.merge) * (width // c.merge)
        lg = vision_graph(self.device, c, patches, merged, patches, width, "full")
        scratch = {
            "pix": self.device.malloc(c.patch_features * patches * 4),
            "pos_bias": self.device.malloc(c.hidden * patches * 4),
            # the vision rope's per-token angles, in the padded permuted head layout
            "rope_cos": self.device.malloc(PADDED * patches * 4),
            "rope_sin": self.device.malloc(PADDED * patches * 4),
            "out": self.device.malloc(c.out_hidden * merged * 4),
            "outb": self.device.malloc(c.out_hidden * merged * 4),
        }
        missing = set(lg._inputs) - set(self._buf) - set(scratch)
        if missing:
            raise RuntimeError(f"the graph wants tensors the tower does not fill: {sorted(missing)[:4]}")
        self.device.h2d(scratch["pos_bias"], host(position_bias(c, self.path, height, width)))
        cos, sin = vision_rope_tables(c, height, width)
        self.device.h2d(scratch["rope_cos"], host(cos))
        self.device.h2d(scratch["rope_sin"], host(sin))
        recipe = self._cache.compile(lg.g)
        self._recipes[key] = recipe
        self._scratch[key] = scratch
        return recipe, scratch

    # ---- the tower
    def forward(self, pixel_values, grid_thw) -> np.ndarray:
        """``pixel_values`` (patches, patch_features) and grid (t, h, w) -> (merged, out_hidden)."""
        grids = np.asarray(grid_thw)
        if grids.ndim == 1:
            grids = grids[None, :]
        if grids.shape[0] != 1:
            raise NotImplementedError(
                f"{grids.shape[0]} images in one prompt; the tower runs one image at a time"
            )
        temporal, height, width = (int(v) for v in grids[0])
        if temporal != 1:
            raise NotImplementedError(f"temporal {temporal}: still images only")
        pixels = np.ascontiguousarray(np.asarray(pixel_values, np.float32))
        c = self.config
        patches = pixels.shape[0]
        if height * width != patches:
            raise ValueError(f"grid {height}x{width} does not match {patches} patches")
        if height % c.merge or width % c.merge:
            raise ValueError(f"patch grid {height}x{width} does not divide by merge {c.merge}")
        recipe, scratch = self._prepare(height, width)
        merged = patches // (c.merge**2)
        self.device.h2d(scratch["pix"], pixels)  # (patches, features) is already transposed
        addr = {**self._buf, **scratch}
        recipe.bind_all(addr)
        recipe.launch()
        self.device.sync()
        return read(self.device, scratch["outb"], (c.out_hidden, merged)).T

    def close(self) -> None:
        for address in self._buf.values():
            self.device.free(address)
        for scratch in self._scratch.values():
            for address in scratch.values():
                self.device.free(address)
        self._buf.clear()
        self._scratch.clear()
        self._recipes.clear()
