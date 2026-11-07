"""Qwen3.8-27B vision tower, on the CPU, as the oracle.

The serving path is `DeviceTower` (`vision_device.py`), which runs this same
tower as one Synapse graph on the card; this module is the reference that path
is diffed against, validated against HF's `Qwen3VLVisionModel` stage by stage
(`docs/vision.md`, and `tools/vision_reference.py` re-runs the comparison). It
runs on torch CPU and imports torch lazily, so nothing that never sees an image
pays for it.

The device tower needs the same arithmetic without torch (it is the serving
path, and torch is not a dependency of the gaudi package), so `vision_rope`
takes the array module to compute with and `interpolate_positions_np` is the
numpy twin of `interpolate_positions`; `tools/vision_reference.py` checks both
against HF.

Layout notes, from the checkpoint's ``vision_config``:

* ``patch_embed.proj`` is a 5-D weight, ``[hidden, in, temporal, patch, patch]``:
  a Conv3d whose temporal kernel is ``temporal_patch_size``. The recipe repeats a
  still frame along that axis, so an image contributes one temporal position.
* ``pos_embed`` is ``[num_position_embeddings, hidden]``, a square grid
  (2304 = 48x48), interpolated bilinearly to the image's patch grid.
* ``deepstack_visual_indexes`` is empty in this checkpoint, so there are no
  per-layer feature taps: the tower's output is the merger's output alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resin_models_gaudi.qwen3_5.checkpoint import Safetensors, bf16_to_f32

VISION_PREFIX = "model.visual."


@dataclass(frozen=True, slots=True)
class VisionConfig:
    depth: int
    hidden: int
    heads: int
    intermediate: int
    patch: int
    temporal: int
    merge: int
    positions: int
    out_hidden: int
    in_channels: int = 3
    eps: float = 1e-6

    @classmethod
    def from_hugging_face(cls, vision: dict) -> VisionConfig:
        """Read ``config.json``'s ``vision_config`` block."""
        return cls(
            depth=int(vision["depth"]),
            hidden=int(vision["hidden_size"]),
            heads=int(vision["num_heads"]),
            intermediate=int(vision["intermediate_size"]),
            patch=int(vision["patch_size"]),
            temporal=int(vision["temporal_patch_size"]),
            merge=int(vision["spatial_merge_size"]),
            positions=int(vision["num_position_embeddings"]),
            out_hidden=int(vision["out_hidden_size"]),
            in_channels=int(vision.get("in_channels", 3)),
            eps=float(vision.get("eps", 1e-6)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_features(self) -> int:
        """The flattened patch vector: channels x temporal x patch x patch."""
        return self.in_channels * self.temporal * self.patch * self.patch

    @property
    def grid(self) -> int:
        """Side of the square position-embedding grid (48 for 2304)."""
        side = int(round(self.positions**0.5))
        if side * side != self.positions:
            raise ValueError(f"{self.positions} position embeddings are not a square grid")
        return side


def _f32(safetensors: Safetensors, name: str) -> np.ndarray:
    array, dtype = safetensors.raw(name)
    if dtype == "BF16":
        return bf16_to_f32(array)
    if dtype == "F32":
        return np.array(array)
    raise ValueError(f"{name}: expected bf16 or f32, got {dtype}")


def interpolate_positions(torch, table, grid: int, height: int, width: int, hidden: int):
    """Bilinear interpolation of the square position grid to (height, width).

    Torch-only: this is the oracle's arithmetic, validated against HF, and it
    stays untouched. The device tower uses `interpolate_positions_np`.
    """
    table = table.reshape(grid, grid, hidden)
    if height == grid and width == grid:
        return table.reshape(height * width, hidden)
    table = table.permute(2, 0, 1)[None]  # (1, hidden, grid, grid)
    # align_corners=True: the reference tower interpolates the position table with the
    # corners mapped onto the corners. False (the obvious default) shifts every patch's
    # position embedding and halves the table's spread; it was the second bug found by
    # diffing against the reference.
    resized = torch.nn.functional.interpolate(
        table, size=(height, width), mode="bilinear", align_corners=True
    )
    return resized[0].permute(1, 2, 0).reshape(height * width, hidden)


def interpolate_positions_np(table: np.ndarray, grid: int, height: int, width: int, hidden: int):
    """`interpolate_positions` for numpy: the device tower's position bias needs no torch.

    Same align_corners=True bilinear, same pass order (lerp along width, then
    height), computed in float32 like the torch kernel. `tools/vision_reference.py`
    diffs this against both the torch version and HF.
    """
    table = np.asarray(table, np.float32).reshape(grid, grid, hidden)
    if height == grid and width == grid:
        return table.reshape(height * width, hidden)

    def weights(out: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # align_corners=True: output i samples src = i * (size-1) / (out-1)
        if out == 1:
            src = np.zeros(1, np.float32)
        else:
            src = np.arange(out, dtype=np.float32) * ((size - 1) / (out - 1))
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, size - 1)
        frac = (src - lo).astype(np.float32)
        return lo, hi, frac

    y0, y1, wy = weights(height, grid)
    x0, x1, wx = weights(width, grid)
    # lerp along width on both source rows, then along height: F.interpolate's order
    top = table[np.ix_(y0, x0)] * (1 - wx)[:, None] + table[np.ix_(y0, x1)] * wx[:, None]
    bottom = table[np.ix_(y1, x0)] * (1 - wx)[:, None] + table[np.ix_(y1, x1)] * wx[:, None]
    out = top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]  # (height, width, hidden)
    return out.reshape(height * width, hidden)


def merge_order(height: int, width: int, merge: int) -> np.ndarray:
    """Raster index of each patch, in the order the model consumes patches.

    The image processor emits patches in spatial-merge-block order: a merge x merge block's
    patches are consecutive, and blocks run across then down. So patch 1 of a 2x2-merge grid
    is the patch at (row 0, col 2), not (row 0, col 1). The tower assumed raster order, which
    gave every patch a neighbour's position embedding and merged the wrong patches together -
    the image survived but its layout did not.
    """
    block_row, block_col = np.meshgrid(
        np.arange(height // merge), np.arange(width // merge), indexing="ij"
    )
    in_row, in_col = np.meshgrid(np.arange(merge), np.arange(merge), indexing="ij")
    row = block_row[:, :, None, None] * merge + in_row[None, None]
    col = block_col[:, :, None, None] * merge + in_col[None, None]
    return (row * width + col).reshape(-1)


def vision_rope(module, head_dim: int, height: int, width: int, merge: int = 1,
                theta: float = 10_000.0):
    """Axial 2D rotary angles for the vision attention: (patches, head_dim) cos and sin.

    Frequencies run over half the head dim (``head_dim // 4`` of them), the h angles and the
    w angles are concatenated, and that pair is duplicated so the rotation covers the whole
    head dim. HF calls this rope "axial"; the tower was written without it, which left the
    attention position-agnostic and every patch's features meaningless in space.

    ``module`` is ``torch`` (the oracle) or ``np`` (the device tower, which must not
    depend on torch); both produce float32 arrays with the same values.
    """
    spatial = head_dim // 2
    inv_freq = module.arange(0, spatial, 2, dtype=module.float32)
    inv_freq = 1.0 / (theta ** (inv_freq / spatial))
    rows, cols = np.divmod(merge_order(height, width, merge), width)
    h = module.asarray(rows, dtype=module.float32)
    w = module.asarray(cols, dtype=module.float32)
    angles = module.concatenate(
        [h[:, None] * inv_freq, w[:, None] * inv_freq], axis=-1
    )
    angles = module.concatenate([angles, angles], axis=-1)  # (patches, head_dim)
    return module.cos(angles), module.sin(angles)


class VisionTower:
    """The ViT that turns patches into prompt embeddings. Torch, CPU, no gradients."""

    def __init__(self, config: VisionConfig, tensors: dict[str, np.ndarray]) -> None:
        import torch

        self.config = config
        self.torch = torch
        self.params = {
            name: torch.from_numpy(np.ascontiguousarray(value, np.float32))
            for name, value in tensors.items()
        }
        missing = self._missing()
        if missing:
            raise ValueError(f"vision tower is missing tensors: {missing}")

    # ---- construction
    def _missing(self) -> list[str]:
        c = self.config
        need = [
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "pos_embed.weight",
            "merger.norm.weight",
            "merger.norm.bias",
            "merger.linear_fc1.weight",
            "merger.linear_fc1.bias",
            "merger.linear_fc2.weight",
            "merger.linear_fc2.bias",
        ]
        for i in range(c.depth):
            for stem in ("norm1", "norm2"):
                need += [f"blocks.{i}.{stem}.weight", f"blocks.{i}.{stem}.bias"]
            need += [
                f"blocks.{i}.attn.qkv.weight",
                f"blocks.{i}.attn.qkv.bias",
                f"blocks.{i}.attn.proj.weight",
                f"blocks.{i}.attn.proj.bias",
                f"blocks.{i}.mlp.linear_fc1.weight",
                f"blocks.{i}.mlp.linear_fc1.bias",
                f"blocks.{i}.mlp.linear_fc2.weight",
                f"blocks.{i}.mlp.linear_fc2.bias",
            ]
        return [name for name in need if name not in self.params]

    @classmethod
    def from_checkpoint(cls, path: str | Path, config: dict) -> VisionTower:
        """Load ``model.visual.*`` from a checkpoint directory."""
        files = Safetensors(path)
        names = [n for n in files.meta if n.startswith(VISION_PREFIX)]
        if not names:
            raise ValueError(f"{path} has no {VISION_PREFIX} tensors")
        tensors = {n[len(VISION_PREFIX) :]: _f32(files, n) for n in names}
        vision = config.get("vision_config")
        if vision is None:
            raise ValueError(f"{path}: config.json has no vision_config")
        return cls(VisionConfig.from_hugging_face(vision), tensors)

    # ---- pieces
    def _linear(self, x, name: str):
        w = self.params[name + ".weight"]
        b = self.params[name + ".bias"]
        if w.dim() > 2:
            # patch_embed is a Conv3d weight, [out, in, temporal, patch, patch]; the
            # recipe hands us the same flattening as a row vector, so a matmul is exact.
            w = w.reshape(w.shape[0], -1)
        return x @ w.t() + b

    def _layer_norm(self, x, name: str):
        torch = self.torch
        w = self.params[name + ".weight"]
        b = self.params[name + ".bias"]
        return torch.nn.functional.layer_norm(x, (self.config.hidden,), w, b, self.config.eps)

    def _pos_embed(self, height: int, width: int):
        """Bilinear interpolation of the square position grid, in patch-consumption order."""
        torch = self.torch
        c = self.config
        table = interpolate_positions(
            torch, self.params["pos_embed.weight"], c.grid, height, width, c.hidden
        )
        order = torch.from_numpy(merge_order(height, width, c.merge))
        return table[order]

    def _attention(self, x, index: int, rope):
        torch = self.torch
        c = self.config
        tokens = x.shape[0]
        qkv = self._linear(x, f"blocks.{index}.attn.qkv").reshape(
            tokens, 3, c.heads, c.head_dim
        )
        # (3, tokens, heads, head_dim) -> per head (heads, tokens, head_dim), which is the
        # layout scaled_dot_product_attention wants: it takes the last two dims as
        # (sequence, head_dim), so attention has to run over tokens, not over heads.
        q, k, v = qkv.permute(1, 2, 0, 3)
        # axial 2D rope on q and k, per patch: cos/sin are (tokens, head_dim), broadcast
        # over the head axis
        cos, sin = (part.unsqueeze(0) for part in rope)

        def rotate(t):
            half = t.shape[-1] // 2
            return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        # The tower is bidirectional: every patch sees every other patch of its image.
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # (heads, tokens, head_dim) -> (tokens, hidden): heads concatenate in order.
        out = out.transpose(0, 1).reshape(tokens, c.hidden)
        return self._linear(out, f"blocks.{index}.attn.proj")

    def stages(self, pixel_values: np.ndarray, grid_thw) -> dict[str, np.ndarray]:
        """The tower's own intermediates, for the harness to diff the card against.

        The harness used to rebuild expectations by hand, and one of them (the merger) kept a
        raster-order assumption after this tower moved to merge-block order - a second
        reference that went stale and produced a number nobody could trust. These taps make
        the tower the only reference: everything the device is compared to comes from the
        code that was validated against HF at 6.8e-07 / 2.0e-05 / 1.5e-04.

        Shapes follow the tower's convention, (patches, features); `merge` is the merged
        (4*hidden, merged) tensor before the merger's two projections.
        """
        torch = self.torch
        c = self.config
        temporal, height, width = (int(v) for v in grid_thw[0])
        if temporal * height * width != pixel_values.shape[0]:
            raise ValueError(
                f"grid {temporal}x{height}x{width} does not match {pixel_values.shape[0]} patches"
            )
        out = {}
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(pixel_values, np.float32))
            x = self._linear(x, "patch_embed.proj") + self._pos_embed(height, width)
            out["embed"] = x.numpy()
            rope = vision_rope(torch, c.head_dim, height, width, c.merge)
            for index in range(c.depth):
                x = self._block(x, index, rope)
                if index == 0:
                    out["block0"] = x.numpy()
            out["blocks"] = x.numpy()
            m = c.merge
            normed = self._layer_norm(x, "merger.norm")
            out["merge"] = normed.reshape(-1, m * m * c.hidden).numpy().T
            out["merged"] = self._merger(x, height, width).numpy()
        return out

    def _block(self, x, index: int, rope):
        torch = self.torch
        x = x + self._attention(self._layer_norm(x, f"blocks.{index}.norm1"), index, rope)
        h = self._layer_norm(x, f"blocks.{index}.norm2")
        h = self._linear(h, f"blocks.{index}.mlp.linear_fc1")
        h = torch.nn.functional.gelu(h, approximate="tanh")
        return x + self._linear(h, f"blocks.{index}.mlp.linear_fc2")

    def _merger(self, x, height: int, width: int):
        """Norm, then merge each spatial_merge x spatial_merge patch block into one row.

        A plain reshape, because the patches already arrive in merge-block order - the
        permute that used to be here was compensating for the raster assumption.
        """
        torch = self.torch
        c = self.config
        m = c.merge
        x = self._layer_norm(x, "merger.norm")
        x = x.reshape(-1, m * m * c.hidden)
        x = self._linear(x, "merger.linear_fc1")
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return self._linear(x, "merger.linear_fc2")

    # ---- forward
    def forward(self, pixel_values: np.ndarray, grid_thw) -> np.ndarray:
        """One image: ``pixel_values`` (patches, patch_features) and grid (t, h, w).

        Returns the merged features, (patches / merge**2, out_hidden), as float32.
        Rows are in patch-grid order, which is the order the recipe's image tokens
        expect.
        """
        torch = self.torch
        c = self.config
        temporal, height, width = (int(v) for v in grid_thw[0])
        if temporal * height * width != pixel_values.shape[0]:
            raise ValueError(
                f"grid {temporal}x{height}x{width} does not match "
                f"{pixel_values.shape[0]} patches"
            )
        if height % c.merge or width % c.merge:
            raise ValueError(f"patch grid {height}x{width} does not divide by merge {c.merge}")

        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(pixel_values, np.float32))
            x = self._linear(x, "patch_embed.proj")
            x = x + self._pos_embed(height, width)
            rope = vision_rope(torch, c.head_dim, height, width, c.merge)
            for index in range(c.depth):
                x = self._block(x, index, rope)
            x = self._merger(x, height, width)
        return x.numpy()
