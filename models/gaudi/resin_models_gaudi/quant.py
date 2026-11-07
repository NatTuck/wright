"""Read the block-FP8 weight scheme out of a checkpoint's ``quantization_config``.

Two spellings of the same scheme are in the wild:

* the official Qwen release, ``quant_method: "fp8"`` with ``weight_block_size``;
* a ``compressed-tensors`` (llm-compressor) export, ``quant_method:
  "compressed-tensors"`` with a config group whose ``weights.strategy`` is
  ``block`` and whose ``block_structure`` is the block size.

Both hold 128x128 block FP8 weights with dynamic activation scales, so the
runners read them the same way. The one difference in the payload is the name of
the block-scale tensor: ``_scale_inv`` in the first, ``_scale`` in the second.
"""

from __future__ import annotations

BLOCK = 128


def weight_block(quantization: object) -> list[int] | None:
    """The FP8 weight block size of a ``quantization_config``, or None if it is not block FP8."""
    if not isinstance(quantization, dict):
        return None
    method = quantization.get("quant_method")
    if method == "fp8":
        # The key is optional in the official format: absent means one 128x128 block size.
        block = quantization.get("weight_block_size") or [BLOCK, BLOCK]
        return list(block)
    if method == "compressed-tensors":
        groups = quantization.get("config_groups")
        for group in groups.values() if isinstance(groups, dict) else ():
            weights = (group or {}).get("weights") if isinstance(group, dict) else None
            if not isinstance(weights, dict):
                continue
            if weights.get("strategy") != "block" or weights.get("num_bits") != 8:
                continue
            block = weights.get("block_structure")
            if block:
                return list(block)
    return None


def scale_suffix(files: object, name: str) -> str:
    """The block-scale tensor of ``name``, under either spelling of the scheme."""
    inverse = name + "_scale_inv"
    return inverse if inverse in files else name + "_scale"  # type: ignore[operator]
