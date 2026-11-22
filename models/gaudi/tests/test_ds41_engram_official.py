"""The host-Engram path against INDEPENDENT oracles: the official 1Cat Engram hash, and torch's e4m3fn.

The production dequant's own test compares against ``ref_lut`` in the same file, and the hash has no
oracle in this repo at all -- so a wrong table row or a wrong id would have shown up only as "the model
repeats one token", with every unit test green. These two gates use implementations that share no code
with the production path:

- the layout (primes, offsets, multipliers) and the ids themselves against
  ``vllm_gaudi.ops.deepseek_v41_engram.EngramHashLayout`` (the official port), on a real token sequence;
- the dequant LUT (256 x 256 (scale, code) pairs) against ``torch.float8_e4m3fn`` and a from-the-spec
  formula.

Skips when the 1Cat tree, the checkpoint or torch's float8 dtype is absent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = Path(
    os.environ.get(
        "DS41_CHECKPOINT",
        "/home/nat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
        "/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170",
    )
)
ONECAT = Path("/home/nat/Envs/onecat/1Cat-vLLM-Gaudi")

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists(), reason="DS41_CHECKPOINT is not set"
)


def config():
    from resin.models.deepseek_v41 import DeepSeekV41Config

    raw = json.loads((CHECKPOINT / "config.json").read_text())
    return raw, DeepSeekV41Config.from_hugging_face(raw)


def test_engram_lut_matches_torch_float8_e4m3fn():
    """Every (scale, code) pair: the production LUT against torch's e4m3fn times 2^(scale-127)."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch has no float8_e4m3fn")
    import resin._resin as native

    lut = np.asarray(native.Ds41Fill._engram_lut(), np.float32)
    codes = np.arange(256, dtype=np.uint8)
    scales = np.arange(256, dtype=np.uint8)
    value = (
        torch.frombuffer(bytearray(codes), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float64)
        .numpy()
    )
    exp = np.exp2(scales.astype(np.int64) - 127).astype(np.float64)
    want = (value[None, :] * exp[:, None]).astype(np.float32)
    # two documented deviations, asserted rather than skipped:
    #   code 0x7F / 0xFF decode to NaN in torch and to 0 in the host oracle (the oracle's choice, kept);
    #   scale 254 is numpy's one-ulp exp2 rounding, scale 255 is E8M0 NaN (rejected at load).
    nan_codes = np.array([0x7F, 0xFF])
    keep = np.ones(256, bool)
    keep[nan_codes] = False
    got, ref = lut[:254][:, keep], want[:254][:, keep]
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} of {got.size} (scale, code) pairs differ"
    )
    assert np.all(lut[:254][:, nan_codes] == 0.0), "0x7F / 0xFF must read 0, as the host oracle does"
    assert np.all(np.isnan(want[:254][:, nan_codes])), "torch reads them as NaN"


def test_engram_ids_match_the_official_hash():
    """Layout and ids against the official 1Cat Engram hash, on a real token sequence."""
    if not (ONECAT / "vllm_gaudi/ops/deepseek_v41_engram.py").exists():
        pytest.skip("the official 1Cat Engram implementation is not on this box")
    sys.path.insert(0, str(ONECAT))
    try:
        from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout
    except Exception as error:  # its own imports are heavy
        pytest.skip(f"the official Engram module does not import here: {error}")
    from tokenizers import Tokenizer

    from resin.models.deepseek_v41.engram import EngramHash

    raw, cfg = config()
    text = raw.get("text_config", raw)
    official = EngramHashLayout.from_config(text)
    prod = EngramHash.build(cfg, Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json")))
    layers = len(cfg.engram_layer_ids)
    assert np.array_equal(prod.primes.reshape(layers, -1), official.primes), "primes"
    assert np.array_equal(prod.offsets, official.offsets), "offsets"
    assert np.array_equal(prod.multipliers, official.multipliers), "multipliers"

    tokens = np.array(
        Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
        .encode("The capital of France is Paris. My name is Ada and I like the sea.")
        .ids,
        np.int64,
    )
    compressed = prod.token_map[tokens]
    pos = np.arange(len(tokens)).astype(np.int64)
    got = prod.ids(compressed, pos)

    # the official id loop, verbatim from deepseek_v41_engram.py: rolling XOR of the n-gram's products,
    # then rolling % prime + offset per (gram size, head)
    n, heads = cfg.engram_max_ngram_size, cfg.engram_n_heads
    rolling = np.zeros((len(tokens), layers), np.int64)
    want = np.empty((len(tokens), layers, (n - 1) * heads), np.int32)
    blocked = np.zeros(len(tokens), bool)
    for shift in range(n):
        lookback = pos - shift
        source = compressed[np.maximum(lookback, 0)]
        blocked |= (lookback < 0) | (source == -1)
        values = np.where(blocked, prod.pad_id, source)
        rolling ^= values[:, None] * official.multipliers[None, :, shift]
        if shift:
            first, last = (shift - 1) * heads, shift * heads
            want[:, :, first:last] = (
                rolling[:, :, None] % official.primes[None, :, first:last]
                + official.offsets[None, :, first:last]
            )
    assert np.array_equal(got, want), f"{int((got != want).sum())} of {got.size} ids differ"
