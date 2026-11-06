"""DeepSeek-V4.1 Engram hash ids against the checkpoint's own inference/engram.py, with DS41_CHECKPOINT set."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from resin.models.deepseek_v41.engram import is_prime, next_prime

CHECKPOINT = Path(os.environ.get("DS41_CHECKPOINT") or "/nonexistent")


def test_primes():
    assert [n for n in range(50) if is_prime(n)] == [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
    ]
    assert is_prime(16000057) == all(16000057 % p for p in range(2, 4001))
    assert next_prime(13, {17}) == 19


@pytest.mark.skipif(
    not (CHECKPOINT / "inference/engram.py").exists(),
    reason="DS41_CHECKPOINT is not set",
)
def test_hash_ids_match_the_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sympy")
    from tokenizers import Tokenizer

    from resin.models.deepseek_v41 import DeepSeekV41Config
    from resin.models.deepseek_v41.engram import EngramHash

    cfg = DeepSeekV41Config.from_hugging_face(
        json.loads((CHECKPOINT / "config.json").read_text())
    )
    backend = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    mine = EngramHash.build(cfg, backend)
    sys.path.insert(0, str(CHECKPOINT / "inference"))
    try:
        import engram as reference
    finally:
        sys.path.pop(0)

    class Tok:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    raw = json.loads((CHECKPOINT / "inference/config.json").read_text())
    args = types.SimpleNamespace(**raw, max_batch_size=1, max_seq_len=1024)
    state = reference.NgramHashState(
        args, reference.EngramLayout.from_args(args), Tok()
    )
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, 260)
    want = state(torch.tensor(tokens[None, :250]), 0)[0].numpy()
    history = mine.compress(tokens)
    np.testing.assert_array_equal(mine.ids(history, np.arange(250)), want)
    for p in range(250, 260):
        step = state(torch.tensor([[int(tokens[p])]]), p)[0, 0].numpy()
        np.testing.assert_array_equal(mine.ids(history, np.array([p]))[0], step)
