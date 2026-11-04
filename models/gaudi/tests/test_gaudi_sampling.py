"""The device sampler (``qwen3_5.sampling``) against independent NumPy oracles of Resin's
sampling semantics (ported from tests/test_sampling_fused_cuda.py and
tests/test_sampling_filters_cuda.py): Philox4x32-10 uniforms, torch.argmax greedy rows, top-k
and full-distribution top-p with ties kept, inverse-CDF draws in vocabulary order, raw logprobs.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)

QWEN_VOCAB = 248320


@pytest.fixture(scope="module")
def device():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    try:
        opened = open_device(None if card is None else int(card))
    except SynapseError as error:
        pytest.skip(f"no Gaudi card available: {error}")
    yield opened


@pytest.fixture(scope="module")
def run(device, tmp_path_factory):
    """run(logits (R, V) f32, params (R, 8) f32) -> (tokens, logprobs, top ids, top logprobs)"""
    from resin_models_gaudi.qwen3_5.sampling import PARAMS, TOP_MAX, sampler
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32

    cache = RecipeCache(device, tmp_path_factory.mktemp("recipes"))
    recipes = {}

    def go(logits, params):
        R, V = logits.shape
        if (R, V) not in recipes:
            graph = Graph(device, f"test_sampler_R{R}_V{V}")
            sampler(
                graph,
                graph.input("logits", (V, R), F32),
                graph.input("params", (PARAMS, R), F32),
            )
            recipes[R, V] = cache.compile(graph)
        recipe = recipes[R, V]
        sizes = {
            "logits": logits.nbytes,
            "params": params.nbytes,
            "tok_out": R * 4,
            "lp_out": R * 4,
            "top_idx": TOP_MAX * R * 4,
            "top_lp": TOP_MAX * R * 4,
        }
        address = {name: device.malloc(max(256, n)) for name, n in sizes.items()}
        device.h2d(address["logits"], np.ascontiguousarray(logits, np.float32))
        device.h2d(address["params"], np.ascontiguousarray(params, np.float32))
        recipe.bind_all(address)
        recipe.launch()
        out = (
            device.d2h(address["tok_out"], (R,), np.int32),
            device.d2h(address["lp_out"], (R,), np.float32),
            device.d2h(address["top_idx"], (R, TOP_MAX), np.int32),
            device.d2h(address["top_lp"], (R, TOP_MAX), np.float32),
        )
        for a in address.values():
            device.free(a)
        return out

    return go


def philox_word(seed: int, offset: int) -> int:
    """Independent Random123 Philox4x32-10 word 0 (tests/test_sampling_fused_cuda.py)."""
    mask = 0xFFFFFFFF
    c0, c1, c2, c3 = 0, 0, offset & mask, (offset >> 32) & mask
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        a, b = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (b >> 32) ^ c1 ^ k0, b & mask, (a >> 32) ^ c3 ^ k1, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0


def reference(logits, temperatures, top_ks, top_ps, seeds, offsets, sample):
    """FP64 oracle (tests/test_sampling_filters_cuda.py): tokens and per-row draw details."""
    result = np.argmax(logits, axis=1)
    details = [None] * len(logits)
    for row, raw in enumerate(logits.astype(np.float64)):
        temp = float(temperatures[row])
        if (
            not sample[row]
            or not 0 < temp < np.inf
            or np.isnan(raw).any()
            or np.isposinf(raw).any()
            or not np.isfinite(raw).any()
        ):
            continue
        k = min(int(top_ks[row]), len(raw))
        keep = np.isfinite(raw) & (raw >= np.partition(raw, -k)[-k])
        probability = np.exp((raw - raw.max()) / temp)
        probability /= probability.sum()
        keep &= probability > 0
        p = float(top_ps[row])
        if p < 1:
            descending = np.sort(probability)[::-1]
            crossing = min(np.searchsorted(np.cumsum(descending), p), len(raw) - 1)
            keep &= probability >= descending[crossing]
        uniform = (philox_word(int(seeds[row]), int(offsets[row])) + 0.5) * 2**-32
        cumulative = np.cumsum(np.where(keep, probability, 0.0))
        total = cumulative[-1]
        target = uniform * total
        selected = int(np.searchsorted(cumulative, target, side="right"))
        result[row] = selected
        before = cumulative[selected - 1] if selected else 0.0
        margin = min(target - before, cumulative[selected] - target) / total
        details[row] = (keep, cumulative, target, total, margin)
    return result, details


def check_draws(logits, tokens, expected, details):
    for row, (token, detail) in enumerate(zip(tokens.tolist(), details, strict=True)):
        assert 0 <= token < logits.shape[1], (row, token)
        if detail is None:
            assert token == int(expected[row]), (row, token, int(expected[row]))
            continue
        keep, cumulative, target, total, margin = detail
        assert keep[token], (row, token)
        if margin > 2e-7:
            assert token == int(expected[row]), (row, token, int(expected[row]), margin)
        else:
            before = cumulative[token - 1] if token else 0.0
            assert before <= target + 2e-7 * total
            assert cumulative[token] > target - 2e-7 * total


def sample_case(rows, vocab, rng):
    row = np.arange(rows)
    return dict(
        temperatures=0.5 + (row % 4) * 0.5,
        top_ks=np.where(row % 3 == 0, vocab, 7 + row % 5),
        top_ps=np.where(row % 3 == 1, 1.0, 0.71),
        seeds=(1 << 62) + 17 + row * (1 << 32),
        offsets=(1 << 40) + 32 + row * 32,
        sample=row % 5 != 4,
    )


def draw(run, logits, case, ntop=None):
    from resin_models_gaudi.qwen3_5.sampling import row_params

    params = row_params(len(logits), logits.shape[1], ntop=ntop, **case)
    return run(logits, params)


def test_philox_matches_the_published_vector():
    from resin_models_gaudi.qwen3_5.sampling import philox_uniform

    assert philox_word(0, 0) == 0x6627E8D5
    u = philox_uniform(np.array([0, (1 << 62) + 5]), np.array([0, (1 << 40) + 64]))
    assert u[0] == (0x6627E8D5 + 0.5) * 2**-32
    assert u[1] == (philox_word((1 << 62) + 5, (1 << 40) + 64) + 0.5) * 2**-32


def test_greedy_is_torch_argmax(run):
    logits = np.random.default_rng(1).standard_normal((8, 2051)).astype(np.float32)
    logits[0] = -np.inf
    logits[1] = 1
    logits[2, [4, 2049]] = np.inf
    logits[3, [7, 1027]] = np.nan
    logits[4] = -5
    logits[4, [1023, 1024, 2048]] = 10
    logits[5] = 0
    logits[5, 0] = -0.0
    logits[6] = np.inf
    logits[7, -1] = np.nan
    tokens, *_ = draw(run, logits, {})
    np.testing.assert_array_equal(tokens, np.argmax(logits, axis=1))


@pytest.mark.parametrize("vocab", [257, 2051, QWEN_VOCAB])
def test_filtered_draws_match_the_oracle(run, vocab):
    rng = np.random.default_rng(vocab)
    logits = (rng.standard_normal((24, vocab)) * 2).astype(np.float32)
    logits[:, ::7] = -np.inf
    case = sample_case(24, vocab, rng)
    tokens, *_ = draw(run, logits, case)
    expected, details = reference(logits, **case)
    check_draws(logits, tokens, expected, details)


def test_nucleus_is_taken_from_the_full_distribution(run):
    logits = np.full((64, 2051), -np.inf, np.float32)
    logits[:, :3] = np.log([0.60, 0.25, 0.15])
    row = np.arange(64)
    case = dict(
        temperatures=np.ones(64),
        top_ks=np.full(64, 2),
        top_ps=np.full(64, 0.65),
        seeds=1000 + row,
        offsets=32 * row,
        sample=np.ones(64, bool),
    )
    tokens, *_ = draw(run, logits, case)
    check_draws(logits, tokens, *reference(logits, **case))
    assert sorted(set(tokens.tolist())) == [0, 1]


@pytest.mark.parametrize(("top_k", "top_p"), [(3, 1.0), (2051, 0.25), (3, 0.25)])
def test_ties_at_the_cutoff_are_kept_across_chunks(run, top_k, top_p):
    logits = np.zeros((64, 2051), np.float32)
    logits[:, [0, 32, 1023, 1024, 2048, 2050]] = 4
    row = np.arange(64)
    case = dict(
        temperatures=0.5 + (row % 4) * 0.5,
        top_ks=np.full(64, top_k),
        top_ps=np.full(64, top_p),
        seeds=7 + row,
        offsets=64 * row,
        sample=np.ones(64, bool),
    )
    tokens, *_ = draw(run, logits, case)
    check_draws(logits, tokens, *reference(logits, **case))
    assert (tokens > 1024).any()


def test_rows_beyond_the_candidates_take_the_full_pass(run):
    """Nuclei and top-k cutoffs larger than the candidate list, and a cutoff among a crowded group
    of tokens, match the oracle."""
    rng = np.random.default_rng(5)
    logits = np.full((4, QWEN_VOCAB), -10.0, np.float32)
    logits[:2, rng.choice(QWEN_VOCAB, 20000, replace=False)] = 1.0  # 20000 tied tokens
    logits[2] = rng.standard_normal(QWEN_VOCAB) * 0.05
    logits[3] = rng.standard_normal(QWEN_VOCAB)
    logits[3, np.arange(20) * 64 + 5] = (
        9.0 - np.arange(20) * 0.01
    )
    case = dict(
        temperatures=np.array([1.0, 1.0, 1.5, 1.0]),
        top_ks=np.array([QWEN_VOCAB, QWEN_VOCAB, 30000, 15]),
        top_ps=np.array([0.9, 0.5, 1.0, 1.0]),
        seeds=np.array([3, 4, 5, 6]),
        offsets=np.array([32, 64, 96, 128]),
        sample=np.ones(4, bool),
    )
    tokens, *_ = draw(run, logits, case)
    check_draws(logits, tokens, *reference(logits, **case))
    assert logits[0, tokens[0]] == 1.0 and logits[1, tokens[1]] == 1.0


@pytest.mark.parametrize("vocab", [2051, QWEN_VOCAB])
def test_logprobs_are_raw_log_softmax(run, vocab):
    rng = np.random.default_rng(11)
    logits = (rng.standard_normal((6, vocab)) * 3).astype(np.float32)
    logits[5, np.arange(30) * 64 + 9] = (
        20.0 - np.arange(30) * 0.5
    )
    case = sample_case(6, vocab, rng)
    tokens, lp, top_idx, top_lp = draw(run, logits, case, ntop=20)
    raw = logits.astype(np.float64)
    lsm = raw - raw.max(1, keepdims=True)
    lsm -= np.log(np.exp(lsm).sum(1, keepdims=True))
    np.testing.assert_allclose(lp, lsm[np.arange(6), tokens], atol=2e-4)
    order = np.lexsort((np.arange(vocab)[None].repeat(6, 0), -raw), axis=1)[:, :20]
    np.testing.assert_array_equal(top_idx[:, :20], order)
    np.testing.assert_allclose(
        top_lp[:, :20], np.take_along_axis(lsm, order, 1), atol=2e-4
    )


def test_score_gather_reads_each_pair(device, tmp_path):
    """Prompt scoring's (row, token) logprob gather: logits[token, row] - lse[row]."""
    from resin_models_gaudi.synapse.graph import Graph, RecipeCache
    from resin_models_gaudi.synapse.runtime import F32, I32

    V, R, N = 1000, 8, 70
    graph = Graph(device, "test_score_gather")
    graph.node(
        "rg_score_gather_f32",
        [
            graph.input("logits", (V, R), F32),
            graph.input("sel_f", (16, R), F32),
            graph.input("pairs", (2, N), I32),
        ],
        [graph.input("out", (N,), F32)],
    )
    recipe = RecipeCache(device, tmp_path).compile(graph)
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((R, V)).astype(np.float32)
    stats = np.zeros((R, 16), np.float32)
    stats[:, 1] = np.arange(R) * 10
    pairs = np.stack([rng.integers(0, R, N), rng.integers(0, V, N)], 1).astype(np.int32)
    pairs[5] = (-1, 0)  # padding
    host = {
        "logits": logits,
        "sel_f": stats,
        "pairs": pairs,
        "out": np.zeros(N, np.float32),
    }
    address = {name: device.malloc(max(256, a.nbytes)) for name, a in host.items()}
    for name, a in host.items():
        device.h2d(address[name], a)
    recipe.bind_all(address)
    recipe.launch()
    got = device.d2h(address["out"], (N,), np.float32)
    for a in address.values():
        device.free(a)
    rows = pairs[:, 0].clip(0)
    expected = np.where(pairs[:, 0] >= 0, logits[rows, pairs[:, 1]] - stats[rows, 1], 0)
    np.testing.assert_array_equal(got, expected)
