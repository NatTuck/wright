"""Device sampling with Resin's semantics (``resin.backend.sampling``).

A sampled row draws from its temperature-scaled distribution, after top-k (ties at the cutoff
kept) and top-p over the full distribution (ties at the crossing kept), by inverse CDF in
vocabulary order with one uniform per request step: Philox4x32-10 keyed by the seed, counter
(0, 0, offset), word 0, as (r + 0.5) * 2^-32. Uniforms are computed here on the host. Greedy rows
(temperature 0, unsampled rows, rows with NaN or +inf logits) take torch.argmax's token.
Logprobs are of the raw logits: the chosen token's and the top ``ntop`` (<= 64).

The device side is five passes (``kernels/samp_*.c``): a chunked scan keeping top candidates,
thresholds from the candidates, a full-vocabulary pass for rows whose candidates fall short,
kept mass per chunk, and the draw.
"""

from __future__ import annotations

import numpy as np

from resin_models_gaudi.synapse.graph import Graph, Tensor
from resin_models_gaudi.synapse.runtime import F32, I32

CHUNKS = 24  # vocabulary chunks per row in the scan and mass passes
DEPTH = 8  # candidates per lane and chunk
TOP_MAX = 64  # longest top-logprob list
PARAMS = 8  # per-row f32 parameters: [T, top_p, top_k, uniform, mode, ntop, logprob, 0]

_MASK = np.uint64(0xFFFFFFFF)


def philox_uniform(seeds: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """f64 uniforms (r + 0.5) * 2^-32 of Philox4x32-10, key = seed, counter = (0, 0, offset)."""
    seeds = np.asarray(seeds, np.uint64)
    offsets = np.asarray(offsets, np.uint64)
    c0 = np.zeros_like(seeds)
    c1 = np.zeros_like(seeds)
    c2, c3 = offsets & _MASK, offsets >> np.uint64(32)
    k0, k1 = seeds & _MASK, seeds >> np.uint64(32)
    for _ in range(10):
        a = np.uint64(0xD2511F53) * c0
        b = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            (b >> np.uint64(32)) ^ c1 ^ k0,
            b & _MASK,
            (a >> np.uint64(32)) ^ c3 ^ k1,
            a & _MASK,
        )
        k0 = (k0 + np.uint64(0x9E3779B9)) & _MASK
        k1 = (k1 + np.uint64(0xBB67AE85)) & _MASK
    return (c0.astype(np.float64) + 0.5) * 2.0**-32


def row_params(
    rows: int,
    vocab: int,
    *,
    temperatures=None,
    top_ps=None,
    top_ks=None,
    seeds=None,
    offsets=None,
    sample=None,
    ntop=None,
    logprobs=None,
) -> np.ndarray:
    """(rows, PARAMS) f32 sampler inputs; rows not sampled (or at temperature 0) are greedy."""
    out = np.zeros((rows, PARAMS), np.float32)
    temperatures = (
        np.zeros(rows) if temperatures is None else np.asarray(temperatures, float)
    )
    sample = (temperatures > 0) if sample is None else np.asarray(sample, bool)
    out[:, 0] = temperatures
    out[:, 1] = 1.0 if top_ps is None else top_ps
    if top_ks is not None:
        k = np.minimum(np.asarray(top_ks, np.int64), vocab)
        out[:, 2] = np.where(k >= vocab, 0, k)
    mode = sample & (temperatures > 0) & np.isfinite(temperatures)
    if seeds is not None and mode.any():  # greedy rows never read their uniform
        out[mode, 3] = philox_uniform(
            np.asarray(seeds, np.uint64)[mode], np.asarray(offsets, np.uint64)[mode]
        ).astype(np.float32)
    out[:, 4] = mode
    if ntop is not None:
        out[:, 5] = np.minimum(ntop, TOP_MAX)
    if logprobs is not None:
        out[:, 6] = logprobs
    return out


OFFSET_STRIDE = 32  # resin.backend.sampling.SAMPLING_OFFSET_STRIDE


def plan_seeds(plan) -> tuple[np.ndarray, np.ndarray]:
    """Each row's effective seed and the Philox offset of its first sampled position
    ((context + query length) * 32, ``batch_sampling_inputs``'s)."""
    seeds = []
    for item in plan.sampling_params:
        if item.seed is None:
            raise ValueError("sampling params must contain effective seeds")
        seeds.append(item.seed)
    lengths = np.diff(np.asarray(plan.token_indptr, np.int64))
    offsets = (np.asarray(plan.context_lengths, np.int64) + lengths) * OFFSET_STRIDE
    return np.array(seeds, np.uint64), offsets


def plan_params(plan, vocab: int) -> np.ndarray:
    """(rows, PARAMS) sampler inputs of an Engine plan, with
    ``resin.backend.sampling.batch_sampling_inputs``'s semantics: each row's uniform comes from
    its effective seed at offset (context + query length) * 32; logprobs (the chosen token's and
    ``top_logprobs`` more) are scored for sampled rows that ask for them, as in
    ``format_output``."""
    params = tuple(plan.sampling_params)
    rows = len(params)
    mask = np.asarray(plan.sampling_mask, bool)
    seeds, offsets = plan_seeds(plan)
    scored = mask & np.array([bool(p.logprobs) for p in params], bool)
    return row_params(
        rows,
        vocab,
        temperatures=np.array([float(p.temperature) for p in params]),
        top_ps=np.array([float(p.top_p) for p in params]),
        top_ks=np.array([vocab if p.top_k is None else int(p.top_k) for p in params]),
        seeds=seeds,
        offsets=offsets.astype(np.uint64),
        sample=mask,
        ntop=np.where(scored, [min(int(p.top_logprobs), vocab) for p in params], 0),
        logprobs=scored,
    )


def statistics(
    graph: Graph, logits: Tensor, params: Tensor, sel_f: Tensor | None = None
):
    """The scan and select passes of ``logits`` (V, R): returns (wkey, wt, stats, sel_i, sel_f,
    top_idx, top_lp); ``sel_f`` (16, R) may be given (a persistent output)."""
    R = logits.shape[1]
    t = graph.tensor
    cand = (64, DEPTH, CHUNKS, R)
    ckey, cidx, cw = t(cand, I32), t(cand, I32), t(cand, F32)
    stats = t((64, CHUNKS, R), F32)
    padded = -(-logits.shape[0] // 64) * 64
    wkey, wt = t((padded, R), I32), t((padded, R), F32)
    graph.node("rg_samp_scan_f32", [logits, params], [ckey, cidx, cw, stats, wkey, wt])
    sel_i, sel_f = t((16, R), I32), sel_f if sel_f is not None else t((16, R), F32)
    top_idx, top_lp = t((TOP_MAX, R), I32), t((TOP_MAX, R), F32)
    graph.node(
        "rg_samp_select_f32",
        [params, ckey, cidx, cw, stats],
        [sel_i, sel_f, top_idx, top_lp],
    )
    return wkey, wt, stats, sel_i, sel_f, top_idx, top_lp


def sampler(graph: Graph, logits: Tensor, params: Tensor) -> None:
    """Add the sampler of ``logits`` (V, R) to ``graph``, with persistent outputs ``tok_out``
    (1, R) i32, ``lp_out`` (1, R) f32, ``top_idx`` (TOP_MAX, R) i32 and ``top_lp`` (TOP_MAX, R)
    f32; ``params`` is (PARAMS, R) f32 (``row_params``)."""
    R = logits.shape[1]
    t = graph.tensor
    wkey, wt, stats, sel_i, sel_f, top_idx, top_lp = statistics(graph, logits, params)
    fin_i = t((16, R), I32)
    graph.node(
        "rg_samp_full_f32",
        [wkey, wt, stats, sel_i, sel_f, top_idx, top_lp],
        [
            fin_i,
            graph.input("top_idx", (TOP_MAX, R), I32),
            graph.input("top_lp", (TOP_MAX, R), F32),
        ],
    )
    cmass = t((1, CHUNKS, R), F32)
    graph.node("rg_samp_mass_f32", [wkey, wt, stats, fin_i, sel_f], [cmass])
    graph.node(
        "rg_samp_draw_f32",
        [logits, wkey, wt, stats, fin_i, sel_f, cmass],
        [graph.input("tok_out", (1, R), I32), graph.input("lp_out", (1, R), F32)],
    )


__all__ = [
    "CHUNKS",
    "PARAMS",
    "TOP_MAX",
    "philox_uniform",
    "plan_params",
    "plan_seeds",
    "row_params",
    "sampler",
    "statistics",
]
