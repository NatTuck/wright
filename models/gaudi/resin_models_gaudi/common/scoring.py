"""Prompt logprobs and output logprob columns of the Gaudi runners.

``score_requests`` reads a plan's prompt positions to score; ``ScoreRequest`` and ``Scores`` carry
them through a model's score launches; ``prompt_columns`` and ``score_columns`` give the
``ForwardOutput`` columns of ``resin.backend.sampling``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from resin_models_gaudi.common.sampling import TOP_MAX


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """Prompt positions of a prefill to score: ``positions`` (indices into its tokens, whose
    next-token distributions are scored), per position the token ids whose raw logprobs are
    wanted (``targets``) and the number of top logprobs (``tops``, at most SCORE_TOPS[-1])."""

    positions: np.ndarray
    targets: Sequence[np.ndarray]
    tops: np.ndarray


@dataclass(frozen=True, slots=True)
class Scores:
    """Per scored position: the targets' logprobs, and the top ids and logprobs."""

    targets: list[np.ndarray]
    top_ids: list[np.ndarray]
    top_logprobs: list[np.ndarray]


def score_requests(plan, vocab: int) -> list[ScoreRequest | None]:
    """Per row, the plan's prompt positions to score (indices into the plan's tokens)."""
    indices = np.asarray(plan.prompt_score_indices, np.int64)
    targets = np.asarray(plan.prompt_score_token_ids, np.int64)
    tops = np.minimum(np.asarray(plan.prompt_score_top_counts, np.int64), vocab)
    extra_ptr = np.asarray(plan.prompt_target_indptr, np.int64)
    extra = np.asarray(plan.prompt_target_token_ids, np.int64)
    indptr = np.asarray(plan.token_indptr, np.int64)
    requests = []
    for r in range(plan.num_sequences):
        inside = np.flatnonzero((indices >= indptr[r]) & (indices < indptr[r + 1]))
        if not len(inside):
            requests.append(None)
            continue
        requests.append(
            ScoreRequest(
                indices[inside] - indptr[r],
                [
                    np.concatenate(
                        ([targets[k]], extra[extra_ptr[k] : extra_ptr[k + 1]])
                    )
                    if len(extra_ptr) > 1
                    else np.array([targets[k]])
                    for k in inside
                ],
                tops[inside],
            )
        )
    return requests


def prompt_columns(plan, prompt_scores) -> dict[str, np.ndarray]:
    """``resin.backend.sampling.PromptScores.output``'s columns, in the plan's order."""
    targets, top_ids, top_lp = [], [], []
    for scored in prompt_scores:
        if scored is not None:
            targets += scored.targets
            top_ids += scored.top_ids
            top_lp += scored.top_logprobs

    def offsets(counts):
        return np.concatenate(([0], np.cumsum(counts))).astype(np.uint32)

    columns = {
        "prompt_logprobs": np.array([t[0] for t in targets], np.float32),
        "prompt_logprob_indptr": np.asarray(plan.prompt_score_indptr, np.uint32),
    }
    counts = np.array([len(t) for t in top_ids], np.int64)
    if counts.max(initial=0):
        columns["prompt_top_indptr"] = offsets(counts)
        columns["prompt_top_token_ids"] = np.concatenate(top_ids).astype(np.uint32)
        columns["prompt_top_logprobs"] = np.concatenate(top_lp).astype(np.float32)
    extra = [t[1:] for t in targets]
    if sum(len(t) for t in extra):
        columns["prompt_target_indptr"] = offsets([len(t) for t in extra])
        columns["prompt_target_token_ids"] = np.asarray(
            plan.prompt_target_token_ids, np.uint32
        )
        columns["prompt_target_logprobs"] = np.concatenate(extra).astype(np.float32)
    return columns


def score_columns(params, counts, lp, top_ids, top_lp) -> dict[str, np.ndarray]:
    """``resin.backend.sampling.format_output``'s logprob columns: every output token of a
    scored row (the params' logprob flag) gets its score, then its row's ntop top entries.
    ``counts``: output tokens per row; lp, top_ids, top_lp: per output token."""
    scored_rows = params[:, 6] > 0.5
    scored = np.repeat(scored_rows, counts)
    tops = np.repeat(np.where(scored_rows, params[:, 5], 0), counts).astype(np.uint32)
    columns = {
        "logprobs": np.ascontiguousarray(lp[scored], np.float32),
        "logprob_indptr": np.concatenate(([0], np.cumsum(counts * scored_rows))).astype(
            np.uint32
        ),
    }
    if tops.max(initial=0):
        keep = (np.arange(TOP_MAX)[None, :] < tops[:, None]) & scored[:, None]
        columns["top_indptr"] = np.concatenate(([0], np.cumsum(tops[scored]))).astype(
            np.uint32
        )
        columns["top_token_ids"] = np.ascontiguousarray(top_ids[keep], np.uint32)
        columns["top_logprobs"] = np.ascontiguousarray(top_lp[keep], np.float32)
    return columns
