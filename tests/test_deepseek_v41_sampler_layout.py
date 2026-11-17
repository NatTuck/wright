"""DeepSeek-V4.1's sampler output layout and parameter table hold the widest decode bucket, and the staged reader
decodes every field of that many rows from its own offsets (on CPU, no card)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("resin_models_gaudi.deepseek_v41.model")


def test_sampler_layout_holds_the_widest_decode_bucket() -> None:
    from resin_models_gaudi.common.sampling import PARAMS, TOP_MAX
    from resin_models_gaudi.deepseek_v41 import model as M

    R = M.MAX_T
    assert R == M.DECODE_BUCKETS[-1]
    sizes = {
        "tok_out": 4 * R,
        "lp_out": 4 * R,
        "top_idx": 4 * TOP_MAX * R,
        "top_lp": 4 * TOP_MAX * R,
    }
    spans = sorted((M._SAMPLED[n], M._SAMPLED[n] + size) for n, size in sizes.items())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans
    assert spans[-1][1] <= M._SAMPLED_BYTES
    assert all(off % 256 == 0 for off in M._SAMPLED.values())
    assert M._SAMPLE_LAST >= 4 * PARAMS * R
    assert M._SAMPLE_TABLE >= M._SAMPLE_LAST + 4


def test_sampled_fields_reads_every_row_from_the_layout() -> None:
    from glm_reference import _sampled_fields

    from resin_models_gaudi.common.sampling import TOP_MAX
    from resin_models_gaudi.deepseek_v41 import model as M

    R, L = M.MAX_T, M._SAMPLED
    out = np.zeros(M._SAMPLED_BYTES, np.uint8)
    tok = np.arange(R, dtype=np.int32) + 7
    lp = -np.arange(R, dtype=np.float32)
    ids = np.arange(R * TOP_MAX, dtype=np.int32).reshape(R, TOP_MAX)
    out[L["tok_out"] : L["tok_out"] + 4 * R] = tok.view(np.uint8)
    out[L["lp_out"] : L["lp_out"] + 4 * R] = lp.view(np.uint8)
    out[L["top_idx"] : L["top_idx"] + ids.nbytes] = ids.ravel().view(np.uint8)
    got = _sampled_fields(out, R, R - 3, L)
    np.testing.assert_array_equal(got[0], tok[: R - 3])
    np.testing.assert_array_equal(got[1], lp[: R - 3])
    np.testing.assert_array_equal(got[2], ids[: R - 3])
