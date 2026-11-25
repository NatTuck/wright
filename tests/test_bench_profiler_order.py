"""The profiler must be armed before the rank's card is claimed.

``claim_card`` opens the device (``open_device(card.accel, wait=0.0)``) to test whether the card is
free. Synapse loads its profiler plugin only when ``HABANA_PROFILE`` is set at the first Synapse call,
and ``libSynapse.so``'s stub ``synSingleton::profilerStart`` answers ``status=18`` (synUnsupported)
once it is too late — which is what a trace bench reported for months. This pins the order.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import scripts.bench as bench


def test_rank_enables_the_profiler_before_it_claims_a_card(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(
        "resin_models_gaudi.synapse.trace.enable_profiler",
        lambda directory: calls.append("profiler"),
    )

    @contextmanager
    def claim_card(rank, **kwargs):
        calls.append("claim")  # claim_card opens the device
        yield SimpleNamespace(card=SimpleNamespace(accel=3), binding=None)

    monkeypatch.setattr("resin_models_gaudi.synapse.runtime.claim_card", claim_card)
    monkeypatch.setattr(bench, "_run_rank", lambda *args, **kwargs: calls.append("run"))

    bench._rank(
        0,
        1,
        str(tmp_path / "report.json"),
        "model",
        "checkpoint",
        [],
        None,
        {"trace_dir": str(tmp_path / "traces"), "numa": "off"},
    )

    assert calls == ["profiler", "claim", "run"]


def test_no_profiler_without_a_trace_directory(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(
        "resin_models_gaudi.synapse.trace.enable_profiler",
        lambda directory: calls.append("profiler"),
    )

    @contextmanager
    def claim_card(rank, **kwargs):
        yield SimpleNamespace(card=SimpleNamespace(accel=3), binding=None)

    monkeypatch.setattr("resin_models_gaudi.synapse.runtime.claim_card", claim_card)
    monkeypatch.setattr(bench, "_run_rank", lambda *args, **kwargs: calls.append("run"))

    bench._rank(
        0,
        1,
        str(tmp_path / "report.json"),
        "model",
        "checkpoint",
        [],
        None,
        {"trace_dir": None, "numa": "off"},
    )

    assert calls == ["run"]
