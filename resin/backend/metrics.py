"""Cumulative execution observations, independent of metric exporters."""

from resin._resin import RunnerMetrics as RunnerMetrics

FORWARD_CATEGORIES = ("prefill", "target_verify", "draft", "decode", "idle")
