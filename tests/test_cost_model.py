# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real, no-mock unit tests for benchmarks/cost_model.py — pure arithmetic, no I/O."""

from __future__ import annotations

import pytest

from benchmarks.cost_model import (
    FRONTIER_PRICING,
    CostComparison,
    GPUCostProfile,
    compare_all_frontier_models,
)


def _example_gpu_profile(utilization: float = 1.0) -> GPUCostProfile:
    # 4x A100 80GB at a realistic RunPod-style rate, serving ~4000 tok/s
    # aggregate under vLLM continuous batching for a 32B coding model — a
    # defensible mid-range figure for real production throughput.
    return GPUCostProfile(
        gpu_count=4,
        hourly_cost_usd_per_gpu=1.89,
        measured_tokens_per_second=4000,
        utilization=utilization,
    )


def test_gpu_cost_profile_hourly_cost_scales_with_gpu_count():
    profile = _example_gpu_profile()
    assert profile.hourly_cost_usd == pytest.approx(4 * 1.89)


def test_gpu_cost_profile_cost_per_million_tokens_is_positive_and_finite():
    profile = _example_gpu_profile()
    assert 0 < profile.cost_per_million_tokens < 100  # sanity bound — self-hosted must be cheap per token


def test_lower_utilization_increases_effective_cost_per_token():
    full = _example_gpu_profile(utilization=1.0)
    half = _example_gpu_profile(utilization=0.5)
    assert half.cost_per_million_tokens == pytest.approx(2 * full.cost_per_million_tokens)


def test_zero_throughput_is_infinite_cost_not_a_crash():
    profile = GPUCostProfile(gpu_count=1, hourly_cost_usd_per_gpu=2.0, measured_tokens_per_second=0)
    assert profile.cost_per_million_tokens == float("inf")


def test_cost_comparison_self_hosted_beats_premium_frontier_models():
    # Against the expensive, high-capability frontier models, self-hosting a
    # well-utilized 4-GPU deployment is a clear win — that's the realistic
    # business case for self-hosting.
    profile = _example_gpu_profile()
    for key in ("claude-opus", "claude-sonnet", "gpt-4o"):
        comparison = CostComparison(
            task_prompt_tokens=5000,
            task_completion_tokens=2000,
            self_hosted=profile,
            frontier_key=key,
        )
        assert comparison.self_hosted_cost_usd < comparison.frontier_cost_usd, key
        assert comparison.savings_multiple > 1.0, key


def test_cost_comparison_is_honest_about_ultra_cheap_commodity_models():
    # gpt-4o-mini is priced low enough ($0.15/$0.60 per 1M) that a modest
    # self-hosted deployment does NOT automatically beat it — the cost model
    # must report that honestly rather than being tuned to always favor
    # self-hosting. This is real, useful signal for a client's actual
    # buy-vs-build decision, not a bug to "fix" by hiding it.
    profile = _example_gpu_profile()
    comparison = CostComparison(
        task_prompt_tokens=5000,
        task_completion_tokens=2000,
        self_hosted=profile,
        frontier_key="gpt-4o-mini",
    )
    # Assert the comparison computes a real, finite ratio in either direction —
    # not that self-hosting wins, which would be a dishonest guarantee.
    assert comparison.savings_multiple > 0
    assert comparison.self_hosted_cost_usd > 0


def test_cost_comparison_unknown_frontier_model_raises():
    profile = _example_gpu_profile()
    comparison = CostComparison(1000, 500, profile, "not-a-real-model")
    with pytest.raises(KeyError, match="Unknown frontier model"):
        _ = comparison.frontier_cost_usd


def test_compare_all_frontier_models_covers_every_known_model():
    profile = _example_gpu_profile()
    results = compare_all_frontier_models(5000, 2000, profile)
    assert len(results) == len(FRONTIER_PRICING)
    assert {r["frontier_model"] for r in results} == {p.model for p in FRONTIER_PRICING.values()}


def test_to_dict_is_json_serializable_and_rounds_cleanly():
    import json

    profile = _example_gpu_profile()
    comparison = CostComparison(5000, 2000, profile, "gpt-4o")
    payload = comparison.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["savings_multiple"] > 0
