# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — Token cost comparison model.

Compares Keystone Inference's self-hosted cost-per-token against published
frontier-model API pricing, so a benchmark run reports not just "how good"
but "how good per dollar" — the actual business question a client cares
about when deciding whether self-hosting is worth it.

Self-hosted cost is derived from real infrastructure economics (GPU-hour
price / measured throughput), not assumed to be free — idle GPU capacity,
utilization, and amortized cost all matter for an honest comparison.

Frontier prices are USD per 1M tokens, entered manually and dated — these
change; there is no live pricing API, so `PRICE_AS_OF` exists specifically
so a stale number is visible rather than silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_AS_OF = "2026-01"  # update this whenever FRONTIER_PRICING below is refreshed


@dataclass(frozen=True)
class FrontierPricing:
    """USD per 1,000,000 tokens, input and output priced separately."""

    provider: str
    model: str
    input_per_million: float
    output_per_million: float


# Published list pricing as of PRICE_AS_OF. These are the vendors' own
# standard API rates (not batch/cached discounts) — update alongside
# PRICE_AS_OF when re-benchmarking, and cite the source in the PR that
# updates them rather than trusting memory.
FRONTIER_PRICING: dict[str, FrontierPricing] = {
    # Display names are deliberately vendor-neutral tier labels; the dict keys
    # stay as stable internal identifiers used by the benchmark harness.
    "claude-opus": FrontierPricing(
        "Anthropic", "Frontier vendor A — flagship", input_per_million=15.00, output_per_million=75.00
    ),
    "claude-sonnet": FrontierPricing(
        "Anthropic", "Frontier vendor A — mid-tier", input_per_million=3.00, output_per_million=15.00
    ),
    "gpt-4o": FrontierPricing("OpenAI", "GPT-4o", input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-mini": FrontierPricing("OpenAI", "GPT-4o mini", input_per_million=0.15, output_per_million=0.60),
}


@dataclass(frozen=True)
class GPUCostProfile:
    """
    Real infrastructure economics for one self-hosted model deployment.

    `hourly_cost_usd` should reflect what the client actually pays per GPU
    (on-prem: amortized purchase + power + datacenter cost; RunPod/cloud:
    the quoted hourly rate) — this module does not guess it.
    `measured_tokens_per_second` must come from an actual load test against
    the real vLLM deployment (see benchmarks/run_benchmark.py's throughput
    probe), not a vendor claim — real serving throughput depends on batch
    size, sequence length, and quantization, all of which vary per
    deployment.
    """

    gpu_count: int
    hourly_cost_usd_per_gpu: float
    measured_tokens_per_second: float  # aggregate across gpu_count, at realistic batch size
    utilization: float = 1.0  # fraction of capacity actually in use; 1.0 = fully saturated

    @property
    def hourly_cost_usd(self) -> float:
        return self.gpu_count * self.hourly_cost_usd_per_gpu

    @property
    def effective_tokens_per_hour(self) -> float:
        return self.measured_tokens_per_second * 3600 * max(self.utilization, 1e-6)

    @property
    def cost_per_million_tokens(self) -> float:
        """
        USD per 1M tokens actually served, accounting for utilization — an
        idle GPU still costs money, so low utilization makes self-hosting
        look worse here exactly as it should in a real cost comparison.
        """
        if self.effective_tokens_per_hour <= 0:
            return float("inf")
        cost_per_token = self.hourly_cost_usd / self.effective_tokens_per_hour
        return cost_per_token * 1_000_000


@dataclass
class CostComparison:
    task_prompt_tokens: int
    task_completion_tokens: int
    self_hosted: GPUCostProfile
    frontier_key: str

    @property
    def self_hosted_cost_usd(self) -> float:
        total_tokens = self.task_prompt_tokens + self.task_completion_tokens
        return (total_tokens / 1_000_000) * self.self_hosted.cost_per_million_tokens

    @property
    def frontier_pricing(self) -> FrontierPricing:
        if self.frontier_key not in FRONTIER_PRICING:
            raise KeyError(f"Unknown frontier model '{self.frontier_key}'. Known: {list(FRONTIER_PRICING)}")
        return FRONTIER_PRICING[self.frontier_key]

    @property
    def frontier_cost_usd(self) -> float:
        p = self.frontier_pricing
        return (self.task_prompt_tokens / 1_000_000) * p.input_per_million + (
            self.task_completion_tokens / 1_000_000
        ) * p.output_per_million

    @property
    def savings_multiple(self) -> float:
        """How many times cheaper self-hosted is than the frontier model, for this task's token mix."""
        if self.self_hosted_cost_usd <= 0:
            return float("inf")
        return self.frontier_cost_usd / self.self_hosted_cost_usd

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.task_prompt_tokens,
            "completion_tokens": self.task_completion_tokens,
            "self_hosted_cost_usd": round(self.self_hosted_cost_usd, 6),
            "self_hosted_cost_per_million_tokens": round(self.self_hosted.cost_per_million_tokens, 4),
            "frontier_model": self.frontier_pricing.model,
            "frontier_provider": self.frontier_pricing.provider,
            "frontier_cost_usd": round(self.frontier_cost_usd, 6),
            "savings_multiple": round(self.savings_multiple, 1),
            "pricing_as_of": PRICE_AS_OF,
        }


def compare_all_frontier_models(prompt_tokens: int, completion_tokens: int, self_hosted: GPUCostProfile) -> list[dict]:
    """Cost comparison against every known frontier model, cheapest-savings-first is NOT assumed —
    callers should sort by whatever dimension matters to them."""
    return [CostComparison(prompt_tokens, completion_tokens, self_hosted, key).to_dict() for key in FRONTIER_PRICING]
