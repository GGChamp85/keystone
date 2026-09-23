# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — pre-task cost estimate and per-task cost breakdown.

**Pre-task estimate** (`estimate_task_cost`): before a task is submitted, what it will likely cost, from
real history rather than an invented throughput number. This tenant's own completed tasks are the primary
source (their prompt+completion token totals are the closest thing to "what a task like this costs" this
deployment has); when this tenant has none yet, every tenant's completed tasks are used instead; only when
the database has no completed task at all does a documented rough default apply. Every estimate states
which of the three it used (`estimate_basis`) — never presented as more precise than it is.

**Per-task breakdown** (`task_cost_breakdown`): after (or during) a task, the same `execution_trace` the
API already returns, grouped by phase and model role, priced by the operator's own
`MODEL_PRICES_PER_MILLION` — the exact rows the durable ledger recorded, not a second estimate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AgentTask, TaskStatus
from src.orchestrator.context import count_tokens
from src.orchestrator.nodes.coding import TOOL_CODING_SYSTEM_PROMPT
from src.orchestrator.repo_map import DEFAULT_MAP_TOKENS

# A completed task with no comparable history at all: a labelled rough default, not a measured number.
# Basis: the token budget one full agentic-loop turn realistically uses (context.py's summarizer keeps
# turns under AGENT_MAX_CONTEXT_TOKENS) times a small number of turns for a typical bug-fix-sized task.
_FALLBACK_TOTAL_TOKENS = 40_000
_FALLBACK_PROMPT_FRACTION = 0.75  # tool-based loops re-send growing context every turn; most tokens are prompt


@dataclass(frozen=True)
class TaskCostEstimate:
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_cost_usd: float | None  # None when the role has no configured price
    pricing_configured: bool
    model_role: str
    best_of_n: int
    sample_size: int  # how many historical tasks the estimate is drawn from (0 = the fallback default)
    estimate_basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_completion_tokens": self.estimated_completion_tokens,
            "estimated_total_tokens": self.estimated_prompt_tokens + self.estimated_completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4) if self.estimated_cost_usd is not None else None,
            "pricing_configured": self.pricing_configured,
            "model_role": self.model_role,
            "best_of_n": self.best_of_n,
            "sample_size": self.sample_size,
            "estimate_basis": self.estimate_basis,
        }


async def _historical_average(tenant_id: UUID | None) -> tuple[float, float, int]:
    """Mean (prompt_tokens, completion_tokens) over completed tasks, and how many contributed. Scoped to
    `tenant_id` when given; the caller falls back to platform-wide when a tenant has none yet."""
    stmt = select(
        func.avg(AgentTask.total_prompt_tokens), func.avg(AgentTask.total_completion_tokens), func.count(AgentTask.id)
    ).where(AgentTask.status == TaskStatus.COMPLETED, AgentTask.total_prompt_tokens > 0)
    if tenant_id is not None:
        stmt = stmt.where(AgentTask.tenant_id == tenant_id)
    async with get_db_context() as db:
        avg_prompt, avg_completion, n = (await db.execute(stmt)).one()
    return float(avg_prompt or 0), float(avg_completion or 0), int(n or 0)


async def estimate_task_cost(
    task_description: str,
    *,
    tenant_id: UUID,
    model_role: str = "coding",
    repository_url: str | None = None,
    best_of_n: int = 1,
) -> TaskCostEstimate:
    """A labelled estimate for a task before it runs. `repository_url` only affects the floor (a repo task
    always pays for the system prompt and, once cloned, the repository map — see the floor below); it does
    not change which historical average is used, since token totals already reflect repo-having tasks."""
    settings = get_settings()
    prices = settings.model_prices_per_million or {}

    prompt_avg, completion_avg, n = await _historical_average(tenant_id)
    basis = f"this tenant's own {n} completed task(s)" if n else ""
    if not n:
        prompt_avg, completion_avg, n = await _historical_average(None)
        basis = f"{n} completed task(s) across this deployment" if n else ""
    if not n:
        prompt_avg = _FALLBACK_TOTAL_TOKENS * _FALLBACK_PROMPT_FRACTION
        completion_avg = _FALLBACK_TOTAL_TOKENS * (1 - _FALLBACK_PROMPT_FRACTION)
        basis = "no completed task yet on this deployment — a rough, documented default, not a measurement"

    # A floor from what is known exactly regardless of history: the fixed system prompt, and the task's
    # own description (counted with the real tokenizer, not guessed).
    floor_prompt = count_tokens(TOOL_CODING_SYSTEM_PROMPT) + count_tokens(task_description)
    if repository_url:
        floor_prompt += DEFAULT_MAP_TOKENS
    prompt_tokens = int(max(prompt_avg, floor_prompt) * max(best_of_n, 1))
    completion_tokens = int(completion_avg * max(best_of_n, 1))

    price = prices.get(model_role)
    cost = (prompt_tokens * price + completion_tokens * price) / 1_000_000 if price else None
    if price:
        basis += f"; priced at ${price}/M tokens for the '{model_role}' role"
    else:
        basis += f"; no price configured for the '{model_role}' role (MODEL_PRICES_PER_MILLION), so no dollar figure"
    if best_of_n > 1:
        basis += f"; x{best_of_n} for best_of_n"

    return TaskCostEstimate(
        estimated_prompt_tokens=prompt_tokens,
        estimated_completion_tokens=completion_tokens,
        estimated_cost_usd=cost,
        pricing_configured=bool(price),
        model_role=model_role,
        best_of_n=max(best_of_n, 1),
        sample_size=n,
        estimate_basis=basis,
    )


@dataclass
class CostBreakdownRow:
    phase: str
    model_role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    iterations: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def task_cost_breakdown(execution_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """The task's own `execution_trace` (one `IterationRecord.to_dict()` per graph pass), grouped by
    (phase, model_role) and priced by the operator's configured rates — the real per-task cost, not an
    estimate. `pricing_configured` mirrors the ledger's usage summary: false means tokens are shown, cost
    is 0, never a made-up number."""
    prices = get_settings().model_prices_per_million or {}
    rows: dict[tuple[str, str], CostBreakdownRow] = defaultdict(lambda: CostBreakdownRow("", ""))
    for record in execution_trace:
        phase = str(record.get("phase") or "unknown")
        role = str(record.get("model_role") or "")
        prompt = int(record.get("prompt_tokens") or 0)
        completion = int(record.get("completion_tokens") or 0)
        if prompt == 0 and completion == 0:
            continue
        key = (phase, role)
        row = rows[key]
        row.phase, row.model_role = phase, role
        row.prompt_tokens += prompt
        row.completion_tokens += completion
        row.iterations += 1

    def _cost(row: CostBreakdownRow) -> float:
        price = prices.get(row.model_role)
        return (row.prompt_tokens * price + row.completion_tokens * price) / 1_000_000 if price else 0.0

    breakdown: list[dict[str, Any]] = [
        {
            "phase": r.phase,
            "model_role": r.model_role,
            "iterations": r.iterations,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "estimated_cost_usd": round(_cost(r), 6),
        }
        for r in sorted(rows.values(), key=lambda r: (r.phase, r.model_role))
    ]
    return {
        "pricing_configured": bool(prices),
        "prices_per_million": {k: float(v) for k, v in prices.items()},
        "rows": breakdown,
        "totals": {
            "prompt_tokens": sum(r["prompt_tokens"] for r in breakdown),
            "completion_tokens": sum(r["completion_tokens"] for r in breakdown),
            "total_tokens": sum(r["total_tokens"] for r in breakdown),
            "estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in breakdown), 6),
        },
    }
