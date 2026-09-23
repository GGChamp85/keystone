# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Temporal workflow.

Durable execution wrapper for a coding agent task: if the worker process
crashes mid-task, Temporal replays/retries the single `run_agent_task`
activity rather than the task simply vanishing (the previous behavior when
tasks ran as a bare `asyncio.create_task`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities import run_agent_task

# Keep in sync with CircuitBreakerConfig.max_wall_clock_seconds
# (src/orchestrator/circuit_breaker.py) — the activity timeout must always be
# >= the circuit breaker's own wall-clock limit, or Temporal would kill the
# activity before the agent's own timeout logic gets a chance to fail cleanly.
DEFAULT_MAX_WALL_CLOCK_SECONDS = (
    86_400  # activity start-to-close: a day — tasks are bounded by max_iterations, not by a clock
)


@dataclass
class AgentWorkflowInput:
    task_id: str
    tenant_id: str
    api_key_id: str
    instruction: str
    repository_url: str | None = None
    branch: str = "main"
    target_files: list[str] = field(default_factory=list)
    context_files: dict[str, str] = field(default_factory=dict)
    max_iterations: int = 15
    preferred_model: str | None = None
    enable_reasoning_review: bool = True
    enable_sandbox_testing: bool = True
    user_id: str | None = None
    user_slug: str | None = None
    quality_blocking_tools: list[str] | None = None
    best_of_n: int | None = None


@workflow.defn
class CodingAgentWorkflow:
    """
    Temporal workflow for a coding agent task. A single activity
    (`run_agent_task`) runs the full LangGraph Plan->Code->Review->Test->Fix
    loop; Temporal's retry policy and heartbeat-timeout-based crash detection
    are what make this durable, not extra workflow-level steps.
    """

    @workflow.run
    async def run(self, input: AgentWorkflowInput) -> dict:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=10),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=2),
            maximum_attempts=2,  # the graph itself already retries fix/test loops internally
            non_retryable_error_types=["CancelledError"],
        )

        return await workflow.execute_activity(
            run_agent_task,
            args=[
                input.task_id,
                input.tenant_id,
                input.api_key_id,
                input.instruction,
                input.repository_url,
                input.branch,
                input.target_files,
                input.context_files,
                input.preferred_model,
                input.max_iterations,
                input.enable_reasoning_review,
                input.enable_sandbox_testing,
                input.user_id,
                input.user_slug,
                input.quality_blocking_tools,
                input.best_of_n,
            ],
            start_to_close_timeout=timedelta(seconds=DEFAULT_MAX_WALL_CLOCK_SECONDS + 300),
            heartbeat_timeout=timedelta(minutes=3),
            retry_policy=retry_policy,
        )
