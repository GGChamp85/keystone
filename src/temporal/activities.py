# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Temporal activity.

A single activity wraps the full LangGraph agent run
(KeystoneEngine._execute_task). Sandbox creation/teardown is NOT a separate
activity here — it's already self-contained per-iteration inside
`testing_node` (src/orchestrator/nodes/testing.py), which creates and
destroys its own sandbox via SandboxManager on every graph pass. Splitting
that into standalone setup/cleanup activities (the original design this
replaces) doesn't match how the graph actually works and only added a
second, inconsistent sandbox lifecycle.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from temporalio import activity

logger = structlog.get_logger(__name__)


@activity.defn
async def run_agent_task(
    task_id: str,
    tenant_id: str,
    api_key_id: str,
    instruction: str,
    repository_url: str | None,
    branch: str,
    target_files: list[str],
    context_files: dict[str, str],
    preferred_model: str | None,
    max_iterations: int,
    enable_reasoning_review: bool,
    enable_sandbox_testing: bool,
    user_id: str | None = None,
    user_slug: str | None = None,
    quality_blocking_tools: list[str] | None = None,
) -> dict[str, Any]:
    """
    Runs the full Plan -> Code -> Review -> Test -> Fix loop for one task.
    Heartbeats after every graph node so Temporal's activity-timeout-based
    crash detection doesn't fire mid-task; if the worker process dies,
    Temporal retries this activity (KeystoneEngine._execute_task's DB writes
    make each retry pick up from a consistent DB row, and testing_node's own
    try/finally still guarantees sandbox cleanup even on a killed retry).
    """
    from src.orchestrator.engine import get_keystone_engine
    from src.orchestrator.events import publish_task_event

    engine = get_keystone_engine()

    async def _heartbeat(node_name: str, state: dict[str, Any]) -> None:
        activity.heartbeat(f"{node_name}:iteration={state.get('iteration', 0)}")
        await publish_task_event(task_id, node_name, state)

    try:
        return await engine._execute_task(
            task_id=UUID(task_id),
            tenant_id=UUID(tenant_id),
            api_key_id=UUID(api_key_id),
            task_description=instruction,
            repository_url=repository_url,
            branch=branch,
            file_paths=target_files,
            model=preferred_model or "coding",
            max_iterations=max_iterations,
            enable_reasoning_review=enable_reasoning_review,
            enable_sandbox_testing=enable_sandbox_testing,
            context_files=context_files,
            user_id=UUID(user_id) if user_id else None,
            user_slug=user_slug,
            quality_blocking_tools=quality_blocking_tools,
            heartbeat_callback=_heartbeat,
        )
    except Exception as exc:
        # Re-raised as-is: Temporal's activity worker already converts any
        # exception raised here into a proper ApplicationFailure with retry
        # semantics per the workflow's RetryPolicy — wrapping it ourselves
        # would just lose the original type/traceback for no benefit.
        logger.error("temporal.activity_failed", task_id=task_id, error=str(exc))
        raise
