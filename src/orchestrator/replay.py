# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — deterministic task replay.

`GET /v1/keystone/tasks/{task_id}/stream` (src/api/routes/agents.py) is a *live* view: it reads
a per-task Redis Stream (src/orchestrator/events.py) that expires 7 days after the task's last
event, or is gone sooner if Redis itself was restarted. But every step of every task is ALSO
mirrored, in full and in order, into `AgentTask.execution_trace` (a JSONB column) when the task
finishes — `IterationRecord.steps` per iteration, written by src/orchestrator/engine.py. This
module reconstructs the exact same SSE-shaped event sequence `/stream` would have produced live,
straight from that durable Postgres copy — so a task can be "replayed" step by step regardless
of Redis state or age.

Deliberately honest about what this is: a faithful replay of an execution that already happened,
built entirely from what was actually recorded — never a re-invocation of the model or the
sandbox, and never a claim that re-running the task would reproduce the same result (it wouldn't;
neither the model nor a task's git/sandbox state is deterministic across a fresh run).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from src.db.connection import get_db_context
from src.db.models import AgentTask


@dataclass
class TaskReplay:
    tenant_id: UUID
    events: list[dict]


async def build_task_replay(task_id: UUID) -> TaskReplay | None:
    """Flatten `AgentTask.execution_trace` (one `IterationRecord` dict per graph iteration, each
    carrying its own `steps` in original order) into a single, ordered event list, then append
    one final event built from the task's own stored outcome — the exact shape
    `publish_task_final` emits live, so `is_final_event` still closes a replay the same way a
    live stream closes. Returns None if the task doesn't exist."""
    async with get_db_context() as db:
        task = await db.get(AgentTask, task_id)
        if task is None:
            return None

        events: list[dict] = []
        for record in task.execution_trace or []:
            events.extend(record.get("steps") or [])

        events.append(
            {
                "event_type": "node",
                "node": "finalize",
                "phase": task.status.value,
                "final": True,
                "result_summary": task.result_summary or "",
                "error_message": task.error_message,
                "branch_name": task.branch_name,
                "commit_sha": task.commit_sha,
                "pr_url": task.pr_url,
                "pr_number": task.pr_number,
                "timestamp": task.completed_at.timestamp() if task.completed_at else time.time(),
            }
        )

        return TaskReplay(tenant_id=task.tenant_id, events=events)
