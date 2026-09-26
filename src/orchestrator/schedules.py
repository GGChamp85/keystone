# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — scheduled and webhook-triggered tasks.

A `TaskSchedule` (src/db/models.py) is a saved task template that submits a real `AgentTask` on
its own: a `cron` schedule advanced by the periodic loop below, or a `webhook` schedule triggered
the instant its unguessable `webhook_token` URL is hit
(`POST /v1/keystone/webhooks/schedules/{token}`, src/api/routes/schedules.py). Both paths call
the same `trigger_schedule`, which is also the one real function
`poll_and_trigger_due_schedules` calls — the webhook route never re-implements submission logic.

Not Temporal-durable — a lightweight periodic asyncio loop (`run_schedule_polling_loop`, started
from src/main.py's lifespan) rather than a Temporal Schedule, the same shape as
src/orchestrator/pr_polling.py and for the same reason: this codebase's own established
"non-durable asyncio fallback" honesty pattern for where a full durable-workflow version doesn't
exist yet. A missed poll (process restart) just gets picked up on the next interval — `next_run_at`
is recomputed from the real current time after each trigger, never accumulated as a backlog of
missed runs to fire all at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

import structlog
from croniter import croniter
from sqlalchemy import select

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import APIKey, ScheduleTrigger, TaskSchedule, Tenant

logger = structlog.get_logger(__name__)

_ModelRole = Literal["coding", "coding_fallback", "reasoning", "auto"]


class InvalidCronExpression(ValueError):
    """Raised for a missing or unparseable cron expression on a `trigger_type='cron'` schedule —
    mapped to a 422 by the route, not a 500."""


def _validate_cron(expression: str) -> None:
    if not croniter.is_valid(expression):
        raise InvalidCronExpression(f"{expression!r} is not a valid cron expression")


def _next_run_after(expression: str, after: datetime) -> datetime:
    return croniter(expression, after).get_next(datetime)


@dataclass
class ScheduleRecord:
    """A plain-data snapshot of a TaskSchedule row — callers outside the DB session boundary
    (API responses, the polling loop) work with this instead of a live ORM instance."""

    id: UUID
    tenant_id: UUID
    api_key_id: UUID
    name: str
    trigger_type: ScheduleTrigger
    cron_expression: str | None
    webhook_token: str | None
    task_description: str
    repository_url: str | None
    branch: str
    model_role: str
    max_iterations: int
    context_files: dict[str, str] | None
    enabled: bool
    last_triggered_at: datetime | None
    last_task_id: UUID | None
    next_run_at: datetime | None
    created_by: str | None

    @classmethod
    def from_orm(cls, row: TaskSchedule) -> ScheduleRecord:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            api_key_id=row.api_key_id,
            name=row.name,
            trigger_type=row.trigger_type,
            cron_expression=row.cron_expression,
            webhook_token=row.webhook_token,
            task_description=row.task_description,
            repository_url=row.repository_url,
            branch=row.branch,
            model_role=row.model_role,
            max_iterations=row.max_iterations,
            context_files=dict(row.context_files) if row.context_files else None,
            enabled=row.enabled,
            last_triggered_at=row.last_triggered_at,
            last_task_id=row.last_task_id,
            next_run_at=row.next_run_at,
            created_by=row.created_by,
        )


async def create_schedule(
    tenant_id: UUID,
    api_key_id: UUID,
    name: str,
    trigger_type: ScheduleTrigger,
    task_description: str,
    *,
    cron_expression: str | None = None,
    repository_url: str | None = None,
    branch: str = "main",
    model_role: _ModelRole = "coding",
    max_iterations: int = 15,
    context_files: dict[str, str] | None = None,
    enabled: bool = True,
    created_by: str | None = None,
) -> ScheduleRecord:
    webhook_token: str | None = None
    next_run_at: datetime | None = None
    stored_cron_expression: str | None = None

    if trigger_type == ScheduleTrigger.CRON:
        if not cron_expression:
            raise InvalidCronExpression("cron_expression is required when trigger_type is 'cron'")
        _validate_cron(cron_expression)
        stored_cron_expression = cron_expression
        next_run_at = _next_run_after(cron_expression, datetime.now(UTC))
    else:
        webhook_token = secrets.token_urlsafe(32)

    async with get_db_context() as db:
        row = TaskSchedule(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            name=name,
            trigger_type=trigger_type,
            cron_expression=stored_cron_expression,
            webhook_token=webhook_token,
            task_description=task_description,
            repository_url=repository_url,
            branch=branch,
            model_role=model_role,
            max_iterations=max_iterations,
            context_files=context_files,
            enabled=enabled,
            next_run_at=next_run_at,
            created_by=created_by,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        record = ScheduleRecord.from_orm(row)
    logger.info("schedules.created", tenant_id=str(tenant_id), name=name, trigger_type=trigger_type.value)
    return record


async def list_schedules(tenant_id: UUID, *, enabled: bool | None = None) -> list[ScheduleRecord]:
    async with get_db_context() as db:
        stmt = select(TaskSchedule).where(TaskSchedule.tenant_id == tenant_id)
        if enabled is not None:
            stmt = stmt.where(TaskSchedule.enabled == enabled)
        stmt = stmt.order_by(TaskSchedule.created_at.desc())
        result = await db.execute(stmt)
        return [ScheduleRecord.from_orm(row) for row in result.scalars().all()]


async def get_schedule(schedule_id: UUID) -> ScheduleRecord | None:
    async with get_db_context() as db:
        row = await db.get(TaskSchedule, schedule_id)
        return ScheduleRecord.from_orm(row) if row else None


async def get_schedule_by_webhook_token(token: str) -> ScheduleRecord | None:
    async with get_db_context() as db:
        result = await db.execute(select(TaskSchedule).where(TaskSchedule.webhook_token == token))
        row = result.scalar_one_or_none()
        return ScheduleRecord.from_orm(row) if row else None


async def set_enabled(schedule_id: UUID, enabled: bool) -> ScheduleRecord | None:
    async with get_db_context() as db:
        row = await db.get(TaskSchedule, schedule_id)
        if row is None:
            return None
        row.enabled = enabled
        await db.flush()
        await db.refresh(row)
        return ScheduleRecord.from_orm(row)


async def delete_schedule(schedule_id: UUID) -> bool:
    async with get_db_context() as db:
        row = await db.get(TaskSchedule, schedule_id)
        if row is None:
            return False
        await db.delete(row)
        return True


async def trigger_schedule(schedule_id: UUID) -> UUID:
    """Submit a real AgentTask from this schedule's template right now, regardless of trigger
    type — the poll loop calls this for a due `cron` schedule; the webhook route calls it the
    instant its token is hit. Advances `last_triggered_at`/`last_task_id` either way, and
    `next_run_at` for a `cron` schedule. Raises LookupError for a missing or disabled schedule,
    so a stale webhook URL or a race with `/disable` fails loudly instead of silently doing
    nothing."""
    from src.api.models.requests import AgentTaskRequest
    from src.api.routes.agents import submit_agent_task

    async with get_db_context() as db:
        row = await db.get(TaskSchedule, schedule_id)
        if row is None or not row.enabled:
            raise LookupError(f"Schedule {schedule_id} not found or disabled")
        tenant = await db.get(Tenant, row.tenant_id)
        api_key = await db.get(APIKey, row.api_key_id)
        if tenant is None or api_key is None:
            raise LookupError(f"Schedule {schedule_id} has no owning tenant/API key")

        req = AgentTaskRequest(
            task=row.task_description,
            repository_url=row.repository_url,
            branch=row.branch,
            model=cast(_ModelRole, row.model_role),
            max_iterations=row.max_iterations,
            context_files=row.context_files,
        )

    task_id = await submit_agent_task(req, api_key, tenant, None)

    async with get_db_context() as db:
        row = await db.get(TaskSchedule, schedule_id)
        assert row is not None  # just fetched above; nothing else deletes a schedule mid-trigger
        row.last_triggered_at = datetime.now(UTC)
        row.last_task_id = task_id
        if row.trigger_type == ScheduleTrigger.CRON and row.cron_expression:
            row.next_run_at = _next_run_after(row.cron_expression, datetime.now(UTC))
        await db.flush()

    logger.info("schedules.triggered", schedule_id=str(schedule_id), task_id=str(task_id))
    return task_id


async def poll_and_trigger_due_schedules() -> list[dict[str, UUID]]:
    """One poll pass over every enabled `cron` schedule whose `next_run_at` has passed. Returns
    the schedules triggered this pass (empty if none were due) — never raises: one schedule's
    submission failing (a concurrency limit, a budget cap, a bad repo URL) is logged and
    skipped, not allowed to abort the whole pass, matching pr_polling.py's own shape."""
    now = datetime.now(UTC)
    async with get_db_context() as db:
        result = await db.execute(
            select(TaskSchedule.id).where(
                TaskSchedule.trigger_type == ScheduleTrigger.CRON,
                TaskSchedule.enabled.is_(True),
                TaskSchedule.next_run_at.isnot(None),
                TaskSchedule.next_run_at <= now,
            )
        )
        due_ids = list(result.scalars().all())

    triggered: list[dict[str, UUID]] = []
    for schedule_id in due_ids:
        try:
            task_id = await trigger_schedule(schedule_id)
        except Exception as exc:
            logger.warning("schedules.trigger_failed", schedule_id=str(schedule_id), error=str(exc))
            continue
        triggered.append({"schedule_id": schedule_id, "task_id": task_id})
    return triggered


async def run_schedule_polling_loop(stop_event: asyncio.Event) -> None:
    """Runs `poll_and_trigger_due_schedules` every `settings.schedule_poll_interval_seconds`
    until `stop_event` is set (src/main.py's lifespan sets it on shutdown). A failed pass is
    logged and the loop keeps running — one bad poll should never silently kill all future
    ones."""
    settings = get_settings()
    interval = settings.schedule_poll_interval_seconds
    while not stop_event.is_set():
        try:
            await poll_and_trigger_due_schedules()
        except Exception as exc:
            logger.error("schedules.poll_loop_failed", error=str(exc))
        with contextlib.suppress(TimeoutError):  # normal — just means it's time for the next pass
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
