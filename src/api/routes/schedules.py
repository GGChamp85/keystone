# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — scheduled and webhook-triggered task routes.

A schedule is a saved task template that submits a real task on its own
(src/orchestrator/schedules.py) — either on a cron schedule, or the instant its webhook URL is
hit. CRUD is under the usual `agent`-scoped, per-tenant `/v1/keystone/schedules` (same pattern as
skills, src/api/routes/skills.py); the trigger route is deliberately separate and unauthenticated
by API key — its own unguessable `webhook_token` in the URL is the auth, the same way a CI system
or external webhook sender would call it without ever holding a Keystone API key.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.middleware.auth import require_scope
from src.api.models.requests import CreateScheduleRequest
from src.api.models.responses import ScheduleResponse
from src.db.models import APIKey, ScheduleTrigger, Tenant, User
from src.orchestrator.concurrency import ConcurrencyLimitExceeded
from src.orchestrator.schedules import (
    InvalidCronExpression,
    ScheduleRecord,
    create_schedule,
    delete_schedule,
    get_schedule,
    get_schedule_by_webhook_token,
    list_schedules,
    set_enabled,
    trigger_schedule,
)

router = APIRouter(prefix="/v1/keystone/schedules", tags=["keystone"])
webhook_router = APIRouter(tags=["keystone"])


def _to_response(record: ScheduleRecord) -> ScheduleResponse:
    return ScheduleResponse(
        id=record.id,
        name=record.name,
        trigger_type=record.trigger_type.value,
        cron_expression=record.cron_expression,
        webhook_url=f"/v1/keystone/webhooks/schedules/{record.webhook_token}" if record.webhook_token else None,
        task=record.task_description,
        repository_url=record.repository_url,
        branch=record.branch,
        model=record.model_role,
        max_iterations=record.max_iterations,
        context_files=record.context_files,
        enabled=record.enabled,
        last_triggered_at=record.last_triggered_at,
        last_task_id=record.last_task_id,
        next_run_at=record.next_run_at,
        created_by=record.created_by,
    )


@router.get("", response_model=list[ScheduleResponse])
async def list_tenant_schedules(
    enabled: bool | None = Query(default=None),
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    records = await list_schedules(tenant.id, enabled=enabled)
    return [_to_response(r) for r in records]


@router.post("", response_model=ScheduleResponse, status_code=201)
async def add_schedule(
    req: CreateScheduleRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]
    user: User | None = auth[2]
    try:
        record = await create_schedule(
            tenant.id,
            api_key.id,
            req.name,
            ScheduleTrigger(req.trigger_type),
            req.task,
            cron_expression=req.cron_expression,
            repository_url=req.repository_url,
            branch=req.branch,
            model_role=req.model,
            max_iterations=req.max_iterations,
            context_files=req.context_files,
            enabled=req.enabled,
            created_by=str(user.id) if user else str(api_key.id),
        )
    except InvalidCronExpression as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(record)


async def _get_owned_schedule(schedule_id: UUID, tenant_id: UUID) -> ScheduleRecord:
    record = await get_schedule(schedule_id)
    if record is None or record.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return record


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_one_schedule(
    schedule_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    record = await _get_owned_schedule(schedule_id, tenant.id)
    return _to_response(record)


@router.post("/{schedule_id}/enable", response_model=ScheduleResponse)
async def enable_schedule(
    schedule_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    await _get_owned_schedule(schedule_id, tenant.id)
    record = await set_enabled(schedule_id, True)
    assert record is not None  # just confirmed it exists above
    return _to_response(record)


@router.post("/{schedule_id}/disable", response_model=ScheduleResponse)
async def disable_schedule(
    schedule_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    await _get_owned_schedule(schedule_id, tenant.id)
    record = await set_enabled(schedule_id, False)
    assert record is not None  # just confirmed it exists above
    return _to_response(record)


@router.delete("/{schedule_id}", status_code=204)
async def remove_schedule(
    schedule_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    await _get_owned_schedule(schedule_id, tenant.id)
    await delete_schedule(schedule_id)


@webhook_router.post("/v1/keystone/webhooks/schedules/{token}", status_code=202)
async def trigger_schedule_webhook(token: str):
    """Deliberately unauthenticated by API key: the unguessable `token` itself is the secret, the
    same trust model as any other webhook receiver (e.g. a git host's own push webhooks) —
    generated once by `create_schedule` (`secrets.token_urlsafe(32)`) and never otherwise
    reachable, since it's returned only on this schedule's own CRUD responses."""
    record = await get_schedule_by_webhook_token(token)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown schedule webhook")
    try:
        task_id = await trigger_schedule(record.id)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConcurrencyLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "pending"}
