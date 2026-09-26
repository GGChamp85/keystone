# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Skills library routes.

A skill is an admin-curated standing instruction applied to every matching task
(src/memory/skills.py) — distinct from memory (src/api/routes/memory.py), which an agent
proposes per-task from what actually happened. Same "agent" scope and per-tenant isolation as
memory, for the same reason: any team member with agent access curates what every task in the
tenant sees, and a skill never crosses a tenant boundary.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.middleware.auth import require_scope
from src.api.models.requests import CreateSkillRequest, MatchSkillsRequest
from src.api.models.responses import SkillResponse
from src.db.models import APIKey, Tenant, User
from src.memory.skills import (
    SkillRecord,
    create_skill,
    delete_skill,
    get_skill,
    list_skills,
    matching_skills,
    set_enabled,
)

router = APIRouter(prefix="/v1/keystone/skills", tags=["keystone"])


def _to_response(record: SkillRecord) -> SkillResponse:
    return SkillResponse(
        id=record.id,
        name=record.name,
        content=record.content,
        trigger_keywords=record.trigger_keywords,
        enabled=record.enabled,
        created_by=record.created_by,
    )


@router.get("", response_model=list[SkillResponse])
async def list_tenant_skills(
    enabled: bool | None = Query(default=None),
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    records = await list_skills(tenant.id, enabled=enabled)
    return [_to_response(r) for r in records]


@router.post("", response_model=SkillResponse, status_code=201)
async def add_skill(
    req: CreateSkillRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]
    user: User | None = auth[2]
    record = await create_skill(
        tenant.id,
        req.name,
        req.content,
        trigger_keywords=req.trigger_keywords,
        enabled=req.enabled,
        created_by=str(user.id) if user else str(api_key.id),
    )
    return _to_response(record)


@router.post("/match", response_model=list[SkillResponse])
async def match_skills(
    req: MatchSkillsRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    """Preview exactly which skills a task description would trigger — the same real function
    nodes/coding.py's prompt building calls, not an approximation of it."""
    tenant: Tenant = auth[1]
    records = await matching_skills(tenant.id, req.task)
    return [_to_response(r) for r in records]


async def _get_owned_skill(skill_id: UUID, tenant_id: UUID) -> SkillRecord:
    record = await get_skill(skill_id)
    if record is None or record.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    return record


@router.get("/{skill_id}", response_model=SkillResponse)
async def get_one_skill(
    skill_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    record = await _get_owned_skill(skill_id, tenant.id)
    return _to_response(record)


@router.post("/{skill_id}/enable", response_model=SkillResponse)
async def enable_skill(
    skill_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    await _get_owned_skill(skill_id, tenant.id)
    record = await set_enabled(skill_id, True)
    assert record is not None  # just confirmed it exists above
    return _to_response(record)


@router.post("/{skill_id}/disable", response_model=SkillResponse)
async def disable_skill(
    skill_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    await _get_owned_skill(skill_id, tenant.id)
    record = await set_enabled(skill_id, False)
    assert record is not None  # just confirmed it exists above
    return _to_response(record)


@router.delete("/{skill_id}", status_code=204)
async def remove_skill(
    skill_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    await _get_owned_skill(skill_id, tenant.id)
    await delete_skill(skill_id)
