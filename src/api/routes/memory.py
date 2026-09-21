# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Memory management routes.

Lets a user see and control what the agent has learned and will recall
into its own context — the thing every previous iteration of this
platform left entirely opaque. Every write here is attributed to the
real user behind the calling key when one exists (Phase 4's `User`
model), falling back to the API key's own id for keys not yet linked to
a person. Approving a tenant-wide (as opposed to repo-scoped) proposed
memory requires the lead or admin role — a tenant-wide convention affects
every repo the team touches, so it needs a team-level sign-off, per the
same governance the plan calls for on adapter promotion.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.middleware.auth import require_scope
from src.api.models.requests import CreateMemoryRequest, RecallMemoryRequest
from src.api.models.responses import MemoryResponse
from src.db.models import APIKey, Tenant, User, UserRole
from src.memory.store import (
    MemoryRecord,
    create_memory,
    get_memory,
    list_memories,
    recall,
    set_pinned,
    set_status,
)

router = APIRouter(prefix="/v1/keystone/memory", tags=["memory"])

_ROLE_RANK = {UserRole.DEVELOPER: 0, UserRole.LEAD: 1, UserRole.ADMIN: 2}


def _require_lead_for_tenant_scope(record: MemoryRecord, user: User | None) -> None:
    if record.scope != "tenant":
        return
    if user is None or _ROLE_RANK[user.role] < _ROLE_RANK[UserRole.LEAD]:
        raise HTTPException(
            status_code=403,
            detail="Approving a tenant-wide memory requires the lead or admin role",
        )


def _to_response(record: MemoryRecord) -> MemoryResponse:
    return MemoryResponse(
        id=record.id,
        repository=record.repository,
        scope=record.scope,
        kind=record.kind,
        content=record.content,
        source=record.source,
        status=record.status,
        pinned=record.pinned,
        confidence=record.confidence,
        hit_count=record.hit_count,
        created_by=record.created_by,
    )


@router.get("", response_model=list[MemoryResponse])
async def list_tenant_memories(
    repository: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    status: str | None = Query(default=None),
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    records = await list_memories(tenant.id, repository=repository, scope=scope, status=status)
    return [_to_response(r) for r in records]


@router.post("", response_model=MemoryResponse, status_code=201)
async def add_memory(
    req: CreateMemoryRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]
    user: User | None = auth[2]
    record = await create_memory(
        tenant.id,
        req.content,
        kind=req.kind,
        repository=req.repository,
        source="user",
        status="approved",
        pinned=req.pinned,
        created_by=str(user.id) if user else str(api_key.id),
    )
    return _to_response(record)


@router.post("/recall", response_model=list[MemoryResponse])
async def recall_memories(
    req: RecallMemoryRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    """Preview exactly what the agent would recall for a given query/repo — the same real ranking
    nodes/coding.py and nodes/planning.py use, not an approximation of it."""
    tenant: Tenant = auth[1]
    records = await recall(tenant.id, req.repository, req.query, budget_tokens=req.budget_tokens)
    return [_to_response(r) for r in records]


@router.post("/{memory_id}/approve", response_model=MemoryResponse)
async def approve_memory(
    memory_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    user: User | None = auth[2]
    existing = await get_memory(memory_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    _require_lead_for_tenant_scope(existing, user)
    record = await set_status(memory_id, "approved")
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _to_response(record)


@router.post("/{memory_id}/forget", response_model=MemoryResponse)
async def forget_memory(
    memory_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    record = await set_status(memory_id, "forgotten")
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _to_response(record)


@router.post("/{memory_id}/pin", response_model=MemoryResponse)
async def pin_memory(
    memory_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    record = await set_pinned(memory_id, True)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _to_response(record)


@router.post("/{memory_id}/unpin", response_model=MemoryResponse)
async def unpin_memory(
    memory_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    record = await set_pinned(memory_id, False)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _to_response(record)


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_one_memory(
    memory_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    record = await get_memory(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _to_response(record)
