# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Agent task management routes.
"""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_scope
from src.api.models.requests import AgentTaskRequest, IngestRepositoryRequest
from src.api.models.responses import AgentTaskResponse, AgentTaskSubmittedResponse
from src.db.models import APIKey, Tenant
from src.memory.ingestion import CodeIngestionPipeline
from src.orchestrator.engine import get_keystone_engine
from src.orchestrator.events import TERMINAL_PHASES, block_for_next_event, read_task_events_from

router = APIRouter(prefix="/v1/keystone", tags=["keystone"])


@router.post("/tasks", response_model=AgentTaskSubmittedResponse, status_code=202)
async def submit_task(
    req: AgentTaskRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]
    engine = get_keystone_engine()

    task_id = await engine.submit_task(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        task_description=req.task,
        repository_url=req.repository_url,
        branch=req.branch,
        file_paths=req.file_paths,
        model=req.model,
        max_iterations=req.max_iterations,
        enable_reasoning_review=req.enable_reasoning_review,
        enable_sandbox_testing=req.enable_sandbox_testing,
        context_files=req.context_files,
    )

    return AgentTaskSubmittedResponse(task_id=task_id, status="pending")


@router.get("/tasks/{task_id}", response_model=AgentTaskResponse)
async def get_task(
    task_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    engine = get_keystone_engine()
    result = await engine.get_task_status(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return AgentTaskResponse(**result)


@router.get("/tasks/{task_id}/stream")
async def stream_task(
    task_id: UUID,
    request: Request,
    auth: tuple = Depends(require_scope("agent")),
):
    """
    Server-Sent Events feed of the live plan/execution trace — replays
    every event recorded so far (so a client connecting mid-task, or
    reconnecting after a network blip, still sees the full plan and
    history-to-date), then streams new events as the agent graph
    progresses through each node (src/orchestrator/events.py). Closes
    itself once the task reaches a terminal phase, or when the client
    disconnects.
    """

    async def event_generator():
        last_id = "0-0"
        history = await read_task_events_from(task_id, last_id)
        for entry_id, payload in history:
            last_id = entry_id
            yield f"id: {entry_id}\ndata: {json.dumps(payload)}\n\n"
            if payload.get("phase") in TERMINAL_PHASES:
                return  # task already finished before this client connected

        while True:
            if await request.is_disconnected():
                break
            result = await block_for_next_event(task_id, last_id, timeout_ms=15_000)
            if result is None:
                yield ": keep-alive\n\n"  # SSE comment — keeps intermediary proxies from timing the connection out
                continue
            entry_id, payload = result
            last_id = entry_id
            yield f"id: {entry_id}\ndata: {json.dumps(payload)}\n\n"
            if payload.get("phase") in TERMINAL_PHASES:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: UUID,
    auth: tuple = Depends(require_scope("agent")),
):
    engine = get_keystone_engine()
    cancelled = await engine.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Task not found or not running")
    return {"status": "cancelled", "task_id": str(task_id)}


@router.post("/ingest")
async def ingest_repository(
    req: IngestRepositoryRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    tenant: Tenant = auth[1]
    pipeline = CodeIngestionPipeline()
    result = await pipeline.ingest_repository(
        repository_url=req.repository_url,
        tenant_id=str(tenant.id),
        branch=req.branch,
        file_extensions=req.file_extensions,
        max_file_size_kb=req.max_file_size_kb,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return result
