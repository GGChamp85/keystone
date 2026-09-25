# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Agent task management routes.
"""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_scope
from src.api.models.requests import (
    AgentTaskRequest,
    IngestRepositoryRequest,
    SubmitTaskFeedbackRequest,
    TaskCostEstimateRequest,
)
from src.api.models.responses import (
    AgentTaskResponse,
    AgentTaskSubmittedResponse,
    AgentTaskSummaryResponse,
    TaskFeedbackResponse,
)
from src.api.routes._inference_common import check_dollar_budget
from src.billing.ledger import month_to_date_cost_usd, usage_summary
from src.db.connection import get_db_context
from src.db.models import AgentTask, APIKey, FeedbackVerdict, TaskFeedback, Tenant, User
from src.memory.ingestion import CodeIngestionPipeline, InvalidRepositoryURLError
from src.orchestrator.concurrency import ConcurrencyLimitExceeded
from src.orchestrator.engine import ImagesRequireVisionModelError, get_keystone_engine
from src.orchestrator.events import block_for_next_event, is_final_event, read_task_events_from
from src.orchestrator.nodes._shared import slugify_for_branch

router = APIRouter(prefix="/v1/keystone", tags=["keystone"])


async def submit_agent_task(req: AgentTaskRequest, api_key: APIKey, tenant: Tenant, user: User | None) -> UUID:
    """The one submission path behind `POST /v1/keystone/tasks` and the MCP `task_submit` tool
    (src/api/routes/mcp.py): a validated AgentTaskRequest plus the authenticated caller, handed
    to the engine with the user's branch slug. Raises ConcurrencyLimitExceeded; each caller maps
    it to its own protocol's error (HTTP 429 here, a ToolError over MCP). The tenant's monthly dollar
    budget is checked here too, so every submission path honours it."""
    await check_dollar_budget(tenant, surface="agent")
    engine = get_keystone_engine()
    return await engine.submit_task(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        user_id=user.id if user else None,
        user_slug=slugify_for_branch(user.email.split("@")[0]) if user else None,
        task_description=req.task,
        repository_url=req.repository_url,
        branch=req.branch,
        file_paths=req.file_paths,
        model=req.model,
        max_iterations=req.max_iterations,
        enable_reasoning_review=req.enable_reasoning_review,
        enable_sandbox_testing=req.enable_sandbox_testing,
        context_files=req.context_files,
        images=[img.model_dump() for img in req.images] if req.images else None,
        quality_blocking_tools=req.quality_blocking_tools,
        best_of_n=req.best_of_n,
    )


@router.post("/tasks", response_model=AgentTaskSubmittedResponse, status_code=202)
async def submit_task(
    req: AgentTaskRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]
    user: User | None = auth[2]

    try:
        task_id = await submit_agent_task(req, api_key, tenant, user)
    except ConcurrencyLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ImagesRequireVisionModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AgentTaskSubmittedResponse(task_id=task_id, status="pending")


@router.get("/tasks", response_model=list[AgentTaskSummaryResponse])
async def list_tasks(
    user_id: UUID | None = Query(default=None, description="Filter to one team member's tasks"),
    repository_url: str | None = Query(default=None, description="Filter to one repository"),
    limit: int = Query(default=50, ge=1, le=200),
    auth: tuple = Depends(require_scope("agent")),
):
    """Team task list — who's working on what, and its PR status. Backs the
    web UI's team view; any valid tenant key can see the whole tenant's
    tasks (not just its own), matching the plan's "task list across the
    team" requirement rather than a narrower per-key view."""
    tenant: Tenant = auth[1]
    engine = get_keystone_engine()
    tasks = await engine.list_tasks(tenant.id, user_id=user_id, repository_url=repository_url, limit=limit)
    return [AgentTaskSummaryResponse(**t) for t in tasks]


@router.post("/tasks/estimate")
async def estimate_task(
    req: TaskCostEstimateRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    """A labelled cost estimate before submitting a task — no task is created. Drawn from this tenant's own
    completed tasks when it has any, else every completed task on this deployment, else a documented rough
    default; `estimate_basis` always says which. Never more precise than the history backing it."""
    from src.inference.model_router import classify_task_to_role
    from src.orchestrator.cost_estimate import estimate_task_cost

    tenant: Tenant = auth[1]
    role = classify_task_to_role(req.task) if req.model == "auto" else req.model
    estimate = await estimate_task_cost(
        req.task,
        tenant_id=tenant.id,
        model_role=role,
        repository_url=req.repository_url,
        best_of_n=req.best_of_n or 1,
    )
    return estimate.to_dict()


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


@router.get("/usage")
async def tenant_usage(
    days: int = 30,
    auth: tuple = Depends(require_scope("agent")),
):
    """This tenant's spend — tokens, requests and dollars per day and model role (src/billing/ledger.py),
    for the last `days` days, plus totals. `pricing_configured` is false until the operator sets
    MODEL_PRICES_PER_MILLION; cost is then 0 rather than a made-up number."""
    tenant: Tenant = auth[1]
    summary = await usage_summary(tenant.id, days=max(1, min(days, 3650)))
    budget = float(tenant.monthly_budget_usd or 0)
    summary["budget"] = {
        "monthly_budget_usd": budget,  # 0 = no budget
        "month_to_date_usd": await month_to_date_cost_usd(tenant.id),
    }
    return summary


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
            if is_final_event(payload):
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
            if is_final_event(payload):
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


@router.post("/tasks/{task_id}/feedback", response_model=TaskFeedbackResponse, status_code=201)
async def submit_task_feedback(
    task_id: UUID,
    req: SubmitTaskFeedbackRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    """A human's own verdict on a completed task — accept/reject/merge/revert
    plus an optional reason src/memory/extract.py can later turn into a
    proposed memory. Complements src/orchestrator/pr_polling.py's automatic
    merged/rejected detection; this is the path for everything a PR's own
    open/closed/merged state can't tell you (accepted-without-a-PR, a
    revert caught by a human, etc.)."""
    tenant: Tenant = auth[1]
    user: User | None = auth[2]

    async with get_db_context() as db:
        task = await db.get(AgentTask, task_id)
        if task is None or task.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Task not found")

        feedback = TaskFeedback(
            task_id=task_id,
            user_id=str(user.id) if user else None,
            verdict=FeedbackVerdict(req.verdict),
            reason=req.reason,
        )
        db.add(feedback)
        await db.flush()
        return TaskFeedbackResponse(
            id=feedback.id,
            task_id=feedback.task_id,
            user_id=feedback.user_id,
            verdict=feedback.verdict.value,
            reason=feedback.reason,
            created_at=feedback.created_at,
        )


@router.post("/ingest")
async def ingest_repository(
    req: IngestRepositoryRequest,
    auth: tuple = Depends(require_scope("agent")),
):
    import git

    tenant: Tenant = auth[1]
    pipeline = CodeIngestionPipeline()
    try:
        result = await pipeline.ingest_repository(
            repository_url=req.repository_url,
            tenant_id=str(tenant.id),
            branch=req.branch,
            file_extensions=req.file_extensions,
            max_file_size_kb=req.max_file_size_kb,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
    except InvalidRepositoryURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except git.exc.GitCommandError as exc:
        raise HTTPException(status_code=400, detail=f"git clone failed: {exc}") from exc
    return result
