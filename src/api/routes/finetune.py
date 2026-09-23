# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Fine-tuning job management routes.

Submission validates `config` against the real trainer Config dataclass's
known fields (src/finetuning/runner.py's `_build_config` otherwise
silently drops anything unrecognized) so a typo in a job's config is a
clear 422 at submission time, not a job that runs for hours and only then
fails — or worse, silently ignores the field and trains with a default
the caller didn't intend.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from src.api.middleware.auth import require_scope
from src.api.models.requests import ApprovePlanRequest, GuidedPlanRequest, StartFineTuneRequest
from src.api.models.responses import FineTuneJobResponse
from src.db.connection import get_db_context
from src.db.models import AdapterStatus, APIKey, AuditLog, FineTuneJob, ModelAdapter, Tenant, User, UserRole
from src.finetuning import guided
from src.finetuning.runner import _JOB_TYPE_CONFIGS, start_finetune_job
from src.finetuning.verdict import compute_verdict
from src.hardware import GPU, detect_gpus
from src.inference.catalog import CATALOG, default_slm
from src.inference.lora_registry import serve_promoted_adapter
from src.memory.ingestion import InvalidRepositoryURLError

router = APIRouter(prefix="/v1/finetune", tags=["finetune"])


def _to_response(job: FineTuneJob) -> FineTuneJobResponse:
    return FineTuneJobResponse(
        id=job.id,
        base_model=job.base_model,
        job_type=job.job_type,
        status=job.status,
        config=job.config or {},
        metrics=job.metrics or {},
        output_model_path=job.output_model_path,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
        error_message=job.error_message,
    )


def _validate_config(job_type: str, config: dict) -> None:
    if job_type not in _JOB_TYPE_CONFIGS:
        raise HTTPException(status_code=422, detail=f"Unknown job_type '{job_type}' (expected one of lora/sft/dpo)")
    module_name, config_cls_name, _run_fn_name = _JOB_TYPE_CONFIGS[job_type]
    import importlib

    config_cls = getattr(importlib.import_module(module_name), config_cls_name)
    known_fields = {f.name for f in dataclasses.fields(config_cls)} - {"base_model"}
    unknown = set(config) - known_fields
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown config field(s) for a {job_type} job: {sorted(unknown)}. Valid fields: {sorted(known_fields)}"
            ),
        )


@router.get("/catalog")
async def model_catalog(auth: tuple = Depends(require_scope("finetune"))):
    """The SLM catalog (src/inference/catalog.py) with VRAM footprints, plus the GPUs detected on this host."""
    return {
        "models": [e.to_dict() for e in CATALOG],
        "default": default_slm().hf_id,
        "detected_gpus": [g.to_dict() for g in detect_gpus()],
    }


@router.post("/plans", status_code=201)
async def create_guided_plan(req: GuidedPlanRequest, auth: tuple = Depends(require_scope("finetune"))):
    """Describe → plan & cost. Builds the dataset from the repositories' real history (and this tenant's
    accepted agent tasks), splits it, picks QLoRA/LoRA for the hardware, and persists a `planned` job.
    Nothing trains until POST /plans/{id}/approve."""
    tenant: Tenant = auth[1]
    try:
        plan = await guided.create_plan(
            tenant.id,
            repositories=req.repositories,
            goal=req.goal,
            base_model=req.base_model,
            holdout_ratio=req.holdout_ratio,
            epochs=req.epochs,
            lora_r=req.lora_r,
            gpus=[GPU(name=g.name, vram_gb=g.vram_gb) for g in req.gpus] if req.gpus is not None else None,
            gpu_hourly_cost_usd=req.gpu_hourly_cost_usd,
            clone=guided.default_clone,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidRepositoryURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return plan.to_dict()


@router.get("/plans/{job_id}")
async def get_guided_plan(job_id: UUID, auth: tuple = Depends(require_scope("finetune"))):
    tenant: Tenant = auth[1]
    job = await _get_owned_job(job_id, tenant.id)
    body = (job.config or {}).get("guided")
    if not body:
        raise HTTPException(status_code=404, detail="not a guided plan")
    return {"job_id": str(job.id), "status": job.status, "base_model": job.base_model, **body}


@router.post("/plans/{job_id}/approve", response_model=FineTuneJobResponse)
async def approve_guided_plan(
    job_id: UUID, req: ApprovePlanRequest | None = None, auth: tuple = Depends(require_scope("finetune"))
):
    """Approve → train: the planned job becomes pending and starts (Temporal, or the asyncio fallback)."""
    tenant: Tenant = auth[1]
    try:
        job = await guided.approve_plan(job_id, tenant.id, force=bool(req and req.force))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except guided.PlanNotApprovable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_response(job)


@router.post("/jobs", response_model=FineTuneJobResponse, status_code=201)
async def start_job(req: StartFineTuneRequest, auth: tuple = Depends(require_scope("finetune"))):
    tenant: Tenant = auth[1]
    config = {**req.config, "training_data": req.training_data_path}
    _validate_config(req.job_type, config)

    async with get_db_context() as db:
        job = FineTuneJob(
            tenant_id=tenant.id,
            base_model=req.base_model,
            job_type=req.job_type,
            status="pending",
            config=config,
        )
        db.add(job)
        await db.flush()
        job_id = job.id
        response = _to_response(job)

    await start_finetune_job(job_id)
    return response


@router.get("/jobs", response_model=list[FineTuneJobResponse])
async def list_jobs(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    auth: tuple = Depends(require_scope("finetune")),
):
    tenant: Tenant = auth[1]
    async with get_db_context() as db:
        stmt = select(FineTuneJob).where(FineTuneJob.tenant_id == tenant.id)
        if status:
            stmt = stmt.where(FineTuneJob.status == status)
        stmt = stmt.order_by(FineTuneJob.created_at.desc()).limit(limit)
        jobs = (await db.execute(stmt)).scalars().all()
    return [_to_response(j) for j in jobs]


async def _get_owned_job(job_id: UUID, tenant_id: UUID) -> FineTuneJob:
    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
    if job is None or job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return job


@router.get("/jobs/{job_id}", response_model=FineTuneJobResponse)
async def get_job(job_id: UUID, auth: tuple = Depends(require_scope("finetune"))):
    tenant: Tenant = auth[1]
    job = await _get_owned_job(job_id, tenant.id)
    return _to_response(job)


@router.get("/jobs/{job_id}/dataset-preview")
async def preview_dataset(
    job_id: UUID,
    lines: int = Query(default=5, ge=1, le=50),
    auth: tuple = Depends(require_scope("finetune")),
):
    """The first `lines` real records from the job's own training_data file —
    what a user should see before a run starts, not a description of it."""
    tenant: Tenant = auth[1]
    job = await _get_owned_job(job_id, tenant.id)
    path_str = (job.config or {}).get("training_data")
    if not path_str:
        raise HTTPException(status_code=404, detail="This job has no training_data path recorded")
    path = Path(path_str)

    result = await asyncio.to_thread(_read_dataset_preview, path, lines)
    if result is None:
        raise HTTPException(status_code=404, detail=f"training_data not found on disk: {path_str}")
    total, records = result
    return {"path": path_str, "total_records": total, "preview": records}


def _read_dataset_preview(path: Path, lines: int) -> tuple[int, list[dict]] | None:
    if not path.exists():
        return None
    records: list[dict] = []
    total = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if len(records) < lines:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({"_raw": line.strip()})
    return total, records


@router.post("/jobs/{job_id}/promote", response_model=FineTuneJobResponse)
async def promote_job(
    job_id: UUID,
    request: Request,
    force: bool = Query(default=False, description="Admin only: promote despite a fail/unknown verdict"),
    auth: tuple = Depends(require_scope("finetune")),
):
    """
    Registers (or updates) this job's adapter as the tenant's promoted
    default for its base model — the one src/inference/model_router.py's
    resolve_served_model_name actually routes real requests to. Demotes
    whatever was previously the default for the same (tenant, base_model)
    first: routing must never have two "default" adapters for the same
    base model at once.

    Gated by the verdict (src/finetuning/verdict.py): the adapter must have
    beaten the base model on the held-out split. A `fail` or `unknown`
    verdict is refused (409) unless an admin passes `force=true`, which is
    written to the audit log. Then the adapter is made servable
    (src/inference/lora_registry.py): manifest for the next restart, live
    load on the serving role now.
    """
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]
    user: User | None = auth[2]
    job = await _get_owned_job(job_id, tenant.id)
    if job.status != "completed" or not job.output_model_path:
        raise HTTPException(
            status_code=409,
            detail="Only a completed job with a real output_model_path can be promoted",
        )
    verdict = compute_verdict(job.metrics)
    if not verdict.passed:
        if not force:
            raise HTTPException(
                status_code=409,
                detail=f"verdict {verdict.status}: {verdict.reason}. An admin may promote anyway with ?force=true.",
            )
        if user is None or user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="force=true requires an admin user linked to this API key")

    async with get_db_context() as db:
        existing_defaults = (
            (
                await db.execute(
                    select(ModelAdapter).where(
                        ModelAdapter.tenant_id == tenant.id,
                        ModelAdapter.base_model_id == job.base_model,
                        ModelAdapter.is_default.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        for adapter in existing_defaults:
            adapter.is_default = False

        existing_for_job = (
            await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_id))
        ).scalar_one_or_none()
        if existing_for_job is not None:
            existing_for_job.status = AdapterStatus.PROMOTED
            existing_for_job.is_default = True
        else:
            db.add(
                ModelAdapter(
                    tenant_id=tenant.id,
                    base_model_id=job.base_model,
                    name=f"tenant-{tenant.id.hex[:8]}-{job.job_type}-{job_id.hex[:8]}",
                    path=job.output_model_path,
                    rank=(job.config or {}).get("lora_r", 64),
                    job_type=job.job_type,
                    job_id=job_id,
                    status=AdapterStatus.PROMOTED,
                    is_default=True,
                    metrics=job.metrics or {},
                )
            )
        promoted = (
            existing_for_job
            or (await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_id))).scalar_one()
        )
        adapter_name, adapter_path = promoted.name, promoted.path
        if not verdict.passed:
            db.add(
                AuditLog(
                    actor=user.email if user else api_key.key_prefix,
                    action="finetune.promote_forced",
                    target_type="finetune_job",
                    target_id=str(job_id),
                    metadata_={"verdict": verdict.to_dict(), "adapter": adapter_name},
                    source_ip=request.client.host if request.client else None,
                )
            )

    serving = await serve_promoted_adapter(adapter_name, adapter_path, job.base_model)
    async with get_db_context() as db:
        refreshed = await db.get(FineTuneJob, job_id)
        if refreshed is not None:
            refreshed.metrics = {**(refreshed.metrics or {}), "serving": serving}
            job = refreshed
    return _to_response(job)


@router.post("/jobs/{job_id}/rollback", response_model=FineTuneJobResponse)
async def rollback_job(job_id: UUID, auth: tuple = Depends(require_scope("finetune"))):
    """Retires this job's adapter — src/inference/model_router.py's
    resolve_served_model_name finds no promoted default anymore and falls
    back to the base model, the real one-click-rollback the plan calls for."""
    tenant: Tenant = auth[1]
    job = await _get_owned_job(job_id, tenant.id)

    async with get_db_context() as db:
        adapter = (await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_id))).scalar_one_or_none()
        if adapter is None:
            raise HTTPException(status_code=404, detail="This job was never promoted to an adapter")
        adapter.status = AdapterStatus.RETIRED
        adapter.is_default = False
    return _to_response(job)


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: UUID, request: Request, auth: tuple = Depends(require_scope("finetune"))):
    """Live SSE progress — the identical replay-then-block contract
    src/api/routes/agents.py's task stream already uses, over
    src/finetuning/events.py's parallel Redis-Streams mechanism."""
    from src.finetuning.events import TERMINAL_STATUSES, block_for_next_finetune_event, read_finetune_events_from

    tenant: Tenant = auth[1]
    await _get_owned_job(job_id, tenant.id)

    async def event_generator():
        last_id = "0-0"
        history = await read_finetune_events_from(job_id, last_id)
        for entry_id, payload in history:
            last_id = entry_id
            yield f"id: {entry_id}\ndata: {json.dumps(payload)}\n\n"
            if payload.get("status") in TERMINAL_STATUSES:
                return

        while True:
            if await request.is_disconnected():
                break
            result = await block_for_next_finetune_event(job_id, last_id, timeout_ms=15_000)
            if result is None:
                yield ": keep-alive\n\n"
                continue
            entry_id, payload = result
            last_id = entry_id
            yield f"id: {entry_id}\ndata: {json.dumps(payload)}\n\n"
            if payload.get("status") in TERMINAL_STATUSES:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
