# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — fine-tuning job runner.

Orchestrates a FineTuneJob's lifecycle (pending -> running ->
completed/failed) the same way src/orchestrator/engine.py orchestrates an
AgentTask's — Temporal-durable when a Temporal server is reachable, a
local asyncio fallback otherwise (src/temporal/finetune_workflow.py /
finetune_activities.py wrap this exact function as the activity body) —
publishing progress to src/finetuning/events.py's identical Redis-Streams
mechanism so a job's live UI experience matches a coding task's.

The actual training call is injectable (`FineTuneRunner`) — the same
pluggable-policy seam src/rl/rollout.py's `PolicyFn` already established:
real orchestration mechanics (status transitions, event publishing, error
handling, config-building from a job's stored JSONB) are fully testable
without a GPU; only `default_finetune_runner`'s real dispatch needs one.
Where the trainer runs is `Settings.finetune_backend`
(src/finetuning/backends/): in this process, on a KubeRay cluster, or on
a rented RunPod pod.

An export (`run_finetune_export`) goes through the same mechanism: a
Temporal workflow when reachable, an asyncio task otherwise, the outcome
recorded on the job's metrics under `exports.<format>`.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import FineTuneJob
from src.finetuning.backends import FineTuneRunner, select_runner
from src.finetuning.backends.inprocess import _JOB_TYPE_CONFIGS, FineTuneJobSnapshot, _build_config
from src.finetuning.events import publish_finetune_event
from src.finetuning.verdict import compute_verdict

__all__ = [
    "_JOB_TYPE_CONFIGS",
    "FineTuneJobSnapshot",
    "FineTuneRunner",
    "_build_config",
    "default_finetune_runner",
    "run_finetune_export",
    "run_finetune_job",
    "start_finetune_export",
    "start_finetune_job",
]

logger = structlog.get_logger(__name__)

_background_jobs: set[asyncio.Task] = set()  # keeps the asyncio-fallback task alive — same pattern engine.py uses


async def default_finetune_runner(job: FineTuneJobSnapshot) -> dict[str, Any]:
    """Runs the job on the configured backend (`FINETUNE_BACKEND`)."""
    runner = select_runner(get_settings().finetune_backend)
    return await runner(job)


async def _load_snapshot(job_id: UUID) -> FineTuneJobSnapshot | None:
    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        if job is None:
            return None
        return {
            "id": job.id,
            "tenant_id": job.tenant_id,
            "base_model": job.base_model,
            "job_type": job.job_type,
            "config": job.config or {},
        }


async def run_finetune_job(job_id: UUID, runner: FineTuneRunner | None = None) -> dict[str, Any]:
    """
    The real orchestration: loads the job, marks it running, calls
    `runner` (defaults to `default_finetune_runner`), persists the result
    or failure, and publishes a progress event at every real state
    transition. Never raises past this function — a failure is recorded
    on the job row and published as a `failed` event, matching
    engine.py's own fail-closed-but-don't-crash-the-caller convention (a
    Temporal activity or an asyncio-fallback background task both need
    the same "always leave the job row in a terminal, truthful state"
    guarantee, not an unhandled exception either can react to
    differently).
    """
    if runner is None:
        runner = default_finetune_runner

    snapshot = await _load_snapshot(job_id)
    if snapshot is None:
        logger.error("finetune.job_not_found", job_id=str(job_id))
        return {"status": "failed", "error": "job not found"}

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        if job is not None:
            job.status = "running"
            job.started_at = datetime.now(UTC)
    await publish_finetune_event(job_id, "running", "Training started")
    logger.info("finetune.job_started", job_id=str(job_id), job_type=snapshot["job_type"])

    try:
        metrics = await runner(snapshot)
    except Exception as exc:
        async with get_db_context() as db:
            job = await db.get(FineTuneJob, job_id)
            if job is not None:
                job.status = "failed"
                job.error_message = str(exc)
                job.completed_at = datetime.now(UTC)
        await publish_finetune_event(job_id, "failed", f"Training failed: {exc}")
        logger.error("finetune.job_failed", job_id=str(job_id), error=str(exc))
        return {"status": "failed", "error": str(exc)}

    metrics = {**metrics, "verdict": compute_verdict(metrics).to_dict()}
    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        if job is not None:
            job.status = "completed"
            job.metrics = metrics
            job.output_model_path = metrics.get("adapter_path")
            job.completed_at = datetime.now(UTC)
    await publish_finetune_event(job_id, "completed", "Training completed", metrics=metrics)
    loggable_metrics = {k: v for k, v in metrics.items() if k != "adapter_path"}
    logger.info("finetune.job_completed", job_id=str(job_id), **loggable_metrics)
    return {"status": "completed", "metrics": metrics}


# ── exports ──────────────────────────────────────────────────


def export_root_for(output_model_path: str) -> str:
    """Exports live next to the adapter: the trainer writes `<output_dir>/adapter`, so
    `<output_dir>/export/` holds `merged/`, the `.gguf` file and `awq/`."""
    return os.path.join(os.path.dirname(output_model_path.rstrip("/")), "export")


async def _record_export(job_id: UUID, fmt: str, entry: dict[str, Any]) -> None:
    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        if job is None:
            return
        metrics = dict(job.metrics or {})
        exports = dict(metrics.get("exports") or {})
        exports[fmt] = {**(exports.get(fmt) or {}), **entry}
        metrics["exports"] = exports
        job.metrics = metrics  # a new dict, so SQLAlchemy sees the JSONB change


async def run_finetune_export(job_id: UUID, fmt: str, quant: str | None = None) -> dict[str, Any]:
    """Runs one export of a completed job's adapter (src/finetuning/export.py) and records the artifact —
    path, size, quant — or the failure under the job's `metrics.exports.<format>`. Never raises past this
    function, for the same reason run_finetune_job never does."""
    from src.finetuning.export import run_export

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        if job is None:
            return {"status": "failed", "error": "job not found"}
        if job.status != "completed" or not job.output_model_path:
            return {"status": "failed", "error": "only a completed job with an adapter on disk can be exported"}
        base_model, adapter_dir = job.base_model, job.output_model_path

    export_root = export_root_for(adapter_dir)
    started: dict[str, Any] = {"status": "running", "quant": quant, "started_at": datetime.now(UTC).isoformat()}
    await _record_export(job_id, fmt, started)
    await publish_finetune_event(job_id, "export_started", f"Export ({fmt}) started", metrics={"format": fmt})
    settings = get_settings()
    loop = asyncio.get_running_loop()
    try:
        artifact = await loop.run_in_executor(
            None,
            lambda: run_export(
                fmt,
                base_model=base_model,
                adapter_dir=adapter_dir,
                export_root=export_root,
                quant=quant,
                hf_token=settings.hf_token,
            ),
        )
    except Exception as exc:
        failed: dict[str, Any] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_at": datetime.now(UTC).isoformat(),
        }
        await _record_export(job_id, fmt, failed)
        await publish_finetune_event(job_id, "export_failed", f"Export ({fmt}) failed: {exc}", metrics={"format": fmt})
        logger.error("finetune.export_failed", job_id=str(job_id), format=fmt, error=str(exc))
        return {"status": "failed", "error": str(exc)}

    entry: dict[str, Any] = {
        "status": "completed",
        "path": artifact.path,
        "size_bytes": artifact.size_bytes,
        "quant": artifact.quant,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    await _record_export(job_id, fmt, entry)
    await publish_finetune_event(
        job_id, "export_completed", f"Export ({fmt}) written to {artifact.path}", metrics=entry
    )
    logger.info("finetune.export_completed", job_id=str(job_id), **entry)
    return {"status": "completed", "export": entry}


# ── starting work: Temporal when reachable, asyncio otherwise ─


_temporal_client = None
_temporal_unavailable = False  # sticky, matching engine.py's own connect-once-then-remember convention


async def _get_temporal_client():
    global _temporal_client, _temporal_unavailable
    if _temporal_unavailable:
        return None
    if _temporal_client is None:
        from temporalio.client import Client

        settings = get_settings()
        try:
            _temporal_client = await Client.connect(settings.temporal_host, namespace=settings.temporal_namespace)
        except Exception as exc:
            logger.warning("finetune.temporal_unreachable", error=str(exc))
            _temporal_unavailable = True
            return None
    return _temporal_client


def _spawn_fallback(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_jobs.add(task)
    task.add_done_callback(_background_jobs.discard)


async def start_finetune_job(job_id: UUID) -> None:
    """
    Starts a submitted job's execution in the background — returns
    immediately, same "202 Accepted, poll/stream for progress" contract
    src/orchestrator/engine.py's submit_task already established for
    coding tasks. Temporal-durable when a Temporal server is reachable
    (src/temporal/finetune_workflow.py), a local asyncio fallback
    otherwise — losing the job on a process restart in that fallback
    case, the same honestly-logged trade-off engine.py's own fallback
    makes, not silently pretended away.
    """
    settings = get_settings()
    client = await _get_temporal_client()

    if client is not None:
        from src.temporal.finetune_workflow import FineTuneJobWorkflow

        workflow_id = f"keystone-finetune-{job_id}"
        await client.start_workflow(
            FineTuneJobWorkflow.run,
            str(job_id),
            id=workflow_id,
            task_queue=settings.temporal_task_queue,
        )
        logger.info("finetune.job_submitted", job_id=str(job_id), execution="temporal", workflow_id=workflow_id)
        return

    if settings.is_production:
        logger.error(
            "finetune.temporal_unavailable_in_production",
            job_id=str(job_id),
            detail="Falling back to non-durable asyncio execution — job will be LOST on restart",
        )
    _spawn_fallback(run_finetune_job(job_id))
    logger.info("finetune.job_submitted", job_id=str(job_id), execution="asyncio_fallback")


async def start_finetune_export(job_id: UUID, fmt: str, quant: str | None = None) -> str:
    """Starts an export in the background through the same mechanism as start_finetune_job; returns
    "temporal" or "asyncio_fallback" so the route can say which."""
    settings = get_settings()
    client = await _get_temporal_client()

    if client is not None:
        from src.temporal.finetune_workflow import FineTuneExportWorkflow

        workflow_id = f"keystone-finetune-export-{job_id}-{fmt}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        await client.start_workflow(
            FineTuneExportWorkflow.run,
            args=[str(job_id), fmt, quant],
            id=workflow_id,
            task_queue=settings.temporal_task_queue,
        )
        logger.info("finetune.export_submitted", job_id=str(job_id), format=fmt, execution="temporal")
        return "temporal"

    if settings.is_production:
        logger.error(
            "finetune.temporal_unavailable_in_production",
            job_id=str(job_id),
            detail="Falling back to non-durable asyncio execution — the export will be LOST on restart",
        )
    _spawn_fallback(run_finetune_export(job_id, fmt, quant))
    logger.info("finetune.export_submitted", job_id=str(job_id), format=fmt, execution="asyncio_fallback")
    return "asyncio_fallback"
