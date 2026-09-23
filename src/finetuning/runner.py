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
without a GPU; only `default_finetune_runner`'s real dispatch to
src/finetuning/{lora,sft,dpo}_train.py needs one — this dev environment
doesn't have one (Apple Silicon, no CUDA; the trainers default to
bitsandbytes 4-bit quantization, which is GPU-only).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import FineTuneJob
from src.finetuning.events import publish_finetune_event
from src.finetuning.verdict import compute_verdict

logger = structlog.get_logger(__name__)

_background_jobs: set[asyncio.Task] = set()  # keeps the asyncio-fallback task alive — same pattern engine.py uses

# A snapshot of the fields default_finetune_runner/a test runner actually
# need — not the live ORM row (which shouldn't be held open across a
# long-running, potentially blocking training call).
FineTuneJobSnapshot = dict[str, Any]

FineTuneRunner = Callable[[FineTuneJobSnapshot], Awaitable[dict[str, Any]]]

_JOB_TYPE_CONFIGS = {
    "lora": ("src.finetuning.lora_train", "LoRATrainingConfig", "run_lora_training"),
    "sft": ("src.finetuning.sft_train", "SFTTrainingConfig", "run_sft_training"),
    "dpo": ("src.finetuning.dpo_train", "DPOTrainingConfig", "run_dpo_training"),
}


def _build_config(config_cls: type, base_model: str, extra: dict[str, Any]):
    """Builds `config_cls` from the job's stored JSONB `config`, keeping only
    keys that are real fields on that dataclass — a job's config dict is
    user/API-submitted input, not something to trust blindly as exact
    constructor kwargs (an unrecognized key would otherwise raise
    TypeError deep inside a long-running job instead of failing clearly
    at submission time, where src/api/routes/finetune.py validates it)."""
    known_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in extra.items() if k in known_fields}
    return config_cls(base_model=base_model, **filtered)


async def default_finetune_runner(job: FineTuneJobSnapshot) -> dict[str, Any]:
    """Dispatches to the real trainer for `job["job_type"]`. The real
    trainers (run_lora_training/run_sft_training/run_dpo_training) are
    synchronous, blocking calls (they call TRL's/Transformers' own
    trainer.train() directly) — run in a thread executor so this doesn't
    block the event loop the rest of the app (SSE streams, other API
    requests) depends on."""
    job_type = job["job_type"]
    if job_type not in _JOB_TYPE_CONFIGS:
        raise ValueError(f"Unknown fine-tuning job_type: {job_type!r} (expected one of {list(_JOB_TYPE_CONFIGS)})")

    module_name, config_cls_name, run_fn_name = _JOB_TYPE_CONFIGS[job_type]
    import importlib

    module = importlib.import_module(module_name)
    config_cls = getattr(module, config_cls_name)
    run_fn = getattr(module, run_fn_name)

    config = _build_config(config_cls, job["base_model"], job.get("config") or {})
    loop = asyncio.get_running_loop()
    if hasattr(config, "progress"):
        job_id = job["id"]

        def report(payload: dict[str, Any]) -> None:
            # Called from the training thread: hand the event to the loop the API runs on.
            step, total = payload.get("step"), payload.get("total_steps")
            message = f"step {step}/{total}" + (
                f", loss {payload['loss']:.4f}" if payload.get("loss") is not None else ""
            )
            asyncio.run_coroutine_threadsafe(publish_finetune_event(job_id, "running", message, metrics=payload), loop)

        config.progress = report
    return await loop.run_in_executor(None, run_fn, config)


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
    task = asyncio.create_task(run_finetune_job(job_id))
    _background_jobs.add(task)
    task.add_done_callback(_background_jobs.discard)
    logger.info("finetune.job_submitted", job_id=str(job_id), execution="asyncio_fallback")
