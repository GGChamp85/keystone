# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Temporal activity for a fine-tuning job.

Wraps src/finetuning/runner.py's `run_finetune_job` exactly the way
src/temporal/activities.py's `run_agent_task` wraps `KeystoneEngine.
_execute_task` — one activity, the module it calls already owns its own
DB-write-based crash recovery (a retried activity re-reads the job row
rather than trusting in-memory state), so there's no second activity for
setup/teardown to keep in sync.

No heartbeat during the training call itself: unlike an agent task's
LangGraph loop (which yields control after every node and can heartbeat
from Python in between), the real trainers
(src/finetuning/{lora,sft,dpo}_train.py) run a single, long, blocking
`trainer.train()` call with no callback hook this activity taps into —
heartbeating through that needs real trainer-callback integration (a
TrainerCallback reporting progress back out), which needs a real GPU
training run to build and verify against, not something to fake here.
`start_to_close_timeout` is set generously instead; Temporal still
retries a genuinely crashed worker via that timeout, just coarser than
per-node heartbeat detection would be.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from temporalio import activity

logger = structlog.get_logger(__name__)


@activity.defn
async def run_finetune_job_activity(job_id: str) -> dict[str, Any]:
    from src.finetuning.runner import run_finetune_job

    try:
        return await run_finetune_job(UUID(job_id))
    except Exception as exc:
        # run_finetune_job itself never raises (it records failures onto the
        # job row and returns a {"status": "failed", ...} dict) — this is a
        # defensive backstop for something going wrong even earlier (e.g.
        # UUID(job_id) itself failing), re-raised as-is so Temporal's own
        # retry/failure semantics apply, same convention as run_agent_task.
        logger.error("temporal.finetune_activity_failed", job_id=job_id, error=str(exc))
        raise


@activity.defn
async def run_finetune_export_activity(job_id: str, fmt: str, quant: str | None) -> dict[str, Any]:
    from src.finetuning.runner import run_finetune_export

    try:
        return await run_finetune_export(UUID(job_id), fmt, quant)
    except Exception as exc:
        logger.error("temporal.finetune_export_activity_failed", job_id=job_id, format=fmt, error=str(exc))
        raise
