# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Temporal workflow for a fine-tuning job.

Durable execution wrapper for src/finetuning/runner.py's job lifecycle —
if the worker process crashes mid-training, Temporal retries the single
`run_finetune_job_activity` rather than the job silently vanishing (the
same problem src/temporal/workflows.py already solved for coding tasks).
A real training run is realistically hours long, not the ~30 minutes a
coding task's own circuit breaker budgets for — the timeout here is set
accordingly, and retries are capped at 1 (a fine-tuning job re-running
from scratch after a crash is expensive in a way retrying a failed
coding-agent step isn't; a human re-submitting a fresh job is the right
recovery path, not an automatic silent restart of a multi-hour run).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.temporal.finetune_activities import run_finetune_job_activity

DEFAULT_MAX_TRAINING_HOURS = 12


@workflow.defn
class FineTuneJobWorkflow:
    """One activity (`run_finetune_job_activity`) runs a job's whole
    lifecycle; the activity's own module (src/finetuning/runner.py)
    already owns DB-write-based state so a retry re-reads the real job
    row rather than trusting in-memory state carried across the retry."""

    @workflow.run
    async def run(self, job_id: str) -> dict:
        retry_policy = RetryPolicy(
            maximum_attempts=1,  # see module docstring — no automatic re-run of an expensive training job
            non_retryable_error_types=["CancelledError"],
        )
        return await workflow.execute_activity(
            run_finetune_job_activity,
            args=[job_id],
            start_to_close_timeout=timedelta(hours=DEFAULT_MAX_TRAINING_HOURS),
            retry_policy=retry_policy,
        )
