# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/finetuning/runner.py — real Postgres
(tests/conftest.py's requires_integration_env), real Redis event stream,
real FineTuneJob rows, and the real `_build_config` filtering logic
against the real trainer Config dataclasses. Uses an injectable async
runner (the same pluggable-policy seam src/rl/rollout.py's PolicyFn
already established) standing in for a real training call — the
orchestration mechanics here (status transitions, event publishing, error
handling) are real and fully exercised; only the *actual* training call
(src/finetuning/{lora,sft,dpo}_train.py's real trainer.train()) needs a
real GPU this dev environment doesn't have.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import FineTuneJob, Tenant, TenantTier
from src.finetuning.events import read_finetune_events_from
from src.finetuning.lora_train import LoRATrainingConfig
from src.finetuning.runner import _build_config, run_finetune_job, start_finetune_job

pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="finetune-runner-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
async def job_id(tenant_id):
    jid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            FineTuneJob(
                id=jid,
                tenant_id=tenant_id,
                base_model="Qwen/Qwen2.5-Coder-0.5B",
                job_type="lora",
                status="pending",
                config={"training_data": "/data/train.jsonl", "num_epochs": 1, "unknown_field": "ignored"},
            )
        )
        await db.flush()
    return jid


# ── _build_config ────────────────────────────────────────────


def test_build_config_keeps_known_fields_and_sets_base_model():
    config = _build_config(LoRATrainingConfig, "Qwen/Qwen2.5-Coder-0.5B", {"num_epochs": 1, "lora_r": 16})
    assert config.base_model == "Qwen/Qwen2.5-Coder-0.5B"
    assert config.num_epochs == 1
    assert config.lora_r == 16


def test_build_config_silently_drops_unknown_keys():
    """A job's config JSONB is user/API-submitted — an unrecognized key must
    not crash deep inside a long-running job; src/api/routes/finetune.py is
    where that should be caught, at submission time."""
    config = _build_config(LoRATrainingConfig, "Qwen/Qwen2.5-Coder-0.5B", {"totally_made_up_field": "x"})
    assert not hasattr(config, "totally_made_up_field")
    assert config.base_model == "Qwen/Qwen2.5-Coder-0.5B"


# ── run_finetune_job ─────────────────────────────────────────


async def test_successful_job_transitions_to_completed_with_real_metrics(job_id):
    async def fake_runner(snapshot):
        assert snapshot["job_type"] == "lora"
        assert snapshot["config"]["training_data"] == "/data/train.jsonl"
        return {"train_loss": 0.42, "eval_loss": 0.5, "adapter_path": "/data/adapters/test-adapter"}

    result = await run_finetune_job(job_id, runner=fake_runner)
    assert result["status"] == "completed"

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        assert job.status == "completed"
        assert job.metrics["train_loss"] == 0.42
        assert job.output_model_path == "/data/adapters/test-adapter"
        assert job.started_at is not None
        assert job.completed_at is not None


async def test_failed_runner_leaves_the_job_in_a_real_failed_state_not_stuck_running(job_id):
    async def failing_runner(snapshot):
        raise RuntimeError("simulated OOM during training")

    result = await run_finetune_job(job_id, runner=failing_runner)
    assert result["status"] == "failed"

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        assert job.status == "failed"
        assert "simulated OOM" in job.error_message
        assert job.completed_at is not None


async def test_missing_job_id_returns_failed_without_raising():
    result = await run_finetune_job(uuid.uuid4(), runner=lambda snapshot: {})
    assert result["status"] == "failed"


async def test_start_finetune_job_really_runs_in_the_background_via_the_asyncio_fallback(tenant_id):
    """No Temporal server is reachable in this dev environment, so
    start_finetune_job must take the real asyncio-fallback path — proven
    here by using an unsupported job_type (no torch/GPU needed to reach a
    real, fast failure) and polling the real DB row until the background
    task has actually run and updated it, not just that submission
    returned without raising."""
    jid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            FineTuneJob(
                id=jid,
                tenant_id=tenant_id,
                base_model="some/model",
                job_type="not-a-real-job-type",
                status="pending",
                config={},
            )
        )
        await db.flush()

    await start_finetune_job(jid)

    for _ in range(50):
        async with get_db_context() as db:
            job = await db.get(FineTuneJob, jid)
            if job.status != "pending":
                break
        await asyncio.sleep(0.05)

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, jid)
        assert job.status == "failed"
        assert "not-a-real-job-type" in job.error_message


async def test_real_events_are_published_to_the_real_redis_stream(job_id):
    async def fake_runner(snapshot):
        return {"train_loss": 0.1, "adapter_path": "/data/adapters/x"}

    await run_finetune_job(job_id, runner=fake_runner)

    events = await read_finetune_events_from(job_id)
    statuses = [payload["status"] for _entry_id, payload in events]
    assert statuses == ["running", "completed"]
    assert events[-1][1]["metrics"]["train_loss"] == 0.1
