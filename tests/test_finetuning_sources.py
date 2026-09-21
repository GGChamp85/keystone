# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/finetuning/sources/trajectories.py — a real
Postgres (tests/conftest.py's requires_integration_env), real AgentTask/
TaskFeedback rows, not fixtures shaped by hand to match what the code
happens to expect.

This gives src/finetuning/data_prep.py's prepare_sft_data a real caller
for the first time (grepped before this file existed: zero real callers
anywhere in this codebase). Full run_*_training() bodies (src/finetuning/
lora_train.py, sft_train.py, dpo_train.py) are NOT exercised in this repo's
dev environment — they default to 4-bit quantization via bitsandbytes,
which needs a CUDA GPU and doesn't work on this (Apple Silicon, no GPU)
machine; that's real infra this pass honestly doesn't have, the same kind
of gap tests/test_git_workflow_integration.py already documents for "no
real Gitea here."
"""

from __future__ import annotations

import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import AgentTask, FeedbackVerdict, TaskFeedback, TaskStatus, Tenant, TenantTier
from src.finetuning.sources.trajectories import build_sft_examples_from_trajectories

pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="finetune-sources-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _make_task(tenant_id, *, description, diff, repository_url=None, status=TaskStatus.COMPLETED) -> uuid.UUID:
    task_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            AgentTask(
                id=task_id,
                tenant_id=tenant_id,
                task_description=description,
                repository_url=repository_url,
                status=status,
                output_diff=diff,
            )
        )
        await db.flush()
    return task_id


async def _add_feedback(task_id, verdict: FeedbackVerdict) -> None:
    async with get_db_context() as db:
        db.add(TaskFeedback(task_id=task_id, verdict=verdict))
        await db.flush()


async def test_accepted_task_with_a_real_diff_becomes_an_sft_example(tenant_id):
    task_id = await _make_task(
        tenant_id, description="Add input validation to the signup form", diff="+def validate(x): ..."
    )
    await _add_feedback(task_id, FeedbackVerdict.ACCEPTED)

    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert len(examples) == 1
    assert examples[0]["task"] == "Add input validation to the signup form"
    assert examples[0]["solution"] == "+def validate(x): ..."


async def test_merged_task_is_also_included(tenant_id):
    task_id = await _make_task(tenant_id, description="Fix off-by-one in pagination", diff="+fix")
    await _add_feedback(task_id, FeedbackVerdict.MERGED)

    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert len(examples) == 1


async def test_rejected_task_is_excluded(tenant_id):
    task_id = await _make_task(tenant_id, description="A bad change", diff="+oops")
    await _add_feedback(task_id, FeedbackVerdict.REJECTED)

    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert examples == []


async def test_task_with_no_feedback_at_all_is_excluded(tenant_id):
    await _make_task(tenant_id, description="Never reviewed", diff="+something")
    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert examples == []


async def test_task_with_no_diff_is_excluded_even_if_accepted(tenant_id):
    task_id = await _make_task(tenant_id, description="Accepted but no real diff", diff=None)
    await _add_feedback(task_id, FeedbackVerdict.ACCEPTED)
    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert examples == []


async def test_non_completed_task_is_excluded_even_if_accepted(tenant_id):
    task_id = await _make_task(
        tenant_id, description="Still running somehow", diff="+partial", status=TaskStatus.RUNNING
    )
    await _add_feedback(task_id, FeedbackVerdict.ACCEPTED)
    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert examples == []


async def test_task_with_both_a_rejected_and_a_later_accepted_feedback_is_included_once(tenant_id):
    task_id = await _make_task(tenant_id, description="Flip-flopped verdict", diff="+final")
    await _add_feedback(task_id, FeedbackVerdict.REJECTED)
    await _add_feedback(task_id, FeedbackVerdict.ACCEPTED)

    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert len(examples) == 1  # not once per feedback row


async def test_repository_url_is_carried_through_for_the_manifest_split(tenant_id):
    task_id = await _make_task(
        tenant_id, description="Repo-scoped task", diff="+x", repository_url="https://git/acme/widgets"
    )
    await _add_feedback(task_id, FeedbackVerdict.ACCEPTED)

    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
    assert examples[0]["repository_url"] == "https://git/acme/widgets"
    assert examples[0]["context"] == "Repository: https://git/acme/widgets"


async def test_filters_by_repository_url_when_given(tenant_id):
    task_a = await _make_task(tenant_id, description="In repo A", diff="+a", repository_url="https://git/a")
    task_b = await _make_task(tenant_id, description="In repo B", diff="+b", repository_url="https://git/b")
    await _add_feedback(task_a, FeedbackVerdict.ACCEPTED)
    await _add_feedback(task_b, FeedbackVerdict.ACCEPTED)

    examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id, repository_url="https://git/a")
    assert len(examples) == 1
    assert examples[0]["task"] == "In repo A"


async def test_does_not_leak_another_tenants_accepted_tasks(tenant_id):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tenant_id, name="other", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    try:
        other_task = await _make_task(other_tenant_id, description="Belongs to someone else", diff="+theirs")
        await _add_feedback(other_task, FeedbackVerdict.ACCEPTED)

        examples = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
        assert examples == []
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)
