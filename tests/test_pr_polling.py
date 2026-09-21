# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/orchestrator/pr_polling.py (against a real
Postgres, tests/conftest.py's requires_integration_env) and the manual
POST /v1/keystone/tasks/{id}/feedback route (src/api/routes/agents.py).

No real Gitea runs in this dev environment (same gap
tests/test_git_workflow_integration.py already documents and skips
without one), so the git-host side is a scripted fake matching this
codebase's own established pattern for exactly that gap — everything
around it (the DB query, the already-recorded skip, the verdict mapping,
the actual row written) is real, not mocked.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import AgentTask, APIKey, FeedbackVerdict, TaskFeedback, TaskStatus, Tenant, TenantTier
from src.main import create_app
from src.orchestrator import pr_polling

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@dataclass(frozen=True)
class _FakePRStatus:
    number: int
    state: str
    merged: bool
    merge_commit_sha: str | None


class _ScriptedGitHost:
    """Returns a pre-scripted status per (owner, repo, number), or raises
    for a number not in the script — the "git host unreachable/PR gone"
    failure path pr_polling.py must skip past without aborting the pass."""

    def __init__(self, script: dict[int, _FakePRStatus]):
        self._script = script

    async def get_pull_request_status(self, owner: str, repo: str, number: int):
        if number not in self._script:
            raise RuntimeError(f"no such PR #{number} (simulated git host failure)")
        return self._script[number]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="pr-polling-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
async def api_key_id(tenant_id):
    _full_key, prefix, key_hash = generate_api_key()
    kid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(APIKey(id=kid, tenant_id=tenant_id, name="pr-poll-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return kid


async def _completed_task_with_pr(tenant_id, api_key_id, pr_number: int) -> uuid.UUID:
    task_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            AgentTask(
                id=task_id,
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                task_description="A real completed task with a real PR.",
                repository_url="https://git.example.com/acme/widgets",
                status=TaskStatus.COMPLETED,
                pr_number=pr_number,
                pr_url=f"https://git.example.com/acme/widgets/pulls/{pr_number}",
            )
        )
        await db.flush()
    return task_id


async def _feedback_for(task_id) -> list[TaskFeedback]:
    from sqlalchemy import select

    async with get_db_context() as db:
        result = await db.execute(select(TaskFeedback).where(TaskFeedback.task_id == task_id))
        return list(result.scalars().all())


async def test_merged_pr_records_merged_feedback(tenant_id, api_key_id, monkeypatch):
    task_id = await _completed_task_with_pr(tenant_id, api_key_id, pr_number=1)
    host = _ScriptedGitHost({1: _FakePRStatus(1, "closed", True, "abc123")})
    monkeypatch.setattr(pr_polling, "get_git_host", lambda: host)

    written = await pr_polling.poll_and_record_pr_feedback()
    assert any(w["task_id"] == task_id and w["verdict"] == "merged" for w in written)

    rows = await _feedback_for(task_id)
    assert len(rows) == 1
    assert rows[0].verdict == FeedbackVerdict.MERGED
    assert "abc123" in rows[0].reason


async def test_closed_without_merge_records_rejected_feedback(tenant_id, api_key_id, monkeypatch):
    task_id = await _completed_task_with_pr(tenant_id, api_key_id, pr_number=2)
    host = _ScriptedGitHost({2: _FakePRStatus(2, "closed", False, None)})
    monkeypatch.setattr(pr_polling, "get_git_host", lambda: host)

    written = await pr_polling.poll_and_record_pr_feedback()
    assert any(w["task_id"] == task_id and w["verdict"] == "rejected" for w in written)

    rows = await _feedback_for(task_id)
    assert rows[0].verdict == FeedbackVerdict.REJECTED


async def test_still_open_pr_records_nothing(tenant_id, api_key_id, monkeypatch):
    task_id = await _completed_task_with_pr(tenant_id, api_key_id, pr_number=3)
    host = _ScriptedGitHost({3: _FakePRStatus(3, "open", False, None)})
    monkeypatch.setattr(pr_polling, "get_git_host", lambda: host)

    written = await pr_polling.poll_and_record_pr_feedback()
    assert not any(w["task_id"] == task_id for w in written)
    assert await _feedback_for(task_id) == []


async def test_already_recorded_task_is_not_polled_again(tenant_id, api_key_id, monkeypatch):
    task_id = await _completed_task_with_pr(tenant_id, api_key_id, pr_number=4)
    host = _ScriptedGitHost({4: _FakePRStatus(4, "closed", True, "sha1")})
    monkeypatch.setattr(pr_polling, "get_git_host", lambda: host)

    first_pass = await pr_polling.poll_and_record_pr_feedback()
    assert len(first_pass) == 1

    second_pass = await pr_polling.poll_and_record_pr_feedback()
    assert not any(w["task_id"] == task_id for w in second_pass)
    assert len(await _feedback_for(task_id)) == 1  # still exactly one row, not duplicated


async def test_one_tasks_git_host_failure_does_not_block_the_rest(tenant_id, api_key_id, monkeypatch):
    broken_task = await _completed_task_with_pr(tenant_id, api_key_id, pr_number=5)
    ok_task = await _completed_task_with_pr(tenant_id, api_key_id, pr_number=6)
    host = _ScriptedGitHost({6: _FakePRStatus(6, "closed", True, "sha2")})  # 5 is missing -> simulated failure
    monkeypatch.setattr(pr_polling, "get_git_host", lambda: host)

    written = await pr_polling.poll_and_record_pr_feedback()
    assert not any(w["task_id"] == broken_task for w in written)
    assert any(w["task_id"] == ok_task and w["verdict"] == "merged" for w in written)


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
            yield client


async def test_manual_feedback_route_persists_a_real_row(tenant_id, api_key_id):
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="feedback-route-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()

    task_id = await _completed_task_with_pr(tenant_id, None, pr_number=99)

    async with running_client() as client:
        resp = await client.post(
            f"/v1/keystone/tasks/{task_id}/feedback",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"verdict": "accepted", "reason": "Looks good, shipped it manually."},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["verdict"] == "accepted"
        assert body["reason"] == "Looks good, shipped it manually."

    rows = await _feedback_for(task_id)
    assert len(rows) == 1
    assert rows[0].verdict == FeedbackVerdict.ACCEPTED


async def test_manual_feedback_route_404s_for_a_task_in_another_tenant(tenant_id, api_key_id):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant_id, name="other-tenant", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        full_key, prefix, key_hash = generate_api_key()
        async with get_db_context() as db:
            db.add(APIKey(tenant_id=tenant_id, name="wrong-tenant-key", key_prefix=prefix, key_hash=key_hash))
            await db.flush()

        foreign_task_id = await _completed_task_with_pr(other_tenant_id, None, pr_number=100)

        async with running_client() as client:
            resp = await client.post(
                f"/v1/keystone/tasks/{foreign_task_id}/feedback",
                headers={"Authorization": f"Bearer {full_key}"},
                json={"verdict": "rejected"},
            )
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)
