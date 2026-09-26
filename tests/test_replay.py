# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for deterministic task replay (src/orchestrator/replay.py) and the
GET /v1/keystone/tasks/{task_id}/replay route — against a real Postgres, a real ASGI request, no
mocking of the flattening logic or the route itself. Also covers the tenant-ownership check now
enforced on GET /v1/keystone/tasks/{task_id} and its /stream sibling (a pre-existing gap: neither
route checked that the task actually belonged to the authenticated tenant).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import AgentTask, APIKey, TaskStatus, Tenant, TenantTier
from src.main import create_app
from src.orchestrator.replay import build_task_replay

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]

_SAMPLE_TRACE = [
    {
        "iteration": 0,
        "phase": "coding",
        "model_role": "coding",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "files_changed": ["a.py"],
        "test_results": [],
        "review_comments": [],
        "quality_findings": [],
        "steps": [
            {"event_type": "tool_call", "node": "coding", "phase": "coding", "name": "read_file", "timestamp": 1.0},
            {
                "event_type": "tool_result",
                "node": "coding",
                "phase": "coding",
                "ok": True,
                "output": "content",
                "timestamp": 2.0,
            },
        ],
        "error": None,
        "duration_ms": 100,
    },
    {
        "iteration": 1,
        "phase": "testing",
        "model_role": "coding",
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "files_changed": [],
        "test_results": [],
        "review_comments": [],
        "quality_findings": [],
        "steps": [
            {
                "event_type": "test_output",
                "node": "testing",
                "phase": "testing",
                "passed": True,
                "timestamp": 3.0,
            }
        ],
        "error": None,
        "duration_ms": 50,
    },
]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="replay-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _real_api_key(tenant_id) -> str:
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="replay-test-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return full_key


async def _completed_task_with_trace(tenant_id, *, api_key_id=None) -> uuid.UUID:
    task_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            AgentTask(
                id=task_id,
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                task_description="A real task with a real recorded trace.",
                status=TaskStatus.COMPLETED,
                execution_trace=_SAMPLE_TRACE,
                result_summary="Fixed the bug.",
                branch_name="fix/bug",
                commit_sha="abc123",
                pr_url="https://git.example.com/acme/widgets/pulls/9",
                pr_number=9,
            )
        )
        await db.flush()
    return task_id


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18084") as client:
            yield client


def _parse_sse(text: str) -> list[dict]:
    events: list[dict] = []
    for block in text.strip().split("\n\n"):
        events.extend(json.loads(line[len("data: ") :]) for line in block.splitlines() if line.startswith("data: "))
    return events


async def test_build_task_replay_flattens_steps_in_order_and_appends_final_event(tenant_id):
    task_id = await _completed_task_with_trace(tenant_id)
    replay = await build_task_replay(task_id)
    assert replay is not None
    assert replay.tenant_id == tenant_id

    event_types = [e["event_type"] for e in replay.events]
    assert event_types == ["tool_call", "tool_result", "test_output", "node"]

    final = replay.events[-1]
    assert final["final"] is True
    assert final["phase"] == "completed"
    assert final["result_summary"] == "Fixed the bug."
    assert final["branch_name"] == "fix/bug"
    assert final["commit_sha"] == "abc123"
    assert final["pr_url"] == "https://git.example.com/acme/widgets/pulls/9"
    assert final["pr_number"] == 9


async def test_build_task_replay_returns_none_for_an_unknown_task():
    assert await build_task_replay(uuid.uuid4()) is None


async def test_replay_route_reproduces_the_exact_recorded_event_sequence(tenant_id):
    full_key = await _real_api_key(tenant_id)
    task_id = await _completed_task_with_trace(tenant_id)

    async with running_client() as client:
        resp = await client.get(f"/v1/keystone/tasks/{task_id}/replay", headers={"Authorization": f"Bearer {full_key}"})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

    assert [e["event_type"] for e in events] == ["tool_call", "tool_result", "test_output", "node"]
    assert events[-1]["final"] is True


async def test_replay_route_404s_for_an_unknown_task(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        resp = await client.get(
            f"/v1/keystone/tasks/{uuid.uuid4()}/replay", headers={"Authorization": f"Bearer {full_key}"}
        )
        assert resp.status_code == 404


async def test_replay_route_404s_for_another_tenants_task(tenant_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant, name="replay-other-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        other_key = await _real_api_key(other_tenant)
        foreign_task_id = await _completed_task_with_trace(tenant_id)

        async with running_client() as client:
            resp = await client.get(
                f"/v1/keystone/tasks/{foreign_task_id}/replay", headers={"Authorization": f"Bearer {other_key}"}
            )
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


async def test_get_task_404s_for_another_tenants_task(tenant_id):
    """The pre-existing GET /tasks/{id} route never checked tenant ownership before this feature
    added the check (mirroring src/orchestrator/replay.py's own tenant check) — any valid
    agent-scoped API key from any tenant could read any other tenant's task description, diffs
    and tool outputs by task id alone."""
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(
                id=other_tenant, name="get-task-other-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE
            )
        )
        await db.flush()
    try:
        other_key = await _real_api_key(other_tenant)
        foreign_task_id = await _completed_task_with_trace(tenant_id)

        async with running_client() as client:
            resp = await client.get(
                f"/v1/keystone/tasks/{foreign_task_id}", headers={"Authorization": f"Bearer {other_key}"}
            )
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


async def test_get_task_200s_for_the_owning_tenant(tenant_id):
    full_key = await _real_api_key(tenant_id)
    task_id = await _completed_task_with_trace(tenant_id)
    async with running_client() as client:
        resp = await client.get(f"/v1/keystone/tasks/{task_id}", headers={"Authorization": f"Bearer {full_key}"})
        assert resp.status_code == 200
        assert resp.json()["id"] == str(task_id)


async def test_stream_task_404s_for_another_tenants_task(tenant_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant, name="stream-other-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        other_key = await _real_api_key(other_tenant)
        foreign_task_id = await _completed_task_with_trace(tenant_id)

        async with running_client() as client:
            resp = await client.get(
                f"/v1/keystone/tasks/{foreign_task_id}/stream", headers={"Authorization": f"Bearer {other_key}"}
            )
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)
