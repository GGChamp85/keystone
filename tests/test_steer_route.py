# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for POST /v1/keystone/tasks/{id}/steer (src/api/routes/agents.py) —
a real Postgres and Redis (tests/conftest.py's requires_integration_env), a real ASGI request
against the real app, no mocking of the route itself.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import AgentTask, APIKey, TaskStatus, Tenant, TenantTier
from src.main import create_app
from src.orchestrator.steering import drain_steering_messages

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="steer-route-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _task(tenant_id, status: TaskStatus) -> uuid.UUID:
    task_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            AgentTask(
                id=task_id,
                tenant_id=tenant_id,
                task_description="A real task for the steer route to act on.",
                status=status,
            )
        )
        await db.flush()
    return task_id


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18081") as client:
            yield client


async def _real_api_key(tenant_id) -> str:
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="steer-route-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return full_key


async def test_steer_a_running_task_queues_the_message_for_real(tenant_id):
    task_id = await _task(tenant_id, TaskStatus.RUNNING)
    full_key = await _real_api_key(tenant_id)

    async with running_client() as client:
        resp = await client.post(
            f"/v1/keystone/tasks/{task_id}/steer",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"message": "also add a test for the empty-input case"},
        )
        assert resp.status_code == 202, resp.text
        assert resp.json() == {"status": "queued", "task_id": str(task_id)}

    assert await drain_steering_messages(task_id) == ["also add a test for the empty-input case"]


async def test_steer_a_pending_task_is_also_allowed(tenant_id):
    task_id = await _task(tenant_id, TaskStatus.PENDING)
    full_key = await _real_api_key(tenant_id)

    async with running_client() as client:
        resp = await client.post(
            f"/v1/keystone/tasks/{task_id}/steer",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"message": "use approach B instead"},
        )
        assert resp.status_code == 202, resp.text

    assert await drain_steering_messages(task_id) == ["use approach B instead"]


async def test_steer_a_completed_task_is_refused_with_409_and_nothing_is_queued(tenant_id):
    task_id = await _task(tenant_id, TaskStatus.COMPLETED)
    full_key = await _real_api_key(tenant_id)

    async with running_client() as client:
        resp = await client.post(
            f"/v1/keystone/tasks/{task_id}/steer",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"message": "too late"},
        )
        assert resp.status_code == 409, resp.text
        assert "completed" in resp.json()["detail"]

    assert await drain_steering_messages(task_id) == []


async def test_steer_a_task_in_another_tenant_404s(tenant_id):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant_id, name="other-tenant", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        task_id = await _task(other_tenant_id, TaskStatus.RUNNING)
        full_key = await _real_api_key(tenant_id)  # a real key, but for the WRONG tenant

        async with running_client() as client:
            resp = await client.post(
                f"/v1/keystone/tasks/{task_id}/steer",
                headers={"Authorization": f"Bearer {full_key}"},
                json={"message": "nope"},
            )
            assert resp.status_code == 404, resp.text
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)


async def test_steer_a_nonexistent_task_404s(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        resp = await client.post(
            f"/v1/keystone/tasks/{uuid.uuid4()}/steer",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"message": "hello"},
        )
        assert resp.status_code == 404, resp.text


async def test_steer_rejects_an_empty_message_with_422(tenant_id):
    task_id = await _task(tenant_id, TaskStatus.RUNNING)
    full_key = await _real_api_key(tenant_id)

    async with running_client() as client:
        resp = await client.post(
            f"/v1/keystone/tasks/{task_id}/steer",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"message": ""},
        )
        assert resp.status_code == 422, resp.text
