# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for the /v1/keystone/schedules routes (src/api/routes/schedules.py) and
the unauthenticated webhook trigger route — a real Postgres, a real ASGI request against the
real app, no mocking of the route itself.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, TenantTier
from src.main import create_app
from src.orchestrator.schedules import get_schedule

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="schedules-route-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _real_api_key(tenant_id) -> str:
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="schedules-route-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return full_key


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18083") as client:
            yield client


async def test_create_list_and_get_a_cron_schedule(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post(
            "/v1/keystone/schedules",
            headers=headers,
            json={
                "name": "Nightly sweep",
                "trigger_type": "cron",
                "cron_expression": "0 2 * * *",
                "task": "A real recurring task description, long enough to pass validation.",
            },
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["trigger_type"] == "cron"
        assert body["cron_expression"] == "0 2 * * *"
        assert body["webhook_url"] is None
        assert body["enabled"] is True
        schedule_id = body["id"]

        listed = await client.get("/v1/keystone/schedules", headers=headers)
        assert listed.status_code == 200
        assert any(s["id"] == schedule_id for s in listed.json())

        fetched = await client.get(f"/v1/keystone/schedules/{schedule_id}", headers=headers)
        assert fetched.status_code == 200
        assert fetched.json()["id"] == schedule_id


async def test_create_cron_schedule_without_expression_returns_422(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        resp = await client.post(
            "/v1/keystone/schedules",
            headers={"Authorization": f"Bearer {full_key}"},
            json={
                "name": "Bad",
                "trigger_type": "cron",
                "task": "A real task description, long enough to pass validation.",
            },
        )
        assert resp.status_code == 422, resp.text


async def test_create_cron_schedule_with_invalid_cron_returns_422(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        resp = await client.post(
            "/v1/keystone/schedules",
            headers={"Authorization": f"Bearer {full_key}"},
            json={
                "name": "Bad",
                "trigger_type": "cron",
                "cron_expression": "not a real cron expression",
                "task": "A real task description, long enough to pass validation.",
            },
        )
        assert resp.status_code == 422, resp.text


async def test_create_webhook_schedule_returns_a_real_webhook_url(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        created = await client.post(
            "/v1/keystone/schedules",
            headers={"Authorization": f"Bearer {full_key}"},
            json={
                "name": "CI trigger",
                "trigger_type": "webhook",
                "task": "A real webhook-triggered task description, long enough to pass validation.",
            },
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["trigger_type"] == "webhook"
        assert body["webhook_url"] is not None
        assert body["webhook_url"].startswith("/v1/keystone/webhooks/schedules/")


async def test_disable_then_enable_a_real_schedule(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post(
            "/v1/keystone/schedules",
            headers=headers,
            json={
                "name": "Test",
                "trigger_type": "webhook",
                "task": "A real task description, long enough to pass validation.",
            },
        )
        schedule_id = created.json()["id"]

        disabled = await client.post(f"/v1/keystone/schedules/{schedule_id}/disable", headers=headers)
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False

        enabled_only = await client.get("/v1/keystone/schedules?enabled=true", headers=headers)
        assert all(s["id"] != schedule_id for s in enabled_only.json())

        re_enabled = await client.post(f"/v1/keystone/schedules/{schedule_id}/enable", headers=headers)
        assert re_enabled.status_code == 200
        assert re_enabled.json()["enabled"] is True


async def test_delete_a_real_schedule(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post(
            "/v1/keystone/schedules",
            headers=headers,
            json={
                "name": "Test",
                "trigger_type": "webhook",
                "task": "A real task description, long enough to pass validation.",
            },
        )
        schedule_id = created.json()["id"]

        deleted = await client.delete(f"/v1/keystone/schedules/{schedule_id}", headers=headers)
        assert deleted.status_code == 204

        gone = await client.get(f"/v1/keystone/schedules/{schedule_id}", headers=headers)
        assert gone.status_code == 404


async def test_a_schedule_in_another_tenant_404s_on_get_and_delete(tenant_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(
                id=other_tenant, name="schedules-other-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE
            )
        )
        await db.flush()
    try:
        other_key = await _real_api_key(other_tenant)
        my_key = await _real_api_key(tenant_id)
        async with running_client() as client:
            created = await client.post(
                "/v1/keystone/schedules",
                headers={"Authorization": f"Bearer {other_key}"},
                json={
                    "name": "Other tenant's schedule",
                    "trigger_type": "webhook",
                    "task": "A real task description, long enough to pass validation.",
                },
            )
            schedule_id = created.json()["id"]

            got = await client.get(
                f"/v1/keystone/schedules/{schedule_id}", headers={"Authorization": f"Bearer {my_key}"}
            )
            assert got.status_code == 404

            deleted = await client.delete(
                f"/v1/keystone/schedules/{schedule_id}", headers={"Authorization": f"Bearer {my_key}"}
            )
            assert deleted.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


async def test_webhook_trigger_submits_a_real_task(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post(
            "/v1/keystone/schedules",
            headers=headers,
            json={
                "name": "CI trigger",
                "trigger_type": "webhook",
                "task": "A real webhook-triggered task description, long enough to pass validation.",
            },
        )
        webhook_url = created.json()["webhook_url"]
        schedule_id = created.json()["id"]

        # The webhook route is deliberately unauthenticated by API key — no Authorization header.
        triggered = await client.post(webhook_url)
        assert triggered.status_code == 202, triggered.text
        assert triggered.json()["status"] == "pending"
        assert triggered.json()["task_id"]

    updated = await get_schedule(uuid.UUID(schedule_id))
    assert updated is not None
    assert updated.last_task_id is not None
    assert str(updated.last_task_id) == triggered.json()["task_id"]


async def test_webhook_trigger_404s_for_an_unknown_token():
    async with running_client() as client:
        resp = await client.post("/v1/keystone/webhooks/schedules/not-a-real-token")
        assert resp.status_code == 404


async def test_webhook_trigger_409s_for_a_disabled_schedule(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post(
            "/v1/keystone/schedules",
            headers=headers,
            json={
                "name": "CI trigger",
                "trigger_type": "webhook",
                "task": "A real task description, long enough to pass validation.",
            },
        )
        webhook_url = created.json()["webhook_url"]
        schedule_id = created.json()["id"]

        await client.post(f"/v1/keystone/schedules/{schedule_id}/disable", headers=headers)

        resp = await client.post(webhook_url)
        assert resp.status_code == 409
