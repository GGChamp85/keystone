# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for the /v1/keystone/skills routes (src/api/routes/skills.py) — a real
Postgres, a real ASGI request against the real app, no mocking of the route itself.
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

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="skills-route-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _real_api_key(tenant_id) -> str:
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="skills-route-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return full_key


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18082") as client:
            yield client


async def test_create_list_and_get_a_skill(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post(
            "/v1/keystone/skills",
            headers=headers,
            json={
                "name": "PCI checklist",
                "content": "Run the PCI compliance checklist.",
                "trigger_keywords": ["payment"],
            },
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["name"] == "PCI checklist"
        assert body["enabled"] is True
        skill_id = body["id"]

        listed = await client.get("/v1/keystone/skills", headers=headers)
        assert listed.status_code == 200
        assert any(s["id"] == skill_id for s in listed.json())

        fetched = await client.get(f"/v1/keystone/skills/{skill_id}", headers=headers)
        assert fetched.status_code == 200
        assert fetched.json()["id"] == skill_id


async def test_disable_then_enable_a_real_skill(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post("/v1/keystone/skills", headers=headers, json={"name": "Test", "content": "content"})
        skill_id = created.json()["id"]

        disabled = await client.post(f"/v1/keystone/skills/{skill_id}/disable", headers=headers)
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False

        enabled_only = await client.get("/v1/keystone/skills?enabled=true", headers=headers)
        assert all(s["id"] != skill_id for s in enabled_only.json())

        re_enabled = await client.post(f"/v1/keystone/skills/{skill_id}/enable", headers=headers)
        assert re_enabled.status_code == 200
        assert re_enabled.json()["enabled"] is True


async def test_delete_a_real_skill(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        created = await client.post("/v1/keystone/skills", headers=headers, json={"name": "Test", "content": "content"})
        skill_id = created.json()["id"]

        deleted = await client.delete(f"/v1/keystone/skills/{skill_id}", headers=headers)
        assert deleted.status_code == 204

        gone = await client.get(f"/v1/keystone/skills/{skill_id}", headers=headers)
        assert gone.status_code == 404


async def test_match_previews_exactly_what_the_agent_would_see(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        headers = {"Authorization": f"Bearer {full_key}"}
        await client.post(
            "/v1/keystone/skills",
            headers=headers,
            json={"name": "Payments", "content": "Run the PCI checklist.", "trigger_keywords": ["payment"]},
        )

        no_match = await client.post(
            "/v1/keystone/skills/match", headers=headers, json={"task": "Fix a typo in the README"}
        )
        assert no_match.json() == []

        match = await client.post(
            "/v1/keystone/skills/match", headers=headers, json={"task": "Add a new payment method"}
        )
        assert len(match.json()) == 1
        assert match.json()[0]["name"] == "Payments"


async def test_a_skill_in_another_tenant_404s_on_get_and_delete(tenant_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant, name="skills-other-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        other_key = await _real_api_key(other_tenant)
        my_key = await _real_api_key(tenant_id)
        async with running_client() as client:
            created = await client.post(
                "/v1/keystone/skills",
                headers={"Authorization": f"Bearer {other_key}"},
                json={"name": "Other tenant's skill", "content": "content"},
            )
            skill_id = created.json()["id"]

            got = await client.get(f"/v1/keystone/skills/{skill_id}", headers={"Authorization": f"Bearer {my_key}"})
            assert got.status_code == 404

            deleted = await client.delete(
                f"/v1/keystone/skills/{skill_id}", headers={"Authorization": f"Bearer {my_key}"}
            )
            assert deleted.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


async def test_create_rejects_too_many_trigger_keywords(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        resp = await client.post(
            "/v1/keystone/skills",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"name": "Test", "content": "content", "trigger_keywords": [f"kw{i}" for i in range(51)]},
        )
        assert resp.status_code == 422, resp.text
