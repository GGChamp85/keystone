# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for Phase 4's users/roles foundation:
- src/db/models.py's User/UserRole + the 4dba9171375c migration
- src/api/middleware/auth.py's require_auth/require_role now resolving a real user
- src/api/routes/keys.py's user CRUD + key-linking routes
- src/api/routes/memory.py's lead/admin gate on approving a tenant-wide memory
- src/api/routes/agents.py's GET /tasks team list route

Against real Postgres/Redis (tests/conftest.py's requires_integration_env),
via the same real-app-over-ASGITransport pattern as tests/test_mcp_server.py
(a running app + real HTTP round trips, not a mock of the auth layer).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, TenantTier
from src.main import create_app

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]

ROOT_ADMIN_TOKEN = "test-root-admin-token-for-users-and-roles"  # noqa: S105 — a test fixture value, not a real credential


@pytest.fixture(autouse=True)
def _root_admin_token(monkeypatch):
    monkeypatch.setenv("KEYSTONE_ROOT_ADMIN_TOKEN", ROOT_ADMIN_TOKEN)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="users-roles-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
async def agent_key(tenant_id):
    """A real tenant-scoped API key with the 'agent' scope, not yet linked to a user."""
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="test-key", key_prefix=prefix, key_hash=key_hash, scopes=["agent"]))
        await db.flush()
    return full_key


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
            yield client


def _admin_headers() -> dict:
    return {"Authorization": f"Bearer {ROOT_ADMIN_TOKEN}"}


def _key_headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def test_create_user_then_list_returns_it(tenant_id):
    async with running_client() as client:
        resp = await client.post(
            f"/v1/admin/tenants/{tenant_id}/users",
            headers=_admin_headers(),
            json={"name": "Ada Lovelace", "email": f"ada-{tenant_id}@test.dev", "role": "lead"},
        )
        assert resp.status_code == 201, resp.text
        created = resp.json()
        assert created["role"] == "lead"
        assert created["is_active"] is True

        listed = await client.get(f"/v1/admin/tenants/{tenant_id}/users", headers=_admin_headers())
        assert listed.status_code == 200
        assert any(u["id"] == created["id"] for u in listed.json())


async def test_duplicate_email_is_rejected(tenant_id):
    async with running_client() as client:
        body = {"name": "Dup", "email": f"dup-{tenant_id}@test.dev", "role": "developer"}
        first = await client.post(f"/v1/admin/tenants/{tenant_id}/users", headers=_admin_headers(), json=body)
        assert first.status_code == 201
        second = await client.post(f"/v1/admin/tenants/{tenant_id}/users", headers=_admin_headers(), json=body)
        assert second.status_code == 409


async def test_key_without_linked_user_cannot_approve_tenant_scope_memory(tenant_id, agent_key):
    async with running_client() as client:
        proposed = await client.post(
            "/v1/keystone/memory",
            headers=_key_headers(agent_key),
            json={"content": "Team-wide convention.", "kind": "convention", "pinned": False},
        )
        assert proposed.status_code == 201
        memory_id = proposed.json()["id"]
        assert proposed.json()["scope"] == "tenant"

        approve = await client.post(f"/v1/keystone/memory/{memory_id}/approve", headers=_key_headers(agent_key))
        assert approve.status_code == 403
        assert "lead or admin" in approve.json()["detail"]


async def test_developer_role_cannot_approve_tenant_scope_but_lead_can(tenant_id, agent_key):
    async with running_client() as client:
        dev = await client.post(
            f"/v1/admin/tenants/{tenant_id}/users",
            headers=_admin_headers(),
            json={"name": "Dev", "email": f"dev-{tenant_id}@test.dev", "role": "developer"},
        )
        lead = await client.post(
            f"/v1/admin/tenants/{tenant_id}/users",
            headers=_admin_headers(),
            json={"name": "Lead", "email": f"lead-{tenant_id}@test.dev", "role": "lead"},
        )
        assert dev.status_code == 201 and lead.status_code == 201

        key_info = await client.get(f"/v1/admin/tenants/{tenant_id}/keys", headers=_admin_headers())
        key_prefix = "-".join(agent_key.split("-")[:2])
        key_id = next(k["id"] for k in key_info.json() if k["key_prefix"] == key_prefix)

        memory = await client.post(
            "/v1/keystone/memory",
            headers=_key_headers(agent_key),
            json={"content": "Another team-wide convention.", "kind": "convention"},
        )
        memory_id = memory.json()["id"]

        link_dev = await client.post(
            f"/v1/admin/tenants/{tenant_id}/keys/{key_id}/link-user",
            headers=_admin_headers(),
            json={"user_id": dev.json()["id"]},
        )
        assert link_dev.status_code == 200
        assert link_dev.json()["user_id"] == dev.json()["id"]

        still_forbidden = await client.post(f"/v1/keystone/memory/{memory_id}/approve", headers=_key_headers(agent_key))
        assert still_forbidden.status_code == 403

        link_lead = await client.post(
            f"/v1/admin/tenants/{tenant_id}/keys/{key_id}/link-user",
            headers=_admin_headers(),
            json={"user_id": lead.json()["id"]},
        )
        assert link_lead.status_code == 200

        now_allowed = await client.post(f"/v1/keystone/memory/{memory_id}/approve", headers=_key_headers(agent_key))
        assert now_allowed.status_code == 200, now_allowed.text
        assert now_allowed.json()["status"] == "approved"


async def test_repo_scope_memory_approval_does_not_require_a_role(tenant_id, agent_key):
    """Only tenant-wide memories need lead/admin sign-off — a repo-scoped one stays
    approvable by any valid agent-scoped key, linked to a user or not."""
    async with running_client() as client:
        memory = await client.post(
            "/v1/keystone/memory",
            headers=_key_headers(agent_key),
            json={"content": "This repo uses tabs.", "kind": "convention", "repository": "https://git/x/y"},
        )
        assert memory.json()["scope"] == "repo"
        approve = await client.post(
            f"/v1/keystone/memory/{memory.json()['id']}/approve", headers=_key_headers(agent_key)
        )
        assert approve.status_code == 200


async def test_task_list_filters_by_user(tenant_id, agent_key):
    async with running_client() as client:
        user = await client.post(
            f"/v1/admin/tenants/{tenant_id}/users",
            headers=_admin_headers(),
            json={"name": "Bob", "email": f"bob-{tenant_id}@test.dev", "role": "developer"},
        )
        key_info = await client.get(f"/v1/admin/tenants/{tenant_id}/keys", headers=_admin_headers())
        key_id = key_info.json()[0]["id"]
        await client.post(
            f"/v1/admin/tenants/{tenant_id}/keys/{key_id}/link-user",
            headers=_admin_headers(),
            json={"user_id": user.json()["id"]},
        )

        submitted = await client.post(
            "/v1/keystone/tasks",
            headers=_key_headers(agent_key),
            json={"task": "A real task description long enough to pass validation."},
        )
        assert submitted.status_code == 202, submitted.text
        task_id = submitted.json()["task_id"]

        listed = await client.get(
            "/v1/keystone/tasks", headers=_key_headers(agent_key), params={"user_id": user.json()["id"]}
        )
        assert listed.status_code == 200
        assert any(t["id"] == task_id for t in listed.json())

        empty = await client.get(
            "/v1/keystone/tasks", headers=_key_headers(agent_key), params={"user_id": str(uuid.uuid4())}
        )
        assert empty.json() == []
