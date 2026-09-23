# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
W4 gateway hardening over the real application with real Postgres/Redis and a real backend process:
legacy (plain SHA-256) keys still authenticate and are re-hashed with the pepper on first use; key
rotation mints a replacement and time-limits the old key, audited; a tenant's monthly dollar budget,
set through the admin limits route, refuses gateway requests and agent tasks with 429 once the priced
ledger reaches it; every response carries X-Request-ID (echoed when the caller sent one) and gateway
responses carry the routing decision; the new metrics appear on /metrics. Requires DATABASE_URL/REDIS_URL.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import select

from src.api.middleware.auth import generate_api_key, legacy_hash_key
from src.billing.ledger import record_usage
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import APIKey, APIKeyStatus, AuditLog, Tenant, TenantTier, UsageRecord
from src.inference.client import InferenceClient
from src.main import create_app

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


class _Router:
    def __init__(self, base_url: str):
        self.client = InferenceClient(base_url=base_url, model_id="test-model")

    async def get_client(self, model_input):
        return self.client, "coding"


@pytest.fixture
async def tenant():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="hardening", email=f"{tid}@test.dev", tier=TenantTier.FREE))
    yield tid
    async with get_db_context() as db:
        # the ledger rows first: deleting a key sets their api_key_id to NULL, which would collide with
        # the tenant's NULL-keyed bucket for the same hour (uq_usage_bucket) — no route hard-deletes keys
        # or tenants, so this ordering only matters to a test's cleanup
        for rec in (await db.execute(select(UsageRecord).where(UsageRecord.tenant_id == tid))).scalars().all():
            await db.delete(rec)
        await db.flush()
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
def admin_token(monkeypatch):
    token = secrets.token_hex(32)
    monkeypatch.setenv("KEYSTONE_ROOT_ADMIN_TOKEN", token)
    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 1000.0}')  # $1 per 1,000 tokens: budgets bite fast
    get_settings.cache_clear()
    yield token
    get_settings.cache_clear()


@asynccontextmanager
async def running_client(backend_url: str) -> AsyncIterator[httpx.AsyncClient]:
    router = _Router(backend_url)
    app = create_app()
    with patch("src.api.routes._inference_common.get_model_router", return_value=router):
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
    await router.client.close()


async def _chat(client: httpx.AsyncClient, headers: dict, content: str, model: str = "coding", **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return await client.post("/v1/chat/completions", headers=headers, json=body)


async def _mint(tenant_id, *, legacy: bool = False, scopes=("inference", "agent")) -> tuple[str, str]:
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(
            APIKey(
                tenant_id=tenant_id,
                name="k",
                key_prefix=prefix,
                key_hash=legacy_hash_key(full_key) if legacy else key_hash,
                scopes=list(scopes),
            )
        )
    return full_key, prefix


async def test_a_legacy_sha256_key_still_works_and_is_rehashed_with_the_pepper(tenant, admin_token, fake_vllm_server):
    key, prefix = await _mint(tenant, legacy=True)
    async with running_client(fake_vllm_server) as client:
        resp = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 200, resp.text
    async with get_db_context() as db:
        row = (await db.execute(select(APIKey).where(APIKey.key_prefix == prefix))).scalar_one()
    from src.api.middleware.auth import hash_key

    assert row.key_hash == hash_key(key) and row.key_hash != legacy_hash_key(key)


async def test_rotation_mints_a_replacement_and_time_limits_the_old_key(tenant, admin_token, fake_vllm_server):
    old_key, old_prefix = await _mint(tenant)
    async with running_client(fake_vllm_server) as client:
        admin = {"Authorization": f"Bearer {admin_token}"}
        rotated = await client.post(f"/v1/admin/tenants/{tenant}/keys/{old_prefix}/rotate", headers=admin, json={})
        assert rotated.status_code == 201, rotated.text
        new_key = rotated.json()["key"]
        assert new_key.startswith("ks-") and new_key != old_key
        assert rotated.json()["scopes"] == ["inference", "agent"]
        # both keys work during the grace window
        for k in (old_key, new_key):
            assert (await client.get("/v1/models", headers={"Authorization": f"Bearer {k}"})).status_code == 200
        # an immediate rotation revokes the old one
        second = await client.post(
            f"/v1/admin/tenants/{tenant}/keys/{rotated.json()['key_prefix']}/rotate",
            headers=admin,
            json={"grace_hours": 0},
        )
        assert second.status_code == 201
        assert (await client.get("/v1/models", headers={"Authorization": f"Bearer {new_key}"})).status_code == 401
        unknown = await client.post(f"/v1/admin/tenants/{tenant}/keys/ks-nope0000/rotate", headers=admin, json={})
        assert unknown.status_code == 404
    async with get_db_context() as db:
        old = (await db.execute(select(APIKey).where(APIKey.key_prefix == old_prefix))).scalar_one()
        assert old.status == APIKeyStatus.ACTIVE and old.expires_at is not None
        assert old.expires_at <= datetime.now(UTC) + timedelta(hours=24, minutes=1)
        audits = (await db.execute(select(AuditLog).where(AuditLog.action == "api_key.rotate"))).scalars().all()
        assert any(a.metadata_ and a.metadata_.get("old_prefix") == old_prefix for a in audits)


async def test_monthly_dollar_budget_refuses_gateway_and_agent_once_spent(tenant, admin_token, fake_vllm_server):
    key, _ = await _mint(tenant)
    user = {"Authorization": f"Bearer {key}"}
    async with running_client(fake_vllm_server) as client:
        admin = {"Authorization": f"Bearer {admin_token}"}
        limits_url = f"/v1/admin/tenants/{tenant}/limits"
        limits = await client.post(limits_url, headers=admin, json={"monthly_budget_usd": 0.05})
        assert limits.status_code == 200 and limits.json()["monthly_budget_usd"] == 0.05
        empty = await client.post(limits_url, headers=admin, json={})
        assert empty.status_code == 422

        ok = await client.post(
            "/v1/chat/completions",
            headers=user,
            json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert ok.status_code == 200, ok.text
        assert ok.headers["x-vs-route-decision"].startswith("requested=coding; served=coding;")

        # spend past the budget through the real ledger ($1 per 1,000 tokens)
        await record_usage(tenant, model_role="coding", prompt_tokens=60, completion_tokens=0)
        refused = await client.post(
            "/v1/chat/completions",
            headers=user,
            json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert refused.status_code == 429, refused.text
        assert "Monthly budget exhausted" in refused.json()["detail"]
        assert float(refused.headers["x-vs-budget-limit-usd"]) == 0.05
        assert float(refused.headers["x-vs-budget-spent-usd"]) >= 0.05

        task = await client.post(
            "/v1/keystone/tasks",
            headers=user,
            json={"task": "Add a docstring to the main module of this service", "model": "coding"},
        )
        assert task.status_code == 429 and "budget" in task.json()["detail"].lower()

        # lifting the budget (0 = none) admits again
        lifted = await client.post(limits_url, headers=admin, json={"monthly_budget_usd": 0})
        assert lifted.status_code == 200
        again = await client.post(
            "/v1/chat/completions",
            headers=user,
            json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert again.status_code == 200


async def test_request_ids_route_decisions_and_metrics(tenant, admin_token, fake_vllm_server):
    key, _ = await _mint(tenant)
    user = {"Authorization": f"Bearer {key}"}
    async with running_client(fake_vllm_server) as client:
        echoed = await client.get("/v1/models", headers={**user, "X-Request-ID": "trace-abc-123"})
        assert echoed.headers["x-request-id"] == "trace-abc-123"
        generated = await client.get("/v1/models", headers=user)
        assert len(generated.headers["x-request-id"]) == 32

        streamed = await _chat(client, user, "hi", stream=True)
        assert streamed.status_code == 200 and "served=coding" in streamed.headers["x-vs-route-decision"]

        metrics = (await client.get("/metrics/")).text
    assert "keystone_time_to_first_token_seconds_bucket" in metrics
    assert 'keystone_route_decisions_total{reason="primary",requested="coding",served="coding"}' in metrics
    assert "keystone_upstream_errors_total" in metrics


async def test_cost_routing_sends_a_simple_auto_request_to_the_fallback_and_says_so(
    tenant, admin_token, fake_vllm_server, monkeypatch
):
    monkeypatch.setenv("GATEWAY_COST_ROUTING", "1")
    get_settings.cache_clear()
    key, _ = await _mint(tenant)
    user = {"Authorization": f"Bearer {key}"}

    class _RecordingRouter(_Router):
        def __init__(self, base_url):
            super().__init__(base_url)
            self.asked: list[str] = []

        async def get_client(self, model_input):
            self.asked.append(model_input)
            return self.client, "coding_fallback" if model_input == "coding_fallback" else "coding"

    router = _RecordingRouter(fake_vllm_server)
    app = create_app()
    try:
        with patch("src.api.routes._inference_common.get_model_router", return_value=router):
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    simple = await _chat(client, user, "Fix a typo in the README", model="auto")
                    assert simple.status_code == 200, simple.text
                    assert (
                        "served=coding_fallback; model=test-model; reason=cost_routing"
                        in simple.headers["x-vs-route-decision"]
                    )
                    hard = "Redesign the authentication architecture across services with migrations"
                    complex_ = await _chat(client, user, hard, model="auto")
                    assert "served=coding;" in complex_.headers["x-vs-route-decision"]
    finally:
        get_settings.cache_clear()
        await router.client.close()
    assert router.asked == ["coding_fallback", "coding"]
