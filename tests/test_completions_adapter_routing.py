# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration test proving src/api/routes/completions.py's
POST /v1/chat/completions — the direct customer-facing OpenAI-compatible
gateway, the highest-value place adapter routing applies — actually routes
a tenant with a promoted ModelAdapter to it. Real Postgres/Redis, a real
FastAPI app over ASGITransport (same pattern as tests/test_mcp_server.py),
and a scripted fake InferenceClient standing in for the model's own reply
(no real GPU/vLLM endpoint needed to prove the *routing*, only real model
output would need one).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import AdapterStatus, APIKey, ModelAdapter, Tenant, TenantTier
from src.main import create_app

pytestmark = pytest.mark.integration


class _ScriptedInferenceClient:
    model_id = "test-coding-model"

    def __init__(self):
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }


class _ScriptedModelRouter:
    def __init__(self, client):
        self._client = client

    async def get_client(self, model_input):
        return self._client, "coding"


@pytest.fixture
async def tenant_and_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="completions-adapter-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="test-key", key_prefix=prefix, key_hash=key_hash, scopes=["inference"]))
        await db.flush()
    yield tid, full_key
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
            yield client


async def test_chat_completions_routes_to_the_tenants_promoted_adapter(tenant_and_key):
    tenant_id, api_key = tenant_and_key
    async with get_db_context() as db:
        db.add(
            ModelAdapter(
                tenant_id=tenant_id,
                base_model_id="test-coding-model",  # matches _ScriptedInferenceClient.model_id
                name="tenant-completions-lora",
                path="/data/adapters/tenant-completions-lora",
                rank=64,
                job_type="lora",
                status=AdapterStatus.PROMOTED,
                is_default=True,
            )
        )
        await db.flush()

    fake_client = _ScriptedInferenceClient()
    with patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)):
        async with running_client() as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200, resp.text

    assert fake_client.calls[0]["model_override"] == "tenant-completions-lora"


async def test_chat_completions_falls_back_to_base_model_when_no_adapter_promoted(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    with patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)):
        async with running_client() as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200, resp.text

    assert fake_client.calls[0]["model_override"] == "test-coding-model"
