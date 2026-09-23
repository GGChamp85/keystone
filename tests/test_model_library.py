# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The Model Library (src/inference/library.py, GET /v1/keystone/models) over the real route with real
Postgres: role state comes from the router's health registry probing real HTTP endpoints (a real
backend process for the coding role, a dead port for the others), this tenant's adapters follow their
base role's state, a running fine-tune marks its base model TRAINING, and the catalog lists what is not
deployed with its VRAM numbers. Requires DATABASE_URL/REDIS_URL.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AdapterStatus, APIKey, FineTuneJob, ModelAdapter, Tenant, TenantTier
from src.inference import client as client_module
from src.inference.health import endpoint_health
from src.main import create_app

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


def _dead_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # closed again on exit: nothing listens there


@pytest.fixture
async def tenant_and_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="model-library-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["inference"]))
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
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_library_reports_live_state_adapters_training_and_catalog(tenant_and_key, fake_vllm_server, monkeypatch):
    tid, key = tenant_and_key
    dead = f"http://127.0.0.1:{_dead_port()}/v1"
    monkeypatch.setenv("VLLM_CODING_URL", fake_vllm_server)
    monkeypatch.setenv("CODING_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")  # a catalog model, so metadata resolves
    monkeypatch.setenv("VLLM_CODING_FALLBACK_URL", dead)
    monkeypatch.setenv("CODING_FALLBACK_MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")
    monkeypatch.setenv("VLLM_REASONING_URL", dead)
    monkeypatch.setenv("INFERENCE_HEALTH_CACHE_SECONDS", "0")
    get_settings.cache_clear()
    endpoint_health.reset()
    await client_module.close_all_clients()

    async with get_db_context() as db:
        db.add(
            ModelAdapter(
                tenant_id=tid,
                base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                name="tenant-lib-lora",
                path="/data/adapters/tenant-lib-lora",
                rank=8,
                job_type="lora",
                status=AdapterStatus.PROMOTED,
                is_default=True,
                metrics={"verdict": {"status": "pass", "reason": "12% better"}},
            )
        )
        db.add(
            ModelAdapter(
                tenant_id=tid,
                base_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                name="tenant-lib-old",
                path="/data/adapters/old",
                rank=8,
                job_type="lora",
                status=AdapterStatus.RETIRED,
                is_default=False,
            )
        )
        db.add(
            FineTuneJob(
                tenant_id=tid,
                base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                job_type="lora",
                status="running",
                config={},
            )
        )
    try:
        async with running_client() as client:
            resp = await client.get("/v1/keystone/models", headers={"Authorization": f"Bearer {key}"})
            assert resp.status_code == 200, resp.text
            body = resp.json()
    finally:
        get_settings.cache_clear()
        endpoint_health.reset()
        await client_module.close_all_clients()

    by_key = {(m["kind"], m["id"]): m for m in body["models"]}

    coding = by_key[("role", "coding")]
    assert coding["state"] == "READY" and coding["provider"] == "local process"
    assert coding["served_model_ids"] == ["test-model"]  # from the endpoint's real /models listing
    assert coding["params_b"] == 7.0
    assert coding["context"] == 4096 and coding["context_source"] == "endpoint"  # the server's max_model_len wins
    fallback_entry = coding["deployment"]
    assert fallback_entry["vram_serve_gb"] == 17.5
    assert coding["api_features"]["tools"] is True and coding["api_features"]["anthropic_messages"] is True
    assert coding["api_features"]["lora_adapters"] is True  # this tenant has an adapter on it
    assert [a["name"] for a in coding["adapters"]] == ["tenant-lib-lora"]

    fallback = by_key[("role", "coding_fallback")]
    assert fallback["state"] == "UNHEALTHY" and fallback["health"]["last_error"]
    assert by_key[("role", "reasoning")]["state"] == "UNHEALTHY"

    adapter = by_key[("adapter", "tenant-lib-lora")]
    assert adapter["state"] == "READY" and adapter["is_default"] and adapter["served_path"] == "coding"
    assert adapter["metrics"]["verdict"]["status"] == "pass"
    retired = by_key[("adapter", "tenant-lib-old")]
    assert retired["state"] == "NOT_DEPLOYED" and retired["adapter_status"] == "retired"

    catalog_ids = {m["id"] for m in body["models"] if m["kind"] == "catalog"}
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" not in catalog_ids  # served by a role, listed there instead
    assert "Qwen/Qwen2.5-Coder-0.5B-Instruct" in catalog_ids
    training = by_key[("catalog", "Qwen/Qwen2.5-Coder-1.5B-Instruct")]
    assert training["state"] == "TRAINING" and training["deployment"]["vram_qlora_gb"] > 0
    assert by_key[("catalog", "Qwen/Qwen2.5-Coder-0.5B-Instruct")]["state"] == "NOT_DEPLOYED"


async def test_library_requires_the_inference_scope(tenant_and_key):
    async with running_client() as client:
        assert (await client.get("/v1/keystone/models")).status_code == 401
