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

    send_usage_chunk = True  # a backend that honors stream_options.include_usage

    async def complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    async def stream(self, messages, **kwargs):
        """The real SSE line shape InferenceClient.stream() yields, with vLLM's usage-only final chunk."""
        self.calls.append(kwargs)
        for piece in ("hel", "lo ", "there"):
            yield f'data: {{"choices":[{{"index":0,"delta":{{"content":"{piece}"}},"finish_reason":null}}]}}\n\n'
        yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        if kwargs.get("include_usage") and self.send_usage_chunk:
            yield 'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":3,"total_tokens":43}}\n\n'
        yield "data: [DONE]\n\n"


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
        db.add(
            APIKey(tenant_id=tid, name="test-key", key_prefix=prefix, key_hash=key_hash, scopes=["inference", "agent"])
        )
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


async def test_chat_completions_has_no_max_tokens_ceiling_by_default(tenant_and_key):
    """MAX_TOKENS_PER_REQUEST defaults to 0 = no deployment ceiling: the model's own limit is the only one."""
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    with patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)):
        async with running_client() as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "coding", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100_000},
            )
            assert resp.status_code == 200, resp.text
    assert fake_client.calls[0]["max_tokens"] == 100_000


async def test_chat_completions_rejects_max_tokens_above_a_configured_deployment_ceiling(tenant_and_key, monkeypatch):
    """When an operator sets MAX_TOKENS_PER_REQUEST, a request above it is rejected up front — before
    routing, so the model is never called — not silently clamped or passed through."""
    from src.config import get_settings

    monkeypatch.setenv("MAX_TOKENS_PER_REQUEST", "32768")
    get_settings.cache_clear()
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    with patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)):
        async with running_client() as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "coding", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100_000},
            )
            assert resp.status_code == 422, resp.text
            assert "exceeds this deployment's limit of 32768" in resp.text

    assert fake_client.calls == []
    get_settings.cache_clear()


async def _stream_frames(client: httpx.AsyncClient, api_key: str, body: dict) -> list[str]:
    frames: list[str] = []
    async with client.stream(
        "POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=body
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        frames.extend([line[len("data: ") :] async for line in resp.aiter_lines() if line.startswith("data: ")])
    return frames


async def test_streaming_bills_the_backends_real_usage_and_hides_the_usage_chunk_unless_asked(tenant_and_key):
    """The old path billed ~1 token per streamed chunk and never counted the prompt. The backend is
    now always asked for stream_options.include_usage; its usage chunk is what gets recorded."""
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    recorded: list[tuple[int, str]] = []

    async def fake_record(tenant_id, tokens, model_role="unknown"):
        recorded.append((tokens, model_role))
        return {}

    with (
        patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)),
        patch("src.api.routes.completions.record_token_usage", fake_record),
    ):
        async with running_client() as client:
            frames = await _stream_frames(
                client, api_key, {"model": "coding", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            )
    assert fake_client.calls[0]["include_usage"] is True  # always requested from the backend
    assert recorded == [(43, "coding")]  # the backend's real total, not 3 chunks
    assert not any('"usage"' in f for f in frames)  # the client did not ask for it, so it is not forwarded
    assert frames[-1] == "[DONE]" and any("hello" in f or "hel" in f for f in frames)


async def test_streaming_forwards_the_usage_chunk_when_the_client_asks(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    with (
        patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)),
        patch("src.api.routes.completions.record_token_usage", lambda *a, **k: _noop()),
    ):
        async with running_client() as client:
            frames = await _stream_frames(
                client,
                api_key,
                {
                    "model": "coding",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
    assert any('"usage"' in f and '"total_tokens":43' in f for f in frames)


async def test_streaming_estimates_with_tiktoken_when_the_backend_sends_no_usage(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    fake_client.send_usage_chunk = False
    recorded: list[int] = []

    async def fake_record(tenant_id, tokens, model_role="unknown"):
        recorded.append(tokens)
        return {}

    with (
        patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)),
        patch("src.api.routes.completions.record_token_usage", fake_record),
    ):
        async with running_client() as client:
            await _stream_frames(
                client, api_key, {"model": "coding", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            )
    # prompt ("hi" + per-message overhead) + completion ("hello there") — a real count, more than 3 chunks
    assert recorded and recorded[0] >= 7


async def _noop():
    return {}


async def test_a_completion_lands_in_the_usage_ledger_with_dollars(tenant_and_key, monkeypatch):
    """Gateway -> ledger -> GET /v1/keystone/usage: the real tokens and the operator's price, end to end."""
    from src.config import get_settings

    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 2.0}')
    get_settings.cache_clear()
    _tenant_id, api_key = tenant_and_key
    fake_client = _ScriptedInferenceClient()
    try:
        with patch("src.api.routes.completions.get_model_router", return_value=_ScriptedModelRouter(fake_client)):
            async with running_client() as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status_code == 200, resp.text
                usage = await client.get("/v1/keystone/usage?days=1", headers={"Authorization": f"Bearer {api_key}"})
        assert usage.status_code == 200, usage.text
        body = usage.json()
        assert body["pricing_configured"] is True
        assert body["totals"] == {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "request_count": 1,
            "estimated_cost_usd": pytest.approx(7 / 1_000_000 * 2.0),
        }
        assert body["rows"][0]["model_role"] == "coding"
    finally:
        get_settings.cache_clear()
