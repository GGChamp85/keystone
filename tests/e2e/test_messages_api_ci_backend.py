# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The Anthropic-compatible gateway end to end against a REAL model: the full FastAPI app (real auth, real
routing through the real ModelRouter, real ledger) with the coding role served by the llama.cpp demo
model (docker-compose `demo-model` profile / the CI job's backend). A real prompt, a real answer, real
token usage — non-streaming and streamed as Anthropic events. Self-skips when VLLM_CODING_URL is unset
or unreachable, like tests/e2e/test_ci_backend.py.
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, TenantTier

from ..conftest import requires_integration_env

_BACKEND_URL = os.environ.get("VLLM_CODING_URL", "")


def _backend_reachable() -> bool:
    if not _BACKEND_URL:
        return False
    try:
        return httpx.get(f"{_BACKEND_URL.rstrip('/')}/models", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    requires_integration_env,
    pytest.mark.skipif(not _backend_reachable(), reason="Needs a reachable model backend at VLLM_CODING_URL"),
]


@pytest.fixture
async def api_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="messages-e2e", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["inference"]))
    yield full_key
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def test_a_real_model_answers_through_the_messages_api(api_key):
    from src.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
            resp = await client.post(
                "/v1/messages",
                headers={"Authorization": f"Bearer {api_key}", "anthropic-version": "2023-06-01"},
                json={
                    "model": "coding",
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["type"] == "message" and body["content"][0]["type"] == "text"
            assert body["content"][0]["text"].strip(), body
            assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0
            assert body["stop_reason"] in ("end_turn", "max_tokens")

            streamed = await client.post(
                "/v1/messages",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "coding",
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Say hi."}],
                },
            )
            assert streamed.status_code == 200, streamed.text
    types = []
    text = ""
    for frame in streamed.text.strip().split("\n\n"):
        _event_line, data_line = frame.split("\n")[:2]
        payload = json.loads(data_line[len("data: ") :])
        types.append(payload["type"])
        if payload["type"] == "content_block_delta":
            text += payload["delta"]["text"]
        if payload["type"] == "message_delta":
            assert payload["usage"]["output_tokens"] > 0  # llama.cpp's real include_usage chunk
    assert types[0] == "message_start" and types[-1] == "message_stop" and "message_delta" in types
    assert text.strip()
