# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The Anthropic-compatible gateway (`POST /v1/messages`, src/api/routes/messages.py) and the tool
passthrough of `/v1/chat/completions`, over the real FastAPI app with the real InferenceClient talking
HTTP to a real backend process (tests/fixtures/fake_vllm_server.py serving vLLM's exact shapes) — real
Postgres/Redis for the API key, the budget and the dollar ledger. Requires DATABASE_URL/REDIS_URL.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import select

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, TenantTier, UsageRecord
from src.inference.client import InferenceClient
from src.main import create_app

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


class _Router:
    """The health-aware router, scripted to the fake backend: the real InferenceClient does the HTTP."""

    def __init__(self, base_url: str):
        self.client = InferenceClient(base_url=base_url, model_id="test-model")

    async def get_client(self, model_input):
        return self.client, "coding"


@pytest.fixture
async def tenant_and_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="messages-api-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["inference"]))
    yield tid, full_key
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@asynccontextmanager
async def running_client(backend_url: str) -> AsyncIterator[tuple[httpx.AsyncClient, _Router]]:
    router = _Router(backend_url)
    app = create_app()
    with patch("src.api.routes._inference_common.get_model_router", return_value=router):
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, router
    await router.client.close()


async def _last_payload(backend_url: str) -> dict:
    async with httpx.AsyncClient() as h:
        return (await h.get(f"{backend_url}/last_payload")).json()


def _parse_sse(body: str) -> list[dict]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = frame.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), frame
        payload = json.loads(lines[1][len("data: ") :])
        assert payload["type"] == lines[0][len("event: ") :]
        events.append(payload)
    return events


async def test_messages_non_streaming_translates_both_ways_and_bills_the_ledger(tenant_and_key, fake_vllm_server):
    tid, key = tenant_and_key
    async with running_client(fake_vllm_server) as (client, _):
        resp = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
            json={
                "model": "coding",
                "max_tokens": 64,
                "system": "Be brief.",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                "stop_sequences": ["END"],
                "temperature": 0.1,
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "message" and body["role"] == "assistant" and body["model"] == "coding"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 5}  # the backend's real usage
    assert resp.headers["x-vs-model"] == "coding"

    sent = await _last_payload(fake_vllm_server)
    assert sent["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"}]
    assert sent["stop"] == ["END"] and sent["max_tokens"] == 64 and sent["temperature"] == 0.1
    assert sent["model"] == "test-model" and sent["stream"] is False

    async with get_db_context() as db:
        rows = (await db.execute(select(UsageRecord).where(UsageRecord.tenant_id == tid))).scalars().all()
    assert sum(r.prompt_tokens for r in rows) == 10 and sum(r.completion_tokens for r in rows) == 5


async def test_messages_tool_use_round_trip(tenant_and_key, fake_vllm_server):
    _, key = tenant_and_key
    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    async with running_client(fake_vllm_server) as (client, _):
        first = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "read app.py"}],
                "tools": tools,
                "tool_choice": {"type": "any"},
            },
        )
        assert first.status_code == 200, first.text
        msg = first.json()
        assert msg["stop_reason"] == "tool_use"
        tool_use = msg["content"][0]
        assert tool_use == {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "app.py"}}
        sent = await _last_payload(fake_vllm_server)
        assert sent["tools"][0]["function"]["name"] == "read_file" and sent["tool_choice"] == "required"

        # the tool result goes back as a `tool` message carrying the call id
        second = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "read app.py"},
                    {"role": "assistant", "content": msg["content"]},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "print(1)"}],
                    },
                ],
            },
        )
        assert second.status_code == 200, second.text
        sent = await _last_payload(fake_vllm_server)
    assert sent["messages"][1]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "app.py"}'}
    assert sent["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"}


async def test_messages_streaming_emits_anthropic_events_with_real_usage(tenant_and_key, fake_vllm_server):
    _, key = tenant_and_key
    async with running_client(fake_vllm_server) as (client, _):
        resp = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "coding", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)
    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta") == "hello"
    assert events[5]["delta"]["stop_reason"] == "end_turn"
    assert events[5]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    sent = await _last_payload(fake_vllm_server)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


async def test_messages_streaming_tool_call_arrives_as_input_json_deltas(tenant_and_key, fake_vllm_server):
    _, key = tenant_and_key
    async with running_client(fake_vllm_server) as (client, _):
        resp = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "read"}],
                "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            },
        )
        events = _parse_sse(resp.text)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "tool_use" and starts[0]["content_block"]["name"] == "read_file"
    partial = "".join(e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta")
    assert json.loads(partial) == {"path": "app.py"}
    assert next(e for e in events if e["type"] == "message_delta")["delta"]["stop_reason"] == "tool_use"


async def test_messages_refuses_images_with_the_anthropic_error_envelope(tenant_and_key, fake_vllm_server):
    _, key = tenant_and_key
    async with running_client(fake_vllm_server) as (client, _):
        resp = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
                        ],
                    }
                ],
            },
        )
        count = await client.post(
            "/v1/messages/count_tokens",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "coding", "system": "sys", "messages": [{"role": "user", "content": "count these words"}]},
        )
    assert resp.status_code == 400
    assert resp.json()["type"] == "error" and resp.json()["error"]["type"] == "invalid_request_error"
    assert "image" in resp.json()["error"]["message"]
    assert count.status_code == 200 and count.json()["input_tokens"] > 3


async def test_chat_completions_passes_tools_and_tool_messages_through(tenant_and_key, fake_vllm_server):
    """An OpenAI client in agent mode: tools out, tool_calls back, a `tool` message in — nothing dropped."""
    _, key = tenant_and_key
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    async with running_client(fake_vllm_server) as (client, _):
        first = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "read"}],
                "tools": tools,
                "tool_choice": "auto",
            },
        )
        assert first.status_code == 200, first.text
        call = first.json()["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "read_file"
        second = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "messages": [
                    {"role": "user", "content": "read"},
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": call["id"], "content": "print(1)"},
                ],
                "tools": tools,
            },
        )
        assert second.status_code == 200, second.text
        sent = await _last_payload(fake_vllm_server)
        assert sent["tools"] == tools
        assert sent["messages"][1]["tool_calls"] == [call] and sent["messages"][2]["tool_call_id"] == call["id"]

        both = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "x"}],
                "tools": tools,
                "response_format": {"type": "json_object"},
            },
        )
        assert both.status_code == 422


async def test_the_real_anthropic_sdk_talks_to_the_gateway(tenant_and_key, fake_vllm_server):
    """The installed `anthropic` SDK, pointed at Keystone with its default `api_key=` (sent as x-api-key):
    a non-streaming message, a tool-use turn, and a stream accumulated by the SDK's own event parser."""
    import anthropic
    import httpx2  # the SDK's own HTTP layer (a pinned fork); the in-process transport must come from it

    _, key = tenant_and_key
    router = _Router(fake_vllm_server)
    app = create_app()
    with patch("src.api.routes._inference_common.get_model_router", return_value=router):
        async with app.router.lifespan_context(app):
            http_client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
            sdk = anthropic.AsyncAnthropic(base_url="http://test", api_key=key, http_client=http_client)
            message = await sdk.messages.create(
                model="coding", max_tokens=32, messages=[{"role": "user", "content": "hello"}]
            )
            assert message.role == "assistant" and message.content[0].text == "hello"
            assert message.usage.input_tokens == 10 and message.usage.output_tokens == 5

            with_tool = await sdk.messages.create(
                model="coding",
                max_tokens=32,
                messages=[{"role": "user", "content": "read"}],
                tools=[{"name": "read_file", "input_schema": {"type": "object"}}],
            )
            block = with_tool.content[0]
            assert block.type == "tool_use" and block.name == "read_file" and block.input == {"path": "app.py"}
            assert with_tool.stop_reason == "tool_use"

            async with sdk.messages.stream(
                model="coding", max_tokens=32, messages=[{"role": "user", "content": "hi"}]
            ) as stream:
                text = "".join([t async for t in stream.text_stream])
                final = await stream.get_final_message()
            assert text == "hello" and final.content[0].text == "hello"
            assert final.usage.output_tokens == 5  # accumulated from message_delta by the SDK itself
            await http_client.aclose()
    await router.client.close()
