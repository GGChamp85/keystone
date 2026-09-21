# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/api/routes/mcp.py — a real MCP JSON-RPC
handshake (initialize -> notifications/initialized -> tools/list ->
tools/call) over the app's real Streamable HTTP endpoint, against a real
Postgres tenant/API key, not a mock of the MCP protocol or the auth layer.
See tests/conftest.py's requires_integration_env.

The running app + client is a plain async context manager entered directly
inside each test body (not a pytest fixture) — the MCP session manager's
`streamable_http_app()` lifespan holds an anyio task group whose cancel
scope is bound to the task that entered it; pytest-asyncio's fixture
setup/teardown for an async-generator fixture can resume across a
different task than setup ran in, which anyio's cancel scope rejects
("Attempted to exit cancel scope in a different task than it was entered
in") — a real, reproduced incompatibility, not a production bug: the app
runs fine under uvicorn's own single lifespan (verified separately via
real curl against a live `uvicorn src.main:app`). Keeping enter/exit in one
uninterrupted test coroutine sidesteps it.
"""

from __future__ import annotations

import json
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


def _parse_sse_json(body: str) -> dict:
    """The streamable-HTTP transport's default (non-json_response) mode replies
    as a single SSE `data:` frame carrying the real JSON-RPC message."""
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"No SSE data frame found in response body: {body!r}")


@pytest.fixture
async def tenant_and_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="mcp-server-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="mcp-test-key", key_prefix=prefix, key_hash=key_hash, scopes=["agent"]))
        await db.flush()
    yield tid, full_key
    async with get_db_context() as db:
        tenant = await db.get(Tenant, tid)
        if tenant is not None:
            await db.delete(tenant)


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        # 127.0.0.1:<port> matches the MCP server's default DNS-rebinding-
        # protection allowlist ("127.0.0.1:*" -- src/api/routes/mcp.py's
        # streamable_http_app()'s default host="127.0.0.1"); the port is
        # required (a bare "127.0.0.1"/http's default port 80, silently
        # dropped from the Host header by httpx same as a browser, doesn't
        # match the wildcard pattern). Confirmed for real: an arbitrary Host
        # header (ASGITransport's usual "http://test" default, or a portless
        # "http://127.0.0.1") is correctly rejected with 421 Misdirected
        # Request before it ever reaches auth.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
            yield client


async def _initialize(client: httpx.AsyncClient, api_key: str) -> str:
    resp = await client.post(
        "/v1/keystone/mcp",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    session_id = resp.headers["mcp-session-id"]
    ack = await client.post(
        "/v1/keystone/mcp",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": session_id,
        },
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert ack.status_code == 202
    return session_id


async def _call_tool(client: httpx.AsyncClient, api_key: str, session_id: str, name: str, arguments: dict) -> dict:
    resp = await client.post(
        "/v1/keystone/mcp",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": session_id,
        },
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
    )
    assert resp.status_code == 200, resp.text
    return _parse_sse_json(resp.text)["result"]


async def test_initialize_handshake_returns_real_server_info(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        resp = await client.post(
            "/v1/keystone/mcp",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "1"},
                },
            },
        )
        assert resp.status_code == 200
        assert "mcp-session-id" in resp.headers
        body = _parse_sse_json(resp.text)
        assert body["result"]["serverInfo"]["name"] == "keystone"


async def test_tools_list_exposes_memory_search_and_memory_add(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)
        resp = await client.post(
            "/v1/keystone/mcp",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
                "Mcp-Session-Id": session_id,
            },
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        tools = {t["name"] for t in _parse_sse_json(resp.text)["result"]["tools"]}
        assert tools == {"memory_search", "memory_add"}


async def test_memory_add_then_search_round_trips_through_the_real_store(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)

        added = await _call_tool(
            client, api_key, session_id, "memory_add", {"content": "Never touch the billing module.", "kind": "avoid"}
        )
        assert added["isError"] is False
        created = added["structuredContent"]
        assert created["kind"] == "avoid"
        assert created["scope"] == "tenant"

        found = await _call_tool(client, api_key, session_id, "memory_search", {"query": "billing module"})
        assert found["isError"] is False
        results = found["structuredContent"]["result"]
        assert any(r["id"] == created["id"] for r in results)


async def test_memory_search_with_invalid_api_key_fails_closed_with_a_readable_message(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)
        result = await _call_tool(client, "ks-totally-bogus-key", session_id, "memory_search", {"query": "anything"})
        assert result["isError"] is True
        assert "Invalid API key" in result["content"][0]["text"]


async def test_memory_search_with_no_authorization_header_fails_closed(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)
        resp = await client.post(
            "/v1/keystone/mcp",
            headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "memory_search", "arguments": {"query": "anything"}},
            },
        )
        result = _parse_sse_json(resp.text)["result"]
        assert result["isError"] is True
        assert "Missing Authorization" in result["content"][0]["text"]
