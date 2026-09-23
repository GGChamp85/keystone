# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/api/routes/mcp.py — a real MCP JSON-RPC
handshake (initialize -> notifications/initialized -> tools/list ->
tools/call) over the app's real Streamable HTTP endpoint, against a real
Postgres tenant/API key, not a mock of the MCP protocol or the auth layer.
Covers the memory tools and the task tools (task_submit / task_status —
the same submission path as POST /v1/keystone/tasks, so a task submitted
over MCP really is persisted and really starts executing). See
tests/conftest.py's requires_integration_env.

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

import asyncio
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


async def test_tools_list_exposes_memory_and_task_tools(tenant_and_key):
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
        assert tools == {"memory_search", "memory_add", "task_submit", "task_status"}


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


# ── task_submit / task_status ────────────────────────────────────────────────
# The same real submission path as POST /v1/keystone/tasks (src/api/routes/agents.py's
# submit_agent_task): a task really is persisted and really starts executing in the
# background. As in tests/test_concurrency.py, every model endpoint is pointed at a port
# nothing listens on so the execution fails fast at the inference call, releases its
# concurrency slot, and cannot outlive the test's event loop.


@pytest.fixture
def _inference_endpoints_unreachable(monkeypatch):
    from src.config import get_settings
    from src.inference import client as inference_client

    for var in ("VLLM_CODING_URL", "VLLM_CODING_FALLBACK_URL", "VLLM_REASONING_URL"):
        monkeypatch.setenv(var, "http://127.0.0.1:1/v1")
    get_settings.cache_clear()
    inference_client._clients.clear()
    yield
    get_settings.cache_clear()
    inference_client._clients.clear()


async def _wait_for_background_execution_to_finish(tenant_id: uuid.UUID) -> None:
    from src.orchestrator.concurrency import current_tenant_concurrency

    for _ in range(300):  # up to ~30 s — a cold embedding-model load precedes the failing inference call
        if await current_tenant_concurrency(tenant_id) == 0:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("the submitted task's background execution did not release its slot")


async def test_task_submit_then_task_status_is_the_same_task_the_rest_route_sees(
    tenant_and_key, _inference_endpoints_unreachable
):
    tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)
        submitted = await _call_tool(
            client,
            api_key,
            session_id,
            "task_submit",
            {"task": "A real task description long enough to pass validation.", "model": "coding"},
        )
        assert submitted["isError"] is False, submitted
        body = submitted["structuredContent"]
        task_id = body["task_id"]
        uuid.UUID(task_id)  # a real id, not a placeholder
        assert body["status"] == "pending"

        status = await _call_tool(client, api_key, session_id, "task_status", {"task_id": task_id})
        assert status["isError"] is False, status
        summary = status["structuredContent"]
        assert summary["id"] == task_id
        assert summary["model_role"] == "coding"
        assert summary["branch"] == "main"
        assert summary["status"] in {"pending", "running", "failed"}
        assert set(summary) >= {"task_description", "repository_url", "pr_url", "pr_number", "error_message"}

        # The REST route lists exactly this task for the same key — one submission path, two protocols.
        listed = await client.get("/v1/keystone/tasks", headers={"Authorization": f"Bearer {api_key}"})
        assert listed.status_code == 200
        assert any(t["id"] == task_id for t in listed.json())

        await _wait_for_background_execution_to_finish(tenant_id)


async def test_task_submit_applies_the_rest_route_s_validation(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)

        too_short = await _call_tool(client, api_key, session_id, "task_submit", {"task": "too short"})
        assert too_short["isError"] is True
        message = too_short["content"][0]["text"]
        assert "task:" in message and "at least 10 characters" in message  # AgentTaskRequest.task min_length=10

        bad_model = await _call_tool(
            client,
            api_key,
            session_id,
            "task_submit",
            {"task": "A real task description long enough to pass validation.", "model": "gpt-9"},
        )
        assert bad_model["isError"] is True
        assert "model:" in bad_model["content"][0]["text"]  # the Literal["coding","coding_fallback","reasoning","auto"]

        zero_iterations = await _call_tool(
            client,
            api_key,
            session_id,
            "task_submit",
            {"task": "A real task description long enough to pass validation.", "max_iterations": 0},
        )
        assert zero_iterations["isError"] is True
        assert "max_iterations:" in zero_iterations["content"][0]["text"]


async def test_task_status_is_tenant_scoped_and_readable_on_bad_input(tenant_and_key, _inference_endpoints_unreachable):
    tenant_id, api_key = tenant_and_key
    other_tid = uuid.uuid4()
    other_key, other_prefix, other_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tid, name="mcp-other-tenant", email=f"{other_tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(
            APIKey(tenant_id=other_tid, name="other", key_prefix=other_prefix, key_hash=other_hash, scopes=["agent"])
        )
        await db.flush()
    try:
        async with running_client() as client:
            session_id = await _initialize(client, api_key)
            submitted = await _call_tool(
                client,
                api_key,
                session_id,
                "task_submit",
                {"task": "A real task description long enough to pass validation."},
            )
            task_id = submitted["structuredContent"]["task_id"]

            not_a_uuid = await _call_tool(client, api_key, session_id, "task_status", {"task_id": "nope"})
            assert not_a_uuid["isError"] is True
            assert "must be a UUID" in not_a_uuid["content"][0]["text"]

            unknown = await _call_tool(client, api_key, session_id, "task_status", {"task_id": str(uuid.uuid4())})
            assert unknown["isError"] is True
            assert "Task not found" in unknown["content"][0]["text"]

            # Another tenant's valid key sees the same id as not found — never the other tenant's task.
            other_session = await _initialize(client, other_key)
            foreign = await _call_tool(client, other_key, other_session, "task_status", {"task_id": task_id})
            assert foreign["isError"] is True
            assert "Task not found" in foreign["content"][0]["text"]

            await _wait_for_background_execution_to_finish(tenant_id)
    finally:
        async with get_db_context() as db:
            other = await db.get(Tenant, other_tid)
            if other is not None:
                await db.delete(other)


async def test_task_tools_fail_closed_without_a_valid_key(tenant_and_key):
    _tenant_id, api_key = tenant_and_key
    async with running_client() as client:
        session_id = await _initialize(client, api_key)

        bogus = await _call_tool(
            client,
            "ks-totally-bogus-key",
            session_id,
            "task_submit",
            {"task": "A real task description long enough to pass validation."},
        )
        assert bogus["isError"] is True
        assert "Invalid API key" in bogus["content"][0]["text"]

        resp = await client.post(
            "/v1/keystone/mcp",
            headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "task_status", "arguments": {"task_id": str(uuid.uuid4())}},
            },
        )
        result = _parse_sse_json(resp.text)["result"]
        assert result["isError"] is True
        assert "Missing Authorization" in result["content"][0]["text"]

        # Nothing was persisted by the rejected submissions.
        listed = await client.get("/v1/keystone/tasks", headers={"Authorization": f"Bearer {api_key}"})
        assert listed.status_code == 200 and listed.json() == []
