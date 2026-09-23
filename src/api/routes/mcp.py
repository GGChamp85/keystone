# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — MCP server.

Exposes the same memory surface as the `keystone` CLI (src/cli/main.py) and
the OpenCode plugin (cli/opencode/plugins/keystone-memory.ts) — plus task
submission and status, the same operations as POST/GET /v1/keystone/tasks —
to any MCP-speaking client (desktop clients, IDEs, a remote `opencode.json`
`mcp` entry) over the real Streamable HTTP transport — mounted into the
main FastAPI app (src/main.py) at /v1/keystone/mcp.

Built on the official `mcp` SDK (MIT, verified via `pip download mcp` +
reading its real METADATA before adding it as a dependency). Its current
(2.x) API renamed the old `mcp.server.fastmcp.FastMCP` to
`mcp.server.mcpserver.MCPServer` — confirmed by reading the installed
package's own source, not assumed from stale docs.

Auth: MCP tool calls run outside FastAPI's own request/Depends cycle (the
SDK dispatches them itself once a session is established), so there's no
`Depends(require_scope(...))` to hang this off of. Instead this reads the
Authorization header directly off `Context.headers` — populated by the
HTTP transport per-request — and resolves it against the real database via
`_resolve_api_key`, the exact same check `require_scope("agent")` performs
for every REST route in this codebase. One real auth implementation, two
call sites, not a second copy that could drift out of sync.

Known limitation: the streamable-HTTP session manager here runs in-process
(default `stateless_http=False`), so MCP sessions are pinned to whichever
uvicorn worker/replica handled `initialize` — fine for the single-worker
dev/API deployment this was verified against, but a horizontally-scaled
deployment needs either sticky routing or `stateless_http=True` (trading
away session-scoped server state, which no tool here currently relies on
anyway). Flagged here rather than silently assumed away.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError
from starlette.applications import Starlette

from src.api.middleware.auth import _resolve_api_key
from src.api.models.requests import AgentTaskRequest
from src.api.models.responses import AgentTaskSubmittedResponse, AgentTaskSummaryResponse
from src.api.routes.agents import submit_agent_task
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, User
from src.memory.store import MemoryRecord, create_memory, recall
from src.orchestrator.concurrency import ConcurrencyLimitExceeded
from src.orchestrator.engine import get_keystone_engine

mcp_server: MCPServer = MCPServer(
    name="keystone",
    title="Keystone Agents",
    instructions=(
        "Tools for Keystone Agents: the team memory — the conventions, preferences, "
        "facts, and things-to-avoid the background coding agent has learned and "
        "recalls into its own prompts — and the background coding tasks themselves "
        "(submit one, check its status). Every call needs the same `ks-...` Bearer "
        "API key used against the rest of the Keystone API."
    ),
    version="1.0.0",
)


async def _authenticate(ctx: Context) -> tuple[APIKey, Tenant, User | None]:
    headers = ctx.headers or {}
    raw = headers.get("authorization") or headers.get("Authorization")
    raw_key = raw[len("Bearer ") :].strip() if raw and raw.lower().startswith("bearer ") else raw
    try:
        async with get_db_context() as db:
            return await _resolve_api_key(raw_key, db)
    except Exception as exc:  # FastAPI's HTTPException — an anticipated auth failure, not a crash
        detail = getattr(exc, "detail", None) or str(exc)
        raise ToolError(str(detail)) from exc


def _record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "repository": record.repository,
        "scope": record.scope,
        "kind": record.kind,
        "content": record.content,
        "status": record.status,
        "pinned": record.pinned,
        "hit_count": record.hit_count,
    }


@mcp_server.tool()
async def memory_search(
    query: str,
    ctx: Context,
    repository: str | None = None,
    budget_tokens: int = 2000,
) -> list[dict[str, Any]]:
    """Recall Keystone's learned team memory relevant to a query.

    Uses the real repo-over-tenant ranked recall (src/memory/store.py) the
    background agent's own planning/coding prompts use — pinned memories
    first, then repo-scope over tenant-scope, then keyword relevance and
    recency, trimmed to `budget_tokens`. Pass `repository` (a git clone URL)
    to include that repo's own memories alongside team-wide ones.
    """
    _api_key, tenant, _user = await _authenticate(ctx)
    records = await recall(tenant.id, repository, query, budget_tokens=budget_tokens)
    return [_record_to_dict(r) for r in records]


@mcp_server.tool()
async def memory_add(
    content: str,
    ctx: Context,
    kind: str = "fact",
    repository: str | None = None,
    pinned: bool = False,
) -> dict[str, Any]:
    """Add a new team memory.

    `kind` is one of convention | preference | fact | avoid. Omit
    `repository` for a tenant-wide memory; pass a git clone URL to scope it
    to one repo. `pinned` memories are always recalled regardless of query
    relevance (e.g. "never touch the legacy billing module").
    """
    api_key, tenant, user = await _authenticate(ctx)
    record = await create_memory(
        tenant.id,
        content,
        kind=kind,
        repository=repository,
        source="user",
        status="approved",
        pinned=pinned,
        created_by=str(user.id) if user else str(api_key.id),
    )
    return _record_to_dict(record)


@mcp_server.tool()
async def task_submit(
    task: str,
    ctx: Context,
    repository_url: str | None = None,
    branch: str = "main",
    model: str = "coding",
    max_iterations: int = 15,
    file_paths: list[str] | None = None,
    quality_blocking_tools: list[str] | None = None,
    best_of_n: int | None = None,
) -> dict[str, Any]:
    """Submit a background coding task to Keystone Agents.

    The agent clones `repository_url` into a sandbox, plans, edits with real
    tools, runs the quality gates and the repo's tests, and opens a pull
    request. `model` is a gateway role — coding | coding_fallback | reasoning —
    or `auto` to let the router classify the task. `max_iterations` is the
    one per-task safety bound (no ceiling). Exactly the validation of
    `POST /v1/keystone/tasks`: the same AgentTaskRequest model and the same
    submission function, so the two surfaces cannot drift. Returns
    `{task_id, status, message}`; follow the task with `task_status` or the
    REST stream `GET /v1/keystone/tasks/{task_id}/stream`.
    """
    api_key, tenant, user = await _authenticate(ctx)
    try:
        req = AgentTaskRequest(
            task=task,
            repository_url=repository_url,
            branch=branch,
            file_paths=file_paths or [],
            model=model,  # type: ignore[arg-type]  # pydantic validates the Literal at runtime → ToolError
            max_iterations=max_iterations,
            quality_blocking_tools=quality_blocking_tools,
            best_of_n=best_of_n,
        )
    except ValidationError as exc:
        raise ToolError(_validation_message(exc)) from exc
    try:
        task_id = await submit_agent_task(req, api_key, tenant, user)
    except ConcurrencyLimitExceeded as exc:
        raise ToolError(str(exc)) from exc
    except HTTPException as exc:  # the tenant's monthly dollar budget is spent (same refusal as the REST route)
        raise ToolError(str(exc.detail)) from exc
    return AgentTaskSubmittedResponse(task_id=task_id, status="pending").model_dump(mode="json")


@mcp_server.tool()
async def task_status(task_id: str, ctx: Context) -> dict[str, Any]:
    """The current status of one background task, in the same summary shape as
    `GET /v1/keystone/tasks` (AgentTaskSummaryResponse): status, model role,
    branch, pull-request URL/number, error message, timestamps. Scoped to the
    calling key's tenant — another tenant's task is "not found".
    """
    _api_key, tenant, _user = await _authenticate(ctx)
    try:
        task_uuid = UUID(task_id)
    except ValueError as exc:
        raise ToolError(f"task_id must be a UUID, got {task_id!r}") from exc
    row = await get_keystone_engine().get_task_summary(tenant.id, task_uuid)
    if row is None:
        raise ToolError("Task not found")
    return AgentTaskSummaryResponse(**row).model_dump(mode="json")


def _validation_message(exc: ValidationError) -> str:
    """pydantic's errors as one readable line per field — the same facts FastAPI's 422 body
    carries for the REST route, in the plain-text form a tool error needs."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "request"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "Invalid task request — " + "; ".join(parts)


def build_mcp_asgi_app() -> Starlette:
    """Streamable-HTTP ASGI app; mounted at /v1/keystone in src/main.py so the
    real endpoint is /v1/keystone/mcp. Its lifespan (session-manager startup)
    must be entered alongside the parent app's own — see src/main.py."""
    settings = get_settings()
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=settings.mcp_allowed_hosts,
        allowed_origins=settings.mcp_allowed_origins,
    )
    return mcp_server.streamable_http_app(transport_security=transport_security)
