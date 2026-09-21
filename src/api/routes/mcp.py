# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — MCP server.

Exposes the same memory surface as the `keystone` CLI (src/cli/main.py) and
the OpenCode plugin (cli/opencode/plugins/keystone-memory.ts) to any
MCP-speaking client (Claude Desktop, Claude Code, other IDEs, a remote
`opencode.json` `mcp` entry) over the real Streamable HTTP transport —
mounted into the main FastAPI app (src/main.py) at /v1/keystone/mcp.

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

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from src.api.middleware.auth import _resolve_api_key
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, User
from src.memory.store import MemoryRecord, create_memory, recall

mcp_server: MCPServer = MCPServer(
    name="keystone",
    title="Keystone Agents",
    instructions=(
        "Tools for Keystone Agents' team memory — the conventions, preferences, "
        "facts, and things-to-avoid the background coding agent has learned and "
        "recalls into its own prompts. Every call needs the same `ks-...` Bearer "
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
