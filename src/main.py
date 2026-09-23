# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone
Main FastAPI application.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from src.config import get_settings
from src.observability import http_request_duration_seconds, http_requests_total

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("vs.starting", env=settings.vs_env.value, platform=settings.vs_platform_name)
    from src.db.connection import init_db

    await init_db()
    logger.info("vs.db_ready")
    from src.api.middleware.rate_limiter import get_redis

    r = await get_redis()
    await r.ping()
    logger.info("vs.redis_ready")

    # The MCP streamable-HTTP sub-app (src/api/routes/mcp.py) owns its own
    # lifespan (starts its session manager's background task) — mounting it
    # with app.mount() alone does NOT cascade FastAPI's lifespan into it, so
    # it's entered explicitly here, scoped to the same stack as everything
    # else this function starts/stops.
    async with AsyncExitStack() as mcp_stack:
        mcp_app = app.state.mcp_asgi_app
        await mcp_stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
        logger.info("vs.mcp_ready")

        # Periodic PR-status poller (src/orchestrator/pr_polling.py) — not
        # Temporal-durable, a plain background asyncio task like the rest of
        # this app's own non-durable fallback path; a missed pass just gets
        # picked up next interval since it only ever reads current state.
        from src.orchestrator.pr_polling import run_pr_polling_loop

        pr_poll_stop = asyncio.Event()
        pr_poll_task = asyncio.create_task(run_pr_polling_loop(pr_poll_stop))
        logger.info("vs.pr_polling_started", interval_seconds=settings.pr_poll_interval_seconds)

        yield

        pr_poll_stop.set()
        await pr_poll_task

    from src.api.middleware.rate_limiter import close_redis
    from src.db.connection import close_db
    from src.inference.client import close_all_clients

    await close_all_clients()
    await close_redis()
    await close_db()
    logger.info("vs.shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Keystone",
        description=(
            "Self-hosted LLM inference with API key management (Keystone Inference) "
            "and autonomous coding agents (Keystone Agents). "
            "Apache 2.0 — Copyright Gaurav Gupta."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.vs_debug else settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        route = request.scope.get("route")
        route_template = getattr(route, "path", request.url.path)
        http_requests_total.labels(request.method, route_template, response.status_code).inc()
        http_request_duration_seconds.labels(request.method, route_template).observe(duration)
        return response

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    from src.api.routes.agents import router as agents_router
    from src.api.routes.completions import router as completions_router
    from src.api.routes.finetune import router as finetune_router
    from src.api.routes.health import router as health_router
    from src.api.routes.keys import router as keys_router
    from src.api.routes.mcp import build_mcp_asgi_app
    from src.api.routes.memory import router as memory_router
    from src.api.routes.messages import router as messages_router
    from src.api.routes.models_library import router as models_library_router

    app.include_router(health_router)
    app.include_router(completions_router)
    app.include_router(messages_router)
    app.include_router(models_library_router)
    app.include_router(keys_router)
    app.include_router(agents_router)
    app.include_router(memory_router)
    app.include_router(finetune_router)

    # MCP server (src/api/routes/mcp.py) — real endpoint at /v1/keystone/mcp.
    # Stored on app.state so `lifespan()` above can enter its session
    # manager's own lifespan alongside the rest of this app's startup.
    mcp_asgi_app = build_mcp_asgi_app()
    app.state.mcp_asgi_app = mcp_asgi_app
    app.mount("/v1/keystone", mcp_asgi_app)

    # Keystone Agents plan/execution-trace UI (web/) — a static build, not
    # served in every environment: absent unless `npm run build` (or the
    # Dockerfile's build stage) has produced web/dist/, so a bare-Python
    # dev checkout without Node.js still boots the API fine, just without
    # the UI at /app/.
    web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if web_dist.is_dir():
        app.mount("/app", StaticFiles(directory=web_dist, html=True), name="web-ui")
        logger.info("vs.web_ui_mounted", path=str(web_dist))
    else:
        logger.info("vs.web_ui_not_built", expected_path=str(web_dist))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.vs_debug,
        log_level=settings.log_level.lower(),
    )
