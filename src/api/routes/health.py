# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Health and readiness endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Response
from sqlalchemy import text

from src.api.models.responses import HealthResponse
from src.inference.client import get_inference_client
from src.inference.health import endpoint_health

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness():
    """Process is up and serving HTTP. Nothing else — a liveness probe that depended on Postgres
    would restart every API pod during a database blip, which makes an outage worse, not better."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(response: Response):
    """Ready to take traffic: Postgres and Redis reachable (503 otherwise). Model endpoints are
    reported but do not gate readiness — the gateway answers 503 + Retry-After per request when a
    role has no healthy endpoint, and an API pod with no model is still the right place to serve
    /health, the admin API and the web UI."""
    components = await _infra_components()
    ready = all(v == "healthy" for v in components.values())
    components.update(_model_components())
    if not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "not_ready", "components": components}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """The full picture (infra + every model role, with breaker state) — cached probes, never one
    round trip per model per call. `status` is "healthy" only when everything is."""
    components = await _infra_components()
    await _probe_models()
    components.update(_model_components())
    overall = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components)


async def _infra_components() -> dict[str, str]:
    components: dict[str, str] = {}
    try:
        from src.api.middleware.rate_limiter import get_redis

        r = await get_redis()
        await r.ping()
        components["redis"] = "healthy"
    except Exception as exc:
        logger.warning("health.redis_check_failed", error=str(exc))
        components["redis"] = "unreachable"

    try:
        from src.db.connection import get_db_context

        async with get_db_context() as db:
            await db.execute(text("SELECT 1"))
        components["postgres"] = "healthy"
    except Exception as exc:
        logger.warning("health.postgres_check_failed", error=str(exc))
        components["postgres"] = "unreachable"
    return components


def _model_components() -> dict[str, str]:
    """Per-role model status from the shared registry (src/inference/health.py) — no network here, so
    /health/ready never blocks on a model; /health awaits `_probe_models()` first."""
    snapshot = endpoint_health.snapshot()
    out: dict[str, str] = {}
    for role in ("coding", "coding_fallback", "reasoning"):
        st = snapshot.get(role)
        if st is None:
            out[f"vllm_{role}"] = "unprobed"
        elif st["breaker"] == "open":
            out[f"vllm_{role}"] = f"breaker_open_{st['open_for_seconds']}s"
        else:
            out[f"vllm_{role}"] = "healthy" if st["healthy"] else "unhealthy"
    return out


async def _probe_models() -> None:
    for role in ("coding", "coding_fallback", "reasoning"):
        try:
            await endpoint_health.is_healthy(role, get_inference_client(role))
        except Exception as exc:
            logger.warning("health.vllm_check_failed", role=role, error=str(exc))


@router.get("/")
async def root():
    return {
        "platform": "Keystone",
        "products": ["Keystone Inference", "Keystone Agents"],
        "version": "1.0.0",
        "docs": "/docs",
    }
