# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Health and readiness endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from sqlalchemy import text

from src.api.models.responses import HealthResponse
from src.inference.client import get_inference_client

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    components = {}
    for role in ("coding", "coding_fallback", "reasoning"):
        try:
            client = get_inference_client(role)
            healthy = await client.health()
            components[f"vllm_{role}"] = "healthy" if healthy else "unhealthy"
        except Exception as exc:
            logger.warning("health.vllm_check_failed", role=role, error=str(exc))
            components[f"vllm_{role}"] = "unreachable"

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

    overall = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components)


@router.get("/")
async def root():
    return {
        "platform": "Keystone",
        "products": ["Keystone Inference", "Keystone Agents"],
        "version": "1.0.0",
        "docs": "/docs",
    }
