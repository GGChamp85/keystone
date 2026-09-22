"""
Keystone — Distributed rate limiter & token budget enforcement.

Two layers:
  1. Request-rate limiter  — sliding-window counter in Redis (requests/min).
  2. Token budget limiter  — daily + monthly token counters per tenant/key.

Both are enforced at the FastAPI middleware level before the request
reaches the vLLM backend, preventing runaway costs.
"""

from __future__ import annotations

import time
from uuid import UUID

import redis.asyncio as aioredis
import structlog
from fastapi import HTTPException, status

from src.config import get_settings
from src.observability import tokens_used_total

logger = structlog.get_logger(__name__)

_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=50,
        )
    return _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None


# ── Sliding-window request rate limiter ───────────────────────

_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit  = tonumber(ARGV[2])
local now    = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now .. '-' .. math.random(1, 1000000))
    redis.call('EXPIRE', key, window)
    return 0
else
    return 1
end
"""


async def check_request_rate(
    tenant_id: UUID,
    api_key_id: UUID,
    requests_per_minute: int = 0,
) -> None:
    """
    Sliding-window rate limiter.  Raises 429 if exceeded. `requests_per_minute`
    0 = no limit (the default policy) — the check is skipped entirely.
    """
    if requests_per_minute <= 0:
        return
    r = await get_redis()
    key = f"vs:rate:{tenant_id}:{api_key_id}"
    window_ms = 60  # seconds
    now = time.time()

    blocked = await r.eval(
        _RATE_LIMIT_SCRIPT,
        1,
        key,
        str(window_ms),
        str(requests_per_minute),
        str(now),
    )

    if blocked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {requests_per_minute} requests/min",
            headers={"Retry-After": "60"},
        )


# ── Token budget enforcement ─────────────────────────────────


def _daily_key(tenant_id: UUID) -> str:
    day = time.strftime("%Y-%m-%d")
    return f"vs:tokens:daily:{tenant_id}:{day}"


def _monthly_key(tenant_id: UUID) -> str:
    month = time.strftime("%Y-%m")
    return f"vs:tokens:monthly:{tenant_id}:{month}"


async def check_token_budget(
    tenant_id: UUID,
    daily_limit: int,
    monthly_limit: int,
) -> None:
    """
    Pre-flight check — ensures the tenant still has budget before we
    forward the request to vLLM.  Raises 429 if exhausted. A limit of 0
    means unlimited and is not checked.
    """
    if daily_limit <= 0 and monthly_limit <= 0:
        return
    r = await get_redis()

    daily_used = int(await r.get(_daily_key(tenant_id)) or 0)
    monthly_used = int(await r.get(_monthly_key(tenant_id)) or 0)

    if daily_limit > 0 and daily_used >= daily_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily token budget exhausted ({daily_used:,}/{daily_limit:,})",
            headers={"X-VS-Daily-Used": str(daily_used), "X-VS-Daily-Limit": str(daily_limit)},
        )

    if monthly_limit > 0 and monthly_used >= monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly token budget exhausted ({monthly_used:,}/{monthly_limit:,})",
            headers={"X-VS-Monthly-Used": str(monthly_used), "X-VS-Monthly-Limit": str(monthly_limit)},
        )


async def record_token_usage(
    tenant_id: UUID,
    tokens: int,
    model_role: str = "unknown",
) -> dict[str, int]:
    """
    Post-request: atomically increment daily + monthly counters.
    Returns updated totals.
    """
    tokens_used_total.labels(str(tenant_id), model_role, "total").inc(tokens)

    r = await get_redis()
    pipe = r.pipeline(transaction=True)

    dk = _daily_key(tenant_id)
    mk = _monthly_key(tenant_id)

    pipe.incrby(dk, tokens)
    pipe.expire(dk, 86400 + 3600)  # TTL = 25 hours (covers timezone drift)

    pipe.incrby(mk, tokens)
    pipe.expire(mk, 31 * 86400 + 3600)  # TTL ≈ 31 days

    results = await pipe.execute()
    daily_total = results[0]
    monthly_total = results[2]

    return {"daily_used": daily_total, "monthly_used": monthly_total}


async def get_budget_status(
    tenant_id: UUID,
    daily_limit: int,
    monthly_limit: int,
) -> dict:
    """Return current budget status for a tenant."""
    r = await get_redis()
    daily_used = int(await r.get(_daily_key(tenant_id)) or 0)
    monthly_used = int(await r.get(_monthly_key(tenant_id)) or 0)

    # A limit of 0 means unlimited: remaining is reported as None, percent as 0.
    return {
        "tenant_id": str(tenant_id),
        "daily_limit": daily_limit,
        "daily_used": daily_used,
        "daily_remaining": max(0, daily_limit - daily_used) if daily_limit > 0 else None,
        "monthly_limit": monthly_limit,
        "monthly_used": monthly_used,
        "monthly_remaining": max(0, monthly_limit - monthly_used) if monthly_limit > 0 else None,
        "percent_daily_used": round(daily_used / daily_limit * 100, 2) if daily_limit > 0 else 0,
        "percent_monthly_used": round(monthly_used / monthly_limit * 100, 2) if monthly_limit > 0 else 0,
    }


# ── Circuit-breaker token guard for agents ────────────────────


async def check_agent_token_guard(
    tenant_id: UUID,
    task_id: UUID,
    iteration: int,
    max_iterations: int,
    tokens_this_task: int,
    max_tokens_per_task: int = 0,
) -> None:
    """
    Agent-specific guard — prevents runaway coding loops.
    Called inside the LangGraph orchestrator before each iteration.
    `max_tokens_per_task` 0 = unlimited; a tenant daily limit of 0 = unlimited.
    """
    if iteration >= max_iterations:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(f"Agent circuit breaker: reached max iterations ({iteration}/{max_iterations}) for task {task_id}"),
        )

    if max_tokens_per_task > 0 and tokens_this_task >= max_tokens_per_task:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Agent circuit breaker: token budget for task {task_id} "
                f"exhausted ({tokens_this_task:,}/{max_tokens_per_task:,})"
            ),
        )

    # Also check tenant-level budget
    settings = get_settings()
    if settings.default_daily_token_limit <= 0:
        return
    r = await get_redis()
    daily_used = int(await r.get(_daily_key(tenant_id)) or 0)

    if daily_used >= settings.default_daily_token_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Tenant daily token budget exhausted mid-agent-task",
        )
