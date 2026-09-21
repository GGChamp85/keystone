# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — per-tenant task concurrency, with per-user fairness.

`Tenant.max_concurrent_agents` (src/db/models.py) caps how many agent
tasks a tenant may run at once — a real limit that was declared on the
model but never actually enforced anywhere (grepped: zero hits before
this). Enforced here via a Redis SET per tenant (membership = currently
running task ids, matching the same atomic-Lua-script pattern
src/api/middleware/rate_limiter.py already uses for its own limits, not a
second inconsistent approach) — SADD/SREM is idempotent, so a slot can
never double-release or double-leak from a retry.

Fairness: a single user can occupy at most half the tenant's capacity
(rounded up, minimum 1) at once, so one person queuing a burst of tasks
can't starve every other developer on the same tenant — a real, simple,
testable rule rather than a full weighted-fair-queue scheduler, which
this increment doesn't need to get an actual concurrent-multi-developer
guarantee right.

Slots are acquired at submission (engine.py's submit_task, atomically —
never "check then increment" as two separate steps) and released once
the task reaches a terminal state (engine.py's _execute_task `finally`
block, the same guaranteed-cleanup point sandbox teardown already uses).
A slot that's never released because a worker process was killed
mid-task still expires via the Redis key's TTL, refreshed on every
acquire, so a crash can't leak a slot forever.
"""

from __future__ import annotations

import math
from uuid import UUID

from src.api.middleware.rate_limiter import get_redis


class ConcurrencyLimitExceeded(Exception):
    """Raised by engine.py's submit_task when the tenant (or the calling
    user's fair share of it) is already at its concurrent-task cap."""

    def __init__(self, tenant_max: int, user_max: int | None = None):
        self.tenant_max = tenant_max
        self.user_max = user_max
        detail = f"tenant is already running its max of {tenant_max} concurrent tasks"
        if user_max is not None:
            detail += f" (your fair share is {user_max})"
        super().__init__(detail)


_SLOT_TTL_SECONDS = 3600  # well above CircuitBreakerConfig's wall-clock cap; a safety net, not the primary release path

_ACQUIRE_SCRIPT = """
local tenant_key = KEYS[1]
local user_key = KEYS[2]
local task_id = ARGV[1]
local tenant_max = tonumber(ARGV[2])
local user_max = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local has_user = ARGV[5] == '1'

local tenant_count = redis.call('SCARD', tenant_key)
if tenant_count >= tenant_max then
    return 0
end

if has_user then
    local user_count = redis.call('SCARD', user_key)
    if user_count >= user_max then
        return 0
    end
end

redis.call('SADD', tenant_key, task_id)
redis.call('EXPIRE', tenant_key, ttl)
if has_user then
    redis.call('SADD', user_key, task_id)
    redis.call('EXPIRE', user_key, ttl)
end
return 1
"""


def _tenant_key(tenant_id: UUID) -> str:
    return f"keystone:concurrency:tenant:{tenant_id}"


def _user_key(tenant_id: UUID, user_id: UUID) -> str:
    return f"keystone:concurrency:user:{tenant_id}:{user_id}"


def per_user_fair_share(tenant_max: int) -> int:
    """No single user may hold more than half the tenant's capacity, rounded up."""
    return max(1, math.ceil(tenant_max / 2))


async def try_acquire_task_slot(
    tenant_id: UUID,
    task_id: UUID,
    tenant_max: int,
    user_id: UUID | None = None,
) -> bool:
    """Atomically claims one of the tenant's (and, if `user_id` is given, that
    user's fair-share of) concurrency slots for `task_id`. Returns False
    without side effects if either cap is already at capacity."""
    r = await get_redis()
    user_max = per_user_fair_share(tenant_max)
    acquired = await r.eval(
        _ACQUIRE_SCRIPT,
        2,
        _tenant_key(tenant_id),
        _user_key(tenant_id, user_id) if user_id else "keystone:concurrency:no-user",
        str(task_id),
        str(tenant_max),
        str(user_max),
        str(_SLOT_TTL_SECONDS),
        "1" if user_id else "0",
    )
    return bool(acquired)


async def release_task_slot(tenant_id: UUID, task_id: UUID, user_id: UUID | None = None) -> None:
    """Idempotent — safe to call even if the slot was never acquired (e.g. the
    task was rejected) or already released."""
    r = await get_redis()
    await r.srem(_tenant_key(tenant_id), str(task_id))
    if user_id:
        await r.srem(_user_key(tenant_id, user_id), str(task_id))


async def current_tenant_concurrency(tenant_id: UUID) -> int:
    """How many task slots the tenant currently holds — for observability/tests."""
    r = await get_redis()
    return int(await r.scard(_tenant_key(tenant_id)))
