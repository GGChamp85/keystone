# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/orchestrator/concurrency.py — a real Redis
(tests/conftest.py's requires_integration_env), no mock of the Lua script
or the atomicity it provides. Also covers slugify_for_branch
(src/orchestrator/nodes/_shared.py) and engine.py's real submit_task/
_execute_task wiring: the slot is acquired before a task's DB row exists
and released once the (real, if fast-failing on an unreachable inference
endpoint) background execution reaches its terminal `finally`.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from src.api.middleware.auth import generate_api_key
from src.api.middleware.rate_limiter import get_redis
from src.db.connection import get_db_context
from src.db.models import APIKey, Tenant, TenantTier
from src.orchestrator.concurrency import (
    ConcurrencyLimitExceeded,
    current_tenant_concurrency,
    per_user_fair_share,
    release_task_slot,
    try_acquire_task_slot,
)
from src.orchestrator.engine import get_keystone_engine
from src.orchestrator.nodes._shared import slugify_for_branch

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(
                id=tid,
                name="concurrency-test",
                email=f"{tid}@test.dev",
                tier=TenantTier.FREE,
                max_concurrent_agents=2,
            )
        )
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
async def api_key_id(tenant_id):
    _full_key, prefix, key_hash = generate_api_key()
    kid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(APIKey(id=kid, tenant_id=tenant_id, name="concurrency-test-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return kid


@pytest.fixture(autouse=True)
async def _clean_redis_slots(tenant_id):
    yield
    r = await get_redis()
    # best-effort cleanup — tests use fresh random tenant/task ids each run,
    # but a failed assertion mid-test could leave a slot held
    async for key in r.scan_iter(match=f"keystone:concurrency:*{tenant_id}*"):
        await r.delete(key)


def test_slugify_for_branch_produces_a_git_ref_safe_slug():
    email_local_part = "Ada.Lovelace+work@x.com".partition("@")[0]
    assert slugify_for_branch(email_local_part) == "ada-lovelace-work"
    assert slugify_for_branch("  Weird   Name!! ") == "weird-name"
    assert slugify_for_branch("") == "user"


def test_per_user_fair_share_is_half_rounded_up_minimum_one():
    assert per_user_fair_share(1) == 1
    assert per_user_fair_share(2) == 1
    assert per_user_fair_share(3) == 2
    assert per_user_fair_share(4) == 2
    assert per_user_fair_share(5) == 3


async def test_acquire_respects_the_tenant_cap(tenant_id):
    t1, t2, t3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert await try_acquire_task_slot(tenant_id, t1, tenant_max=2) is True
    assert await try_acquire_task_slot(tenant_id, t2, tenant_max=2) is True
    assert await current_tenant_concurrency(tenant_id) == 2

    assert await try_acquire_task_slot(tenant_id, t3, tenant_max=2) is False
    assert await current_tenant_concurrency(tenant_id) == 2  # rejected acquire has no side effect

    await release_task_slot(tenant_id, t1)
    assert await current_tenant_concurrency(tenant_id) == 1
    assert await try_acquire_task_slot(tenant_id, t3, tenant_max=2) is True
    assert await current_tenant_concurrency(tenant_id) == 2


async def test_release_is_idempotent(tenant_id):
    task_id = uuid.uuid4()
    assert await try_acquire_task_slot(tenant_id, task_id, tenant_max=2) is True
    await release_task_slot(tenant_id, task_id)
    await release_task_slot(tenant_id, task_id)  # second release — must not raise or go negative
    assert await current_tenant_concurrency(tenant_id) == 0


async def test_per_user_fairness_blocks_one_user_from_taking_the_whole_tenant_cap(tenant_id):
    """tenant_max=2 -> per_user_fair_share=1, so user A's second task is rejected
    even though the tenant itself still has a free slot for user B."""
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    a_task_1, a_task_2, b_task_1 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    assert await try_acquire_task_slot(tenant_id, a_task_1, tenant_max=2, user_id=user_a) is True
    assert await try_acquire_task_slot(tenant_id, a_task_2, tenant_max=2, user_id=user_a) is False
    assert await current_tenant_concurrency(tenant_id) == 1  # user A's rejected 2nd task took no tenant slot

    assert await try_acquire_task_slot(tenant_id, b_task_1, tenant_max=2, user_id=user_b) is True
    assert await current_tenant_concurrency(tenant_id) == 2

    await release_task_slot(tenant_id, a_task_1, user_id=user_a)
    assert await try_acquire_task_slot(tenant_id, a_task_2, tenant_max=2, user_id=user_a) is True


async def test_submit_task_raises_when_a_slot_is_already_held(tenant_id, api_key_id):
    """A slot pre-occupied by another (simulated in-flight) task should make a
    real engine.submit_task call fail closed with ConcurrencyLimitExceeded —
    the exact path src/api/routes/agents.py converts to HTTP 429."""
    occupying_task = uuid.uuid4()
    assert await try_acquire_task_slot(tenant_id, occupying_task, tenant_max=2) is True
    assert await try_acquire_task_slot(tenant_id, uuid.uuid4(), tenant_max=2) is True  # fill the 2-slot cap

    engine = get_keystone_engine()
    with pytest.raises(ConcurrencyLimitExceeded):
        await engine.submit_task(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            task_description="Should be rejected — tenant is at its concurrency cap.",
        )


async def test_submit_task_acquires_and_releases_a_real_slot_end_to_end(tenant_id, api_key_id):
    """No repository_url -> the background execution fails fast at the (in
    this test env, unreachable) inference endpoint and hits _execute_task's
    `finally`, which must release the slot — proving the real acquire (in
    submit_task) -> release (in _execute_task) wiring, not just the Redis
    primitives in isolation above."""
    engine = get_keystone_engine()
    task_id = await engine.submit_task(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        task_description="A real task that will fail fast — no repository_url, no reachable inference endpoint.",
    )
    assert task_id is not None
    assert await current_tenant_concurrency(tenant_id) == 1

    for _ in range(50):  # up to ~5s for the background task to fail and release
        if await current_tenant_concurrency(tenant_id) == 0:
            break
        await asyncio.sleep(0.1)
    assert await current_tenant_concurrency(tenant_id) == 0
