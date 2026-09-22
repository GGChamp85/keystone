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


@pytest.fixture(autouse=True)
def _inference_endpoints_unreachable(monkeypatch):
    """
    The submit_task tests below assert what happens when a task's real
    background execution fails at the inference endpoint: the concurrency
    slot must still be released, and the task must reach a terminal state
    quickly. They used to get that for free because no model endpoint was
    reachable in any test environment. Now that CI runs a real model
    backend for the whole job (VLLM_CODING_URL → the demo-model llama.cpp
    server), a submitted task would genuinely run the agent loop instead,
    outlive the test, and leak a background task across event loops. So
    the assumption is enforced here explicitly rather than inherited from
    the environment: every role points at a port nothing listens on, and
    the client cache is cleared so no previously built client survives.
    """
    from src.config import get_settings
    from src.inference import client as inference_client

    for var in ("VLLM_CODING_URL", "VLLM_CODING_FALLBACK_URL", "VLLM_REASONING_URL"):
        monkeypatch.setenv(var, "http://127.0.0.1:1/v1")
    get_settings.cache_clear()
    inference_client._clients.clear()
    yield
    get_settings.cache_clear()
    inference_client._clients.clear()


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
    """No repository_url -> the background execution fails at the inference
    endpoint (made unreachable for every role by the autouse fixture above,
    so this holds even when the environment has a real model backend) and
    hits _execute_task's `finally`, which must release the slot — proving
    the real acquire (in submit_task) -> release (in _execute_task) wiring,
    not just the Redis primitives in isolation above. When a real Qdrant is configured
    (QDRANT_HOST), RAG retrieval runs for real first — including, on a cold
    process, loading the real embedding model — before the inference call
    fails, so the wait allows for that rather than assuming an instant
    failure."""
    engine = get_keystone_engine()
    task_id = await engine.submit_task(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        task_description="A real task that will fail — no repository_url, no reachable inference endpoint.",
    )
    assert task_id is not None
    assert await current_tenant_concurrency(tenant_id) == 1

    for _ in range(300):  # up to ~30s — covers a cold embedding-model load when Qdrant is real
        if await current_tenant_concurrency(tenant_id) == 0:
            break
        await asyncio.sleep(0.1)
    assert await current_tenant_concurrency(tenant_id) == 0


async def test_submit_task_with_auto_model_resolves_to_a_real_concrete_role_in_the_db(tenant_id, api_key_id):
    """model="auto" must never reach the DB — AgentTask.model_role is a
    real enum column (coding|coding_fallback|reasoning) that can't hold
    "auto" at all, so this also proves engine.submit_task resolves it
    before persisting, not just that the resolver function works in
    isolation."""
    from src.db.connection import get_db_context
    from src.db.models import AgentTask

    engine = get_keystone_engine()
    task_id = await engine.submit_task(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        task_description="Review this code for security issues before we ship it.",
        model="auto",
    )
    async with get_db_context() as db:
        task = await db.get(AgentTask, task_id)
    assert task.model_role.value == "reasoning"  # classify_task_to_role's real reasoning-keyword match

    for _ in range(300):
        if await current_tenant_concurrency(tenant_id) == 0:
            break
        await asyncio.sleep(0.1)


async def test_submit_task_with_auto_model_routes_a_simple_task_to_coding_fallback_when_enabled(
    tenant_id, api_key_id, monkeypatch
):
    from src.config import get_settings
    from src.db.connection import get_db_context
    from src.db.models import AgentTask

    monkeypatch.setenv("TASK_COMPLEXITY_ROUTING_ENABLED", "true")
    get_settings.cache_clear()
    try:
        engine = get_keystone_engine()
        task_id = await engine.submit_task(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            task_description="Fix a typo in the README",
            model="auto",
        )
        async with get_db_context() as db:
            task = await db.get(AgentTask, task_id)
        assert task.model_role.value == "coding_fallback"

        for _ in range(300):
            if await current_tenant_concurrency(tenant_id) == 0:
                break
            await asyncio.sleep(0.1)
    finally:
        get_settings.cache_clear()


def test_per_user_fair_share_is_uncapped_when_the_tenant_is_uncapped():
    assert per_user_fair_share(0) == 0


async def test_zero_cap_means_unlimited_for_the_tenant_and_every_user(tenant_id):
    """The default: no policy cap. Many tasks from one user and from several users all acquire,
    membership is still tracked (so release/observability work), and nothing is ever rejected."""
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    tasks = [uuid.uuid4() for _ in range(12)]
    for i, task in enumerate(tasks):
        assert await try_acquire_task_slot(tenant_id, task, tenant_max=0, user_id=user_a if i % 3 else user_b) is True
    assert await current_tenant_concurrency(tenant_id) == 12
    for i, task in enumerate(tasks):
        await release_task_slot(tenant_id, task, user_id=user_a if i % 3 else user_b)
    assert await current_tenant_concurrency(tenant_id) == 0
