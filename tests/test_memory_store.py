# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/memory/store.py — against a real Postgres (the same one Alembic migrations were
verified against for real), not a mock of the ORM/DB layer. See tests/conftest.py's requires_integration_env.
"""

from __future__ import annotations

import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import Tenant, TenantTier
from src.memory.store import (
    create_memory,
    get_memory,
    list_memories,
    recall,
    render_memories_for_prompt,
    set_pinned,
    set_status,
)

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="memory-store-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def test_create_memory_without_repository_is_tenant_scope(tenant_id):
    record = await create_memory(tenant_id, "Always use type hints.", kind="convention")
    assert record.scope == "tenant"
    assert record.repository is None


async def test_create_memory_with_repository_is_repo_scope(tenant_id):
    record = await create_memory(
        tenant_id, "This repo uses tabs, not spaces.", kind="convention", repository="https://git/x/y"
    )
    assert record.scope == "repo"
    assert record.repository == "https://git/x/y"


async def test_get_memory_round_trips(tenant_id):
    created = await create_memory(tenant_id, "Prefer dataclasses over TypedDict.", kind="preference")
    fetched = await get_memory(created.id)
    assert fetched is not None
    assert fetched.content == "Prefer dataclasses over TypedDict."


async def test_get_memory_missing_returns_none():
    assert await get_memory(uuid.uuid4()) is None


async def test_list_memories_filters_by_repository(tenant_id):
    await create_memory(tenant_id, "tenant-wide fact", kind="fact")
    await create_memory(tenant_id, "repo A fact", kind="fact", repository="https://git/a")
    await create_memory(tenant_id, "repo B fact", kind="fact", repository="https://git/b")

    repo_a = await list_memories(tenant_id, repository="https://git/a")
    assert [m.content for m in repo_a] == ["repo A fact"]

    all_tenant = await list_memories(tenant_id)
    assert len(all_tenant) == 3


async def test_set_status_approves_a_proposed_memory(tenant_id):
    record = await create_memory(tenant_id, "auto-proposed thing", kind="fact", source="auto", status="proposed")
    assert record.status == "proposed"
    approved = await set_status(record.id, "approved")
    assert approved is not None
    assert approved.status == "approved"


async def test_set_status_forget_is_a_soft_delete(tenant_id):
    record = await create_memory(tenant_id, "temporary note", kind="fact")
    forgotten = await set_status(record.id, "forgotten")
    assert forgotten is not None
    assert forgotten.status == "forgotten"
    # Still fetchable directly — forgetting doesn't erase the row/attribution.
    assert await get_memory(record.id) is not None


async def test_set_pinned_toggles(tenant_id):
    record = await create_memory(tenant_id, "pin me", kind="preference")
    assert record.pinned is False
    pinned = await set_pinned(record.id, True)
    assert pinned is not None
    assert pinned.pinned is True
    unpinned = await set_pinned(record.id, False)
    assert unpinned is not None
    assert unpinned.pinned is False


async def test_recall_excludes_proposed_and_forgotten_memories(tenant_id):
    approved = await create_memory(tenant_id, "approved memory about caching", kind="fact")
    await create_memory(tenant_id, "proposed memory about caching", kind="fact", status="proposed")
    forgotten = await create_memory(tenant_id, "forgotten memory about caching", kind="fact")
    await set_status(forgotten.id, "forgotten")

    recalled = await recall(tenant_id, None, "caching")
    contents = {m.content for m in recalled}
    assert approved.content in contents
    assert "proposed memory about caching" not in contents
    assert "forgotten memory about caching" not in contents


async def test_recall_pinned_memory_included_even_without_keyword_match(tenant_id):
    pinned = await create_memory(tenant_id, "Never use eval() anywhere in this codebase.", kind="avoid")
    await set_pinned(pinned.id, True)

    recalled = await recall(tenant_id, None, "how do I add pagination to an endpoint")
    assert any(m.id == pinned.id for m in recalled)


async def test_recall_repo_scope_ranks_before_tenant_scope_on_same_topic(tenant_id):
    tenant_wide = await create_memory(tenant_id, "logging conventions apply broadly", kind="convention")
    repo_specific = await create_memory(
        tenant_id, "logging conventions for this specific repo", kind="convention", repository="https://git/x"
    )

    recalled = await recall(tenant_id, "https://git/x", "logging conventions")
    ids_in_order = [m.id for m in recalled]
    assert ids_in_order.index(repo_specific.id) < ids_in_order.index(tenant_wide.id)


async def test_recall_tenant_scope_only_when_no_repository_given(tenant_id):
    await create_memory(tenant_id, "repo-only convention", kind="convention", repository="https://git/x")
    tenant_wide = await create_memory(tenant_id, "tenant-wide convention", kind="convention")

    recalled = await recall(tenant_id, None, "convention")
    contents = {m.content for m in recalled}
    assert tenant_wide.content in contents
    assert "repo-only convention" not in contents


async def test_recall_respects_token_budget(tenant_id):
    # Each memory's content is long enough that only a couple fit in a tiny budget.
    long_text = "x " * 500
    for i in range(5):
        await create_memory(tenant_id, f"{long_text} memory number {i}", kind="fact")

    recalled = await recall(tenant_id, None, "memory", budget_tokens=50)
    assert 1 <= len(recalled) < 5


async def test_recall_increments_hit_count(tenant_id):
    record = await create_memory(tenant_id, "hit count target about widgets", kind="fact")
    assert record.hit_count == 0

    await recall(tenant_id, None, "widgets")
    await recall(tenant_id, None, "widgets")

    updated = await get_memory(record.id)
    assert updated is not None
    assert updated.hit_count == 2


def test_render_memories_for_prompt_empty_list_returns_empty_string():
    assert render_memories_for_prompt([]) == ""


async def test_render_memories_for_prompt_formats_scope_and_kind(tenant_id):
    record = await create_memory(tenant_id, "use ruff", kind="convention", repository="https://git/x")
    text = render_memories_for_prompt([record])
    assert "convention/repo" in text
    assert "use ruff" in text
