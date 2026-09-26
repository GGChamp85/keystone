# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/memory/skills.py — against a real Postgres, no mock of the ORM/DB
layer. See tests/conftest.py's requires_integration_env.
"""

from __future__ import annotations

import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import Tenant, TenantTier
from src.memory.skills import (
    create_skill,
    delete_skill,
    get_skill,
    list_skills,
    matching_skills,
    render_skills_for_prompt,
    set_enabled,
)

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="skills-store-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def test_create_skill_defaults_to_enabled_with_no_keywords(tenant_id):
    record = await create_skill(tenant_id, "PCI checklist", "Run the PCI compliance checklist.")
    assert record.enabled is True
    assert record.trigger_keywords == []


async def test_get_skill_returns_a_real_row(tenant_id):
    created = await create_skill(tenant_id, "Test", "content")
    fetched = await get_skill(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.content == "content"


async def test_get_skill_returns_none_for_a_missing_id():
    assert await get_skill(uuid.uuid4()) is None


async def test_list_skills_filters_by_tenant_and_enabled(tenant_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant, name="other-skills-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        await create_skill(tenant_id, "Enabled one", "content", enabled=True)
        await create_skill(tenant_id, "Disabled one", "content", enabled=False)
        await create_skill(other_tenant, "Someone else's", "content")

        all_for_tenant = await list_skills(tenant_id)
        assert {s.name for s in all_for_tenant} == {"Enabled one", "Disabled one"}

        only_enabled = await list_skills(tenant_id, enabled=True)
        assert {s.name for s in only_enabled} == {"Enabled one"}
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


async def test_set_enabled_toggles_a_real_row(tenant_id):
    created = await create_skill(tenant_id, "Test", "content", enabled=True)
    disabled = await set_enabled(created.id, False)
    assert disabled is not None
    assert disabled.enabled is False
    re_enabled = await set_enabled(created.id, True)
    assert re_enabled is not None
    assert re_enabled.enabled is True


async def test_delete_skill_really_removes_the_row(tenant_id):
    created = await create_skill(tenant_id, "Test", "content")
    assert await delete_skill(created.id) is True
    assert await get_skill(created.id) is None
    assert await delete_skill(created.id) is False  # already gone


async def test_matching_skills_with_no_keywords_applies_to_every_task(tenant_id):
    await create_skill(tenant_id, "Always", "Always do this.")
    matches = await matching_skills(tenant_id, "Fix a totally unrelated bug in parser.py")
    assert len(matches) == 1
    assert matches[0].name == "Always"


async def test_matching_skills_with_keywords_only_applies_when_one_appears(tenant_id):
    await create_skill(tenant_id, "Payments", "Run the PCI checklist.", trigger_keywords=["payment", "billing"])
    no_match = await matching_skills(tenant_id, "Fix a typo in the README")
    assert no_match == []

    match = await matching_skills(tenant_id, "Add a new payment method to checkout")
    assert len(match) == 1
    assert match[0].name == "Payments"


async def test_matching_skills_keyword_match_is_case_insensitive(tenant_id):
    await create_skill(tenant_id, "Payments", "content", trigger_keywords=["PAYMENT"])
    match = await matching_skills(tenant_id, "fix the payment flow")
    assert len(match) == 1


async def test_matching_skills_ignores_disabled_skills(tenant_id):
    await create_skill(tenant_id, "Disabled always-on", "content", enabled=False)
    matches = await matching_skills(tenant_id, "anything at all")
    assert matches == []


async def test_render_skills_for_prompt_is_empty_string_for_no_skills():
    assert render_skills_for_prompt([]) == ""


async def test_render_skills_for_prompt_includes_name_and_content(tenant_id):
    skill = await create_skill(tenant_id, "PCI checklist", "Run the PCI compliance checklist.")
    rendered = render_skills_for_prompt([skill])
    assert "## Skills" in rendered
    assert "PCI checklist" in rendered
    assert "Run the PCI compliance checklist." in rendered
