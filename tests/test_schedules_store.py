# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/orchestrator/schedules.py — against a real Postgres, no mock of
the ORM/DB layer. A triggered schedule submits a real task through the real
src/orchestrator/engine.py (no repository_url, matching the same bare-minimum-task pattern
tests/test_users_and_roles.py already uses to exercise POST /v1/keystone/tasks for real without
needing a full agent run to complete). See tests/conftest.py's requires_integration_env.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import APIKey, ScheduleTrigger, Tenant, TenantTier
from src.orchestrator.schedules import (
    InvalidCronExpression,
    create_schedule,
    delete_schedule,
    get_schedule,
    get_schedule_by_webhook_token,
    list_schedules,
    poll_and_trigger_due_schedules,
    set_enabled,
    trigger_schedule,
)

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="schedules-store-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
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
        db.add(
            APIKey(
                id=kid,
                tenant_id=tenant_id,
                name="schedules-store-key",
                key_prefix=prefix,
                key_hash=key_hash,
                scopes=["agent"],
            )
        )
        await db.flush()
    return kid


async def test_create_cron_schedule_computes_next_run_at(tenant_id, api_key_id):
    record = await create_schedule(
        tenant_id,
        api_key_id,
        "Nightly repo sweep",
        ScheduleTrigger.CRON,
        "A real recurring task description, long enough to pass validation.",
        cron_expression="0 2 * * *",
    )
    assert record.trigger_type == ScheduleTrigger.CRON
    assert record.cron_expression == "0 2 * * *"
    assert record.webhook_token is None
    assert record.next_run_at is not None
    assert record.next_run_at > datetime.now(UTC)


async def test_create_cron_schedule_without_expression_raises(tenant_id, api_key_id):
    with pytest.raises(InvalidCronExpression):
        await create_schedule(
            tenant_id, api_key_id, "Bad", ScheduleTrigger.CRON, "A real task description, long enough."
        )


async def test_create_cron_schedule_with_invalid_expression_raises(tenant_id, api_key_id):
    with pytest.raises(InvalidCronExpression):
        await create_schedule(
            tenant_id,
            api_key_id,
            "Bad",
            ScheduleTrigger.CRON,
            "A real task description, long enough.",
            cron_expression="not a cron expression",
        )


async def test_create_webhook_schedule_generates_a_real_token(tenant_id, api_key_id):
    record = await create_schedule(
        tenant_id,
        api_key_id,
        "CI-triggered task",
        ScheduleTrigger.WEBHOOK,
        "A real webhook-triggered task description, long enough to pass validation.",
    )
    assert record.trigger_type == ScheduleTrigger.WEBHOOK
    assert record.cron_expression is None
    assert record.webhook_token is not None
    assert len(record.webhook_token) > 20

    fetched = await get_schedule_by_webhook_token(record.webhook_token)
    assert fetched is not None
    assert fetched.id == record.id


async def test_get_schedule_by_webhook_token_returns_none_for_unknown_token():
    assert await get_schedule_by_webhook_token("not-a-real-token") is None


async def test_list_schedules_filters_by_tenant_and_enabled(tenant_id, api_key_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(
                id=other_tenant, name="other-schedules-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE
            )
        )
        await db.flush()
    try:
        await create_schedule(
            tenant_id,
            api_key_id,
            "Enabled one",
            ScheduleTrigger.WEBHOOK,
            "A real task description, long enough to pass validation.",
            enabled=True,
        )
        await create_schedule(
            tenant_id,
            api_key_id,
            "Disabled one",
            ScheduleTrigger.WEBHOOK,
            "A real task description, long enough to pass validation.",
            enabled=False,
        )
        await create_schedule(
            other_tenant,
            api_key_id,
            "Someone else's",
            ScheduleTrigger.WEBHOOK,
            "A real task description, long enough to pass validation.",
        )

        all_for_tenant = await list_schedules(tenant_id)
        assert {s.name for s in all_for_tenant} == {"Enabled one", "Disabled one"}

        only_enabled = await list_schedules(tenant_id, enabled=True)
        assert {s.name for s in only_enabled} == {"Enabled one"}
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


async def test_set_enabled_toggles_a_real_row(tenant_id, api_key_id):
    created = await create_schedule(
        tenant_id,
        api_key_id,
        "Test",
        ScheduleTrigger.WEBHOOK,
        "A real task description, long enough to pass validation.",
        enabled=True,
    )
    disabled = await set_enabled(created.id, False)
    assert disabled is not None
    assert disabled.enabled is False
    re_enabled = await set_enabled(created.id, True)
    assert re_enabled is not None
    assert re_enabled.enabled is True


async def test_delete_schedule_really_removes_the_row(tenant_id, api_key_id):
    created = await create_schedule(
        tenant_id,
        api_key_id,
        "Test",
        ScheduleTrigger.WEBHOOK,
        "A real task description, long enough to pass validation.",
    )
    assert await delete_schedule(created.id) is True
    assert await get_schedule(created.id) is None
    assert await delete_schedule(created.id) is False  # already gone


async def test_trigger_schedule_submits_a_real_task_and_advances_bookkeeping(tenant_id, api_key_id):
    created = await create_schedule(
        tenant_id,
        api_key_id,
        "Manual trigger test",
        ScheduleTrigger.CRON,
        "A real task description, long enough to pass validation.",
        cron_expression="0 0 1 1 *",  # once a year — never due on its own during this test
    )
    first_next_run = created.next_run_at
    assert first_next_run is not None

    task_id = await trigger_schedule(created.id)
    assert task_id is not None

    updated = await get_schedule(created.id)
    assert updated is not None
    assert updated.last_task_id == task_id
    assert updated.last_triggered_at is not None
    assert updated.next_run_at is not None
    # Recomputed from "now" after triggering, not left stale — for this once-a-year expression
    # the real next occurrence is still the same Jan 1, so equal is the correct outcome here;
    # what matters is it was recomputed (>=), not that it necessarily moved forward.
    assert updated.next_run_at >= first_next_run


async def test_trigger_schedule_raises_for_a_disabled_schedule(tenant_id, api_key_id):
    created = await create_schedule(
        tenant_id,
        api_key_id,
        "Disabled",
        ScheduleTrigger.WEBHOOK,
        "A real task description, long enough to pass validation.",
        enabled=False,
    )
    with pytest.raises(LookupError):
        await trigger_schedule(created.id)


async def test_trigger_schedule_raises_for_an_unknown_schedule():
    with pytest.raises(LookupError):
        await trigger_schedule(uuid.uuid4())


async def test_poll_and_trigger_due_schedules_only_triggers_due_cron_schedules(tenant_id, api_key_id):
    due = await create_schedule(
        tenant_id,
        api_key_id,
        "Due now",
        ScheduleTrigger.CRON,
        "A real task description, long enough to pass validation.",
        cron_expression="0 0 1 1 *",
    )
    not_due = await create_schedule(
        tenant_id,
        api_key_id,
        "Not due yet",
        ScheduleTrigger.CRON,
        "A real task description, long enough to pass validation.",
        cron_expression="0 0 1 1 *",
    )
    webhook_only = await create_schedule(
        tenant_id,
        api_key_id,
        "Webhook only",
        ScheduleTrigger.WEBHOOK,
        "A real task description, long enough to pass validation.",
    )

    # Force "due" into the past directly on the row — cron_expression alone can't produce a
    # next_run_at in the past for a real (future) test run.
    async with get_db_context() as db:
        from src.db.models import TaskSchedule

        row = await db.get(TaskSchedule, due.id)
        assert row is not None
        row.next_run_at = datetime.now(UTC) - timedelta(minutes=5)
        await db.flush()

    triggered = await poll_and_trigger_due_schedules()
    triggered_ids = {t["schedule_id"] for t in triggered}
    assert due.id in triggered_ids
    assert not_due.id not in triggered_ids
    assert webhook_only.id not in triggered_ids

    updated_due = await get_schedule(due.id)
    assert updated_due is not None
    assert updated_due.last_task_id is not None
