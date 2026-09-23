# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The usage ledger (src/billing/ledger.py) against a real Postgres: atomic
hour-bucket upserts under concurrency, dollars from the operator's prices,
per-day/per-role summaries, and the per-task path from an execution trace.
Requires DATABASE_URL (tests/conftest.py's requires_integration_env).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from src.billing.ledger import cost_usd, hour_bucket, record_task_usage, record_usage, usage_summary
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import Tenant, TenantTier, UsageRecord

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="ledger-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 0.40, "reasoning": 1.20}')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_hour_bucket_and_cost(priced):
    assert hour_bucket(datetime(2026, 9, 22, 17, 42, 9, tzinfo=UTC)) == datetime(2026, 9, 22, 17, tzinfo=UTC)
    assert cost_usd("coding", 1_000_000) == 0.40
    assert cost_usd("reasoning", 500_000) == 0.60
    assert cost_usd("coding_fallback", 1_000_000) == 0.0  # unpriced role: 0, not invented


async def test_concurrent_records_land_in_one_bucket_without_losing_an_increment(tenant_id, priced):
    key_id = None  # a keyless caller (an agent task) — NULL api_key_id is one bucket, not many
    await asyncio.gather(
        *[
            record_usage(tenant_id, model_role="coding", prompt_tokens=100, completion_tokens=10, api_key_id=key_id)
            for _ in range(25)
        ]
    )
    async with get_db_context() as db:
        rows = (await db.execute(select(UsageRecord).where(UsageRecord.tenant_id == tenant_id))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens, row.request_count) == (2500, 250, 2750, 25)
    assert row.estimated_cost_usd == pytest.approx(2750 / 1_000_000 * 0.40)


async def test_summary_groups_by_day_and_role_and_reports_pricing(tenant_id, priced):
    await record_usage(tenant_id, model_role="coding", prompt_tokens=1000, completion_tokens=200)
    await record_usage(tenant_id, model_role="reasoning", prompt_tokens=300, completion_tokens=100)
    summary = await usage_summary(tenant_id, days=7)
    assert summary["pricing_configured"] is True
    by_role = {r["model_role"]: r for r in summary["rows"]}
    assert by_role["coding"]["total_tokens"] == 1200 and by_role["reasoning"]["total_tokens"] == 400
    assert summary["totals"]["total_tokens"] == 1600
    assert summary["totals"]["estimated_cost_usd"] == pytest.approx(1200 / 1e6 * 0.40 + 400 / 1e6 * 1.20)


async def test_summary_without_prices_says_so(tenant_id, monkeypatch):
    monkeypatch.delenv("MODEL_PRICES_PER_MILLION", raising=False)
    get_settings.cache_clear()
    try:
        await record_usage(tenant_id, model_role="coding", prompt_tokens=10, completion_tokens=5)
        summary = await usage_summary(tenant_id)
        assert summary["pricing_configured"] is False
        assert summary["totals"]["total_tokens"] == 15 and summary["totals"]["estimated_cost_usd"] == 0.0
    finally:
        get_settings.cache_clear()


async def test_task_usage_is_recorded_per_role_from_the_trace(tenant_id, priced):
    trace = [
        {"phase": "planning", "model_role": "coding", "prompt_tokens": 500, "completion_tokens": 50},
        {"phase": "coding", "model_role": "coding", "prompt_tokens": 2000, "completion_tokens": 400},
        {"phase": "review", "model_role": "reasoning", "prompt_tokens": 800, "completion_tokens": 100},
        {"phase": "testing", "model_role": "sandbox", "prompt_tokens": 0, "completion_tokens": 0},
    ]
    await record_task_usage(tenant_id, None, trace)
    summary = await usage_summary(tenant_id)
    by_role = {r["model_role"]: r for r in summary["rows"]}
    assert set(by_role) == {"coding", "reasoning"}
    assert by_role["coding"]["total_tokens"] == 2950 and by_role["reasoning"]["total_tokens"] == 900
    assert by_role["coding"]["request_count"] == 0  # tasks are billed by tokens, not counted as gateway requests
