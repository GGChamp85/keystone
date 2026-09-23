# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Pre-task cost estimate and per-task cost breakdown (src/orchestrator/cost_estimate.py) against real
Postgres: with no completed task anywhere, the estimate uses the documented fallback; once a tenant
has completed tasks, its own average drives the estimate; a tenant with none but the platform with some
uses the platform average. The breakdown groups a real execution_trace by phase and role and prices it
from MODEL_PRICES_PER_MILLION, matching what the durable ledger would have recorded for the same trace.
Also the real routes: POST /v1/keystone/tasks/estimate and the cost_breakdown field on GET .../tasks/{id}.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AgentTask, APIKey, TaskStatus, Tenant, TenantTier
from src.main import create_app
from src.orchestrator.cost_estimate import estimate_task_cost, task_cost_breakdown

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def tenant_and_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="cost-estimate", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["agent"]))
    yield tid, full_key
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _completed_task(tenant_id, prompt_tokens: int, completion_tokens: int) -> None:
    async with get_db_context() as db:
        db.add(
            AgentTask(
                tenant_id=tenant_id,
                task_description="a completed task with real recorded token totals for the estimate to learn from",
                status=TaskStatus.COMPLETED,
                total_prompt_tokens=prompt_tokens,
                total_completion_tokens=completion_tokens,
            )
        )


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_estimate_uses_this_tenants_own_history_when_it_has_any(tenant_and_key, monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 10.0}')
    get_settings.cache_clear()
    tid, _ = tenant_and_key
    await _completed_task(tid, 8000, 2000)
    await _completed_task(tid, 12000, 4000)
    try:
        est = await estimate_task_cost("Fix the off-by-one error in the pagination cursor", tenant_id=tid)
    finally:
        get_settings.cache_clear()
    assert "this tenant's own 2 completed task(s)" in est.estimate_basis
    assert est.sample_size == 2
    # mean of (8000,12000) and (2000,4000)
    assert est.estimated_prompt_tokens == 10000
    assert est.estimated_completion_tokens == 3000
    assert est.pricing_configured is True
    assert est.estimated_cost_usd == pytest.approx((10000 + 3000) * 10.0 / 1_000_000)


async def test_estimate_falls_back_to_platform_average_then_to_the_documented_default(tenant_and_key, monkeypatch):
    monkeypatch.delenv("MODEL_PRICES_PER_MILLION", raising=False)
    get_settings.cache_clear()
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tenant_id, name="other", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE))
    await _completed_task(other_tenant_id, 20000, 5000)

    tid, _ = tenant_and_key  # this tenant has no completed tasks of its own
    try:
        platform = await estimate_task_cost("Add a retry with backoff to the HTTP client", tenant_id=tid)
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)
        get_settings.cache_clear()
    assert "completed task(s) across this deployment" in platform.estimate_basis
    assert platform.pricing_configured is False and platform.estimated_cost_usd is None
    assert "no price configured" in platform.estimate_basis

    # with nothing completed anywhere (a brand-new tenant on the other_tenant's row also gone), the fallback applies
    empty = await estimate_task_cost("Add a retry with backoff to the HTTP client", tenant_id=tid)
    assert empty.sample_size == 0
    assert "rough, documented default, not a measurement" in empty.estimate_basis
    assert empty.estimated_prompt_tokens > 0 and empty.estimated_completion_tokens > 0


async def test_estimate_floor_covers_the_system_prompt_and_a_repository_map_budget(tenant_and_key):
    # A tiny historical average (well below the real system-prompt + description floor) so the floor,
    # not the average, drives the estimate — otherwise a small average would always win via max().
    tid, _ = tenant_and_key
    await _completed_task(tid, 100, 20)
    task = "Add input validation to the signup form"
    no_repo = await estimate_task_cost(task, tenant_id=tid)
    with_repo = await estimate_task_cost(task, tenant_id=tid, repository_url="https://git.test/o/r")
    assert with_repo.estimated_prompt_tokens > no_repo.estimated_prompt_tokens  # the repo map budget adds to the floor
    assert no_repo.estimated_prompt_tokens > 100  # the real system prompt + task tokens exceed the tiny average


async def test_estimate_scales_with_best_of_n(tenant_and_key):
    tid, _ = tenant_and_key
    await _completed_task(tid, 10000, 2000)
    single = await estimate_task_cost("Add input validation to the signup form", tenant_id=tid, best_of_n=1)
    triple = await estimate_task_cost("Add input validation to the signup form", tenant_id=tid, best_of_n=3)
    assert triple.estimated_prompt_tokens == single.estimated_prompt_tokens * 3
    assert triple.best_of_n == 3 and "x3 for best_of_n" in triple.estimate_basis


def test_cost_breakdown_groups_a_real_trace_by_phase_and_role_and_prices_it(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 2.0, "reasoning": 5.0}')
    get_settings.cache_clear()
    trace = [
        {"phase": "planning", "model_role": "reasoning", "prompt_tokens": 1000, "completion_tokens": 200},
        {"phase": "coding", "model_role": "coding", "prompt_tokens": 3000, "completion_tokens": 800},
        {"phase": "coding", "model_role": "coding", "prompt_tokens": 2000, "completion_tokens": 500},
        {"phase": "quality", "model_role": "tools", "prompt_tokens": 0, "completion_tokens": 0},  # no tokens: dropped
    ]
    try:
        body = task_cost_breakdown(trace)
    finally:
        get_settings.cache_clear()
    assert body["pricing_configured"] is True
    rows = {(r["phase"], r["model_role"]): r for r in body["rows"]}
    assert set(rows) == {("planning", "reasoning"), ("coding", "coding")}
    coding = rows[("coding", "coding")]
    assert coding["iterations"] == 2 and coding["prompt_tokens"] == 5000 and coding["completion_tokens"] == 1300
    assert coding["estimated_cost_usd"] == pytest.approx((5000 + 1300) * 2.0 / 1_000_000)
    planning = rows[("planning", "reasoning")]
    assert planning["estimated_cost_usd"] == pytest.approx((1000 + 200) * 5.0 / 1_000_000)
    assert body["totals"]["total_tokens"] == 5000 + 1300 + 1000 + 200
    assert body["totals"]["estimated_cost_usd"] == pytest.approx(
        coding["estimated_cost_usd"] + planning["estimated_cost_usd"]
    )


def test_cost_breakdown_of_an_empty_trace_is_empty_not_an_error():
    body = task_cost_breakdown([])
    assert body["rows"] == [] and body["totals"]["total_tokens"] == 0


async def test_estimate_route_and_task_route_expose_cost_data(tenant_and_key, monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 1.0}')
    get_settings.cache_clear()
    tid, key = tenant_and_key
    await _completed_task(tid, 5000, 1000)
    async with get_db_context() as db:
        task = AgentTask(
            tenant_id=tid,
            task_description="a task whose stored execution_trace the route must price",
            status=TaskStatus.COMPLETED,
            total_prompt_tokens=4000,
            total_completion_tokens=900,
            execution_trace=[
                {"phase": "coding", "model_role": "coding", "prompt_tokens": 4000, "completion_tokens": 900}
            ],
        )
        db.add(task)
        await db.flush()
        task_id = task.id
    try:
        async with running_client() as client:
            headers = {"Authorization": f"Bearer {key}"}
            est = await client.post(
                "/v1/keystone/tasks/estimate", headers=headers, json={"task": "Add a health check endpoint"}
            )
            assert est.status_code == 200, est.text
            body = est.json()
            # this tenant has 2 completed tasks by now (the helper above plus the one seeded below for the
            # breakdown check); the point here is that its own history was used, not the exact count
            assert body["sample_size"] >= 1 and "this tenant's own" in body["estimate_basis"]
            assert body["pricing_configured"] is True
            assert body["estimated_cost_usd"] is not None

            got = await client.get(f"/v1/keystone/tasks/{task_id}", headers=headers)
            assert got.status_code == 200, got.text
            breakdown = got.json()["cost_breakdown"]
            assert breakdown["totals"]["prompt_tokens"] == 4000
            assert breakdown["totals"]["estimated_cost_usd"] == pytest.approx(4900 / 1_000_000)
    finally:
        async with get_db_context() as db:
            row = await db.get(AgentTask, task_id)
            if row is not None:
                await db.delete(row)
        get_settings.cache_clear()
