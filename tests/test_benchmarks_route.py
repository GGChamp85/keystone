# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""`GET /v1/keystone/benchmarks` over the real route with real Postgres: rows persisted the way the
benchmark runner persists them come back as per-backend totals and a task-by-backend matrix of the
latest run per task, filtered by `days`, and the route needs the inference scope."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from sqlalchemy import delete

from benchmarks.runs import persist_results
from src.api.middleware.auth import generate_api_key
from src.api.routes.benchmarks import summarize
from src.db.connection import get_db_context
from src.db.models import APIKey, BenchmarkRun, Tenant, TenantTier
from src.main import create_app

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]

SUITE = f"repo-test-{uuid.uuid4().hex[:6]}"


@pytest.fixture
async def api_key():
    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="bench", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["inference"]))
    yield full_key
    async with get_db_context() as db:
        await db.execute(delete(BenchmarkRun).where(BenchmarkRun.suite == SUITE))
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def _result(task_id: str, solved: bool, ms: int, tokens: int, status: str = "completed") -> dict:
    return {
        "task_id": task_id,
        "status": status,
        "solved": solved,
        "fail_to_pass": {f"tests/test_{task_id}.py::test_bug": solved},
        "pass_to_pass": {f"tests/test_{task_id}.py::test_ok": True},
        "total_prompt_tokens": tokens,
        "total_completion_tokens": tokens // 10,
        "duration_ms": ms,
        "pr_url": f"https://git.test/pr/{task_id}" if solved else None,
    }


async def test_benchmark_comparison_reflects_the_latest_persisted_run_per_task(api_key):
    await persist_results(
        [_result("token_bucket", False, 90_000, 5000), _result("lru_cache_recency", True, 60_000, 4000)],
        backend_label="base",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        suite=SUITE,
    )
    await persist_results(
        [_result("token_bucket", True, 70_000, 6000)], backend_label="base", model_id=None, suite=SUITE
    )
    await persist_results(
        [_result("token_bucket", True, 40_000, 9000), _result("lru_cache_recency", True, 30_000, 8000)],
        backend_label="frontier",
        model_id="frontier-model",
        suite=SUITE,
    )
    async with running_client() as client:
        resp = await client.get(
            f"/v1/keystone/benchmarks?suite={SUITE}", headers={"Authorization": f"Bearer {api_key}"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (await client.get(f"/v1/keystone/benchmarks?suite={SUITE}")).status_code == 401
        recent = await client.get(
            f"/v1/keystone/benchmarks?suite={SUITE}&days=1", headers={"Authorization": f"Bearer {api_key}"}
        )
        assert recent.json()["runs_considered"] == 5

    assert body["runs_considered"] == 5
    by_label = {b["label"]: b for b in body["backends"]}
    assert by_label["base"]["tasks_run"] == 2 and by_label["base"]["solved"] == 2  # the later token_bucket run wins
    assert by_label["base"]["solve_rate"] == 1.0 and by_label["base"]["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert by_label["frontier"]["avg_duration_ms"] == 35_000
    tasks = {t["task_id"]: t for t in body["tasks"]}
    assert tasks["token_bucket"]["language"] == "python"
    assert tasks["token_bucket"]["results"]["base"]["duration_ms"] == 70_000  # latest, not the first
    assert tasks["token_bucket"]["results"]["frontier"]["pr_url"].endswith("/token_bucket")
    assert tasks["lru_cache_recency"]["results"]["base"]["solved"] is True


def test_summarize_of_nothing_is_empty_not_an_error():
    body = summarize([])
    assert body["backends"] == [] and body["tasks"] == [] and body["runs_considered"] == 0
