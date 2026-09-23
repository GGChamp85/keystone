# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Persisted benchmark runs (benchmarks/runs.py) and the report rendered from
them (benchmarks/report.py): real rows in a real Postgres, the FK from
model_adapters, and a report whose numbers come only from those rows.
Requires DATABASE_URL (tests/conftest.py's requires_integration_env) for
the DB tests; the rendering tests are pure.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select

from benchmarks.compare import Backend
from benchmarks.report import latest_per_task, render
from benchmarks.runs import load_runs, persist_results
from src.db.connection import get_db_context
from src.db.models import BenchmarkRun, ModelAdapter, Tenant, TenantTier

from .conftest import requires_integration_env


def _run(task_id: str, label: str, solved: bool, created: datetime, **kw) -> BenchmarkRun:
    return BenchmarkRun(
        id=uuid.uuid4(),
        suite="repo",
        task_id=task_id,
        backend_label=label,
        model_id=kw.get("model_id", "m"),
        status=kw.get("status", "completed"),
        solved=solved,
        prompt_tokens=kw.get("prompt", 1000),
        completion_tokens=kw.get("completion", 100),
        duration_ms=kw.get("duration_ms", 60_000),
        pr_url=kw.get("pr_url"),
        created_at=created,
        fail_to_pass={},
        pass_to_pass={},
        details={},
    )


def test_latest_run_per_task_supersedes_older_attempts():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    t1 = datetime(2026, 9, 2, tzinfo=UTC)
    runs = [_run("a", "coding", False, t0), _run("a", "coding", True, t1), _run("b", "coding", False, t1)]
    latest = latest_per_task(runs)
    assert latest["coding"]["a"].solved is True
    assert set(latest["coding"]) == {"a", "b"}


def test_render_reports_solve_rate_per_backend_and_per_task_grid(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("MODEL_PRICES_PER_MILLION", '{"coding": 1.0}')
    get_settings.cache_clear()
    try:
        t = datetime(2026, 9, 2, tzinfo=UTC)
        runs = [
            _run("token_bucket", "coding", True, t, pr_url="https://git/pr/1", duration_ms=68_000),
            _run("lru", "coding", False, t, status="failed"),
            _run("token_bucket", "frontier", True, t, model_id="frontier"),
            _run("lru", "frontier", True, t, model_id="frontier"),
        ]
        text = render(runs, generated_at=t)
        assert "| `coding` | `m` | **1/2** (50%) | 2 | 2,000 | 200 | 128s | $0.0022 |" in text
        assert "| `frontier` | `frontier` | **2/2** (100%) | 2 |" in text and "not priced |" in text
        assert "| `token_bucket` | ✅ [68s](https://git/pr/1) | ✅ 60s |" in text
        assert "| `lru` | ❌ failed | ✅ 60s |" in text
    finally:
        get_settings.cache_clear()


def test_render_with_no_runs_says_how_to_produce_some():
    assert "No benchmark runs have been persisted yet" in render([])


def test_backend_spec_parsing():
    b = Backend.parse("frontier=http://localhost:8091/v1:frontier-model")
    assert (b.label, b.url, b.model_id) == ("frontier", "http://localhost:8091/v1", "frontier-model")
    c = Backend.parse("coding=http://vllm:8000/v1")
    assert (c.label, c.url, c.model_id) == ("coding", "http://vllm:8000/v1", None)


@pytest.mark.integration
@requires_integration_env
async def test_persist_and_reload_runs_and_the_adapter_fk():
    results = [
        {
            "task_id": "token_bucket",
            "agent_task_id": None,
            "status": "completed",
            "solved": True,
            "fail_to_pass": {"tests/test_x.py::test_a": True},
            "pass_to_pass": {"tests/test_x.py::test_b": True},
            "total_prompt_tokens": 12_345,
            "total_completion_tokens": 678,
            "duration_ms": 67_600,
            "pr_url": "https://gitea/x/pull/1",
        }
    ]
    label = f"test-{uuid.uuid4().hex[:6]}"
    ids = await persist_results(results, backend_label=label, model_id="demo-coder")
    tid = uuid.uuid4()
    try:
        runs = [r for r in await load_runs() if r.backend_label == label]
        assert len(runs) == 1 and runs[0].id == ids[0]
        assert runs[0].solved and runs[0].prompt_tokens == 12_345 and runs[0].details["pr_url"] == results[0]["pr_url"]

        # model_adapters.benchmark_run_id is now a real FK: SET NULL when the run is deleted.
        async with get_db_context() as db:
            db.add(Tenant(id=tid, name="bench-fk", email=f"{tid}@test.dev", tier=TenantTier.FREE))
            await db.flush()
            db.add(
                ModelAdapter(
                    tenant_id=tid,
                    base_model_id="base",
                    name="adapter",
                    path="/adapters/a",
                    rank=8,
                    job_type="lora",
                    benchmark_run_id=ids[0],
                )
            )
        async with get_db_context() as db:
            await db.execute(delete(BenchmarkRun).where(BenchmarkRun.id == ids[0]))
        async with get_db_context() as db:
            adapter = (await db.execute(select(ModelAdapter).where(ModelAdapter.tenant_id == tid))).scalar_one()
            assert adapter.benchmark_run_id is None
    finally:
        async with get_db_context() as db:
            await db.execute(delete(BenchmarkRun).where(BenchmarkRun.backend_label == label))
            row = await db.get(Tenant, tid)
            if row is not None:
                await db.delete(row)
