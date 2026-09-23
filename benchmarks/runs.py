# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Persisted benchmark runs — the record behind every published number.

`persist_results` writes one `BenchmarkRun` row per repo task the agent
runner produced (benchmarks/agent_runner.py `--persist`), tagged with the
backend label it ran against ("coding" for the configured open-weight
role, "frontier" for benchmarks/frontier_proxy.py, an adapter's name for a
fine-tuned model, ...). `load_runs` reads them back for
benchmarks/report.py, which regenerates docs/benchmarks/latest.md from
the database rather than from a hand-edited table — so the doc can never
say something the rows don't.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from src.db.connection import get_db_context
from src.db.models import BenchmarkRun

SUITE_REPO = "repo"


def _uuid_or_none(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value else None
    except ValueError:
        return None


async def persist_results(
    results: list[dict[str, Any]], *, backend_label: str, model_id: str | None, suite: str = SUITE_REPO
) -> list[uuid.UUID]:
    """One row per runner result. Returns the new run ids in the same order."""
    ids: list[uuid.UUID] = []
    async with get_db_context() as db:
        for result in results:
            run = BenchmarkRun(
                suite=suite,
                task_id=str(result.get("task_id", "")),
                backend_label=backend_label,
                model_id=model_id,
                agent_task_id=_uuid_or_none(result.get("agent_task_id")),
                status=str(result.get("status", "unknown")),
                solved=bool(result.get("solved", False)),
                fail_to_pass=dict(result.get("fail_to_pass") or {}),
                pass_to_pass=dict(result.get("pass_to_pass") or {}),
                prompt_tokens=int(result.get("total_prompt_tokens") or 0),
                completion_tokens=int(result.get("total_completion_tokens") or 0),
                duration_ms=int(result.get("duration_ms") or 0),
                pr_url=result.get("pr_url"),
                error_message=result.get("error_message") or result.get("error"),
                details=result,
            )
            db.add(run)
            await db.flush()
            ids.append(run.id)
    return ids


async def load_runs(*, suite: str = SUITE_REPO, days: int | None = None) -> list[BenchmarkRun]:
    """Every persisted run for `suite` (optionally only the last `days` days), oldest first."""
    stmt = select(BenchmarkRun).where(BenchmarkRun.suite == suite).order_by(BenchmarkRun.created_at)
    if days is not None:
        stmt = stmt.where(BenchmarkRun.created_at >= datetime.now(UTC) - timedelta(days=days))
    async with get_db_context() as db:
        return list((await db.execute(stmt)).scalars().all())
