# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — `GET /v1/keystone/benchmarks`: the persisted benchmark results (`benchmark_runs`, written by
`benchmarks/agent_runner.py --persist` and `benchmarks/compare.py`) as a comparison across backends —
solve rate, time and tokens per backend, and a task-by-backend matrix of the latest run of each task.
The same rows `benchmarks/report.py` renders to docs/benchmarks/latest.md; nothing here is computed
from anything but those rows.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from src.api.middleware.auth import require_scope
from src.db.connection import get_db_context
from src.db.models import BenchmarkRun

router = APIRouter(prefix="/v1/keystone", tags=["keystone"])

_TASKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "tasks" / "repo"


def _task_languages() -> dict[str, str]:
    """task id -> language from the task definitions, when the checkout ships them (not required)."""
    out: dict[str, str] = {}
    if not _TASKS_DIR.is_dir():
        return out
    for task_json in _TASKS_DIR.glob("*/task.json"):
        try:
            data = json.loads(task_json.read_text())
        except (OSError, ValueError):
            continue
        out[str(data.get("id", task_json.parent.name))] = str(data.get("language", "python"))
    return out


def summarize(runs: list[BenchmarkRun], languages: dict[str, str] | None = None) -> dict[str, Any]:
    """Per-backend totals over the latest run of each task, plus the task-by-backend matrix. `runs` oldest
    first, so a later run of the same task replaces an earlier one."""
    languages = languages or {}
    latest: dict[str, dict[str, BenchmarkRun]] = defaultdict(dict)
    for row in runs:
        latest[row.backend_label][row.task_id] = row

    backends: list[dict[str, Any]] = []
    for label, by_task in sorted(latest.items()):
        rows = list(by_task.values())
        solved = sum(1 for r in rows if r.solved)
        latest_at = max((r.created_at for r in rows if r.created_at), default=None)
        backends.append(
            {
                "label": label,
                "model_id": next((r.model_id for r in reversed(rows) if r.model_id), None),
                "tasks_run": len(rows),
                "solved": solved,
                "solve_rate": round(solved / len(rows), 4) if rows else 0.0,
                "avg_duration_ms": int(sum(r.duration_ms for r in rows) / len(rows)) if rows else 0,
                "total_prompt_tokens": sum(r.prompt_tokens for r in rows),
                "total_completion_tokens": sum(r.completion_tokens for r in rows),
                "latest_at": latest_at.isoformat() if latest_at else None,
            }
        )

    task_ids = sorted({t for by_task in latest.values() for t in by_task})
    tasks = []
    for task_id in task_ids:
        results: dict[str, Any] = {}
        for label, by_task in latest.items():
            run: BenchmarkRun | None = by_task.get(task_id)
            if run is None:
                continue
            results[label] = {
                "run_id": str(run.id),
                "solved": run.solved,
                "status": run.status,
                "duration_ms": run.duration_ms,
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "pr_url": run.pr_url,
                "error_message": run.error_message,
                "fail_to_pass": run.fail_to_pass,
                "pass_to_pass": run.pass_to_pass,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
        tasks.append({"task_id": task_id, "language": languages.get(task_id, "python"), "results": results})

    return {
        "generated_at": int(time.time()),
        "runs_considered": len(runs),
        "backends": backends,
        "tasks": tasks,
    }


@router.get("/benchmarks")
async def benchmarks(
    days: int | None = Query(default=None, ge=1, description="Only runs from the last N days (default: all)"),
    suite: str = Query(default="repo"),
    auth: tuple = Depends(require_scope("inference")),
):
    """The persisted benchmark comparison: which backend solved which repository task, how fast, at what
    token cost — the latest run per task per backend. Empty until `benchmarks/agent_runner.py --persist`
    or `benchmarks/compare.py` has written rows."""
    stmt = select(BenchmarkRun).where(BenchmarkRun.suite == suite).order_by(BenchmarkRun.created_at)
    if days is not None:
        stmt = stmt.where(BenchmarkRun.created_at >= datetime.now(UTC) - timedelta(days=days))
    async with get_db_context() as db:
        runs = list((await db.execute(stmt)).scalars().all())
    return summarize(runs, _task_languages())
