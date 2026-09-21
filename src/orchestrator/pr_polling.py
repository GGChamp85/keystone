# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — automatic PR-status polling.

`TaskFeedback` (src/db/models.py) existed since Phase 3 as the raw signal
src/memory/extract.py turns into proposed memories, but nothing in this
codebase ever wrote a row to it — no route, no poller (grepped: zero call
sites before this file). This is the automatic half the plan calls for:
a completed task with an open PR gets its real merge/close state checked
against the git host (src/git/host.py's `GitHost.get_pull_request_status`,
already implemented for Gitea) and a real feedback row recorded, without a
human having to remember to do it by hand. `POST
/v1/keystone/tasks/{id}/feedback` (src/api/routes/agents.py) is the manual
complement for a human's own verdict/reason.

Deliberately scoped to what a single PR-status check can actually tell
you: `merged` -> `FeedbackVerdict.MERGED`, closed-without-merging ->
`FeedbackVerdict.REJECTED` (the human explicitly didn't want this
direction). `FeedbackVerdict.REVERTED` is NOT produced here — detecting a
later revert of an already-merged PR needs watching the base branch's own
history after the fact, a materially different (and open) piece of work,
not something `get_pull_request_status`'s open/closed/merged view can
tell you; claiming to detect it from this alone would be exactly the kind
of stub this project doesn't ship.

Not Temporal-durable — a lightweight periodic asyncio loop
(`run_pr_polling_loop`, started from src/main.py's lifespan) rather than a
Temporal Schedule, matching this codebase's own existing "non-durable
asyncio fallback" honesty pattern (src/orchestrator/engine.py) for where a
full durable-workflow version doesn't exist yet. A missed poll (process
restart) just gets picked up on the next interval — nothing here is lost,
since it only ever reads current PR state, never a delta.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import select

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AgentTask, FeedbackVerdict, TaskFeedback, TaskStatus
from src.git.host import get_git_host, parse_owner_repo

logger = structlog.get_logger(__name__)


async def _already_recorded(task_id: UUID) -> bool:
    async with get_db_context() as db:
        result = await db.execute(
            select(TaskFeedback.id).where(
                TaskFeedback.task_id == task_id,
                TaskFeedback.verdict.in_([FeedbackVerdict.MERGED, FeedbackVerdict.REJECTED]),
            )
        )
        return result.scalar_one_or_none() is not None


async def poll_and_record_pr_feedback() -> list[dict[str, Any]]:
    """One poll pass over every COMPLETED task with an open PR that hasn't
    already had its merge/close state recorded. Returns the feedback rows
    written this pass (empty if nothing changed) — never raises: a single
    task's git-host lookup failing (unreachable host, deleted PR, ...) is
    logged and skipped, not allowed to abort the whole pass."""
    async with get_db_context() as db:
        result = await db.execute(
            select(AgentTask).where(
                AgentTask.status == TaskStatus.COMPLETED,
                AgentTask.pr_number.isnot(None),
                AgentTask.repository_url.isnot(None),
            )
        )
        tasks = result.scalars().all()

    written: list[dict[str, Any]] = []
    for task in tasks:
        if await _already_recorded(task.id):
            continue

        try:
            owner, repo = parse_owner_repo(task.repository_url)
            host = get_git_host()
            status = await host.get_pull_request_status(owner, repo, task.pr_number)
        except Exception as exc:
            logger.warning("keystone.pr_poll_check_failed", task_id=str(task.id), error=str(exc))
            continue

        if status.merged:
            verdict, reason = (
                FeedbackVerdict.MERGED,
                f"Auto-detected: PR #{task.pr_number} merged (commit {status.merge_commit_sha})",
            )
        elif status.state == "closed":
            verdict, reason = FeedbackVerdict.REJECTED, f"Auto-detected: PR #{task.pr_number} closed without merging"
        else:
            continue  # still open — nothing to record yet

        async with get_db_context() as db:
            feedback = TaskFeedback(task_id=task.id, user_id=None, verdict=verdict, reason=reason)
            db.add(feedback)
            await db.flush()
            feedback_id = feedback.id

        logger.info("keystone.pr_poll_feedback_recorded", task_id=str(task.id), verdict=verdict.value)
        written.append({"task_id": task.id, "feedback_id": feedback_id, "verdict": verdict.value})

    return written


async def run_pr_polling_loop(stop_event: asyncio.Event) -> None:
    """Runs `poll_and_record_pr_feedback` every `settings.pr_poll_interval_seconds`
    until `stop_event` is set (src/main.py's lifespan sets it on shutdown).
    A failed pass is logged and the loop keeps running — one bad poll
    should never silently kill all future ones."""
    settings = get_settings()
    interval = settings.pr_poll_interval_seconds
    while not stop_event.is_set():
        try:
            await poll_and_record_pr_feedback()
        except Exception as exc:
            logger.error("keystone.pr_poll_loop_failed", error=str(exc))
        with contextlib.suppress(TimeoutError):  # normal — just means it's time for the next pass
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
