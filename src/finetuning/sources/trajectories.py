# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — fine-tuning data source: real completed task history.

`src/finetuning/data_prep.py`'s `prepare_sft_data`/`prepare_dpo_data` have
existed since early in this project with zero real callers (grepped: the
only reference anywhere was a comment in src/rl/rollout.py, not an actual
call) — every fine-tuning run would have needed someone to hand-assemble
`instruction_pairs`/`preference_pairs` by hand. This is the first real
source: a tenant's own accepted/merged `AgentTask` history (src/db/models.py),
scored by the real human verdicts in `TaskFeedback` (src/api/routes/agents.py's
manual feedback route, and src/orchestrator/pr_polling.py's automatic
merged/rejected detection) — turning "the agent did this and a human kept
it" into a real instruction -> real diff training example.

DPO pair extraction from production task history is NOT implemented here —
unlike src/rl/rollout.py's rollouts (deliberately multiple attempts at the
same instruction, so pairing chosen/rejected within one trajectory is
well-defined), a normal AgentTask is one-shot: there is no second attempt
at the same instruction to prefer over the first. Building real DPO pairs
from production history needs a real notion of two comparable outputs for
the same input (e.g. an accepted task and a *later reverted* one on the
same repository/area) — a genuinely different, larger piece of work, not
stubbed out here as something it isn't.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from src.db.connection import get_db_context
from src.db.models import AgentTask, FeedbackVerdict, TaskFeedback, TaskStatus

_POSITIVE_VERDICTS = (FeedbackVerdict.ACCEPTED, FeedbackVerdict.MERGED)


async def build_sft_examples_from_trajectories(
    tenant_id: UUID | None = None,
    repository_url: str | None = None,
) -> list[dict]:
    """
    Real SFT instruction/output pairs from completed tasks a human accepted
    or merged. Shape matches `data_prep.prepare_sft_data`'s `internal_tasks`
    parameter: `{"task": str, "context": str, "solution": str}`.

    A task needs a real diff to teach anything (`output_diff` non-empty) and
    at least one ACCEPTED/MERGED `TaskFeedback` row — a task with no
    feedback at all, or only REJECTED/REVERTED verdicts, is excluded: this
    is a source of *positive* examples only (what to imitate), matching
    `prepare_sft_data`'s own shape. A rejected/reverted task is exactly the
    kind of signal src/memory/extract.py already turns into an `avoid`
    memory instead — a different mechanism for a different purpose, not
    something this also tries to do as negative SFT examples.
    """
    async with get_db_context() as db:
        stmt = (
            select(AgentTask)
            .join(TaskFeedback, TaskFeedback.task_id == AgentTask.id)
            .where(
                AgentTask.status == TaskStatus.COMPLETED,
                AgentTask.output_diff.isnot(None),
                TaskFeedback.verdict.in_(_POSITIVE_VERDICTS),
            )
            .order_by(AgentTask.created_at)
        )
        if tenant_id is not None:
            stmt = stmt.where(AgentTask.tenant_id == tenant_id)
        if repository_url is not None:
            stmt = stmt.where(AgentTask.repository_url == repository_url)
        tasks = (await db.execute(stmt)).unique().scalars().all()

    examples: list[dict] = []
    seen_task_ids: set[UUID] = set()
    for task in tasks:
        if task.id in seen_task_ids:  # a task can carry more than one positive feedback row
            continue
        seen_task_ids.add(task.id)
        if not task.output_diff or not task.output_diff.strip():
            continue
        examples.append(
            {
                "task": task.task_description,
                "context": f"Repository: {task.repository_url}" if task.repository_url else "",
                "solution": task.output_diff,
                # Carried through for split_by_repository() below, not part
                # of prepare_sft_data's own {"task","context","solution"}
                # contract — callers that hand this straight to
                # prepare_sft_data should drop it first.
                "repository_url": task.repository_url,
            }
        )
    return examples
