# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — the skills library.

A skill is an admin-curated, reusable standing instruction — "when touching payment code, run
the PCI checklist" — applied to every matching task, not learned per-task the way
`src/memory/store.py`'s `AgentMemory` is (an agent proposes a memory from what happened on ONE
task; a human writes a skill once, deliberately, for every future task it should apply to).
Tenant-scoped, not repo-scoped, since a skill is meant to travel with every repo the tenant
touches.

`matching_skills` is the one real function nodes/coding.py's prompt building calls — not an
approximation of it — so what the agent actually sees is provably the same as what
`POST /v1/keystone/skills/match` (a preview route, same shape as memory's `/recall`) reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import structlog
from sqlalchemy import select

from src.db.connection import get_db_context
from src.db.models import Skill

logger = structlog.get_logger(__name__)


@dataclass
class SkillRecord:
    """A plain-data snapshot of a Skill row — callers outside the DB session boundary (nodes/, API
    responses) work with this instead of a live ORM instance."""

    id: UUID
    tenant_id: UUID
    name: str
    content: str
    trigger_keywords: list[str]
    enabled: bool
    created_by: str | None

    @classmethod
    def from_orm(cls, row: Skill) -> SkillRecord:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            name=row.name,
            content=row.content,
            trigger_keywords=list(row.trigger_keywords or []),
            enabled=row.enabled,
            created_by=row.created_by,
        )


async def create_skill(
    tenant_id: UUID,
    name: str,
    content: str,
    *,
    trigger_keywords: list[str] | None = None,
    enabled: bool = True,
    created_by: str | None = None,
) -> SkillRecord:
    async with get_db_context() as db:
        row = Skill(
            tenant_id=tenant_id,
            name=name,
            content=content,
            trigger_keywords=trigger_keywords or [],
            enabled=enabled,
            created_by=created_by,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        record = SkillRecord.from_orm(row)
    logger.info("skills.created", tenant_id=str(tenant_id), name=name, enabled=enabled)
    return record


async def list_skills(tenant_id: UUID, *, enabled: bool | None = None) -> list[SkillRecord]:
    async with get_db_context() as db:
        stmt = select(Skill).where(Skill.tenant_id == tenant_id)
        if enabled is not None:
            stmt = stmt.where(Skill.enabled == enabled)
        stmt = stmt.order_by(Skill.created_at.desc())
        result = await db.execute(stmt)
        return [SkillRecord.from_orm(row) for row in result.scalars().all()]


async def get_skill(skill_id: UUID) -> SkillRecord | None:
    async with get_db_context() as db:
        row = await db.get(Skill, skill_id)
        return SkillRecord.from_orm(row) if row else None


async def set_enabled(skill_id: UUID, enabled: bool) -> SkillRecord | None:
    async with get_db_context() as db:
        row = await db.get(Skill, skill_id)
        if row is None:
            return None
        row.enabled = enabled
        await db.flush()
        await db.refresh(row)
        return SkillRecord.from_orm(row)


async def delete_skill(skill_id: UUID) -> bool:
    async with get_db_context() as db:
        row = await db.get(Skill, skill_id)
        if row is None:
            return False
        await db.delete(row)
        return True


async def matching_skills(tenant_id: UUID, task_description: str) -> list[SkillRecord]:
    """Every enabled skill whose `trigger_keywords` is empty (applies to every task) or has at
    least one keyword appearing case-insensitively in `task_description`. Real Postgres query
    plus a real (if simple) substring match — no heuristic scoring to get wrong, since a skill
    either applies or it doesn't."""
    description_lower = task_description.lower()
    enabled = await list_skills(tenant_id, enabled=True)
    return [
        skill
        for skill in enabled
        if not skill.trigger_keywords or any(kw.lower() in description_lower for kw in skill.trigger_keywords)
    ]


def render_skills_for_prompt(skills: list[SkillRecord]) -> str:
    if not skills:
        return ""
    lines = [f"- {s.name}: {s.content}" for s in skills]
    return "## Skills (standing instructions for this kind of task)\n" + "\n".join(lines)
