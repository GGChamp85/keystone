# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — persistent memory store.

Layered repo-over-tenant: a repo-scope memory always outranks a tenant-
scope one on the same topic, since it's more specific. `recall()` is what
the agent loop actually injects into its prompts (nodes/coding.py,
nodes/planning.py, via a dedicated slice of the token budget — see
src/orchestrator/context.py).

Ranking today is real and immediately useful — pinned first (a pinned
memory is a standing instruction, always included regardless of the
query), then repo-scope before tenant-scope, then keyword overlap with
the query, then recency — trimmed to a real token budget via
`context.count_tokens`, not a length heuristic. This is honestly a
simpler ranking than semantic (embedding) search, not a placeholder
pretending to be one; a dedicated Qdrant collection for memories is a
real future upgrade once there's enough memory volume per tenant for
semantic ranking to matter more than recency/pins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

import structlog
from sqlalchemy import func, select, update

from src.db.connection import get_db_context
from src.db.models import AgentMemory, MemoryKind, MemoryScope, MemorySource, MemoryStatus
from src.memory.hybrid_search import or_query
from src.orchestrator.context import count_tokens

logger = structlog.get_logger(__name__)

DEFAULT_RECALL_BUDGET_TOKENS = 2000
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class MemoryRecord:
    """A plain-data snapshot of an AgentMemory row — callers outside the DB session boundary (nodes/, API
    responses) work with this instead of a live ORM instance."""

    id: UUID
    tenant_id: UUID
    repository: str | None
    scope: str
    kind: str
    content: str
    source: str
    status: str
    pinned: bool
    confidence: float
    hit_count: int
    created_by: str | None

    @classmethod
    def from_orm(cls, row: AgentMemory) -> MemoryRecord:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            repository=row.repository,
            scope=row.scope.value if hasattr(row.scope, "value") else row.scope,
            kind=row.kind.value if hasattr(row.kind, "value") else row.kind,
            content=row.content,
            source=row.source.value if hasattr(row.source, "value") else row.source,
            status=row.status.value if hasattr(row.status, "value") else row.status,
            pinned=row.pinned,
            confidence=row.confidence,
            hit_count=row.hit_count,
            created_by=row.created_by,
        )


async def create_memory(
    tenant_id: UUID,
    content: str,
    kind: str,
    *,
    repository: str | None = None,
    source: str = MemorySource.USER.value,
    status: str = MemoryStatus.APPROVED.value,
    pinned: bool = False,
    confidence: float = 1.0,
    created_by: str | None = None,
) -> MemoryRecord:
    scope = MemoryScope.REPO if repository else MemoryScope.TENANT
    async with get_db_context() as db:
        row = AgentMemory(
            tenant_id=tenant_id,
            repository=repository,
            scope=scope,
            kind=MemoryKind(kind),
            content=content,
            source=MemorySource(source),
            status=MemoryStatus(status),
            pinned=pinned,
            confidence=confidence,
            created_by=created_by,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        record = MemoryRecord.from_orm(row)
    logger.info(
        "memory.created", tenant_id=str(tenant_id), repository=repository, kind=kind, source=source, status=status
    )
    return record


async def list_memories(
    tenant_id: UUID,
    *,
    repository: str | None = None,
    scope: str | None = None,
    status: str | None = None,
) -> list[MemoryRecord]:
    async with get_db_context() as db:
        stmt = select(AgentMemory).where(AgentMemory.tenant_id == tenant_id)
        if repository is not None:
            stmt = stmt.where(AgentMemory.repository == repository)
        if scope is not None:
            stmt = stmt.where(AgentMemory.scope == MemoryScope(scope))
        if status is not None:
            stmt = stmt.where(AgentMemory.status == MemoryStatus(status))
        stmt = stmt.order_by(AgentMemory.created_at.desc())
        result = await db.execute(stmt)
        return [MemoryRecord.from_orm(row) for row in result.scalars().all()]


async def get_memory(memory_id: UUID) -> MemoryRecord | None:
    async with get_db_context() as db:
        row = await db.get(AgentMemory, memory_id)
        return MemoryRecord.from_orm(row) if row else None


async def set_status(memory_id: UUID, status: str) -> MemoryRecord | None:
    """Used for both approve (proposed -> approved) and forget (any -> forgotten) — a soft delete that
    keeps the row (and its attribution/history) rather than actually removing it."""
    async with get_db_context() as db:
        row = await db.get(AgentMemory, memory_id)
        if row is None:
            return None
        row.status = MemoryStatus(status)
        await db.flush()
        await db.refresh(row)
        return MemoryRecord.from_orm(row)


async def set_pinned(memory_id: UUID, pinned: bool) -> MemoryRecord | None:
    async with get_db_context() as db:
        row = await db.get(AgentMemory, memory_id)
        if row is None:
            return None
        row.pinned = pinned
        await db.flush()
        await db.refresh(row)
        return MemoryRecord.from_orm(row)


async def _record_hits(memory_ids: list[UUID]) -> None:
    if not memory_ids:
        return
    async with get_db_context() as db:
        await db.execute(
            update(AgentMemory).where(AgentMemory.id.in_(memory_ids)).values(hit_count=AgentMemory.hit_count + 1)
        )


def _keyword_overlap(content: str, query_words: set[str]) -> int:
    if not query_words:
        return 0
    content_words = {w.lower() for w in _WORD_RE.findall(content)}
    return len(content_words & query_words)


async def recall(
    tenant_id: UUID,
    repository: str | None,
    query: str,
    *,
    budget_tokens: int = DEFAULT_RECALL_BUDGET_TOKENS,
) -> list[MemoryRecord]:
    """
    Everything approved for this tenant (tenant-scope) plus this specific repo (repo-scope), ranked
    pinned-first / repo-over-tenant / keyword-overlap / recency, trimmed to `budget_tokens`. Increments
    hit_count for every memory actually returned (not every memory considered).
    """
    async with get_db_context() as db:
        stmt = select(AgentMemory).where(
            AgentMemory.tenant_id == tenant_id,
            AgentMemory.status == MemoryStatus.APPROVED,
        )
        if repository is not None:
            stmt = stmt.where(
                (AgentMemory.scope == MemoryScope.TENANT) | (AgentMemory.repository == repository),
            )
        else:
            stmt = stmt.where(AgentMemory.scope == MemoryScope.TENANT)
        # Postgres full-text rank of each memory against the query (stemming, stop words, phrase-aware),
        # computed in the query itself; a memory that matches nothing ranks 0 and falls back to recency.
        tsq = func.websearch_to_tsquery("english", or_query(query)) if or_query(query) else None
        rank_expr = func.ts_rank_cd(func.to_tsvector("english", AgentMemory.content), tsq) if tsq is not None else None
        stmt = stmt.order_by(AgentMemory.created_at.desc())
        result = await db.execute(stmt)
        candidates = list(result.scalars().all())
        fts_rank: dict[UUID, float] = {}
        if rank_expr is not None and candidates:
            rank_rows = await db.execute(
                select(AgentMemory.id, rank_expr).where(AgentMemory.id.in_([m.id for m in candidates]))
            )
            fts_rank = {row[0]: float(row[1] or 0.0) for row in rank_rows.all()}

    query_words = {w.lower() for w in _WORD_RE.findall(query)}
    ranked = sorted(
        candidates,
        key=lambda m: (
            not m.pinned,  # pinned (False) sorts before unpinned (True)
            m.scope != MemoryScope.REPO,  # repo-scope sorts before tenant-scope
            -fts_rank.get(m.id, 0.0),  # full-text rank first
            -_keyword_overlap(m.content, query_words),  # then plain overlap as the tie-breaker
            -(m.created_at.timestamp() if m.created_at else 0),
        ),
    )

    selected: list[AgentMemory] = []
    used_tokens = 0
    for memory in ranked:
        cost = count_tokens(memory.content)
        if selected and used_tokens + cost > budget_tokens:
            continue
        selected.append(memory)
        used_tokens += cost

    await _record_hits([m.id for m in selected])
    return [MemoryRecord.from_orm(m) for m in selected]


def render_memories_for_prompt(memories: list[MemoryRecord]) -> str:
    """Formats recalled memories for injection into a coding/planning/review prompt."""
    if not memories:
        return ""
    lines = ["## Team Memory (learned conventions, preferences, and things to avoid)"]
    for m in memories:
        scope_label = "repo" if m.scope == MemoryScope.REPO.value else "team"
        lines.append(f"- [{m.kind}/{scope_label}] {m.content}")
    return "\n".join(lines)
