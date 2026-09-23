# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — hybrid code retrieval: vector similarity + full-text rank, fused.

An embedding finds "what this is about"; a full-text index finds the exact identifier the task names
(`_refill`, `TokenBucket`, `X-VS-Route-Decision`) that an embedding smooths over. Each source returns
its own ranked list and Reciprocal Rank Fusion merges them: score = Σ 1/(k + rank), so a chunk near the
top of either list ranks high and a chunk in both ranks highest. No tuned weights, no score scales to
reconcile.

The full-text side is Postgres (`code_chunks.tsv`, a generated tsvector with a GIN index; migration
b3c7d9e1f2a4), written by the ingestion pipeline next to every Qdrant point with the same id.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select

from src.db.connection import get_db_context
from src.db.models import CodeChunk
from src.memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)

RRF_K = 60  # the standard constant; rank 1 in one list ≈ 1/61, in both ≈ 2/61
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def or_query(text: str) -> str:
    """The query words joined with OR for websearch_to_tsquery: ranking, not filtering — a chunk that
    matches three of five words should rank, not vanish because the other two are absent."""
    words = []
    seen: set[str] = set()
    for w in _WORD_RE.findall(text):
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            words.append(w)
    return " OR ".join(words)


async def fulltext_search(
    tenant_id: UUID, query: str, *, repository: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """Postgres full-text matches for `query` (websearch syntax: quotes, OR, -), ranked by ts_rank_cd."""
    tsq = func.websearch_to_tsquery("english", or_query(query))
    rank = func.ts_rank_cd(CodeChunk.tsv, tsq).label("rank")
    stmt = (
        select(CodeChunk, rank)
        .where(CodeChunk.tenant_id == tenant_id, CodeChunk.tsv.op("@@")(tsq))
        .order_by(rank.desc())
        .limit(limit)
    )
    if repository:
        stmt = stmt.where(CodeChunk.repository_url == repository)
    async with get_db_context() as db:
        rows = (await db.execute(stmt)).all()
    return [
        {
            "id": str(chunk.id),
            "content": chunk.content,
            "file_path": chunk.file_path,
            "language": chunk.language or "",
            "repository": chunk.repository_url,
            "chunk_index": chunk.chunk_index,
            "score": float(r),
        }
        for chunk, r in rows
    ]


def rrf_merge(lists: list[list[dict[str, Any]]], *, k: int = RRF_K, limit: int = 10) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over ranked lists keyed by (repository, file_path, chunk_index)."""
    fused: dict[tuple[str, str, int], dict[str, Any]] = {}
    for ranked in lists:
        for position, item in enumerate(ranked, start=1):
            key = (item.get("repository", ""), item.get("file_path", ""), int(item.get("chunk_index", 0)))
            entry = fused.setdefault(key, {**item, "rrf": 0.0, "sources": []})
            entry["rrf"] += 1.0 / (k + position)
            entry["sources"].append({"score": item.get("score"), "rank": position})
    merged = sorted(fused.values(), key=lambda e: -e["rrf"])
    return merged[:limit]


async def hybrid_search(
    vs: VectorStore,
    tenant_id: UUID,
    query: str,
    *,
    repository: str | None = None,
    limit: int = 5,
    candidates: int = 10,
) -> list[dict[str, Any]]:
    """Top `limit` chunks by RRF over the vector search and the full-text search (each asked for
    `candidates`). A source that fails is logged and skipped — retrieval degrades to the other, never to
    nothing because one store hiccupped."""
    lists: list[list[dict[str, Any]]] = []
    try:
        # a low threshold: the vector side is a ranked candidate list for fusion, not a filter
        lists.append(
            await vs.search(str(tenant_id), query, repository=repository, limit=candidates, score_threshold=0.05)
        )
    except Exception as exc:
        logger.warning("hybrid_search.vector_failed", error=str(exc))
    try:
        lists.append(await fulltext_search(tenant_id, query, repository=repository, limit=candidates))
    except Exception as exc:
        logger.warning("hybrid_search.fulltext_failed", error=str(exc))
    return rrf_merge(lists, limit=limit)
