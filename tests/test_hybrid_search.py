# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Hybrid retrieval (src/memory/hybrid_search.py) against real Qdrant and real Postgres: an exact identifier
that embeddings blur is found by the full-text side, a topical query by the vector side, and RRF puts a
chunk found by both first. Also: memory recall (src/memory/store.py) ranks by Postgres full-text rank.
Requires DATABASE_URL/REDIS_URL and a reachable Qdrant.
"""

from __future__ import annotations

import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import CodeChunk, Tenant, TenantTier
from src.memory.hybrid_search import fulltext_search, hybrid_search, rrf_merge
from src.memory.ingestion import _chunk_id
from src.memory.vector_store import VectorStore

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


def test_rrf_puts_a_chunk_found_by_both_lists_first_and_keeps_source_ranks():
    vector = [
        {"repository": "r", "file_path": "a.py", "chunk_index": 0, "score": 0.9},
        {"repository": "r", "file_path": "b.py", "chunk_index": 0, "score": 0.8},
    ]
    fulltext = [
        {"repository": "r", "file_path": "c.py", "chunk_index": 0, "score": 0.5},
        {"repository": "r", "file_path": "b.py", "chunk_index": 0, "score": 0.4},
    ]
    merged = rrf_merge([vector, fulltext], limit=3)
    assert merged[0]["file_path"] == "b.py" and len(merged[0]["sources"]) == 2
    assert {m["file_path"] for m in merged} == {"a.py", "b.py", "c.py"}
    assert rrf_merge([], limit=3) == []


@pytest.fixture
async def tenant():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="hybrid", email=f"{tid}@test.dev", tier=TenantTier.FREE))
    vs = VectorStore()
    try:
        yield tid, vs
    finally:
        await vs.delete_by_tenant(str(tid))
        await vs.close()
        async with get_db_context() as db:
            row = await db.get(Tenant, tid)
            if row is not None:
                await db.delete(row)


async def _index(tid: uuid.UUID, vs: VectorStore, repo: str, files: dict[str, str]) -> None:
    chunks = [
        {
            "id": _chunk_id(repo, path, 0),
            "content": content,
            "file_path": path,
            "language": "python",
            "tenant_id": str(tid),
            "repository": repo,
            "chunk_index": 0,
            "file_hash": "h",
        }
        for path, content in files.items()
    ]
    await vs.upsert_chunks(str(tid), chunks)
    async with get_db_context() as db:
        for c in chunks:
            db.add(
                CodeChunk(
                    id=uuid.UUID(c["id"]),
                    tenant_id=tid,
                    repository_url=repo,
                    file_path=c["file_path"],
                    language="python",
                    chunk_index=0,
                    content=c["content"],
                )
            )


async def test_exact_identifiers_are_found_by_full_text_and_fused_with_vector_results(tenant):
    tid, vs = tenant
    repo = "https://gitea.internal.keystone.local/org/hybrid"
    await _index(
        tid,
        vs,
        repo,
        {
            "bucket.py": "class TokenBucket:\n    def _refill(self):\n"
            "        self.tokens = min(self.capacity, self.tokens + self.rate)\n",
            "http.py": "def parse_headers(raw):\n    return dict(line.split(': ', 1) for line in raw.splitlines())\n",
            "db.py": "async def connect_pool(url):\n    return await asyncpg.create_pool(url)\n",
        },
    )
    fts = await fulltext_search(tid, "_refill TokenBucket", repository=repo)
    assert fts and fts[0]["file_path"] == "bucket.py"

    merged = await hybrid_search(
        vs, tid, "the token bucket _refill method must cap at capacity", repository=repo, limit=3
    )
    assert merged and merged[0]["file_path"] == "bucket.py"
    assert any(len(m["sources"]) == 2 for m in merged)  # found by both the vector and the full-text side

    other_repo = await hybrid_search(vs, tid, "_refill", repository="https://example.invalid/x", limit=3)
    assert other_repo == []  # repository scoping applies to both sides


async def test_memory_recall_ranks_by_full_text_rank(tenant):
    from src.memory.store import create_memory, recall

    tid, _ = tenant
    repo = "https://gitea.internal.keystone.local/org/hybrid"
    await create_memory(
        tid, "Always run the migrations before the tests in this repository.", "convention", repository=repo
    )
    await create_memory(tid, "The CSV exporter quotes every field, including numbers.", "fact", repository=repo)
    await create_memory(tid, "Use ruff, not flake8, for linting.", "avoid", repository=repo)
    hits = await recall(tid, repo, "which migration step comes first when testing?")
    assert hits and "migrations" in hits[0].content  # stemming: migration ~ migrations, testing ~ tests
