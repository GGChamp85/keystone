# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for the large-repo-scaling fixes in src/memory/ingestion.py and
src/memory/vector_store.py:

1. `_plan_file_ingestion` (ingestion.py) — the core incremental decision
   (skip an unchanged file's re-embedding; find a shrunk file's now-stale
   chunk indices) as a pure function, so it's tested directly rather than
   only indirectly through a full repo clone.
2. Per-tenant Qdrant collection isolation (vector_store.py) — against a
   real Qdrant (tests/conftest.py's requires_integration_env), not a mock:
   two tenants' identical content lives in genuinely separate collections,
   proven by searching tenant B and finding zero results for content only
   tenant A ever upserted — the actual bug this fixes (a single shared
   collection filtered by a `tenant_id` field, where any call site that
   forgot to pass it would leak across tenants).

Full end-to-end `ingest_repository()` (real git clone -> real chunking ->
real CodebaseIndex read/write -> real Qdrant) is NOT exercised here for
the same reason tests/test_git_workflow_integration.py already documents
and skips without one: it needs a real git host reachable over https/ssh
(ingestion.py's own SSRF guard deliberately rejects file:// and bare
git://), and no Gitea instance exists in this dev environment. What *is*
real and tested here — the decision logic and the Qdrant isolation — are
the two pieces that were actually broken/missing before this change.
"""

from __future__ import annotations

import uuid

import pytest

from src.memory.ingestion import _chunk_id, _plan_file_ingestion
from src.memory.vector_store import VectorStore

pytestmark = pytest.mark.integration


# ── _plan_file_ingestion (pure, no infra needed) ────────────────────


def test_never_seen_file_always_needs_embedding_with_nothing_stale():
    needs_reembedding, stale = _plan_file_ingestion(None, 0, "hash-a", 3)
    assert needs_reembedding is True
    assert list(stale) == []


def test_unchanged_file_is_skipped():
    needs_reembedding, stale = _plan_file_ingestion("hash-a", 3, "hash-a", 3)
    assert needs_reembedding is False
    assert list(stale) == []


def test_changed_file_with_same_chunk_count_has_nothing_stale():
    needs_reembedding, stale = _plan_file_ingestion("hash-a", 3, "hash-b", 3)
    assert needs_reembedding is True
    assert list(stale) == []


def test_file_that_grew_has_nothing_stale():
    needs_reembedding, stale = _plan_file_ingestion("hash-a", 2, "hash-b", 5)
    assert needs_reembedding is True
    assert list(stale) == []


def test_file_that_shrank_reports_its_old_trailing_indices_as_stale():
    needs_reembedding, stale = _plan_file_ingestion("hash-a", 5, "hash-b", 2)
    assert needs_reembedding is True
    assert list(stale) == [2, 3, 4]


def test_chunk_id_is_deterministic_and_distinguishes_index_and_path():
    a = _chunk_id("https://git/x/y", "src/main.py", 0)
    b = _chunk_id("https://git/x/y", "src/main.py", 0)
    c = _chunk_id("https://git/x/y", "src/main.py", 1)
    d = _chunk_id("https://git/x/y", "src/other.py", 0)
    assert a == b
    assert len({a, c, d}) == 3


# ── Per-tenant Qdrant collection isolation (real Qdrant) ────────────


def _qdrant_env_ready() -> bool:
    import os

    # QDRANT_HOST isn't part of the base requires_integration_env set
    # (Postgres/Redis) — most of the suite doesn't touch Qdrant at all, so
    # gate these specifically on it being explicitly pointed at a real
    # instance, matching how test_git_workflow_integration.py gates on
    # GIT_HOST_TOKEN rather than reusing an unrelated env var as a proxy.
    return bool(os.environ.get("QDRANT_HOST"))


requires_qdrant = pytest.mark.skipif(
    not _qdrant_env_ready(), reason="Needs QDRANT_HOST (and QDRANT_PORT) pointed at a real Qdrant instance"
)


@pytest.fixture
async def vector_store():
    vs = VectorStore()
    yield vs
    await vs.close()


@pytest.fixture
def tenant_a():
    return str(uuid.uuid4())


@pytest.fixture
def tenant_b():
    return str(uuid.uuid4())


@requires_qdrant
async def test_tenants_get_genuinely_separate_collections(vector_store, tenant_a, tenant_b):
    await vector_store.ensure_collection(tenant_a)
    await vector_store.ensure_collection(tenant_b)
    assert vector_store._collection_name(tenant_a) != vector_store._collection_name(tenant_b)


@requires_qdrant
async def test_searching_tenant_b_never_finds_tenant_as_content(vector_store, tenant_a, tenant_b):
    unique_marker = f"unique-marker-{uuid.uuid4().hex}"
    await vector_store.upsert_chunks(
        tenant_a,
        [
            {
                "id": str(uuid.uuid4()),
                "content": f"def handle_payment(): return '{unique_marker}'",
                "file_path": "payments.py",
                "language": "python",
                "repository": "https://git/acme/payments",
                "chunk_index": 0,
                "file_hash": "h1",
            }
        ],
    )

    # Tenant A finds its own real content.
    found_by_a = await vector_store.search(tenant_a, unique_marker, limit=5, score_threshold=0.0)
    assert any(unique_marker in r["content"] for r in found_by_a)

    # Tenant B's collection was never written to — searching it for the
    # exact same query must come back empty, not filtered-but-present.
    found_by_b = await vector_store.search(tenant_b, unique_marker, limit=5, score_threshold=0.0)
    assert found_by_b == []


@requires_qdrant
async def test_delete_by_ids_removes_only_the_given_points_in_that_tenants_collection(vector_store, tenant_a):
    keep_id, remove_id = str(uuid.uuid4()), str(uuid.uuid4())
    await vector_store.upsert_chunks(
        tenant_a,
        [
            {
                "id": keep_id,
                "content": "def kept(): pass",
                "file_path": "a.py",
                "language": "python",
                "repository": "https://git/x/y",
                "chunk_index": 0,
                "file_hash": "h",
            },
            {
                "id": remove_id,
                "content": "def removed(): pass",
                "file_path": "a.py",
                "language": "python",
                "repository": "https://git/x/y",
                "chunk_index": 1,
                "file_hash": "h",
            },
        ],
    )

    await vector_store.delete_by_ids(tenant_a, [remove_id])

    results = await vector_store.search(tenant_a, "def", limit=10, score_threshold=0.0)
    contents = {r["content"] for r in results}
    assert "def kept(): pass" in contents
    assert "def removed(): pass" not in contents


@requires_qdrant
async def test_delete_by_tenant_drops_the_whole_collection(vector_store, tenant_a):
    await vector_store.upsert_chunks(
        tenant_a,
        [
            {
                "id": str(uuid.uuid4()),
                "content": "def x(): pass",
                "file_path": "a.py",
                "language": "python",
                "repository": "https://git/x/y",
                "chunk_index": 0,
                "file_hash": "h",
            }
        ],
    )
    await vector_store.delete_by_tenant(tenant_a)

    # A search after the collection is gone must transparently recreate it
    # (ensure_collection) and return empty, not raise.
    results = await vector_store.search(tenant_a, "def", limit=5, score_threshold=0.0)
    assert results == []
