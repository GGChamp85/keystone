# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/finetuning/sources/repo_pretrain.py — a
real Postgres (tests/conftest.py's requires_integration_env), real
CodebaseIndex rows written the same way src/memory/ingestion.py's
incremental ingestion writes them.
"""

from __future__ import annotations

import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import CodebaseIndex, Tenant, TenantTier
from src.finetuning.sources.repo_pretrain import list_pretraining_eligible_repositories

pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="repo-pretrain-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _index_file(tenant_id, repository_url, file_path="a.py") -> None:
    async with get_db_context() as db:
        db.add(
            CodebaseIndex(
                tenant_id=tenant_id,
                repository_url=repository_url,
                file_path=file_path,
                file_hash="deadbeef",
                chunk_count=1,
                language="python",
            )
        )
        await db.flush()


async def test_no_ingested_repos_returns_empty_list(tenant_id):
    assert await list_pretraining_eligible_repositories(tenant_id) == []


async def test_returns_distinct_repositories_sorted(tenant_id):
    await _index_file(tenant_id, "https://git/b/repo", "a.py")
    await _index_file(tenant_id, "https://git/a/repo", "a.py")
    await _index_file(tenant_id, "https://git/b/repo", "b.py")  # same repo, second file — must not duplicate

    repos = await list_pretraining_eligible_repositories(tenant_id)
    assert repos == ["https://git/a/repo", "https://git/b/repo"]


async def test_does_not_include_another_tenants_ingested_repositories(tenant_id):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tenant_id, name="other", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    try:
        await _index_file(other_tenant_id, "https://git/theirs/repo")
        assert await list_pretraining_eligible_repositories(tenant_id) == []
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)
