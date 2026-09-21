# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — fine-tuning data source: repositories eligible for
continued pretraining.

`src/finetuning/data_prep.py`'s `prepare_continued_pretraining_data` reads
real files off local disk — it has no notion of *which* repositories a
tenant actually cares about, and no durable local checkout of any repo's
content survives past one `CodeIngestionPipeline.ingest_repository()` call
(the clone happens into a temp dir that's removed when that call returns —
src/memory/ingestion.py's `finally: shutil.rmtree(...)`). This module is
the "which repos" decision, driven by real data: `CodebaseIndex`
(src/db/models.py) is the durable record of every repository a tenant has
actually ingested for RAG before, via `distinct(repository_url)`.

It deliberately does NOT re-clone those repositories into a fresh
`repo_paths` list ready for `prepare_continued_pretraining_data` — that
needs the same real, reachable https/ssh git host `ingest_repository()`
itself needs (this dev environment doesn't have one — see
tests/test_git_workflow_integration.py's own documented gap), and a
caller with real infrastructure can do that clone itself (reusing
`src.memory.ingestion._validate_repo_url` and `git.Repo.clone_from`,
exactly as `ingest_repository()` already does) once it has this real list
of which repositories to bother cloning at all.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from src.db.connection import get_db_context
from src.db.models import CodebaseIndex


async def list_pretraining_eligible_repositories(tenant_id: UUID) -> list[str]:
    """Every distinct repository URL this tenant has real ingested history
    for (src/memory/ingestion.py's incremental ingestion, via CodebaseIndex),
    sorted for a deterministic, reviewable order — the real candidate list
    for continued-pretraining data, not a config value someone has to type
    in by hand."""
    async with get_db_context() as db:
        result = await db.execute(
            select(CodebaseIndex.repository_url).where(CodebaseIndex.tenant_id == tenant_id).distinct()
        )
        return sorted(result.scalars().all())
