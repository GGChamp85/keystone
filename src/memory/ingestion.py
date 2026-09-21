# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Codebase ingestion pipeline.
Clones repos, chunks code files, and upserts into Qdrant.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import shutil
import socket
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import structlog
from sqlalchemy import select

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import CodebaseIndex
from src.memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)


def _chunk_id(repository_url: str, file_path: str, chunk_index: int) -> str:
    """Deterministic per-(repo, file, chunk-index) id — same input always
    produces the same id, so re-ingesting an unchanged file is a no-op
    upsert rather than a duplicate, and a shrunk file's now-gone higher
    indices can be reconstructed for deletion without a Qdrant round trip."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repository_url}:{file_path}:{chunk_index}"))


def _plan_file_ingestion(
    existing_hash: str | None,
    existing_chunk_count: int,
    new_hash: str,
    new_chunk_count: int,
) -> tuple[bool, range]:
    """The core incremental-ingestion decision, pulled out as a pure function
    so it's directly testable without a real git clone: does this file need
    re-chunking/re-embedding at all (no, if its content hash hasn't
    changed — the skip that makes a repeat ingest of a large, mostly-
    unchanged repo cheap), and if it does, which of its *old* chunk indices
    no longer exist in the new version and must be explicitly deleted (an
    upsert only ever touches the ids it's given — a file that shrank from
    5 chunks to 3 leaves indices 3 and 4 behind forever unless something
    deletes them).

    `existing_hash`/`existing_chunk_count` are `None`/`0` for a file with no
    prior CodebaseIndex row (never ingested before — always needs
    embedding, nothing stale to delete)."""
    if existing_hash is not None and existing_hash == new_hash:
        return False, range(0)
    stale_range = range(new_chunk_count, existing_chunk_count) if existing_chunk_count > new_chunk_count else range(0)
    return True, stale_range


class InvalidRepositoryURLError(ValueError):
    """Raised when a repository_url fails host/scheme allowlisting."""


def _validate_repo_url(url: str, allowed_hosts: list[str]) -> None:
    """
    Guard against SSRF via `git clone`: a tenant-supplied repository_url must
    use https/ssh (never file:// or bare git://), resolve to an allowlisted
    internal Git host (Gitea in air-gapped deployments), and never resolve to
    a loopback/link-local/private address — a coding agent given a
    repository_url is otherwise a generic "make this server fetch an
    arbitrary URL" primitive.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("https", "ssh"):
        raise InvalidRepositoryURLError(f"repository_url must use https:// or ssh://, got scheme '{parsed.scheme}'")

    host = parsed.hostname
    if not host:
        raise InvalidRepositoryURLError("repository_url has no resolvable host")

    if allowed_hosts:
        if host not in allowed_hosts:
            raise InvalidRepositoryURLError(
                f"repository_url host '{host}' is not in the allowlisted Git hosts {allowed_hosts}"
            )
        # An explicitly operator-allowlisted host (e.g. the internal Gitea
        # instance) is *expected* to resolve to a private/internal address in
        # an air-gapped deployment — that's not an SSRF risk, it's the point.
        # Skip the IP-range check below for allowlisted hosts.
        return

    # No allowlist configured (open/dev mode): fall back to blocking the
    # classic SSRF targets — cloud metadata endpoints, loopback, link-local,
    # and other private ranges a tenant-supplied URL should never reach.
    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidRepositoryURLError(f"repository_url host '{host}' does not resolve: {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved or ip.is_multicast:
            raise InvalidRepositoryURLError(
                f"repository_url host '{host}' resolves to a disallowed address ({ip}) — "
                f"internal/loopback/link-local targets are blocked (SSRF protection)"
            )


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".vue": "vue",
    ".jsx": "javascript",
    ".sh": "bash",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
}


class CodeIngestionPipeline:
    def __init__(self):
        self._vector_store: VectorStore | None = None

    async def _get_vs(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    async def ingest_repository(
        self,
        repository_url: str,
        tenant_id: str,
        branch: str = "main",
        file_extensions: list[str] | None = None,
        max_file_size_kb: int = 500,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> dict:
        """
        Incremental: a file whose content hash matches what's already
        recorded in CodebaseIndex is skipped entirely (no re-chunk, no
        re-embed, no upsert) — the first ingest of a large repo pays the
        full cost once, every ingest after that only touches what actually
        changed. A file removed from the repo (or shrunk to fewer chunks
        than before) has its now-stale Qdrant points deleted, not left
        behind forever — the gap this pipeline had before: `file_hash` was
        computed and stored in every chunk's payload, but nothing ever read
        CodebaseIndex back to compare against it.
        """
        import git

        settings = get_settings()
        _validate_repo_url(repository_url, settings.git_allowed_hosts)

        if file_extensions is None:
            file_extensions = list(LANGUAGE_MAP.keys())

        tenant_uuid = uuid.UUID(tenant_id)
        async with get_db_context() as db:
            existing_rows = (
                (
                    await db.execute(
                        select(CodebaseIndex).where(
                            CodebaseIndex.tenant_id == tenant_uuid,
                            CodebaseIndex.repository_url == repository_url,
                        )
                    )
                )
                .scalars()
                .all()
            )
        existing_by_path = {row.file_path: row for row in existing_rows}

        tmp_dir = tempfile.mkdtemp(prefix="vs_ingest_")
        try:
            logger.info("ingestion.cloning", repo=repository_url, branch=branch)
            repo = git.Repo.clone_from(repository_url, tmp_dir, branch=branch, depth=1)
            commit_sha = repo.head.commit.hexsha

            chunks: list[dict] = []
            stale_ids: list[str] = []
            # (path, hash, chunk_count, language, existing CodebaseIndex id or None)
            index_upserts: list[tuple[str, str, int, str, uuid.UUID | None]] = []
            seen_paths: set[str] = set()
            files_processed = 0
            files_unchanged = 0
            files_skipped = 0

            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}
            for root, dirs, files in os.walk(tmp_dir):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                for fname in files:
                    fpath = Path(root) / fname
                    ext = fpath.suffix.lower()
                    if ext not in file_extensions:
                        continue
                    if fpath.stat().st_size > max_file_size_kb * 1024:
                        files_skipped += 1
                        continue
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        files_skipped += 1
                        continue

                    rel_path = str(fpath.relative_to(tmp_dir))
                    seen_paths.add(rel_path)
                    language = LANGUAGE_MAP.get(ext, "text")
                    file_hash = hashlib.sha256(content.encode()).hexdigest()

                    existing = existing_by_path.get(rel_path)
                    file_chunks = self._chunk_code(content, chunk_size, chunk_overlap)
                    needs_reembedding, stale_range = _plan_file_ingestion(
                        existing.file_hash if existing else None,
                        existing.chunk_count if existing else 0,
                        file_hash,
                        len(file_chunks),
                    )
                    if not needs_reembedding:
                        files_unchanged += 1
                        continue

                    for idx, chunk_text in enumerate(file_chunks):
                        chunks.append(
                            {
                                "id": _chunk_id(repository_url, rel_path, idx),
                                "content": chunk_text,
                                "file_path": rel_path,
                                "language": language,
                                "tenant_id": tenant_id,
                                "repository": repository_url,
                                "chunk_index": idx,
                                "file_hash": file_hash,
                            }
                        )
                    stale_ids.extend(_chunk_id(repository_url, rel_path, idx) for idx in stale_range)
                    existing_id = existing.id if existing is not None else None
                    index_upserts.append((rel_path, file_hash, len(file_chunks), language, existing_id))
                    files_processed += 1

            deleted_paths = set(existing_by_path) - seen_paths
            deleted_ids = [existing_by_path[path].id for path in deleted_paths]
            for path in deleted_paths:
                stale_row = existing_by_path[path]
                stale_ids.extend(_chunk_id(repository_url, path, idx) for idx in range(stale_row.chunk_count))

            vs = await self._get_vs()
            if stale_ids:
                await vs.delete_by_ids(tenant_id, stale_ids)
            upserted = await vs.upsert_chunks(tenant_id, chunks)

            # A fresh session for the writeback — existing_by_path's rows were
            # loaded in (and detached from) the earlier read-only session, so
            # each is re-fetched here by primary key rather than reused
            # directly, avoiding any ambiguity around reattaching a
            # cross-session ORM instance.
            async with get_db_context() as db:
                for path, file_hash, chunk_count, language, existing_id in index_upserts:
                    row = await db.get(CodebaseIndex, existing_id) if existing_id is not None else None
                    if row is not None:
                        row.file_hash = file_hash
                        row.chunk_count = chunk_count
                        row.language = language
                        row.last_indexed_at = datetime.now(UTC)
                    else:
                        db.add(
                            CodebaseIndex(
                                tenant_id=tenant_uuid,
                                repository_url=repository_url,
                                file_path=path,
                                file_hash=file_hash,
                                chunk_count=chunk_count,
                                language=language,
                            )
                        )
                for row_id in deleted_ids:
                    to_delete = await db.get(CodebaseIndex, row_id)
                    if to_delete is not None:
                        await db.delete(to_delete)

            result = {
                "repository": repository_url,
                "commit_sha": commit_sha,
                "branch": branch,
                "files_processed": files_processed,
                "files_unchanged": files_unchanged,
                "files_deleted": len(deleted_paths),
                "files_skipped": files_skipped,
                "chunks_created": len(chunks),
                "chunks_upserted": upserted,
                "stale_chunks_deleted": len(stale_ids),
            }
            logger.info("ingestion.complete", **result)
            return result

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _chunk_code(self, code: str, chunk_size: int, overlap: int) -> list[str]:
        if len(code) <= chunk_size:
            return [code]

        chunks = []
        lines = code.split("\n")
        current_chunk: list[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                overlap_lines = []
                overlap_len = 0
                for prev_line in reversed(current_chunk):
                    if overlap_len + len(prev_line) + 1 > overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_len += len(prev_line) + 1
                current_chunk = overlap_lines
                current_len = overlap_len
            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks
