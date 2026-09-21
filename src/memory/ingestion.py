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
from pathlib import Path
from urllib.parse import urlparse

import structlog

from src.config import get_settings
from src.memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)


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
            await self._vector_store.ensure_collection()
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
        import git

        settings = get_settings()
        _validate_repo_url(repository_url, settings.git_allowed_hosts)

        if file_extensions is None:
            file_extensions = list(LANGUAGE_MAP.keys())

        tmp_dir = tempfile.mkdtemp(prefix="vs_ingest_")
        try:
            logger.info("ingestion.cloning", repo=repository_url, branch=branch)
            repo = git.Repo.clone_from(repository_url, tmp_dir, branch=branch, depth=1)
            commit_sha = repo.head.commit.hexsha

            chunks = []
            files_processed = 0
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
                    language = LANGUAGE_MAP.get(ext, "text")
                    file_hash = hashlib.sha256(content.encode()).hexdigest()

                    file_chunks = self._chunk_code(content, chunk_size, chunk_overlap)
                    for idx, chunk_text in enumerate(file_chunks):
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repository_url}:{rel_path}:{idx}"))
                        chunks.append(
                            {
                                "id": chunk_id,
                                "content": chunk_text,
                                "file_path": rel_path,
                                "language": language,
                                "tenant_id": tenant_id,
                                "repository": repository_url,
                                "chunk_index": idx,
                                "file_hash": file_hash,
                            }
                        )
                    files_processed += 1

            vs = await self._get_vs()
            upserted = await vs.upsert_chunks(chunks)

            result = {
                "repository": repository_url,
                "commit_sha": commit_sha,
                "branch": branch,
                "files_processed": files_processed,
                "files_skipped": files_skipped,
                "chunks_created": len(chunks),
                "chunks_upserted": upserted,
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
