# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Git hosting abstraction.

Background agent tasks end with a real pull/merge request, not a JSON blob
of file contents. This module is the seam between that behavior and
whichever git host a client actually runs — Gitea by default (the
self-hosted, air-gap-friendly host this project already assumes via
`settings.git_allowed_hosts`), GitLab or GitHub Enterprise as later
implementations. Callers depend only on the `GitHost` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PullRequestInfo:
    number: int
    url: str
    html_url: str


@dataclass(frozen=True)
class PullRequestStatus:
    number: int
    state: str  # open | closed | merged
    merged: bool
    merge_commit_sha: str | None


class GitHostError(RuntimeError):
    """Raised when a git-hosting API call fails — never silently ignored,
    since a failed PR open means the agent's work exists only on an unpushed
    branch and the caller must decide how to surface that."""


class GitHost(Protocol):
    """
    A self-hosted git server's PR/MR surface. Implementations talk to one
    specific host's REST API; nothing else in the orchestrator should know
    which one is configured (`settings.git_host_kind` selects it via
    `get_git_host()` below).
    """

    async def get_default_branch(self, owner: str, repo: str) -> str: ...

    async def open_pull_request(
        self,
        owner: str,
        repo: str,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestInfo: ...

    async def get_pull_request_status(self, owner: str, repo: str, number: int) -> PullRequestStatus: ...

    async def aclose(self) -> None: ...


def parse_owner_repo(repository_url: str) -> tuple[str, str]:
    """
    `https://gitea.internal.keystone.local/myorg/myapi(.git)` -> ("myorg", "myapi").
    Shared by every GitHost implementation so URL parsing isn't duplicated
    per-host; raises ValueError on a URL that doesn't fit the owner/repo
    shape every one of these hosts uses.
    """
    from urllib.parse import urlparse

    path = urlparse(repository_url).path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Cannot parse owner/repo from repository_url: {repository_url!r}")
    return parts[0], parts[1]


_host_singleton: GitHost | None = None


def get_git_host() -> GitHost:
    """Factory keyed on settings.git_host_kind — the only place call sites need to touch to add a new host."""
    global _host_singleton
    if _host_singleton is not None:
        return _host_singleton

    from src.config import get_settings

    settings = get_settings()
    kind = settings.git_host_kind
    if kind == "gitea":
        from src.git.gitea import GiteaHost

        _host_singleton = GiteaHost()
    else:
        raise ValueError(f"Unsupported git_host_kind: {kind!r} (supported: gitea)")
    return _host_singleton


async def reset_git_host_singleton_for_tests() -> None:
    """Test-only: allow a fresh GitHost (fresh httpx client, fresh settings) between tests."""
    global _host_singleton
    if _host_singleton is not None:
        await _host_singleton.aclose()
    _host_singleton = None
