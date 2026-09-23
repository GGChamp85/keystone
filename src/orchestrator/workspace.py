# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — sandboxed git workspace.

One `Workspace` = one real clone of the target repository inside a single
sandbox for the lifetime of a task — the foundation the real git workflow
(clone -> branch -> commit -> push -> PR) and the quality/testing nodes are
built on. Wraps `SandboxManager` rather than replacing it; every operation
here is just `execute()`/`write_file()`/`read_file()` calls against the
sandbox, so it inherits the daemon's egress policy, resource limits, and
zero-retention teardown for free.

Git credentials are never embedded in the remote URL (which would leak into
`git remote -v` / error messages / `.git/config`) or written into the
sandbox's persistent environment — instead passed per-command via
`git -c http.extraHeader=...`, which git accepts on argv without ever
writing the header value to disk in the sandbox. Visible only momentarily
in that command's own process listing during execution, the same exposure
window any CI system's git-credential-in-argv approach accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from src.config import get_settings
from src.memory.ingestion import InvalidRepositoryURLError, _validate_repo_url
from src.sandbox.manager import SandboxManager

logger = structlog.get_logger(__name__)

REPO_DIR = "repo"  # relative to the sandbox's /workspace cwd


class WorkspaceCommandError(RuntimeError):
    """A workspace git/shell command exited non-zero. Carries stdout/stderr for the caller to act on or surface."""

    def __init__(self, command: str, exit_code: int, stdout: str, stderr: str):
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"command failed ({exit_code}): {command}\n{stderr}")


@dataclass
class CloneResult:
    default_branch: str
    commit_sha: str


class Workspace:
    """
    Reuse note: this is deliberately thin — it does not reimplement sandbox
    lifecycle (SandboxManager.get_or_create already handles idempotent
    creation for Temporal retries) or repository-URL SSRF validation
    (reuses `src.memory.ingestion._validate_repo_url`, the same guard the
    RAG ingestion pipeline already uses, rather than a second copy of the
    same logic).
    """

    def __init__(self, manager: SandboxManager, task_id: str, tenant_id: str = "default"):
        self._manager = manager
        self._task_id = task_id
        self._tenant_id = tenant_id
        self._handle: str | None = None
        self._repo_cwd = f"/workspace/{REPO_DIR}"
        self.language = "python"  # the runtime template this sandbox was created with
        self._network_enabled = True

    async def ensure_sandbox(self, *, network_enabled: bool = True, language: str | None = None) -> str:
        if self._handle is None:
            self.language = language or self.language
            self._network_enabled = network_enabled
            self._handle = await self._manager.get_or_create(
                self._task_id, tenant_id=self._tenant_id, language=self.language, network_enabled=network_enabled
            )
        return self._handle

    async def switch_runtime(self, language: str) -> str:
        """Replace this task's sandbox with one built from another runtime image (`language` is a
        template name the daemon maps to an image: python, javascript, typescript, go, ...). The
        working tree is not carried over — call `clone` again afterwards. Used once, right after the
        first clone, when the repository turns out to be Node or Go rather than Python."""
        if self._handle is not None:
            await self._manager.destroy(self._handle)
            self._manager.forget(self._task_id)
            self._handle = None
        self.language = language
        return await self.ensure_sandbox(network_enabled=self._network_enabled, language=language)

    def _repo_relative(self, path: str) -> str:
        """
        `SandboxManager.write_file`/`read_file` always resolve their `path`
        relative to the sandbox's fixed /workspace root (a leading "/" is
        stripped, not treated as absolute — verified against the daemon's
        real behavior). `Workspace` methods take a path relative to the
        repo checkout (matching `run()`'s default cwd); this bridges the
        two so callers never have to know about that daemon quirk.
        """
        clean = path.lstrip("/")
        if clean.startswith(f"{REPO_DIR}/") or clean == REPO_DIR:
            return clean  # already repo-rooted (e.g. passed self._repo_cwd + "/x")
        return f"{REPO_DIR}/{clean}"

    async def write_file(self, path: str, content: str) -> None:
        """`path` relative to the repo root, e.g. "src/app.py" — or the full `self._repo_cwd`-prefixed form."""
        handle = await self.ensure_sandbox()
        await self._manager.write_file(handle, self._repo_relative(path), content)

    async def read_file(self, path: str) -> str:
        handle = await self.ensure_sandbox()
        return await self._manager.read_file(handle, self._repo_relative(path))

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int = 120,
        env_vars: dict[str, str] | None = None,
        check: bool = True,
    ) -> dict[str, Any]:
        handle = await self.ensure_sandbox()
        result = await self._manager.execute(
            handle, command, timeout=timeout, cwd=cwd or self._repo_cwd, env_vars=env_vars
        )
        if check and result.get("exit_code") != 0:
            raise WorkspaceCommandError(
                command, result.get("exit_code", -1), result.get("stdout", ""), result.get("stderr", "")
            )
        return result

    async def clone(self, repository_url: str, branch: str) -> CloneResult:
        """
        Clone `repository_url` into the sandbox's /workspace/repo. Validates
        the URL the same way RAG ingestion does (allowlisted git host,
        https/ssh only, no SSRF targets) before ever handing it to `git`.
        Uses `--filter=blob:none` (a partial/blobless clone) so a large
        monorepo doesn't pull every historical blob into a short-lived
        per-task sandbox — file contents are fetched on demand as the
        working tree is checked out and read.
        """
        settings = get_settings()
        try:
            _validate_repo_url(repository_url, settings.git_allowed_hosts)
        except InvalidRepositoryURLError:
            raise

        token = settings.git_host_token.get_secret_value() if settings.git_host_token else None
        auth_header = f"Authorization: token {token}" if token else None

        await self.run("mkdir -p /workspace", cwd="/", check=True)
        if auth_header:
            clone_cmd = (
                f'git -c http.extraHeader="{auth_header}" clone --filter=blob:none '
                f"--branch {branch} {repository_url} {REPO_DIR}"
            )
        else:
            clone_cmd = f"git clone --filter=blob:none --branch {branch} {repository_url} {REPO_DIR}"
        await self.run(clone_cmd, cwd="/workspace", timeout=300)

        sha_result = await self.run("git rev-parse HEAD")
        commit_sha = sha_result["stdout"].strip()
        logger.info("workspace.cloned", task_id=self._task_id, repo=repository_url, branch=branch, sha=commit_sha[:12])
        return CloneResult(default_branch=branch, commit_sha=commit_sha)

    async def list_files(self, max_depth: int = 2) -> set[str]:
        """Relative file/dir names near the repo root — feeds `repo_profile.detect_repo_profile`."""
        result = await self.run(f"find . -maxdepth {max_depth} -mindepth 1 -printf '%P\\n'", check=False)
        return {line.strip() for line in result.get("stdout", "").splitlines() if line.strip()}

    async def list_tracked_files(self) -> set[str]:
        """Every git-tracked path (relative, any depth) — feeds `test_scope.related_test_command`."""
        result = await self.run("git ls-files", check=False)
        return {line.strip() for line in result.get("stdout", "").splitlines() if line.strip()}

    async def read_package_json_scripts(self) -> dict[str, str]:
        result = await self.run("cat package.json", check=False)
        if result.get("exit_code") != 0:
            return {}
        import json

        try:
            return json.loads(result["stdout"]).get("scripts", {})
        except (json.JSONDecodeError, AttributeError):
            return {}

    async def create_branch(self, branch_name: str) -> None:
        await self.run(f"git checkout -b {branch_name}")

    async def diff_stat(self, base: str = "HEAD") -> str:
        result = await self.run(f"git diff --stat {base}", check=False)
        return result.get("stdout", "")

    async def diff(self, base: str = "HEAD") -> str:
        result = await self.run(f"git diff {base}", check=False)
        return result.get("stdout", "")

    async def commit_all(self, message: str) -> str:
        settings = get_settings()
        await self.run(
            f'git -c user.name="{settings.git_host_commit_author_name}" '
            f'-c user.email="{settings.git_host_commit_author_email}" add -A'
        )
        await self.run(
            f'git -c user.name="{settings.git_host_commit_author_name}" '
            f'-c user.email="{settings.git_host_commit_author_email}" '
            f'commit -m "{message}"'
        )
        sha_result = await self.run("git rev-parse HEAD")
        return sha_result["stdout"].strip()

    async def push(self, branch_name: str) -> None:
        """Pushes HEAD to `branch_name` on the `origin` remote set up by `clone()`."""
        settings = get_settings()
        token = settings.git_host_token.get_secret_value() if settings.git_host_token else None
        if not token:
            raise WorkspaceCommandError("git push", -1, "", "git_host_token is not configured — cannot push")
        auth_header = f"Authorization: token {token}"
        push_cmd = f'git -c http.extraHeader="{auth_header}" push origin HEAD:{branch_name}'
        await self.run(push_cmd, timeout=120)
        logger.info("workspace.pushed", task_id=self._task_id, branch=branch_name)

    async def close(self) -> None:
        if self._handle is not None:
            await self._manager.destroy(self._handle)
            self._handle = None
