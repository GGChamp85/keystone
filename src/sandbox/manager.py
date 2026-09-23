# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Sandbox Manager.

Thin async client for the self-hosted sandbox execution daemon
(`keystoned`, src/sandbox/daemon.py) — Firecracker microVMs, or gVisor
containers where KVM isn't available. Replaces the previous direct
dependency on E2B's hosted cloud SDK: nothing in this module talks to any
third-party SaaS, only to `keystoned` inside the client's own network.

Security measures (enforced by the daemon, not this client):
  - Egress filtering: host-side network-namespace rules, allowlisted
    internal mirrors only — the sandboxed workload cannot alter its own
    firewall rules (unlike the old sandbox-side-iptables approach).
  - Ephemeral secrets: injected per-execution, wiped on close.
  - Resource limits: CPU, memory, disk, execution time.
  - Zero retention: instant destroy on close.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


class SandboxManager:
    """
    Manages sandbox lifecycle via keystoned: create, file I/O, execute, destroy.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = httpx.AsyncClient(base_url=self.settings.sandbox_daemon_url, timeout=30.0)
        # task_id -> handle, so a retried Temporal activity can find its own
        # sandbox instead of leaking one (see get_or_create()).
        self._handles_by_task: dict[str, str] = {}

    async def create(
        self,
        template: str = "python",
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
        tenant_id: str = "default",
        network_enabled: bool = False,
    ) -> str:
        """
        Create a new self-hosted sandbox. Returns the sandbox handle.
        """
        safe_env = self._sanitize_env_vars(env_vars or {})
        resp = await self._client.post(
            "/sandboxes",
            json={
                "tenant_id": tenant_id,
                "template": template,
                "timeout_seconds": timeout or self.settings.sandbox_timeout_seconds,
                "env_vars": safe_env,
                "network_enabled": network_enabled,
                "max_concurrent_per_tenant": self.settings.sandbox_max_concurrent,
            },
            # A network-enabled sandbox applies one real iptables rule per
            # configured egress host (src/sandbox/security.py's
            # build_egress_policy) before the daemon responds — each one a
            # real OS-level DNS lookup that can legitimately take several
            # seconds. The client-wide 30s default is tuned for the common
            # no-network case; this call specifically needs real headroom.
            timeout=90.0 if network_enabled else 30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info("sandbox.created", handle=data["handle"], backend=data["backend"])
        return data["handle"]

    async def get_or_create(
        self,
        task_id: str,
        tenant_id: str = "default",
        language: str = "python",
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
        network_enabled: bool = False,
    ) -> str:
        """
        Idempotent create, keyed by task_id — safe to call from a retried
        Temporal activity without leaking an orphaned sandbox per retry.
        `network_enabled` only takes effect on the first call for a given
        task_id (the one that actually creates the sandbox) — a later call
        passing a different value does not retroactively change an
        already-created sandbox's egress.
        """
        existing = self._handles_by_task.get(task_id)
        if existing is not None:
            return existing
        handle = await self.create(
            template=language,
            timeout=timeout,
            env_vars=env_vars,
            tenant_id=tenant_id,
            network_enabled=network_enabled,
        )
        self._handles_by_task[task_id] = handle
        return handle

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        resp = await self._client.post(f"/sandboxes/{sandbox_id}/files", json={"path": path, "content": content})
        resp.raise_for_status()

    async def read_file(self, sandbox_id: str, path: str) -> str:
        resp = await self._client.get(f"/sandboxes/{sandbox_id}/files", params={"path": path})
        resp.raise_for_status()
        return resp.json()["content"]

    async def execute(
        self,
        sandbox_id: str,
        command: str,
        timeout: int = 120,
        cwd: str = "/workspace",
        env_vars: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a command in the sandbox.
        Returns {exit_code, stdout, stderr, duration_ms}.
        """
        resp = await self._client.post(
            f"/sandboxes/{sandbox_id}/exec",
            json={
                "command": command,
                "cwd": cwd,
                "timeout_seconds": timeout,
                "env_vars": self._sanitize_env_vars(env_vars or {}),
            },
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        return resp.json()

    async def execute_python(
        self,
        sandbox_id: str,
        code: str,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Execute Python code by writing it to a scratch file and running it."""
        scratch_path = "_keystone_exec_scratch.py"
        await self.write_file(sandbox_id, scratch_path, code)
        return await self.execute(sandbox_id, f"python {scratch_path}", timeout=timeout)

    async def destroy(self, sandbox_id: str) -> None:
        """Destroy a sandbox — zero-retention instant wipe."""
        try:
            resp = await self._client.delete(f"/sandboxes/{sandbox_id}")
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("sandbox.destroy_error", sandbox_id=sandbox_id, error=str(exc))
        self._handles_by_task = {k: v for k, v in self._handles_by_task.items() if v != sandbox_id}
        logger.info("sandbox.destroyed", sandbox_id=sandbox_id)

    def forget(self, task_id: str) -> None:
        """Drop the cached handle for `task_id` (after destroying it) so the next get_or_create builds a new
        sandbox — how a task switches runtime image once the clone reveals the repository's ecosystem."""
        self._handles_by_task.pop(task_id, None)

    async def destroy_all(self) -> None:
        """Destroy all sandboxes this manager instance knows about."""
        ids = list(set(self._handles_by_task.values()))
        for sid in ids:
            await self.destroy(sid)

    async def health(self) -> dict[str, Any]:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Internal helpers ──────────────────────────────────────

    def _sanitize_env_vars(self, env_vars: dict[str, str]) -> dict[str, str]:
        """
        Strip dangerous env vars. Never pass through cloud credentials,
        SSH keys, or platform secrets.
        """
        blocked_prefixes = (
            "AWS_",
            "AZURE_",
            "GCP_",
            "GOOGLE_",
            "GITHUB_TOKEN",
            "SSH_",
            "VS_SECRET",
            "DATABASE_URL",
            "REDIS_",
            "QDRANT_API",
            "KEYSTONE_ROOT_ADMIN",
        )
        return {k: v for k, v in env_vars.items() if not any(k.upper().startswith(bp) for bp in blocked_prefixes)}


_manager_singleton: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    """
    Process-wide shared SandboxManager — needed so `get_or_create(task_id)`'s
    idempotency (its `_handles_by_task` cache) actually spans multiple
    LangGraph node calls for the same task, not just retries within one
    node. A task's `Workspace` (src/orchestrator/workspace.py) is built on
    this so the same sandboxed git checkout persists across
    planning/coding/review/testing/fixing iterations instead of a fresh
    sandbox (and a fresh, unrelated clone) on every node call.
    """
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = SandboxManager()
    return _manager_singleton
