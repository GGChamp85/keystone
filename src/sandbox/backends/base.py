# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Sandbox backend interface.

Every self-hosted execution backend (Firecracker microVMs, gVisor containers)
implements this interface. Callers (src/sandbox/daemon.py) are backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


class SandboxBackend(ABC):
    """
    Common interface for self-hosted sandbox execution backends.

    A "handle" is an opaque backend-specific identifier (a container ID for
    gVisor, a microVM ID for Firecracker) that the caller treats as an opaque
    string and passes back into every subsequent call.
    """

    name: str = "base"

    @abstractmethod
    async def create(
        self,
        *,
        runtime_image: str,
        cpu_limit: float,
        memory_mb: int,
        network_enabled: bool,
        env_vars: dict[str, str],
        timeout_seconds: int,
    ) -> str:
        """Create a new isolated sandbox. Returns an opaque handle."""

    @abstractmethod
    async def write_file(self, handle: str, path: str, content: str) -> None:
        """Write a file into the sandbox's working directory."""

    @abstractmethod
    async def read_file(self, handle: str, path: str) -> str:
        """Read a file from the sandbox's working directory."""

    @abstractmethod
    async def execute(
        self,
        handle: str,
        command: str,
        *,
        cwd: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
    ) -> BackendExecResult:
        """Execute a shell command inside the sandbox."""

    @abstractmethod
    async def destroy(self, handle: str) -> None:
        """Destroy the sandbox — zero-retention, instant wipe."""

    @abstractmethod
    async def apply_egress_policy(self, handle: str, allowed_destinations: list[dict[str, Any]]) -> None:
        """
        Apply network egress filtering to the sandbox FROM THE HOST SIDE
        (the network namespace / veth pair the backend controls), not by
        running iptables inside the sandbox itself — a workload with shell
        access inside the sandbox must never be able to alter its own
        firewall rules.
        """

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Backend health/capacity info, surfaced on the daemon's /health route."""
