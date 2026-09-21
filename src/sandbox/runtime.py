"""
Keystone — Sandbox Runtime Management
Manages sandbox lifecycle, resource limits, runtime selection,
and health monitoring for self-hosted Firecracker/gVisor execution
environments (via the keystoned daemon — src/sandbox/manager.py).

Copyright 2024-2026 Gaurav Gupta — Apache 2.0 License
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from src.sandbox.manager import SandboxManager
from src.sandbox.security import (
    EgressPolicy,
    ScanResult,
    SecretVault,
    sanitize_environment,
    scan_sandbox_output,
)

logger = structlog.get_logger()


# ── Enums ────────────────────────────────────────────────────────────


class RuntimeLanguage(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    JAVA = "java"
    SHELL = "shell"


class SandboxStatus(StrEnum):
    CREATING = "creating"
    READY = "ready"
    EXECUTING = "executing"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"


# ── Resource Limits ──────────────────────────────────────────────────


@dataclass
class ResourceLimits:
    """Resource constraints for a sandbox instance."""

    max_cpu_seconds: int = 120  # CPU time limit
    max_wall_clock_seconds: int = 300  # Wall-clock timeout
    max_memory_mb: int = 2048  # Memory limit in MB
    max_disk_mb: int = 1024  # Disk space limit in MB
    max_processes: int = 64  # Process count limit
    max_open_files: int = 1024  # File descriptor limit
    max_output_bytes: int = 10_485_760  # 10 MB output cap
    network_enabled: bool = False  # Network access (default off)

    def to_backend_options(self) -> dict[str, Any]:
        """Convert to self-hosted sandbox backend options (daemon create() kwargs)."""
        return {
            "timeout": self.max_wall_clock_seconds,
            "memory_mb": self.max_memory_mb,
            "cpu_limit": max(1.0, self.max_cpu_seconds / 60),
            "network_enabled": self.network_enabled,
        }

    def to_ulimit_commands(self) -> list[str]:
        """Generate ulimit commands for shell-based execution."""
        return [
            f"ulimit -t {self.max_cpu_seconds}",
            f"ulimit -v {self.max_memory_mb * 1024}",  # KB
            f"ulimit -u {self.max_processes}",
            f"ulimit -n {self.max_open_files}",
            f"ulimit -f {self.max_disk_mb * 1024}",  # KB
        ]


# Default resource limits per tier
TIER_LIMITS: dict[str, ResourceLimits] = {
    "free": ResourceLimits(
        max_cpu_seconds=30,
        max_wall_clock_seconds=60,
        max_memory_mb=512,
        max_disk_mb=256,
        max_processes=16,
    ),
    "pro": ResourceLimits(
        max_cpu_seconds=120,
        max_wall_clock_seconds=300,
        max_memory_mb=2048,
        max_disk_mb=1024,
        max_processes=64,
    ),
    "enterprise": ResourceLimits(
        max_cpu_seconds=600,
        max_wall_clock_seconds=1800,
        max_memory_mb=8192,
        max_disk_mb=4096,
        max_processes=256,
        network_enabled=True,  # Enterprise can opt-in to network
    ),
}


# ── Runtime Templates ────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeTemplate:
    """Pre-configured runtime environment template."""

    language: RuntimeLanguage
    runtime_key: str  # key into daemon.py's RUNTIME_IMAGES map
    default_command: str  # How to run code
    file_extension: str  # Default file extension
    install_command: str  # Package install command
    test_command: str  # Test runner command
    setup_commands: list[str] = field(default_factory=list)

    def run_command(self, filepath: str) -> str:
        """Generate the command to run a specific file."""
        return self.default_command.format(filepath=filepath)


RUNTIME_TEMPLATES: dict[RuntimeLanguage, RuntimeTemplate] = {
    RuntimeLanguage.PYTHON: RuntimeTemplate(
        language=RuntimeLanguage.PYTHON,
        runtime_key="python",
        default_command="python {filepath}",
        file_extension=".py",
        install_command="pip install",
        test_command="python -m pytest {filepath} -v --tb=short 2>&1",
        setup_commands=[
            "pip install --quiet pytest httpx pydantic structlog 2>/dev/null",
        ],
    ),
    RuntimeLanguage.JAVASCRIPT: RuntimeTemplate(
        language=RuntimeLanguage.JAVASCRIPT,
        runtime_key="javascript",
        default_command="node {filepath}",
        file_extension=".js",
        install_command="npm install --save",
        test_command="npx jest {filepath} --verbose 2>&1",
        setup_commands=[
            "npm init -y 2>/dev/null",
        ],
    ),
    RuntimeLanguage.TYPESCRIPT: RuntimeTemplate(
        language=RuntimeLanguage.TYPESCRIPT,
        runtime_key="typescript",
        default_command="npx tsx {filepath}",
        file_extension=".ts",
        install_command="npm install --save",
        test_command="npx jest {filepath} --verbose 2>&1",
        setup_commands=[
            "npm init -y 2>/dev/null",
            "npm install --save-dev typescript tsx @types/node jest ts-jest 2>/dev/null",
        ],
    ),
    RuntimeLanguage.GO: RuntimeTemplate(
        language=RuntimeLanguage.GO,
        runtime_key="go",
        default_command="go run {filepath}",
        file_extension=".go",
        install_command="go get",
        test_command="go test -v -run {filepath} ./... 2>&1",
        setup_commands=[
            "go mod init sandbox 2>/dev/null || true",
        ],
    ),
    RuntimeLanguage.RUST: RuntimeTemplate(
        language=RuntimeLanguage.RUST,
        runtime_key="rust",
        default_command="rustc {filepath} -o /tmp/sandbox_bin && /tmp/sandbox_bin",
        file_extension=".rs",
        install_command="cargo add",
        test_command="rustc --test {filepath} -o /tmp/sandbox_test && /tmp/sandbox_test 2>&1",
        setup_commands=[],
    ),
    RuntimeLanguage.RUBY: RuntimeTemplate(
        language=RuntimeLanguage.RUBY,
        runtime_key="ruby",
        default_command="ruby {filepath}",
        file_extension=".rb",
        install_command="gem install",
        test_command="ruby -Ilib -Itest {filepath} 2>&1",
        setup_commands=[],
    ),
    RuntimeLanguage.SHELL: RuntimeTemplate(
        language=RuntimeLanguage.SHELL,
        runtime_key="base",
        default_command="bash {filepath}",
        file_extension=".sh",
        install_command="apt-get install -y",
        test_command="bash -n {filepath} && bash {filepath} 2>&1",
        setup_commands=[],
    ),
}


def detect_language(filepath: str, content: str = "") -> RuntimeLanguage:
    """Detect the programming language from file extension or content."""
    ext_map = {
        ".py": RuntimeLanguage.PYTHON,
        ".js": RuntimeLanguage.JAVASCRIPT,
        ".mjs": RuntimeLanguage.JAVASCRIPT,
        ".ts": RuntimeLanguage.TYPESCRIPT,
        ".tsx": RuntimeLanguage.TYPESCRIPT,
        ".go": RuntimeLanguage.GO,
        ".rs": RuntimeLanguage.RUST,
        ".rb": RuntimeLanguage.RUBY,
        ".sh": RuntimeLanguage.SHELL,
        ".bash": RuntimeLanguage.SHELL,
    }

    for ext, lang in ext_map.items():
        if filepath.endswith(ext):
            return lang

    # Content-based detection
    if content:
        if content.startswith("#!/usr/bin/env python") or "import " in content[:200]:
            return RuntimeLanguage.PYTHON
        if content.startswith("#!/usr/bin/env node") or "require(" in content[:200]:
            return RuntimeLanguage.JAVASCRIPT
        if "package main" in content[:200]:
            return RuntimeLanguage.GO
        if "fn main()" in content[:200]:
            return RuntimeLanguage.RUST

    return RuntimeLanguage.PYTHON  # Default


# ── Sandbox Instance ─────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Result of a single command execution in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    killed: bool = False
    scan_result: ScanResult | None = None


@dataclass
class SandboxInstance:
    """
    Represents a live sandbox instance with its lifecycle state.
    Wraps self-hosted sandbox (Firecracker/gVisor, via keystoned) operations
    with security controls.
    """

    instance_id: str = field(default_factory=lambda: f"sbx-{uuid.uuid4().hex[:12]}")
    tenant_id: str = ""
    task_id: str = ""
    language: RuntimeLanguage = RuntimeLanguage.PYTHON
    status: SandboxStatus = SandboxStatus.CREATING
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    egress_policy: EgressPolicy = field(default_factory=EgressPolicy)
    secret_vault: SecretVault = field(default_factory=SecretVault)

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    terminated_at: float | None = None

    total_cpu_seconds: float = 0.0
    total_executions: int = 0
    _manager: Any = None  # Shared SandboxManager (injected by SandboxPool)
    _handle: str | None = None  # keystoned sandbox handle

    @property
    def uptime_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.terminated_at or time.time()
        return end - self.started_at

    @property
    def is_alive(self) -> bool:
        return self.status in (SandboxStatus.READY, SandboxStatus.EXECUTING)

    async def initialize(self, manager: SandboxManager) -> None:
        """
        Create and initialize the sandbox environment via keystoned.
        Sets up the runtime, applies security policies, and prepares for execution.
        """
        try:
            self.status = SandboxStatus.CREATING
            self._manager = manager
            template = RUNTIME_TEMPLATES.get(self.language)

            self._handle = await manager.create(
                template=template.runtime_key if template else "base",
                timeout=self.limits.max_wall_clock_seconds,
                tenant_id=self.tenant_id or "default",
                network_enabled=self.limits.network_enabled,
            )

            # Run setup commands (egress policy is applied by keystoned itself,
            # host-side, at create() time — not by running iptables in-sandbox)
            if template:
                for cmd in template.setup_commands:
                    try:
                        await manager.execute(self._handle, cmd, timeout=30)
                    except Exception:
                        logger.debug("setup_cmd_failed", cmd=cmd)

            self.status = SandboxStatus.READY
            self.started_at = time.time()

            logger.info(
                "sandbox_initialized",
                instance=self.instance_id,
                handle=self._handle,
                language=self.language.value,
                tenant=self.tenant_id,
            )

        except Exception as e:
            self.status = SandboxStatus.ERROR
            logger.error("sandbox_init_failed", error=str(e), instance=self.instance_id)
            raise

    async def execute(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        scan_output: bool = True,
    ) -> ExecutionResult:
        """
        Execute a command in the sandbox with security controls.
        """
        if not self.is_alive:
            raise RuntimeError(f"Sandbox {self.instance_id} is not alive (status: {self.status})")

        # Check wall-clock limit
        if self.uptime_seconds > self.limits.max_wall_clock_seconds:
            await self.terminate(reason="wall_clock_exceeded")
            raise TimeoutError(f"Sandbox exceeded wall-clock limit of {self.limits.max_wall_clock_seconds}s")

        self.status = SandboxStatus.EXECUTING
        exec_timeout = timeout or min(
            self.limits.max_cpu_seconds,
            self.limits.max_wall_clock_seconds - int(self.uptime_seconds),
        )

        # Sanitize environment variables
        safe_env = sanitize_environment(env or {})
        # Inject ephemeral secrets
        safe_env.update(self.secret_vault.to_env_vars())

        if self._manager is None or self._handle is None:
            raise RuntimeError(
                f"Sandbox {self.instance_id} has no backing keystoned sandbox — "
                f"initialize() must succeed before execute() is called"
            )

        start_time = time.time()

        try:
            result = await self._manager.execute(
                self._handle,
                command,
                timeout=exec_timeout,
                env_vars=safe_env,
            )
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            exit_code = result.get("exit_code", -1)
            timed_out = bool(result.get("timed_out", False))

        except Exception as e:
            stdout = ""
            stderr = str(e)
            exit_code = -1
            timed_out = False

        duration_ms = int((time.time() - start_time) * 1000)
        self.total_executions += 1
        self.total_cpu_seconds += duration_ms / 1000

        # Truncate output to limit
        max_output = self.limits.max_output_bytes
        stdout = stdout[:max_output]
        stderr = stderr[:max_output]

        # Security scan
        scan_result = None
        if scan_output:
            scan_result = scan_sandbox_output(stdout, stderr)
            if not scan_result.is_safe:
                logger.warning(
                    "sandbox_security_alert",
                    instance=self.instance_id,
                    risk_score=scan_result.risk_score,
                    findings=len(scan_result.findings),
                )

        self.status = SandboxStatus.READY

        logger.info(
            "sandbox_executed",
            instance=self.instance_id,
            exit_code=exit_code,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
            scan_result=scan_result,
        )

    async def write_file(self, path: str, content: str) -> None:
        """Write a file into the sandbox filesystem."""
        if self._manager and self._handle:
            await self._manager.write_file(self._handle, path, content)
        else:
            logger.debug("sandbox_write_skip", path=path, reason="no_backing_sandbox")

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox filesystem."""
        if self._manager and self._handle:
            return await self._manager.read_file(self._handle, path)
        return ""

    async def terminate(self, reason: str = "normal") -> None:
        """
        Terminate the sandbox. Zero-retention: instant wipe on close.
        """
        if self.status == SandboxStatus.TERMINATED:
            return

        self.status = SandboxStatus.STOPPING

        # Revoke all ephemeral secrets
        self.secret_vault.revoke_all()

        # Destroy the self-hosted sandbox
        if self._manager and self._handle:
            try:
                await self._manager.destroy(self._handle)
            except Exception as e:
                logger.warning("sandbox_kill_error", error=str(e))
            self._handle = None

        self.status = SandboxStatus.TERMINATED
        self.terminated_at = time.time()

        logger.info(
            "sandbox_terminated",
            instance=self.instance_id,
            reason=reason,
            uptime_s=round(self.uptime_seconds, 1),
            total_executions=self.total_executions,
        )


# ── Runtime Pool ─────────────────────────────────────────────────────


class SandboxPool:
    """
    Pool manager for sandbox instances.
    Enforces global concurrency limits and handles cleanup.
    """

    def __init__(
        self,
        max_concurrent: int = 20,
        manager: SandboxManager | None = None,
    ):
        self._instances: dict[str, SandboxInstance] = {}
        self._max_concurrent = max_concurrent
        self._manager = manager or SandboxManager()
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._instances.values() if s.is_alive)

    async def acquire(
        self,
        tenant_id: str,
        task_id: str,
        language: RuntimeLanguage = RuntimeLanguage.PYTHON,
        tier: str = "pro",
    ) -> SandboxInstance:
        """
        Acquire a new sandbox instance from the pool.
        Blocks if the pool is at capacity.
        """
        async with self._lock:
            if self.active_count >= self._max_concurrent:
                # Try to reclaim dead instances
                await self._cleanup_dead()
                if self.active_count >= self._max_concurrent:
                    raise RuntimeError(f"Sandbox pool exhausted ({self._max_concurrent} max concurrent)")

            limits = TIER_LIMITS.get(tier, TIER_LIMITS["pro"])
            instance = SandboxInstance(
                tenant_id=tenant_id,
                task_id=task_id,
                language=language,
                limits=limits,
            )

            await instance.initialize(self._manager)
            self._instances[instance.instance_id] = instance
            return instance

    async def release(self, instance_id: str) -> None:
        """Release and terminate a sandbox instance."""
        instance = self._instances.get(instance_id)
        if instance:
            await instance.terminate(reason="released")
            del self._instances[instance_id]

    async def terminate_all(self) -> None:
        """Terminate all sandbox instances (shutdown)."""
        tasks = [instance.terminate(reason="pool_shutdown") for instance in self._instances.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._instances.clear()
        logger.info("sandbox_pool_shutdown", terminated=len(tasks))

    async def _cleanup_dead(self) -> int:
        """Remove terminated or errored instances."""
        dead = [iid for iid, inst in self._instances.items() if not inst.is_alive]
        for iid in dead:
            del self._instances[iid]
        return len(dead)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_instances": len(self._instances),
            "active": self.active_count,
            "max_concurrent": self._max_concurrent,
            "by_status": {
                status.value: sum(1 for s in self._instances.values() if s.status == status) for status in SandboxStatus
            },
        }
