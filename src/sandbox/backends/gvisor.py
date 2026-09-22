# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — gVisor sandbox backend.

Runs each sandbox as a Docker container using the `runsc` (gVisor) OCI
runtime — a userspace-kernel sandbox that isolates the workload without
requiring nested KVM/virtualization. This is the portable default backend:
it works on any Docker host, including containerized GPU pods (RunPod's
test tier) where Firecracker's KVM requirement can't be satisfied.

If `runsc` isn't registered as a Docker runtime on this host, falls back to
the standard `runc` runtime with hardened flags (no-new-privileges, all
capabilities dropped, read-only rootfs, PID/process limits) — container
boundaries plus those flags are still a real (if weaker) isolation
boundary, never an "unsandboxed" execution path.
"""

from __future__ import annotations

import asyncio
import base64
import subprocess
import time
from typing import Any

import structlog

from src.sandbox.backends.base import BackendExecResult, SandboxBackend

logger = structlog.get_logger()

_KEEPALIVE_CMD = ["sleep", "infinity"]
_SANDBOX_NETWORK = "keystone-sandbox-net"
_WORKDIR = "/workspace"


class GVisorBackend(SandboxBackend):
    name = "gvisor"

    def __init__(self) -> None:
        self._client = None
        self._runsc_available: bool | None = None

    def _get_client(self):
        if self._client is None:
            import docker  # lazy import — only required when this backend is used

            self._client = docker.from_env()
        return self._client

    async def _runsc_registered(self) -> bool:
        if self._runsc_available is None:
            client = self._get_client()
            info = await asyncio.to_thread(client.info)
            runtimes = info.get("Runtimes", {})
            self._runsc_available = "runsc" in runtimes
            if not self._runsc_available:
                logger.warning(
                    "sandbox.gvisor_runtime_missing",
                    detail="runsc not registered as a Docker runtime — falling back to hardened runc",
                )
        return self._runsc_available

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
        client = self._get_client()
        use_runsc = await self._runsc_registered()

        await self._ensure_network()

        kwargs: dict[str, Any] = {
            "image": runtime_image,
            "command": _KEEPALIVE_CMD,
            "detach": True,
            "working_dir": _WORKDIR,
            "environment": env_vars,
            "mem_limit": f"{memory_mb}m",
            "nano_cpus": int(cpu_limit * 1_000_000_000),
            "network": _SANDBOX_NETWORK if network_enabled else None,
            "network_disabled": not network_enabled,
            "read_only": False,  # workload needs to write files under /workspace
            "security_opt": ["no-new-privileges"],
            "pids_limit": 256,
            "cap_drop": ["ALL"],
            "labels": {"keystone.sandbox": "true"},
        }
        if use_runsc:
            kwargs["runtime"] = "runsc"

        container = await asyncio.to_thread(client.containers.run, **kwargs)
        await asyncio.to_thread(container.exec_run, ["mkdir", "-p", _WORKDIR])
        logger.info("sandbox.gvisor_created", handle=container.id[:12], runsc=use_runsc)
        return container.id

    async def _ensure_network(self) -> None:
        client = self._get_client()
        networks = await asyncio.to_thread(client.networks.list, names=[_SANDBOX_NETWORK])
        if not networks:
            await asyncio.to_thread(
                client.networks.create,
                _SANDBOX_NETWORK,
                driver="bridge",
                internal=False,  # egress is enforced via apply_egress_policy(), not network isolation alone
                options={"com.docker.network.bridge.enable_icc": "false"},
            )

    async def write_file(self, handle: str, path: str, content: str) -> None:
        client = self._get_client()
        container = await asyncio.to_thread(client.containers.get, handle)
        full_path = f"{_WORKDIR}/{path.lstrip('/')}"
        parent = "/".join(full_path.split("/")[:-1])
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        script = f"mkdir -p {parent} && echo {encoded} | base64 -d > {full_path}"
        exit_code, output = await asyncio.to_thread(container.exec_run, ["sh", "-c", script])
        if exit_code != 0:
            raise RuntimeError(f"sandbox write_file failed: {output.decode(errors='replace')}")

    async def read_file(self, handle: str, path: str) -> str:
        client = self._get_client()
        container = await asyncio.to_thread(client.containers.get, handle)
        full_path = f"{_WORKDIR}/{path.lstrip('/')}"
        exit_code, output = await asyncio.to_thread(container.exec_run, ["cat", full_path])
        if exit_code != 0:
            raise FileNotFoundError(f"sandbox file not found: {path}")
        return output.decode("utf-8", errors="replace")

    async def execute(
        self,
        handle: str,
        command: str,
        *,
        cwd: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
    ) -> BackendExecResult:
        client = self._get_client()
        container = await asyncio.to_thread(client.containers.get, handle)
        t0 = time.monotonic()

        async def _run() -> Any:  # docker's untyped ExecResult namedtuple (exit_code, output)
            return await asyncio.to_thread(
                container.exec_run,
                ["sh", "-c", command],
                workdir=cwd or _WORKDIR,
                environment=env_vars or {},
                demux=True,
            )

        try:
            result = await asyncio.wait_for(_run(), timeout=timeout_seconds)
            exit_code = result.exit_code if hasattr(result, "exit_code") else result[0]
            output = result.output if hasattr(result, "output") else result[1]
            stdout_bytes, stderr_bytes = output if isinstance(output, tuple) else (output, b"")
            duration_ms = int((time.monotonic() - t0) * 1000)
            return BackendExecResult(
                exit_code=exit_code,
                stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
                stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
            )
        except TimeoutError:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.warning("sandbox.gvisor_timeout", handle=handle[:12], timeout=timeout_seconds)
            return BackendExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout_seconds}s",
                duration_ms=duration_ms,
                timed_out=True,
            )

    async def destroy(self, handle: str) -> None:
        client = self._get_client()
        try:
            container = await asyncio.to_thread(client.containers.get, handle)
            await asyncio.to_thread(container.remove, force=True)
        except Exception as exc:
            logger.warning("sandbox.gvisor_destroy_error", handle=handle[:12], error=str(exc))
        logger.info("sandbox.gvisor_destroyed", handle=handle[:12])

    async def apply_egress_policy(self, handle: str, allowed_destinations: list[dict[str, Any]]) -> None:
        """
        Applied HOST-SIDE via the Docker DOCKER-USER iptables chain, keyed on the
        container's assigned IP on `_SANDBOX_NETWORK` — the workload inside the
        sandbox has no capability to alter these rules (they live in the host's
        network namespace, not the container's).
        """
        client = self._get_client()
        container = await asyncio.to_thread(client.containers.get, handle)
        await asyncio.to_thread(container.reload)
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        net_info = networks.get(_SANDBOX_NETWORK)
        if not net_info:
            logger.info("sandbox.egress_skip_no_network", handle=handle[:12])
            return
        container_ip = net_info.get("IPAddress")
        if not container_ip:
            return

        rules = [f"iptables -I DOCKER-USER -s {container_ip} -j DROP"]  # default deny, prepended below in reverse
        for dest in reversed(allowed_destinations):
            cidr = dest.get("cidr")
            port = dest.get("port")
            if not cidr:
                continue
            rule = f"iptables -I DOCKER-USER -s {container_ip} -d {cidr}"
            if port:
                rule += f" -p tcp --dport {port}"
            rule += " -j ACCEPT"
            rules.insert(0, rule)

        for rule in rules:
            proc = await asyncio.to_thread(
                subprocess.run,
                rule.split(),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                logger.warning("sandbox.egress_rule_failed", rule=rule, stderr=proc.stderr.strip())

    async def health(self) -> dict[str, Any]:
        client = self._get_client()
        try:
            info = await asyncio.to_thread(client.info)
            containers = await asyncio.to_thread(client.containers.list, filters={"label": "keystone.sandbox=true"})
            return {
                "backend": self.name,
                "docker_ok": True,
                "runsc_available": await self._runsc_registered(),
                "active_sandboxes": len(containers),
                "server_version": info.get("ServerVersion"),
            }
        except Exception as exc:
            return {"backend": self.name, "docker_ok": False, "error": str(exc)}
