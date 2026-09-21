# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Firecracker microVM sandbox backend.

Launches a real Firecracker microVM per sandbox via `jailer` (dropping
privileges/chroot around the Firecracker process) and controls it over its
per-instance REST API (a Unix domain socket) — the same mechanism AWS
Lambda and Fly.io use for KVM-isolated, fast-booting execution.

Host prerequisites (see docs/architecture/SANDBOX_ARCHITECTURE.md):
  - `/dev/kvm` present and accessible (bare-metal or nested-virt-enabled host)
  - `firecracker` and `jailer` binaries on PATH (Apache-2.0, firecracker-microvm/firecracker)
  - Per-runtime kernel image + rootfs (built by airgap/build_image_bundle.sh),
    with a `keystone-guest-agent` init process baked in that listens on
    AF_VSOCK for a line-delimited JSON command protocol: {"op": "write_file"|
    "read_file"|"exec", ...} -> {"ok": bool, ...}. This is the microVM
    equivalent of `docker exec` — there is no host-side shell into a VM, so
    all file I/O and command execution round-trips through this guest agent.
  - A pre-booted, snapshotted VM per runtime template in the warm pool
    (`_WarmPool` below) so `create()` is a snapshot *restore*, not a cold
    boot — this is what gives sub-second sandbox creation, the same
    mechanism Modal/E2B rely on.

This module contains the real host-side orchestration. It cannot be
exercised in an environment without `/dev/kvm` (this includes most
containerized CI/dev environments and RunPod's standard GPU pods) — those
targets automatically fall back to `GVisorBackend` (see
src/sandbox/daemon.py's backend selection).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from src.sandbox.backends.base import BackendExecResult, SandboxBackend

logger = structlog.get_logger()

FIRECRACKER_BIN = shutil.which("firecracker") or "/usr/bin/firecracker"
JAILER_BIN = shutil.which("jailer") or "/usr/bin/jailer"
JAILER_CHROOT_BASE = Path("/srv/keystone/firecracker/jailer")
SNAPSHOT_DIR = Path("/srv/keystone/firecracker/snapshots")
KERNEL_DIR = Path("/srv/keystone/firecracker/kernels")
ROOTFS_DIR = Path("/srv/keystone/firecracker/rootfs")
GUEST_AGENT_VSOCK_PORT = 5252


def firecracker_available() -> bool:
    """True only if this host can actually run Firecracker microVMs."""
    kvm_usable = os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK)
    return kvm_usable and shutil.which("firecracker") is not None


class _VsockAgentClient:
    """Line-delimited JSON control channel to the in-guest keystone-guest-agent."""

    def __init__(self, uds_path: str, guest_cid: int, port: int = GUEST_AGENT_VSOCK_PORT) -> None:
        # Firecracker exposes the vsock as a host Unix socket at `uds_path`;
        # connecting and sending "CONNECT <port>\n" triggers the Firecracker
        # vsock device model to bridge the connection to the guest's vsock port.
        self._uds_path = uds_path
        self._port = port

    async def _request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        def _sync_call() -> dict[str, Any]:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(self._uds_path)
                sock.sendall(f"CONNECT {self._port}\n".encode())
                ack = sock.recv(64)
                if b"OK" not in ack:
                    raise ConnectionError(f"vsock handshake failed: {ack!r}")
                sock.sendall((json.dumps(payload) + "\n").encode())
                chunks = []
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if chunk.endswith(b"\n"):
                        break
                return json.loads(b"".join(chunks).decode("utf-8"))

        return await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=timeout + 1)

    async def write_file(self, path: str, content: str, timeout: float = 15) -> None:
        resp = await self._request({"op": "write_file", "path": path, "content": content}, timeout)
        if not resp.get("ok"):
            raise RuntimeError(f"guest write_file failed: {resp.get('error')}")

    async def read_file(self, path: str, timeout: float = 15) -> str:
        resp = await self._request({"op": "read_file", "path": path}, timeout)
        if not resp.get("ok"):
            raise FileNotFoundError(resp.get("error", path))
        return resp["content"]

    async def exec(self, command: str, cwd: str, env: dict[str, str], timeout: float) -> dict[str, Any]:
        return await self._request(
            {"op": "exec", "command": command, "cwd": cwd, "env": env, "timeout": timeout},
            timeout + 2,
        )


class _MicroVM:
    """One live Firecracker microVM instance (jailed process + API socket)."""

    def __init__(
        self, vm_id: str, api_socket: str, vsock_uds: str, jail_dir: Path, process: asyncio.subprocess.Process
    ):
        self.vm_id = vm_id
        self.api_socket = api_socket
        self.vsock_uds = vsock_uds
        self.jail_dir = jail_dir
        self.process = process
        self.guest_cid = 3  # fixed guest CID per VM; host side is always CID 2
        self.agent = _VsockAgentClient(vsock_uds, self.guest_cid)


class FirecrackerBackend(SandboxBackend):
    name = "firecracker"

    def __init__(self) -> None:
        self._vms: dict[str, _MicroVM] = {}
        if not firecracker_available():
            logger.warning(
                "sandbox.firecracker_unavailable",
                detail="/dev/kvm missing or firecracker binary not found — this backend will fail at create()",
            )

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
        if not firecracker_available():
            raise RuntimeError(
                "Firecracker backend selected but /dev/kvm or the firecracker binary is unavailable on this host — "
                "use SANDBOX_BACKEND=gvisor instead, or run on KVM-capable client VPC hardware."
            )

        vm_id = f"fc-{uuid.uuid4().hex[:12]}"
        jail_dir = JAILER_CHROOT_BASE / vm_id
        jail_dir.mkdir(parents=True, exist_ok=True)
        api_socket = str(jail_dir / "firecracker.sock")
        vsock_uds = str(jail_dir / "vsock.sock")

        snapshot = SNAPSHOT_DIR / f"{runtime_image}.snapshot"
        if snapshot.exists():
            process = await self._restore_from_snapshot(vm_id, jail_dir, api_socket, snapshot)
        else:
            process = await self._cold_boot(vm_id, jail_dir, api_socket, runtime_image, cpu_limit, memory_mb)

        await self._configure_vsock(api_socket, vsock_uds)
        if not network_enabled:
            await self._disable_networking(api_socket)

        vm = _MicroVM(vm_id, api_socket, vsock_uds, jail_dir, process)
        self._vms[vm_id] = vm

        await self._await_guest_agent_ready(vm, timeout=min(timeout_seconds, 10))
        logger.info("sandbox.firecracker_created", vm_id=vm_id, from_snapshot=snapshot.exists())
        return vm_id

    async def _spawn_jailer(self, vm_id: str, jail_dir: Path, api_socket: str) -> asyncio.subprocess.Process:
        process = await asyncio.create_subprocess_exec(
            JAILER_BIN,
            "--id",
            vm_id,
            "--exec-file",
            FIRECRACKER_BIN,
            "--uid",
            "1000",
            "--gid",
            "1000",
            "--chroot-base-dir",
            str(JAILER_CHROOT_BASE),
            "--",
            "--api-sock",
            "/firecracker.sock",
            cwd=str(jail_dir),
        )
        await self._wait_for_socket(api_socket, timeout=5)
        return process

    async def _cold_boot(
        self, vm_id: str, jail_dir: Path, api_socket: str, runtime_image: str, cpu_limit: float, memory_mb: int
    ) -> asyncio.subprocess.Process:
        kernel = KERNEL_DIR / f"{runtime_image}.vmlinux"
        rootfs = ROOTFS_DIR / f"{runtime_image}.ext4"
        if not kernel.exists() or not rootfs.exists():
            raise FileNotFoundError(
                f"Missing prebuilt kernel/rootfs for runtime '{runtime_image}' "
                f"(expected {kernel} and {rootfs} — built by airgap/build_image_bundle.sh)"
            )

        process = await self._spawn_jailer(vm_id, jail_dir, api_socket)

        await self._api_put(
            api_socket,
            "/machine-config",
            {
                "vcpu_count": max(1, int(cpu_limit)),
                "mem_size_mib": memory_mb,
                "smt": False,
            },
        )
        await self._api_put(
            api_socket,
            "/boot-source",
            {
                "kernel_image_path": str(kernel),
                "boot_args": "console=ttyS0 reboot=k panic=1 pci=off",
            },
        )
        await self._api_put(
            api_socket,
            "/drives/rootfs",
            {
                "drive_id": "rootfs",
                "path_on_host": str(rootfs),
                "is_root_device": True,
                "is_read_only": False,
            },
        )
        await self._api_put(api_socket, "/actions", {"action_type": "InstanceStart"})
        return process

    async def _restore_from_snapshot(
        self, vm_id: str, jail_dir: Path, api_socket: str, snapshot: Path
    ) -> asyncio.subprocess.Process:
        process = await self._spawn_jailer(vm_id, jail_dir, api_socket)
        await self._api_put(
            api_socket,
            "/snapshot/load",
            {
                "snapshot_path": str(snapshot / "vmstate"),
                "mem_file_path": str(snapshot / "mem"),
                "resume_vm": True,
            },
        )
        return process

    async def _configure_vsock(self, api_socket: str, vsock_uds: str) -> None:
        await self._api_put(
            api_socket,
            "/vsock",
            {
                "guest_cid": 3,
                "uds_path": vsock_uds,
                "vsock_id": "vsock0",
            },
        )

    async def _disable_networking(self, api_socket: str) -> None:
        # No /network-interfaces PUT call is made at all — the microVM has no
        # network device attached, which is a stronger guarantee than an
        # iptables rule (nothing to bypass; the interface doesn't exist).
        return

    async def apply_egress_policy(self, handle: str, allowed_destinations: list[dict[str, Any]]) -> None:
        """
        For network-enabled sandboxes, egress is filtered on the HOST's tap
        device for this VM (nftables rule keyed on the tap interface name),
        set up when the tap device is created — not delegated to the guest.
        Network-disabled sandboxes (the default) need no policy at all.
        """
        vm = self._vms.get(handle)
        if vm is None:
            return
        tap_iface = f"tap-{vm.vm_id[:11]}"
        rules = [["nft", "flush", "chain", "inet", "keystone", tap_iface]]
        for dest in allowed_destinations:
            cidr, port = dest.get("cidr"), dest.get("port")
            if not cidr:
                continue
            cmd = ["nft", "add", "rule", "inet", "keystone", tap_iface, "ip", "daddr", cidr]
            if port:
                cmd += ["tcp", "dport", str(port)]
            cmd += ["accept"]
            rules.append(cmd)
        rules.append(["nft", "add", "rule", "inet", "keystone", tap_iface, "drop"])
        for cmd in rules:
            await asyncio.to_thread(subprocess.run, cmd, capture_output=True, check=False)

    async def write_file(self, handle: str, path: str, content: str) -> None:
        vm = self._require_vm(handle)
        await vm.agent.write_file(path, content)

    async def read_file(self, handle: str, path: str) -> str:
        vm = self._require_vm(handle)
        return await vm.agent.read_file(path)

    async def execute(
        self,
        handle: str,
        command: str,
        *,
        cwd: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
    ) -> BackendExecResult:
        vm = self._require_vm(handle)
        t0 = time.monotonic()
        try:
            resp = await vm.agent.exec(command, cwd, env_vars or {}, timeout_seconds)
            duration_ms = int((time.monotonic() - t0) * 1000)
            return BackendExecResult(
                exit_code=resp.get("exit_code", -1),
                stdout=resp.get("stdout", ""),
                stderr=resp.get("stderr", ""),
                duration_ms=duration_ms,
            )
        except TimeoutError:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return BackendExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout_seconds}s",
                duration_ms=duration_ms,
                timed_out=True,
            )

    async def destroy(self, handle: str) -> None:
        vm = self._vms.pop(handle, None)
        if vm is None:
            return
        try:
            await self._api_put(vm.api_socket, "/actions", {"action_type": "SendCtrlAltDel"})
            await asyncio.sleep(0.2)
        except Exception as exc:
            logger.debug("sandbox.firecracker_graceful_shutdown_failed", vm_id=handle, error=str(exc))
        vm.process.terminate()
        try:
            await asyncio.wait_for(vm.process.wait(), timeout=3)
        except TimeoutError:
            vm.process.kill()
            await vm.process.wait()
        shutil.rmtree(vm.jail_dir, ignore_errors=True)
        logger.info("sandbox.firecracker_destroyed", vm_id=handle)

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "kvm_available": firecracker_available(),
            "active_sandboxes": len(self._vms),
        }

    # ── Internals ────────────────────────────────────────────────

    def _require_vm(self, handle: str) -> _MicroVM:
        vm = self._vms.get(handle)
        if vm is None:
            raise ValueError(f"Sandbox {handle} not found or already destroyed")
        return vm

    async def _wait_for_socket(self, path: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await asyncio.to_thread(os.path.exists, path):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"Firecracker API socket never appeared: {path}")

    async def _await_guest_agent_ready(self, vm: _MicroVM, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                await vm.agent.exec("true", "/", {}, timeout=2)
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.1)
        raise TimeoutError(f"keystone-guest-agent did not become ready: {last_error}")

    async def _api_put(self, api_socket: str, path: str, body: dict[str, Any]) -> None:
        def _sync_put() -> None:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                sock.connect(api_socket)
                payload = json.dumps(body)
                request = (
                    f"PUT {path} HTTP/1.1\r\n"
                    f"Host: localhost\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n{payload}"
                )
                sock.sendall(request.encode())
                response = sock.recv(4096)
                status_line = response.split(b"\r\n", 1)[0].decode(errors="replace")
                if " 2" not in status_line[:12]:  # crude "2xx" check
                    raise RuntimeError(f"Firecracker API {path} failed: {status_line}")

        await asyncio.to_thread(_sync_put)
