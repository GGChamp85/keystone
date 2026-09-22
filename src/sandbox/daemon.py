# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — self-hosted sandbox execution daemon ("keystoned").

An internal-only HTTP service that owns the privileged access sandbox
execution needs (the Docker socket for the gVisor backend, /dev/kvm for the
Firecracker backend) so the main API tier never has to run with that
privilege itself — the app calls this daemon over the network instead.

Backend selection is host-capability-driven: Firecracker is used when this
host actually has KVM available (bare-metal/nested-virt client VPC nodes);
otherwise this falls back to the gVisor backend automatically, regardless
of the configured preference, so a misconfigured host fails safe into a
still-sandboxed backend rather than refusing to start.

Deployed as its own container (docker/sandbox-daemon.Dockerfile) — never
colocated with the main API process.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from src.config import get_settings
from src.observability import sandbox_executions_total
from src.sandbox.backends.base import SandboxBackend
from src.sandbox.backends.firecracker import FirecrackerBackend, firecracker_available
from src.sandbox.backends.gvisor import GVisorBackend
from src.sandbox.security import build_egress_policy

logger = structlog.get_logger()

RUNTIME_IMAGES: dict[str, str] = {
    "python": "keystone-sandbox-python:latest",
    "javascript": "keystone-sandbox-node:latest",
    "typescript": "keystone-sandbox-node:latest",
    "go": "keystone-sandbox-go:latest",
    "rust": "keystone-sandbox-rust:latest",
    "ruby": "keystone-sandbox-ruby:latest",
    "shell": "keystone-sandbox-base:latest",
    "base": "keystone-sandbox-base:latest",
}


def _select_backend() -> SandboxBackend:
    settings = get_settings()
    preferred = settings.sandbox_backend
    if preferred == "firecracker" and firecracker_available():
        logger.info("keystoned.backend_selected", backend="firecracker")
        return FirecrackerBackend()
    if preferred == "firecracker" and not firecracker_available():
        logger.warning("keystoned.firecracker_unavailable_falling_back", detail="no /dev/kvm on this host")
    logger.info("keystoned.backend_selected", backend="gvisor")
    return GVisorBackend()


class _TenantConcurrency:
    """Per-tenant concurrent-sandbox cap, enforced daemon-side so a compromised
    app-tier pod can't bypass limits by talking to the daemon directly."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, tenant_id: str, max_concurrent: int) -> None:
        """`max_concurrent` 0 = no cap (the default): the Docker host's real capacity is the limit."""
        async with self._lock:
            current = self._counts.get(tenant_id, 0)
            if max_concurrent > 0 and current >= max_concurrent:
                raise HTTPException(
                    status_code=429,
                    detail=f"Tenant sandbox concurrency limit reached ({max_concurrent})",
                )
            self._counts[tenant_id] = current + 1

    async def release(self, tenant_id: str) -> None:
        async with self._lock:
            self._counts[tenant_id] = max(0, self._counts.get(tenant_id, 1) - 1)


backend = _select_backend()
concurrency = _TenantConcurrency()
_handle_owner: dict[str, str] = {}  # handle -> tenant_id, for release() bookkeeping

app = FastAPI(title="keystoned", description="Keystone self-hosted sandbox execution daemon")
app.mount("/metrics", make_asgi_app())


class CreateRequest(BaseModel):
    tenant_id: str
    template: str = "python"
    timeout_seconds: int | None = None
    env_vars: dict[str, str] = {}
    network_enabled: bool = False
    cpu_limit: float = 1.0
    memory_mb: int = 1024
    max_concurrent_per_tenant: int = 0  # 0 = no cap


class CreateResponse(BaseModel):
    handle: str
    backend: str


class WriteFileRequest(BaseModel):
    path: str
    content: str


class ExecRequest(BaseModel):
    command: str
    cwd: str = "/workspace"
    timeout_seconds: int = 120
    env_vars: dict[str, str] = {}


class ExecResponse(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


@app.post("/sandboxes", response_model=CreateResponse)
async def create_sandbox(req: CreateRequest):
    settings = get_settings()
    await concurrency.acquire(req.tenant_id, req.max_concurrent_per_tenant)
    try:
        runtime_image = RUNTIME_IMAGES.get(req.template, RUNTIME_IMAGES["base"])
        handle = await backend.create(
            runtime_image=runtime_image,
            cpu_limit=req.cpu_limit,
            memory_mb=req.memory_mb,
            network_enabled=req.network_enabled,
            env_vars=req.env_vars,
            timeout_seconds=req.timeout_seconds or settings.sandbox_timeout_seconds,
        )
    except Exception:
        await concurrency.release(req.tenant_id)
        raise
    _handle_owner[handle] = req.tenant_id

    policy = build_egress_policy(
        settings.git_allowed_hosts,
        settings.pip_index_url,
        settings.npm_registry_url,
        settings.go_proxy_url,
    )
    allowed = [
        {"cidr": r.destination, "port": r.port}
        for r in policy.rules
        if r.action.value == "allow" and r.destination not in ("*",)
    ]
    try:
        await backend.apply_egress_policy(handle, allowed)
    except Exception as exc:
        logger.warning("keystoned.egress_policy_failed", handle=handle, error=str(exc))

    return CreateResponse(handle=handle, backend=backend.name)


@app.post("/sandboxes/{handle}/files")
async def write_file(handle: str, req: WriteFileRequest):
    await backend.write_file(handle, req.path, req.content)
    return {"status": "ok"}


@app.get("/sandboxes/{handle}/files")
async def read_file(handle: str, path: str):
    try:
        content = await backend.read_file(handle, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"File not found: {path}") from exc
    return {"content": content}


@app.post("/sandboxes/{handle}/exec", response_model=ExecResponse)
async def execute(handle: str, req: ExecRequest):
    result = await backend.execute(
        handle,
        req.command,
        cwd=req.cwd,
        timeout_seconds=req.timeout_seconds,
        env_vars=req.env_vars,
    )
    status_label = "timeout" if result.timed_out else ("success" if result.exit_code == 0 else "error")
    sandbox_executions_total.labels(backend.name, status_label).inc()
    return ExecResponse(**result.__dict__)


@app.delete("/sandboxes/{handle}")
async def destroy_sandbox(handle: str):
    await backend.destroy(handle)
    tenant_id = _handle_owner.pop(handle, None)
    if tenant_id:
        await concurrency.release(tenant_id)
    return {"status": "destroyed"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return await backend.health()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.sandbox.daemon:app", host="0.0.0.0", port=9000, log_level="info")
