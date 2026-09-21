# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for the gVisor sandbox backend — no mocks: this
actually creates and destroys Docker containers.

Requires: a working Docker daemon, and the `keystone-sandbox-python:latest`
image present (tag any Python 3.x image as that name for local testing;
see docs/architecture/SANDBOX_ARCHITECTURE.md for the real build).
"""

from __future__ import annotations

import shutil

import pytest

import docker
from src.sandbox.backends.gvisor import GVisorBackend

pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


def _test_image_available() -> bool:
    if not _docker_available():
        return False
    try:
        docker.from_env().images.get("keystone-sandbox-python:latest")
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
requires_test_image = pytest.mark.skipif(
    not _test_image_available(),
    reason="Tag a Python image as keystone-sandbox-python:latest first, e.g.: "
    "docker tag python:3.12-slim keystone-sandbox-python:latest",
)


@pytest.fixture
async def backend():
    b = GVisorBackend()
    yield b


@requires_docker
@requires_test_image
async def test_create_write_execute_destroy_lifecycle(backend: GVisorBackend):
    handle = await backend.create(
        runtime_image="keystone-sandbox-python:latest",
        cpu_limit=1.0,
        memory_mb=512,
        network_enabled=False,
        env_vars={},
        timeout_seconds=30,
    )
    try:
        await backend.write_file(handle, "script.py", "print(6 * 7)")
        result = await backend.execute(handle, "python script.py", cwd="/workspace", timeout_seconds=15)
        assert result.exit_code == 0
        assert result.stdout.strip() == "42"
    finally:
        await backend.destroy(handle)

    client = docker.from_env()
    with pytest.raises(docker.errors.NotFound):
        client.containers.get(handle)


@requires_docker
@requires_test_image
async def test_network_disabled_sandbox_cannot_reach_internet(backend: GVisorBackend):
    handle = await backend.create(
        runtime_image="keystone-sandbox-python:latest",
        cpu_limit=1.0,
        memory_mb=512,
        network_enabled=False,
        env_vars={},
        timeout_seconds=30,
    )
    try:
        result = await backend.execute(
            handle,
            "python -c \"import socket; socket.create_connection(('8.8.8.8', 53), timeout=3)\"",
            cwd="/workspace",
            timeout_seconds=15,
        )
        assert result.exit_code != 0  # connection must fail — no network device attached
    finally:
        await backend.destroy(handle)


@requires_docker
@requires_test_image
async def test_read_file_roundtrip(backend: GVisorBackend):
    handle = await backend.create(
        runtime_image="keystone-sandbox-python:latest",
        cpu_limit=1.0,
        memory_mb=512,
        network_enabled=False,
        env_vars={},
        timeout_seconds=30,
    )
    try:
        await backend.write_file(handle, "data.txt", "hello sandbox\nline two")
        content = await backend.read_file(handle, "data.txt")
        assert content == "hello sandbox\nline two"
    finally:
        await backend.destroy(handle)


@requires_docker
@requires_test_image
async def test_command_timeout_is_enforced(backend: GVisorBackend):
    handle = await backend.create(
        runtime_image="keystone-sandbox-python:latest",
        cpu_limit=1.0,
        memory_mb=512,
        network_enabled=False,
        env_vars={},
        timeout_seconds=30,
    )
    try:
        result = await backend.execute(
            handle,
            "python -c 'import time; time.sleep(30)'",
            cwd="/workspace",
            timeout_seconds=2,
        )
        assert result.timed_out
        assert result.exit_code == -1
    finally:
        await backend.destroy(handle)


@requires_docker
async def test_health_reports_docker_connectivity(backend: GVisorBackend):
    health = await backend.health()
    assert health["backend"] == "gvisor"
    assert health["docker_ok"] is True
