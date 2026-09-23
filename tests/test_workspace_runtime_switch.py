# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""A task's sandbox can switch runtime image once the repository's ecosystem is known
(src/orchestrator/workspace.py `switch_runtime`): against the real sandbox daemon, a Python sandbox has
no `go`, switching to the Go runtime gives one (and destroys the old sandbox), and switching back to
Python works too. Requires the daemon (SANDBOX_DAEMON_URL) and the node/go runtime images
(`make sandbox-images`)."""

from __future__ import annotations

import os
import subprocess
import uuid

import pytest

from src.orchestrator.workspace import Workspace
from src.sandbox.manager import SandboxManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("SANDBOX_DAEMON_URL"), reason="Needs the sandbox daemon"),
]


def _image_built(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True).returncode == 0  # noqa: S603, S607


@pytest.mark.skipif(
    not _image_built("keystone-sandbox-go:latest"), reason="Needs keystone-sandbox-go (make sandbox-images)"
)
async def test_switching_to_the_go_runtime_replaces_the_sandbox_and_brings_the_toolchain():
    manager = SandboxManager()
    ws = Workspace(manager, task_id=f"runtime-switch-{uuid.uuid4().hex[:8]}")
    try:
        python_handle = await ws.ensure_sandbox(network_enabled=False)
        assert ws.language == "python"
        assert (await ws.run("python3 --version", cwd="/workspace", check=False))["exit_code"] == 0
        assert (await ws.run("go version", cwd="/workspace", check=False))["exit_code"] != 0

        go_handle = await ws.switch_runtime("go")
        assert go_handle != python_handle and ws.language == "go"
        out = await ws.run("go version", cwd="/workspace", check=True)
        assert "go1." in out["stdout"]
        assert (await ws.run("ctags --version", cwd="/workspace", check=False))[
            "exit_code"
        ] == 0  # repo map works here too

        # the previous sandbox is gone from the daemon; only the current one is tracked for this task
        health = await manager.health()
        assert health.get("docker_ok") is True
        assert manager._handles_by_task[ws._task_id] == go_handle
    finally:
        await ws.close()
