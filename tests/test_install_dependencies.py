# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
`install_dependencies` (src/orchestrator/nodes/_shared.py) against a real
sandbox: the repo's own install command actually runs once, its outcome is
recorded on the state, and a failure is attributed — not raised, not hidden.
Requires SANDBOX_DAEMON_URL; skipped otherwise.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.orchestrator.nodes._shared import install_dependencies
from src.orchestrator.state import AgentState
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import SandboxManager

pytestmark = pytest.mark.integration

requires_sandbox = pytest.mark.skipif(
    not os.environ.get("SANDBOX_DAEMON_URL"),
    reason="Needs SANDBOX_DAEMON_URL pointed at a running keystoned",
)


@pytest.fixture
async def ws():
    manager = SandboxManager()
    workspace = Workspace(manager, task_id=f"deps-test-{uuid.uuid4().hex[:8]}")
    await workspace.ensure_sandbox(network_enabled=False)
    await workspace.run("mkdir -p /workspace/repo", cwd="/")
    try:
        yield workspace
    finally:
        await workspace.close()


@requires_sandbox
async def test_install_runs_the_repos_real_install_command_once(ws):
    # httpx is already in the sandbox image, so `pip install -r requirements.txt` succeeds
    # without any network — proving the mechanism (detection -> real command -> recorded).
    await ws.write_file("requirements.txt", "httpx\n")
    state = AgentState(repository_url="https://example.invalid/r.git")
    await install_dependencies(state, ws)
    assert state.deps_installed is True
    assert state.deps_install_error == ""

    # Second call is a no-op (the flag, not a re-run).
    await ws.write_file("requirements.txt", "this-package-does-not-exist-anywhere-xyz\n")
    await install_dependencies(state, ws)
    assert state.deps_install_error == ""


@requires_sandbox
async def test_install_failure_is_recorded_and_attributed_not_raised(ws):
    await ws.write_file("requirements.txt", "this-package-does-not-exist-anywhere-xyz\n")
    state = AgentState(repository_url="https://example.invalid/r.git")
    await install_dependencies(state, ws)
    assert state.deps_installed is True
    assert state.deps_install_error.startswith("`pip install -r requirements.txt` exited ")
    assert "this-package-does-not-exist-anywhere-xyz" in state.deps_install_error


@requires_sandbox
async def test_repo_without_an_install_command_is_marked_done_with_no_error(ws):
    await ws.write_file("README.md", "nothing to install\n")
    state = AgentState(repository_url="https://example.invalid/r.git")
    await install_dependencies(state, ws)
    assert state.deps_installed is True
    assert state.deps_install_error == ""
