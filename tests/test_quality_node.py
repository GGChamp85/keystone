# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for nodes/quality.py — runs against a real sandbox with the real ruff/mypy/bandit
binaries baked into the sandbox image (docker/sandbox-runtimes/python.Dockerfile), on a real git working tree.
Not a mock of any of these tools: the findings asserted on here are what ruff/bandit/mypy actually produce for
deliberately-flawed real files.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import pytest

from src.orchestrator.nodes.quality import quality_node
from src.orchestrator.state import AgentPhase, AgentState
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import SandboxManager

pytestmark = pytest.mark.integration

requires_sandbox = pytest.mark.skipif(
    not os.environ.get("SANDBOX_DAEMON_URL"),
    reason="Needs SANDBOX_DAEMON_URL pointed at a running keystoned — see tests/test_tool_impl.py",
)


@pytest.fixture
async def ws():
    manager = SandboxManager()
    workspace = Workspace(manager, task_id=f"quality-node-test-{uuid.uuid4().hex[:8]}")
    await workspace.ensure_sandbox(network_enabled=False)
    await workspace.run("mkdir -p /workspace/repo", cwd="/")
    await workspace.write_file(
        "pyproject.toml",
        "[tool.ruff]\nline-length = 100\n[tool.mypy]\npython_version = '3.12'\n[tool.bandit]\n",
    )
    await workspace.run(
        "git init -q && git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m init"
    )
    try:
        yield workspace
    finally:
        await workspace.close()


def _make_state() -> AgentState:
    state = AgentState()
    state.repository_url = "https://example.invalid/scripted/repo.git"
    state.repo_cloned = True
    state.enable_reasoning_review = False
    state.enable_sandbox_testing = False
    return state


@requires_sandbox
async def test_clean_file_passes_with_no_blocking_findings(ws):
    await ws.write_file("app.py", "def greet(name: str) -> str:\n    return f'hello {name}'\n")
    await ws.run(
        "git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m edit"
    )
    state = _make_state()

    with patch("src.orchestrator.nodes._shared.Workspace", return_value=ws):
        result = await quality_node(state)

    blocking = [f for f in result.quality_findings if f.tool in ("bandit", "mypy") and f.severity == "error"]
    assert blocking == []
    assert result.phase == AgentPhase.COMPLETE
    assert result.consecutive_quality_failures == 0


@requires_sandbox
async def test_bandit_finds_a_real_security_issue_and_blocks(ws):
    # A genuinely real bandit finding: eval() on untrusted-looking input (B307).
    await ws.write_file("app.py", "def run(user_input):\n    return eval(user_input)\n")
    await ws.run(
        "git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m edit"
    )
    state = _make_state()

    with patch("src.orchestrator.nodes._shared.Workspace", return_value=ws):
        result = await quality_node(state)

    bandit_findings = [f for f in result.quality_findings if f.tool == "bandit"]
    assert bandit_findings, "expected bandit to report a real finding for eval()"
    assert any(f.severity == "error" for f in bandit_findings)
    assert result.phase == AgentPhase.FIXING
    assert result.consecutive_quality_failures == 1
    assert result.error_message and "bandit" in result.error_message


@requires_sandbox
async def test_mypy_finds_a_real_type_error_and_blocks(ws):
    await ws.write_file("app.py", "def add(a: int, b: int) -> int:\n    return a + b\n\nresult: str = add(1, 2)\n")
    await ws.run(
        "git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m edit"
    )
    state = _make_state()

    with patch("src.orchestrator.nodes._shared.Workspace", return_value=ws):
        result = await quality_node(state)

    mypy_findings = [f for f in result.quality_findings if f.tool == "mypy"]
    assert mypy_findings, "expected mypy to report a real type error"
    assert result.phase == AgentPhase.FIXING


@requires_sandbox
async def test_ruff_findings_alone_do_not_block():
    """ruff findings are informational only in this cut (see quality.py's module docstring) — a file with
    only a style issue (unused import) must NOT block, unlike a bandit/mypy error."""
    manager = SandboxManager()
    workspace = Workspace(manager, task_id=f"quality-node-ruffonly-{uuid.uuid4().hex[:8]}")
    await workspace.ensure_sandbox(network_enabled=False)
    await workspace.run("mkdir -p /workspace/repo", cwd="/")
    await workspace.write_file("pyproject.toml", "[tool.ruff]\nline-length = 100\n")
    await workspace.write_file("app.py", "import os\n\ndef greet(name):\n    return f'hello {name}'\n")
    await workspace.run(
        "git init -q && git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m init"
    )
    state = _make_state()

    try:
        with patch("src.orchestrator.nodes._shared.Workspace", return_value=workspace):
            result = await quality_node(state)

        ruff_findings = [f for f in result.quality_findings if f.tool == "ruff"]
        assert ruff_findings, "expected ruff to flag the unused import"
        assert result.phase == AgentPhase.COMPLETE, "ruff-only findings must not block"
    finally:
        await workspace.close()


@requires_sandbox
async def test_standalone_mode_skips_quality_gates_entirely(ws):
    state = _make_state()
    state.repository_url = None
    state.repo_cloned = False

    result = await quality_node(state)

    assert result.quality_findings == []
    assert result.phase == AgentPhase.COMPLETE


async def test_disabled_quality_gates_skips_straight_through():
    state = _make_state()
    state.enable_quality_gates = False

    result = await quality_node(state)

    assert result.quality_findings == []
    assert result.phase == AgentPhase.COMPLETE
