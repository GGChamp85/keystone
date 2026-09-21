# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/orchestrator/tools/impl.py — every tool executed against a real sandbox
(`keystoned`, gVisor backend) and a real git working tree inside it, not a mock of `Workspace`. Requires
SANDBOX_DAEMON_URL pointed at a running daemon with the keystone-sandbox-python image built (`make
sandbox-images`); skipped otherwise, matching tests/test_git_workflow_integration.py's pattern.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.orchestrator.tools.impl import apply_patch, grep, list_dir, read_file, run_command
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import SandboxManager

pytestmark = pytest.mark.integration

requires_sandbox = pytest.mark.skipif(
    not os.environ.get("SANDBOX_DAEMON_URL"),
    reason="Needs SANDBOX_DAEMON_URL pointed at a running keystoned — see module docstring",
)


@pytest.fixture
async def ws():
    manager = SandboxManager()
    workspace = Workspace(manager, task_id=f"tool-impl-test-{uuid.uuid4().hex[:8]}")
    await workspace.ensure_sandbox(network_enabled=False)
    await workspace.run("mkdir -p /workspace/repo", cwd="/")
    await workspace.write_file("app.py", "def greet(name):\n    return f'hello {name}'\n")
    await workspace.run(
        "git init -q && git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m init"
    )
    try:
        yield workspace
    finally:
        await workspace.close()


@requires_sandbox
async def test_read_file_returns_real_content(ws):
    result = await read_file(ws, "app.py")
    assert result.ok
    assert "def greet" in result.output


@requires_sandbox
async def test_read_file_missing_path_is_a_clean_failure_not_an_exception(ws):
    result = await read_file(ws, "does_not_exist.py")
    assert not result.ok
    assert "does not exist" in result.error


@requires_sandbox
async def test_list_dir_shows_the_file_we_created(ws):
    result = await list_dir(ws, ".")
    assert result.ok
    assert "app.py" in result.output


@requires_sandbox
async def test_grep_finds_a_real_match(ws):
    result = await grep(ws, pattern="def greet")
    assert result.ok
    assert "app.py" in result.output


@requires_sandbox
async def test_grep_no_match_is_ok_not_an_error(ws):
    result = await grep(ws, pattern="this_pattern_does_not_exist_anywhere")
    assert result.ok
    assert result.output == "(no matches)"


@requires_sandbox
async def test_apply_patch_search_replace_edits_the_real_file(ws):
    result = await apply_patch(ws, path="app.py", search="hello {name}", replace="hi {name}")
    assert result.ok
    content = (await read_file(ws, "app.py")).output
    assert "hi {name}" in content
    assert "hello {name}" not in content


@requires_sandbox
async def test_apply_patch_search_not_found_gives_actionable_error(ws):
    result = await apply_patch(ws, path="app.py", search="this text is not in the file", replace="x")
    assert not result.ok
    assert "not found" in result.error


@requires_sandbox
async def test_apply_patch_ambiguous_search_gives_actionable_error(ws):
    create_result = await apply_patch(ws, path="ambiguous.py", create=True, replace="x = 1\nx = 1\n")
    assert create_result.ok, create_result.error
    result = await apply_patch(ws, path="ambiguous.py", search="x = 1", replace="x = 2")
    assert not result.ok
    assert "2 places" in result.error


@requires_sandbox
async def test_apply_patch_create_new_file(ws):
    result = await apply_patch(ws, path="new_module.py", create=True, replace="VALUE = 42\n")
    assert result.ok
    content = (await read_file(ws, "new_module.py")).output
    assert "VALUE = 42" in content


@requires_sandbox
async def test_apply_patch_create_refuses_to_clobber_existing_file(ws):
    result = await apply_patch(ws, path="app.py", create=True, replace="overwritten\n")
    assert not result.ok
    assert "already exists" in result.error
    content = (await read_file(ws, "app.py")).output
    assert "def greet" in content  # untouched


@requires_sandbox
async def test_apply_patch_delete_removes_the_real_file(ws):
    await apply_patch(ws, path="scratch.py", create=True, replace="x = 1\n")
    result = await apply_patch(ws, path="scratch.py", delete=True)
    assert result.ok
    follow_up = await read_file(ws, "scratch.py")
    assert not follow_up.ok


@requires_sandbox
async def test_apply_patch_unified_diff_applies_a_real_git_diff(ws):
    # Generate a real diff the way the model would receive one to send back:
    # modify the file directly, capture `git diff`, revert, then apply it
    # through the tool to prove the tool's own git-apply path round-trips.
    await ws.write_file("app.py", "def greet(name):\n    return f'HELLO {name}'\n")
    diff_result = await ws.run("git diff")
    real_diff = diff_result["stdout"]
    assert real_diff.strip(), "expected a non-empty real git diff"
    await ws.run("git checkout -- app.py")
    assert "HELLO" not in (await read_file(ws, "app.py")).output

    result = await apply_patch(ws, unified_diff=real_diff)
    assert result.ok
    content = (await read_file(ws, "app.py")).output
    assert "HELLO" in content


@requires_sandbox
async def test_apply_patch_unified_diff_conflict_is_a_clean_failure(ws):
    garbage_diff = (
        "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n-this line does not exist in the real file\n+replacement\n"
    )
    result = await apply_patch(ws, unified_diff=garbage_diff)
    assert not result.ok
    assert result.error


@requires_sandbox
async def test_run_command_success_shows_exit_code_and_stdout(ws):
    result = await run_command(ws, "echo hello-from-sandbox")
    assert result.ok
    assert "hello-from-sandbox" in result.output
    assert "exit code 0" in result.output


@requires_sandbox
async def test_run_command_nonzero_exit_is_ok_with_exit_code_visible(ws):
    result = await run_command(ws, "exit 7")
    assert result.ok  # a failing command is real information, not a tool failure
    assert "exit code 7" in result.output


@requires_sandbox
async def test_run_command_real_test_suite_passes(ws):
    await ws.write_file(
        "test_app.py",
        "from app import greet\n\ndef test_greet():\n    assert greet('world') == \"hello world\"\n",
    )
    result = await run_command(ws, "python3 -m pytest test_app.py -q")
    assert result.ok
    assert "exit code 0" in result.output
    assert "1 passed" in result.output
