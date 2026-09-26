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

from src.orchestrator.tools.impl import (
    apply_patch,
    get_repo_map,
    grep,
    list_dir,
    lsp_diagnostics,
    read_file,
    run_command,
    run_tests,
)
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
async def test_read_file_line_range_is_numbered_and_bounded(ws):
    await ws.write_file("multi.py", "\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    result = await read_file(ws, "multi.py", start_line=3, end_line=5)
    assert result.ok
    assert result.output == "multi.py lines 3-5 of 10\n3: line3\n4: line4\n5: line5"
    tail = await read_file(ws, "multi.py", start_line=9)
    assert tail.ok and tail.output.endswith("10: line10")
    past_end = await read_file(ws, "multi.py", start_line=50)
    assert not past_end.ok and "only 10 lines" in past_end.error


@requires_sandbox
async def test_get_repo_map_returns_real_symbols_from_ctags(ws):
    result = await get_repo_map(ws, max_tokens=1000)
    assert result.ok
    assert "app.py:" in result.output
    assert "1: function greet(name)" in result.output


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
async def test_apply_patch_tolerates_a_whitespace_only_mismatch_and_says_so(ws):
    # The model's search block is re-indented relative to the real file.
    result = await apply_patch(
        ws,
        path="app.py",
        search="def greet(name):\n  return f'hello {name}'",
        replace="def greet(name):\n    return f'hey {name}'\n",
    )
    assert result.ok, result.error
    assert "fuzzy match at lines 1-2" in result.output
    assert "whitespace" in result.output
    content = (await read_file(ws, "app.py")).output
    assert content == "def greet(name):\n    return f'hey {name}'\n"


@requires_sandbox
async def test_apply_patch_refuses_a_fuzzy_match_that_could_mean_two_places(ws):
    await apply_patch(ws, path="twice.py", create=True, replace="a = 1\nb = 2\na = 1\n")
    result = await apply_patch(ws, path="twice.py", search="a  =  1", replace="a = 9")
    assert not result.ok
    assert "2 places" in result.error and "lines 1-1" in result.error and "lines 3-3" in result.error
    assert (await read_file(ws, "twice.py")).output == "a = 1\nb = 2\na = 1\n"  # untouched


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
async def test_run_tests_related_then_all_against_the_real_repo_tooling(ws):
    await ws.write_file("pytest.ini", "[pytest]\n")
    await ws.write_file(
        "test_app.py", "from app import greet\n\n\ndef test_greet():\n    assert greet('w') == 'hello w'\n"
    )
    await ws.write_file("test_other.py", "def test_other():\n    assert 1 == 1\n")
    await ws.run("git add -A")

    related = await run_tests(ws, scope="related", touched_paths=["app.py"])
    assert related.ok
    assert related.output.startswith("$ pytest -q test_app.py\n(exit code 0) related tests passed")
    assert "1 passed" in related.output

    everything = await run_tests(ws, scope="all", touched_paths=["app.py"])
    assert everything.ok
    assert everything.output.startswith("$ pytest\n(exit code 0) full suite passed")
    assert "2 passed" in everything.output

    nothing_related = await run_tests(ws, scope="related", touched_paths=["unrelated.py"])
    assert "no tests related" in nothing_related.output and "2 passed" in nothing_related.output

    await ws.write_file("app.py", "def greet(name):\n    return f'bye {name}'\n")
    failing = await run_tests(ws, scope="related", touched_paths=["app.py"])
    assert failing.ok  # a failing test is information, not a tool error
    assert "(exit code 1) related tests FAILED" in failing.output
    assert "assert" in failing.output


@requires_sandbox
async def test_run_tests_without_a_detectable_test_command_is_a_clean_error(ws):
    result = await run_tests(ws, scope="all")
    assert not result.ok
    assert "No test command" in result.error


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


@requires_sandbox
async def test_lsp_diagnostics_finds_a_real_undefined_name(ws):
    """Real proof this is a real language server, not a heuristic: pylsp (python-lsp-server,
    baked into the sandbox image — docker/sandbox-runtimes/python.Dockerfile) actually running
    inside the sandbox, speaking the real LSP stdio protocol, catching a real undefined-name
    error pyflakes would flag."""
    await ws.write_file("broken.py", "def greet(name):\n    return f'hello {undefined_variable}'\n")
    result = await lsp_diagnostics(ws, "broken.py")
    assert result.ok
    assert "broken.py:2:" in result.output
    assert "undefined_variable" in result.output.lower() or "undefined name" in result.output.lower()


@requires_sandbox
async def test_lsp_diagnostics_reports_clean_for_a_real_valid_file(ws):
    result = await lsp_diagnostics(ws, "app.py")
    assert result.ok
    assert "No diagnostics" in result.output


@requires_sandbox
async def test_lsp_diagnostics_refuses_an_unsupported_extension_with_a_clear_reason(ws):
    await ws.write_file("script.sh", "#!/bin/bash\necho hi\n")
    result = await lsp_diagnostics(ws, "script.sh")
    assert not result.ok
    assert "No language server configured" in result.error
    assert ".sh" in result.error
