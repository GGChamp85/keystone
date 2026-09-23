# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Testing node.

Two modes, chosen by whether the task has a `repository_url`:

- **Repo mode** (real git workflow, src/orchestrator/workspace.py): reuses
  the task's already-cloned working tree
  (`nodes/_shared.get_or_clone_workspace` — the same helper
  `nodes/coding.py`'s tool loop uses, so cloning happens exactly once no
  matter which node runs first). By the time this node runs, `coding_node`
  has already written any changes directly into that working tree via
  `apply_patch`, so this node's job is purely to detect the repo's own real
  tooling (src/orchestrator/repo_profile.py) and run its real test command —
  replacing the old heuristic guessing (py_compile / "pytest if a path
  contains 'test'" / bare import) entirely. The sandbox is NOT destroyed
  here — it needs to survive into the next fixing/testing iteration and,
  on success, into the commit/push/PR step engine.py runs after the graph
  completes.
- **Standalone mode** (no repository_url — e.g. a benchmark task or a
  context-files-only request with nothing to clone): unchanged from
  before — a fresh ephemeral sandbox per call, generated files written
  directly, heuristic per-language smoke commands, destroyed at the end.
"""

from __future__ import annotations

import time
from uuid import UUID

import structlog

from src.config import get_settings
from src.orchestrator.events import publish_task_step
from src.orchestrator.nodes._shared import detect_profile, get_or_clone_workspace
from src.orchestrator.state import (
    AgentPhase,
    AgentState,
    IterationRecord,
    TestResult,
)
from src.orchestrator.test_scope import related_test_command
from src.orchestrator.workspace import Workspace, WorkspaceCommandError
from src.sandbox.manager import SandboxManager

logger = structlog.get_logger(__name__)


async def testing_node(state: AgentState) -> AgentState:
    """
    Deploy code changes into a sandboxed environment, run tests,
    and capture results.
    """
    if not state.enable_sandbox_testing:
        state.tests_passed = True
        state.phase = AgentPhase.COMPLETE
        state.result_summary = "Code generation complete (sandbox testing disabled)."
        return state

    if state.repository_url:
        return await _test_in_repo_workspace(state)

    if not state.file_changes:
        state.tests_passed = False
        state.phase = AgentPhase.FAILED
        state.error_message = "No file changes to test."
        return state

    return await _test_standalone(state)


async def _test_in_repo_workspace(state: AgentState) -> AgentState:
    t0 = time.monotonic()

    try:
        ws = await get_or_clone_workspace(state)
        state.sandbox_id = await ws.ensure_sandbox(network_enabled=True)

        diff_stat = await ws.diff_stat()
        if not diff_stat.strip():
            state.tests_passed = False
            state.phase = AgentPhase.FAILED
            state.error_message = "No file changes to test (coding produced an empty diff)."
            return state

        profile = await detect_profile(ws)

        state.test_results = []
        steps: list[dict] = []
        if not profile.test_cmd:
            state.test_results.append(
                TestResult(
                    test_name="repo_profile",
                    passed=False,
                    error=f"Could not detect a test command for ecosystem={profile.ecosystem!r}",
                )
            )
            all_passed = False
        else:
            # Related tests first (fast, targeted feedback), the full suite only
            # once they pass — a failure in the scoped run goes straight back to
            # fixing with the relevant output instead of a wall of unrelated tests.
            scoped = related_test_command(profile, state.files_touched, await ws.list_tracked_files())
            all_passed = True
            if scoped:
                all_passed = await _run_command_as_test(
                    ws,
                    f"related: {scoped}",
                    scoped,
                    state.test_results,
                    no_tests_collected_ok=True,
                    task_id=state.task_id,
                    steps=steps,
                )
            if all_passed:
                all_passed = await _run_command_as_test(
                    ws, profile.test_cmd, profile.test_cmd, state.test_results, task_id=state.task_id, steps=steps
                )

        state.tests_passed = all_passed
        state.sandbox_output = "\n".join(
            f"[{'PASS' if tr.passed else 'FAIL'}] {tr.test_name}: {tr.output}" for tr in state.test_results
        )

        files_modified = len(state.files_touched) or len(state.file_changes)
        if all_passed:
            state.consecutive_test_failures = 0
            state.phase = AgentPhase.COMPLETE
            state.result_summary = (
                f"Tests passed ({profile.test_cmd!r}, {profile.ecosystem}). Modified {files_modified} file(s)."
            )
        else:
            state.consecutive_test_failures += 1
            state.phase = AgentPhase.FIXING

        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="testing",
                model_role="sandbox",
                test_results=[
                    {"test_name": tr.test_name, "passed": tr.passed, "error": tr.error} for tr in state.test_results
                ],
                steps=steps,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )

        logger.info(
            "testing.repo_complete",
            task_id=str(state.task_id),
            passed=all_passed,
            ecosystem=profile.ecosystem,
        )

    except Exception as exc:
        logger.error("testing.repo_failed", task_id=str(state.task_id), error=str(exc))
        state.tests_passed = False
        state.consecutive_test_failures += 1
        state.phase = AgentPhase.FIXING
        state.error_message = f"Sandbox testing error: {exc}"

    # Deliberately no `finally: ws.close()` here — the sandbox must survive
    # into the next fixing/testing iteration and, on success, into
    # engine.py's post-graph commit/push/PR step. It's destroyed exactly
    # once, by the engine, after the graph fully completes.
    return state


async def _run_command_as_test(
    ws: Workspace,
    name: str,
    command: str,
    results: list[TestResult],
    *,
    no_tests_collected_ok: bool = False,
    task_id: UUID | str | None = None,
    steps: list[dict] | None = None,
) -> bool:
    try:
        result = await ws.run(command, timeout=get_settings().agent_test_timeout_seconds, check=False)
        # pytest exits 5 when it collected nothing — for a scoped run that means "no related
        # tests after all", not a failure; the full suite that follows is the real verdict.
        passed = result["exit_code"] == 0 or (no_tests_collected_ok and result["exit_code"] == 5)
        if task_id is not None:
            event = await publish_task_step(
                task_id,
                "test_output",
                "testing",
                {
                    "name": name,
                    "command": command,
                    "exit_code": result["exit_code"],
                    "passed": passed,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "duration_ms": result.get("duration_ms", 0),
                },
            )
            if steps is not None:
                steps.append(event)
        results.append(
            TestResult(
                test_name=name,
                passed=passed,
                output=result.get("stdout", ""),
                error=result.get("stderr", ""),
                duration_ms=result.get("duration_ms", 0),
            )
        )
        return passed
    except WorkspaceCommandError as exc:
        results.append(TestResult(test_name=name, passed=False, error=str(exc)))
        return False


async def _test_standalone(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    sandbox = SandboxManager()

    try:
        # Create sandbox and upload files
        sandbox_id = await sandbox.create()
        state.sandbox_id = sandbox_id

        logger.info(
            "testing.sandbox_created",
            task_id=str(state.task_id),
            sandbox_id=sandbox_id,
        )

        # Upload all generated files
        for fc in state.file_changes:
            if fc.action != "delete":
                await sandbox.write_file(sandbox_id, fc.path, fc.new_content)

        # Upload context files
        for fname, content in state.context_files.items():
            await sandbox.write_file(sandbox_id, fname, content)

        # Determine test strategy based on file types
        test_commands = _build_test_commands(state)

        state.test_results = []
        all_passed = True

        for cmd_name, cmd in test_commands:
            try:
                result = await sandbox.execute(sandbox_id, cmd, timeout=120)
                passed = result["exit_code"] == 0

                state.test_results.append(
                    TestResult(
                        test_name=cmd_name,
                        passed=passed,
                        output=result.get("stdout", ""),
                        error=result.get("stderr", ""),
                        duration_ms=result.get("duration_ms", 0),
                    )
                )

                if not passed:
                    all_passed = False
                    logger.warning(
                        "testing.test_failed",
                        test=cmd_name,
                        exit_code=result["exit_code"],
                        stderr=result.get("stderr", ""),
                    )

            except Exception as exc:
                state.test_results.append(
                    TestResult(
                        test_name=cmd_name,
                        passed=False,
                        error=str(exc),
                    )
                )
                all_passed = False

        state.tests_passed = all_passed
        state.sandbox_output = "\n".join(
            f"[{'PASS' if tr.passed else 'FAIL'}] {tr.test_name}: {tr.output}" for tr in state.test_results
        )

        if all_passed:
            state.consecutive_test_failures = 0
            state.phase = AgentPhase.COMPLETE
            state.result_summary = (
                f"All {len(state.test_results)} tests passed. Modified {len(state.file_changes)} files."
            )
        else:
            state.consecutive_test_failures += 1
            state.phase = AgentPhase.FIXING  # Go back to coding to fix

        test_data = [{"test_name": tr.test_name, "passed": tr.passed, "error": tr.error} for tr in state.test_results]

        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="testing",
                model_role="sandbox",
                test_results=test_data,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )

        logger.info(
            "testing.complete",
            task_id=str(state.task_id),
            passed=all_passed,
            total_tests=len(state.test_results),
            failures=sum(1 for tr in state.test_results if not tr.passed),
        )

    except Exception as exc:
        logger.error("testing.failed", error=str(exc))
        state.tests_passed = False
        state.consecutive_test_failures += 1
        state.phase = AgentPhase.FIXING
        state.error_message = f"Sandbox testing error: {exc}"

    finally:
        # Always clean up sandbox
        if state.sandbox_id:
            try:
                await sandbox.destroy(state.sandbox_id)
            except Exception as exc:
                logger.warning("testing.sandbox_cleanup_failed", sandbox_id=state.sandbox_id, error=str(exc))

    return state


def _build_test_commands(state: AgentState) -> list[tuple[str, str]]:
    """
    Build a list of (test_name, shell_command) based on the files generated.
    Standalone-mode only (no repository_url) — repo-mode uses the repo's own
    real tooling via repo_profile.detect_repo_profile instead.
    """
    commands = []
    extensions = {fc.path.rsplit(".", 1)[-1] if "." in fc.path else "" for fc in state.file_changes}
    paths = [fc.path for fc in state.file_changes if fc.action != "delete"]

    if "py" in extensions:
        # Syntax check all Python files
        py_files = [p for p in paths if p.endswith(".py")]
        if py_files:
            commands.append(("python_syntax", f"python -m py_compile {' '.join(py_files)}"))

        # Look for test files
        test_files = [p for p in py_files if "test" in p.lower()]
        if test_files:
            commands.append(("pytest", f"python -m pytest {' '.join(test_files)} -v --tb=short 2>&1"))
        elif py_files:
            # Try importing to check for runtime errors
            for pf in py_files[:3]:
                module = pf.replace("/", ".").replace(".py", "")
                commands.append((f"import_{module}", f"python -c 'import {module}' 2>&1"))

    if "js" in extensions or "ts" in extensions:
        js_files = [p for p in paths if p.endswith((".js", ".ts"))]
        if js_files:
            commands.append(("node_syntax", f"node --check {' '.join(f for f in js_files if f.endswith('.js'))}"))
        test_files = [p for p in js_files if "test" in p.lower() or "spec" in p.lower()]
        if test_files:
            commands.append(("jest", "npx jest --passWithNoTests 2>&1"))

    if "go" in extensions:
        commands.append(("go_build", "go build ./... 2>&1"))
        commands.append(("go_test", "go test ./... -v 2>&1"))

    if "rs" in extensions:
        commands.append(("cargo_check", "cargo check 2>&1"))
        commands.append(("cargo_test", "cargo test 2>&1"))

    # If no language-specific commands, at least check file existence
    if not commands:
        check_cmds = " && ".join(f"test -f {p}" for p in paths[:10])
        if check_cmds:
            commands.append(("files_exist", check_cmds))

    return commands
