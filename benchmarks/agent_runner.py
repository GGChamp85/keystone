# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — Phase 6 repo-task benchmark runner.

This is the real thing the plan's Phase 6 calls for and
benchmarks/run_benchmark.py is not: a repo-scale, multi-file task run
through the *actual* agent loop (src/orchestrator/engine.py's real
KeystoneEngine, the real LangGraph graph, real tools, real quality gates),
not a single free-standing function completion. For each task in
benchmarks/tasks/repo/<name>/:

  1. Seed a fresh real Gitea repo with that task's starting code
     (benchmarks/tasks/repo/<name>/repo/) via Gitea's real contents API.
  2. Submit a real AgentTask against it through the real engine — the
     `coding` (and `reasoning`, if enabled) model role is whatever
     VLLM_CODING_URL/VLLM_REASONING_URL point at; point them at a real
     vLLM endpoint or benchmarks/frontier_proxy.py for a real frontier
     model — never mocked or scripted.
  3. Poll the real AgentTask row until it reaches a terminal status.
  4. If it completed and pushed a branch, clone that exact branch in a
     *fresh* sandbox (independent from whatever sandbox the agent used)
     and run the task's real fail_to_pass/pass_to_pass tests individually
     — pass/fail is the real pytest exit code per test, nothing inferred
     from the agent's own summary.

Requires: SANDBOX_DAEMON_URL, DATABASE_URL, REDIS_URL reachable; GIT_HOST_API_URL/
GIT_HOST_TOKEN/GIT_ALLOWED_HOSTS pointed at a real, network-reachable-from-the-
sandbox Gitea instance (see tests/test_git_workflow_integration.py's module
docstring for how to stand one up); VLLM_CODING_URL reachable.

Usage:
  python -m benchmarks.agent_runner                       # every benchmarks/tasks/repo/* task
  python -m benchmarks.agent_runner --task token_bucket    # just one
  python -m benchmarks.agent_runner --max-iterations 8 --timeout-seconds 600
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

REPO_TASKS_DIR = Path(__file__).parent / "tasks" / "repo"


def load_repo_tasks(only: str | None = None) -> list[dict[str, Any]]:
    tasks = []
    for task_dir in sorted(REPO_TASKS_DIR.iterdir()):
        task_json = task_dir / "task.json"
        if not task_json.exists():
            continue
        data = json.loads(task_json.read_text())
        if only and data["id"] != only:
            continue
        data["_dir"] = task_dir
        tasks.append(data)
    return tasks


_SEED_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _list_repo_files(repo_source_dir: Path) -> list[tuple[str, str]]:
    """Every file of the task's repo except local tool caches — a seeded repo must be exactly the source."""
    return [
        (path.relative_to(repo_source_dir).as_posix(), base64.b64encode(path.read_bytes()).decode())
        for path in sorted(repo_source_dir.rglob("*"))
        if path.is_file()
        and not (set(path.relative_to(repo_source_dir).parts[:-1]) & _SEED_SKIP_DIRS)
        and path.suffix != ".pyc"
    ]


async def seed_gitea_repo(repo_source_dir: Path, gitea_api_url: str, gitea_token: str, repo_name: str) -> str:
    """
    Creates a fresh Gitea repo under the token's own user and pushes every
    file in `repo_source_dir` into it via Gitea's real contents API — no
    local git needed for seeding, matching
    tests/test_git_workflow_integration.py's own seeded_repo fixture
    pattern, generalized to walk an arbitrary directory tree instead of a
    fixed file dict.
    """
    async with httpx.AsyncClient(
        base_url=gitea_api_url.rstrip("/"), headers={"Authorization": f"token {gitea_token}"}, timeout=30.0
    ) as client:
        create_resp = await client.post("/user/repos", json={"name": repo_name, "private": False, "auto_init": True})
        create_resp.raise_for_status()
        repo_info = create_resp.json()
        owner = repo_info["owner"]["login"]

        files = await asyncio.to_thread(_list_repo_files, repo_source_dir)
        for rel_path, content_b64 in files:
            resp = await client.get(f"/repos/{owner}/{repo_name}/contents/{rel_path}")
            payload = {"content": content_b64, "message": f"seed: {rel_path}", "branch": "main"}
            if resp.status_code == 200:
                payload["sha"] = resp.json()["sha"]
                put_resp = await client.put(f"/repos/{owner}/{repo_name}/contents/{rel_path}", json=payload)
                put_resp.raise_for_status()
            else:
                post_resp = await client.post(f"/repos/{owner}/{repo_name}/contents/{rel_path}", json=payload)
                post_resp.raise_for_status()

    base_url = gitea_api_url.rstrip("/").removesuffix("/api/v1")
    return f"{base_url}/{owner}/{repo_name}.git"


async def _create_benchmark_tenant_and_key() -> tuple[uuid.UUID, uuid.UUID]:
    from src.api.middleware.auth import generate_api_key
    from src.db.connection import get_db_context
    from src.db.models import APIKey, Tenant, TenantTier

    tenant_id = uuid.uuid4()
    _full_key, prefix, key_hash = generate_api_key()
    api_key_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(
                id=tenant_id,
                name=f"benchmark-{tenant_id.hex[:8]}",
                email=f"{tenant_id}@benchmark.test",
                tier=TenantTier.FREE,
                max_concurrent_agents=5,
            )
        )
        await db.flush()
        db.add(
            APIKey(
                id=api_key_id,
                tenant_id=tenant_id,
                name="agent-runner",
                key_prefix=prefix,
                key_hash=key_hash,
                scopes=["agent"],
            )
        )
        await db.flush()
    return tenant_id, api_key_id


async def _wait_for_terminal_status(task_id: uuid.UUID, timeout_seconds: int, poll_interval: float = 3.0) -> Any:
    from src.db.connection import get_db_context
    from src.db.models import AgentTask, TaskStatus

    terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT}
    elapsed = 0.0
    while elapsed < timeout_seconds:
        async with get_db_context() as db:
            task = await db.get(AgentTask, task_id)
        if task is not None and task.status in terminal:
            return task
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Task {task_id} did not reach a terminal status within {timeout_seconds}s")


def _held_out_prefix(task: dict[str, Any]) -> str:
    """The command a single held-out test id is appended to: one test at a time, so pytest's own
    invocation is used for Python; Node and Go tasks declare a prefix-style `test_command`."""
    return "python -m pytest" if task.get("language", "python") == "python" else task["test_command"]


# task.json "language" -> the sandbox daemon's runtime template (src/sandbox/daemon.py RUNTIME_IMAGES)
_RUNTIME_TEMPLATE = {"python": "python", "node": "javascript", "typescript": "typescript", "go": "go"}


async def _run_held_out_tests(
    repo_url: str, branch_name: str, test_ids: list[str], test_command_prefix: str, language: str = "python"
) -> dict:
    """Clones `branch_name` in a brand-new sandbox — independent of whatever
    sandbox the agent itself used — built from the runtime image for the task's
    language, and runs each held-out test id individually so one unrelated
    failure doesn't mask the rest."""
    from src.orchestrator.workspace import Workspace
    from src.sandbox.manager import SandboxManager

    manager = SandboxManager()
    results: dict[str, bool] = {}
    task_id = f"verify-{uuid.uuid4().hex[:8]}"
    ws = Workspace(manager, task_id=task_id, tenant_id="benchmark-verify")
    try:
        await ws.ensure_sandbox(network_enabled=True, language=_RUNTIME_TEMPLATE.get(language, "python"))
        await ws.clone(repo_url, branch=branch_name)
        for test_id in test_ids:
            outcome = await ws.run(f"{test_command_prefix} {test_id}", timeout=120, check=False)
            results[test_id] = outcome.get("exit_code") == 0
    finally:
        await ws.close()
        await manager.aclose()
    return results


async def run_one_task(task: dict[str, Any], *, max_iterations: int, timeout_seconds: int) -> dict[str, Any]:
    import os

    from src.orchestrator.engine import KeystoneEngine

    repo_name = f"{task['id']}-{uuid.uuid4().hex[:8]}"
    gitea_api_url = os.environ["GIT_HOST_API_URL"]
    gitea_token = os.environ["GIT_HOST_TOKEN"]

    logger.info("agent_runner.seeding_repo", task_id=task["id"], repo_name=repo_name)
    repo_url = await seed_gitea_repo(task["_dir"] / "repo", gitea_api_url, gitea_token, repo_name)

    tenant_id, api_key_id = await _create_benchmark_tenant_and_key()

    engine = KeystoneEngine()
    started = time.monotonic()
    agent_task_id = await engine.submit_task(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        task_description=task["instruction"],
        repository_url=repo_url,
        branch="main",
        model="coding",
        max_iterations=max_iterations,
    )
    logger.info("agent_runner.task_submitted", task_id=task["id"], agent_task_id=str(agent_task_id))

    try:
        final = await _wait_for_terminal_status(agent_task_id, timeout_seconds)
    except TimeoutError as exc:
        return {
            "task_id": task["id"],
            "agent_task_id": str(agent_task_id),
            "status": "timeout",
            "error": str(exc),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    result: dict[str, Any] = {
        "task_id": task["id"],
        "agent_task_id": str(agent_task_id),
        "status": final.status.value,
        "branch_name": final.branch_name,
        "pr_url": final.pr_url,
        "error_message": final.error_message,
        "total_prompt_tokens": final.total_prompt_tokens,
        "total_completion_tokens": final.total_completion_tokens,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }

    if final.status.value != "completed" or not final.branch_name:
        result["fail_to_pass"] = dict.fromkeys(task["fail_to_pass"], False)
        result["pass_to_pass"] = dict.fromkeys(task["pass_to_pass"], False)
        return result

    logger.info("agent_runner.verifying_held_out_tests", task_id=task["id"], branch=final.branch_name)
    all_test_ids = task["fail_to_pass"] + task["pass_to_pass"]
    held_out = await _run_held_out_tests(
        repo_url, final.branch_name, all_test_ids, _held_out_prefix(task), task.get("language", "python")
    )

    result["fail_to_pass"] = {t: held_out.get(t, False) for t in task["fail_to_pass"]}
    result["pass_to_pass"] = {t: held_out.get(t, False) for t in task["pass_to_pass"]}
    result["solved"] = all(result["fail_to_pass"].values()) and all(result["pass_to_pass"].values())
    return result


async def run_suite(only: str | None, max_iterations: int, timeout_seconds: int) -> list[dict[str, Any]]:
    tasks = load_repo_tasks(only)
    if not tasks:
        raise RuntimeError(f"No repo tasks found in {REPO_TASKS_DIR}" + (f" matching {only!r}" if only else ""))
    results = []
    for task in tasks:
        print(f"\n=== Running {task['id']} ===")
        result = await run_one_task(task, max_iterations=max_iterations, timeout_seconds=timeout_seconds)
        results.append(result)
        solved = result.get("solved", False)
        print(f"[{'SOLVED' if solved else 'NOT SOLVED'}] {task['id']} (status={result['status']})")
        if result.get("pr_url"):
            print(f"  PR: {result['pr_url']}")
        for test_id, passed in {**result.get("fail_to_pass", {}), **result.get("pass_to_pass", {})}.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {test_id}")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, help="Run only this task id")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--persist", action="store_true", help="Write every result to the benchmark_runs table (benchmarks/runs.py)"
    )
    parser.add_argument(
        "--backend-label",
        default="coding",
        help="How to tag persisted runs: the backend behind the coding role (e.g. coding, frontier, my-adapter)",
    )
    return parser


async def _main() -> int:
    args = _build_arg_parser().parse_args()
    results = await run_suite(args.task, args.max_iterations, args.timeout_seconds)

    solved_count = sum(1 for r in results if r.get("solved"))
    print(f"\n=== Summary: {solved_count}/{len(results)} solved ===")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Full report written to {args.output}")

    if args.persist:
        from benchmarks.runs import persist_results
        from src.config import get_settings

        ids = await persist_results(results, backend_label=args.backend_label, model_id=get_settings().coding_model_id)
        print(f"Persisted {len(ids)} run(s) to benchmark_runs as backend {args.backend_label!r}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
