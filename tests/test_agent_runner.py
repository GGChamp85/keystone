# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for benchmarks/agent_runner.py's pure task-loading logic.

The rest of agent_runner.py (seed_gitea_repo, run_one_task, run_suite) was
verified for real by hand against a live Gitea, a live sandbox daemon, and
a live Anthropic API through benchmarks/frontier_proxy.py — the full
plan→code→quality→review→test→PR loop, independently re-verified by
cloning the resulting branch in a fresh sandbox and running the task's
real fail_to_pass/pass_to_pass tests (both passed). That path isn't wired
into the automated suite: it needs a live Gitea reachable from inside a
sandbox container (this project has no standing one — see
tests/test_git_workflow_integration.py's module docstring for how to
stand one up) and spends real frontier-model tokens on every run, neither
of which belongs in a CI job that runs on every push.
"""

from __future__ import annotations

from benchmarks.agent_runner import REPO_TASKS_DIR, _list_repo_files, load_repo_tasks


def test_the_token_bucket_task_loads_with_its_real_fields():
    tasks = load_repo_tasks()
    ids = [t["id"] for t in tasks]
    assert "token_bucket" in ids

    task = next(t for t in tasks if t["id"] == "token_bucket")
    assert task["fail_to_pass"] == ["tests/test_token_bucket.py::test_tokens_never_exceed_capacity_after_long_idle"]
    assert task["pass_to_pass"] == ["tests/test_token_bucket.py::test_consume_within_capacity"]
    assert task["test_command"]
    assert task["instruction"]
    assert task["_dir"] == REPO_TASKS_DIR / "token_bucket"


def test_load_repo_tasks_filters_by_id():
    tasks = load_repo_tasks(only="token_bucket")
    assert len(tasks) == 1
    assert tasks[0]["id"] == "token_bucket"

    tasks = load_repo_tasks(only="does-not-exist")
    assert tasks == []


def test_list_repo_files_walks_the_real_seed_repo_and_base64_encodes_content():
    repo_dir = REPO_TASKS_DIR / "token_bucket" / "repo"
    files = _list_repo_files(repo_dir)
    paths = {p for p, _content in files}
    assert "token_bucket.py" in paths
    assert "tests/test_token_bucket.py" in paths
    assert "pyproject.toml" in paths

    import base64

    content_by_path = dict(files)
    decoded = base64.b64decode(content_by_path["token_bucket.py"]).decode()
    assert "class TokenBucket" in decoded


def test_the_seed_repo_really_has_the_expected_pass_fail_split():
    """Not agent_runner.py's own logic, but a guard against the seed repo
    silently drifting out of sync with its own task.json — if this ever
    fails, the fail_to_pass/pass_to_pass ids in task.json no longer
    describe what the repo actually does."""
    import subprocess
    import sys

    repo_dir = REPO_TASKS_DIR / "token_bucket" / "repo"
    task = load_repo_tasks(only="token_bucket")[0]

    for test_id in task["pass_to_pass"]:
        result = subprocess.run(  # noqa: S603 — sys.executable + test ids from this repo's own task.json, not user input
            [sys.executable, "-m", "pytest", test_id, "-q"], cwd=repo_dir, capture_output=True, text=True
        )
        assert result.returncode == 0, f"pass_to_pass test {test_id} does not actually pass:\n{result.stdout}"

    for test_id in task["fail_to_pass"]:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", test_id, "-q"], cwd=repo_dir, capture_output=True, text=True
        )
        assert result.returncode != 0, (
            f"fail_to_pass test {test_id} already passes on the unfixed seed:\n{result.stdout}"
        )
