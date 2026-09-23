# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Every repo benchmark task (benchmarks/tasks/repo/<id>/) must be honest
before an agent ever sees it: on the UNFIXED repository, each `fail_to_pass`
test must actually fail and each `pass_to_pass` test must actually pass.
Otherwise a "solved" run would be meaningless. This runs the task's own
pytest against a scratch copy of its repo — real execution, no reading of
the tests as text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_TASKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "tasks" / "repo"
TASK_DIRS = sorted(p for p in REPO_TASKS_DIR.iterdir() if (p / "task.json").exists())


def _pytest_outcomes(repo: Path, test_ids: list[str]) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for test_id in test_ids:
        proc = subprocess.run(  # noqa: S603 — the task's own pytest, on a scratch copy of its own repo
            [sys.executable, "-m", "pytest", test_id, "-q", "-p", "no:cacheprovider"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120,
        )
        outcomes[test_id] = proc.returncode == 0
    return outcomes


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=[p.name for p in TASK_DIRS])
def test_task_premise_holds_on_the_unfixed_repo(task_dir: Path, tmp_path: Path):
    task = json.loads((task_dir / "task.json").read_text())
    assert task["id"] == task_dir.name
    assert task["fail_to_pass"] and task["pass_to_pass"], "a task needs both kinds of held-out test"
    assert len(task["instruction"]) > 80, "the instruction must describe the bug, not just name it"

    repo = tmp_path / "repo"
    shutil.copytree(task_dir / "repo", repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    failing = _pytest_outcomes(repo, task["fail_to_pass"])
    assert all(not passed for passed in failing.values()), f"fail_to_pass must fail before the fix: {failing}"
    passing = _pytest_outcomes(repo, task["pass_to_pass"])
    assert all(passing.values()), f"pass_to_pass must pass before the fix: {passing}"


def test_there_are_at_least_nine_repo_tasks():
    assert len(TASK_DIRS) >= 9, [p.name for p in TASK_DIRS]
