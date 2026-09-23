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


# language -> (the runtime image the sandbox daemon uses for it, the toolchain binary a host run needs)
_RUNTIMES = {"node": ("keystone-sandbox-node:latest", "node"), "go": ("keystone-sandbox-go:latest", "go")}


def _host_or_image(language: str) -> tuple[str, str | None]:
    """Where to run a non-Python task's tests: on the host when its toolchain is installed, else inside
    the task's real sandbox runtime image when that is built, else skip with the reason."""
    image, binary = _RUNTIMES[language]
    if shutil.which(binary):
        return "host", None
    docker = shutil.which("docker")
    if docker:
        probe = subprocess.run([docker, "image", "inspect", image], capture_output=True, text=True)  # noqa: S603
        if probe.returncode == 0:
            return "image", image
    pytest.skip(f"{language} task premise needs `{binary}` on PATH or the {image} image (make sandbox-images)")


def _run_test(repo: Path, language: str, test_command: str, test_id: str) -> bool:
    if language == "python":
        argv = [sys.executable, "-m", "pytest", test_id, "-q", "-p", "no:cacheprovider"]
        return subprocess.run(argv, cwd=repo, capture_output=True, text=True, timeout=120).returncode == 0  # noqa: S603
    where, image = _host_or_image(language)
    command = f"{test_command} {test_id}"
    if where == "host":
        proc = subprocess.run(command, cwd=repo, shell=True, capture_output=True, text=True, timeout=300)  # noqa: S602
        return proc.returncode == 0
    docker = shutil.which("docker") or "docker"
    argv = [
        docker,
        "run",
        "--rm",
        "-v",
        f"{repo}:/workspace/repo",
        "-w",
        "/workspace/repo",
        str(image),
        "sh",
        "-c",
        command,
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=600).returncode == 0  # noqa: S603


def _outcomes(repo: Path, language: str, test_command: str, test_ids: list[str]) -> dict[str, bool]:
    return {test_id: _run_test(repo, language, test_command, test_id) for test_id in test_ids}


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=[p.name for p in TASK_DIRS])
def test_task_premise_holds_on_the_unfixed_repo(task_dir: Path, tmp_path: Path):
    task = json.loads((task_dir / "task.json").read_text())
    assert task["id"] == task_dir.name
    assert task["fail_to_pass"] and task["pass_to_pass"], "a task needs both kinds of held-out test"
    assert len(task["instruction"]) > 80, "the instruction must describe the bug, not just name it"

    repo = tmp_path / "repo"
    shutil.copytree(task_dir / "repo", repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    language = task.get("language", "python")
    failing = _outcomes(repo, language, task["test_command"], task["fail_to_pass"])
    assert all(not passed for passed in failing.values()), f"fail_to_pass must fail before the fix: {failing}"
    passing = _outcomes(repo, language, task["test_command"], task["pass_to_pass"])
    assert all(passing.values()), f"pass_to_pass must pass before the fix: {passing}"


def test_there_are_at_least_twelve_repo_tasks_across_three_ecosystems():
    assert len(TASK_DIRS) >= 12, [p.name for p in TASK_DIRS]
    languages = {json.loads((p / "task.json").read_text()).get("language", "python") for p in TASK_DIRS}
    assert {"python", "node", "go"} <= languages
