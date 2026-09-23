# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Best-of-N coding (src/orchestrator/best_of_n.py) in a real sandbox with a real git repository: three
candidate loops make real edits (one breaks the tests, one passes them but adds a lint finding, one
passes cleanly), each is scored by the repository's own pytest and ruff, the clean one wins, its change
lands on the working branch as ordinary uncommitted edits, the candidate branches are gone, and the
trace carries every candidate's score plus the ranking. Only the model call is scripted — the loop
function — everything it acts on is real. Requires the sandbox daemon (SANDBOX_DAEMON_URL).
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.orchestrator.best_of_n import run_best_of_n
from src.orchestrator.state import AgentState
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import SandboxManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("SANDBOX_DAEMON_URL"), reason="Needs the sandbox daemon"),
]

REPO_FILES = {
    "mathx.py": "def add(a, b):\n    return a + b\n",
    "tests/test_mathx.py": "from mathx import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    "requirements.txt": "",
    "pyproject.toml": (
        '[tool.ruff]\nline-length = 100\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n'
    ),
}


@pytest.fixture
async def ws():
    manager = SandboxManager()
    workspace = Workspace(manager, task_id=f"best-of-n-{uuid.uuid4().hex[:8]}")
    await workspace.ensure_sandbox(network_enabled=False)
    for path, content in REPO_FILES.items():
        await workspace.write_file(path, content)
    await workspace.run(
        "git init -q && git -c user.name=t -c user.email=t@t.dev add -A && "
        "git -c user.name=t -c user.email=t@t.dev commit -qm base && git checkout -q -b keystone/work",
        check=True,
    )
    try:
        yield workspace
    finally:
        await workspace.close()


async def test_the_candidate_that_passes_the_repos_own_checks_wins(ws, monkeypatch):
    monkeypatch.setenv("REDIS_URL", os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    state = AgentState(task_id=uuid.uuid4(), tenant_id=uuid.uuid4(), task_description="make add handle three args")
    state.quality_blocking_tools = ["ruff"]
    attempts = iter(
        [
            # 1: breaks the existing test
            ("def add(a, b, c=0):\n    return a - b + c\n", "changed add (wrong)"),
            # 2: passes the test but leaves an unused import for ruff (F401)
            ("import os\n\n\ndef add(a, b, c=0):\n    return a + b + c\n", "changed add (lint finding)"),
            # 3: passes cleanly
            ("def add(a, b, c=0):\n    return a + b + c\n", "changed add (clean)"),
        ]
    )

    async def scripted_loop(st, workspace):
        content, summary = next(attempts)
        await workspace.write_file("mathx.py", content)
        return True, summary, ["mathx.py"], []

    completed, summary, touched, steps = await run_best_of_n(state, ws, 3, run_loop=scripted_loop)

    assert completed and touched == ["mathx.py"]
    assert summary == "changed add (clean)"
    # the winner's edit is on the working branch as an uncommitted change; nothing else remains
    assert (await ws.read_file("mathx.py")) == "def add(a, b, c=0):\n    return a + b + c\n"
    status = (await ws.run("git status --porcelain", check=True))["stdout"]
    assert "M mathx.py" in status and ".patch" not in status
    branches = (await ws.run("git branch --list 'keystone-candidate-*'", check=True))["stdout"].strip()
    assert branches == ""
    assert (await ws.run("git rev-parse --abbrev-ref HEAD", check=True))["stdout"].strip() == "keystone/work"

    candidates = [s for s in steps if s.get("event_type") == "candidate"]
    scored = [c for c in candidates if c.get("winner") is None]  # step payloads are flattened onto the event
    final = [c for c in candidates if c.get("winner") is not None]
    assert len(scored) == 3 and len(final) == 1
    by_index = {c["index"]: c for c in scored}
    assert by_index[1]["tests_passed"] is False  # the wrong arithmetic really failed pytest
    assert by_index[2]["tests_passed"] is True and by_index[2]["blocking_findings"] >= 1  # ruff really flagged F401
    assert by_index[3]["tests_passed"] is True and by_index[3]["blocking_findings"] == 0
    assert final[0]["winner"] == 3
    assert next(r["index"] for r in final[0]["ranking"]) == 3
