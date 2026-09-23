# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — best-of-N coding: N independent attempts, one winner chosen by real checks.

Opt-in (`AGENT_BEST_OF_N` / `AgentTaskRequest.best_of_n`, default 1 = off — it multiplies model spend
by N). Each candidate runs the same agentic loop from the same clean base commit on its own git
branch inside the task's sandbox, is scored by the repository's own quality commands and its related
tests (the same gates the task will face), and the winner's changes are applied back onto the working
branch as ordinary uncommitted edits — so everything downstream (quality, review, tests, commit, PR)
is unchanged and sees exactly one candidate. Losers are recorded in the trace with their scores and
their branches are deleted; nothing is hidden.

Score, best first: related tests passed, fewer blocking quality findings, fewer findings overall, the
model signalled completion, fewer files touched (a smaller diff that passes beats a larger one).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import get_settings
from src.orchestrator.events import publish_task_step
from src.orchestrator.nodes.quality import _findings_for_command, _is_blocking
from src.orchestrator.repo_profile import RepoProfile, detect_repo_profile
from src.orchestrator.state import AgentState
from src.orchestrator.test_scope import related_test_command
from src.orchestrator.workspace import Workspace

logger = structlog.get_logger(__name__)

LoopFn = Callable[[AgentState, Workspace], Awaitable[tuple[bool, str, list[str], list[dict]]]]


@dataclass
class Candidate:
    index: int
    branch: str
    completed: bool
    summary: str
    touched: list[str]
    steps: list[dict] = field(default_factory=list)
    patch: str = ""
    tests_passed: bool | None = None  # None: no related test command for this ecosystem
    test_command: str | None = None
    blocking_findings: int = 0
    findings: int = 0
    duration_ms: int = 0

    @property
    def score(self) -> tuple[int, int, int, int, int]:
        """Higher is better."""
        return (
            1 if self.tests_passed is not False else 0,
            -self.blocking_findings,
            -self.findings,
            1 if self.completed else 0,
            -len(self.touched),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "branch": self.branch,
            "completed": self.completed,
            "summary": self.summary,
            "files_touched": self.touched,
            "tests_passed": self.tests_passed,
            "test_command": self.test_command,
            "blocking_findings": self.blocking_findings,
            "findings": self.findings,
            "duration_ms": self.duration_ms,
            "score": list(self.score),
        }


async def _profile(ws: Workspace) -> RepoProfile:
    files = await ws.list_files()
    scripts = await ws.read_package_json_scripts() if "package.json" in files else {}
    return detect_repo_profile(files, scripts)


async def score_candidate(state: AgentState, ws: Workspace, profile: RepoProfile, candidate: Candidate) -> None:
    """The task's own gates, run on the candidate's working tree: lint/typecheck findings and the tests
    related to what it touched. Nothing here is a model's opinion."""
    settings = get_settings()
    commands = list(profile.lint_cmds)
    if profile.typecheck_cmd:
        commands.append(profile.typecheck_cmd)
    findings = []
    for command in commands:
        result = await ws.run(command, timeout=settings.agent_quality_timeout_seconds, check=False)
        if result.get("exit_code", 0) != 0:
            findings.extend(_findings_for_command(command, f"{result.get('stdout', '')}\n{result.get('stderr', '')}"))
    candidate.findings = len(findings)
    candidate.blocking_findings = sum(1 for f in findings if _is_blocking(f, state.quality_blocking_tools))

    test_cmd = related_test_command(profile, candidate.touched, await ws.list_tracked_files())
    candidate.test_command = test_cmd
    if test_cmd:
        result = await ws.run(test_cmd, timeout=settings.agent_test_timeout_seconds, check=False)
        candidate.tests_passed = result.get("exit_code") in (0, 5)  # 5: pytest collected nothing related


async def run_best_of_n(
    state: AgentState, ws: Workspace, n: int, *, run_loop: LoopFn
) -> tuple[bool, str, list[str], list[dict]]:
    """Same contract as the single agentic loop: (completed, summary, touched, steps) — for the winner,
    with every candidate's `candidate` step event included so the trace shows the whole contest."""
    base = (await ws.run("git rev-parse HEAD", check=True))["stdout"].strip()
    working_branch = (await ws.run("git rev-parse --abbrev-ref HEAD", check=True))["stdout"].strip()
    profile = await _profile(ws)
    candidates: list[Candidate] = []
    all_steps: list[dict] = []

    for i in range(n):
        branch = f"keystone-candidate-{i + 1}"
        t0 = time.monotonic()
        await ws.run(f"git checkout -q -B {branch} {base}", check=True)
        completed, summary, touched, steps = await run_loop(state, ws)
        cand = Candidate(i + 1, branch, completed, summary, touched, steps)
        try:
            await score_candidate(state, ws, profile, cand)
        except Exception as exc:  # a broken tree is a losing candidate, not a crashed task
            logger.warning("best_of_n.score_failed", candidate=i + 1, error=str(exc))
            cand.tests_passed = False
        # the candidate's whole change as one patch (untracked files included), then a clean tree for the next
        await ws.run("git add -A", check=False)
        cand.patch = (await ws.run(f"git diff --cached --binary {base}", check=True))["stdout"]
        await ws.run("git reset -q --hard && git clean -qfd", check=False)
        cand.duration_ms = int((time.monotonic() - t0) * 1000)
        candidates.append(cand)
        all_steps.extend(steps)
        all_steps.append(
            await publish_task_step(
                state.task_id, "candidate", "coding", {"of": n, **cand.to_dict(), "winner": None}, phase="coding"
            )
        )

    best = max(candidates, key=lambda c: c.score)
    await ws.run(f"git checkout -q {working_branch}", check=True)
    if best.patch.strip():
        await ws.write_file(".keystone-best-of-n.patch", best.patch)
        await ws.run("git apply --binary --whitespace=nowarn .keystone-best-of-n.patch", check=True)
        await ws.run("rm -f .keystone-best-of-n.patch", check=False)
    for c in candidates:
        await ws.run(f"git branch -q -D {c.branch}", check=False)
    all_steps.append(
        await publish_task_step(
            state.task_id,
            "candidate",
            "coding",
            {
                "of": n,
                "winner": best.index,
                "ranking": [c.to_dict() for c in sorted(candidates, key=lambda c: c.score, reverse=True)],
            },
            phase="coding",
        )
    )
    logger.info(
        "best_of_n.selected",
        task_id=str(state.task_id),
        n=n,
        winner=best.index,
        scores=[c.score for c in candidates],
    )
    return best.completed, best.summary, best.touched, all_steps
