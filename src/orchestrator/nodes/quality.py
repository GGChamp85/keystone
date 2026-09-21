# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Quality gates node.

Sits between CODING and REVIEW: runs the repo's own real lint/typecheck/
security tooling (`repo_profile.py`'s `lint_cmds`/`typecheck_cmd` — already
detected there since Phase 1, but until now nothing ever actually ran
them) against the same already-cloned working tree `coding_node`'s tool
loop just edited. Repo-mode only, matching every other repo-aware node —
standalone-mode tasks (no repository_url) have no real repo tooling to run
against, so this node no-ops straight through for them.

Bandit (security) and mypy (type errors) findings block — real production
code with an unresolved security or type issue must not sail through to
review/tests. ruff and every other tool's findings are informational only
for this cut: real strictness per repo (what to block on, what to ignore)
is repo-memory-driven work (Phase 3), and until that governance exists,
blocking on every style nit would fail tasks for reasons a human reviewer
wouldn't actually care about. Fails closed on infra error (can't run the
tools at all) — same policy as testing_node and review_node: FIXING, not a
silent pass.
"""

from __future__ import annotations

import json
import re
import time

import structlog

from src.orchestrator.nodes._shared import get_or_clone_workspace, next_gate_phase
from src.orchestrator.repo_profile import detect_repo_profile
from src.orchestrator.state import AgentPhase, AgentState, IterationRecord, QualityFinding

logger = structlog.get_logger(__name__)

QUALITY_TOOL_TIMEOUT = 180
_MYPY_LINE_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?:\d+:)?\s*error:\s*(?P<message>.+?)(?:\s*\[(?P<code>[\w-]+)])?$"
)


def _parse_ruff_json(output: str) -> list[QualityFinding]:
    try:
        items = json.loads(output)
    except json.JSONDecodeError:
        return []
    findings = []
    for item in items:
        loc = item.get("location", {})
        findings.append(
            QualityFinding(
                tool="ruff",
                path=item.get("filename", ""),
                line=loc.get("row"),
                severity="warning",
                code=item.get("code") or "",
                message=item.get("message", ""),
            )
        )
    return findings


def _parse_bandit_json(output: str) -> list[QualityFinding]:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    findings = []
    for item in data.get("results", []):
        issue_severity = str(item.get("issue_severity", "LOW")).lower()
        severity = "error" if issue_severity in ("high", "medium") else "info"
        findings.append(
            QualityFinding(
                tool="bandit",
                path=item.get("filename", ""),
                line=item.get("line_number"),
                severity=severity,
                code=item.get("test_id", ""),
                message=item.get("issue_text", ""),
            )
        )
    return findings


def _parse_mypy_text(output: str) -> list[QualityFinding]:
    findings = []
    for line in output.splitlines():
        match = _MYPY_LINE_RE.match(line.strip())
        if match:
            findings.append(
                QualityFinding(
                    tool="mypy",
                    path=match.group("path"),
                    line=int(match.group("line")),
                    severity="error",
                    code=match.group("code") or "",
                    message=match.group("message"),
                )
            )
    return findings


def _parse_generic(tool: str, output: str) -> list[QualityFinding]:
    """Fallback for a tool without a structured parser (go vet, cargo clippy, eslint/tsc's text output): one
    opaque finding carrying the raw output — still real and actionable, just not per-line structured."""
    stripped = output.strip()
    if not stripped:
        return []
    return [QualityFinding(tool=tool, path="", line=None, severity="warning", code="", message=stripped[:4000])]


def _findings_for_command(command: str, output: str) -> list[QualityFinding]:
    if "ruff" in command:
        return _parse_ruff_json(output) or _parse_generic("ruff", output)
    if "bandit" in command:
        return _parse_bandit_json(output) or _parse_generic("bandit", output)
    if "mypy" in command:
        return _parse_mypy_text(output) or _parse_generic("mypy", output)
    tool = command.split()[0] if command.split() else "unknown"
    return _parse_generic(tool, output)


def _is_blocking(finding: QualityFinding) -> bool:
    return finding.tool in ("bandit", "mypy") and finding.severity == "error"


async def quality_node(state: AgentState) -> AgentState:
    if not state.repository_url or not state.enable_quality_gates:
        state.phase = next_gate_phase(state)
        return state

    t0 = time.monotonic()
    try:
        ws = await get_or_clone_workspace(state)
        files = await ws.list_files()
        scripts = await ws.read_package_json_scripts() if "package.json" in files else {}
        profile = detect_repo_profile(files, scripts)

        commands = list(profile.lint_cmds)
        if profile.typecheck_cmd:
            commands.append(profile.typecheck_cmd)

        all_findings: list[QualityFinding] = []
        for command in commands:
            result = await ws.run(command, timeout=QUALITY_TOOL_TIMEOUT, check=False)
            if result.get("exit_code", 0) == 0:
                continue
            output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
            all_findings.extend(_findings_for_command(command, output))

        state.quality_findings = all_findings
        blocking = [f for f in all_findings if _is_blocking(f)]

        if blocking:
            state.consecutive_quality_failures += 1
            state.phase = AgentPhase.FIXING
            state.error_message = f"{len(blocking)} blocking quality finding(s): " + "; ".join(
                f"{f.tool}:{f.path}:{f.line} {f.message[:100]}" for f in blocking[:5]
            )
        else:
            state.consecutive_quality_failures = 0
            state.phase = next_gate_phase(state)

        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="quality",
                model_role="tools",
                quality_findings=[
                    {
                        "tool": f.tool,
                        "path": f.path,
                        "line": f.line,
                        "severity": f.severity,
                        "code": f.code,
                        "message": f.message[:300],
                    }
                    for f in all_findings
                ],
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )

        logger.info(
            "quality.complete",
            task_id=str(state.task_id),
            commands=len(commands),
            findings=len(all_findings),
            blocking=len(blocking),
        )

    except Exception as exc:
        logger.error("quality.failed", task_id=str(state.task_id), error=str(exc))
        state.consecutive_quality_failures += 1
        state.phase = AgentPhase.FIXING
        state.error_message = f"Quality gate error: {exc}"

    return state
