# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Fixing node.

Normally just re-enters coding with the test failures/review comments as
extra context, so the model can fix its own mistakes. But repeated failures
(>=2 consecutive test or review failures) are a signal that *execution*
tweaks aren't working — the plan itself may be wrong. In that case, before
just trying again, a root-cause step asks the reasoning model for its best
hypothesis and whether the plan should be redone from scratch; a genuine
re-plan (back to PLANNING, carrying the hypothesis forward) only happens
when the model says so, not on every repeated failure — otherwise this
would just add cost without changing the outcome.
"""

from __future__ import annotations

import structlog

from src.inference.client import get_inference_client
from src.orchestrator.nodes.coding import coding_node
from src.orchestrator.state import AgentPhase, AgentState

logger = structlog.get_logger(__name__)

ROOT_CAUSE_FAILURE_THRESHOLD = 2

ROOT_CAUSE_SYSTEM_PROMPT = """\
You are the root-cause analysis module of Keystone Agents, an autonomous coding agent by Keystone. The coding \
agent has failed the same test or review gate repeatedly on this task. Your job is not to fix the code — it's \
to diagnose WHY the fixes keep failing, using the failure ledger you're given.

Decide: is this a bad EXECUTION (the plan is right, but each fix attempt has a bug or misses a detail) or a bad \
PLAN (the overall approach is wrong — e.g. editing the wrong files, missing a required step, a design that \
can't actually satisfy the task)? Only recommend replanning when the plan itself is the problem — replanning \
throws away plan-level context and costs real time, so don't recommend it just because the execution has been \
sloppy.
"""

ROOT_CAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "description": "Best guess at the actual root cause, one or two sentences."},
        "evidence": {"type": "string", "description": "What in the failure ledger supports this hypothesis."},
        "replan": {
            "type": "boolean",
            "description": "True only if the current plan itself needs to be redone, not just re-executed.",
        },
    },
    "required": ["hypothesis", "evidence", "replan"],
}


def _build_failure_ledger(state: AgentState) -> str:
    parts = []
    if state.test_results:
        failing = [tr for tr in state.test_results if not tr.passed]
        if failing:
            parts.append("Test failures:")
            parts.extend(f"- {tr.test_name}: {tr.error[:500]}" for tr in failing)
    if state.review_comments:
        blocking = [rc for rc in state.review_comments if rc.severity in ("error", "critical")]
        if blocking:
            parts.append("Review comments:")
            parts.extend(f"- [{rc.severity}] {rc.file_path}:{rc.line}: {rc.message}" for rc in blocking)
    if state.quality_findings:
        blocking_findings = [
            f for f in state.quality_findings if f.tool in ("bandit", "mypy") and f.severity == "error"
        ]
        if blocking_findings:
            parts.append("Quality gate findings:")
            parts.extend(f"- [{f.tool}] {f.path}:{f.line}: {f.message}" for f in blocking_findings)
    if state.root_cause_notes:
        parts.append("Root-cause hypotheses already tried this task (don't repeat these):")
        parts.extend(f"- {note}" for note in state.root_cause_notes)
    return "\n".join(parts) or "(no specific failure detail recorded)"


async def _run_root_cause_analysis(state: AgentState) -> dict:
    """
    Advisory, not a gate: unlike review (which fails closed), a root-cause
    step that can't produce an answer just means "no extra insight this
    round" — fixing falls back to its normal re-coding path, which is
    still safe and correct on its own.
    """
    client = get_inference_client("reasoning")
    user_content = (
        f"## Task\n{state.task_description}\n\n"
        f"## Current Plan\n{state.plan}\n\n"
        f"## Failure Ledger (consecutive: test={state.consecutive_test_failures}, "
        f"review={state.consecutive_review_failures})\n{_build_failure_ledger(state)}"
    )
    messages = [
        {"role": "system", "content": ROOT_CAUSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return await client.chat_structured(
            messages, ROOT_CAUSE_SCHEMA, schema_name="root_cause", on_usage=state.add_tokens
        )
    except Exception as exc:
        logger.warning("fixing.root_cause_unavailable", task_id=str(state.task_id), error=str(exc))
        return {"hypothesis": "", "evidence": "", "replan": False}


async def fixing_node(state: AgentState) -> AgentState:
    """
    Fixing is normally just coding with extra context (failures/comments).
    We set the phase back to CODING and re-run the coding node, which
    already knows how to read test_results and review_comments — unless
    repeated failures trigger a root-cause step that decides to replan.
    """
    logger.info(
        "fixing.entering",
        task_id=str(state.task_id),
        iteration=state.iteration,
        test_failures=sum(1 for tr in state.test_results if not tr.passed),
        review_issues=len([rc for rc in state.review_comments if rc.severity in ("error", "critical")]),
        quality_findings=len(state.quality_findings),
        consecutive_test_failures=state.consecutive_test_failures,
        consecutive_review_failures=state.consecutive_review_failures,
        consecutive_quality_failures=state.consecutive_quality_failures,
    )

    if (
        state.consecutive_test_failures >= ROOT_CAUSE_FAILURE_THRESHOLD
        or state.consecutive_review_failures >= ROOT_CAUSE_FAILURE_THRESHOLD
        or state.consecutive_quality_failures >= ROOT_CAUSE_FAILURE_THRESHOLD
    ):
        analysis = await _run_root_cause_analysis(state)
        hypothesis = (analysis.get("hypothesis") or "").strip()
        if hypothesis:
            state.root_cause_notes.append(hypothesis)
        logger.info(
            "fixing.root_cause_analyzed",
            task_id=str(state.task_id),
            replan=bool(analysis.get("replan")),
            hypothesis=hypothesis,
        )
        if analysis.get("replan"):
            state.phase = AgentPhase.PLANNING
            state.consecutive_test_failures = 0
            state.consecutive_review_failures = 0
            state.consecutive_quality_failures = 0
            return state

    # The coding node reads state.test_results and state.review_comments
    # when they contain failures, and includes them in the prompt.
    state.phase = AgentPhase.CODING
    state = await coding_node(state)

    return state
