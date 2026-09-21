# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Review node.
Uses the reasoning model (DeepSeek-R1) as a critic to review
generated code before it goes to testing.

Fails closed, not open: a review the model can't produce (unparseable
output even after chat_structured's self-correction retry, the reasoning
model unreachable, no diff to review) blocks the task (FIXING, with an
explanatory comment) rather than silently waving it through as "passed" —
code nobody actually reviewed must never reach testing/PR labeled
"reviewed". Likewise `approved=true` from the model is overridden to
`false` if it's inconsistent with its own output (a `critical`/`error`
comment or a positive `blocking_issues` count) — the model's boolean is
advisory, not the last word.
"""

from __future__ import annotations

import time

import structlog

from src.inference.client import StructuredOutputError, get_inference_client
from src.inference.model_router import resolve_model_name_for_client
from src.orchestrator.nodes._shared import get_or_clone_workspace
from src.orchestrator.state import (
    AgentPhase,
    AgentState,
    IterationRecord,
    ReviewComment,
)

logger = structlog.get_logger(__name__)

REVIEW_SYSTEM_PROMPT = """\
You are the code review module of Keystone Agents, an autonomous coding agent by Keystone.
You are powered by a deep reasoning model and your job is to critically review code changes.

Review criteria:
1. **Correctness**: Does the code do what the task requires? Any logic errors?
2. **Security**: Any injection risks, hardcoded secrets, unsafe operations?
3. **Performance**: Any obvious N+1 queries, unbounded loops, memory leaks?
4. **Style**: Follows language conventions? Proper naming? Clean structure?
5. **Edge cases**: Missing null checks, boundary conditions, error handling?
6. **Completeness**: Does the implementation cover the full task spec?

Be thorough but pragmatic. Only flag real issues, not style nitpicks. `approved` must be consistent with your
own `comments`/`blocking_issues` — never approve while a comment is `error` or `critical` severity, or while
`blocking_issues` is greater than zero.
"""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "summary": {"type": "string", "description": "Overall assessment in 1-2 sentences."},
        "comments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
                    "message": {"type": "string"},
                    "suggestion": {"type": ["string", "null"]},
                },
                "required": ["file_path", "line", "severity", "message", "suggestion"],
            },
        },
        "blocking_issues": {"type": "integer", "description": "Count of comments that must be fixed before merge."},
    },
    "required": ["approved", "summary", "comments", "blocking_issues"],
}


def _fail_closed(state: AgentState, reason: str) -> AgentState:
    state.review_passed = False
    state.consecutive_review_failures += 1
    state.phase = AgentPhase.FIXING
    state.review_comments = [
        ReviewComment(file_path="", severity="error", message=f"Review unavailable: {reason}"),
    ]
    state.error_message = f"Review unavailable: {reason}"
    logger.warning(
        "review.fail_closed", task_id=str(state.task_id), reason=reason, consecutive=state.consecutive_review_failures
    )
    return state


async def review_node(state: AgentState) -> AgentState:
    """
    Send all code changes to the reasoning model for review.
    """
    if not state.enable_reasoning_review:
        state.review_passed = True
        state.phase = AgentPhase.TESTING if state.enable_sandbox_testing else AgentPhase.COMPLETE
        return state

    t0 = time.monotonic()
    client = get_inference_client("reasoning")
    model_name = await resolve_model_name_for_client(client, state.tenant_id)

    try:
        user_parts = [f"## Task\n{state.task_description}"]

        if state.plan:
            user_parts.append(f"\n## Plan\n{state.plan}")

        if state.repository_url:
            # Repo-mode tasks are edited directly in the sandboxed working tree
            # via apply_patch (src/orchestrator/nodes/coding.py) — state.file_changes
            # stays empty there, so the real diff comes from git, not from state.
            ws = await get_or_clone_workspace(state)
            diff = await ws.diff()
            if not diff.strip():
                return _fail_closed(state, "coding produced no working-tree changes to review")
            user_parts.append(f"\n## Code Changes to Review (git diff)\n```diff\n{diff[:60_000]}\n```")
        else:
            user_parts.append("\n## Code Changes to Review")
            user_parts.extend(
                f"\n### {fc.path} ({fc.action})\n```{fc.language}\n{fc.new_content}\n```" for fc in state.file_changes
            )

        if state.context_files:
            user_parts.append("\n## Original Context Files")
            for fname, content in list(state.context_files.items())[:3]:
                truncated = content[:4000] if len(content) > 4000 else content
                user_parts.append(f"\n### {fname}\n```\n{truncated}\n```")

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        call_tokens = {"prompt": 0, "completion": 0}

        def _track_usage(prompt: int, completion: int) -> None:
            call_tokens["prompt"] += prompt
            call_tokens["completion"] += completion
            state.add_tokens(prompt, completion)

        try:
            parsed = await client.chat_structured(
                messages,
                REVIEW_SCHEMA,
                schema_name="code_review",
                temperature=0.1,
                max_tokens=8192,
                on_usage=_track_usage,
                model_override=model_name,
            )
        except StructuredOutputError as exc:
            return _fail_closed(state, f"model did not return a valid review after a self-correction retry ({exc})")

        state.review_comments = [
            ReviewComment(
                file_path=c.get("file_path", ""),
                line=c.get("line"),
                severity=c.get("severity", "info"),
                message=c.get("message", ""),
                suggestion=c.get("suggestion"),
            )
            for c in parsed.get("comments", [])
        ]
        blocking_issues = parsed.get("blocking_issues", 0)
        has_blocking_comment = any(rc.severity in ("error", "critical") for rc in state.review_comments)
        approved = bool(parsed.get("approved", False)) and blocking_issues <= 0 and not has_blocking_comment

        if approved:
            state.review_passed = True
            state.consecutive_review_failures = 0
            state.phase = AgentPhase.TESTING if state.enable_sandbox_testing else AgentPhase.COMPLETE
            if not state.enable_sandbox_testing:
                state.result_summary = parsed.get("summary", "Review passed.")
        else:
            state.review_passed = False
            state.consecutive_review_failures += 1
            state.phase = AgentPhase.FIXING

        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="review",
                model_role="reasoning",
                prompt_tokens=call_tokens["prompt"],
                completion_tokens=call_tokens["completion"],
                review_comments=[
                    {"file_path": rc.file_path, "severity": rc.severity, "message": rc.message}
                    for rc in state.review_comments
                ],
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )

        logger.info(
            "review.complete",
            task_id=str(state.task_id),
            approved=state.review_passed,
            comments=len(state.review_comments),
            blocking_issues=blocking_issues,
            consecutive_failures=state.consecutive_review_failures,
        )

    except Exception as exc:
        logger.error("review.failed", error=str(exc), task_id=str(state.task_id))
        return _fail_closed(state, str(exc))

    return state
