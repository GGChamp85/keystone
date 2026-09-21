# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Coding node.

Two modes, chosen by whether the task has a `repository_url` — matching
the same split `nodes/testing.py` uses, for the same reason:

- **Repo mode** (real git workflow): a real agentic tool-use loop against
  the task's cloned working tree (`src/orchestrator/nodes/_shared.py`'s
  shared `Workspace`) — read/list/grep to explore the real code, edit it
  with `apply_patch`, optionally run a command, for up to
  `state.max_tool_steps` turns, ending when the model replies without a
  tool call (its final message becomes the change summary) or the turn
  budget runs out. This is the actual "core of agentic coding": no full-file
  rewrites from memory, no blind guesses about what a file currently
  contains — every edit is grounded in something the model just read.
- **Standalone mode** (no repository_url — benchmarks, context-files-only
  requests with nothing to clone): unchanged full-file JSON-envelope
  generation, since there's no real working tree for tools to act on.
"""

from __future__ import annotations

import json
import re
import time

import structlog

from src.inference.client import get_inference_client
from src.inference.model_router import resolve_model_name_for_client
from src.orchestrator.context import trim_turns_to_budget
from src.orchestrator.nodes._shared import get_or_clone_workspace
from src.orchestrator.state import (
    AgentPhase,
    AgentState,
    FileChange,
    IterationRecord,
)
from src.orchestrator.tools.impl import dispatch_tool_call, touched_path
from src.orchestrator.tools.protocol import ToolCallError, get_tool_protocol
from src.orchestrator.workspace import Workspace

logger = structlog.get_logger(__name__)

TOOL_CODING_SYSTEM_PROMPT = """\
You are the coding module of Keystone Agents, an autonomous coding agent by Keystone. You are working \
directly inside a real, already-cloned git repository, using tools — you never write a full file from \
memory or guess at what it currently contains.

Rules:
1. Explore before you edit: use list_dir/grep/read_file to see the real current code before changing \
anything. If you already modified a file earlier in this task, re-read it — don't assume you remember its \
exact current content.
2. Make the smallest correct change. Use apply_patch's search/replace mode for most edits (search must match \
the file's real current text exactly); use create=true only for a genuinely new file; use unified_diff only \
for a multi-hunk change you've verified against the real file content.
3. Write complete, production-quality code: proper error handling and type hints, following the surrounding \
codebase's own conventions (check a neighboring file if unsure) — no stubs, no TODOs, no placeholders.
4. If a tool call fails, its error message tells you exactly what went wrong — act on it (re-read the file, \
narrow an ambiguous search, fix a bad argument) rather than repeating the same call.
5. You may run a command (e.g. a quick syntax check) via run_command, but the repo's real test suite runs \
automatically after you finish — don't try to reinvent it.
6. When every necessary change for this task is made, reply with a plain message and NO tool call, summarizing \
what you changed and why. That ends this session — only do this once you are actually done.
"""

STANDALONE_CODING_SYSTEM_PROMPT = """\
You are the coding module of Keystone Agents, an autonomous coding agent by Keystone.

You write production-quality code. Follow these rules:
1. Write complete, working implementations — no stubs, no TODOs, no placeholders.
2. Include proper error handling, type hints, and docstrings.
3. Follow the conventions of the target language and codebase.
4. If modifying existing code, show the complete updated file.
5. If test failures or review comments are provided, address every one.

Respond in JSON:
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "action": "create | modify | delete",
      "language": "python",
      "content": "complete file content here"
    }
  ],
  "explanation": "Brief explanation of changes made",
  "tests_suggested": ["test descriptions if any new tests should be added"]
}
"""


async def coding_node(state: AgentState) -> AgentState:
    """Generate code changes for the current task."""
    if state.repository_url:
        return await _code_with_tools(state)
    return await _code_standalone(state)


# ── Repo mode: real agentic tool-use loop ──────────────────────


def _build_agentic_user_context(state: AgentState) -> str:
    parts = [f"## Task\n{state.task_description}"]

    if state.plan_steps:
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state.plan_steps))
        parts.append(f"\n## Plan\n{plan_text}")
    elif state.plan:
        parts.append(f"\n## Plan\n{state.plan}")

    if state.files_touched:
        already = "\n".join(f"- {p}" for p in state.files_touched)
        parts.append(f"\n## Already modified earlier this task (re-read before editing further)\n{already}")

    if state.test_results and not state.tests_passed:
        parts.append("\n## Test Failures To Fix")
        parts.extend(f"- **{tr.test_name}**: {tr.error[:1000]}" for tr in state.test_results if not tr.passed)

    if state.review_comments and not state.review_passed:
        parts.append("\n## Review Comments To Address")
        for rc in state.review_comments:
            if rc.severity in ("error", "critical"):
                line = f"- [{rc.severity}] {rc.file_path}:{rc.line}: {rc.message}"
                if rc.suggestion:
                    line += f"\n  Suggestion: {rc.suggestion}"
                parts.append(line)

    blocking_quality = [f for f in state.quality_findings if f.tool in ("bandit", "mypy") and f.severity == "error"]
    if blocking_quality:
        parts.append("\n## Quality Gate Findings To Fix (bandit/mypy — blocking)")
        parts.extend(f"- [{f.tool}] {f.path}:{f.line} ({f.code}): {f.message}" for f in blocking_quality)

    if state.memory_context:
        parts.append(f"\n{state.memory_context}")

    if state.rag_context:
        parts.append(f"\n## Codebase Context (retrieved)\n{state.rag_context}")

    parts.append(
        "\n## Repository\nAlready cloned and checked out on your working branch. "
        "Start with list_dir('.') and grep to orient yourself before editing."
    )
    return "\n".join(parts)


async def _run_agentic_loop(state: AgentState, ws: Workspace) -> tuple[bool, str, list[str]]:
    """Returns (model_signaled_done, final_summary_text, paths_touched_this_call)."""
    client = get_inference_client(state.primary_model)
    model_name = await resolve_model_name_for_client(client, state.tenant_id)
    protocol = get_tool_protocol(state.tool_protocol)

    # Turn-structured, not a flat message list: each entry is one or more
    # messages that must move together (an assistant tool_calls message
    # plus its matching tool-result messages), so trim_turns_to_budget can
    # drop the oldest ones wholesale as the conversation grows across
    # max_tool_steps turns without ever splitting a pair — a split would
    # produce a malformed request against a strict OpenAI-compatible
    # backend. Turn 0 (system + task context) is never dropped.
    turns: list[list[dict]] = [
        [
            {"role": "system", "content": TOOL_CODING_SYSTEM_PROMPT + protocol.system_prompt_suffix()},
            {"role": "user", "content": _build_agentic_user_context(state)},
        ]
    ]
    touched: set[str] = set()
    summary = ""
    completed = False

    for _step in range(state.max_tool_steps):
        messages = trim_turns_to_budget(turns, max_tokens=state.max_context_tokens, keep_head_turns=1)
        response = await client.complete(
            messages=messages, temperature=0.15, max_tokens=8192, model_override=model_name, **protocol.request_kwargs()
        )
        usage = response.get("usage", {})
        state.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        message = response["choices"][0]["message"]
        calls = protocol.parse_tool_calls(message)
        turn = [protocol.format_assistant_turn(message, calls)]

        if not calls:
            summary = (message.get("content") or "").strip()
            completed = True
            turns.append(turn)
            break

        for call in calls:
            if isinstance(call, ToolCallError):
                turn.extend(protocol.format_tool_result(call, f"ERROR: {call.error}"))
                continue
            result = await dispatch_tool_call(ws, call.name, call.arguments)
            turn.extend(protocol.format_tool_result(call, result.to_content()))
            path = touched_path(call.name, call.arguments, result)
            if path:
                touched.add(path)
        turns.append(turn)
    else:
        summary = f"Reached the {state.max_tool_steps}-turn tool budget before signaling completion."

    return completed, summary, sorted(touched)


async def _code_with_tools(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    prompt_tokens_before = state.total_prompt_tokens
    completion_tokens_before = state.total_completion_tokens

    try:
        ws = await get_or_clone_workspace(state)
        completed, summary, touched = await _run_agentic_loop(state, ws)

        for path in touched:
            if path not in state.files_touched:
                state.files_touched.append(path)

        if not completed and not touched:
            state.phase = AgentPhase.FAILED
            state.error_message = summary or "The coding agent made no changes and did not signal completion."
        else:
            state.current_plan_step = len(state.plan_steps)
            state.result_summary = summary or f"Modified {len(touched)} file(s)."
            state.phase = AgentPhase.QUALITY

        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="coding",
                model_role=state.primary_model,
                prompt_tokens=state.total_prompt_tokens - prompt_tokens_before,
                completion_tokens=state.total_completion_tokens - completion_tokens_before,
                files_changed=touched,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )
        logger.info(
            "coding.tools_complete",
            task_id=str(state.task_id),
            files_touched=len(touched),
            completed=completed,
            tokens=(state.total_prompt_tokens - prompt_tokens_before)
            + (state.total_completion_tokens - completion_tokens_before),
        )

    except Exception as exc:
        logger.error("coding.tools_failed", error=str(exc), task_id=str(state.task_id))
        state.phase = AgentPhase.FAILED
        state.error_message = f"Coding failed: {exc}"

    return state


# ── Standalone mode: legacy full-file JSON envelope ────────────


async def _code_standalone(state: AgentState) -> AgentState:
    """
    Generate code changes for the current plan step (no repository_url — benchmarks/context-files-only
    tasks). Unchanged from the pre-tools implementation: full-file JSON envelope, one plan step per call.
    """
    t0 = time.monotonic()
    client = get_inference_client(state.primary_model)
    model_name = await resolve_model_name_for_client(client, state.tenant_id)

    user_parts = []

    if state.plan_steps and state.current_plan_step < len(state.plan_steps):
        step_desc = state.plan_steps[state.current_plan_step]
        user_parts.append(f"## Current Step ({state.current_plan_step + 1}/{len(state.plan_steps)})\n{step_desc}")
    else:
        user_parts.append(f"## Task\n{state.task_description}")

    if state.plan:
        user_parts.append(f"\n## Full Plan\n{state.plan}")

    if state.file_changes:
        user_parts.append("\n## Files Already Modified")
        user_parts.extend(
            f"\n### {fc.path} ({fc.action})\n```{fc.language}\n{fc.new_content}\n```" for fc in state.file_changes
        )

    if state.test_results and not state.tests_passed:
        user_parts.append("\n## Test Failures to Fix")
        user_parts.extend(f"- **{tr.test_name}**: {tr.error}" for tr in state.test_results if not tr.passed)

    if state.review_comments and not state.review_passed:
        user_parts.append("\n## Review Comments to Address")
        for rc in state.review_comments:
            if rc.severity in ("error", "critical"):
                user_parts.append(f"- [{rc.severity}] {rc.file_path}:{rc.line}: {rc.message}")
                if rc.suggestion:
                    user_parts.append(f"  Suggestion: {rc.suggestion}")

    if state.memory_context:
        user_parts.append(f"\n{state.memory_context}")

    if state.rag_context:
        user_parts.append(f"\n## Codebase Context\n{state.rag_context}")

    if state.context_files:
        for fname, content in list(state.context_files.items())[:5]:
            truncated = content[:8000] if len(content) > 8000 else content
            user_parts.append(f"\n## Reference: {fname}\n```\n{truncated}\n```")

    messages = [
        {"role": "system", "content": STANDALONE_CODING_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    prompt_tokens = completion_tokens = 0
    new_changes: list[FileChange] = []
    parsed = None

    try:
        response = await client.complete(
            messages=messages, temperature=0.15, max_tokens=16384, model_override=model_name
        )

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        state.add_tokens(prompt_tokens, completion_tokens)

        content = response["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if parsed and "files" in parsed:
            new_changes = [
                FileChange(
                    path=f["path"],
                    action=f.get("action", "create"),
                    language=f.get("language", ""),
                    new_content=f.get("content", ""),
                )
                for f in parsed["files"]
            ]

            existing_paths = {fc.path: i for i, fc in enumerate(state.file_changes)}
            for nc in new_changes:
                if nc.path in existing_paths:
                    idx = existing_paths[nc.path]
                    state.file_changes[idx] = nc
                else:
                    state.file_changes.append(nc)

            state.current_plan_step += 1

            if state.current_plan_step >= len(state.plan_steps):
                # quality_node no-ops straight through for standalone tasks
                # (no repository_url) and decides the real next phase
                # (review/testing/complete) via next_gate_phase — set as a
                # fallback here since nothing else will if it reaches
                # COMPLETE without review/testing ever running.
                state.result_summary = parsed.get("explanation", "Code generation complete.")
                state.phase = AgentPhase.QUALITY
            else:
                state.phase = AgentPhase.CODING

        else:
            logger.warning("coding.no_files_in_response", task_id=str(state.task_id))
            state.phase = AgentPhase.FAILED
            state.error_message = "Coding model did not return any file changes."

        files_changed = [fc.path for fc in (new_changes if parsed else [])]
        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="coding",
                model_role=state.primary_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                files_changed=files_changed,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )

        logger.info(
            "coding.complete",
            task_id=str(state.task_id),
            files=len(files_changed),
            step=state.current_plan_step,
            tokens=prompt_tokens + completion_tokens,
        )

    except Exception as exc:
        logger.error("coding.failed", error=str(exc), task_id=str(state.task_id))
        state.phase = AgentPhase.FAILED
        state.error_message = f"Coding failed: {exc}"

    return state


def _extract_json(text: str) -> dict | None:
    """Try to extract JSON from a response that may contain markdown fences."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    patterns = [
        r"```json\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, IndexError):
                continue

    return None
