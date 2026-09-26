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

from src.config import get_settings
from src.inference.client import get_inference_client
from src.inference.model_router import resolve_model_name_for_client
from src.orchestrator.best_of_n import run_best_of_n
from src.orchestrator.context import Summarizer, fit_to_tokens, trim_turns_to_budget_async
from src.orchestrator.events import publish_task_step
from src.orchestrator.nodes._shared import ensure_repo_map, get_or_clone_workspace
from src.orchestrator.nodes.quality import _is_blocking
from src.orchestrator.state import (
    AgentPhase,
    AgentState,
    FileChange,
    IterationRecord,
)
from src.orchestrator.steering import drain_steering_messages
from src.orchestrator.tools.impl import ToolResult, dispatch_tool_call, touched_path
from src.orchestrator.tools.protocol import ToolCallError, get_tool_protocol
from src.orchestrator.workspace import Workspace
from src.security.prompt_injection import wrap_tool_output

logger = structlog.get_logger(__name__)

TOOL_CODING_SYSTEM_PROMPT = """\
You are the coding module of Keystone Agents, an autonomous coding agent by Keystone. You are working \
directly inside a real, already-cloned git repository, using tools — you never write a full file from \
memory or guess at what it currently contains.

Rules:
1. Explore before you edit: the repository map (get_repo_map) tells you which file and line a symbol lives \
at; read_file with start_line/end_line shows exactly that region; grep finds every call site. See the real \
current code before changing anything. If you already modified a file earlier in this task, re-read it — \
don't assume you remember its exact current content.
2. Make the smallest correct change. Use apply_patch's search/replace mode for most edits (copy the file's \
real current text as `search`; a whitespace-only or near-identical mismatch is tolerated when unambiguous, \
and the result tells you when that happened); use create=true only for a genuinely new file; use \
unified_diff only for a multi-hunk change you've verified against the real file content.
3. Write complete, production-quality code: proper error handling and type hints, following the surrounding \
codebase's own conventions (check a neighboring file if unsure) — no stubs, no TODOs, no placeholders.
4. If a tool call fails, its error message tells you exactly what went wrong — act on it (re-read the file, \
narrow an ambiguous search, fix a bad argument) rather than repeating the same call.
5. Verify before you finish: after your edits, call run_tests (scope='related' for fast feedback on the \
files you changed; scope='all' once before you're done) and fix what fails. The full suite also runs \
automatically after you finish, and a failure there costs a whole extra round — catch it here first.
6. When every necessary change for this task is made and run_tests passes, reply with a plain message and NO \
tool call, summarizing what you changed and why. That ends this session — only do this once you are actually done.
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


def _prompt_budget(state: AgentState) -> int:
    """Tokens one section of the task context may take: half the loop's context budget, so the task, the
    map and the conversation always fit alongside it. Derived from configuration, never a fixed number."""
    return max(2_000, state.max_context_tokens // 2)


def _with_image_parts(text: str, images: list[dict[str, str]]) -> str | list[dict]:
    """OpenAI-style content-part list when the task has attached images, plain text otherwise —
    only a vision-capable backend (src/inference/catalog.py's `vision` flag) does anything with
    the image_url parts; a text-only backend would either ignore or error on them, which is why
    the task is refused up front in agents.py if the resolved role isn't vision-capable."""
    if not images:
        return text
    parts: list[dict] = [{"type": "text", "text": text}]
    parts.extend(
        {"type": "image_url", "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"}} for img in images
    )
    return parts


def _build_agentic_user_context(state: AgentState) -> str | list[dict]:
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
        budget = _prompt_budget(state)
        parts.extend(
            f"- **{tr.test_name}**:\n{fit_to_tokens(tr.error or tr.output, budget, keep='tail', what='test output')}"
            for tr in state.test_results
            if not tr.passed
        )

    if state.review_comments and not state.review_passed:
        parts.append("\n## Review Comments To Address")
        for rc in state.review_comments:
            if rc.severity in ("error", "critical"):
                line = f"- [{rc.severity}] {rc.file_path}:{rc.line}: {rc.message}"
                if rc.suggestion:
                    line += f"\n  Suggestion: {rc.suggestion}"
                parts.append(line)

    blocking_quality = [f for f in state.quality_findings if _is_blocking(f, state.quality_blocking_tools)]
    if blocking_quality:
        tools = "/".join(state.quality_blocking_tools)
        parts.append(f"\n## Quality Gate Findings To Fix ({tools} — blocking)")
        parts.extend(f"- [{f.tool}] {f.path}:{f.line} ({f.code}): {f.message}" for f in blocking_quality)

    if state.memory_context:
        parts.append(f"\n{state.memory_context}")

    if state.skills_context:
        parts.append(f"\n{state.skills_context}")

    if state.rag_context:
        parts.append(f"\n## Codebase Context (retrieved)\n{state.rag_context}")

    if state.deps_install_error:
        parts.append(
            "\n## Dependency install (automatic, after clone) FAILED\n"
            f"{fit_to_tokens(state.deps_install_error, _prompt_budget(state), keep='tail', what='install output')}\n"
            "A test that fails on a missing import may be failing for this reason, not because of your change."
        )

    if state.repo_map:
        parts.append(f"\n## Repository map\n{state.repo_map}")
        parts.append(
            "\n## Repository\nAlready cloned and checked out on your working branch. The map above shows "
            "where the relevant symbols live — go to those files (read_file with a line range) rather than "
            "exploring from scratch; use grep to confirm call sites before changing a signature."
        )
    else:
        parts.append(
            "\n## Repository\nAlready cloned and checked out on your working branch. "
            "Start with get_repo_map, then list_dir/grep to orient yourself before editing."
        )
    return _with_image_parts("\n".join(parts), state.images)


_TOOL_DETAIL_ARG: dict[str, str] = {
    "read_file": "path",
    "list_dir": "path",
    "get_repo_map": "max_tokens",
    "apply_patch": "path",
    "grep": "pattern",
    "run_command": "command",
    "run_tests": "scope",
}


def _wrapped_tool_content(name: str, arguments: dict, result: ToolResult) -> str:
    """
    Every successful tool result reflects content Keystone doesn't control
    the origin of (a file, a grep match, command output) — wrapped in a
    real untrusted-data delimiter (src/security/prompt_injection.py)
    before it becomes part of the model's context, real defense-in-depth
    against a repository that tries to smuggle instructions through
    exactly this channel. Error messages are Keystone's own generated
    text, not repository content, so they pass through unwrapped.
    """
    if not result.ok:
        return result.to_content()
    detail = str(arguments.get(_TOOL_DETAIL_ARG.get(name, ""), ""))
    return wrap_tool_output(result.output, tool=name, detail=detail)


_SUMMARY_SYSTEM_PROMPT = """\
You are compressing the earlier part of a coding agent's tool-use session so the agent can continue with \
less context. Write a dense, factual summary (at most ~300 words) covering: files read and what was learned \
about them (paths, key symbols, line numbers); edits already applied (file and what changed); command and \
test results seen; open problems. Facts only — no preamble, no advice.
"""


def _render_messages_for_summary(messages: list[dict], *, budget_tokens: int) -> str:
    """The dropped turns as a transcript for the summariser, fitted to ITS context budget (every message
    gets an equal share, head-kept so tool calls and the start of each result survive)."""
    per_message = max(200, budget_tokens // max(1, len(messages)))
    parts = []
    for message in messages:
        content = message.get("content") or ""
        if message.get("tool_calls"):
            calls = [
                {"name": tc.get("function", {}).get("name"), "arguments": tc.get("function", {}).get("arguments")}
                for tc in message["tool_calls"]
            ]
            content = f"{content}\n{json.dumps(calls)}".strip()
        parts.append(f"[{message.get('role')}] {fit_to_tokens(content, per_message, what='message')}")
    return "\n\n".join(parts)


def _make_turn_summarizer(state: AgentState) -> Summarizer:
    """The real model call behind context.py's summarise-on-overflow: the cheaper coding_fallback role
    condenses the turns being dropped; its tokens count against the task like any other call."""

    async def summarize(messages: list[dict]) -> str:
        client = get_inference_client("coding_fallback")
        model_name = await resolve_model_name_for_client(client, state.tenant_id)
        response = await client.complete(
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _render_messages_for_summary(messages, budget_tokens=state.max_context_tokens),
                },
            ],
            temperature=0.0,
            max_tokens=600,
            model_override=model_name,
        )
        usage = response.get("usage", {})
        state.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return response["choices"][0]["message"].get("content") or ""

    return summarize


async def _run_agentic_loop(state: AgentState, ws: Workspace) -> tuple[bool, str, list[str], list[dict]]:
    """Returns (model_signaled_done, final_summary_text, paths_touched_this_call, step_events)."""
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
    steps: list[dict] = []

    async def step(event_type: str, payload: dict) -> None:
        steps.append(await publish_task_step(state.task_id, event_type, "coding", payload, phase="coding"))

    summarizer = _make_turn_summarizer(state)
    stream_turns = get_settings().agent_stream_turns

    for _step in range(state.max_tool_steps):
        steering_messages = await drain_steering_messages(state.task_id)
        if steering_messages:
            joined = "\n".join(f"- {m}" for m in steering_messages)
            header = "## New instructions from the user (received while this task was running)"
            turns.append([{"role": "user", "content": f"{header}\n{joined}"}])
            await step("steering", {"turn": _step, "messages": steering_messages, "status": "applied"})

        messages = await trim_turns_to_budget_async(
            turns, max_tokens=state.max_context_tokens, keep_head_turns=1, summarizer=summarizer
        )
        request: dict = dict(
            messages=messages, temperature=0.15, max_tokens=8192, model_override=model_name, **protocol.request_kwargs()
        )
        # Streamed by default: tool calls and text are reassembled from deltas into the same
        # response shape, so a long turn shows progress instead of sitting on a read timeout.
        response = await (client.stream_to_message(**request) if stream_turns else client.complete(**request))
        usage = response.get("usage", {})
        state.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        message = response["choices"][0]["message"]
        calls = protocol.parse_tool_calls(message)
        turn = [protocol.format_assistant_turn(message, calls)]
        if message.get("content"):
            await step("model_text", {"turn": _step, "content": message["content"], "final": not calls})

        if not calls:
            summary = (message.get("content") or "").strip()
            completed = True
            turns.append(turn)
            break

        for call in calls:
            if isinstance(call, ToolCallError):
                await step("tool_call", {"turn": _step, "name": getattr(call, "name", ""), "error": call.error})
                turn.extend(protocol.format_tool_result(call, f"ERROR: {call.error}"))
                continue
            await step("tool_call", {"turn": _step, "name": call.name, "arguments": call.arguments})
            result = await dispatch_tool_call(
                ws, call.name, call.arguments, touched_paths=sorted(touched | set(state.files_touched))
            )
            await step(
                "tool_result",
                {"turn": _step, "name": call.name, "ok": result.ok, "output": result.output, "error": result.error},
            )
            turn.extend(protocol.format_tool_result(call, _wrapped_tool_content(call.name, call.arguments, result)))
            path = touched_path(call.name, call.arguments, result)
            if path:
                touched.add(path)
        turns.append(turn)
    else:
        summary = f"Reached the {state.max_tool_steps}-turn tool budget before signaling completion."

    return completed, summary, sorted(touched), steps


async def _code_with_tools(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    prompt_tokens_before = state.total_prompt_tokens
    completion_tokens_before = state.total_completion_tokens

    try:
        ws = await get_or_clone_workspace(state)
        await ensure_repo_map(state, ws)
        if state.best_of_n > 1:
            completed, summary, touched, steps = await run_best_of_n(
                state, ws, state.best_of_n, run_loop=_run_agentic_loop
            )
        else:
            completed, summary, touched, steps = await _run_agentic_loop(state, ws)

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
                steps=steps,
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
        per_file = max(1_000, _prompt_budget(state) // max(1, len(state.context_files)))
        for fname, content in state.context_files.items():
            user_parts.append(f"\n## Reference: {fname}\n```\n{fit_to_tokens(content, per_file, what=fname)}\n```")

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
