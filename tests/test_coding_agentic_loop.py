# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for nodes/coding.py's repo-mode agentic tool-use loop — mechanics only. Uses a real
sandbox and a real cloned/git-initialized working tree (same fixture pattern as test_tool_impl.py) with a
scripted fake InferenceClient standing in for the model's own creative output, the same "scripted policy"
pattern used elsewhere this session (tests/fixtures/fake_vllm_server.py, tests/test_git_workflow_integration.py)
— real tool dispatch, real file I/O, real turn-budget/error-recovery behavior; only the model's replies are
pre-scripted. Never a substitute for verifying model quality itself, which needs a real GPU/vLLM endpoint.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import pytest

from src.orchestrator.nodes.coding import coding_node
from src.orchestrator.state import AgentPhase, AgentState
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import SandboxManager

pytestmark = pytest.mark.integration

requires_sandbox = pytest.mark.skipif(
    not os.environ.get("SANDBOX_DAEMON_URL"),
    reason="Needs SANDBOX_DAEMON_URL pointed at a running keystoned — see tests/test_tool_impl.py",
)


class ScriptedClient:
    """Stands in for InferenceClient — returns pre-scripted `complete()` responses in order, one per call."""

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._script:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        return self._script.pop(0)


def _tool_call_response(name: str, arguments: dict, prompt_tokens=50, completion_tokens=20) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _final_response(content: str, prompt_tokens=50, completion_tokens=20) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


@pytest.fixture
async def repo_workspace():
    manager = SandboxManager()
    task_id = f"coding-loop-test-{uuid.uuid4().hex[:8]}"
    ws = Workspace(manager, task_id=task_id)
    await ws.ensure_sandbox(network_enabled=False)
    await ws.run("mkdir -p /workspace/repo", cwd="/")
    await ws.write_file("app.py", "def greet(name):\n    return f'hello {name}'\n")
    await ws.run(
        "git init -q && git -c user.name=test -c user.email=test@test.dev add -A "
        "&& git -c user.name=test -c user.email=test@test.dev commit -q -m init"
    )
    try:
        yield task_id, ws
    finally:
        await ws.close()


def _make_state() -> AgentState:
    """
    `repo_cloned=True` plus patching `_shared.Workspace` to return the
    fixture's already-cloned `ws` means `get_or_clone_workspace` never
    actually clones `repository_url` — it's a placeholder, never dialed.
    """
    state = AgentState()
    state.task_description = "Change the greeting to say HI instead of hello."
    state.repository_url = "https://example.invalid/scripted/repo.git"
    state.repo_cloned = True
    state.enable_reasoning_review = False
    state.enable_sandbox_testing = False
    state.max_tool_steps = 5
    return state


@requires_sandbox
async def test_agentic_loop_applies_a_real_patch_and_completes(repo_workspace):
    _task_id, ws = repo_workspace
    state = _make_state()

    script = ScriptedClient(
        [
            _tool_call_response("apply_patch", {"path": "app.py", "search": "hello {name}", "replace": "HI {name}"}),
            _final_response("Changed the greeting from hello to HI."),
        ]
    )

    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        result = await coding_node(state)

    assert result.phase == AgentPhase.QUALITY
    assert "app.py" in result.files_touched
    assert result.result_summary == "Changed the greeting from hello to HI."
    content = (await ws.read_file("app.py")).strip()
    assert "HI {name}" in content or "HI " in content


@requires_sandbox
async def test_agentic_loop_includes_recalled_memory_in_the_model_prompt(repo_workspace):
    """Real proof that src/orchestrator/engine.py's memory recall (state.memory_context) actually reaches the
    model's prompt, not just that the plumbing compiles — the exact rendered text from
    memory.store.render_memories_for_prompt must appear in what the scripted client received."""
    _task_id, ws = repo_workspace
    state = _make_state()
    state.memory_context = (
        "## Team Memory (learned conventions, preferences, and things to avoid)\n"
        "- [avoid/repo] Never touch the legacy billing module."
    )

    script = ScriptedClient([_final_response("Nothing to do here.")])

    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        await coding_node(state)

    sent_text = "\n".join(m.get("content") or "" for call in script.calls for m in call["messages"])
    assert "Never touch the legacy billing module" in sent_text


@requires_sandbox
async def test_agentic_loop_recovers_from_a_bad_search_and_still_completes(repo_workspace):
    _task_id, ws = repo_workspace
    state = _make_state()

    script = ScriptedClient(
        [
            # First attempt: search text that doesn't exist — the tool
            # returns an actionable error, and the model (scripted here to
            # mimic a real self-correction) retries with the right text.
            _tool_call_response("apply_patch", {"path": "app.py", "search": "NOPE NOT THERE", "replace": "x"}),
            _tool_call_response("apply_patch", {"path": "app.py", "search": "hello {name}", "replace": "HI {name}"}),
            _final_response("Fixed after the first search didn't match."),
        ]
    )

    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        result = await coding_node(state)

    assert result.phase == AgentPhase.QUALITY
    assert "app.py" in result.files_touched
    # The tool's error message for the first (bad) call should have reached the model as a tool result.
    tool_messages = [m for call in script.calls for m in call["messages"] if m.get("role") == "tool"]
    assert any("not found" in (m.get("content") or "") for m in tool_messages)


@requires_sandbox
async def test_agentic_loop_fails_cleanly_when_budget_exhausted_with_no_changes(repo_workspace):
    _task_id, ws = repo_workspace
    state = _make_state()
    state.max_tool_steps = 2

    # The model keeps asking to read a file that doesn't exist, never
    # editing anything and never signaling completion — should exhaust the
    # turn budget and fail cleanly rather than loop forever or crash.
    script = ScriptedClient(
        [
            _tool_call_response("read_file", {"path": "does_not_exist.py"}),
            _tool_call_response("read_file", {"path": "still_does_not_exist.py"}),
        ]
    )

    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        result = await coding_node(state)

    assert result.phase == AgentPhase.FAILED
    assert result.files_touched == []
    assert result.error_message


@requires_sandbox
async def test_agentic_loop_trims_old_turns_once_the_context_budget_is_exceeded(repo_workspace):
    """Real end-to-end proof that context.trim_turns_to_budget is actually wired into the loop: each
    run_command call here returns several thousand tokens of output, so an unbounded conversation would grow
    turn over turn — with a small max_context_tokens, later turns must NOT carry every earlier turn's full
    output, and a drop notice must appear once trimming has happened."""
    _task_id, ws = repo_workspace
    state = _make_state()
    state.max_tool_steps = 6
    state.max_context_tokens = 600  # deliberately tiny — forces trimming after just one or two big turns

    big_output_cmd = "python3 -c \"print('word ' * 2000)\""
    script = ScriptedClient(
        [
            _tool_call_response("run_command", {"command": big_output_cmd}),
            _tool_call_response("run_command", {"command": big_output_cmd}),
            _tool_call_response("run_command", {"command": big_output_cmd}),
            _final_response("Done exploring."),
        ]
    )

    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        result = await coding_node(state)

    assert result.phase == AgentPhase.QUALITY

    # The 4th call (final) is the one most likely to have been trimmed —
    # by then 3 big tool-result turns exist, well over the 600-token budget.
    last_call_messages = script.calls[-1]["messages"]
    tool_result_count = sum(1 for m in last_call_messages if m.get("role") == "tool")
    assert tool_result_count < 3, "expected at least one earlier big tool-result turn to have been dropped"
    assert any("dropped" in (m.get("content") or "") for m in last_call_messages)
