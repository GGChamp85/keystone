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

    model_id = "test-coding-model"  # real InferenceClient attribute — resolve_model_name_for_client reads it

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._script:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        return self._script.pop(0)

    async def stream_to_message(self, messages, on_text=None, **kwargs):
        # The loop streams each turn by default; the scripted client answers the same way either way.
        return await self.complete(messages, **kwargs)


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
async def test_agentic_loop_includes_matching_skills_in_the_model_prompt(repo_workspace):
    """Real proof that src/orchestrator/engine.py's skill matching (state.skills_context) actually
    reaches the model's prompt — the exact rendered text from memory.skills.render_skills_for_prompt
    must appear in what the scripted client received, mirroring the memory test above."""
    _task_id, ws = repo_workspace
    state = _make_state()
    state.skills_context = "## Skills (standing instructions for this kind of task)\n- PCI checklist: Run it."

    script = ScriptedClient([_final_response("Nothing to do here.")])

    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        await coding_node(state)

    sent_text = "\n".join(m.get("content") or "" for call in script.calls for m in call["messages"])
    assert "PCI checklist: Run it." in sent_text


@requires_sandbox
async def test_agentic_loop_applies_a_queued_steering_message_as_a_new_turn_then_stops_resending_it(repo_workspace):
    """Real proof of src/orchestrator/steering.py's contract: a message queued via
    enqueue_steering_message (the exact function POST /v1/keystone/tasks/{id}/steer calls) is
    injected as a whole new user turn with the expected header on the very next model call, and
    is never sent again once drained — proving the queue is really cleared, not resent every
    iteration."""
    from src.orchestrator.steering import enqueue_steering_message

    _task_id, ws = repo_workspace
    state = _make_state()
    await enqueue_steering_message(state.task_id, "also add a docstring to greet()")

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
        await coding_node(state)

    assert len(script.calls) == 2
    first_call_text = "\n".join(m.get("content") or "" for m in script.calls[0]["messages"])
    second_call_text = "\n".join(m.get("content") or "" for m in script.calls[1]["messages"])
    assert "## New instructions from the user" in first_call_text
    assert "also add a docstring to greet()" in first_call_text
    # The injected turn legitimately stays in conversation history on the second call (it's a
    # real turn now, same as any other) — what must NOT happen is a *second* injection of the
    # same message, which would mean the queue wasn't really cleared by the first drain.
    assert second_call_text.count("also add a docstring to greet()") == 1


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


@requires_sandbox
async def test_agentic_loop_routes_to_the_tenants_promoted_adapter(repo_workspace):
    """Real end-to-end proof that src/inference/model_router.py's adapter
    routing actually reaches the coding node's real inference call — not
    just that resolve_model_name_for_client works in isolation
    (tests/test_model_router_adapters.py already covers that): a real
    Postgres-backed Tenant + promoted ModelAdapter, and the scripted
    client's own model_id set to match the adapter's base_model_id, so
    the *only* way model_override could come out right is if coding.py
    really called resolve_model_name_for_client with this tenant's id."""
    from src.db.connection import get_db_context
    from src.db.models import AdapterStatus, ModelAdapter, Tenant, TenantTier

    _task_id, ws = repo_workspace
    state = _make_state()
    tenant_id = uuid.uuid4()

    async with get_db_context() as db:
        db.add(Tenant(id=tenant_id, name="adapter-routing-test", email=f"{tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(
            ModelAdapter(
                tenant_id=tenant_id,
                base_model_id="test-coding-model",  # matches ScriptedClient.model_id
                name="tenant-coding-lora-v1",
                path="/data/adapters/tenant-coding-lora-v1",
                rank=64,
                job_type="lora",
                status=AdapterStatus.PROMOTED,
                is_default=True,
            )
        )
        await db.flush()

    try:
        state.tenant_id = tenant_id
        script = ScriptedClient([_final_response("No changes needed.")])

        with (
            patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
            patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
        ):
            await coding_node(state)

        assert script.calls[0]["kwargs"]["model_override"] == "tenant-coding-lora-v1"
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, tenant_id)
            if row is not None:
                await db.delete(row)


@requires_sandbox
async def test_agentic_loop_records_every_step_in_full_on_the_iteration_record(repo_workspace):
    """The durable copy of the live trace: each tool call, its complete result, and the model's
    final message land on the coding IterationRecord's `steps`, in order, untruncated."""
    _task_id, ws = repo_workspace
    state = _make_state()
    script = ScriptedClient(
        [
            _tool_call_response("read_file", {"path": "app.py"}),
            _tool_call_response("apply_patch", {"path": "app.py", "search": "hello {name}", "replace": "HI {name}"}),
            _final_response("Changed the greeting."),
        ]
    )
    with (
        patch("src.orchestrator.nodes.coding.get_inference_client", return_value=script),
        patch("src.orchestrator.nodes._shared.Workspace", return_value=ws),
    ):
        result = await coding_node(state)

    steps = result.trace[-1].steps
    assert [s["event_type"] for s in steps] == ["tool_call", "tool_result", "tool_call", "tool_result", "model_text"]
    assert steps[0]["name"] == "read_file" and steps[0]["arguments"] == {"path": "app.py"}
    assert steps[1]["ok"] is True and steps[1]["output"] == "def greet(name):\n    return f'hello {name}'\n"
    assert steps[3]["ok"] is True and "Updated app.py" in steps[3]["output"]
    assert steps[4]["content"] == "Changed the greeting." and steps[4]["final"] is True
    assert all(s["node"] == "coding" and s["timestamp"] > 0 for s in steps)
