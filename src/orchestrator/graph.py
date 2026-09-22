# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — LangGraph agent graph.

Defines the full agent execution graph:
  PLANNING → CODING → QUALITY → (REVIEW) → (TESTING) → COMPLETE
                  ↑         ↓        ↓          ↓            ↓
                  └── FIXING ←───────┴──────────┴────────────┘

Each node transforms AgentState and sets the next phase.
The circuit breaker is checked before every node execution.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.orchestrator.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerTripped,
)
from src.orchestrator.nodes.coding import coding_node
from src.orchestrator.nodes.planning import planning_node
from src.orchestrator.nodes.quality import quality_node
from src.orchestrator.nodes.review import review_node
from src.orchestrator.nodes.testing import testing_node
from src.orchestrator.nodes.tool_execution import fixing_node
from src.orchestrator.state import AgentPhase, AgentState

logger = structlog.get_logger(__name__)


def _phase_router(state: dict[str, Any]) -> str:
    """
    Conditional edge: decide next node based on current phase.
    """
    phase = state.get("phase", AgentPhase.PLANNING)

    if isinstance(phase, str):
        phase = AgentPhase(phase)

    routing = {
        AgentPhase.PLANNING: "planning",
        AgentPhase.CODING: "coding",
        AgentPhase.QUALITY: "quality",
        AgentPhase.REVIEW: "review",
        AgentPhase.TESTING: "testing",
        AgentPhase.FIXING: "fixing",
        AgentPhase.COMPLETE: END,
        AgentPhase.FAILED: END,
        AgentPhase.CANCELLED: END,
    }
    return routing.get(phase, END)


HeartbeatCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _wrap_with_circuit_breaker(
    node_fn,
    state_dict: dict[str, Any],
    breaker: CircuitBreaker,
    on_iteration: HeartbeatCallback | None = None,
) -> dict[str, Any]:
    """
    Wrap a node function with circuit breaker checks.
    Converts between dict (LangGraph format) and AgentState.
    """
    state = _dict_to_state(state_dict)

    try:
        await breaker.check(state)
    except CircuitBreakerTripped as exc:
        logger.error(
            "circuit_breaker.tripped",
            reason=exc.reason,
            task_id=str(state.task_id),
            iteration=state.iteration,
            total_tokens=state.total_tokens,
        )
        state.phase = AgentPhase.FAILED
        state.error_message = f"Circuit breaker: {exc.reason}"
        return _state_to_dict(state)

    try:
        state = await node_fn(state)
    except Exception as exc:
        logger.error(
            "node.unhandled_error",
            node=node_fn.__name__,
            error=str(exc),
            task_id=str(state.task_id),
        )
        state.phase = AgentPhase.FAILED
        state.error_message = f"Unhandled error in {node_fn.__name__}: {exc}"

    result = _state_to_dict(state)

    # Let the caller (e.g. a Temporal activity) know execution is still
    # progressing after every node, so a crash-detection timeout doesn't
    # fire during a long-running planning/coding/testing step.
    if on_iteration is not None:
        try:
            await on_iteration(node_fn.__name__, result)
        except Exception as exc:
            logger.warning("graph.heartbeat_failed", node=node_fn.__name__, error=str(exc))

    return result


def build_agent_graph(
    breaker_config: CircuitBreakerConfig | None = None,
    on_iteration: HeartbeatCallback | None = None,
) -> CompiledStateGraph:
    """
    Build and compile the Keystone Agents agent graph.

    `on_iteration`, if given, is awaited after every node completes — this is
    the hook Temporal-backed execution uses to heartbeat, so a 30-minute
    agent task doesn't trip Temporal's own crash-detection timeout just
    because a single LLM call inside one node takes a while.
    """
    if breaker_config is None:
        breaker_config = CircuitBreakerConfig()

    breaker = CircuitBreaker(breaker_config)
    breaker.start()

    # The state is the plain dict `_state_to_dict` produces from AgentState (the
    # nodes read/write it by key); LangGraph's generics want a TypedDict/model
    # class here, but a plain dict is what actually flows, so it's annotated as such.
    graph: Any = StateGraph(dict)

    # ── Nodes (each wrapped with circuit breaker) ─────────────

    async def plan_node(state: dict) -> dict:
        return await _wrap_with_circuit_breaker(planning_node, state, breaker, on_iteration)

    async def code_node(state: dict) -> dict:
        return await _wrap_with_circuit_breaker(coding_node, state, breaker, on_iteration)

    async def qual_node(state: dict) -> dict:
        return await _wrap_with_circuit_breaker(quality_node, state, breaker, on_iteration)

    async def rev_node(state: dict) -> dict:
        return await _wrap_with_circuit_breaker(review_node, state, breaker, on_iteration)

    async def test_node(state: dict) -> dict:
        return await _wrap_with_circuit_breaker(testing_node, state, breaker, on_iteration)

    async def fix_node(state: dict) -> dict:
        return await _wrap_with_circuit_breaker(fixing_node, state, breaker, on_iteration)

    graph.add_node("planning", plan_node)
    graph.add_node("coding", code_node)
    graph.add_node("quality", qual_node)
    graph.add_node("review", rev_node)
    graph.add_node("testing", test_node)
    graph.add_node("fixing", fix_node)

    # ── Edges ─────────────────────────────────────────────────

    graph.set_entry_point("planning")

    # After each node, route based on the phase set by that node
    for node_name in ["planning", "coding", "quality", "review", "testing", "fixing"]:
        graph.add_conditional_edges(node_name, _phase_router)

    return graph.compile()


# ── State conversion helpers ──────────────────────────────────


def _state_to_dict(state: AgentState) -> dict[str, Any]:
    """Convert AgentState dataclass to a dict for LangGraph."""
    return {
        "task_id": str(state.task_id),
        "tenant_id": str(state.tenant_id) if state.tenant_id else None,
        "api_key_id": str(state.api_key_id) if state.api_key_id else None,
        "user_slug": state.user_slug,
        "task_description": state.task_description,
        "repository_url": state.repository_url,
        "branch": state.branch,
        "target_files": state.target_files,
        "context_files": state.context_files,
        "phase": state.phase.value if isinstance(state.phase, AgentPhase) else state.phase,
        "iteration": state.iteration,
        "max_iterations": state.max_iterations,
        "primary_model": state.primary_model,
        "enable_reasoning_review": state.enable_reasoning_review,
        "enable_sandbox_testing": state.enable_sandbox_testing,
        "enable_quality_gates": state.enable_quality_gates,
        "plan": state.plan,
        "plan_steps": state.plan_steps,
        "current_plan_step": state.current_plan_step,
        "file_changes": [
            {"path": fc.path, "action": fc.action, "language": fc.language, "new_content": fc.new_content}
            for fc in state.file_changes
        ],
        "review_comments": [
            {
                "file_path": rc.file_path,
                "line": rc.line,
                "severity": rc.severity,
                "message": rc.message,
                "suggestion": rc.suggestion,
            }
            for rc in state.review_comments
        ],
        "review_passed": state.review_passed,
        "consecutive_review_failures": state.consecutive_review_failures,
        "test_results": [
            {"test_name": tr.test_name, "passed": tr.passed, "output": tr.output, "error": tr.error}
            for tr in state.test_results
        ],
        "tests_passed": state.tests_passed,
        "consecutive_test_failures": state.consecutive_test_failures,
        "quality_findings": [
            {
                "tool": f.tool,
                "path": f.path,
                "line": f.line,
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
            }
            for f in state.quality_findings
        ],
        "consecutive_quality_failures": state.consecutive_quality_failures,
        "sandbox_id": state.sandbox_id,
        "sandbox_output": state.sandbox_output,
        "repo_cloned": state.repo_cloned,
        "working_branch": state.working_branch,
        "commit_sha": state.commit_sha,
        "pr_url": state.pr_url,
        "pr_number": state.pr_number,
        "max_tool_steps": state.max_tool_steps,
        "tool_protocol": state.tool_protocol,
        "files_touched": state.files_touched,
        "max_context_tokens": state.max_context_tokens,
        "root_cause_notes": state.root_cause_notes,
        "total_prompt_tokens": state.total_prompt_tokens,
        "total_completion_tokens": state.total_completion_tokens,
        "max_tokens_per_task": state.max_tokens_per_task,
        "trace": [
            {
                "iteration": r.iteration,
                "phase": r.phase,
                "model_role": r.model_role,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "files_changed": r.files_changed,
                "test_results": r.test_results,
                "review_comments": r.review_comments,
                "quality_findings": r.quality_findings,
                "error": r.error,
                "duration_ms": r.duration_ms,
            }
            for r in state.trace
        ],
        "messages": state.messages,
        "result_summary": state.result_summary,
        "error_message": state.error_message,
        "rag_context": state.rag_context,
        "memory_context": state.memory_context,
    }


def _dict_to_state(d: dict[str, Any]) -> AgentState:
    """Convert a dict back to AgentState."""
    from uuid import UUID as _UUID

    from src.orchestrator.state import (
        FileChange,
        IterationRecord,
        QualityFinding,
        ReviewComment,
        TestResult,
    )

    state = AgentState()
    state.task_id = _UUID(d["task_id"]) if isinstance(d.get("task_id"), str) else d.get("task_id", state.task_id)
    state.tenant_id = _UUID(d["tenant_id"]) if d.get("tenant_id") else None
    state.api_key_id = _UUID(d["api_key_id"]) if d.get("api_key_id") else None
    state.user_slug = d.get("user_slug")
    state.task_description = d.get("task_description", "")
    state.repository_url = d.get("repository_url")
    state.branch = d.get("branch", "main")
    state.target_files = d.get("target_files", [])
    state.context_files = d.get("context_files", {})

    phase_val = d.get("phase", "planning")
    state.phase = AgentPhase(phase_val) if isinstance(phase_val, str) else phase_val

    state.iteration = d.get("iteration", 0)
    state.max_iterations = d.get("max_iterations", 15)
    state.primary_model = d.get("primary_model", "coding")
    state.enable_reasoning_review = d.get("enable_reasoning_review", True)
    state.enable_sandbox_testing = d.get("enable_sandbox_testing", True)
    state.enable_quality_gates = d.get("enable_quality_gates", True)
    state.plan = d.get("plan", "")
    state.plan_steps = d.get("plan_steps", [])
    state.current_plan_step = d.get("current_plan_step", 0)

    state.file_changes = [
        FileChange(
            path=fc["path"],
            action=fc.get("action", "create"),
            language=fc.get("language", ""),
            new_content=fc.get("new_content", ""),
        )
        for fc in d.get("file_changes", [])
    ]

    state.review_comments = [
        ReviewComment(
            file_path=rc["file_path"],
            line=rc.get("line"),
            severity=rc.get("severity", "info"),
            message=rc.get("message", ""),
            suggestion=rc.get("suggestion"),
        )
        for rc in d.get("review_comments", [])
    ]
    state.review_passed = d.get("review_passed", False)
    state.consecutive_review_failures = d.get("consecutive_review_failures", 0)

    state.test_results = [
        TestResult(
            test_name=tr["test_name"], passed=tr["passed"], output=tr.get("output", ""), error=tr.get("error", "")
        )
        for tr in d.get("test_results", [])
    ]
    state.tests_passed = d.get("tests_passed", False)
    state.consecutive_test_failures = d.get("consecutive_test_failures", 0)

    state.quality_findings = [
        QualityFinding(
            tool=f["tool"],
            path=f.get("path", ""),
            line=f.get("line"),
            severity=f.get("severity", "warning"),
            code=f.get("code", ""),
            message=f.get("message", ""),
        )
        for f in d.get("quality_findings", [])
    ]
    state.consecutive_quality_failures = d.get("consecutive_quality_failures", 0)

    state.sandbox_id = d.get("sandbox_id")
    state.sandbox_output = d.get("sandbox_output", "")
    state.repo_cloned = d.get("repo_cloned", False)
    state.working_branch = d.get("working_branch", "")
    state.commit_sha = d.get("commit_sha")
    state.pr_url = d.get("pr_url")
    state.pr_number = d.get("pr_number")
    state.max_tool_steps = d.get("max_tool_steps", 25)
    state.tool_protocol = d.get("tool_protocol", "native")
    state.files_touched = d.get("files_touched", [])
    state.max_context_tokens = d.get("max_context_tokens", 24_000)
    state.root_cause_notes = d.get("root_cause_notes", [])
    state.total_prompt_tokens = d.get("total_prompt_tokens", 0)
    state.total_completion_tokens = d.get("total_completion_tokens", 0)
    state.max_tokens_per_task = d.get("max_tokens_per_task", 0)

    state.trace = [
        IterationRecord(
            iteration=t["iteration"],
            phase=t["phase"],
            model_role=t["model_role"],
            prompt_tokens=t.get("prompt_tokens", 0),
            completion_tokens=t.get("completion_tokens", 0),
            files_changed=t.get("files_changed", []),
            test_results=t.get("test_results", []),
            review_comments=t.get("review_comments", []),
            quality_findings=t.get("quality_findings", []),
            error=t.get("error"),
            duration_ms=t.get("duration_ms", 0),
        )
        for t in d.get("trace", [])
    ]

    state.messages = d.get("messages", [])
    state.result_summary = d.get("result_summary", "")
    state.error_message = d.get("error_message")
    state.rag_context = d.get("rag_context", "")
    state.memory_context = d.get("memory_context", "")

    return state
