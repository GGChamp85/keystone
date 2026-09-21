# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Unit tests for nodes/tool_execution.py's root-cause/re-plan step — fixing_node's own branching logic, with a
fake InferenceClient standing in for the reasoning model (coding_node's own call is also faked out, since these
tests are about fixing_node's decision of WHETHER to replan, not about coding/planning behavior itself, which
have their own test files).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.orchestrator.nodes.tool_execution import fixing_node
from src.orchestrator.state import AgentPhase, AgentState, ReviewComment, TestResult


class FakeRootCauseClient:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.call_count = 0

    async def chat_structured(
        self, messages, schema, *, schema_name="response", temperature=0.1, max_tokens=4096, on_usage=None
    ):
        self.call_count += 1
        if self._error is not None:
            raise self._error
        if on_usage is not None:
            on_usage(80, 40)
        return self._response


def _make_state(consecutive_test_failures=0, consecutive_review_failures=0) -> AgentState:
    state = AgentState()
    state.task_description = "Fix the flaky checkout total calculation."
    state.plan = "1. Fix rounding in totals.py"
    state.consecutive_test_failures = consecutive_test_failures
    state.consecutive_review_failures = consecutive_review_failures
    state.test_results = [TestResult(test_name="test_totals", passed=False, error="AssertionError: 9.99 != 10.00")]
    return state


async def test_below_threshold_skips_root_cause_and_goes_straight_to_coding():
    state = _make_state(consecutive_test_failures=1)
    client = FakeRootCauseClient()

    async def fake_coding_node(s):
        s.phase = AgentPhase.TESTING
        return s

    with (
        patch("src.orchestrator.nodes.tool_execution.get_inference_client", return_value=client),
        patch("src.orchestrator.nodes.tool_execution.coding_node", side_effect=fake_coding_node),
    ):
        result = await fixing_node(state)

    assert client.call_count == 0
    assert result.phase == AgentPhase.TESTING
    assert result.root_cause_notes == []


async def test_at_threshold_with_replan_false_still_goes_to_coding():
    state = _make_state(consecutive_test_failures=2)
    client = FakeRootCauseClient(
        {"hypothesis": "The fix attempts have small off-by-one bugs.", "evidence": "...", "replan": False}
    )

    async def fake_coding_node(s):
        s.phase = AgentPhase.TESTING
        return s

    with (
        patch("src.orchestrator.nodes.tool_execution.get_inference_client", return_value=client),
        patch("src.orchestrator.nodes.tool_execution.coding_node", side_effect=fake_coding_node),
    ):
        result = await fixing_node(state)

    assert client.call_count == 1
    assert result.phase == AgentPhase.TESTING
    assert result.root_cause_notes == ["The fix attempts have small off-by-one bugs."]
    assert result.total_prompt_tokens == 80
    assert result.total_completion_tokens == 40


async def test_replan_true_routes_to_planning_and_resets_failure_counters():
    state = _make_state(consecutive_test_failures=3, consecutive_review_failures=1)
    client = FakeRootCauseClient(
        {
            "hypothesis": "The plan edits the wrong module entirely — totals are computed in pricing.py.",
            "evidence": "Every fix touches totals.py but the failing test imports pricing.py.",
            "replan": True,
        }
    )

    # coding_node must NOT be called on a replan — patch it to blow up if it is.
    with (
        patch("src.orchestrator.nodes.tool_execution.get_inference_client", return_value=client),
        patch("src.orchestrator.nodes.tool_execution.coding_node", side_effect=AssertionError("should not run")),
    ):
        result = await fixing_node(state)

    assert result.phase == AgentPhase.PLANNING
    assert result.consecutive_test_failures == 0
    assert result.consecutive_review_failures == 0
    assert len(result.root_cause_notes) == 1
    assert "pricing.py" in result.root_cause_notes[0]


async def test_root_cause_model_unavailable_degrades_to_normal_fixing():
    """Root-cause is advisory, not a gate (unlike review) — an unreachable reasoning model must not block fixing."""
    state = _make_state(consecutive_test_failures=2)
    client = FakeRootCauseClient(error=ConnectionError("reasoning model unreachable"))

    async def fake_coding_node(s):
        s.phase = AgentPhase.TESTING
        return s

    with (
        patch("src.orchestrator.nodes.tool_execution.get_inference_client", return_value=client),
        patch("src.orchestrator.nodes.tool_execution.coding_node", side_effect=fake_coding_node),
    ):
        result = await fixing_node(state)

    assert result.phase == AgentPhase.TESTING
    assert result.root_cause_notes == []


@pytest.mark.parametrize("test_failures,review_failures", [(2, 0), (0, 2), (2, 2)])
async def test_threshold_triggers_on_either_test_or_review_failures(test_failures, review_failures):
    state = _make_state(consecutive_test_failures=test_failures, consecutive_review_failures=review_failures)
    state.review_comments = [ReviewComment(file_path="pricing.py", severity="error", message="wrong rounding mode")]
    client = FakeRootCauseClient({"hypothesis": "x", "evidence": "y", "replan": False})

    async def fake_coding_node(s):
        s.phase = AgentPhase.TESTING
        return s

    with (
        patch("src.orchestrator.nodes.tool_execution.get_inference_client", return_value=client),
        patch("src.orchestrator.nodes.tool_execution.coding_node", side_effect=fake_coding_node),
    ):
        await fixing_node(state)

    assert client.call_count == 1
