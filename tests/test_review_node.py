# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Unit tests for nodes/review.py's fail-closed behavior — the review node's own branching logic, exercised with a
fake InferenceClient standing in for the reasoning model (real HTTP/retry/JSON-schema behavior is already
covered for real in test_inference_client.py against tests/fixtures/fake_vllm_server.py). Standalone mode
(state.file_changes, no repository_url) needs no sandbox, so these run unconditionally.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.inference.client import StructuredOutputError
from src.orchestrator.nodes.review import review_node
from src.orchestrator.state import AgentPhase, AgentState, FileChange


class FakeReviewClient:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self._response = response
        self._error = error

    async def chat_structured(
        self, messages, schema, *, schema_name="response", temperature=0.1, max_tokens=4096, on_usage=None
    ):
        if self._error is not None:
            raise self._error
        if on_usage is not None:
            on_usage(100, 50)
        return self._response


def _make_state() -> AgentState:
    state = AgentState()
    state.task_description = "Add input validation to the login endpoint."
    state.enable_reasoning_review = True
    state.enable_sandbox_testing = True
    state.file_changes = [FileChange(path="app.py", action="modify", language="python", new_content="def f(): ...")]
    return state


async def test_approved_with_no_issues_passes_review():
    state = _make_state()
    client = FakeReviewClient({"approved": True, "summary": "Looks good.", "comments": [], "blocking_issues": 0})
    with patch("src.orchestrator.nodes.review.get_inference_client", return_value=client):
        result = await review_node(state)

    assert result.review_passed is True
    assert result.phase == AgentPhase.TESTING
    assert result.consecutive_review_failures == 0
    assert result.total_prompt_tokens == 100
    assert result.total_completion_tokens == 50


async def test_approved_true_overridden_by_a_critical_comment():
    """The model's own `approved` boolean is advisory, not authoritative — see review.py's module docstring."""
    state = _make_state()
    client = FakeReviewClient(
        {
            "approved": True,
            "summary": "Mostly fine but one real bug.",
            "comments": [
                {
                    "file_path": "app.py",
                    "line": 10,
                    "severity": "critical",
                    "message": "SQL injection",
                    "suggestion": None,
                }
            ],
            "blocking_issues": 0,
        }
    )
    with patch("src.orchestrator.nodes.review.get_inference_client", return_value=client):
        result = await review_node(state)

    assert result.review_passed is False
    assert result.phase == AgentPhase.FIXING
    assert result.consecutive_review_failures == 1


async def test_approved_true_overridden_by_positive_blocking_issues():
    state = _make_state()
    client = FakeReviewClient(
        {"approved": True, "summary": "x", "comments": [], "blocking_issues": 2},
    )
    with patch("src.orchestrator.nodes.review.get_inference_client", return_value=client):
        result = await review_node(state)

    assert result.review_passed is False
    assert result.phase == AgentPhase.FIXING


async def test_structured_output_error_fails_closed_not_open():
    """A review the model can't produce must block the task, never silently pass it through."""
    state = _make_state()
    client = FakeReviewClient(error=StructuredOutputError("model never returned valid JSON"))
    with patch("src.orchestrator.nodes.review.get_inference_client", return_value=client):
        result = await review_node(state)

    assert result.review_passed is False
    assert result.phase == AgentPhase.FIXING
    assert result.consecutive_review_failures == 1
    assert any("Review unavailable" in rc.message for rc in result.review_comments)


async def test_reasoning_model_unreachable_fails_closed_not_open():
    state = _make_state()
    client = FakeReviewClient(error=ConnectionError("reasoning model endpoint unreachable"))
    with patch("src.orchestrator.nodes.review.get_inference_client", return_value=client):
        result = await review_node(state)

    assert result.review_passed is False
    assert result.phase == AgentPhase.FIXING
    assert any("Review unavailable" in rc.message for rc in result.review_comments)


async def test_disabled_review_skips_straight_through():
    state = _make_state()
    state.enable_reasoning_review = False
    result = await review_node(state)

    assert result.review_passed is True
    assert result.phase == AgentPhase.TESTING


@pytest.mark.parametrize("attempt", [1, 2, 3])
async def test_consecutive_failures_accumulate_across_calls(attempt):
    """Sanity check that the circuit breaker's max_consecutive_review_failures has something real to count."""
    state = _make_state()
    state.consecutive_review_failures = attempt - 1
    client = FakeReviewClient(error=StructuredOutputError("boom"))
    with patch("src.orchestrator.nodes.review.get_inference_client", return_value=client):
        result = await review_node(state)

    assert result.consecutive_review_failures == attempt
