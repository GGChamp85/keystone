# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real unit tests for the image-attachment plumbing in
src/orchestrator/nodes/coding.py::_build_agentic_user_context — pure, no sandbox/inference needed.
"""

from __future__ import annotations

from src.orchestrator.nodes.coding import _build_agentic_user_context, _with_image_parts
from src.orchestrator.state import AgentState


def test_with_image_parts_returns_plain_text_when_there_are_no_images():
    assert _with_image_parts("hello", []) == "hello"


def test_with_image_parts_returns_content_part_list_when_images_are_present():
    images = [{"media_type": "image/png", "data": "aGVsbG8="}]
    result = _with_image_parts("## Task\ndo the thing", images)
    assert result == [
        {"type": "text", "text": "## Task\ndo the thing"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]


def test_with_image_parts_handles_multiple_images_in_order():
    images = [
        {"media_type": "image/png", "data": "AAAA"},
        {"media_type": "image/jpeg", "data": "BBBB"},
    ]
    result = _with_image_parts("text", images)
    assert result[0] == {"type": "text", "text": "text"}
    assert result[1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert result[2]["image_url"]["url"] == "data:image/jpeg;base64,BBBB"


def test_build_agentic_user_context_is_plain_string_without_images():
    state = AgentState(task_description="Fix the bug in parser.py")
    result = _build_agentic_user_context(state)
    assert isinstance(result, str)
    assert "Fix the bug in parser.py" in result


def test_build_agentic_user_context_becomes_content_parts_with_images():
    state = AgentState(
        task_description="Implement this UI mockup exactly.",
        images=[{"media_type": "image/png", "data": "aGVsbG8="}],
    )
    result = _build_agentic_user_context(state)
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert "Implement this UI mockup exactly." in result[0]["text"]
    assert result[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
