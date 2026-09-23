# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Step-level task events (src/orchestrator/events.py) against a real Redis:
node events, step events with full payloads, the engine's final event, and
the SSE route's closing rule — plus the no-trimming guarantee (a long task's
earliest steps are never dropped). Requires REDIS_URL (tests/conftest.py's
requires_integration_env); skipped otherwise.
"""

from __future__ import annotations

import uuid

import pytest

from src.api.middleware.rate_limiter import get_redis
from src.orchestrator.events import (
    STEP_EVENT_TYPES,
    is_final_event,
    publish_task_event,
    publish_task_final,
    publish_task_step,
    read_task_events_from,
)

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


@pytest.fixture
async def task_id():
    tid = uuid.uuid4()
    yield tid
    r = await get_redis()
    await r.delete(f"keystone:task:{tid}:events")


async def test_step_events_carry_full_payloads_in_order_and_close_only_on_final(task_id):
    await publish_task_event(task_id, "planning", {"phase": "coding", "iteration": 0})
    big_output = "line\n" * 50_000  # 250 kB of tool output — kept whole, no payload cap
    await publish_task_step(task_id, "tool_call", "coding", {"name": "read_file", "arguments": {"path": "a.py"}})
    await publish_task_step(task_id, "tool_result", "coding", {"name": "read_file", "ok": True, "output": big_output})
    # The graph's own terminal phase does NOT close the stream any more — the git workflow follows it.
    await publish_task_event(task_id, "testing", {"phase": "complete", "iteration": 3})
    await publish_task_step(task_id, "diff", "finalize", {"diff": "+x", "branch_name": "b", "commit_sha": "abc"})
    await publish_task_final(task_id, "complete", result_summary="done", git_result={"pr_url": "u", "pr_number": 7})

    events = [payload for _id, payload in await read_task_events_from(task_id)]
    assert [e["event_type"] for e in events] == ["node", "tool_call", "tool_result", "node", "diff", "node"]
    assert events[2]["output"] == big_output
    assert events[1]["arguments"] == {"path": "a.py"}
    assert [is_final_event(e) for e in events] == [False, False, False, False, False, True]
    assert events[-1]["pr_url"] == "u" and events[-1]["pr_number"] == 7 and events[-1]["phase"] == "complete"


async def test_stream_is_never_trimmed(task_id):
    # Previously MAXLEN=500 (approximate) silently dropped a long task's earliest steps.
    for i in range(1_200):
        await publish_task_step(task_id, "model_text", "coding", {"turn": i, "content": f"turn {i}"})
    events = await read_task_events_from(task_id)
    assert len(events) == 1_200
    assert events[0][1]["turn"] == 0 and events[-1][1]["turn"] == 1_199


def test_legacy_terminal_event_without_event_type_still_closes_the_stream():
    assert is_final_event({"phase": "complete"}) is True  # recorded before step events existed
    assert is_final_event({"event_type": "node", "phase": "complete"}) is False  # graph terminal, git work pending
    assert is_final_event({"event_type": "node", "phase": "complete", "final": True}) is True


async def test_unknown_step_type_is_a_programming_error(task_id):
    with pytest.raises(ValueError, match="unknown step event type"):
        await publish_task_step(task_id, "not_a_type", "coding", {})
    assert "tool_call" in STEP_EVENT_TYPES
