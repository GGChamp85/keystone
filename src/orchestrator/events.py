# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — live task execution event stream.

Publishes a curated event after every LangGraph node transition (see
`HeartbeatCallback` in src/orchestrator/graph.py, which this hooks into —
the same extension point Temporal's activity heartbeat already uses, not
a separate mechanism) to a per-task Redis Stream, so a frontend can render
the plan/execution trace live as the agent works: `GET
/v1/keystone/tasks/{task_id}/stream` (src/api/routes/agents.py) replays
this stream's history then blocks for new entries, converting each to a
Server-Sent Event.

Redis Streams (not plain pub/sub) deliberately: a client that connects
*after* planning already finished still needs to see the plan — pub/sub
has no history, a Stream does (XRANGE from the start, then XREAD BLOCK for
what comes next). Streams are the one Redis messaging primitive this
codebase uses, deliberately — pub/sub is not introduced alongside it.
"""

from __future__ import annotations

import json
import time
from typing import Any, cast
from uuid import UUID

import structlog

from src.api.middleware.rate_limiter import get_redis

logger = structlog.get_logger(__name__)

_STREAM_PREFIX = "keystone:task:"
_STREAM_SUFFIX = ":events"
_STREAM_MAXLEN = 500  # a task capped at circuit-breaker's max_iterations=hundreds of nodes, generously bounded
_STREAM_TTL_SECONDS = 24 * 3600  # events outlive the task by a day so a late-loading UI can still replay them

TERMINAL_PHASES = {"complete", "failed", "cancelled"}


def _stream_key(task_id: UUID | str) -> str:
    return f"{_STREAM_PREFIX}{task_id}{_STREAM_SUFFIX}"


def _summarize(node_name: str, state: dict[str, Any]) -> dict[str, Any]:
    """
    Trim the full AgentState dict (see graph.py's _state_to_dict — this
    mirrors its exact field names) down to what a live UI actually needs —
    not the whole thing, which carries full file contents/diffs and would
    make every event payload grow with task size.
    """
    phase = state.get("phase", "")
    if hasattr(phase, "value"):  # AgentPhase enum, in case a caller passes the raw dataclass form
        phase = phase.value

    file_changes = state.get("file_changes") or []
    review_comments = state.get("review_comments") or []
    test_results = state.get("test_results") or []

    return {
        "node": node_name,
        "phase": str(phase),
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations"),
        "plan": state.get("plan", ""),
        "plan_steps": state.get("plan_steps") or [],
        "current_plan_step": state.get("current_plan_step", 0),
        "files_changed": [fc.get("path") for fc in file_changes if isinstance(fc, dict)],
        "review_passed": state.get("review_passed"),
        "review_comment_count": len(review_comments),
        "tests_passed": state.get("tests_passed"),
        "test_pass_count": sum(1 for t in test_results if isinstance(t, dict) and t.get("passed")),
        "test_total_count": len(test_results),
        "total_tokens": (state.get("total_prompt_tokens") or 0) + (state.get("total_completion_tokens") or 0),
        "result_summary": state.get("result_summary", ""),
        "error_message": state.get("error_message"),
        "timestamp": time.time(),
    }


async def publish_task_event(task_id: UUID | str, node_name: str, state: dict[str, Any]) -> None:
    """
    Append one execution-trace event for `task_id`. Never raises — a
    streaming-UI hiccup must not fail or slow down the actual agent run;
    matches how graph.py already treats `on_iteration` heartbeat failures
    as log-and-continue, not fatal.
    """
    try:
        redis = await get_redis()
        key = _stream_key(task_id)
        payload = _summarize(node_name, state)
        await redis.xadd(key, {"data": json.dumps(payload)}, maxlen=_STREAM_MAXLEN, approximate=True)
        await redis.expire(key, _STREAM_TTL_SECONDS)
    except Exception as exc:
        logger.warning("task_events.publish_failed", task_id=str(task_id), node=node_name, error=str(exc))


async def read_task_events_from(task_id: UUID | str, last_id: str = "0-0") -> list[tuple[str, dict[str, Any]]]:
    """One-shot read of every event after `last_id` (replay use, "0-0" = from the start)."""
    redis = await get_redis()
    key = _stream_key(task_id)
    # redis-py types stream replies as bytes-or-str unions; this client decodes responses, so they are str.
    entries = cast(
        "list[tuple[str, dict[str, str]]]",
        await redis.xrange(key, min=f"({last_id}" if last_id != "0-0" else "-", max="+"),
    )
    return [(entry_id, json.loads(fields["data"])) for entry_id, fields in entries]


async def block_for_next_event(
    task_id: UUID | str, last_id: str, timeout_ms: int = 15_000
) -> tuple[str, dict[str, Any]] | None:
    """Block until the next event after `last_id`, or return None on timeout (caller re-polls/checks disconnect)."""
    redis = await get_redis()
    key = _stream_key(task_id)
    result = cast(
        "list[tuple[str, list[tuple[str, dict[str, str]]]]]",
        await redis.xread({key: last_id}, count=1, block=timeout_ms),
    )
    if not result:
        return None
    _key, entries = result[0]
    entry_id, fields = entries[0]
    return entry_id, json.loads(fields["data"])
