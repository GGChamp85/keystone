# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — live fine-tuning job event stream.

The identical Redis-Streams mechanism src/orchestrator/events.py already
uses for coding tasks (append-only per-job stream, history replay for a
client that connects late, block-for-next for live updates) — reused
rather than introduced as a second primitive, just keyed and shaped for a
FineTuneJob instead of an AgentState. A fine-tuning job's UI experience
(GET /v1/finetune/jobs/{id}/stream) should feel like a coding task's, not
a differently-behaved thing bolted on beside it.
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

import structlog

from src.api.middleware.rate_limiter import get_redis

logger = structlog.get_logger(__name__)

_STREAM_PREFIX = "keystone:finetune:"
_STREAM_SUFFIX = ":events"
_STREAM_MAXLEN = 200
_STREAM_TTL_SECONDS = 24 * 3600

TERMINAL_STATUSES = {"completed", "failed"}


def _stream_key(job_id: UUID | str) -> str:
    return f"{_STREAM_PREFIX}{job_id}{_STREAM_SUFFIX}"


async def publish_finetune_event(
    job_id: UUID | str,
    status: str,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Never raises — a streaming-UI hiccup must not fail or slow down the
    actual training run, matching src/orchestrator/events.py's own
    publish_task_event convention."""
    try:
        redis = await get_redis()
        key = _stream_key(job_id)
        payload = {"status": status, "message": message, "metrics": metrics or {}, "timestamp": time.time()}
        await redis.xadd(key, {"data": json.dumps(payload)}, maxlen=_STREAM_MAXLEN, approximate=True)
        await redis.expire(key, _STREAM_TTL_SECONDS)
    except Exception as exc:
        logger.warning("finetune_events.publish_failed", job_id=str(job_id), status=status, error=str(exc))


async def read_finetune_events_from(job_id: UUID | str, last_id: str = "0-0") -> list[tuple[str, dict[str, Any]]]:
    """One-shot read of every event after `last_id` ("0-0" = from the start)."""
    redis = await get_redis()
    key = _stream_key(job_id)
    entries = await redis.xrange(key, min=f"({last_id}" if last_id != "0-0" else "-", max="+")
    return [(entry_id, json.loads(fields["data"])) for entry_id, fields in entries]


async def block_for_next_finetune_event(
    job_id: UUID | str, last_id: str, timeout_ms: int = 15_000
) -> tuple[str, dict[str, Any]] | None:
    """Block until the next event after `last_id`, or None on timeout (caller re-polls/checks disconnect)."""
    redis = await get_redis()
    key = _stream_key(job_id)
    result = await redis.xread({key: last_id}, count=1, block=timeout_ms)
    if not result:
        return None
    _key, entries = result[0]
    entry_id, fields = entries[0]
    return entry_id, json.loads(fields["data"])
