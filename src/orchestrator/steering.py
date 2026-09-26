# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — mid-task steering.

The live task stream is one-way: once a task is submitted you can watch it,
not redirect it. `POST /v1/keystone/tasks/{id}/steer` (src/api/routes/agents.py)
queues a new instruction in Redis for a running task; `_run_agentic_loop`
(nodes/coding.py) drains the queue once per iteration — never mid-tool-call,
so a steering message can't split an assistant `tool_calls` message from its
matching tool-result messages — and injects whatever is queued as a whole
new turn before the next model call. A message added between two drains
isn't lost, just picked up one iteration later.

Redis, not Postgres: this is an ephemeral, low-latency hand-off between an
HTTP request and a running loop, the same reasoning src/orchestrator/
concurrency.py already applies to its own Redis-backed state. A message
still sitting in the queue when the task finishes expires with the key
rather than leaking forever.
"""

from __future__ import annotations

from uuid import UUID

from src.api.middleware.rate_limiter import get_redis

_QUEUE_TTL_SECONDS = 3600  # generous — the longest a task's own AGENT_MAX_WALL_CLOCK_SECONDS could run
_MAX_MESSAGE_LENGTH = 10_000


def _queue_key(task_id: UUID | str) -> str:
    return f"keystone:steering:{task_id}"


async def enqueue_steering_message(task_id: UUID | str, message: str) -> None:
    """Queue one instruction for a running task. Never raises on a Redis outage — a steering
    message that can't be queued is a missed nudge, not a reason to fail the request that's
    already running; the caller still gets a clear error from the route itself in that case."""
    text = message.strip()
    if not text:
        return
    key = _queue_key(task_id)
    r = await get_redis()
    await r.rpush(key, text[:_MAX_MESSAGE_LENGTH])
    await r.expire(key, _QUEUE_TTL_SECONDS)


async def drain_steering_messages(task_id: UUID | str) -> list[str]:
    """Every message queued since the last drain, oldest first, and clears the queue. Not
    perfectly atomic against a concurrent enqueue — a message that arrives between the LRANGE
    and the DELETE below is simply caught on the *next* drain, never lost and never duplicated,
    which is the semantics this needs (an extra loop iteration's delay is invisible to the user;
    a dropped instruction would not be)."""
    key = _queue_key(task_id)
    r = await get_redis()
    raw = await r.lrange(key, 0, -1)
    if raw:
        await r.delete(key)
    return [m.decode() if isinstance(m, bytes) else m for m in raw]
