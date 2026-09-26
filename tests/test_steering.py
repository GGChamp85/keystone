# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/orchestrator/steering.py — a real Redis (tests/conftest.py's
requires_integration_env), no mocking of the queue.
"""

from __future__ import annotations

import uuid

import pytest

from src.api.middleware.rate_limiter import get_redis
from src.orchestrator.steering import drain_steering_messages, enqueue_steering_message

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


async def test_drain_with_nothing_queued_returns_empty():
    task_id = uuid.uuid4()
    assert await drain_steering_messages(task_id) == []


async def test_enqueue_then_drain_returns_the_message():
    task_id = uuid.uuid4()
    await enqueue_steering_message(task_id, "also add a test for the empty-input case")
    assert await drain_steering_messages(task_id) == ["also add a test for the empty-input case"]


async def test_multiple_messages_drain_in_fifo_order():
    task_id = uuid.uuid4()
    await enqueue_steering_message(task_id, "first")
    await enqueue_steering_message(task_id, "second")
    await enqueue_steering_message(task_id, "third")
    assert await drain_steering_messages(task_id) == ["first", "second", "third"]


async def test_drain_clears_the_queue_a_second_drain_is_empty():
    task_id = uuid.uuid4()
    await enqueue_steering_message(task_id, "only once")
    assert await drain_steering_messages(task_id) == ["only once"]
    assert await drain_steering_messages(task_id) == []


async def test_blank_message_is_not_queued():
    task_id = uuid.uuid4()
    await enqueue_steering_message(task_id, "   ")
    assert await drain_steering_messages(task_id) == []


async def test_different_tasks_have_independent_queues():
    task_a, task_b = uuid.uuid4(), uuid.uuid4()
    await enqueue_steering_message(task_a, "for a")
    await enqueue_steering_message(task_b, "for b")
    assert await drain_steering_messages(task_a) == ["for a"]
    assert await drain_steering_messages(task_b) == ["for b"]


async def test_queue_key_carries_a_real_ttl_not_left_forever():
    task_id = uuid.uuid4()
    await enqueue_steering_message(task_id, "hello")
    r = await get_redis()
    ttl = await r.ttl(f"keystone:steering:{task_id}")
    assert 0 < ttl <= 3600
