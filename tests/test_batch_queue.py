# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/inference/batch.py — against a real Redis,
no mocks. Exercises the Redis Streams consumer-group mechanics directly
(submit -> queued -> pending -> read -> ack) without invoking the actual
embedding model or vLLM, which are already-existing, separately-testable
components; this module's own new logic is the queue semantics.
"""

from __future__ import annotations

import pytest

from src.inference.batch import (
    CONSUMER_GROUP,
    STREAM_KEY,
    BatchJobType,
    _ensure_group,
    get_batch_job_status,
    submit_batch_job,
)

pytestmark = pytest.mark.integration


async def test_submit_creates_queued_status():
    job_id = await submit_batch_job("test-tenant", BatchJobType.EMBEDDING, ["hello", "world"])
    status = await get_batch_job_status(job_id)
    assert status is not None
    assert status["status"] == "queued"
    assert status["job"]["tenant_id"] == "test-tenant"
    assert status["job"]["items"] == ["hello", "world"]


async def test_submit_rejects_empty_items():
    with pytest.raises(ValueError, match="non-empty"):
        await submit_batch_job("test-tenant", BatchJobType.EMBEDDING, [])


async def test_unknown_job_id_returns_none():
    status = await get_batch_job_status("not-a-real-job-id")
    assert status is None


async def test_stream_consumer_group_ack_cycle():
    from src.api.middleware.rate_limiter import get_redis

    job_id = await submit_batch_job("test-tenant", BatchJobType.COMPLETION, ["prompt one"])

    r = await get_redis()
    await _ensure_group(r)

    resp = await r.xreadgroup(CONSUMER_GROUP, "pytest-consumer", {STREAM_KEY: ">"}, count=10, block=2000)
    assert resp, "expected at least one pending message on the stream"
    _stream, messages = resp[0]

    matching = [(mid, fields) for mid, fields in messages if fields["id"] == job_id]
    assert matching, f"submitted job {job_id} was not found on the stream"
    message_id, fields = matching[0]
    assert fields["job_type"] == "completion"

    pending_before = await r.xpending(STREAM_KEY, CONSUMER_GROUP)
    assert pending_before["pending"] >= 1

    await r.xack(STREAM_KEY, CONSUMER_GROUP, message_id)
