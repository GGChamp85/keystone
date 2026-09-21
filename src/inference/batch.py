# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — Batch/async inference queue.

For embeddings, re-ranking, and dataset-generation workloads that don't
need the low-latency interactive path (/v1/chat/completions) — submitted
jobs are queued in Redis (a Redis Stream, for consumer-group semantics: a
job is only removed from the pending list once a worker acknowledges it,
so a worker crash mid-job doesn't silently drop the job) and drained by a
separate worker process (batch_worker.py), so a large batch never competes
with interactive traffic for the same request path.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any

import redis.asyncio as aioredis
import structlog

from src.api.middleware.rate_limiter import get_redis

logger = structlog.get_logger(__name__)

STREAM_KEY = "keystone:batch:jobs"
CONSUMER_GROUP = "keystone-batch-workers"
RESULT_KEY_PREFIX = "keystone:batch:result:"
RESULT_TTL_SECONDS = 24 * 3600


class BatchJobType(StrEnum):
    EMBEDDING = "embedding"
    COMPLETION = "completion"


class BatchJobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchJob:
    id: str
    tenant_id: str
    job_type: BatchJobType
    items: list[str]  # texts to embed, or prompts to complete
    model_role: str = "coding"  # used only for job_type=completion
    status: BatchJobStatus = BatchJobStatus.QUEUED
    created_at: float = field(default_factory=time.time)

    def to_stream_fields(self) -> dict[str, str]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "job_type": self.job_type.value,
            "items": json.dumps(self.items),
            "model_role": self.model_role,
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_stream_fields(cls, fields: dict[str, str]) -> BatchJob:
        return cls(
            id=fields["id"],
            tenant_id=fields["tenant_id"],
            job_type=BatchJobType(fields["job_type"]),
            items=json.loads(fields["items"]),
            model_role=fields.get("model_role", "coding"),
            created_at=float(fields.get("created_at", time.time())),
        )


async def _ensure_group(r: aioredis.Redis) -> None:
    try:
        await r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def submit_batch_job(
    tenant_id: str,
    job_type: BatchJobType,
    items: list[str],
    model_role: str = "coding",
) -> str:
    """Queue a batch job. Returns the job ID immediately; processing happens
    out-of-band via batch_worker.py."""
    if not items:
        raise ValueError("items must be non-empty")

    job = BatchJob(id=str(uuid.uuid4()), tenant_id=tenant_id, job_type=job_type, items=items, model_role=model_role)

    r = await get_redis()
    await _ensure_group(r)
    await r.xadd(STREAM_KEY, job.to_stream_fields())
    await r.set(
        f"{RESULT_KEY_PREFIX}{job.id}",
        json.dumps({"status": BatchJobStatus.QUEUED.value, "job": asdict(job, dict_factory=_enum_safe_dict)}),
        ex=RESULT_TTL_SECONDS,
    )
    logger.info("batch.job_submitted", job_id=job.id, job_type=job_type.value, item_count=len(items))
    return job.id


def _enum_safe_dict(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in items}


async def get_batch_job_status(job_id: str) -> dict | None:
    r = await get_redis()
    raw = await r.get(f"{RESULT_KEY_PREFIX}{job_id}")
    if raw is None:
        return None
    return json.loads(raw)


async def _mark_result(r: aioredis.Redis, job_id: str, status: BatchJobStatus, **extra: Any) -> None:
    payload = {"status": status.value, "job_id": job_id, **extra}
    await r.set(f"{RESULT_KEY_PREFIX}{job_id}", json.dumps(payload), ex=RESULT_TTL_SECONDS)


async def process_one(r: aioredis.Redis, consumer_name: str, block_ms: int = 5000) -> bool:
    """
    Read and process a single job from the stream, using consumer-group
    semantics: XACK only happens after the job's result is durably written,
    so a worker crash mid-job leaves the entry pending for another consumer
    to pick up via XCLAIM (not implemented here — see the stream's pending-
    entries list, XPENDING, for manual/cron-based recovery in production).
    Returns True if a job was processed, False on a read timeout (no job).
    """
    await _ensure_group(r)
    resp = await r.xreadgroup(CONSUMER_GROUP, consumer_name, {STREAM_KEY: ">"}, count=1, block=block_ms)
    if not resp:
        return False

    _stream, messages = resp[0]
    message_id, fields = messages[0]
    job = BatchJob.from_stream_fields(fields)

    logger.info("batch.job_processing", job_id=job.id, job_type=job.job_type.value)
    await _mark_result(r, job.id, BatchJobStatus.PROCESSING)

    try:
        if job.job_type == BatchJobType.EMBEDDING:
            result = await _process_embedding_job(job)
        else:
            result = await _process_completion_job(job)
        await _mark_result(r, job.id, BatchJobStatus.COMPLETED, result=result)
        logger.info("batch.job_completed", job_id=job.id)
    except Exception as exc:
        logger.error("batch.job_failed", job_id=job.id, error=str(exc))
        await _mark_result(r, job.id, BatchJobStatus.FAILED, error=str(exc))
    finally:
        await r.xack(STREAM_KEY, CONSUMER_GROUP, message_id)

    return True


async def _process_embedding_job(job: BatchJob) -> dict[str, Any]:
    from src.memory.embeddings import EmbeddingService

    embedder = EmbeddingService()
    vectors = await embedder.embed_batch(job.items)
    return {"embeddings": vectors, "count": len(vectors)}


async def _process_completion_job(job: BatchJob) -> dict[str, Any]:
    from src.inference.client import get_inference_client

    client = get_inference_client(job.model_role)
    completions = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for item in job.items:
        response = await client.complete(messages=[{"role": "user", "content": item}])
        usage = response.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        completions.append(response["choices"][0]["message"]["content"])
    return {
        "completions": completions,
        "count": len(completions),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }
