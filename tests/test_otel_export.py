# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for OpenTelemetry-style trace export (src/orchestrator/otel_export.py)
and the GET /v1/keystone/tasks/{task_id}/trace route — against a real Postgres, a real ASGI
request, and the REAL opentelemetry-sdk/opentelemetry-exporter-otlp-proto-common encoder (no
hand-rolled JSON, no mock of the OTel library). There is no live OTel collector in this dev
environment, so what's verified is exactly what CAN be verified without one: the real library's
own encoder accepts the span tree without error, every span shares one real trace id, every
non-root span's parent id matches its actual parent's span id, the span count matches the
recorded trace exactly, and every span's start time is at or before its end time.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import AgentTask, APIKey, TaskStatus, Tenant, TenantTier
from src.main import create_app
from src.orchestrator.otel_export import build_task_otel_export

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]

_SAMPLE_TRACE = [
    {
        "iteration": 0,
        "phase": "coding",
        "model_role": "coding",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "files_changed": ["a.py"],
        "test_results": [],
        "review_comments": [],
        "quality_findings": [],
        "steps": [
            {"event_type": "tool_call", "node": "coding", "phase": "coding", "name": "read_file", "timestamp": 1.0},
            {
                "event_type": "tool_result",
                "node": "coding",
                "phase": "coding",
                "ok": True,
                "output": "content",
                "timestamp": 2.0,
            },
        ],
        "error": None,
        "duration_ms": 100,
    },
    {
        "iteration": 1,
        "phase": "testing",
        "model_role": "coding",
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "files_changed": [],
        "test_results": [],
        "review_comments": [],
        "quality_findings": [],
        "steps": [
            {"event_type": "test_output", "node": "testing", "phase": "testing", "passed": True, "timestamp": 3.0}
        ],
        "error": None,
        "duration_ms": 50,
    },
    {
        # No steps recorded — a planning/review-only iteration never calls publish_task_step —
        # exercises the duration_ms fallback path.
        "iteration": 2,
        "phase": "review",
        "model_role": "reasoning",
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "files_changed": [],
        "test_results": [],
        "review_comments": [],
        "quality_findings": [],
        "steps": [],
        "error": None,
        "duration_ms": 200,
    },
]


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="otel-export-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _real_api_key(tenant_id) -> str:
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="otel-export-key", key_prefix=prefix, key_hash=key_hash))
        await db.flush()
    return full_key


async def _completed_task_with_trace(tenant_id) -> uuid.UUID:
    task_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            AgentTask(
                id=task_id,
                tenant_id=tenant_id,
                task_description="A real task with a real recorded trace, for OTel export.",
                status=TaskStatus.COMPLETED,
                execution_trace=_SAMPLE_TRACE,
                result_summary="Fixed the bug.",
                total_prompt_tokens=125,
                total_completion_tokens=65,
            )
        )
        await db.flush()
    return task_id


def _spans(payload: dict) -> list[dict]:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


async def test_build_task_otel_export_returns_none_for_an_unknown_task():
    assert await build_task_otel_export(uuid.uuid4()) is None


async def test_build_task_otel_export_produces_a_real_valid_span_tree(tenant_id):
    task_id = await _completed_task_with_trace(tenant_id)
    export = await build_task_otel_export(task_id)
    assert export is not None
    assert export.tenant_id == tenant_id

    spans = _spans(export.payload)
    # 1 root + 3 iterations + 3 steps (2 in iteration 0, 1 in iteration 1, 0 in iteration 2)
    assert export.span_count == len(spans) == 1 + 3 + 3

    root = spans[0]
    assert "parentSpanId" not in root
    assert root["name"] == "agent_task"
    root_trace_id = root["traceId"]

    for span in spans:
        assert span["traceId"] == root_trace_id, "every span in one task's export must share one real trace id"
        assert int(span["startTimeUnixNano"]) <= int(span["endTimeUnixNano"])

    non_root = spans[1:]
    assert all("parentSpanId" in s for s in non_root), "every non-root span must declare a real parent"
    all_span_ids = {s["spanId"] for s in spans}
    assert all(s["parentSpanId"] in all_span_ids for s in non_root), "every parent id must resolve to a real span"

    iteration_spans = [s for s in spans if s["name"].startswith("iteration.")]
    assert len(iteration_spans) == 3
    assert all(s["parentSpanId"] == root["spanId"] for s in iteration_spans)

    tool_call_span = next(s for s in spans if s["name"] == "coding.tool_call")
    assert tool_call_span["kind"] == "SPAN_KIND_CLIENT"
    attrs = {a["key"]: a["value"] for a in tool_call_span["attributes"]}
    assert attrs["name"]["stringValue"] == "read_file"


async def test_otel_export_route_returns_a_real_otlp_payload(tenant_id):
    full_key = await _real_api_key(tenant_id)
    task_id = await _completed_task_with_trace(tenant_id)

    async with running_client() as client:
        resp = await client.get(f"/v1/keystone/tasks/{task_id}/trace", headers={"Authorization": f"Bearer {full_key}"})
        assert resp.status_code == 200
        payload = resp.json()

    spans = _spans(payload)
    assert len(spans) == 1 + 3 + 3
    assert spans[0]["name"] == "agent_task"


async def test_otel_export_route_404s_for_an_unknown_task(tenant_id):
    full_key = await _real_api_key(tenant_id)
    async with running_client() as client:
        resp = await client.get(
            f"/v1/keystone/tasks/{uuid.uuid4()}/trace", headers={"Authorization": f"Bearer {full_key}"}
        )
        assert resp.status_code == 404


async def test_otel_export_route_404s_for_another_tenants_task(tenant_id):
    other_tenant = uuid.uuid4()
    async with get_db_context() as db:
        db.add(
            Tenant(id=other_tenant, name="otel-other-tenant", email=f"{other_tenant}@test.dev", tier=TenantTier.FREE)
        )
        await db.flush()
    try:
        other_key = await _real_api_key(other_tenant)
        foreign_task_id = await _completed_task_with_trace(tenant_id)

        async with running_client() as client:
            resp = await client.get(
                f"/v1/keystone/tasks/{foreign_task_id}/trace", headers={"Authorization": f"Bearer {other_key}"}
            )
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant)
            if row is not None:
                await db.delete(row)


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18085") as client:
            yield client
