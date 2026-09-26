# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — OpenTelemetry-style trace export.

Converts a task's durable execution record (`AgentTask.execution_trace`, the same data
src/orchestrator/replay.py already reads) into a real OTel span tree — one root span for the
task, one child span per graph iteration, one grandchild span per recorded step inside that
iteration (a tool call, a test run, a quality-gate pass, ...) — and encodes it with the actual
`opentelemetry-sdk`/`opentelemetry-exporter-otlp-proto-common` library, not hand-rolled JSON that
merely resembles the OTLP schema. The result is exactly what `encode_spans` would send to a real
OTLP collector, so it can be fed straight into Jaeger, Tempo, Honeycomb, or any OTLP-compatible
backend.

Deliberately built after the fact from already-recorded data, the same honesty stance as
replay.py: this is not a live tracer instrumenting the agent as it runs (a materially bigger,
riskier change needing a real collector to develop against) — every timestamp below comes from a
step's own recorded `timestamp` (events.py) or, for an iteration that recorded no steps of its own
(a planning- or review-only pass never calls `publish_task_step`), from `IterationRecord.duration_ms`
chained onto the running cursor. Never a fabricated or estimated time with no basis in what
actually happened.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from src.db.connection import get_db_context
from src.db.models import AgentTask, TaskStatus

_ATTR_VALUE_MAX_CHARS = 4096
_TOOL_EVENT_TYPES = frozenset({"tool_call", "tool_result"})
_STEP_METADATA_KEYS = frozenset({"event_type", "node", "phase", "timestamp", "final"})
_RESOURCE = Resource.create({"service.name": "keystone-agents"})


def _otel_attr_value(value: Any) -> str | bool | int | float | None:
    """Coerce one field into an OTel-attribute-safe scalar. OTel attributes are scalars or
    homogeneous sequences of scalars, never a nested dict/list — so anything else (a tool call's
    `arguments` dict, a list of test results) is JSON-encoded, then capped, since an uncapped
    string ends up in every span exported to a real backend, unlike the live event stream
    (events.py), which deliberately has no payload cap because it is the record, not an export
    of it."""
    if value is None:
        return None
    if isinstance(value, str | bool | int | float):
        coerced: str | bool | int | float = value
    else:
        coerced = json.dumps(value, default=str)
    if isinstance(coerced, str) and len(coerced) > _ATTR_VALUE_MAX_CHARS:
        coerced = coerced[:_ATTR_VALUE_MAX_CHARS] + f"...<truncated, {len(coerced)} chars total>"
    return coerced


def _attributes(fields: dict[str, Any]) -> dict[str, str | bool | int | float]:
    result: dict[str, str | bool | int | float] = {}
    for key, value in fields.items():
        coerced = _otel_attr_value(value)
        if coerced is not None:
            result[key] = coerced
    return result


def _step_attributes(step: dict[str, Any]) -> dict[str, str | bool | int | float]:
    return _attributes({k: v for k, v in step.items() if k not in _STEP_METADATA_KEYS})


@dataclass
class TaskOtelExport:
    tenant_id: UUID
    trace_id_hex: str
    span_count: int
    payload: dict[str, Any]


async def build_task_otel_export(task_id: UUID) -> TaskOtelExport | None:
    """Build and encode the real OTLP export for one task. Returns None if the task doesn't
    exist. Never raises on a malformed or missing individual field within the trace — every
    field is read defensively, since `execution_trace` is historical data from potentially many
    past code versions."""
    async with get_db_context() as db:
        task = await db.get(AgentTask, task_id)
        if task is None:
            return None

        trace_id = random.getrandbits(128)
        trace_flags = TraceFlags(TraceFlags.SAMPLED)

        def new_context() -> SpanContext:
            return SpanContext(
                trace_id=trace_id, span_id=random.getrandbits(64), is_remote=False, trace_flags=trace_flags
            )

        root_ctx = new_context()
        execution_trace = task.execution_trace or []

        cursor = task.started_at.timestamp() if task.started_at else time.time()
        earliest = cursor
        latest = cursor
        spans: list[ReadableSpan] = []

        for i, record in enumerate(execution_trace):
            steps = record.get("steps") or []
            step_timestamps = [s["timestamp"] for s in steps if isinstance(s.get("timestamp"), int | float)]
            if step_timestamps:
                iter_start, iter_end = min(step_timestamps), max(step_timestamps)
            else:
                iter_start = cursor
                iter_end = iter_start + (record.get("duration_ms") or 0) / 1000.0
            cursor = max(cursor, iter_end)
            earliest = min(earliest, iter_start)
            latest = max(latest, iter_end)

            iter_ctx = new_context()
            spans.append(
                ReadableSpan(
                    name=f"iteration.{record.get('phase', 'unknown')}",
                    context=iter_ctx,
                    parent=root_ctx,
                    resource=_RESOURCE,
                    attributes=_attributes(
                        {
                            "iteration": record.get("iteration", i),
                            "phase": record.get("phase"),
                            "model_role": record.get("model_role"),
                            "prompt_tokens": record.get("prompt_tokens", 0),
                            "completion_tokens": record.get("completion_tokens", 0),
                            "error": record.get("error"),
                        }
                    ),
                    kind=SpanKind.INTERNAL,
                    status=Status(StatusCode.ERROR if record.get("error") else StatusCode.OK),
                    start_time=int(iter_start * 1e9),
                    end_time=int(iter_end * 1e9),
                )
            )

            step_cursor = iter_start
            for step in steps:
                ts = step.get("timestamp")
                step_time = ts if isinstance(ts, int | float) else step_cursor
                step_cursor = max(step_cursor, step_time)
                event_type = step.get("event_type", "step")
                spans.append(
                    ReadableSpan(
                        name=f"{step.get('node', record.get('phase', 'unknown'))}.{event_type}",
                        context=new_context(),
                        parent=iter_ctx,
                        resource=_RESOURCE,
                        attributes=_step_attributes(step),
                        kind=SpanKind.CLIENT if event_type in _TOOL_EVENT_TYPES else SpanKind.INTERNAL,
                        status=Status(StatusCode.OK if step.get("ok", True) else StatusCode.ERROR),
                        start_time=int(step_time * 1e9),
                        end_time=int(step_time * 1e9),
                    )
                )

        root_start = task.started_at.timestamp() if task.started_at else earliest
        root_end = task.completed_at.timestamp() if task.completed_at else latest
        spans.insert(
            0,
            ReadableSpan(
                name="agent_task",
                context=root_ctx,
                parent=None,
                resource=_RESOURCE,
                attributes=_attributes(
                    {
                        "task.id": str(task.id),
                        "tenant_id": str(task.tenant_id),
                        "status": task.status.value,
                        "model_role": task.model_role.value if task.model_role else None,
                        "total_prompt_tokens": task.total_prompt_tokens or 0,
                        "total_completion_tokens": task.total_completion_tokens or 0,
                    }
                ),
                kind=SpanKind.INTERNAL,
                status=Status(StatusCode.ERROR if task.status == TaskStatus.FAILED else StatusCode.OK),
                start_time=int(root_start * 1e9),
                end_time=int(root_end * 1e9),
            ),
        )

        payload = MessageToDict(encode_spans(spans))
        return TaskOtelExport(
            tenant_id=task.tenant_id,
            trace_id_hex=format(trace_id, "032x"),
            span_count=len(spans),
            payload=payload,
        )
