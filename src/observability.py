# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — Prometheus metrics registry.

Every custom business metric lives here so /metrics (mounted in src/main.py)
actually exposes something beyond process defaults. Scraped by the
self-hosted Prometheus (observability/prometheus/prometheus.yml).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

http_requests_total = Counter(
    "keystone_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "keystone_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

tokens_used_total = Counter(
    "keystone_tokens_used_total",
    "Total tokens consumed",
    ["tenant_id", "model_role", "kind"],  # kind: prompt | completion
)

sandbox_executions_total = Counter(
    "keystone_sandbox_executions_total",
    "Total sandbox command executions",
    ["backend", "status"],  # status: success | error | timeout
)

circuit_breaker_trips_total = Counter(
    "keystone_circuit_breaker_trips_total",
    "Total agent circuit breaker trips",
    ["reason"],
)

agent_task_duration_seconds = Histogram(
    "keystone_agent_task_duration_seconds",
    "Autonomous agent task duration in seconds",
    ["status"],
)

time_to_first_token_seconds = Histogram(
    "keystone_time_to_first_token_seconds",
    "Streamed gateway requests: seconds from request to the first content chunk",
    ["model_role"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

upstream_errors_total = Counter(
    "keystone_upstream_errors_total",
    "Model endpoint failures as seen by the router (probe failures and request errors)",
    ["model_role"],
)

route_decisions_total = Counter(
    "keystone_route_decisions_total",
    "Gateway routing decisions: which role served what was requested, and why",
    ["requested", "served", "reason"],
)

budget_rejections_total = Counter(
    "keystone_budget_rejections_total",
    "Requests and tasks refused because a tenant's monthly dollar budget is exhausted",
    ["tenant_id", "surface"],  # surface: gateway | agent
)
