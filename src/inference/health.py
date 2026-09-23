# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — per-role endpoint health: cached probes and a circuit breaker.

The router used to call `GET /models` on the model endpoint for EVERY
request and, when nothing answered, hand back the unhealthy client
anyway. Two real production problems: a probe per request doubles the
round trips to a remote (RunPod) endpoint, and a scale-to-zero worker
that is cold on the first request after idle would fail every one of
those probes and still be "served". This module fixes both:

- **Cached health**: a probe's result is reused for
  `Settings.inference_health_cache_seconds` (default 10 s), so a burst of
  requests costs one probe, not one each.
- **Breaker per role**: `Settings.inference_breaker_failure_threshold`
  consecutive real failures (a failed probe, or a request that errored)
  open the breaker for `Settings.inference_breaker_open_seconds`; while
  open the role is skipped by the fallback chain without a probe, and one
  request is let through when the cooldown ends (half-open) to see if the
  endpoint is back. A success closes it.
- **Never serve the unhealthy**: when no role in the chain is available
  the router raises `NoHealthyModelError`, which the gateway turns into
  HTTP 503 with a `Retry-After` header — an honest answer instead of a
  request that fails downstream with a confusing error.

Pure in-memory per process (each API replica keeps its own view); state is
small, and a shared view would just add a Redis round trip to the hot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

from src.config import get_settings
from src.observability import upstream_errors_total

logger = structlog.get_logger(__name__)


class _Probeable(Protocol):
    async def health(self) -> bool: ...


class NoHealthyModelError(RuntimeError):
    """No endpoint in the requested role's fallback chain is available right now."""

    def __init__(self, role: str, retry_after_seconds: int):
        self.role = role
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"no healthy model endpoint for role {role!r}; retry in {retry_after_seconds}s")


@dataclass
class RoleHealth:
    healthy: bool | None = None  # None = never probed
    probed_at: float = 0.0
    consecutive_failures: int = 0
    open_until: float = 0.0  # breaker open while now < open_until
    last_error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)  # last transitions, for /health and the trace

    def _note(self, event: str, **extra: Any) -> None:
        self.history.append({"event": event, "at": time.time(), **extra})
        if len(self.history) > 50:
            del self.history[:-50]


class EndpointHealthRegistry:
    def __init__(self) -> None:
        self._roles: dict[str, RoleHealth] = {}

    def reset(self) -> None:
        self._roles.clear()

    def state(self, role: str) -> RoleHealth:
        return self._roles.setdefault(role, RoleHealth())

    # ── breaker ───────────────────────────────────────────────

    def is_open(self, role: str, now: float | None = None) -> bool:
        return (now or time.monotonic()) < self.state(role).open_until

    def record_success(self, role: str) -> None:
        st = self.state(role)
        if st.consecutive_failures or st.open_until:
            st._note("closed", after_failures=st.consecutive_failures)
            logger.info("inference_health.breaker_closed", role=role)
        st.consecutive_failures = 0
        st.open_until = 0.0
        st.healthy = True
        st.probed_at = time.monotonic()
        st.last_error = None

    def record_failure(self, role: str, error: str) -> None:
        settings = get_settings()
        upstream_errors_total.labels(role).inc()
        st = self.state(role)
        st.consecutive_failures += 1
        st.healthy = False
        st.probed_at = time.monotonic()
        st.last_error = error
        threshold = settings.inference_breaker_failure_threshold
        if threshold > 0 and st.consecutive_failures >= threshold and not self.is_open(role):
            st.open_until = time.monotonic() + settings.inference_breaker_open_seconds
            st._note("opened", failures=st.consecutive_failures, error=error)
            logger.warning(
                "inference_health.breaker_opened",
                role=role,
                failures=st.consecutive_failures,
                open_seconds=settings.inference_breaker_open_seconds,
                error=error,
            )

    # ── cached probe ──────────────────────────────────────────

    async def is_healthy(self, role: str, client: _Probeable) -> bool:
        """Cached `client.health()`; a probe failure counts toward the breaker, a success closes it.
        Returns False without probing while the breaker is open."""
        st = self.state(role)
        now = time.monotonic()
        if self.is_open(role, now):
            return False
        ttl = get_settings().inference_health_cache_seconds
        if st.healthy is not None and now - st.probed_at < ttl:
            return st.healthy
        try:
            healthy = bool(await client.health())
        except Exception as exc:  # a probe that raises is a failure, not an exception for the caller
            self.record_failure(role, f"probe raised: {exc}")
            return False
        if healthy:
            self.record_success(role)
        else:
            self.record_failure(role, "probe returned unhealthy")
        return healthy

    def retry_after_seconds(self, roles: list[str]) -> int:
        """How long until the soonest breaker among `roles` half-opens (at least 1 s)."""
        now = time.monotonic()
        waits = [self.state(r).open_until - now for r in roles if self.is_open(r, now)]
        if not waits:
            return max(1, get_settings().inference_health_cache_seconds)
        return max(1, int(min(waits)) + 1)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        return {
            role: {
                "healthy": st.healthy,
                "breaker": "open" if self.is_open(role, now) else "closed",
                "consecutive_failures": st.consecutive_failures,
                "open_for_seconds": max(0, int(st.open_until - now)) if self.is_open(role, now) else 0,
                "last_error": st.last_error,
                "probed_seconds_ago": int(now - st.probed_at) if st.probed_at else None,
            }
            for role, st in self._roles.items()
        }


endpoint_health = EndpointHealthRegistry()
