# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Circuit breaker for autonomous coding agents.

Prevents runaway loops by enforcing:
  1. Max iteration count per task
  2. Max total tokens per task
  3. Consecutive failure limits (test / review)
  4. Wall-clock timeout per task
  5. Tenant-level daily/monthly budget (via Redis)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from src.api.middleware.rate_limiter import get_redis
from src.observability import circuit_breaker_trips_total
from src.orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


class CircuitBreakerTripped(Exception):
    """Raised when any circuit breaker condition fires."""

    def __init__(self, reason: str, state: AgentState):
        self.reason = reason
        self.state = state
        super().__init__(reason)


@dataclass
class CircuitBreakerConfig:
    max_iterations: int = 15
    max_tokens_per_task: int = 2_000_000
    max_consecutive_test_failures: int = 3
    max_consecutive_review_failures: int = 3
    max_consecutive_quality_failures: int = 3
    max_wall_clock_seconds: int = 1800  # 30 minutes
    tenant_daily_limit: int | None = None
    tenant_monthly_limit: int | None = None


class CircuitBreaker:
    """
    Stateless checker — call `check()` before every agent iteration.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.monotonic()

    def _trip(self, reason_code: str, message: str, state: AgentState) -> CircuitBreakerTripped:
        circuit_breaker_trips_total.labels(reason_code).inc()
        return CircuitBreakerTripped(message, state)

    async def check(self, state: AgentState) -> None:
        """
        Raises CircuitBreakerTripped if any limit is exceeded.
        Must be called at the top of every graph iteration.
        """
        # 1. Iteration limit
        if state.iteration >= self.config.max_iterations:
            raise self._trip(
                "max_iterations",
                f"Max iterations reached ({state.iteration}/{self.config.max_iterations}). "
                f"The agent may be stuck in a fix-test-fail loop.",
                state,
            )

        # 2. Token budget
        if state.total_tokens >= self.config.max_tokens_per_task:
            raise self._trip(
                "token_budget",
                f"Token budget exhausted ({state.total_tokens:,}/{self.config.max_tokens_per_task:,}). "
                f"The task consumed too many tokens.",
                state,
            )

        # 3. Consecutive test failures
        if state.consecutive_test_failures >= self.config.max_consecutive_test_failures:
            raise self._trip(
                "consecutive_test_failures",
                f"Too many consecutive test failures "
                f"({state.consecutive_test_failures}/{self.config.max_consecutive_test_failures}). "
                f"The agent cannot fix the failing tests.",
                state,
            )

        # 4. Consecutive review failures
        if state.consecutive_review_failures >= self.config.max_consecutive_review_failures:
            raise self._trip(
                "consecutive_review_failures",
                f"Too many consecutive review rejections "
                f"({state.consecutive_review_failures}/{self.config.max_consecutive_review_failures}). "
                f"The reasoning critic keeps rejecting the code.",
                state,
            )

        # 5. Consecutive quality-gate failures — without this the quality
        # node (nodes/quality.py) could route CODING -> QUALITY -> FIXING
        # indefinitely, bounded only by the iteration and token limits.
        if state.consecutive_quality_failures >= self.config.max_consecutive_quality_failures:
            raise self._trip(
                "consecutive_quality_failures",
                f"Too many consecutive quality-gate failures "
                f"({state.consecutive_quality_failures}/{self.config.max_consecutive_quality_failures}). "
                f"The agent cannot get the code past lint/typecheck/security scanning.",
                state,
            )

        # 6. Wall-clock timeout
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self.config.max_wall_clock_seconds:
                raise self._trip(
                    "wall_clock_timeout",
                    f"Wall-clock timeout ({elapsed:.0f}s / {self.config.max_wall_clock_seconds}s). "
                    f"The task is taking too long.",
                    state,
                )

        # 7. Tenant daily budget (Redis check)
        if state.tenant_id and self.config.tenant_daily_limit:
            await self._check_tenant_budget(state)

    async def _check_tenant_budget(self, state: AgentState) -> None:
        limit = self.config.tenant_daily_limit
        if limit is None:
            return  # no budget configured — the caller already skips, this keeps the method self-contained
        try:
            r = await get_redis()
            day = time.strftime("%Y-%m-%d")
            daily_key = f"vs:tokens:daily:{state.tenant_id}:{day}"
            daily_used = int(await r.get(daily_key) or 0)

            if daily_used >= limit:
                raise self._trip(
                    "tenant_daily_budget",
                    f"Tenant daily token budget exhausted "
                    f"({daily_used:,}/{self.config.tenant_daily_limit:,}) "
                    f"while running agent task {state.task_id}.",
                    state,
                )

            if self.config.tenant_monthly_limit:
                month = time.strftime("%Y-%m")
                monthly_key = f"vs:tokens:monthly:{state.tenant_id}:{month}"
                monthly_used = int(await r.get(monthly_key) or 0)

                if monthly_used >= self.config.tenant_monthly_limit:
                    raise self._trip(
                        "tenant_monthly_budget",
                        f"Tenant monthly token budget exhausted "
                        f"({monthly_used:,}/{self.config.tenant_monthly_limit:,}).",
                        state,
                    )
        except CircuitBreakerTripped:
            raise
        except Exception as exc:
            logger.warning(
                "circuit_breaker.redis_check_failed",
                error=str(exc),
                task_id=str(state.task_id),
            )
