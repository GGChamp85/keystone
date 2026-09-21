# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real, no-mock unit tests for src/orchestrator/circuit_breaker.py."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.orchestrator.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerTripped,
)
from src.orchestrator.state import AgentState


def _state(**overrides) -> AgentState:
    state = AgentState(task_id=uuid4(), task_description="test task")
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


async def test_trips_on_max_iterations():
    breaker = CircuitBreaker(CircuitBreakerConfig(max_iterations=3))
    breaker.start()
    with pytest.raises(CircuitBreakerTripped, match="Max iterations"):
        await breaker.check(_state(iteration=3))


async def test_does_not_trip_below_max_iterations():
    breaker = CircuitBreaker(CircuitBreakerConfig(max_iterations=3))
    breaker.start()
    await breaker.check(_state(iteration=2))  # must not raise


async def test_trips_on_token_budget_exhausted():
    breaker = CircuitBreaker(CircuitBreakerConfig(max_tokens_per_task=1000))
    breaker.start()
    with pytest.raises(CircuitBreakerTripped, match="Token budget"):
        await breaker.check(_state(total_prompt_tokens=800, total_completion_tokens=300))


async def test_trips_on_consecutive_test_failures():
    breaker = CircuitBreaker(CircuitBreakerConfig(max_consecutive_test_failures=2))
    breaker.start()
    with pytest.raises(CircuitBreakerTripped, match="consecutive test failures"):
        await breaker.check(_state(consecutive_test_failures=2))


async def test_trips_on_consecutive_review_failures():
    breaker = CircuitBreaker(CircuitBreakerConfig(max_consecutive_review_failures=2))
    breaker.start()
    with pytest.raises(CircuitBreakerTripped, match="consecutive review rejections"):
        await breaker.check(_state(consecutive_review_failures=2))


async def test_trips_on_wall_clock_timeout():
    import asyncio

    breaker = CircuitBreaker(CircuitBreakerConfig(max_wall_clock_seconds=0))
    breaker.start()
    await asyncio.sleep(0.05)
    with pytest.raises(CircuitBreakerTripped, match="Wall-clock timeout"):
        await breaker.check(_state())


async def test_healthy_state_does_not_trip():
    breaker = CircuitBreaker(CircuitBreakerConfig())
    breaker.start()
    await breaker.check(_state(iteration=1, consecutive_test_failures=0, consecutive_review_failures=0))
