# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Shared pytest fixtures.

Tests marked `integration` require real backing services (Postgres, Redis,
Qdrant, Docker for the sandbox daemon) — no mocks. Run `docker compose up -d
postgres redis qdrant` (and `python -m uvicorn src.sandbox.daemon:app`
separately, since it needs the host's Docker socket) before running these.
Unit tests have no such requirement and run anywhere.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires real Postgres/Redis/Qdrant/Docker")


@pytest.fixture(autouse=True)
async def _reset_redis_pool_per_test():
    """
    src.api.middleware.rate_limiter.get_redis() caches a module-level
    connection pool. pytest-asyncio gives each async test function its own
    event loop by default, so a pool created in one test's loop is invalid
    in the next test's loop ("RuntimeError: Event loop is closed") — close
    and clear it after every test so the next one creates a fresh
    connection bound to its own loop, exactly like a fresh process would.
    """
    yield
    from src.api.middleware import rate_limiter

    if rate_limiter._redis_pool is not None:
        await rate_limiter.close_redis()


@pytest.fixture(autouse=True)
async def _reset_db_engine_per_test():
    """
    Same per-test-event-loop problem as Redis above, for
    src.db.connection's module-level async engine/session-factory: a real
    RuntimeError ("Event loop is closed") caught by actually running
    tests/test_memory_store.py's real-Postgres tests back to back, not a
    hypothetical — close and clear the engine after every test so the next
    one creates a fresh one bound to its own loop.
    """
    yield
    from src.db import connection

    if connection._engine is not None:
        await connection.close_db()


def _integration_env_ready() -> bool:
    return bool(os.environ.get("DATABASE_URL")) and bool(os.environ.get("REDIS_URL"))


requires_integration_env = pytest.mark.skipif(
    not _integration_env_ready(),
    reason="Integration tests need DATABASE_URL/REDIS_URL pointed at real services — see tests/conftest.py",
)
