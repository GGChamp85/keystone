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

# The test suite is a deployment: API keys are hashed with this pepper (ADR 0004) and the app refuses to
# start in production without a stable one. CI sets it; a developer's shell may not.
os.environ.setdefault("VS_SECRET_KEY", "test-suite-pepper-" + "0" * 46)


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


@pytest.fixture
async def fake_vllm_server():
    """tests/fixtures/fake_vllm_server.py as a real subprocess: a real HTTP server speaking vLLM's
    OpenAI-compatible shapes (non-streaming + streaming, tool calls split across chunks, the usage-only
    final chunk), for route tests that need a real backend without a GPU. Yields its base URL."""
    import asyncio
    import socket
    import sys
    from pathlib import Path

    import httpx

    fixture = Path(__file__).parent / "fixtures" / "fake_vllm_server.py"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "fake_vllm_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        cwd=str(fixture.parent),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as probe:
            for _ in range(60):
                try:
                    await probe.get(f"{base_url}/last_payload", timeout=1.0)
                    break
                except httpx.TransportError:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("fake_vllm_server did not start in time")
        yield base_url
    finally:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)
