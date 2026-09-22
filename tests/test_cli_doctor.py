# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for src/cli/doctor.py — every check connects to real
infrastructure (or a deliberately-unreachable one, to prove failure
reporting is real too), no mocked clients.
"""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from src.cli.doctor import (
    CheckStatus,
    check_docker,
    check_git_host,
    check_gpu,
    check_package_mirrors,
    check_postgres,
    check_qdrant,
    check_redis,
    check_runpod,
    check_sandbox_daemon,
    check_secrets,
)
from src.config import Settings

pytestmark = pytest.mark.integration


def _settings(**overrides) -> Settings:
    """
    Settings() already reads DATABASE_URL/REDIS_URL/QDRANT_HOST/
    QDRANT_PORT/QDRANT_API_KEY/SANDBOX_DAEMON_URL from the real ambient
    environment (this dev machine's docker-compose port mappings, or
    CI's — see .github/workflows/ci.yml's service containers, which use
    different ports than any local dev setup) — so these checks must be
    built against whatever's actually there, never hardcoded literal
    connection details for one specific environment.
    """
    base: dict = {
        "postgres_password": SecretStr("testpass"),
        "keystone_root_admin_token": SecretStr("a-real-non-placeholder-token"),
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


async def test_postgres_check_succeeds_against_the_real_test_database():
    result = await check_postgres(_settings())
    assert result.status == CheckStatus.OK


async def test_postgres_check_fails_honestly_against_an_unreachable_port():
    result = await check_postgres(_settings(database_url="postgresql+asyncpg://x:x@localhost:1/nope"))
    assert result.status == CheckStatus.FAIL
    assert "cannot connect" in result.message


async def test_runpod_check_skips_when_unconfigured():
    result = await check_runpod(_settings(runpod_api_key=None, runpod_endpoint_url=None))
    assert result.status == CheckStatus.SKIP


async def test_runpod_check_warns_when_only_the_key_is_set():
    result = await check_runpod(_settings(runpod_api_key=SecretStr("rpa_not_real"), runpod_endpoint_url=None))
    assert result.status == CheckStatus.WARN
    assert "RUNPOD_ENDPOINT_URL is empty" in result.message


async def test_runpod_check_warns_when_only_the_url_is_set():
    result = await check_runpod(
        _settings(runpod_api_key=None, runpod_endpoint_url="https://api.runpod.ai/v2/x/openai/v1")
    )
    assert result.status == CheckStatus.WARN
    assert "RUNPOD_API_KEY is empty" in result.message


async def test_runpod_check_fails_honestly_against_an_unreachable_endpoint():
    result = await check_runpod(
        _settings(runpod_api_key=SecretStr("rpa_not_real"), runpod_endpoint_url="http://localhost:1/openai/v1")
    )
    assert result.status == CheckStatus.FAIL
    assert "cannot reach" in result.message


@pytest.mark.skipif(
    not (os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_URL")),
    reason="Needs a real RunPod Serverless endpoint: RUNPOD_API_KEY + RUNPOD_ENDPOINT_URL",
)
async def test_runpod_check_succeeds_against_the_real_endpoint():
    result = await check_runpod(
        _settings(
            runpod_api_key=SecretStr(os.environ["RUNPOD_API_KEY"]),
            runpod_endpoint_url=os.environ["RUNPOD_ENDPOINT_URL"],
        )
    )
    assert result.status == CheckStatus.OK, result.message


async def test_redis_check_succeeds_against_the_real_test_redis():
    result = await check_redis(_settings())
    assert result.status == CheckStatus.OK


async def test_redis_check_fails_honestly_against_an_unreachable_port():
    result = await check_redis(_settings(redis_url="redis://localhost:1/0"))
    assert result.status == CheckStatus.FAIL


async def test_qdrant_check_succeeds_against_the_real_test_qdrant():
    result = await check_qdrant(_settings())
    assert result.status == CheckStatus.OK


async def test_qdrant_check_fails_honestly_against_an_unreachable_port():
    result = await check_qdrant(_settings(qdrant_port=1))
    assert result.status == CheckStatus.FAIL


async def test_sandbox_daemon_check_succeeds_against_the_real_running_daemon():
    result = await check_sandbox_daemon(_settings())
    assert result.status == CheckStatus.OK
    assert "gvisor" in result.message


async def test_sandbox_daemon_check_fails_honestly_when_unreachable():
    result = await check_sandbox_daemon(_settings(sandbox_daemon_url="http://localhost:1"))
    assert result.status == CheckStatus.FAIL


async def test_git_host_check_warns_on_the_default_placeholder():
    result = await check_git_host(_settings())
    assert result.status == CheckStatus.WARN
    assert "placeholder" in result.message


async def test_git_host_check_fails_honestly_when_configured_but_unreachable():
    result = await check_git_host(_settings(git_host_api_url="https://this-host-does-not-exist.invalid/api/v1"))
    assert result.status == CheckStatus.FAIL


async def test_package_mirrors_skip_when_unconfigured():
    results = await check_package_mirrors(_settings())
    assert all(r.status == CheckStatus.SKIP for r in results)


def test_secrets_check_flags_the_real_documented_placeholder_values():
    results = check_secrets(_settings(postgres_password=SecretStr("changeme")))
    postgres_result = next(r for r in results if "POSTGRES_PASSWORD" in r.name)
    assert postgres_result.status in (CheckStatus.WARN, CheckStatus.FAIL)
    assert "placeholder" in postgres_result.message


def test_secrets_check_passes_a_real_non_placeholder_value():
    results = check_secrets(_settings())
    postgres_result = next(r for r in results if "POSTGRES_PASSWORD" in r.name)
    assert postgres_result.status == CheckStatus.OK


def test_docker_check_runs_for_real():
    result = check_docker()
    assert result.status in (CheckStatus.OK, CheckStatus.FAIL)


def test_gpu_check_runs_for_real_and_never_raises():
    result = check_gpu()
    assert result.status in (CheckStatus.OK, CheckStatus.WARN, CheckStatus.SKIP)
