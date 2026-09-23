# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — `keystone doctor` real infrastructure diagnostics.

Unlike GET /health (src/api/routes/health.py), which only exists once the
app is already up, every check here connects to configured infrastructure
directly — asyncpg to Postgres, redis.asyncio to Redis, real HTTP to
Qdrant/the sandbox daemon/model endpoints/the git host — so it's useful
for diagnosing a setup that doesn't work *yet*, the actual point of a
`doctor` command. Every check reports exactly what's wrong and, where
there's an obvious fix, what to do about it — never just "failed".
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum

import httpx

from src.config import _WEAK_SECRET_SENTINELS, Settings


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str


async def check_postgres(settings: Settings) -> CheckResult:
    import asyncpg

    if settings.database_url is None:
        return CheckResult("Postgres", CheckStatus.FAIL, "DATABASE_URL is not set")
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        conn = await asyncpg.connect(dsn, timeout=5.0)
        try:
            await conn.execute("SELECT 1")
        finally:
            await conn.close()
        return CheckResult(
            "Postgres", CheckStatus.OK, f"reachable at {settings.postgres_host}:{settings.postgres_port}"
        )
    except Exception as exc:
        return CheckResult(
            "Postgres", CheckStatus.FAIL, f"cannot connect ({exc}) — is it running? Check DATABASE_URL/POSTGRES_*"
        )


async def check_redis(settings: Settings) -> CheckResult:
    import redis.asyncio as aioredis

    try:
        client = aioredis.from_url(settings.redis_url, socket_timeout=5.0)
        try:
            await client.ping()
        finally:
            await client.aclose()
        return CheckResult("Redis", CheckStatus.OK, "reachable and responding to PING")
    except Exception as exc:
        return CheckResult("Redis", CheckStatus.FAIL, f"cannot connect ({exc}) — is it running? Check REDIS_URL")


async def check_qdrant(settings: Settings) -> CheckResult:
    scheme = "https" if settings.qdrant_use_tls else "http"
    url = f"{scheme}://{settings.qdrant_host}:{settings.qdrant_port}/collections"
    headers = {}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key.get_secret_value()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return CheckResult("Qdrant", CheckStatus.OK, f"reachable at {settings.qdrant_host}:{settings.qdrant_port}")
        return CheckResult("Qdrant", CheckStatus.FAIL, f"reachable but returned HTTP {resp.status_code}")
    except Exception as exc:
        return CheckResult(
            "Qdrant", CheckStatus.FAIL, f"cannot connect ({exc}) — is it running? Check QDRANT_HOST/PORT"
        )


async def check_sandbox_daemon(settings: Settings) -> CheckResult:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.sandbox_daemon_url.rstrip('/')}/health")
        if resp.status_code == 200:
            body = resp.json()
            if not body.get("docker_ok", True):
                return CheckResult("Sandbox daemon", CheckStatus.WARN, "reachable, but its Docker socket check failed")
            return CheckResult("Sandbox daemon", CheckStatus.OK, f"reachable, backend={body.get('backend', '?')}")
        return CheckResult("Sandbox daemon", CheckStatus.FAIL, f"reachable but returned HTTP {resp.status_code}")
    except Exception as exc:
        return CheckResult(
            "Sandbox daemon",
            CheckStatus.FAIL,
            f"cannot connect ({exc}) — no agentic coding task can run without this. "
            f"Check SANDBOX_DAEMON_URL and that `make sandbox-images` has been run.",
        )


async def check_model_endpoint(role: str, url: str) -> CheckResult:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/models")
        if resp.status_code == 200:
            return CheckResult(f"Model endpoint ({role})", CheckStatus.OK, f"reachable at {url}")
        return CheckResult(f"Model endpoint ({role})", CheckStatus.WARN, f"{url} returned HTTP {resp.status_code}")
    except Exception as exc:
        return CheckResult(
            f"Model endpoint ({role})",
            CheckStatus.WARN,
            f"{url} unreachable ({exc}) — expected if you haven't started a vLLM endpoint or "
            f"benchmarks/frontier_proxy.py for this role yet",
        )


async def check_runpod(settings: Settings) -> CheckResult:
    """
    RunPod Serverless is optional, so an unconfigured pair is SKIP, not a
    failure. Half-configured (key without URL, or URL without key) is a
    WARN naming the missing half — the most common real mistake. When
    both are present this makes a real authenticated call to the
    endpoint's OpenAI-compatible `/models` route, which is served by
    RunPod's endpoint layer rather than a worker, so it answers even when
    the endpoint has scaled to zero workers — a cold endpoint still
    reports OK here, and rightly so.
    """
    key = settings.runpod_api_key.get_secret_value() if settings.runpod_api_key else ""
    url = (settings.runpod_endpoint_url or "").rstrip("/")
    if not key and not url:
        return CheckResult("RunPod", CheckStatus.SKIP, "RUNPOD_API_KEY/RUNPOD_ENDPOINT_URL not set — fine if unused")
    if key and not url:
        return CheckResult("RunPod", CheckStatus.WARN, "RUNPOD_API_KEY is set but RUNPOD_ENDPOINT_URL is empty")
    if url and not key:
        return CheckResult("RunPod", CheckStatus.WARN, "RUNPOD_ENDPOINT_URL is set but RUNPOD_API_KEY is empty")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"})
        if resp.status_code == 200:
            return CheckResult("RunPod", CheckStatus.OK, f"endpoint reachable and authenticated at {url}")
        if resp.status_code in (401, 403):
            return CheckResult("RunPod", CheckStatus.FAIL, f"{url} rejected the key (HTTP {resp.status_code})")
        return CheckResult("RunPod", CheckStatus.WARN, f"{url}/models returned HTTP {resp.status_code}")
    except Exception as exc:
        return CheckResult("RunPod", CheckStatus.FAIL, f"cannot reach {url} ({exc})")


async def check_git_host(settings: Settings) -> CheckResult:
    if settings.git_host_api_url == "http://gitea.internal.keystone.local:3000/api/v1":
        return CheckResult(
            "Git host",
            CheckStatus.WARN,
            "still the default placeholder — agentic coding tasks that open PRs need a real git host. "
            "See the README's 'Connect your own git server'.",
        )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.git_host_api_url.rstrip('/')}/version")
        if resp.status_code == 200:
            return CheckResult("Git host", CheckStatus.OK, f"reachable at {settings.git_host_api_url}")
        return CheckResult(
            "Git host", CheckStatus.WARN, f"{settings.git_host_api_url} returned HTTP {resp.status_code}"
        )
    except Exception as exc:
        return CheckResult(
            "Git host", CheckStatus.FAIL, f"configured but unreachable at {settings.git_host_api_url} ({exc})"
        )


async def check_package_mirrors(settings: Settings) -> list[CheckResult]:
    results = []
    for name, url in (
        ("pip index", settings.pip_index_url),
        ("npm registry", settings.npm_registry_url),
        ("Go proxy", settings.go_proxy_url),
    ):
        if not url:
            results.append(
                CheckResult(
                    f"Mirror ({name})", CheckStatus.SKIP, "not configured — public internet will be used if reachable"
                )
            )
            continue
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
            results.append(
                CheckResult(f"Mirror ({name})", CheckStatus.OK if resp.status_code < 500 else CheckStatus.FAIL, url)
            )
        except Exception as exc:
            results.append(CheckResult(f"Mirror ({name})", CheckStatus.FAIL, f"{url} unreachable ({exc})"))
    return results


def check_secrets(settings: Settings) -> list[CheckResult]:
    """The real, previously-unused _WEAK_SECRET_SENTINELS set from
    src/config.py — defined but never actually checked against until now."""
    results = []
    secrets_to_check = [
        ("POSTGRES_PASSWORD", settings.postgres_password.get_secret_value()),
        (
            "KEYSTONE_ROOT_ADMIN_TOKEN",
            settings.keystone_root_admin_token.get_secret_value() if settings.keystone_root_admin_token else "",
        ),
        ("VS_SECRET_KEY", settings.vs_secret_key.get_secret_value()),
    ]
    if settings.redis_password:
        secrets_to_check.append(("REDIS_PASSWORD", settings.redis_password.get_secret_value()))
    if settings.qdrant_api_key:
        secrets_to_check.append(("QDRANT_API_KEY", settings.qdrant_api_key.get_secret_value()))

    if settings.secret_key_is_ephemeral:
        results.append(
            CheckResult(
                "Secret (VS_SECRET_KEY)",
                CheckStatus.FAIL if settings.is_production else CheckStatus.WARN,
                "not set — a random one is generated per process, so API-key hashes (peppered with it) stop "
                "verifying after a restart; set a stable value: `openssl rand -hex 32`",
            )
        )
        secrets_to_check = [s for s in secrets_to_check if s[0] != "VS_SECRET_KEY"]

    for name, value in secrets_to_check:
        if value.lower() in _WEAK_SECRET_SENTINELS:
            severity = CheckStatus.FAIL if settings.is_production else CheckStatus.WARN
            results.append(
                CheckResult(
                    f"Secret ({name})",
                    severity,
                    "still a placeholder/default value — generate a real one with `openssl rand -hex 32`",
                )
            )
        else:
            results.append(CheckResult(f"Secret ({name})", CheckStatus.OK, "not a known placeholder"))
    return results


def check_gpu() -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return CheckResult(
            "GPU",
            CheckStatus.SKIP,
            "no nvidia-smi found — fine if you're using benchmarks/frontier_proxy.py instead of self-hosted GPU models",
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().splitlines()
            return CheckResult("GPU", CheckStatus.OK, f"{len(gpus)} GPU(s) detected: {'; '.join(gpus)}")
        return CheckResult("GPU", CheckStatus.WARN, "nvidia-smi present but reported no GPUs")
    except Exception as exc:
        return CheckResult("GPU", CheckStatus.WARN, f"nvidia-smi present but failed to run ({exc})")


def check_docker() -> CheckResult:
    if shutil.which("docker") is None:
        return CheckResult(
            "Docker",
            CheckStatus.FAIL,
            "docker CLI not found on PATH — required for the sandbox daemon and the compose stack",
        )
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)  # noqa: S607
        if result.returncode == 0:
            return CheckResult("Docker", CheckStatus.OK, "daemon reachable")
        return CheckResult(
            "Docker",
            CheckStatus.FAIL,
            f"docker CLI present but the daemon isn't reachable: {result.stderr.strip()}",
        )
    except Exception as exc:
        return CheckResult("Docker", CheckStatus.FAIL, f"failed to run `docker info` ({exc})")


async def run_all_checks(settings: Settings) -> list[CheckResult]:
    results: list[CheckResult] = [check_docker(), check_gpu()]
    results.append(await check_postgres(settings))
    results.append(await check_redis(settings))
    results.append(await check_qdrant(settings))
    results.append(await check_sandbox_daemon(settings))
    for role, url in (
        ("coding", settings.vllm_coding_url),
        ("coding_fallback", settings.vllm_coding_fallback_url),
        ("reasoning", settings.vllm_reasoning_url),
    ):
        results.append(await check_model_endpoint(role, url))
    results.append(await check_runpod(settings))
    results.append(await check_git_host(settings))
    results.extend(await check_package_mirrors(settings))
    results.extend(check_secrets(settings))
    return results
