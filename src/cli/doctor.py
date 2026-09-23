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


def check_tls(settings: Settings) -> CheckResult:
    """The API's certificate: present, parseable, not expired, and how long it has left."""
    if not settings.tls_cert_path and not settings.tls_key_path:
        return CheckResult(
            "TLS", CheckStatus.SKIP, "TLS_CERT_PATH/TLS_KEY_PATH not set — TLS terminates elsewhere (nginx/ingress)"
        )
    from pathlib import Path

    cert_path = Path(settings.tls_cert_path or "")
    key_path = Path(settings.tls_key_path or "")
    if not cert_path.is_file() or not key_path.is_file():
        return CheckResult(
            "TLS", CheckStatus.FAIL, f"certificate or key file missing ({cert_path}, {key_path}) — `make certs`"
        )
    try:
        from datetime import UTC, datetime

        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except Exception as exc:
        return CheckResult("TLS", CheckStatus.FAIL, f"{cert_path} is not a PEM certificate ({exc})")
    days_left = (cert.not_valid_after_utc - datetime.now(UTC)).days
    subject = cert.subject.rfc4514_string()
    if days_left < 0:
        return CheckResult(
            "TLS", CheckStatus.FAIL, f"{subject} expired {-days_left} day(s) ago — re-issue with `make certs-ca`"
        )
    if days_left < 14:
        return CheckResult("TLS", CheckStatus.WARN, f"{subject} expires in {days_left} day(s) — re-issue soon")
    return CheckResult("TLS", CheckStatus.OK, f"{subject} valid for {days_left} more day(s)")


def check_disk(settings: Settings) -> list[CheckResult]:
    """Free space where the platform writes: the fine-tuning output/data dirs when they exist, else the cwd."""
    from pathlib import Path

    results = []
    candidates = [settings.finetuning_output_dir, settings.finetuning_data_dir, "."]
    seen: set[str] = set()
    for c in candidates:
        path = Path(c)
        if not path.exists():
            continue
        root = str(path.resolve())
        if root in seen:
            continue
        seen.add(root)
        usage = shutil.disk_usage(root)
        free_pct = usage.free / usage.total * 100 if usage.total else 0
        free_gb = usage.free / 1e9
        label = f"Disk ({c})" if c != "." else "Disk (working directory)"
        if free_pct < 5:
            results.append(
                CheckResult(
                    label,
                    CheckStatus.FAIL,
                    f"{free_gb:.1f} GB free ({free_pct:.0f}%) — model pulls and adapters will fail",
                )
            )
        elif free_pct < 15:
            results.append(CheckResult(label, CheckStatus.WARN, f"{free_gb:.1f} GB free ({free_pct:.0f}%)"))
        else:
            results.append(CheckResult(label, CheckStatus.OK, f"{free_gb:.1f} GB free ({free_pct:.0f}%)"))
    return results


async def check_migrations(settings: Settings) -> CheckResult:
    """The database's Alembic revision against this code's head — an upgrade that forgot `alembic upgrade head`
    shows up here, before a request hits a missing column."""
    import asyncpg
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    if settings.database_url is None:
        return CheckResult("Migrations", CheckStatus.SKIP, "DATABASE_URL is not set")
    try:
        heads = ScriptDirectory.from_config(Config("alembic.ini")).get_heads()
    except Exception as exc:
        return CheckResult(
            "Migrations", CheckStatus.SKIP, f"cannot read the migration scripts here ({exc}) — run from a checkout"
        )
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        conn = await asyncpg.connect(dsn, timeout=5.0)
        try:
            rows = await conn.fetch("SELECT version_num FROM alembic_version")
        finally:
            await conn.close()
    except Exception as exc:
        if "alembic_version" in str(exc):
            return CheckResult(
                "Migrations",
                CheckStatus.FAIL,
                "no alembic_version table — the database has never been migrated: `alembic upgrade head`",
            )
        return CheckResult("Migrations", CheckStatus.SKIP, f"Postgres unreachable ({exc}); see the Postgres check")
    current = {r["version_num"] for r in rows}
    if current == set(heads):
        return CheckResult("Migrations", CheckStatus.OK, f"database at head {', '.join(sorted(heads))}")
    return CheckResult(
        "Migrations",
        CheckStatus.FAIL,
        f"database at {', '.join(sorted(current)) or 'nothing'}, code head is {', '.join(sorted(heads))} — "
        "run `alembic upgrade head`",
    )


async def check_app_ready(base_url: str | None) -> CheckResult:
    """The running application's own readiness (`/health/ready`), when KEYSTONE_INFERENCE_URL points at one."""
    if not base_url:
        return CheckResult(
            "App /health/ready",
            CheckStatus.SKIP,
            "KEYSTONE_INFERENCE_URL not set — the checks above talk to the services directly",
        )
    url = f"{base_url.rstrip('/')}/health/ready"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
    except Exception as exc:
        return CheckResult(
            "App /health/ready",
            CheckStatus.FAIL,
            f"{url} unreachable ({exc}) — is the app up? `keystone up`, `make status`",
        )
    try:
        body = resp.json()
    except ValueError:
        body = {}
    components = body.get("components", {}) if isinstance(body, dict) else {}
    unhealthy = [k for k, v in components.items() if v not in ("healthy", "unprobed")]
    if resp.status_code == 200:
        note = f"; models: {', '.join(unhealthy)}" if unhealthy else ""
        return CheckResult("App /health/ready", CheckStatus.OK, f"ready at {base_url}{note}")
    return CheckResult(
        "App /health/ready", CheckStatus.FAIL, f"HTTP {resp.status_code}: {', '.join(unhealthy) or body}"
    )


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


async def run_all_checks(settings: Settings, *, app_base_url: str | None = None) -> list[CheckResult]:
    results: list[CheckResult] = [check_docker(), check_gpu()]
    results.extend(check_disk(settings))
    results.append(await check_postgres(settings))
    results.append(await check_migrations(settings))
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
    results.append(check_tls(settings))
    results.append(await check_app_ready(app_base_url))
    return results
