# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for the `keystone ingest` CLI command (src/cli/main.py) — the
actual Typer command, run via CliRunner, making real HTTP calls against a
real running Keystone server, real Postgres/Qdrant, and (for the happy-path
clone test) a real git host. No mocking of the HTTP layer, the API route,
or CodeIngestionPipeline.

Needs a live server + KEYSTONE_ROOT_ADMIN_TOKEN to bootstrap a tenant/key,
matching tests/test_cli_admin.py's own real-infra-required pattern. The
happy-path clone test additionally needs the live server's own configured
GIT_ALLOWED_HOSTS to include a reachable host — set
KEYSTONE_INGEST_TEST_REPO_URL to a real https clone URL on that allowlist
(and KEYSTONE_INGEST_TEST_REPO_BRANCH if its default branch isn't "main")
to exercise it; it self-skips otherwise, same as
tests/test_git_workflow_integration.py's GITEA_TEST_REPO_URL gate.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest
from typer.testing import CliRunner

from src.cli.main import app

pytestmark = pytest.mark.integration

requires_admin_cli_env = pytest.mark.skipif(
    not (os.environ.get("KEYSTONE_INFERENCE_URL") and os.environ.get("KEYSTONE_ROOT_ADMIN_TOKEN")),
    reason="Needs a live Keystone server: KEYSTONE_INFERENCE_URL + KEYSTONE_ROOT_ADMIN_TOKEN",
)

requires_ingest_test_repo = pytest.mark.skipif(
    not os.environ.get("KEYSTONE_INGEST_TEST_REPO_URL"),
    reason="Needs KEYSTONE_INGEST_TEST_REPO_URL — a real clone URL the live server's GIT_ALLOWED_HOSTS permits",
)

runner = CliRunner()


@pytest.fixture
async def cleanup_tenant():
    """Same real-DB-cleanup fixture as tests/test_cli_admin.py — an async
    fixture so pytest-asyncio manages its event loop, avoiding the "Event
    loop is closed" conflict with conftest.py's autouse async fixtures."""
    from sqlalchemy import delete

    from src.db.connection import get_db_context
    from src.db.models import APIKey, AuditLog, CodebaseIndex, Tenant, User

    tenant_ids: list[str] = []
    yield tenant_ids

    async with get_db_context() as db:
        for tid in tenant_ids:
            await db.execute(delete(CodebaseIndex).where(CodebaseIndex.tenant_id == tid))
            await db.execute(delete(AuditLog).where(AuditLog.target_id == tid))
            await db.execute(delete(APIKey).where(APIKey.tenant_id == tid))
            await db.execute(delete(User).where(User.tenant_id == tid))
            await db.execute(delete(Tenant).where(Tenant.id == tid))


def _bootstrap_tenant_and_key(cleanup_tenant, name: str) -> str:
    """Creates the tenant via the real `keystone tenants create` CLI command
    (simple, single-line output, safe to parse) and the API key via a
    direct HTTP call to the real admin route — keys-create's own output is
    Rich-wrapped across lines at CliRunner's terminal width, which makes
    the full ks-... value unsafe to recover by parsing text."""
    email = f"{uuid.uuid4().hex[:8]}@cli-ingest-test.dev"
    create_result = runner.invoke(app, ["tenants", "create", name, email])
    assert create_result.exit_code == 0, create_result.output
    tenant_id = create_result.output.split("Created tenant")[1].strip().split()[0]
    cleanup_tenant.append(tenant_id)

    resp = httpx.post(
        f"{os.environ['KEYSTONE_INFERENCE_URL']}/v1/admin/tenants/{tenant_id}/keys",
        headers={"Authorization": f"Bearer {os.environ['KEYSTONE_ROOT_ADMIN_TOKEN']}"},
        json={"name": f"{name}-key", "scopes": ["agent"], "expires_in_days": None},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["key"]


@requires_admin_cli_env
@requires_ingest_test_repo
def test_ingest_clones_a_real_repo_and_is_incremental_on_a_second_run(cleanup_tenant, monkeypatch):
    key_prefix = _bootstrap_tenant_and_key(cleanup_tenant, "cli-ingest-repo-test")
    monkeypatch.setenv("KEYSTONE_API_KEY", key_prefix)

    repo_url = os.environ["KEYSTONE_INGEST_TEST_REPO_URL"]
    branch = os.environ.get("KEYSTONE_INGEST_TEST_REPO_BRANCH", "main")
    first = runner.invoke(app, ["ingest", repo_url, "--branch", branch])
    assert first.exit_code == 0, first.output
    assert "Ingested" in first.output
    assert "chunk(s) created" in first.output

    second = runner.invoke(app, ["ingest", repo_url, "--branch", branch])
    assert second.exit_code == 0, second.output
    assert "0 chunk(s) created" in second.output


@requires_admin_cli_env
def test_ingest_rejects_a_disallowed_url_scheme_with_a_clean_error_not_a_500(cleanup_tenant, monkeypatch):
    key_prefix = _bootstrap_tenant_and_key(cleanup_tenant, "cli-ingest-scheme-test")
    monkeypatch.setenv("KEYSTONE_API_KEY", key_prefix)

    result = runner.invoke(app, ["ingest", "file:///etc/passwd"])
    assert result.exit_code == 1
    assert "Internal Server Error" not in result.output
    assert "scheme" in result.output


def test_ingest_fails_clearly_without_a_tenant_api_key(monkeypatch):
    monkeypatch.setenv("KEYSTONE_INFERENCE_URL", "http://localhost:1")
    monkeypatch.delenv("KEYSTONE_API_KEY", raising=False)
    result = runner.invoke(app, ["ingest", "https://github.com/octocat/Hello-World"])
    assert result.exit_code == 1
    assert "KEYSTONE_API_KEY" in result.output
