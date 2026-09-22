# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for the `keystone tenants`/`keystone users`/`keystone keys-create`
CLI commands — the actual Typer commands (src/cli/main.py), run via
CliRunner, making real HTTP calls (the CLI's own httpx.Client, unmodified)
against a real running Keystone server and a real Postgres. No mocking of
the HTTP layer, the API routes, or the database.

Needs a live server the same way the rest of the `keystone` CLI does
(KEYSTONE_INFERENCE_URL) plus KEYSTONE_ROOT_ADMIN_TOKEN — self-skips
without both, matching tests/test_git_workflow_integration.py's own
real-infra-required pattern.
"""

from __future__ import annotations

import os
import uuid

import pytest
from typer.testing import CliRunner

from src.cli.main import app

pytestmark = pytest.mark.integration

requires_admin_cli_env = pytest.mark.skipif(
    not (os.environ.get("KEYSTONE_INFERENCE_URL") and os.environ.get("KEYSTONE_ROOT_ADMIN_TOKEN")),
    reason="Needs a live Keystone server: KEYSTONE_INFERENCE_URL + KEYSTONE_ROOT_ADMIN_TOKEN",
)

runner = CliRunner()


@pytest.fixture
async def cleanup_tenant():
    """
    Yields a list the test appends tenant ids to; deletes them on
    teardown. An *async* fixture, not a sync one calling its own
    asyncio.run() — this test file's tests are plain sync functions
    (CliRunner().invoke() is sync), but conftest.py's autouse
    _reset_db_engine_per_test is itself an async fixture that
    pytest-asyncio runs in a real per-test event loop regardless; a
    second, independent asyncio.run() here would bind asyncpg
    connections to a different loop than that one closes, and asyncpg's
    own cancel-on-close path then fails with "Event loop is closed".
    Letting pytest-asyncio manage this fixture in the same loop avoids
    that entirely.
    """
    from sqlalchemy import delete

    from src.db.connection import get_db_context
    from src.db.models import APIKey, AuditLog, Tenant, User

    tenant_ids: list[str] = []
    yield tenant_ids

    async with get_db_context() as db:
        for tid in tenant_ids:
            await db.execute(delete(AuditLog).where(AuditLog.target_id == tid))
            await db.execute(delete(APIKey).where(APIKey.tenant_id == tid))
            await db.execute(delete(User).where(User.tenant_id == tid))
            await db.execute(delete(Tenant).where(Tenant.id == tid))


@requires_admin_cli_env
async def test_tenant_create_makes_a_real_tenant(cleanup_tenant):
    """async def, not a sync test calling its own asyncio.run() for the DB
    check below — see cleanup_tenant's docstring for why that conflicts
    with conftest.py's autouse async fixtures. runner.invoke() itself
    stays a plain sync call; only the DB verification needs the loop."""
    email = f"{uuid.uuid4().hex[:8]}@cli-test.dev"
    result = runner.invoke(app, ["tenants", "create", "cli-test-tenant", email])
    assert result.exit_code == 0, result.output
    assert "Created tenant" in result.output

    tenant_id = result.output.split("Created tenant")[1].strip().split()[0]
    cleanup_tenant.append(tenant_id)

    from src.db.connection import get_db_context
    from src.db.models import Tenant

    async with get_db_context() as db:
        tenant = await db.get(Tenant, tenant_id)
    assert tenant is not None
    assert tenant.email == email


@requires_admin_cli_env
def test_user_add_and_list_round_trip(cleanup_tenant):
    email = f"{uuid.uuid4().hex[:8]}@cli-test.dev"
    create_result = runner.invoke(app, ["tenants", "create", "cli-user-test", email])
    tenant_id = create_result.output.split("Created tenant")[1].strip().split()[0]
    cleanup_tenant.append(tenant_id)

    user_email = f"{uuid.uuid4().hex[:8]}@cli-test.dev"
    add_result = runner.invoke(app, ["users", "add", tenant_id, "Test User", user_email, "--role", "lead"])
    assert add_result.exit_code == 0, add_result.output
    assert "Added user" in add_result.output

    list_result = runner.invoke(app, ["users", "list", tenant_id])
    assert list_result.exit_code == 0
    assert "Test User" in list_result.output
    assert user_email in list_result.output
    assert "lead" in list_result.output


@requires_admin_cli_env
def test_keys_create_prints_a_real_usable_key(cleanup_tenant):
    email = f"{uuid.uuid4().hex[:8]}@cli-test.dev"
    create_result = runner.invoke(app, ["tenants", "create", "cli-key-test", email])
    tenant_id = create_result.output.split("Created tenant")[1].strip().split()[0]
    cleanup_tenant.append(tenant_id)

    result = runner.invoke(app, ["keys-create", tenant_id, "--name", "cli-test-key"])
    assert result.exit_code == 0, result.output
    assert "ks-" in result.output


@requires_admin_cli_env
def test_user_add_rejects_a_duplicate_email(cleanup_tenant):
    email = f"{uuid.uuid4().hex[:8]}@cli-test.dev"
    create_result = runner.invoke(app, ["tenants", "create", "cli-dup-test", email])
    tenant_id = create_result.output.split("Created tenant")[1].strip().split()[0]
    cleanup_tenant.append(tenant_id)

    user_email = f"{uuid.uuid4().hex[:8]}@cli-test.dev"
    first = runner.invoke(app, ["users", "add", tenant_id, "First", user_email])
    assert first.exit_code == 0

    second = runner.invoke(app, ["users", "add", tenant_id, "Second", user_email])
    assert second.exit_code == 1
    assert "Conflict" in second.output


def test_admin_commands_fail_clearly_without_a_root_admin_token(monkeypatch):
    monkeypatch.setenv("KEYSTONE_INFERENCE_URL", "http://localhost:1")
    monkeypatch.delenv("KEYSTONE_ROOT_ADMIN_TOKEN", raising=False)
    result = runner.invoke(app, ["users", "list", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 1
    assert "KEYSTONE_ROOT_ADMIN_TOKEN" in result.output
