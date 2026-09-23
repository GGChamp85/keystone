# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The quickstart (docs/getting-started/quickstart.md), executed literally where CI can: step 1's
`keystone init --backend demo-cpu --yes` writes the .env the doc says it writes; step 3 creates a tenant
and a key through the real admin routes and gets a real completion — through the real router, from the
real llama.cpp demo model — with the OpenAI route, the Anthropic route, and the Model Library reporting
the coding role READY. Self-skips without a reachable VLLM_CODING_URL and the integration env.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from src.cli.main import app as cli_app
from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import Tenant

from ..conftest import requires_integration_env

_BACKEND_URL = os.environ.get("VLLM_CODING_URL", "")


def _backend_reachable() -> bool:
    if not _BACKEND_URL:
        return False
    try:
        return httpx.get(f"{_BACKEND_URL.rstrip('/')}/models", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    requires_integration_env,
    pytest.mark.skipif(not _backend_reachable(), reason="Needs a reachable model backend at VLLM_CODING_URL"),
]


def test_step_1_init_writes_the_env_the_quickstart_describes(tmp_path: Path):
    out = tmp_path / ".env"
    result = CliRunner().invoke(cli_app, ["init", "--backend", "demo-cpu", "--yes", "--output", str(out)])
    assert result.exit_code == 0, result.output
    env = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line and not line.startswith("#"))
    assert env["VLLM_CODING_URL"] == "http://demo-model:8000/v1"
    assert env["CODING_MODEL_ID"] == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert len(env["POSTGRES_PASSWORD"]) == 64 and len(env["KEYSTONE_ROOT_ADMIN_TOKEN"]) == 64  # real secrets
    assert "keystone up --with-demo-model" in result.output


async def test_step_3_tenant_key_and_a_real_completion_through_the_real_router(monkeypatch):
    from src.inference import client as client_module
    from src.inference.health import endpoint_health
    from src.main import create_app

    admin_token = secrets.token_hex(32)
    monkeypatch.setenv("KEYSTONE_ROOT_ADMIN_TOKEN", admin_token)
    get_settings.cache_clear()
    endpoint_health.reset()
    await client_module.close_all_clients()
    tenant_id: str | None = None
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
                admin = {"Authorization": f"Bearer {admin_token}"}
                created = await client.post(
                    "/v1/admin/tenants", headers=admin, json={"name": "quickstart", "email": "qs@test.dev"}
                )
                assert created.status_code in (200, 201), created.text
                tenant_id = created.json()["id"]
                key = await client.post(
                    f"/v1/admin/tenants/{tenant_id}/keys",
                    headers=admin,
                    json={"name": "laptop", "scopes": ["inference", "agent", "finetune"]},
                )
                assert key.status_code in (200, 201), key.text
                api_key = key.json()["key"]
                assert api_key.startswith("ks-")
                user = {"Authorization": f"Bearer {api_key}"}

                # the quickstart's curl, verbatim
                completion = await client.post(
                    "/v1/chat/completions",
                    headers=user,
                    json={
                        "model": "coding",
                        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                        "max_tokens": 8,
                    },
                )
                assert completion.status_code == 200, completion.text
                body = completion.json()
                assert body["choices"][0]["message"]["content"].strip()
                assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] > 0

                # "the same key works for the Anthropic Messages API"
                message = await client.post(
                    "/v1/messages",
                    headers=user,
                    json={"model": "coding", "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]},
                )
                assert message.status_code == 200, message.text
                assert message.json()["content"][0]["text"].strip()

                # "open Models to see the coding role READY"
                library = await client.get("/v1/keystone/models", headers=user)
                assert library.status_code == 200, library.text
                coding = next(m for m in library.json()["models"] if m["kind"] == "role" and m["id"] == "coding")
                assert coding["state"] == "READY"
    finally:
        get_settings.cache_clear()
        endpoint_health.reset()
        await client_module.close_all_clients()
        if tenant_id:
            async with get_db_context() as db:
                row = await db.get(Tenant, tenant_id)
                if row is not None:
                    await db.delete(row)
