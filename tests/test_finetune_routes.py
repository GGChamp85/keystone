# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/api/routes/finetune.py — real Postgres,
real HTTP round trips over a real FastAPI app (ASGITransport, the same
pattern as tests/test_mcp_server.py/test_users_and_roles.py). Job
submission really calls src/finetuning/runner.py's start_finetune_job
(the real asyncio-fallback path, since no Temporal server is reachable
here) — a submitted job with an unsupported job_type fails fast and real
for tests that don't need to wait on it; tests that need a specific
terminal state build the FineTuneJob row directly instead of going
through the real (torch-requiring, GPU-less-here) training path.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import AdapterStatus, APIKey, FineTuneJob, ModelAdapter, Tenant, TenantTier
from src.finetuning.events import publish_finetune_event
from src.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="finetune-routes-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
async def api_key(tenant_id):
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(
            APIKey(
                tenant_id=tenant_id,
                name="finetune-test-key",
                key_prefix=prefix,
                key_hash=key_hash,
                scopes=["finetune"],
            )
        )
        await db.flush()
    return full_key


@pytest.fixture(autouse=True)
def _finetuning_output_dir(tmp_path, monkeypatch):
    """Promotion writes the real lora_modules manifest into FINETUNING_OUTPUT_DIR; give it a writable one."""
    from src.config import get_settings

    monkeypatch.setenv("FINETUNING_OUTPUT_DIR", str(tmp_path / "finetuning-out"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
            yield client


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# Metrics a real trainer run writes when the adapter beat the base model on the held-out split — what the
# promotion verdict gate (src/finetuning/verdict.py) requires before a non-forced promote.
PASSING = {"base_eval_loss": 1.0, "eval_loss": 0.8}


async def _make_job(tenant_id, **overrides) -> uuid.UUID:
    jid = uuid.uuid4()
    fields = {
        "id": jid,
        "tenant_id": tenant_id,
        "base_model": "Qwen/Qwen2.5-Coder-0.5B",
        "job_type": "lora",
        "status": "pending",
        "config": {"training_data": "/tmp/does-not-matter.jsonl"},  # noqa: S108
    }
    fields.update(overrides)
    async with get_db_context() as db:
        db.add(FineTuneJob(**fields))
        await db.flush()
    return jid


async def test_submitting_a_job_with_an_unknown_job_type_is_rejected(tenant_id, api_key):
    async with running_client() as client:
        resp = await client.post(
            "/v1/finetune/jobs",
            headers=_headers(api_key),
            json={"job_type": "not-real", "training_data_path": "/data/train.jsonl", "config": {}},
        )
        assert resp.status_code == 422


async def test_submitting_a_job_with_an_unknown_config_field_is_rejected(tenant_id, api_key):
    async with running_client() as client:
        resp = await client.post(
            "/v1/finetune/jobs",
            headers=_headers(api_key),
            json={
                "job_type": "lora",
                "training_data_path": "/data/train.jsonl",
                "config": {"totally_made_up_field": 1},
            },
        )
        assert resp.status_code == 422
        assert "totally_made_up_field" in resp.text


async def test_submitting_a_valid_job_persists_a_real_row_and_starts_it(tenant_id, api_key):
    async with running_client() as client:
        resp = await client.post(
            "/v1/finetune/jobs",
            headers=_headers(api_key),
            json={
                "job_type": "lora",
                "training_data_path": "/tmp/does-not-matter.jsonl",  # noqa: S108
                "config": {"num_epochs": 1},
            },
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == "pending"
        assert body["config"]["num_epochs"] == 1
        assert body["config"]["training_data"] == "/tmp/does-not-matter.jsonl"  # noqa: S108

        get_resp = await client.get(f"/v1/finetune/jobs/{body['id']}", headers=_headers(api_key))
        assert get_resp.status_code == 200


async def test_list_jobs_only_returns_this_tenants_jobs(tenant_id, api_key):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tenant_id, name="other", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    try:
        await _make_job(tenant_id)
        await _make_job(other_tenant_id)

        async with running_client() as client:
            resp = await client.get("/v1/finetune/jobs", headers=_headers(api_key))
            assert resp.status_code == 200
            assert len(resp.json()) == 1
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)


async def test_list_jobs_filters_by_status(tenant_id, api_key):
    await _make_job(tenant_id, status="pending")
    await _make_job(tenant_id, status="completed")

    async with running_client() as client:
        resp = await client.get("/v1/finetune/jobs", headers=_headers(api_key), params={"status": "completed"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["status"] == "completed"


async def test_get_job_404s_for_another_tenants_job(tenant_id, api_key):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tenant_id, name="other2", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    try:
        foreign_job_id = await _make_job(other_tenant_id)
        async with running_client() as client:
            resp = await client.get(f"/v1/finetune/jobs/{foreign_job_id}", headers=_headers(api_key))
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)


async def test_dataset_preview_reads_real_file_content(tenant_id, api_key, tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"task": "a"}\n{"task": "b"}\n{"task": "c"}\n')
    job_id = await _make_job(tenant_id, config={"training_data": str(data_path)})

    async with running_client() as client:
        resp = await client.get(
            f"/v1/finetune/jobs/{job_id}/dataset-preview",
            headers=_headers(api_key),
            params={"lines": 2},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 3
        assert len(body["preview"]) == 2
        assert body["preview"][0]["task"] == "a"


async def test_dataset_preview_404s_when_the_file_does_not_exist(tenant_id, api_key):
    job_id = await _make_job(tenant_id, config={"training_data": "/tmp/definitely-not-a-real-file.jsonl"})  # noqa: S108
    async with running_client() as client:
        resp = await client.get(f"/v1/finetune/jobs/{job_id}/dataset-preview", headers=_headers(api_key))
        assert resp.status_code == 404


async def test_promote_requires_a_completed_job_with_an_output_path(tenant_id, api_key):
    job_id = await _make_job(tenant_id, status="pending")
    async with running_client() as client:
        resp = await client.post(f"/v1/finetune/jobs/{job_id}/promote", headers=_headers(api_key))
        assert resp.status_code == 409


async def test_promote_refuses_a_job_without_a_held_out_comparison(tenant_id, api_key):
    """No base_eval_loss -> verdict unknown -> 409 with the reason (an admin may force; see
    tests/test_finetune_promote_gate.py)."""
    job_id = await _make_job(
        tenant_id, status="completed", output_model_path="/data/adapters/no-base", metrics={"eval_loss": 0.3}
    )
    async with running_client() as client:
        resp = await client.post(f"/v1/finetune/jobs/{job_id}/promote", headers=_headers(api_key))
        assert resp.status_code == 409
        assert "verdict unknown" in resp.json()["detail"] and "force=true" in resp.json()["detail"]


async def test_promote_creates_a_real_default_adapter(tenant_id, api_key):
    job_id = await _make_job(
        tenant_id,
        status="completed",
        output_model_path="/data/adapters/real-adapter",
        metrics=PASSING,
    )
    async with running_client() as client:
        resp = await client.post(f"/v1/finetune/jobs/{job_id}/promote", headers=_headers(api_key))
        assert resp.status_code == 200

    async with get_db_context() as db:
        from sqlalchemy import select

        adapter = (await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_id))).scalar_one()
        assert adapter.status == AdapterStatus.PROMOTED
        assert adapter.is_default is True
        assert adapter.path == "/data/adapters/real-adapter"
        assert adapter.base_model_id == "Qwen/Qwen2.5-Coder-0.5B"
    serving = resp.json()["metrics"]["serving"]
    assert serving["manifest_error"] is None and serving["manifest"].endswith("lora_modules.json")
    manifest = json.loads(await asyncio.to_thread(Path(serving["manifest"]).read_text))
    assert any(m["path"] == "/data/adapters/real-adapter" for m in manifest["modules"]["Qwen/Qwen2.5-Coder-0.5B"])


async def test_promoting_a_second_job_demotes_the_previous_default(tenant_id, api_key):
    job_a = await _make_job(tenant_id, status="completed", output_model_path="/data/adapters/a", metrics=PASSING)
    job_b = await _make_job(tenant_id, status="completed", output_model_path="/data/adapters/b", metrics=PASSING)

    async with running_client() as client:
        assert (await client.post(f"/v1/finetune/jobs/{job_a}/promote", headers=_headers(api_key))).status_code == 200
        assert (await client.post(f"/v1/finetune/jobs/{job_b}/promote", headers=_headers(api_key))).status_code == 200

    async with get_db_context() as db:
        from sqlalchemy import select

        adapter_a = (await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_a))).scalar_one()
        adapter_b = (await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_b))).scalar_one()
        assert adapter_a.is_default is False
        assert adapter_b.is_default is True


async def test_rollback_retires_the_adapter_and_clears_default(tenant_id, api_key):
    job_id = await _make_job(tenant_id, status="completed", output_model_path="/data/adapters/x", metrics=PASSING)
    async with running_client() as client:
        await client.post(f"/v1/finetune/jobs/{job_id}/promote", headers=_headers(api_key))
        resp = await client.post(f"/v1/finetune/jobs/{job_id}/rollback", headers=_headers(api_key))
        assert resp.status_code == 200

    async with get_db_context() as db:
        from sqlalchemy import select

        adapter = (await db.execute(select(ModelAdapter).where(ModelAdapter.job_id == job_id))).scalar_one()
        assert adapter.status == AdapterStatus.RETIRED
        assert adapter.is_default is False


async def test_rollback_404s_for_a_job_never_promoted(tenant_id, api_key):
    job_id = await _make_job(tenant_id, status="completed", output_model_path="/data/adapters/x")
    async with running_client() as client:
        resp = await client.post(f"/v1/finetune/jobs/{job_id}/rollback", headers=_headers(api_key))
        assert resp.status_code == 404


async def test_stream_replays_real_published_events(tenant_id, api_key):
    job_id = await _make_job(tenant_id)
    await publish_finetune_event(job_id, "running", "Training started")
    await publish_finetune_event(job_id, "completed", "Training completed", metrics={"train_loss": 0.1})

    stream_url = f"/v1/finetune/jobs/{job_id}/stream"
    async with running_client() as client, client.stream("GET", stream_url, headers=_headers(api_key)) as resp:
        assert resp.status_code == 200
        lines = []
        async for line in resp.aiter_lines():
            lines.append(line)
            if len(lines) > 20:
                break

    data_lines = [json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ")]
    statuses = [d["status"] for d in data_lines]
    assert statuses == ["running", "completed"]
