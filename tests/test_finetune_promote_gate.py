# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The promotion gate over the real route: a completed job whose adapter did
not beat the base model is refused (409); an admin may force it, which is
audited; a passing verdict promotes, writes the lora_modules manifest and
attempts the live load on the serving role (reported, not required). Also
the LoRA registry's pure rendering. Requires DATABASE_URL.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from sqlalchemy import select

from src.api.middleware.auth import generate_api_key
from src.db.connection import get_db_context
from src.db.models import APIKey, AuditLog, FineTuneJob, ModelAdapter, Tenant, TenantTier, User, UserRole
from src.inference.lora_registry import LoraModule, render_lora_modules_args, render_manifest, role_for_base_model

from .conftest import requires_integration_env


def test_lora_modules_argv_and_manifest_rendering():
    mods = [
        LoraModule("t1-lora", "/adapters/t1", "base/A", "t1"),
        LoraModule("t2-lora", "/adapters/t2", "base/A", "t2"),
    ]
    assert render_lora_modules_args(mods) == ["--lora-modules", "t1-lora=/adapters/t1", "t2-lora=/adapters/t2"]
    assert render_lora_modules_args([]) == []
    manifest = render_manifest(mods)
    assert manifest["argv"]["base/A"][0] == "--lora-modules" and len(manifest["modules"]["base/A"]) == 2


def test_role_for_base_model_reads_the_configured_model_ids(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("CODING_MODEL_ID", "org/CoderX")
    get_settings.cache_clear()
    try:
        assert role_for_base_model("org/CoderX") == "coding"
        assert role_for_base_model("org/Nothing") is None
    finally:
        get_settings.cache_clear()


@pytest.fixture
async def tenant_key_admin():
    """A tenant with two keys: one linked to an ADMIN user, one to nobody."""
    tid = uuid.uuid4()
    admin_key, a_prefix, a_hash = generate_api_key()
    plain_key, p_prefix, p_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="promote-gate", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        admin = User(tenant_id=tid, name="Admin", email=f"admin-{tid}@test.dev", role=UserRole.ADMIN)
        db.add(admin)
        await db.flush()
        db.add(
            APIKey(
                tenant_id=tid, name="admin", key_prefix=a_prefix, key_hash=a_hash, scopes=["finetune"], user_id=admin.id
            )
        )
        db.add(APIKey(tenant_id=tid, name="plain", key_prefix=p_prefix, key_hash=p_hash, scopes=["finetune"]))
    yield tid, admin_key, plain_key
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _completed_job(tenant_id: uuid.UUID, metrics: dict, tmp_path: Path) -> uuid.UUID:
    adapter_dir = tmp_path / f"adapter-{uuid.uuid4().hex[:6]}"
    adapter_dir.mkdir()
    async with get_db_context() as db:
        job = FineTuneJob(
            tenant_id=tenant_id,
            base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            job_type="lora",
            status="completed",
            config={"lora_r": 8},
            metrics=metrics,
            output_model_path=str(adapter_dir),
        )
        db.add(job)
        await db.flush()
        return job.id


@pytest.mark.integration
@requires_integration_env
async def test_promote_is_gated_by_the_verdict_forced_only_by_an_admin_and_then_served(
    tenant_key_admin, tmp_path, monkeypatch
):
    from src.config import get_settings
    from src.main import create_app

    tid, admin_key, plain_key = tenant_key_admin
    monkeypatch.setenv("FINETUNING_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    failing = await _completed_job(tid, {"base_eval_loss": 1.0, "eval_loss": 1.2}, tmp_path)
    passing = await _completed_job(tid, {"base_eval_loss": 1.0, "eval_loss": 0.8}, tmp_path)
    app = create_app()
    live_loads: list[tuple[str, str, str]] = []

    async def fake_load(base_url, name, path, *, api_key=None):
        live_loads.append((base_url, name, path))
        return {"loaded": True, "detail": "fake vllm accepted"}

    try:
        with patch("src.inference.lora_registry.load_adapter_at_runtime", AsyncMock(side_effect=fake_load)):
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    plain = {"Authorization": f"Bearer {plain_key}"}
                    admin = {"Authorization": f"Bearer {admin_key}"}

                    refused = await client.post(f"/v1/finetune/jobs/{failing}/promote", headers=plain)
                    assert refused.status_code == 409 and "verdict fail" in refused.json()["detail"]

                    not_admin = await client.post(f"/v1/finetune/jobs/{failing}/promote?force=true", headers=plain)
                    assert not_admin.status_code == 403

                    forced = await client.post(f"/v1/finetune/jobs/{failing}/promote?force=true", headers=admin)
                    assert forced.status_code == 200, forced.text

                    ok = await client.post(f"/v1/finetune/jobs/{passing}/promote", headers=plain)
                    assert ok.status_code == 200, ok.text
                    assert ok.json()["metrics"]["serving"]["manifest"].endswith("lora_modules.json")
    finally:
        get_settings.cache_clear()

    async with get_db_context() as db:
        adapters = (await db.execute(select(ModelAdapter).where(ModelAdapter.tenant_id == tid))).scalars().all()
        audit = (
            (
                await db.execute(
                    select(AuditLog).where(
                        AuditLog.action == "finetune.promote_forced", AuditLog.target_id == str(failing)
                    )
                )
            )
            .scalars()
            .all()
        )
    # The passing job's adapter is the default now (the forced one was demoted when it was promoted later).
    defaults = [a for a in adapters if a.is_default]
    assert len(adapters) == 2 and len(defaults) == 1 and defaults[0].job_id == passing
    assert len(audit) == 1 and audit[0].metadata_["verdict"]["status"] == "fail"
    assert audit[0].actor.startswith("admin-")
    # Both promotions were made servable: manifest written, live load attempted for each.
    manifest = json.loads((tmp_path / "out" / "lora_modules.json").read_text())
    assert {m["name"] for m in manifest["modules"]["Qwen/Qwen2.5-Coder-0.5B-Instruct"]} == {a.name for a in adapters}
    assert len(live_loads) == 2 or live_loads == []  # live load only when a role serves this base model
