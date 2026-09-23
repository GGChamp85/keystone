# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Tests for src/finetuning/export.py and the `POST /v1/finetune/jobs/{id}/export`
route — everything that needs no torch. The real merge and GGUF conversion
run in tests/test_trainer_smoke.py (the `.venv-train` environment, gated
on KEYSTONE_TRAINING_SMOKE=1) on the smoke's real 0.5B adapter; here the
refusals are real (a wrong quant, a missing converter, AWQ on a host with
no CUDA), the converter pin is held to the training image's, and the route
is exercised over real HTTP against a real Postgres, including the asyncio
fallback recording an honest failure when the ML stack is absent.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from src.finetuning import export
from src.finetuning.export import (
    EXPORT_FORMATS,
    GGUF_QUANT_TYPES,
    LLAMA_CPP_COMMIT,
    ExportUnavailable,
    artifact_size,
    export_awq,
    export_gguf,
    run_export,
)
from src.finetuning.runner import export_root_for

_ROOT = Path(__file__).resolve().parent.parent


# ── unit ────────────────────────────────────────────────────────


def test_the_converter_pin_is_the_one_the_training_image_vendors():
    dockerfile = (_ROOT / "docker" / "training.Dockerfile").read_text()
    match = re.search(r"^ARG LLAMA_CPP_COMMIT=([0-9a-f]{40})$", dockerfile, re.MULTILINE)
    assert match, "docker/training.Dockerfile must pin LLAMA_CPP_COMMIT to a full commit sha"
    assert match.group(1) == LLAMA_CPP_COMMIT
    assert "convert_hf_to_gguf.py" in dockerfile and "conversion/*" in dockerfile and "gguf-py/*" in dockerfile


def test_gguf_extra_is_declared():
    pyproject = (_ROOT / "pyproject.toml").read_text()
    assert re.search(r'"gguf>=[0-9.]+"', pyproject)
    assert "finetuning-ray" in pyproject and "ray[train]" in pyproject


def test_export_gguf_rejects_an_unknown_quant_before_touching_anything(tmp_path):
    with pytest.raises(ValueError, match="q4_k_m"):
        export_gguf(tmp_path, tmp_path / "x.gguf", quant="q4_k_m")
    assert GGUF_QUANT_TYPES == ("f32", "f16", "bf16", "q8_0")


def test_export_gguf_without_the_converter_says_where_it_should_be(tmp_path, monkeypatch):
    monkeypatch.setenv(export.CONVERTER_DIR_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(ExportUnavailable, match=LLAMA_CPP_COMMIT):
        export_gguf(tmp_path, tmp_path / "x.gguf")


def test_export_awq_refuses_on_a_host_without_cuda_or_the_ml_stack(tmp_path):
    with pytest.raises(ExportUnavailable, match="AWQ export needs"):
        export_awq(tmp_path, tmp_path / "awq")


def test_run_export_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="onnx"):
        run_export("onnx", base_model="m", adapter_dir=tmp_path, export_root=tmp_path)
    assert EXPORT_FORMATS == ("merged", "gguf", "awq")


def test_artifact_size_sums_a_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert artifact_size(tmp_path) == 15
    assert artifact_size(tmp_path / "a.bin") == 10


def test_export_root_sits_next_to_the_adapter():
    assert export_root_for("/data/finetuning/output/guided/abc/adapter") == "/data/finetuning/output/guided/abc/export"
    assert export_root_for("/data/out/adapter/") == "/data/out/export"


# ── the route, over real HTTP against a real Postgres ───────────


@pytest.fixture
async def tenant_id():
    from src.db.connection import get_db_context
    from src.db.models import Tenant, TenantTier

    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="finetune-export-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.fixture
async def api_key(tenant_id):
    from src.api.middleware.auth import generate_api_key
    from src.db.connection import get_db_context
    from src.db.models import APIKey

    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(
            APIKey(
                tenant_id=tenant_id, name="export-test-key", key_prefix=prefix, key_hash=key_hash, scopes=["finetune"]
            )
        )
        await db.flush()
    return full_key


@asynccontextmanager
async def running_client() -> AsyncIterator[httpx.AsyncClient]:
    from src.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
            yield client


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def _make_job(tenant_id, **overrides) -> uuid.UUID:
    from src.db.connection import get_db_context
    from src.db.models import FineTuneJob

    jid = uuid.uuid4()
    fields = {
        "id": jid,
        "tenant_id": tenant_id,
        "base_model": "Qwen/Qwen2.5-Coder-0.5B",
        "job_type": "lora",
        "status": "completed",
        "config": {"training_data": "/tmp/does-not-matter.jsonl"},  # noqa: S108
        "output_model_path": "/data/finetuning/output/x/adapter",
        "metrics": {"train_loss": 0.5},
    }
    fields.update(overrides)
    async with get_db_context() as db:
        db.add(FineTuneJob(**fields))
        await db.flush()
    return jid


@pytest.mark.integration
async def test_export_refuses_a_job_that_is_not_completed(tenant_id, api_key):
    job_id = await _make_job(tenant_id, status="running", output_model_path=None)
    async with running_client() as client:
        resp = await client.post(
            f"/v1/finetune/jobs/{job_id}/export", headers=_headers(api_key), json={"format": "gguf"}
        )
        assert resp.status_code == 409
        assert "completed" in resp.json()["detail"]


@pytest.mark.integration
async def test_export_validates_the_format_and_the_quant(tenant_id, api_key):
    job_id = await _make_job(tenant_id)
    async with running_client() as client:
        assert (
            await client.post(f"/v1/finetune/jobs/{job_id}/export", headers=_headers(api_key), json={"format": "onnx"})
        ).status_code == 422
        resp = await client.post(
            f"/v1/finetune/jobs/{job_id}/export", headers=_headers(api_key), json={"format": "gguf", "quant": "q4_k_m"}
        )
        assert resp.status_code == 422 and "q8_0" in resp.json()["detail"]
        resp = await client.post(
            f"/v1/finetune/jobs/{job_id}/export", headers=_headers(api_key), json={"format": "merged", "quant": "q8_0"}
        )
        assert resp.status_code == 422 and "gguf format only" in resp.json()["detail"]


@pytest.mark.integration
async def test_export_404s_for_another_tenants_job(tenant_id, api_key):
    from src.db.connection import get_db_context
    from src.db.models import Tenant, TenantTier

    other = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other, name="other-export", email=f"{other}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    try:
        foreign = await _make_job(other)
        async with running_client() as client:
            resp = await client.post(f"/v1/finetune/jobs/{foreign}/export", headers=_headers(api_key), json={})
            assert resp.status_code == 404
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other)
            if row is not None:
                await db.delete(row)


@pytest.mark.integration
async def test_export_refuses_a_second_run_while_one_is_running(tenant_id, api_key):
    job_id = await _make_job(tenant_id, metrics={"exports": {"gguf": {"status": "running"}}})
    async with running_client() as client:
        resp = await client.post(
            f"/v1/finetune/jobs/{job_id}/export", headers=_headers(api_key), json={"format": "gguf"}
        )
        assert resp.status_code == 409 and "already running" in resp.json()["detail"]


@pytest.mark.integration
async def test_export_starts_in_the_background_and_records_an_honest_outcome(tenant_id, api_key, tmp_path):
    """No Temporal here, so the asyncio fallback runs the export; this environment has no torch, so the
    real outcome is a failure recorded on the job with the reason — never a fake artifact."""
    adapter = tmp_path / "out" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    job_id = await _make_job(tenant_id, output_model_path=str(adapter))
    async with running_client() as client:
        resp = await client.post(
            f"/v1/finetune/jobs/{job_id}/export", headers=_headers(api_key), json={"format": "gguf", "quant": "f16"}
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "started" and body["format"] == "gguf" and body["quant"] == "f16"
        assert body["execution"] == "asyncio_fallback"
        assert body["export_root"] == str(tmp_path / "out" / "export")

        for _ in range(100):
            job = (await client.get(f"/v1/finetune/jobs/{job_id}", headers=_headers(api_key))).json()
            entry = (job["metrics"].get("exports") or {}).get("gguf") or {}
            if entry.get("status") in ("completed", "failed"):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(f"export never reached a terminal state: {job['metrics']}")
        assert entry["quant"] == "f16" and entry["started_at"]
        assert job["metrics"]["train_loss"] == 0.5, "the training metrics are kept, the export is added beside them"
        assert job["status"] == "completed", "an export never changes the job's own status"
        if entry["status"] == "failed":
            assert "torch" in entry["error"] or "peft" in entry["error"] or "transformers" in entry["error"], entry
        else:
            assert await asyncio.to_thread(Path(entry["path"]).is_file) and entry["size_bytes"] > 0
