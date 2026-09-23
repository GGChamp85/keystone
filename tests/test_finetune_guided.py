# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The guided fine-tune path (src/finetuning/guided.py, planner.py,
src/inference/catalog.py, src/hardware.py): pure planning tests, then the
real thing — a temporary git repository with real commits (one carrying
PII, one carrying a credential) turned into a planned FineTuneJob in a
real Postgres, approved into a pending job with the trainer start patched
out, and the same flow over the HTTP routes. Integration parts need
DATABASE_URL (tests/conftest.py's requires_integration_env).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.finetuning import guided
from src.finetuning.planner import plan_training
from src.hardware import GPU, parse_nvidia_smi_line
from src.inference.catalog import CATALOG, default_slm, find, largest_that_trains_on

from .conftest import requires_integration_env

# ── pure ──────────────────────────────────────────────────────


def test_catalog_footprints_are_monotonic_and_the_default_fits_a_24gb_gpu():
    sizes = [e.params_b for e in CATALOG]
    assert sizes == sorted(sizes)
    seven_b = default_slm()
    assert seven_b.hf_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert seven_b.vram_qlora_gb <= 22.5 < seven_b.vram_lora_bf16_gb  # QLoRA on an L4; bf16 LoRA needs ~24 GB+
    assert seven_b.vram_serve_gb <= 24.0
    assert largest_that_trains_on(22.5).hf_id == "Qwen/Qwen2.5-Coder-14B-Instruct"
    assert largest_that_trains_on(3.0).hf_id == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    with pytest.raises(KeyError):
        find("nobody/NoSuchModel")


def test_planner_picks_the_method_counts_steps_exactly_and_labels_estimates():
    l4 = [GPU("NVIDIA L4", 22.5)]
    plan = plan_training(default_slm(), l4, train_examples=100, holdout_examples=25, epochs=2, gpu_hourly_cost_usd=0.69)
    assert plan.method == "qlora" and plan.fits
    assert plan.total_steps == 14  # ceil(100 / (2*8)) = 7 steps/epoch x 2
    assert plan.lora_r == 8  # Fireworks-style default for a guided run
    assert plan.estimated_hours and plan.estimated_cost_usd and "estimate" in plan.estimate_basis
    assert "$0.69/GPU-hour" in plan.estimate_basis

    big = plan_training(default_slm(), [GPU("NVIDIA A100 80GB PCIe", 79.2)], train_examples=100, holdout_examples=25)
    assert big.method == "lora" and big.estimated_cost_usd is None and "no GPU price" in big.estimate_basis

    none = plan_training(default_slm(), [], train_examples=2, holdout_examples=0)
    assert not none.fits and none.method == "unfit" and none.estimated_hours is None
    assert any("no GPU" in r for r in none.reasons) and any("fewer than 3" in r for r in none.reasons)

    measured = plan_training(
        default_slm(), l4, train_examples=100, holdout_examples=25, measured_tokens_per_second=1000.0
    )
    assert "measured" in measured.estimate_basis


def test_resolve_base_model_auto_steps_down_when_the_default_does_not_fit():
    assert guided.resolve_base_model("auto", [GPU("L4", 22.5)]).hf_id == default_slm().hf_id
    assert guided.resolve_base_model("auto", [GPU("T4", 15.0)]).hf_id == default_slm().hf_id  # 7B QLoRA fits 16 GB
    assert guided.resolve_base_model("auto", [GPU("RTX 3070", 8.0)]).hf_id == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert guided.resolve_base_model("auto", []).hf_id == default_slm().hf_id  # unknown hardware: keep the default
    assert guided.resolve_base_model("Qwen/Qwen2.5-Coder-0.5B-Instruct", []).params_b == 0.5


def test_credential_shaped_content_is_recognised():
    assert guided.looks_like_credential('API_KEY = "sk-live-000000000000000000000000"')
    assert guided.looks_like_credential("aws_key = AKIAIOSFODNN7EXAMPLE")
    assert guided.looks_like_credential("-----BEGIN RSA PRIVATE KEY-----")
    assert not guided.looks_like_credential("def add(a, b):\n    return a + b\n")
    assert not guided.looks_like_credential("password reset email template")  # a word, not a value


def test_render_example_is_a_fixed_layout_the_trainer_can_read():
    text = guided.render_example({"task": "Fix it", "context": "Repository: r", "solution": "diff --git a b"})
    assert text == "### Task\nFix it\n\n### Context\nRepository: r\n\n### Solution\ndiff --git a b\n"
    assert "### Context" not in guided.render_example({"task": "t", "solution": "s"})


def test_nvidia_smi_line_parsing():
    assert parse_nvidia_smi_line("NVIDIA L4, 23034") == GPU("NVIDIA L4", 22.5)
    assert parse_nvidia_smi_line("garbage") is None


# ── real repository → real plan ───────────────────────────────


_GIT = shutil.which("git") or "git"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 — real git on a scratch repository the test itself created
        [_GIT, "-c", "user.name=t", "-c", "user.email=t@t.dev", *args], cwd=repo, check=True, capture_output=True
    )


def _read(path: str | Path) -> str:
    return Path(path).read_text()


def _make_repo(root: Path) -> Path:
    """A real repository with real commits: ordinary fixes, one commit message with PII, one diff with a
    credential (must be dropped), and a tiny message (must be ignored by the source's minimum length)."""
    repo = root / "src-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "app.py").write_text("def add(a, b):\n    return a - b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Initial commit of the calculator module")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Fix add() which subtracted instead of adding")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Add sub() as requested by jane.doe@example.com in the review")
    (repo / "settings.py").write_text('API_KEY = "sk-live-000000000000000000000000"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Add settings module with the service credential")
    (repo / "README.md").write_text("# calc\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Document the module for new contributors")
    (repo / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Add multiplication alongside add and sub")
    return repo


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path)


@pytest.fixture
def copy_clone(local_repo: Path) -> guided.CloneFn:
    def clone(_url: str, destination: str) -> None:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(local_repo, destination)

    return clone


@pytest.fixture
async def tenant_id():
    from src.db.connection import get_db_context
    from src.db.models import Tenant, TenantTier

    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="guided-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


@pytest.mark.integration
@requires_integration_env
async def test_create_plan_builds_a_sanitised_dataset_and_a_planned_job(tenant_id, copy_clone, tmp_path, monkeypatch):
    from src.config import get_settings
    from src.db.connection import get_db_context
    from src.db.models import FineTuneJob

    monkeypatch.setenv("FINETUNING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FINETUNING_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    try:
        plan = await guided.create_plan(
            tenant_id,
            repositories=["https://git.example.invalid/acme/calc.git"],
            goal="Make the agent follow this repository's conventions when fixing arithmetic bugs",
            base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            holdout_ratio=0.25,
            gpus=[GPU("NVIDIA L4", 22.5)],
            gpu_hourly_cost_usd=0.69,
            clone=copy_clone,
        )
    finally:
        get_settings.cache_clear()

    ds = plan.dataset
    # 5 qualifying commits (the initial commit has no parent); the credential commit is dropped by the scan.
    assert ds["per_repository"] == {"https://git.example.invalid/acme/calc.git": 4}
    assert ds["dropped_by_secret_scan"] == 1
    assert ds["records_with_pii_redacted"] == 1 and ds["warnings"] == []
    assert plan.manifest["train_count"] + plan.manifest["holdout_count"] == 4
    assert plan.plan.method == "lora" and plan.plan.fits and plan.plan.estimated_cost_usd is not None

    rows = [json.loads(line) for line in _read(plan.trainer_config["training_data"]).splitlines()]
    assert rows and all(r["text"].startswith("### Task\n") for r in rows)
    everything = _read(plan.trainer_config["training_data"]) + _read(plan.trainer_config["eval_data"])
    assert "jane.doe@example.com" not in everything and "sk-live-" not in everything
    assert "[EMAIL]" in everything or "REDACTED" in everything.upper()

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, plan.job_id)
    assert job is not None and job.status == "planned" and job.base_model == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert job.config["use_4bit"] is False and job.config["guided"]["plan"]["total_steps"] >= 1


@pytest.mark.integration
@requires_integration_env
async def test_approve_flips_planned_to_pending_and_starts_the_job(tenant_id, copy_clone, tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("FINETUNING_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    try:
        plan = await guided.create_plan(
            tenant_id,
            repositories=["https://git.example.invalid/acme/calc.git"],
            goal="Learn this repository's conventions for small fixes",
            base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            gpus=[GPU("NVIDIA L4", 22.5)],
            clone=copy_clone,
        )
        unfit = await guided.create_plan(
            tenant_id,
            repositories=["https://git.example.invalid/acme/calc.git"],
            goal="Learn this repository's conventions for small fixes",
            base_model="Qwen/Qwen2.5-Coder-32B-Instruct",
            gpus=[GPU("NVIDIA L4", 22.5)],
            clone=copy_clone,
        )
    finally:
        get_settings.cache_clear()

    started: list[uuid.UUID] = []
    with patch("src.finetuning.runner.start_finetune_job", AsyncMock(side_effect=started.append)):
        job = await guided.approve_plan(plan.job_id, tenant_id)
        assert job.status == "pending" and started == [plan.job_id]

        with pytest.raises(guided.PlanNotApprovable, match="not 'planned'"):
            await guided.approve_plan(plan.job_id, tenant_id)  # already pending
        with pytest.raises(guided.PlanNotApprovable, match="does not fit"):
            await guided.approve_plan(unfit.job_id, tenant_id)
        forced = await guided.approve_plan(unfit.job_id, tenant_id, force=True)
        assert forced.status == "pending"
        with pytest.raises(LookupError):
            await guided.approve_plan(uuid.uuid4(), tenant_id)


@pytest.mark.integration
@requires_integration_env
async def test_guided_routes_end_to_end(tenant_id, copy_clone, tmp_path, monkeypatch):
    from src.api.middleware.auth import generate_api_key
    from src.config import get_settings
    from src.db.connection import get_db_context
    from src.db.models import APIKey
    from src.main import create_app

    monkeypatch.setenv("FINETUNING_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(APIKey(tenant_id=tenant_id, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["finetune"]))
    headers = {"Authorization": f"Bearer {full_key}"}
    app = create_app()
    try:
        with (
            patch("src.finetuning.guided.default_clone", copy_clone),
            patch("src.finetuning.runner.start_finetune_job", AsyncMock()),
        ):
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    catalog = await client.get("/v1/finetune/catalog", headers=headers)
                    assert catalog.status_code == 200 and catalog.json()["default"] == default_slm().hf_id

                    created = await client.post(
                        "/v1/finetune/plans",
                        headers=headers,
                        json={
                            "repositories": ["https://git.example.invalid/acme/calc.git"],
                            "goal": "Learn this repository's conventions for small fixes",
                            "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                            "gpus": [{"name": "NVIDIA L4", "vram_gb": 22.5}],
                            "gpu_hourly_cost_usd": 0.69,
                        },
                    )
                    assert created.status_code == 201, created.text
                    body = created.json()
                    assert body["status"] == "planned" and body["plan"]["fits"] is True
                    job_id = body["job_id"]

                    fetched = await client.get(f"/v1/finetune/plans/{job_id}", headers=headers)
                    assert fetched.status_code == 200 and fetched.json()["plan"]["method"] == "lora"

                    approved = await client.post(f"/v1/finetune/plans/{job_id}/approve", headers=headers, json={})
                    assert approved.status_code == 200, approved.text
                    assert approved.json()["status"] == "pending"

                    again = await client.post(f"/v1/finetune/plans/{job_id}/approve", headers=headers, json={})
                    assert again.status_code == 409

                    bad_model = await client.post(
                        "/v1/finetune/plans",
                        headers=headers,
                        json={
                            "repositories": ["https://x.invalid/r.git"],
                            "goal": "something long enough",
                            "base_model": "no/such",
                        },
                    )
                    assert bad_model.status_code == 422
    finally:
        get_settings.cache_clear()
