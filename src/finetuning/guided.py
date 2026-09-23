# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the guided fine-tune: describe → plan & cost → approve → train.

The library pieces existed and nothing wired them together: sources
(git history, accepted trajectories), the stratified manifest, the
trainers, the registry. This module is the wiring, in the order an admin
experiences it:

1. `build_dataset` — for each repository URL (allow-listed, cloned to a
   scratch checkout) collect real SFT examples from its commit history,
   plus the tenant's accepted agent trajectories; every record goes
   through PII redaction and the secret scanner before it is kept.
2. `split_by_repository` + `write_manifest` — a repository-stratified
   train/holdout split with real content hashes.
3. `plan_training` — the method that fits the hardware, exact step count,
   labelled time/cost estimates.
4. `create_plan` — persists a `FineTuneJob` with `status="planned"` whose
   `config` is the complete trainer config plus the plan, so approval is
   nothing more than flipping it to pending and starting it.

The trainer reads `{"text": ...}` JSONL; each example is rendered with
`render_example` in a fixed instruction layout so held-out eval loss is
comparable across runs and against the base model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import FineTuneJob
from src.finetuning.manifest import split_by_repository, write_manifest
from src.finetuning.planner import TrainingPlan, plan_training
from src.finetuning.sources.git_history import build_sft_examples_from_git_history
from src.finetuning.sources.trajectories import build_sft_examples_from_trajectories
from src.hardware import GPU, detect_gpus
from src.inference.catalog import CatalogEntry, default_slm, find
from src.memory.ingestion import _validate_repo_url
from src.sandbox.security import scan_sandbox_output
from src.security.pii_redaction import redact_pii

logger = structlog.get_logger(__name__)

CloneFn = Callable[[str, str], None]  # (repository_url, destination_dir) -> None

# Credential-shaped CONTENT (the sandbox scanner looks at env-var names and exfiltration commands, not at
# whether a diff pastes a key). A training record that carries one of these is dropped, never redacted:
# a redacted secret is still a leak of its shape and location.
_CREDENTIAL_PATTERNS = (
    re.compile(r"\b(?:sk|rk)[-_](?:live|test|proj)?[-_]?[A-Za-z0-9]{16,}"),  # Stripe/OpenAI-style keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub tokens
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),  # GitLab PATs
    re.compile(r"\bxox[bprs]-[A-Za-z0-9-]{10,}\b"),  # Slack tokens
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd)\b\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"
    ),
)


def looks_like_credential(text: str) -> bool:
    return any(p.search(text) for p in _CREDENTIAL_PATTERNS)


def render_example(example: dict[str, Any]) -> str:
    """One SFT record as the trainer's `text`: a fixed layout, so eval loss means the same thing every run."""
    context = (example.get("context") or "").strip()
    parts = ["### Task", (example.get("task") or "").strip(), ""]
    if context:
        parts += ["### Context", context, ""]
    parts += ["### Solution", (example.get("solution") or "").strip(), ""]
    return "\n".join(parts)


def default_clone(repository_url: str, destination: str) -> None:
    """A real `git clone` of an allow-listed repository into `destination`, using the configured bot token."""
    settings = get_settings()
    _validate_repo_url(repository_url, settings.git_allowed_hosts)
    cmd = ["git"]
    token = settings.git_host_token.get_secret_value() if settings.git_host_token else None
    if token:
        cmd += ["-c", f"http.extraHeader=Authorization: token {token}"]
    cmd += ["clone", "--quiet", repository_url, destination]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)  # noqa: S603


@dataclass
class DatasetReport:
    examples: list[dict[str, Any]] = field(default_factory=list)
    per_repository: dict[str, int] = field(default_factory=dict)
    trajectories: int = 0
    dropped_secret_scan: int = 0
    pii_redacted_records: int = 0
    warnings: list[str] = field(default_factory=list)


def _sanitise(example: dict[str, Any], report: DatasetReport) -> dict[str, Any] | None:
    """PII out, secrets out: a record that carries a credential is dropped, one with PII is redacted."""
    body = f"{example.get('task', '')}\n{example.get('solution', '')}"
    scan = scan_sandbox_output(stdout="", stderr="", code=body)
    if not scan.is_safe or looks_like_credential(body):
        report.dropped_secret_scan += 1
        return None
    cleaned = dict(example)
    redacted_any = False
    for key in ("task", "context", "solution"):
        value = cleaned.get(key)
        if isinstance(value, str) and value:
            result = redact_pii(value)
            if result.matches:
                redacted_any = True
                cleaned[key] = result.redacted_text
    if redacted_any:
        report.pii_redacted_records += 1
    return cleaned


async def build_dataset(
    tenant_id: UUID | None,
    repositories: list[str],
    *,
    max_commits: int = 500,
    clone: CloneFn = default_clone,
) -> DatasetReport:
    report = DatasetReport()
    for url in repositories:
        workdir = tempfile.mkdtemp(prefix="keystone-guided-")
        try:
            clone(url, workdir)
            raw = build_sft_examples_from_git_history(workdir, max_commits=max_commits, repository_url=url)
        except Exception as exc:
            report.warnings.append(f"{url}: {exc}")
            logger.warning("guided.repository_failed", repository=url, error=str(exc))
            continue
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        kept = [c for c in (_sanitise(e, report) for e in raw) if c is not None]
        report.per_repository[url] = len(kept)
        report.examples.extend(kept)

    if tenant_id is not None:
        try:
            trajectories = await build_sft_examples_from_trajectories(tenant_id=tenant_id)
        except Exception as exc:
            trajectories = []
            report.warnings.append(f"trajectories: {exc}")
        kept = [c for c in (_sanitise(e, report) for e in trajectories) if c is not None]
        report.trajectories = len(kept)
        report.examples.extend(kept)
    return report


def _write_text_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps({"text": render_example(r)}) + "\n" for r in rows))


@dataclass(frozen=True)
class GuidedPlan:
    job_id: UUID
    base_model: str
    dataset_dir: str
    manifest: dict[str, Any]
    dataset: dict[str, Any]
    plan: TrainingPlan
    trainer_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "status": "planned",
            "base_model": self.base_model,
            "dataset_dir": self.dataset_dir,
            "manifest": self.manifest,
            "dataset": self.dataset,
            "plan": self.plan.to_dict(),
            "trainer_config": self.trainer_config,
        }


def resolve_base_model(base_model: str, gpus: list[GPU]) -> CatalogEntry:
    """ "auto" → the catalog default (7B), or the largest SLM that trains on the smallest detected GPU when
    the default would not fit; a named model must be in the catalog."""
    if base_model != "auto":
        return find(base_model)
    entry = default_slm()
    if gpus:
        vram = min(g.vram_gb for g in gpus)
        if entry.vram_qlora_gb > vram:
            from src.inference.catalog import largest_that_trains_on

            entry = largest_that_trains_on(vram) or entry
    return entry


async def create_plan(
    tenant_id: UUID,
    *,
    repositories: list[str],
    goal: str,
    base_model: str = "auto",
    holdout_ratio: float = 0.2,
    epochs: int = 1,
    lora_r: int | None = None,
    gpus: list[GPU] | None = None,
    gpu_hourly_cost_usd: float | None = None,
    clone: CloneFn = default_clone,
) -> GuidedPlan:
    """The whole "describe → plan" step: dataset, split, manifest, plan, and a `planned` FineTuneJob."""
    hardware = gpus if gpus is not None else detect_gpus()
    entry = resolve_base_model(base_model, hardware)
    report = await build_dataset(tenant_id, repositories, clone=clone)

    train, holdout = split_by_repository(report.examples, holdout_ratio=holdout_ratio)
    job_id = uuid.uuid4()
    dataset_dir = Path(get_settings().finetuning_data_dir) / "guided" / str(job_id)
    manifest = write_manifest(train, holdout, str(dataset_dir))
    train_text = dataset_dir / "train.text.jsonl"
    holdout_text = dataset_dir / "holdout.text.jsonl"
    _write_text_jsonl(train, train_text)
    _write_text_jsonl(holdout, holdout_text)

    plan = plan_training(
        entry,
        hardware,
        train_examples=len(train),
        holdout_examples=len(holdout),
        epochs=epochs,
        lora_r=lora_r,
        gpu_hourly_cost_usd=gpu_hourly_cost_usd,
    )
    trainer_config: dict[str, Any] = {
        "training_data": str(train_text),
        "eval_data": str(holdout_text) if holdout else None,
        "num_epochs": epochs,
        "lora_r": plan.lora_r,
        "lora_alpha": plan.lora_r * 2,
        "per_device_batch_size": plan.per_device_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "use_4bit": plan.method == "qlora",
        "output_dir": os.path.join(get_settings().finetuning_output_dir, "guided", str(job_id)),
    }
    dataset_summary = {
        "goal": goal,
        "repositories": repositories,
        "per_repository": report.per_repository,
        "trajectories": report.trajectories,
        "total_examples": len(report.examples),
        "dropped_by_secret_scan": report.dropped_secret_scan,
        "records_with_pii_redacted": report.pii_redacted_records,
        "warnings": report.warnings,
    }
    async with get_db_context() as db:
        db.add(
            FineTuneJob(
                id=job_id,
                tenant_id=tenant_id,
                base_model=entry.hf_id,
                job_type="lora",
                status="planned",
                config={
                    **trainer_config,
                    "guided": {"dataset": dataset_summary, "manifest": manifest, "plan": plan.to_dict()},
                },
            )
        )
    logger.info(
        "guided.plan_created",
        job_id=str(job_id),
        base_model=entry.hf_id,
        method=plan.method,
        fits=plan.fits,
        train=len(train),
        holdout=len(holdout),
    )
    return GuidedPlan(
        job_id=job_id,
        base_model=entry.hf_id,
        dataset_dir=str(dataset_dir),
        manifest=manifest,
        dataset=dataset_summary,
        plan=plan,
        trainer_config=trainer_config,
    )


class PlanNotApprovable(RuntimeError):
    """The plan cannot be started as it stands (does not fit, or has no training data)."""


async def approve_plan(job_id: UUID, tenant_id: UUID, *, force: bool = False) -> FineTuneJob:
    """Flip a `planned` job to `pending` and start it — refusing a plan that does not fit or has no data
    unless `force` (an admin overriding the fit check on hardware the planner could not see)."""
    from src.finetuning.runner import start_finetune_job

    async with get_db_context() as db:
        job = await db.get(FineTuneJob, job_id)
        if job is None or job.tenant_id != tenant_id:
            raise LookupError("plan not found")
        if job.status != "planned":
            raise PlanNotApprovable(f"job is {job.status!r}, not 'planned'")
        guided = (job.config or {}).get("guided") or {}
        plan = guided.get("plan") or {}
        if plan.get("train_examples", 0) < 3:
            raise PlanNotApprovable("fewer than 3 training examples — add repositories with history or accepted tasks")
        if not plan.get("fits") and not force:
            raise PlanNotApprovable(
                f"the plan does not fit the detected hardware ({plan.get('reasons')}); pass force=true to start anyway"
            )
        job.status = "pending"
        await db.flush()
        snapshot = job
    await start_finetune_job(job_id)
    return snapshot
