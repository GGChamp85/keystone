# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The in-process training backend — the trainer runs in this process, on
this host's GPUs (or CPU for a smoke). Every other backend ends up calling
`run_training` too, just inside a Ray Train worker or a RunPod pod, so the
job snapshot → trainer config mapping lives here once.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
from collections.abc import Callable
from typing import Any

from src.finetuning.events import publish_finetune_event

# A snapshot of the fields a runner needs — not the live ORM row (which shouldn't be held open across a
# long-running, potentially blocking training call).
FineTuneJobSnapshot = dict[str, Any]

ProgressFn = Callable[[dict[str, Any]], None]

_JOB_TYPE_CONFIGS = {
    "lora": ("src.finetuning.lora_train", "LoRATrainingConfig", "run_lora_training"),
    "sft": ("src.finetuning.sft_train", "SFTTrainingConfig", "run_sft_training"),
    "dpo": ("src.finetuning.dpo_train", "DPOTrainingConfig", "run_dpo_training"),
}


def _build_config(config_cls: type, base_model: str, extra: dict[str, Any]):
    """Builds `config_cls` from the job's stored JSONB `config`, keeping only
    keys that are real fields on that dataclass — a job's config dict is
    user/API-submitted input, not something to trust blindly as exact
    constructor kwargs (an unrecognized key would otherwise raise
    TypeError deep inside a long-running job instead of failing clearly
    at submission time, where src/api/routes/finetune.py validates it)."""
    known_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in extra.items() if k in known_fields}
    return config_cls(base_model=base_model, **filtered)


def resolve_trainer(job_type: str) -> tuple[type, Callable[[Any], dict[str, Any]]]:
    """The (Config dataclass, run function) pair for a job type — imported lazily, since the trainers pull
    in torch/transformers, which the API process does not have."""
    if job_type not in _JOB_TYPE_CONFIGS:
        raise ValueError(f"Unknown fine-tuning job_type: {job_type!r} (expected one of {list(_JOB_TYPE_CONFIGS)})")
    module_name, config_cls_name, run_fn_name = _JOB_TYPE_CONFIGS[job_type]
    module = importlib.import_module(module_name)
    return getattr(module, config_cls_name), getattr(module, run_fn_name)


def run_training(
    job: FineTuneJobSnapshot,
    *,
    progress: ProgressFn | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Synchronous, blocking: builds the trainer config from the snapshot and runs the real trainer.
    `output_dir` overrides the stored one (a Ray worker or a RunPod pod writes where its own storage is)."""
    config_cls, run_fn = resolve_trainer(job["job_type"])
    config = _build_config(config_cls, job["base_model"], job.get("config") or {})
    if output_dir and hasattr(config, "output_dir"):
        config.output_dir = output_dir
    if progress is not None and hasattr(config, "progress"):
        config.progress = progress
    return run_fn(config)


def format_progress(payload: dict[str, Any]) -> str:
    step, total = payload.get("step"), payload.get("total_steps")
    message = f"step {step}/{total}"
    if payload.get("loss") is not None:
        message += f", loss {payload['loss']:.4f}"
    return message


async def run_inprocess(job: FineTuneJobSnapshot) -> dict[str, Any]:
    """Dispatches to the real trainer for `job["job_type"]`. The real
    trainers (run_lora_training/run_sft_training/run_dpo_training) are
    synchronous, blocking calls (they call TRL's/Transformers' own
    trainer.train() directly) — run in a thread executor so this doesn't
    block the event loop the rest of the app (SSE streams, other API
    requests) depends on."""
    resolve_trainer(job["job_type"])  # fail fast on an unknown type, before the executor
    loop = asyncio.get_running_loop()
    job_id = job["id"]

    def report(payload: dict[str, Any]) -> None:
        # Called from the training thread: hand the event to the loop the API runs on.
        asyncio.run_coroutine_threadsafe(
            publish_finetune_event(job_id, "running", format_progress(payload), metrics=payload), loop
        )

    return await loop.run_in_executor(None, lambda: run_training(job, progress=report))
