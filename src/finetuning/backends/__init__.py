# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Where a fine-tuning job's trainer runs. `Settings.finetune_backend` picks one:

- `inprocess` — this process, this host's GPUs (the default; the CPU smoke).
- `ray` — a Ray Train `TorchTrainer` job submitted to the KubeRay cluster
  at `RAY_ADDRESS` (ADR 0002); the app talks to the Ray Jobs REST API and
  needs no `ray` package itself.
- `runpod_pod` — an on-demand RunPod GPU pod running the training image,
  the adapter synced to a RunPod network volume (ADR 0005).

Each backend is a `FineTuneRunner`: an async callable from a job snapshot
to the trainer's metrics dict — the same seam src/finetuning/runner.py's
`run_finetune_job` accepts for tests.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from src.finetuning.backends.inprocess import FineTuneJobSnapshot

FineTuneRunner = Callable[[FineTuneJobSnapshot], Awaitable[dict[str, Any]]]

BACKENDS = ("inprocess", "ray", "runpod_pod")


def select_runner(name: str) -> FineTuneRunner:
    """The runner for `Settings.finetune_backend`; an unknown name is a configuration error, raised here
    (at job start) rather than deep inside a job."""
    if name == "inprocess":
        from src.finetuning.backends.inprocess import run_inprocess

        return run_inprocess
    if name == "ray":
        from src.finetuning.backends.ray_train import run_ray_job

        return run_ray_job
    if name == "runpod_pod":
        from src.finetuning.backends.runpod_pod import run_runpod_pod_job

        return run_runpod_pod_job
    raise ValueError(f"Unknown FINETUNE_BACKEND {name!r} (expected one of {BACKENDS})")
