# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — train on more than one GPU with Ray Train (`FINETUNE_BACKEND=ray`).

ADR 0002: Ray Train on KubeRay, the same Ray the chart already runs for
multi-node vLLM. Two halves in one module:

The control plane (the API/worker process, which has no torch and no ray):
`ray_job_spec` builds a Ray Jobs API submission whose entrypoint is this
module, `RayJobsClient` talks to the Ray dashboard at `RAY_ADDRESS` over
its documented REST API (`POST /api/jobs/`, `GET /api/jobs/{id}`,
`GET /api/jobs/{id}/logs`, `POST /api/jobs/{id}/stop`), and `run_ray_job`
polls the job, relays progress lines from its log as fine-tune events, and
returns the metrics line the driver printed.

The driver (`python -m src.finetuning.backends.ray_train`, run by Ray on
the cluster inside the training image): builds a real
`ray.train.torch.TorchTrainer` — `ScalingConfig` from
`build_scaling_config`, `RunConfig` from `build_run_config`, the same
in-process trainer wrapped in `train_loop_per_worker` — calls `fit()`, and
prints the rank-0 metrics behind a marker the control plane parses.

Verified without a cluster: the builders are pure and unit-tested
(tests/test_ray_train_backend.py) and, where `ray` is importable, the
kwargs construct the real `ScalingConfig`/`RunConfig`/`TorchTrainer`
objects; the polling loop runs against a local server speaking the Ray
Jobs REST shapes (tests/fixtures/fake_ray_jobs_server.py). Not verified:
a real multi-worker run — no Ray cluster or GPU was available.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.finetuning.backends.inprocess import FineTuneJobSnapshot, format_progress
from src.finetuning.events import publish_finetune_event

logger = structlog.get_logger(__name__)

ENTRYPOINT = "python -m src.finetuning.backends.ray_train"
JOB_ENV = "KEYSTONE_FINETUNE_JOB"
NUM_WORKERS_ENV = "KEYSTONE_RAY_NUM_WORKERS"
USE_GPU_ENV = "KEYSTONE_RAY_USE_GPU"
OUTPUT_DIR_ENV = "KEYSTONE_FINETUNE_OUTPUT_DIR"

PROGRESS_MARKER = "KEYSTONE_FINETUNE_PROGRESS "
METRICS_MARKER = "KEYSTONE_FINETUNE_METRICS "

# Ray Jobs API `JobStatus` values.
JOB_TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
JOB_STATUSES = {"PENDING", "RUNNING", *JOB_TERMINAL}

_RAY_IMPORT_HELP = (
    "Ray is not installed in this environment. The `finetuning-ray` extra (pip install -e '.[finetuning-ray]') "
    "adds it; docker/training.Dockerfile's image, which the KubeRay cluster runs, includes it."
)


# ── pure builders ───────────────────────────────────────────────


def build_scaling_config(
    num_workers: int,
    use_gpu: bool,
    *,
    resources_per_worker: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Keyword arguments for `ray.train.ScalingConfig`: one worker per GPU (data parallel), a GPU each."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    kwargs: dict[str, Any] = {"num_workers": num_workers, "use_gpu": use_gpu}
    if resources_per_worker:
        kwargs["resources_per_worker"] = dict(resources_per_worker)
    return kwargs


def build_run_config(job_id: Any, output_dir: str) -> dict[str, Any]:
    """Keyword arguments for `ray.train.RunConfig`: the run named after the job, checkpoints and
    results under FINETUNING_OUTPUT_DIR/ray (the shared PVC the chart mounts on every Ray pod)."""
    return {"name": f"keystone-finetune-{job_id}", "storage_path": os.path.join(output_dir, "ray")}


def workers_for(job: FineTuneJobSnapshot, configured: int) -> int:
    """RAY_TRAIN_NUM_WORKERS when set, else the guided plan's gpu_count, else 1."""
    if configured > 0:
        return configured
    plan = ((job.get("config") or {}).get("guided") or {}).get("plan") or {}
    return max(int(plan.get("gpu_count") or 0), 1)


def ray_job_spec(
    job: FineTuneJobSnapshot,
    *,
    num_workers: int,
    use_gpu: bool,
    output_dir: str,
    working_dir: str | None = None,
) -> dict[str, Any]:
    """The Ray Jobs API submission body: this module as the entrypoint, the job in the runtime env, a
    submission id that makes a resubmit of the same job a 400 (`submission_id` must be unique) rather
    than a second silent run."""
    job_id = str(job["id"])
    env_vars = {
        JOB_ENV: json.dumps(
            {
                "id": job_id,
                "job_type": job["job_type"],
                "base_model": job["base_model"],
                "config": job.get("config") or {},
            },
            default=str,
        ),
        NUM_WORKERS_ENV: str(num_workers),
        USE_GPU_ENV: "1" if use_gpu else "0",
        OUTPUT_DIR_ENV: output_dir,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    runtime_env: dict[str, Any] = {"env_vars": env_vars}
    if working_dir:
        runtime_env["working_dir"] = working_dir
    return {
        "entrypoint": ENTRYPOINT,
        "submission_id": f"keystone-finetune-{job_id}",
        "runtime_env": runtime_env,
        "metadata": {"keystone_job_id": job_id, "keystone_job_type": str(job["job_type"])},
    }


def parse_marker_lines(logs: str, marker: str) -> list[dict[str, Any]]:
    """Every JSON payload printed behind `marker`, in order; malformed lines are skipped, not fatal."""
    found: list[dict[str, Any]] = []
    for line in logs.splitlines():
        idx = line.find(marker)
        if idx < 0:
            continue
        try:
            found.append(json.loads(line[idx + len(marker) :]))
        except ValueError:
            continue
    return found


# ── the Ray Jobs REST client ────────────────────────────────────


class RayJobsClient:
    """Ray's Jobs REST API on the dashboard (what `ray job submit` itself uses) — no `ray` package needed."""

    def __init__(self, dashboard_url: str, *, timeout: float = 30.0):
        if not dashboard_url:
            raise ValueError("RAY_ADDRESS (the Ray dashboard URL) is required")
        self._client = httpx.AsyncClient(base_url=dashboard_url.rstrip("/"), timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def submit(self, spec: dict[str, Any]) -> str:
        resp = await self._client.post("/api/jobs/", json=spec)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ray Jobs API refused the submission ({resp.status_code}): {resp.text}")
        return resp.json()["submission_id"]

    async def status(self, submission_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/api/jobs/{submission_id}")
        resp.raise_for_status()
        return resp.json()

    async def logs(self, submission_id: str) -> str:
        resp = await self._client.get(f"/api/jobs/{submission_id}/logs")
        resp.raise_for_status()
        return resp.json().get("logs", "")

    async def stop(self, submission_id: str) -> None:
        resp = await self._client.post(f"/api/jobs/{submission_id}/stop")
        if resp.status_code != 404:
            resp.raise_for_status()


async def wait_for_job(
    client: RayJobsClient,
    submission_id: str,
    job_id: Any,
    *,
    poll_interval_seconds: float,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Poll until terminal; relay new progress lines as fine-tune events; return the metrics line."""
    started = time.monotonic()
    seen_progress = 0
    while True:
        info = await client.status(submission_id)
        status = info.get("status")
        logs = await client.logs(submission_id)
        progress_lines = parse_marker_lines(logs, PROGRESS_MARKER)
        for payload in progress_lines[seen_progress:]:
            await publish_finetune_event(job_id, "running", format_progress(payload), metrics=payload)
        seen_progress = len(progress_lines)

        if status == "SUCCEEDED":
            metrics_lines = parse_marker_lines(logs, METRICS_MARKER)
            if not metrics_lines:
                raise RuntimeError(f"Ray job {submission_id} succeeded but printed no {METRICS_MARKER.strip()} line")
            return metrics_lines[-1]
        if status in JOB_TERMINAL:
            tail = "\n".join(logs.strip().splitlines()[-20:])
            raise RuntimeError(f"Ray job {submission_id} {status}: {info.get('message') or ''}\n{tail}".rstrip())
        if timeout_seconds and time.monotonic() - started > timeout_seconds:
            await client.stop(submission_id)
            raise RuntimeError(f"Ray job {submission_id} exceeded RAY_JOB_TIMEOUT_SECONDS ({timeout_seconds}s)")
        await asyncio.sleep(poll_interval_seconds)


async def run_ray_job(job: FineTuneJobSnapshot) -> dict[str, Any]:
    """The `FineTuneRunner` for `FINETUNE_BACKEND=ray`."""
    s = get_settings()
    num_workers = workers_for(job, s.ray_train_num_workers)
    spec = ray_job_spec(job, num_workers=num_workers, use_gpu=s.ray_train_use_gpu, output_dir=s.finetuning_output_dir)
    client = RayJobsClient(s.ray_address)
    try:
        submission_id = await client.submit(spec)
        logger.info("ray_train.submitted", job_id=str(job["id"]), submission_id=submission_id, workers=num_workers)
        await publish_finetune_event(
            job["id"],
            "running",
            f"Ray job {submission_id} submitted ({num_workers} worker(s))",
            metrics={"ray_submission_id": submission_id, "ray_num_workers": num_workers},
        )
        metrics = await wait_for_job(
            client,
            submission_id,
            job["id"],
            poll_interval_seconds=s.ray_job_poll_interval_seconds,
            timeout_seconds=s.ray_job_timeout_seconds,
        )
        metrics["ray_submission_id"] = submission_id
        metrics["ray_num_workers"] = num_workers
        return metrics
    finally:
        await client.aclose()


# ── the driver (runs on the Ray cluster) ────────────────────────


def _train_loop_per_worker(loop_config: dict[str, Any]) -> None:
    """Runs on every Ray Train worker. Ray has already set up the torch process group and the
    torchrun-style environment (RANK/WORLD_SIZE/LOCAL_RANK), which transformers' Trainer reads to run
    data-parallel. Rank 0 owns the adapter path and the reported metrics; other ranks train into a
    scratch directory so their checkpoints never overwrite it."""
    import ray.train

    from src.finetuning.backends.inprocess import run_training

    context = ray.train.get_context()
    rank = context.get_world_rank()
    job = loop_config["job"]
    output_dir = loop_config["output_dir"]
    worker_output = output_dir if rank == 0 else os.path.join(output_dir, f"worker-{rank}")

    def progress(payload: dict[str, Any]) -> None:
        if rank == 0:
            print(PROGRESS_MARKER + json.dumps(payload, default=str), flush=True)

    metrics = run_training(job, progress=progress, output_dir=worker_output)
    ray.train.report(metrics if rank == 0 else {})


def build_trainer(job: FineTuneJobSnapshot, *, num_workers: int, use_gpu: bool, output_dir: str) -> Any:
    """A real `TorchTrainer` from the pure builders — importable only where `ray[train]` is installed."""
    try:
        from ray.train import RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer
    except ImportError as exc:
        raise RuntimeError(_RAY_IMPORT_HELP) from exc

    job_output = os.path.join(output_dir, "ray-jobs", str(job["id"]))
    return TorchTrainer(
        _train_loop_per_worker,
        train_loop_config={"job": job, "output_dir": job_output},
        scaling_config=ScalingConfig(**build_scaling_config(num_workers, use_gpu)),
        run_config=RunConfig(**build_run_config(job["id"], output_dir)),
    )


def main(environ: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environ is None else environ)
    raw = env.get(JOB_ENV)
    if not raw:
        print(f"{JOB_ENV} is not set — this driver expects the job the control plane submitted", file=sys.stderr)
        return 2
    job = json.loads(raw)
    num_workers = int(env.get(NUM_WORKERS_ENV) or 1)
    use_gpu = env.get(USE_GPU_ENV, "1") == "1"
    output_dir = env.get(OUTPUT_DIR_ENV) or get_settings().finetuning_output_dir

    trainer = build_trainer(job, num_workers=num_workers, use_gpu=use_gpu, output_dir=output_dir)
    result = trainer.fit()
    metrics = dict(result.metrics or {})
    if result.error is not None:
        print(f"training failed: {result.error}", file=sys.stderr)
        return 1
    print(METRICS_MARKER + json.dumps(metrics, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
