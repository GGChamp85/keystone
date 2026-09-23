# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Tests for src/finetuning/backends/ray_train.py — the Ray Train backend (ADR 0002).

The builders are pure and asserted exactly, so CI verifies the configuration
a real cluster would receive without one. Where `ray[train]` is importable
(the `.venv-train` environment, the training image) the same kwargs
construct the real `ScalingConfig`, `RunConfig` and `TorchTrainer`. The
submission and polling loop runs against a real local server speaking the
Ray Jobs REST API (tests/fixtures/fake_ray_jobs_server.py). Not tested
here: a real multi-worker run — no Ray cluster or GPU was available.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import sys
import uuid
from pathlib import Path

import httpx
import pytest

from src.finetuning.backends import ray_train
from src.finetuning.backends.ray_train import (
    ENTRYPOINT,
    METRICS_MARKER,
    PROGRESS_MARKER,
    RayJobsClient,
    build_run_config,
    build_scaling_config,
    parse_marker_lines,
    ray_job_spec,
    wait_for_job,
    workers_for,
)

_FIXTURES = Path(__file__).parent / "fixtures"

_JOB = {
    "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    "tenant_id": uuid.uuid4(),
    "job_type": "lora",
    "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "config": {"training_data": "/data/finetuning/data/train.jsonl", "guided": {"plan": {"gpu_count": 4}}},
}


# ── pure builders ───────────────────────────────────────────────


def test_scaling_config_is_one_worker_per_gpu():
    assert build_scaling_config(4, True) == {"num_workers": 4, "use_gpu": True}
    assert build_scaling_config(1, False, resources_per_worker={"CPU": 8}) == {
        "num_workers": 1,
        "use_gpu": False,
        "resources_per_worker": {"CPU": 8},
    }
    with pytest.raises(ValueError):
        build_scaling_config(0, True)


def test_run_config_stores_under_the_finetuning_output_dir():
    assert build_run_config(_JOB["id"], "/data/finetuning/output") == {
        "name": "keystone-finetune-12345678-1234-5678-1234-567812345678",
        "storage_path": "/data/finetuning/output/ray",
    }


def test_workers_come_from_the_setting_then_the_plan_then_one():
    assert workers_for(_JOB, 2) == 2
    assert workers_for(_JOB, 0) == 4
    assert workers_for({**_JOB, "config": {}}, 0) == 1


def test_ray_job_spec_submits_this_module_with_the_job_in_the_runtime_env():
    spec = ray_job_spec(_JOB, num_workers=4, use_gpu=True, output_dir="/data/finetuning/output")
    assert spec["entrypoint"] == ENTRYPOINT == "python -m src.finetuning.backends.ray_train"
    assert spec["submission_id"] == "keystone-finetune-12345678-1234-5678-1234-567812345678"
    env = spec["runtime_env"]["env_vars"]
    assert json.loads(env["KEYSTONE_FINETUNE_JOB"]) == {
        "id": str(_JOB["id"]),
        "job_type": "lora",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "config": _JOB["config"],
    }
    assert env["KEYSTONE_RAY_NUM_WORKERS"] == "4" and env["KEYSTONE_RAY_USE_GPU"] == "1"
    assert env["KEYSTONE_FINETUNE_OUTPUT_DIR"] == "/data/finetuning/output"
    assert "working_dir" not in spec["runtime_env"], "the training image already carries src/"
    assert spec["metadata"] == {"keystone_job_id": str(_JOB["id"]), "keystone_job_type": "lora"}
    assert set(spec) == {"entrypoint", "submission_id", "runtime_env", "metadata"}


def test_marker_lines_are_parsed_in_order_and_bad_lines_skipped():
    logs = (
        "2026-09-23 10:00:00 (TrainTrainable pid=1) some ray noise\n"
        f"{PROGRESS_MARKER}{json.dumps({'step': 1})}\n"
        f"{PROGRESS_MARKER}not json\n"
        f"(RayTrainWorker pid=2) {PROGRESS_MARKER}{json.dumps({'step': 2})}\n"
        f"{METRICS_MARKER}{json.dumps({'train_loss': 0.5})}\n"
    )
    assert parse_marker_lines(logs, PROGRESS_MARKER) == [{"step": 1}, {"step": 2}]
    assert parse_marker_lines(logs, METRICS_MARKER) == [{"train_loss": 0.5}]


# ── the real Ray objects, where ray is installed ────────────────


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="needs ray[train] (the finetuning-ray extra)")
def test_the_builders_construct_the_real_torch_trainer_configuration():
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    scaling = ScalingConfig(**build_scaling_config(2, use_gpu=False))
    assert scaling.num_workers == 2 and scaling.use_gpu is False
    run = RunConfig(**build_run_config(_JOB["id"], "/tmp/keystone-out"))  # noqa: S108 — a name, nothing is written
    assert run.name == "keystone-finetune-12345678-1234-5678-1234-567812345678"
    assert str(run.storage_path).rstrip("/").endswith("keystone-out/ray")
    trainer = ray_train.build_trainer(_JOB, num_workers=2, use_gpu=False, output_dir="/tmp/keystone-out")  # noqa: S108
    assert isinstance(trainer, TorchTrainer)


def test_build_trainer_explains_the_missing_extra_when_ray_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setitem(sys.modules, "ray.train", None)
    with pytest.raises(RuntimeError, match="finetuning-ray"):
        ray_train.build_trainer(_JOB, num_workers=1, use_gpu=False, output_dir="/tmp/x")  # noqa: S108


# ── the Ray Jobs API loop against a local server ────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def fake_ray():
    port = _free_port()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "fake_ray_jobs_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        cwd=str(_FIXTURES),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as probe:
            for _ in range(100):
                try:
                    await probe.get(f"{base_url}/recorded", timeout=1.0)
                    break
                except httpx.TransportError:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("fake_ray_jobs_server did not start in time")
        yield base_url
    finally:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)


@pytest.fixture
def published(monkeypatch):
    events: list[tuple[str, str, dict]] = []

    async def capture(job_id, status, message, metrics=None):
        events.append((status, message, metrics or {}))

    monkeypatch.setattr(ray_train, "publish_finetune_event", capture)
    return events


async def _scenario(base_url: str, **body):
    async with httpx.AsyncClient() as h:
        assert (await h.post(f"{base_url}/scenario", json=body)).status_code == 200


async def test_submit_poll_relay_progress_and_return_the_metrics_line(fake_ray, published):
    metrics = {"train_loss": 0.41, "eval_loss": 0.39, "adapter_path": "/data/finetuning/output/ray-jobs/x/adapter"}
    await _scenario(
        fake_ray,
        statuses=["PENDING", "RUNNING", "RUNNING", "SUCCEEDED"],
        logs_by_poll=[
            "",
            f"(RayTrainWorker pid=7) {PROGRESS_MARKER}{json.dumps({'step': 1, 'total_steps': 2, 'loss': 1.2})}\n",
            f"(RayTrainWorker pid=7) {PROGRESS_MARKER}{json.dumps({'step': 1, 'total_steps': 2, 'loss': 1.2})}\n"
            f"(RayTrainWorker pid=7) {PROGRESS_MARKER}{json.dumps({'step': 2, 'total_steps': 2, 'loss': 1.0})}\n",
            f"(RayTrainWorker pid=7) {PROGRESS_MARKER}{json.dumps({'step': 1, 'total_steps': 2, 'loss': 1.2})}\n"
            f"(RayTrainWorker pid=7) {PROGRESS_MARKER}{json.dumps({'step': 2, 'total_steps': 2, 'loss': 1.0})}\n"
            f"{METRICS_MARKER}{json.dumps(metrics)}\n",
        ],
    )
    client = RayJobsClient(fake_ray)
    try:
        spec = ray_job_spec(_JOB, num_workers=2, use_gpu=True, output_dir="/data/finetuning/output")
        submission_id = await client.submit(spec)
        assert submission_id == spec["submission_id"]
        result = await wait_for_job(client, submission_id, _JOB["id"], poll_interval_seconds=0.02)
    finally:
        await client.aclose()
    assert result == metrics
    assert [m["step"] for _, _, m in published] == [1, 2], "each progress line relayed once"
    assert published[0][1] == "step 1/2, loss 1.2000"


async def test_a_duplicate_submission_is_refused_by_the_server_not_run_twice(fake_ray):
    await _scenario(fake_ray)
    client = RayJobsClient(fake_ray)
    try:
        spec = ray_job_spec(_JOB, num_workers=1, use_gpu=False, output_dir="/o")
        await client.submit(spec)
        with pytest.raises(RuntimeError, match="already exists"):
            await client.submit(spec)
    finally:
        await client.aclose()


async def test_a_failed_job_raises_with_rays_message_and_the_log_tail(fake_ray, published):
    await _scenario(
        fake_ray,
        statuses=["RUNNING", "FAILED"],
        logs_by_poll=["", "Traceback ...\nRuntimeError: CUDA out of memory\n"],
        message="Job entrypoint command failed with exit code 1",
    )
    client = RayJobsClient(fake_ray)
    try:
        submission_id = await client.submit(ray_job_spec(_JOB, num_workers=1, use_gpu=True, output_dir="/o"))
        with pytest.raises(RuntimeError) as excinfo:
            await wait_for_job(client, submission_id, _JOB["id"], poll_interval_seconds=0.02)
    finally:
        await client.aclose()
    assert "FAILED" in str(excinfo.value) and "exit code 1" in str(excinfo.value)
    assert "CUDA out of memory" in str(excinfo.value)


async def test_a_succeeded_job_without_a_metrics_line_is_an_error_not_empty_metrics(fake_ray, published):
    await _scenario(fake_ray, statuses=["SUCCEEDED"], logs_by_poll=["nothing useful\n"])
    client = RayJobsClient(fake_ray)
    try:
        submission_id = await client.submit(ray_job_spec(_JOB, num_workers=1, use_gpu=True, output_dir="/o"))
        with pytest.raises(RuntimeError, match="printed no KEYSTONE_FINETUNE_METRICS"):
            await wait_for_job(client, submission_id, _JOB["id"], poll_interval_seconds=0.02)
    finally:
        await client.aclose()


async def test_the_timeout_stops_the_job(fake_ray, published):
    await _scenario(fake_ray, statuses=["RUNNING"])
    client = RayJobsClient(fake_ray)
    try:
        submission_id = await client.submit(ray_job_spec(_JOB, num_workers=1, use_gpu=True, output_dir="/o"))
        with pytest.raises(RuntimeError, match="RAY_JOB_TIMEOUT_SECONDS"):
            await wait_for_job(client, submission_id, _JOB["id"], poll_interval_seconds=0.02, timeout_seconds=1)
        assert (await client.status(submission_id))["status"] == "STOPPED"
    finally:
        await client.aclose()


def test_client_requires_an_address():
    with pytest.raises(ValueError, match="RAY_ADDRESS"):
        RayJobsClient("")
