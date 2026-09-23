# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Tests for src/finetuning/backends/__init__.py's dispatch and the RunPod pod
entrypoint's status server (src/finetuning/backends/pod_entrypoint.py) —
the part of the pod process that needs no torch: a real stdlib HTTP server
answering the status the control plane polls, and the shutdown handshake.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request

import pytest

from src.finetuning.backends import BACKENDS, select_runner
from src.finetuning.backends.inprocess import format_progress, run_inprocess
from src.finetuning.backends.pod_entrypoint import (
    JOB_ENV,
    PodStatus,
    linger,
    load_job_from_env,
    serve_status,
)
from src.finetuning.backends.ray_train import run_ray_job
from src.finetuning.backends.runpod_pod import run_runpod_pod_job


def test_select_runner_maps_every_documented_backend_and_refuses_the_rest():
    assert select_runner("inprocess") is run_inprocess
    assert select_runner("ray") is run_ray_job
    assert select_runner("runpod_pod") is run_runpod_pod_job
    assert BACKENDS == ("inprocess", "ray", "runpod_pod")
    with pytest.raises(ValueError, match="FINETUNE_BACKEND"):
        select_runner("kubeflow")


def test_format_progress_reads_like_the_watch_command():
    assert format_progress({"step": 3, "total_steps": 10, "loss": 1.23456}) == "step 3/10, loss 1.2346"
    assert format_progress({"step": 1, "total_steps": 2}) == "step 1/2"


def test_load_job_from_env_requires_the_job_and_its_identity():
    with pytest.raises(SystemExit, match=JOB_ENV):
        load_job_from_env({})
    with pytest.raises(SystemExit, match="base_model"):
        load_job_from_env({JOB_ENV: json.dumps({"id": "x", "job_type": "lora"})})
    job = load_job_from_env({JOB_ENV: json.dumps({"id": "x", "job_type": "lora", "base_model": "m"})})
    assert job["config"] == {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 — a 127.0.0.1 test URL
        return json.loads(resp.read())


def _post(url: str) -> dict:
    req = urllib.request.Request(url, method="POST")  # noqa: S310 — a 127.0.0.1 test URL
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
        return json.loads(resp.read())


def test_the_pod_status_server_reports_progress_then_the_result_and_honours_shutdown():
    status = PodStatus()
    port = _free_port()
    server = serve_status(status, port)
    try:
        base = f"http://127.0.0.1:{port}"
        assert _get(f"{base}/health") == {"ok": True}
        assert _get(f"{base}/status")["state"] == "starting"

        status.update(state="running", progress={"step": 1, "total_steps": 2, "loss": 2.0})
        body = _get(f"{base}/status")
        assert body["state"] == "running" and body["progress"]["step"] == 1
        assert body["updated_at"] >= body["started_at"]

        status.update(state="completed", metrics={"train_loss": 1.9, "adapter_path": "/runpod-volume/adapters/x"})
        body = _get(f"{base}/status")
        assert body["state"] == "completed" and body["metrics"]["train_loss"] == 1.9

        waiter = threading.Thread(target=linger, args=(status, 0))  # 0 = until asked
        waiter.start()
        assert waiter.is_alive()
        assert _post(f"{base}/shutdown") == {"ok": True, "state": "completed"}
        waiter.join(timeout=5)
        assert not waiter.is_alive(), "the shutdown request must release the lingering entrypoint"

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(f"{base}/nope")
        assert excinfo.value.code == 404
    finally:
        server.shutdown()


def test_linger_gives_up_after_the_safety_bound():
    status = PodStatus()
    linger(status, 1)  # returns after a second without a shutdown request — a dead control plane cannot bill forever
