# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — train on a rented RunPod GPU pod (`FINETUNE_BACKEND=runpod_pod`).

ADR 0005: training does not run serverless; a short-lived on-demand pod
runs docker/training.Dockerfile's image with the job in its environment
and writes the adapter to a RunPod network volume
(`/runpod-volume/adapters/<job>`) that a serverless endpoint created with
the same volume can serve. This module is the control-plane half:

1. `build_pod_payload` — a `PodCreateInput` for RunPod's REST API
   (`POST /pods`), field names taken from RunPod's published OpenAPI
   document (tests/fixtures/runpod_openapi_pods.json) and asserted in
   tests/test_runpod_pod_backend.py, exactly as the serverless client is.
2. `RunPodPods` — the real HTTP client (create / get / stop / delete, and
   the pod's own status endpoint through RunPod's proxy).
3. `decide` + `PodTrainingBackend.run` — the polling state machine: wait
   while the pod runs and its entrypoint reports progress (relayed as
   fine-tune events), finish on the entrypoint's `completed`, fail on its
   `failed` or on a pod that exited or was terminated first, and always
   delete the pod afterwards so billing stops.

Honest scope: no pod has been launched — the RunPod account has no credit
(the same real HTTP 500 `keystone deploy runpod-serverless` met). The
request shapes are checked against the OpenAPI document and the whole
loop runs against a local HTTP server that speaks those shapes
(tests/fixtures/fake_runpod_server.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from src.cli.runpod_serverless import REST_BASE_URL, explain_api_error
from src.config import get_settings
from src.finetuning.backends.inprocess import FineTuneJobSnapshot, format_progress
from src.finetuning.backends.pod_entrypoint import (
    JOB_ENV,
    LINGER_ENV,
    OUTPUT_DIR_ENV,
    STATUS_PORT_ENV,
)
from src.finetuning.events import publish_finetune_event

logger = structlog.get_logger(__name__)

VOLUME_MOUNT_PATH = "/runpod-volume"
ENTRYPOINT_CMD = ["python", "-m", "src.finetuning.backends.pod_entrypoint"]
DEFAULT_PROXY_URL_TEMPLATE = "https://{pod_id}-{port}.proxy.runpod.net"

# `Pod.desiredStatus` enum values from the OpenAPI document.
POD_RUNNING = "RUNNING"
POD_EXITED = "EXITED"
POD_TERMINATED = "TERMINATED"


def adapter_volume_path(job_id: Any) -> str:
    """Where the pod writes the adapter on the network volume — the path a serverless worker mounting the
    same volume will find it at."""
    return f"{VOLUME_MOUNT_PATH}/adapters/{job_id}"


def pod_status_url(pod_id: str, port: int, template: str = DEFAULT_PROXY_URL_TEMPLATE) -> str:
    return template.format(pod_id=pod_id, port=port).rstrip("/") + "/status"


def pod_shutdown_url(pod_id: str, port: int, template: str = DEFAULT_PROXY_URL_TEMPLATE) -> str:
    return template.format(pod_id=pod_id, port=port).rstrip("/") + "/shutdown"


def build_pod_payload(
    *,
    job: FineTuneJobSnapshot,
    image: str,
    network_volume_id: str,
    gpu_type_ids: list[str],
    gpu_count: int = 1,
    container_disk_gb: int = 50,
    cloud_type: str = "SECURE",
    status_port: int = 8000,
    linger_seconds: int | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """`PodCreateInput` for a training pod: the training image, the job as environment, the network volume
    at /runpod-volume (adapters outlive the pod), the status port exposed over RunPod's HTTP proxy, and the
    entrypoint as `dockerStartCmd`. `volumeInGb` is 0 on purpose — nothing worth keeping lives on the pod's
    own disk. Only keys the OpenAPI document declares."""
    if not network_volume_id:
        raise ValueError("RUNPOD_NETWORK_VOLUME_ID is required: the adapter must outlive the pod")
    if not gpu_type_ids:
        raise ValueError("RUNPOD_TRAINING_GPU_TYPE_IDS must name at least one gpuTypeId")
    job_id = str(job["id"])
    env: dict[str, str] = {
        JOB_ENV: json.dumps(
            {
                "id": job_id,
                "job_type": job["job_type"],
                "base_model": job["base_model"],
                "config": job.get("config") or {},
            },
            default=str,
        ),
        OUTPUT_DIR_ENV: adapter_volume_path(job_id),
        STATUS_PORT_ENV: str(status_port),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if linger_seconds is not None:
        env[LINGER_ENV] = str(linger_seconds)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return {
        "name": f"keystone-finetune-{job_id[:8]}",
        "imageName": image,
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": list(gpu_type_ids),
        "gpuTypePriority": "availability",
        "gpuCount": gpu_count,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": 0,
        "volumeMountPath": VOLUME_MOUNT_PATH,
        "networkVolumeId": network_volume_id,
        "ports": [f"{status_port}/http"],
        "env": env,
        "dockerStartCmd": list(ENTRYPOINT_CMD),
        "interruptible": False,
    }


# ── the state machine ───────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    action: str  # "wait" | "completed" | "failed"
    reason: str = ""
    metrics: dict[str, Any] | None = None


def decide(pod: dict[str, Any] | None, status: dict[str, Any] | None) -> Decision:
    """One step of the loop, from the latest `GET /pods/{id}` body and the latest reply of the pod's
    status endpoint (None while unreachable — the container may still be pulling the image).

    The entrypoint's own verdict wins: `completed` carries the metrics, `failed` the error. A pod that
    is EXITED or TERMINATED without having said `completed` failed — the container died, or RunPod
    reclaimed it — and a pod record that has disappeared (None) means the same. Everything else waits.
    """
    if status is not None:
        state = status.get("state")
        if state == "completed":
            return Decision("completed", metrics=dict(status.get("metrics") or {}))
        if state == "failed":
            return Decision("failed", reason=f"training failed in the pod: {status.get('error') or 'no detail'}")
    if pod is None:
        return Decision("failed", reason="the pod record is gone (deleted outside Keystone, or never created)")
    desired = pod.get("desiredStatus")
    if desired in (POD_EXITED, POD_TERMINATED):
        return Decision("failed", reason=f"the pod is {desired} before the entrypoint reported completion")
    return Decision("wait")


# ── the client ──────────────────────────────────────────────────


class RunPodPods:
    """Thin, real async client over RunPod's REST pods endpoints plus the pod's proxied status port."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = REST_BASE_URL,
        proxy_url_template: str = DEFAULT_PROXY_URL_TEMPLATE,
        status_port: int = 8000,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError("A RunPod API key is required (RUNPOD_API_KEY)")
        self._proxy_url_template = proxy_url_template
        self._status_port = status_port
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> RunPodPods:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post("/pods", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(explain_api_error(resp.status_code, resp.text))
        return resp.json()

    async def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        resp = await self._client.get(f"/pods/{pod_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def stop_pod(self, pod_id: str) -> None:
        resp = await self._client.post(f"/pods/{pod_id}/stop")
        if resp.status_code != 404:
            resp.raise_for_status()

    async def delete_pod(self, pod_id: str) -> None:
        resp = await self._client.delete(f"/pods/{pod_id}")
        if resp.status_code != 404:
            resp.raise_for_status()

    async def read_status(self, pod_id: str) -> dict[str, Any] | None:
        """The entrypoint's status, or None while the proxy has nothing to reach yet."""
        url = pod_status_url(pod_id, self._status_port, self._proxy_url_template)
        try:
            resp = await self._client.get(url, headers={"Authorization": ""})
        except httpx.TransportError:
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    async def request_shutdown(self, pod_id: str) -> None:
        url = pod_shutdown_url(pod_id, self._status_port, self._proxy_url_template)
        # The pod is deleted right after; this only lets the entrypoint exit cleanly first.
        with contextlib.suppress(httpx.TransportError):
            await self._client.post(url, headers={"Authorization": ""})


class PodTrainingBackend:
    def __init__(
        self,
        client: RunPodPods,
        *,
        image: str,
        network_volume_id: str,
        gpu_type_ids: list[str],
        gpu_count: int = 1,
        container_disk_gb: int = 50,
        cloud_type: str = "SECURE",
        status_port: int = 8000,
        poll_interval_seconds: float = 15.0,
        timeout_seconds: int = 0,
        hf_token: str | None = None,
    ):
        self._client = client
        self._image = image
        self._network_volume_id = network_volume_id
        self._gpu_type_ids = gpu_type_ids
        self._gpu_count = gpu_count
        self._container_disk_gb = container_disk_gb
        self._cloud_type = cloud_type
        self._status_port = status_port
        self._poll_interval = poll_interval_seconds
        self._timeout = timeout_seconds
        self._hf_token = hf_token

    async def run(self, job: FineTuneJobSnapshot) -> dict[str, Any]:
        job_id = job["id"]
        payload = build_pod_payload(
            job=job,
            image=self._image,
            network_volume_id=self._network_volume_id,
            gpu_type_ids=self._gpu_type_ids,
            gpu_count=self._gpu_count,
            container_disk_gb=self._container_disk_gb,
            cloud_type=self._cloud_type,
            status_port=self._status_port,
            hf_token=self._hf_token,
        )
        created = await self._client.create_pod(payload)
        pod_id = created["id"]
        started = time.monotonic()
        logger.info("runpod_pod.created", job_id=str(job_id), pod_id=pod_id, gpu_type_ids=self._gpu_type_ids)
        await publish_finetune_event(job_id, "running", f"RunPod pod {pod_id} created", metrics={"pod_id": pod_id})

        last_step: int | None = None
        try:
            while True:
                pod = await self._client.get_pod(pod_id)
                status = await self._client.read_status(pod_id)
                decision = decide(pod, status)
                if decision.action == "completed":
                    metrics = dict(decision.metrics or {})
                    metrics["adapter_path"] = adapter_volume_path(job_id)
                    metrics["pod_id"] = pod_id
                    metrics["pod_seconds"] = round(time.monotonic() - started, 1)
                    if pod and pod.get("costPerHr") is not None:
                        metrics["pod_cost_per_hour_usd"] = pod["costPerHr"]
                    await self._client.request_shutdown(pod_id)
                    return metrics
                if decision.action == "failed":
                    raise RuntimeError(decision.reason)
                progress = (status or {}).get("progress") or {}
                if progress.get("step") is not None and progress.get("step") != last_step:
                    last_step = progress["step"]
                    await publish_finetune_event(job_id, "running", format_progress(progress), metrics=progress)
                if self._timeout and time.monotonic() - started > self._timeout:
                    raise RuntimeError(f"the pod exceeded RUNPOD_POD_TIMEOUT_SECONDS ({self._timeout}s)")
                await asyncio.sleep(self._poll_interval)
        finally:
            # Whatever happened, the pod must not keep billing.
            try:
                await self._client.delete_pod(pod_id)
                logger.info("runpod_pod.deleted", job_id=str(job_id), pod_id=pod_id)
            except Exception as exc:
                logger.error("runpod_pod.delete_failed", job_id=str(job_id), pod_id=pod_id, error=str(exc))


def backend_from_settings() -> PodTrainingBackend:
    s = get_settings()
    api_key = s.runpod_api_key.get_secret_value() if s.runpod_api_key else ""
    if not api_key:
        raise ValueError("FINETUNE_BACKEND=runpod_pod needs RUNPOD_API_KEY")
    if not s.runpod_network_volume_id:
        raise ValueError(
            "FINETUNE_BACKEND=runpod_pod needs RUNPOD_NETWORK_VOLUME_ID (the adapter must outlive the pod)"
        )
    client = RunPodPods(
        api_key,
        base_url=s.runpod_rest_base_url,
        proxy_url_template=s.runpod_proxy_url_template,
        status_port=s.runpod_pod_status_port,
    )
    return PodTrainingBackend(
        client,
        image=s.runpod_training_image,
        network_volume_id=s.runpod_network_volume_id,
        gpu_type_ids=list(s.runpod_training_gpu_type_ids),
        gpu_count=s.runpod_training_gpu_count,
        container_disk_gb=s.runpod_training_container_disk_gb,
        cloud_type=s.runpod_training_cloud_type,
        status_port=s.runpod_pod_status_port,
        poll_interval_seconds=s.runpod_pod_poll_interval_seconds,
        timeout_seconds=s.runpod_pod_timeout_seconds,
        hf_token=s.hf_token,
    )


async def run_runpod_pod_job(job: FineTuneJobSnapshot) -> dict[str, Any]:
    """The `FineTuneRunner` for `FINETUNE_BACKEND=runpod_pod`."""
    backend = backend_from_settings()
    try:
        return await backend.run(job)
    finally:
        await backend._client.aclose()
