# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Tests for src/finetuning/backends/runpod_pod.py — the RunPod pod training backend.

The `PodCreateInput` the backend sends is asserted against RunPod's
published OpenAPI document (tests/fixtures/runpod_openapi_pods.json, a
verbatim subset fetched from https://rest.runpod.io/v1/openapi.json), the
same way tests/test_runpod_serverless.py keeps the serverless shapes
honest; the polling state machine is tested as a pure function and then
end to end against a real local HTTP server speaking those shapes
(tests/fixtures/fake_runpod_server.py). No pod is launched: the RunPod
account has no credit, and creating a pod is a billable act.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import uuid
from pathlib import Path

import httpx
import pytest

from src.finetuning.backends import runpod_pod
from src.finetuning.backends.pod_entrypoint import JOB_ENV, OUTPUT_DIR_ENV, STATUS_PORT_ENV
from src.finetuning.backends.runpod_pod import (
    ENTRYPOINT_CMD,
    VOLUME_MOUNT_PATH,
    Decision,
    PodTrainingBackend,
    RunPodPods,
    adapter_volume_path,
    build_pod_payload,
    decide,
    pod_status_url,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_SPEC = json.loads((_FIXTURES / "runpod_openapi_pods.json").read_text())
_POD_CREATE = _SPEC["components"]["schemas"]["PodCreateInput"]["properties"]

_JOB = {
    "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    "tenant_id": uuid.uuid4(),
    "job_type": "lora",
    "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "config": {"training_data": "/runpod-volume/data/train.jsonl", "max_steps": 2},
}


def _payload(**overrides):
    kwargs = {
        "job": _JOB,
        "image": "registry.example/keystone-training:2026.09",
        "network_volume_id": "vol_abc",
        "gpu_type_ids": ["NVIDIA L4", "NVIDIA RTX A5000"],
    }
    kwargs.update(overrides)
    return build_pod_payload(**kwargs)


# ── request shapes against the OpenAPI document ─────────────────


def test_the_fixture_is_the_published_document_not_a_hand_written_one():
    assert _SPEC["_provenance"]["source"] == "https://rest.runpod.io/v1/openapi.json"
    assert _SPEC["paths"]["/pods"]["post"]["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/PodCreateInput"
    }
    assert _SPEC["paths"]["/pods/{podId}"]["delete"]["responses"]["204"] is not None


def test_pod_payload_uses_only_documented_pod_create_input_fields():
    payload = _payload()
    assert set(payload) <= set(_POD_CREATE), sorted(set(payload) - set(_POD_CREATE))


def test_pod_payload_values_match_the_documented_enums_and_types():
    payload = _payload()
    assert payload["cloudType"] in _POD_CREATE["cloudType"]["enum"]
    assert payload["computeType"] in _POD_CREATE["computeType"]["enum"]
    assert payload["gpuTypePriority"] in _POD_CREATE["gpuTypePriority"]["enum"]
    assert set(payload["gpuTypeIds"]) <= set(_POD_CREATE["gpuTypeIds"]["items"]["enum"])
    assert isinstance(payload["gpuCount"], int) and isinstance(payload["containerDiskInGb"], int)
    assert isinstance(payload["ports"], list) and all(isinstance(p, str) for p in payload["ports"])
    assert isinstance(payload["dockerStartCmd"], list)
    assert isinstance(payload["env"], dict) and all(isinstance(v, str) for v in payload["env"].values())


def test_pod_payload_runs_the_training_image_with_the_job_in_its_environment():
    payload = _payload()
    assert payload["imageName"] == "registry.example/keystone-training:2026.09"
    assert payload["dockerStartCmd"] == ENTRYPOINT_CMD
    job = json.loads(payload["env"][JOB_ENV])
    assert job == {
        "id": str(_JOB["id"]),
        "job_type": "lora",
        "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "config": {"training_data": "/runpod-volume/data/train.jsonl", "max_steps": 2},
    }
    assert (
        payload["env"][OUTPUT_DIR_ENV]
        == adapter_volume_path(_JOB["id"])
        == f"{VOLUME_MOUNT_PATH}/adapters/{_JOB['id']}"
    )
    assert payload["env"][STATUS_PORT_ENV] == "8000"
    assert payload["ports"] == ["8000/http"]
    assert payload["name"] == "keystone-finetune-12345678"
    assert "HF_TOKEN" not in payload["env"]


def test_pod_payload_puts_the_adapter_on_the_network_volume_not_the_pods_disk():
    payload = _payload()
    assert payload["networkVolumeId"] == "vol_abc"
    assert payload["volumeMountPath"] == VOLUME_MOUNT_PATH
    assert payload["volumeInGb"] == 0
    assert payload["interruptible"] is False  # a spot pod reclaimed mid-run would lose the training


def test_pod_payload_refuses_to_train_without_a_network_volume():
    with pytest.raises(ValueError, match="RUNPOD_NETWORK_VOLUME_ID"):
        _payload(network_volume_id="")


def test_pod_payload_passes_hf_token_and_linger_only_when_given():
    fixture_token = "hf_not_a_real_token"  # noqa: S105 — a test fixture value, not a credential
    env = _payload(hf_token=fixture_token, linger_seconds=600)["env"]
    assert env["HF_TOKEN"] == fixture_token
    assert env["KEYSTONE_POD_LINGER_SECONDS"] == "600"


def test_status_url_follows_runpods_proxy_pattern():
    assert pod_status_url("abc123", 8000) == "https://abc123-8000.proxy.runpod.net/status"


# ── the state machine ───────────────────────────────────────────


def test_decide_waits_while_the_pod_runs_and_nothing_has_reported():
    assert decide({"desiredStatus": "RUNNING"}, None) == Decision("wait")
    assert decide({"desiredStatus": "RUNNING"}, {"state": "running", "progress": {"step": 3}}) == Decision("wait")


def test_decide_completes_on_the_entrypoints_verdict_with_its_metrics():
    decision = decide({"desiredStatus": "RUNNING"}, {"state": "completed", "metrics": {"train_loss": 0.4}})
    assert decision.action == "completed" and decision.metrics == {"train_loss": 0.4}


def test_decide_fails_on_the_entrypoints_failure_or_a_pod_that_died_first():
    assert decide({"desiredStatus": "RUNNING"}, {"state": "failed", "error": "OOM"}).action == "failed"
    assert "OOM" in decide({"desiredStatus": "RUNNING"}, {"state": "failed", "error": "OOM"}).reason
    assert decide({"desiredStatus": "EXITED"}, None).action == "failed"
    assert decide({"desiredStatus": "TERMINATED"}, {"state": "running"}).action == "failed"
    assert decide(None, None).action == "failed"


def test_decide_lets_a_completed_report_win_over_a_pod_that_exited_afterwards():
    # The entrypoint exits after lingering; a poll that lands after that must still read the result.
    assert decide({"desiredStatus": "EXITED"}, {"state": "completed", "metrics": {}}).action == "completed"


# ── end to end against the local server ─────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def fake_runpod():
    port = _free_port()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "fake_runpod_server:app",
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
                raise RuntimeError("fake_runpod_server did not start in time")
        yield base_url
    finally:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)


@pytest.fixture
def published(monkeypatch):
    """Captures the fine-tune events the backend publishes (they go to Redis in production)."""
    events: list[tuple[str, str, dict]] = []

    async def capture(job_id, status, message, metrics=None):
        events.append((status, message, metrics or {}))

    monkeypatch.setattr(runpod_pod, "publish_finetune_event", capture)
    return events


def _backend(base_url: str, **overrides) -> PodTrainingBackend:
    client = RunPodPods(
        "rp_test_key",
        base_url=base_url,
        proxy_url_template=f"{base_url}/proxy/{{pod_id}}",
        status_port=8000,
    )
    kwargs = {
        "image": "registry.example/keystone-training:2026.09",
        "network_volume_id": "vol_abc",
        "gpu_type_ids": ["NVIDIA L4"],
        "poll_interval_seconds": 0.05,
    }
    kwargs.update(overrides)
    return PodTrainingBackend(client, **kwargs)


async def _scenario(base_url: str, **body):
    async with httpx.AsyncClient() as h:
        assert (await h.post(f"{base_url}/scenario", json=body)).status_code == 200


async def _recorded(base_url: str) -> dict:
    async with httpx.AsyncClient() as h:
        return (await h.get(f"{base_url}/recorded")).json()


async def test_the_whole_loop_creates_polls_relays_progress_reads_metrics_and_deletes_the_pod(fake_runpod, published):
    await _scenario(
        fake_runpod,
        status_script=[
            None,  # image still pulling: the proxy has nothing to reach
            {"state": "running", "progress": {"step": 1, "total_steps": 2, "loss": 2.5}},
            {"state": "running", "progress": {"step": 1, "total_steps": 2, "loss": 2.5}},  # same step: no new event
            {"state": "running", "progress": {"step": 2, "total_steps": 2, "loss": 2.1}},
            {"state": "completed", "metrics": {"train_loss": 2.3, "eval_loss": 2.0, "adapter_path": "/ignored"}},
        ],
    )
    backend = _backend(fake_runpod)
    try:
        metrics = await backend.run(_JOB)
    finally:
        await backend._client.aclose()

    assert metrics["train_loss"] == 2.3 and metrics["eval_loss"] == 2.0
    assert metrics["adapter_path"] == f"/runpod-volume/adapters/{_JOB['id']}"
    assert metrics["pod_id"].startswith("pod") and metrics["pod_cost_per_hour_usd"] == 0.44
    assert metrics["pod_seconds"] >= 0

    recorded = await _recorded(fake_runpod)
    methods = [(m, p.split("/")[1]) for m, p, _ in recorded["requests"]]
    assert methods[0] == ("POST", "pods")
    assert ("DELETE", "pods") in methods, "the pod must be deleted once the result is in — billing stops"
    assert recorded["pods"] == {}, "no pod left behind"
    assert recorded["shutdown_requests"] == 1
    created = recorded["requests"][0][2]
    assert set(created) <= set(_POD_CREATE)

    statuses = [s for s, _, _ in published]
    assert statuses[0] == "running" and "created" in published[0][1]
    progress_events = [m for s, _, m in published if "step" in m]
    assert [m["step"] for m in progress_events] == [1, 2], "one event per new step, not per poll"
    assert published[1][1] == "step 1/2, loss 2.5000"


async def test_a_pod_that_exits_before_reporting_is_a_failure_and_is_still_deleted(fake_runpod, published):
    await _scenario(
        fake_runpod,
        status_script=[{"state": "running", "progress": {}}, {"state": "running", "progress": {}}],
        pod_status_after={"polls": 2, "desiredStatus": "EXITED"},
    )
    backend = _backend(fake_runpod)
    try:
        with pytest.raises(RuntimeError, match="EXITED before the entrypoint reported completion"):
            await backend.run(_JOB)
    finally:
        await backend._client.aclose()
    recorded = await _recorded(fake_runpod)
    assert recorded["pods"] == {}


async def test_the_entrypoints_failure_is_the_error_the_job_records(fake_runpod, published):
    await _scenario(fake_runpod, status_script=[{"state": "failed", "error": "RuntimeError: CUDA out of memory"}])
    backend = _backend(fake_runpod)
    try:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            await backend.run(_JOB)
    finally:
        await backend._client.aclose()
    assert (await _recorded(fake_runpod))["pods"] == {}


async def test_the_timeout_stops_and_deletes_a_pod_that_never_finishes(fake_runpod, published):
    await _scenario(fake_runpod, status_script=[{"state": "running", "progress": {}}])
    backend = _backend(fake_runpod, timeout_seconds=1)
    try:
        with pytest.raises(RuntimeError, match="RUNPOD_POD_TIMEOUT_SECONDS"):
            await backend.run(_JOB)
    finally:
        await backend._client.aclose()
    assert (await _recorded(fake_runpod))["pods"] == {}


async def test_an_unfunded_account_is_explained_not_a_raw_500(fake_runpod, published):
    # The exact body RunPod returned for real on 2026-09-22 when creating a serverless endpoint with $0 credit.
    body = '{"error":"create pod: graphql: You must have at least $0.01 in your account balance.","status":500}'
    await _scenario(fake_runpod, create_error={"status": 500, "body": body})
    backend = _backend(fake_runpod)
    try:
        with pytest.raises(RuntimeError, match="no credit"):
            await backend.run(_JOB)
    finally:
        await backend._client.aclose()


async def test_the_server_rejects_a_payload_with_an_undocumented_field(fake_runpod):
    """The fixture enforces the schema, so a wrong key in the builder would fail the loop test above."""
    async with httpx.AsyncClient(base_url=fake_runpod) as h:
        resp = await h.post("/pods", json={**_payload(), "gpuTypeId": "NVIDIA L4"})
    assert resp.status_code == 400 and "gpuTypeId" in resp.text


def test_client_refuses_an_empty_api_key():
    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        RunPodPods("")


def test_backend_from_settings_names_what_is_missing(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")
    monkeypatch.delenv("RUNPOD_NETWORK_VOLUME_ID", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="RUNPOD_NETWORK_VOLUME_ID"):
            runpod_pod.backend_from_settings()
    finally:
        get_settings.cache_clear()
