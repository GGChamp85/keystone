# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — RunPod Serverless deployment (`keystone deploy runpod-serverless`).

A real client for RunPod's REST API (https://rest.runpod.io/v1, bearer auth
with the account API key) that stands up a serverless vLLM endpoint serving
an open-weight model, scaled to zero when idle. Request shapes are taken
verbatim from RunPod's published OpenAPI document (TemplateCreateInput,
EndpointCreateInput), not guessed.

Why REST and not the OpenTofu provider used for on-demand pods
(infra/opentofu/modules/runpod-gpu-pool): the provider ships `runpod_pod`,
`runpod_endpoint`, and `runpod_network_volume` resources but no template
resource, while `runpod_endpoint` requires a `template_id` — so a serverless
endpoint cannot be declared end to end in OpenTofu. One tool for one
deployment is the honest choice; pods stay on OpenTofu.

Pure request builders are separated from the HTTP client so payload shapes
are unit-testable without touching the API, matching src/cli/ops.py's argv
builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

REST_BASE_URL = "https://rest.runpod.io/v1"

# RunPod's maintained vLLM worker — OpenAI-compatible, streaming, tool
# calling via TOOL_CALL_PARSER. Tag = the latest GitHub release, present on
# Docker Hub (checked 2026-09-22). Bump deliberately; do not float `latest`.
WORKER_VLLM_IMAGE = "runpod/worker-v1-vllm:v2.27.1"

# The 24GB serverless tier — exact `gpuTypeIds` enum values from the OpenAPI
# document. RunPod schedules a worker on whichever of these is available.
GPU_TIER_24GB = ["NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"]
GPU_TIER_80GB = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_SERVED_NAME = "keystone-coder"


def endpoint_openai_base_url(endpoint_id: str) -> str:
    """RunPod's documented OpenAI-compatible base URL for a serverless endpoint."""
    return f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"


def build_template_payload(
    *,
    name: str,
    model: str,
    served_model_name: str = DEFAULT_SERVED_NAME,
    image: str = WORKER_VLLM_IMAGE,
    max_model_len: int = 16384,
    tool_call_parser: str | None = "hermes",
    container_disk_gb: int = 50,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """`TemplateCreateInput` for a serverless vLLM worker serving `model`.

    `tool_call_parser="hermes"` is vLLM's parser for Qwen2.5's tool-call
    format (the same value src/inference/config.py uses for the
    coding_fallback role); pass None for a model with no native tool calling.
    """
    env: dict[str, str] = {
        "MODEL_NAME": model,
        "MAX_MODEL_LEN": str(max_model_len),
        "OPENAI_SERVED_MODEL_NAME_OVERRIDE": served_model_name,
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if tool_call_parser:
        env["ENABLE_AUTO_TOOL_CHOICE"] = "true"
        env["TOOL_CALL_PARSER"] = tool_call_parser
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return {
        "name": name,
        "imageName": image,
        "isServerless": True,
        "category": "NVIDIA",
        "containerDiskInGb": container_disk_gb,
        "env": env,
        "ports": [],
        "readme": f"Keystone serverless vLLM worker for {model}",
    }


def build_endpoint_payload(
    *,
    name: str,
    template_id: str,
    gpu_type_ids: list[str] | None = None,
    gpu_count: int = 1,
    workers_min: int = 0,
    workers_max: int = 2,
    idle_timeout_seconds: int = 60,
    execution_timeout_ms: int = 600_000,
    flashboot: bool = True,
    network_volume_id: str | None = None,
) -> dict[str, Any]:
    """`EndpointCreateInput`. `workers_min=0` is scale-to-zero: idle cost is $0,
    at the price of a cold start on the first request after `idle_timeout`."""
    payload: dict[str, Any] = {
        "name": name,
        "templateId": template_id,
        "computeType": "GPU",
        "gpuTypeIds": list(gpu_type_ids or GPU_TIER_24GB),
        "gpuCount": gpu_count,
        "workersMin": workers_min,
        "workersMax": workers_max,
        "idleTimeout": idle_timeout_seconds,
        "executionTimeoutMs": execution_timeout_ms,
        "flashboot": flashboot,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
    }
    if network_volume_id:
        payload["networkVolumeId"] = network_volume_id
    return payload


@dataclass(frozen=True)
class Deployment:
    template_id: str
    endpoint_id: str
    name: str
    model: str
    served_model_name: str
    created_template: bool
    created_endpoint: bool

    @property
    def openai_base_url(self) -> str:
        return endpoint_openai_base_url(self.endpoint_id)


class RunPodServerless:
    """Thin, real client over RunPod's REST API. Every method makes a real call."""

    def __init__(self, api_key: str, *, base_url: str = REST_BASE_URL, timeout: float = 30.0):
        if not api_key:
            raise ValueError("A RunPod API key is required (RUNPOD_API_KEY)")
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RunPodServerless:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── raw operations ────────────────────────────────────────────

    def _get(self, path: str) -> Any:
        resp = self._client.get(path)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> None:
        resp = self._client.delete(path)
        if resp.status_code != 404:
            resp.raise_for_status()

    def list_templates(self) -> list[dict[str, Any]]:
        return _as_list(self._get("/templates"))

    def list_endpoints(self) -> list[dict[str, Any]]:
        return _as_list(self._get("/endpoints"))

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any]:
        return self._get(f"/endpoints/{endpoint_id}")

    def create_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/templates", payload)

    def create_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/endpoints", payload)

    def delete_endpoint(self, endpoint_id: str) -> None:
        self._delete(f"/endpoints/{endpoint_id}")

    def delete_template(self, template_id: str) -> None:
        self._delete(f"/templates/{template_id}")

    def find_template_by_name(self, name: str) -> dict[str, Any] | None:
        return next((t for t in self.list_templates() if t.get("name") == name), None)

    def find_endpoint_by_name(self, name: str) -> dict[str, Any] | None:
        return next((e for e in self.list_endpoints() if e.get("name") == name), None)

    # ── the real operation `keystone deploy runpod-serverless` performs ──

    def deploy(
        self,
        *,
        name: str,
        model: str = DEFAULT_MODEL,
        served_model_name: str = DEFAULT_SERVED_NAME,
        gpu_type_ids: list[str] | None = None,
        max_model_len: int = 16384,
        workers_max: int = 2,
        idle_timeout_seconds: int = 60,
        hf_token: str | None = None,
        tool_call_parser: str | None = "hermes",
    ) -> Deployment:
        """Idempotent by name: re-running reuses an existing template and
        endpoint of the same name rather than creating billable duplicates."""
        template = self.find_template_by_name(name)
        created_template = template is None
        if template is None:
            template = self.create_template(
                build_template_payload(
                    name=name,
                    model=model,
                    served_model_name=served_model_name,
                    max_model_len=max_model_len,
                    tool_call_parser=tool_call_parser,
                    hf_token=hf_token,
                )
            )

        endpoint = self.find_endpoint_by_name(name)
        created_endpoint = endpoint is None
        if endpoint is None:
            endpoint = self.create_endpoint(
                build_endpoint_payload(
                    name=name,
                    template_id=template["id"],
                    gpu_type_ids=gpu_type_ids,
                    workers_max=workers_max,
                    idle_timeout_seconds=idle_timeout_seconds,
                )
            )

        return Deployment(
            template_id=template["id"],
            endpoint_id=endpoint["id"],
            name=name,
            model=model,
            served_model_name=served_model_name,
            created_template=created_template,
            created_endpoint=created_endpoint,
        )

    def destroy(self, name: str) -> dict[str, bool]:
        """Deletes the endpoint (stops all billing) and then its template."""
        removed = {"endpoint": False, "template": False}
        endpoint = self.find_endpoint_by_name(name)
        if endpoint is not None:
            self.delete_endpoint(endpoint["id"])
            removed["endpoint"] = True
        template = self.find_template_by_name(name)
        if template is not None:
            self.delete_template(template["id"])
            removed["template"] = True
        return removed


def explain_api_error(status_code: int, body: str) -> str:
    """Turn a RunPod REST error into the action the operator needs to take.

    Observed for real on 2026-09-22: creating an endpoint on an unfunded
    account returns HTTP 500 with `"You must have at least $0.01 in your
    account balance to create an endpoint"` — a 500, not a 402, so the
    status code alone says nothing useful. The template step before it
    succeeds (templates are free), and `deploy` is idempotent by name, so
    the fix is to add credit and re-run the same command.
    """
    lowered = body.lower()
    if "account balance" in lowered:
        return (
            f"RunPod refused to create the endpoint ({status_code}): the account has no credit. "
            "Add credit at runpod.io → Billing, then re-run this exact command — the template "
            "already created is reused, nothing is duplicated."
        )
    if status_code in (401, 403):
        return f"RunPod rejected the API key ({status_code}). Check RUNPOD_API_KEY in .env: {body}"
    return f"RunPod API error {status_code}: {body}"


def _as_list(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in ("endpoints", "templates", "data", "items"):
            if isinstance(body.get(key), list):
                return body[key]
    return []
