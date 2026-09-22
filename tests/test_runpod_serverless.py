# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Tests for src/cli/runpod_serverless.py. The request builders are pure and
asserted against the field names in RunPod's published OpenAPI document
(TemplateCreateInput / EndpointCreateInput) — a payload with the wrong key
is a 4xx at deploy time, so this is what keeps the shapes honest without
paying for an endpoint. The one real-API test is read-only (a list call,
free) and self-skips without RUNPOD_API_KEY; creating an endpoint is a
deliberate, billable act done by `keystone deploy runpod-serverless`, not
by the test suite.
"""

from __future__ import annotations

import os

import pytest

from src.cli.runpod_serverless import (
    GPU_TIER_24GB,
    GPU_TIER_80GB,
    WORKER_VLLM_IMAGE,
    RunPodServerless,
    _as_list,
    build_endpoint_payload,
    build_template_payload,
    endpoint_openai_base_url,
    explain_api_error,
)

# Exact `gpuTypeIds` enum values from the OpenAPI document (a subset).
_KNOWN_GPU_TYPE_IDS = {
    "NVIDIA L4",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
}


def test_gpu_tiers_only_use_documented_gpu_type_ids():
    assert set(GPU_TIER_24GB) <= _KNOWN_GPU_TYPE_IDS
    assert set(GPU_TIER_80GB) <= _KNOWN_GPU_TYPE_IDS


def test_openai_base_url_matches_runpods_documented_pattern():
    assert endpoint_openai_base_url("abc123xyz") == "https://api.runpod.ai/v2/abc123xyz/openai/v1"


def test_template_payload_matches_template_create_input():
    payload = build_template_payload(name="keystone-coder", model="Qwen/Qwen2.5-Coder-7B-Instruct")
    # Required by the schema
    assert payload["name"] == "keystone-coder"
    assert payload["imageName"] == WORKER_VLLM_IMAGE
    # A serverless worker, not a pod template
    assert payload["isServerless"] is True
    assert payload["category"] == "NVIDIA"
    assert isinstance(payload["containerDiskInGb"], int)
    assert payload["ports"] == []
    # worker-vllm's own env contract
    env = payload["env"]
    assert env["MODEL_NAME"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert env["MAX_MODEL_LEN"] == "16384"
    assert env["OPENAI_SERVED_MODEL_NAME_OVERRIDE"] == "keystone-coder"
    assert env["TOOL_CALL_PARSER"] == "hermes"
    assert env["ENABLE_AUTO_TOOL_CHOICE"] == "true"
    assert "HF_TOKEN" not in env
    # Only documented top-level keys
    assert set(payload) <= {
        "name",
        "imageName",
        "isServerless",
        "category",
        "containerDiskInGb",
        "env",
        "ports",
        "readme",
    }


def test_template_payload_omits_tool_env_for_a_model_without_native_tool_calling():
    env = build_template_payload(name="n", model="m", tool_call_parser=None)["env"]
    assert "TOOL_CALL_PARSER" not in env
    assert "ENABLE_AUTO_TOOL_CHOICE" not in env


def test_template_payload_passes_hf_token_only_when_given():
    fixture_token = "hf_not_a_real_token"  # noqa: S105 — a test fixture value, not a credential
    env = build_template_payload(name="n", model="m", hf_token=fixture_token)["env"]
    assert env["HF_TOKEN"] == fixture_token


def test_endpoint_payload_matches_endpoint_create_input_and_scales_to_zero():
    payload = build_endpoint_payload(name="keystone-coder", template_id="tpl123")
    assert payload["templateId"] == "tpl123"  # the schema's only required field
    assert payload["name"] == "keystone-coder"
    assert payload["computeType"] == "GPU"
    assert payload["gpuTypeIds"] == GPU_TIER_24GB
    assert payload["gpuCount"] == 1
    assert payload["workersMin"] == 0  # scale-to-zero: $0 idle
    assert payload["workersMax"] == 2
    assert payload["idleTimeout"] == 60
    assert payload["executionTimeoutMs"] == 600_000
    assert payload["flashboot"] is True
    assert payload["scalerType"] == "QUEUE_DELAY"
    assert "networkVolumeId" not in payload
    assert set(payload) <= {
        "name",
        "templateId",
        "computeType",
        "gpuTypeIds",
        "gpuCount",
        "workersMin",
        "workersMax",
        "idleTimeout",
        "executionTimeoutMs",
        "flashboot",
        "scalerType",
        "scalerValue",
        "networkVolumeId",
    }


def test_endpoint_payload_attaches_a_network_volume_only_when_given():
    payload = build_endpoint_payload(name="n", template_id="t", network_volume_id="vol1", gpu_type_ids=GPU_TIER_80GB)
    assert payload["networkVolumeId"] == "vol1"
    assert payload["gpuTypeIds"] == GPU_TIER_80GB


def test_as_list_accepts_the_shapes_the_api_may_return():
    assert _as_list([{"id": "a"}]) == [{"id": "a"}]
    assert _as_list({"endpoints": [{"id": "b"}]}) == [{"id": "b"}]
    assert _as_list({"templates": [{"id": "c"}]}) == [{"id": "c"}]
    assert _as_list({"unexpected": 1}) == []


def test_explain_api_error_turns_the_real_unfunded_account_500_into_an_instruction():
    # The exact body RunPod returned on 2026-09-22 for an account with $0 credit.
    body = (
        '{"error":"create endpoint: create endpoint: graphql: You must have at least $0.01 '
        'in your account balance to create an endpoint.","status":500}'
    )
    message = explain_api_error(500, body)
    assert "no credit" in message
    assert "Billing" in message
    assert "re-run" in message


def test_explain_api_error_names_the_key_on_401_and_passes_other_errors_through():
    assert "RUNPOD_API_KEY" in explain_api_error(401, '{"error":"unauthorized"}')
    assert explain_api_error(500, "boom") == "RunPod API error 500: boom"


def test_client_refuses_an_empty_api_key():
    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        RunPodServerless("")


@pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="Needs a real RUNPOD_API_KEY (read-only list call)")
def test_real_api_key_authenticates_and_lists_endpoints():
    with RunPodServerless(os.environ["RUNPOD_API_KEY"]) as rp:
        endpoints = rp.list_endpoints()
        templates = rp.list_templates()
    assert isinstance(endpoints, list)
    assert isinstance(templates, list)
