# ADR 0005 — RunPod Serverless endpoints are created through RunPod's REST API, not the OpenTofu provider

**Status**: accepted (2026-09-22)

## Context

The first real GPU tier for end-to-end verification is RunPod **Serverless** (scale-to-zero, $0 idle, 24 GB tier serving Qwen2.5-Coder-7B-Instruct). On-demand pods are already provisioned with OpenTofu (`infra/opentofu/modules/runpod-gpu-pool`, provider `runpod/runpod`, which resolves to v1.0.9). A serverless endpoint needs a **template** (image + env) and an **endpoint** (GPU tier, worker bounds, idle timeout). The provider ships `runpod_pod`, `runpod_endpoint` and `runpod_network_volume` resources but **no template resource**, and `runpod_endpoint` requires a `template_id` — so a serverless endpoint cannot be declared end to end in OpenTofu.

## Decision

`keystone deploy runpod-serverless` (`src/cli/runpod_serverless.py`) calls RunPod's REST API (`https://rest.runpod.io/v1`, bearer auth) directly: `POST /templates` then `POST /endpoints`, with request bodies taken from RunPod's published OpenAPI document (`TemplateCreateInput`, `EndpointCreateInput`) and asserted in `tests/test_runpod_serverless.py`. Idempotent by name; `--destroy` removes both. Pods stay on OpenTofu.

## Consequences

- Two provisioning tools for two RunPod products, each the right one; documented side by side in `docs/deployment/RUNPOD_SETUP.md`.
- The worker image is RunPod's maintained `runpod/worker-v1-vllm` (pinned tag); its env → vLLM flag mapping (`ENABLE_LORA`, `LORA_MODULES`, `MAX_LORA_RANK`, …) was verified from its source, which is what "served in seconds" for a promoted adapter relies on.
- Training does not run serverless (request-scoped, execution-timeout-bound, no clean loss streaming): a short-lived on-demand pod runs the training image and syncs the adapter to a network volume the serverless template mounts.
- Real finding: endpoint creation on an unfunded account returns HTTP 500 with a plain-text reason; the CLI translates it into an instruction.
