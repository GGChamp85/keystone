# RunPod deployment (cloud GPU test/staging tier)

Rent real GPUs by the hour instead of buying hardware — the fastest way to run Keystone against a real vLLM-served model without owning a GPU. This is the dev/test tier: internet-connected, single GPU pod, no HA. For an air-gapped or multi-node production deployment, see `docs/deployment/KUBERNETES_CLIENT_VPC.md` instead.

## What this actually provisions

`infra/opentofu/environments/runpod-test/` provisions one real RunPod GPU pod running vLLM (`infra/opentofu/modules/runpod-gpu-pool/`), sized down to Qwen2.5-Coder-32B-Instruct on a single GPU — not the full 8-GPU GLM-5.3-Flash production profile, which needs far more VRAM than one test pod has (see the [Models](../../README.md#models) table). Everything else — Postgres, Redis, Qdrant, the Keystone app itself — still runs via the normal `docker-compose.yml`, just pointed at the RunPod pod's public URL instead of a local vLLM container.

## Prerequisites

- A [RunPod](https://runpod.io) account and API key (RunPod console → Settings → API Keys).
- OpenTofu 1.6+ (or Terraform 1.6+) and Docker + Compose v2 on the machine running Keystone's own stack.

## Steps

```bash
# 1. Provision the GPU pod
cd infra/opentofu/environments/runpod-test
export TF_VAR_runpod_api_key=<your RunPod API key>   # never commit this — see main.tf's variable description
tofu init
tofu apply
# Confirm the GPU type ID in main.tf ("NVIDIA A100 80GB PCIe") against
# `runpodctl get gputypes` or the RunPod console before applying — these
# IDs aren't stable enough to hardcode a default.

# 2. Get the pod's real public URL
tofu output coding_pool_proxy_urls
# -> ["https://<pod-id>-8000.proxy.runpod.net"]
# This is RunPod's documented HTTP proxy pattern for a pod with an exposed
# "8000/http" port — use this, not the internal cluster_ip, from outside
# RunPod's network.

# 3. Point Keystone's coding role at it (back in the repo root)
cd ../../../..
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, REDIS_PASSWORD, QDRANT_API_KEY, KEYSTONE_ROOT_ADMIN_TOKEN,
# and VLLM_CODING_URL=https://<pod-id>-8000.proxy.runpod.net/v1

# 4. Bring up the rest of the stack locally, same as any dev deployment
make certs && make build && make up && make db-migrate
make sandbox-images
make api-key

# 5. Confirm the RunPod-hosted model is actually reachable
curl http://localhost:8080/health
# components.vllm_coding should read "healthy" — if it doesn't, curl the
# proxy URL directly (https://<pod-id>-8000.proxy.runpod.net/health) to
# tell apart a RunPod-side problem from a Keystone-side one.
```

From here, everything in the main [README](../../README.md) — submitting a background task, an interactive OpenCode session, the benchmark suite — works exactly as it does against a local vLLM instance, because it's the same OpenAI-compatible interface either way.

## Sizing beyond one GPU

`infra/opentofu/environments/runpod-test/main.tf`'s own comment describes the pattern: add a second `module "reasoning_gpu_pool" { source = "../../modules/runpod-gpu-pool" ... }` block (same module, different `pool_name`/`gpu_type_id`/`docker_args`) for each additional role you want RunPod-hosted, and point the matching `VLLM_*_URL` at its own `proxy_urls` output. `helm/keystone/values-runpod-test.yaml` is a separate, heavier profile for running the *entire* Helm chart (including in-cluster vLLM) on a GPU-enabled Kubernetes cluster rather than standalone RunPod pods — use it only if you already have such a cluster; this repo doesn't include a script that provisions one on RunPod.

## Tear down

```bash
cd infra/opentofu/environments/runpod-test
tofu destroy   # stops billing for the GPU pod immediately
```
