# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Keystone dev/test tier on RunPod: one GPU pool for the coding model.
# Reasoning/codestral pools follow the same module, added as separate
# `module` blocks when the test budget covers more than one GPU pod — kept
# to one pool by default here to match values-runpod-test.yaml's sized-down
# single-GPU profile.

terraform {
  required_version = ">= 1.6.0" # OpenTofu 1.6+ or Terraform 1.6+ — the HCL here has no provider-specific syntax
  required_providers {
    runpod = {
      source  = "runpod/runpod"
      version = "~> 1.0"
    }
  }
}

variable "runpod_api_key" {
  description = "RunPod API key. Set via TF_VAR_runpod_api_key or a .tfvars file — never commit it."
  type        = string
  sensitive   = true
}

variable "image_registry" {
  description = "Where to pull the vLLM image from. Defaults to Docker Hub for the internet-connected test tier; point at the internal Harbor registry once the air-gap bundle exists."
  type        = string
  default     = "docker.io"
}

provider "runpod" {
  api_key = var.runpod_api_key
}

module "coding_gpu_pool" {
  source = "../../modules/runpod-gpu-pool"

  pool_name   = "keystone-coding-test"
  gpu_type_id = "NVIDIA A100 80GB PCIe" # confirm exact ID against the RunPod console before applying
  gpu_count   = 1
  pod_count   = 1
  pod_type    = "ON_DEMAND"
  image_name  = "${var.image_registry}/vllm/vllm-openai:v0.6.3"

  ports = "8000/http"
  env = [
    "HF_HUB_DISABLE_TELEMETRY=1",
    "VLLM_NO_USAGE_STATS=1",
  ]
  docker_args = join(" ", [
    "--model Qwen/Qwen2.5-Coder-32B-Instruct",
    "--served-model-name qwen-coder-32b",
    "--tensor-parallel-size 1",
    "--max-model-len 32768",
    "--gpu-memory-utilization 0.92",
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--trust-remote-code",
    "--port 8000",
  ])
}

output "coding_pool_proxy_urls" {
  value = module.coding_gpu_pool.proxy_urls
}
