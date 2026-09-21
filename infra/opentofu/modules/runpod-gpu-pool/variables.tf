# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "pool_name" {
  description = "Name prefix for pods in this pool (e.g. \"keystone-coding-test\")."
  type        = string
}

variable "gpu_type_id" {
  description = "RunPod GPU type identifier, e.g. \"NVIDIA A100 80GB PCIe\". See `runpodctl get gputypes` or the RunPod console for exact current IDs — these are not stable enough to hardcode a default here."
  type        = string
}

variable "gpu_count" {
  description = "GPUs per pod in this pool."
  type        = number
  default     = 1
}

variable "pod_count" {
  description = "Number of pods to provision in this pool (usually 1 for the dev/test tier)."
  type        = number
  default     = 1
}

variable "pod_type" {
  description = "RunPod v2 pod type: ON_DEMAND or SPOT. SPOT (interruptible) is cheaper but can be reclaimed — never use it for anything stateful without checkpointing."
  type        = string
  default     = "ON_DEMAND"

  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.pod_type)
    error_message = "pod_type must be ON_DEMAND or SPOT."
  }
}

variable "image_name" {
  description = "Container image to run, e.g. \"vllm/vllm-openai:v0.6.3\" — pull from the internal registry in air-gapped/production use."
  type        = string
}

variable "container_disk_gb" {
  type    = number
  default = 50
}

variable "volume_gb" {
  description = "Persistent network volume size for model weights (survives pod restarts)."
  type        = number
  default     = 200
}

variable "volume_mount_path" {
  type    = string
  default = "/root/.cache/huggingface"
}

variable "ports" {
  description = "RunPod port config string, e.g. \"8000/http\" or \"8000/http,22/tcp\" for multiple exposed ports."
  type        = string
  default     = "8000/http"
}

variable "env" {
  description = "Environment variables for the pod's container, as \"KEY=VALUE\" strings — the runpod_pod resource's `env` attribute is a list of strings, not a map."
  type        = list(string)
  default     = []
}

variable "docker_args" {
  description = "Full docker run args/command line override (a single string), e.g. the vLLM serve invocation for this model role."
  type        = string
  default     = null
}

variable "cloud_type" {
  description = "RunPod cloud tier: SECURE (RunPod-owned datacenters) or COMMUNITY (cheaper, less isolation — do not use for anything handling client data). Deprecated in the v2 API in favor of a scheduling block, but still the only way to set this via the current provider schema."
  type        = string
  default     = "SECURE"

  validation {
    condition     = contains(["SECURE", "COMMUNITY"], var.cloud_type)
    error_message = "cloud_type must be SECURE or COMMUNITY."
  }
}
