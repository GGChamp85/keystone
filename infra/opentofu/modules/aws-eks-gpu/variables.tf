# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "cluster_name" {
  description = "EKS cluster name. Also prefixes every VPC/IAM/EFS resource this module creates."
  type        = string
}

variable "kubernetes_version" {
  description = "EKS Kubernetes minor version (`aws eks describe-addon-versions` lists what the region supports)."
  type        = string
  default     = "1.33"
}

variable "vpc_cidr" {
  description = "CIDR for the VPC this module creates. Two public and two private /20 subnets are carved from it (two AZs — EKS requires subnets in at least two)."
  type        = string
  default     = "10.40.0.0/16"
}

variable "system_instance_type" {
  description = "Instance type for the CPU system pool (app, worker, Postgres, Redis, Qdrant, Temporal, OpenBao, observability)."
  type        = string
  default     = "m6i.xlarge"
}

variable "system_node_count" {
  description = "Desired size of the CPU system pool."
  type        = number
  default     = 2
}

variable "system_node_min" {
  type    = number
  default = 2
}

variable "system_node_max" {
  type    = number
  default = 4
}

variable "gpu_instance_type" {
  description = <<-EOT
    Instance type for the GPU pool. Default g6.2xlarge: one NVIDIA L4 (24 GB),
    8 vCPU, 32 GiB host RAM. g6.xlarge carries the same single L4 but only
    16 GiB of host RAM, which is too tight for vLLM to load the ~15 GB of
    Qwen2.5-Coder-7B weights next to the kubelet and system pods — choose it
    only for models under ~4B parameters.
  EOT
  type        = string
  default     = "g6.2xlarge"
}

variable "gpu_node_count" {
  description = "Desired size of the GPU pool. One node = one L4 on the default instance type."
  type        = number
  default     = 1
}

variable "gpu_node_min" {
  description = "Minimum size of the GPU pool. 0 lets the cluster autoscaler (not installed by this module) scale GPUs to nothing when idle."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  type    = number
  default = 2
}

variable "gpu_ami_type" {
  description = "EKS-optimized AMI family for the GPU pool. AL2023_x86_64_NVIDIA ships the NVIDIA driver and container toolkit pre-installed; BOTTLEROCKET_x86_64_NVIDIA also works."
  type        = string
  default     = "AL2023_x86_64_NVIDIA"

  validation {
    condition     = contains(["AL2023_x86_64_NVIDIA", "BOTTLEROCKET_x86_64_NVIDIA", "AL2_x86_64_GPU"], var.gpu_ami_type)
    error_message = "gpu_ami_type must be one of AL2023_x86_64_NVIDIA, BOTTLEROCKET_x86_64_NVIDIA, AL2_x86_64_GPU."
  }
}

variable "gpu_node_disk_gb" {
  description = "Root volume size for GPU nodes — the vLLM image alone is >10 GB and the container runtime keeps every pulled layer here."
  type        = number
  default     = 200
}

variable "gpu_nodepool_label" {
  description = "Value of the `nodepool` label on GPU nodes — the Helm chart's vllm.*.nodeSelector and training.nodeSelector match on it (values-aws.yaml)."
  type        = string
  default     = "gpu-coding"
}

variable "install_nvidia_device_plugin" {
  description = "Install NVIDIA's k8s-device-plugin Helm chart so `nvidia.com/gpu` becomes a schedulable resource. The EKS NVIDIA AMIs ship the driver but not the device plugin."
  type        = bool
  default     = true
}

variable "nvidia_device_plugin_version" {
  description = "Chart version of nvidia/nvidia-device-plugin (https://nvidia.github.io/k8s-device-plugin)."
  type        = string
  default     = "0.17.1"
}

variable "efs_throughput_mode" {
  description = "EFS throughput mode for the model cache: `elastic` (pay per use, best for bursty weight loads) or `bursting`."
  type        = string
  default     = "elastic"

  validation {
    condition     = contains(["elastic", "bursting"], var.efs_throughput_mode)
    error_message = "efs_throughput_mode must be elastic or bursting."
  }
}

variable "tags" {
  description = "Tags applied to every taggable resource."
  type        = map(string)
  default     = {}
}
