# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "cluster_name" {
  description = "AKS cluster name. Also prefixes the resource group, VNet and subnet this module creates."
  type        = string
}

variable "location" {
  description = "Azure region. NVadsA10_v5 sizes are not in every region — `az vm list-skus --size Standard_NV36ads_A10_v5 --output table` shows where."
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group to create for the cluster. Empty = `<cluster_name>-rg`."
  type        = string
  default     = ""
}

variable "kubernetes_version" {
  description = "AKS Kubernetes version. null = the region's current default (`az aks get-versions --location <region>`)."
  type        = string
  default     = null
}

variable "sku_tier" {
  description = "AKS control-plane tier: Free (no SLA, no hourly fee) or Standard (uptime SLA, billed hourly)."
  type        = string
  default     = "Free"

  validation {
    condition     = contains(["Free", "Standard"], var.sku_tier)
    error_message = "sku_tier must be Free or Standard."
  }
}

variable "vnet_cidr" {
  description = "Address space of the VNet this module creates."
  type        = string
  default     = "10.41.0.0/16"
}

variable "nodes_subnet_cidr" {
  description = "Subnet for every node pool (Azure CNI overlay: pods get their own overlay range, not VNet addresses)."
  type        = string
  default     = "10.41.0.0/20"
}

variable "pod_cidr" {
  description = "Overlay CIDR for pods (Azure CNI overlay mode)."
  type        = string
  default     = "10.244.0.0/16"
}

variable "service_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "dns_service_ip" {
  type    = string
  default = "10.0.0.10"
}

variable "system_vm_size" {
  description = "VM size for the CPU system pool (app, worker, Postgres, Redis, Qdrant, Temporal, OpenBao, observability)."
  type        = string
  default     = "Standard_D4s_v5"
}

variable "system_node_count" {
  type    = number
  default = 2
}

variable "system_node_min" {
  type    = number
  default = 2
}

variable "system_node_max" {
  type    = number
  default = 4
}

variable "gpu_vm_size" {
  description = <<-EOT
    VM size for the GPU pool. Default Standard_NV36ads_A10_v5: one full NVIDIA
    A10 (24 GB), 36 vCPU, 440 GiB RAM — the smallest NVadsA10_v5 size that
    exposes a whole GPU (NV6/NV12/NV18 are fractional). Standard_NC24ads_A100_v4
    (one 80 GB A100) is the next step up.
  EOT
  type        = string
  default     = "Standard_NV36ads_A10_v5"
}

variable "gpu_node_count" {
  type    = number
  default = 1
}

variable "gpu_node_min" {
  description = "Minimum GPU nodes. 0 lets the AKS cluster autoscaler scale the pool to nothing when idle."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  type    = number
  default = 2
}

variable "gpu_node_disk_gb" {
  description = "OS disk for GPU nodes — the vLLM image alone is >10 GB."
  type        = number
  default     = 200
}

variable "gpu_nodepool_label" {
  description = "Value of the `nodepool` label on GPU nodes — the Helm chart's vllm.*.nodeSelector and training.nodeSelector match on it (values-azure.yaml)."
  type        = string
  default     = "gpu-coding"
}

variable "install_nvidia_device_plugin" {
  description = "Install NVIDIA's k8s-device-plugin Helm chart. AKS installs the NVIDIA driver on GPU VM sizes itself but not the device plugin that exposes `nvidia.com/gpu`."
  type        = bool
  default     = true
}

variable "nvidia_device_plugin_version" {
  description = "Chart version of nvidia/nvidia-device-plugin (https://nvidia.github.io/k8s-device-plugin)."
  type        = string
  default     = "0.17.1"
}

variable "tags" {
  description = "Tags applied to every taggable resource."
  type        = map(string)
  default     = {}
}
