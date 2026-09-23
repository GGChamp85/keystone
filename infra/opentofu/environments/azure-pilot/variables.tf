# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "subscription_id" {
  description = "Azure subscription to deploy into (`az account show --query id -o tsv`). Set via terraform.tfvars or TF_VAR_subscription_id."
  type        = string
}

variable "location" {
  description = "Azure region. NVadsA10_v5 sizes are not in every region — `az vm list-skus --size Standard_NV36ads_A10_v5 --output table`."
  type        = string
  default     = "eastus"
}

variable "cluster_name" {
  type    = string
  default = "keystone-pilot"
}

variable "kubernetes_version" {
  description = "null = the region's current AKS default."
  type        = string
  default     = null
}

variable "sku_tier" {
  description = "Free (no SLA, no hourly control-plane fee) or Standard."
  type        = string
  default     = "Free"
}

variable "system_vm_size" {
  type    = string
  default = "Standard_D4s_v5"
}

variable "system_node_count" {
  type    = number
  default = 2
}

variable "gpu_vm_size" {
  description = "One full NVIDIA A10 (24 GB) per node."
  type        = string
  default     = "Standard_NV36ads_A10_v5"
}

variable "gpu_node_count" {
  type    = number
  default = 1
}

variable "gpu_node_min" {
  type    = number
  default = 0
}

variable "gpu_node_max" {
  type    = number
  default = 2
}

variable "tags" {
  type = map(string)
  default = {
    Project     = "keystone"
    Environment = "pilot"
  }
}
