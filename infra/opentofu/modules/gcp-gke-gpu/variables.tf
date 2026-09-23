# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "project_id" {
  description = "GCP project to create the cluster in (billing must be enabled)."
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name. Also prefixes the VPC, subnet, router and service account this module creates."
  type        = string
}

variable "region" {
  description = "Region for the VPC subnet and Cloud NAT."
  type        = string
  default     = "us-central1"
}

variable "location" {
  description = "Cluster location: a zone (e.g. us-central1-a) for a zonal cluster — one control plane, no management fee on the first zonal cluster per billing account — or a region for a regional (3-master) cluster. L4 availability varies by zone: `gcloud compute accelerator-types list --filter=\"name=nvidia-l4\"`."
  type        = string
  default     = "us-central1-a"
}

variable "release_channel" {
  description = "GKE release channel for automatic control-plane/node upgrades."
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "release_channel must be RAPID, REGULAR or STABLE."
  }
}

variable "subnet_cidr" {
  description = "Primary range for nodes."
  type        = string
  default     = "10.42.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary range for pods (VPC-native cluster)."
  type        = string
  default     = "10.44.0.0/14"
}

variable "services_cidr" {
  description = "Secondary range for Services."
  type        = string
  default     = "10.48.0.0/20"
}

variable "master_ipv4_cidr_block" {
  description = "/28 for the private control-plane endpoint peering."
  type        = string
  default     = "172.16.0.0/28"
}

variable "system_machine_type" {
  description = "Machine type for the CPU system pool (app, worker, Postgres, Redis, Qdrant, Temporal, OpenBao, observability)."
  type        = string
  default     = "e2-standard-4"
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

variable "gpu_machine_type" {
  description = <<-EOT
    Machine type for the GPU pool. Default g2-standard-8: one NVIDIA L4
    (24 GB), 8 vCPU, 32 GiB RAM. G2 machine types have a fixed GPU count;
    gpu_accelerator_count must match it (g2-standard-8/12/16/32 = 1 L4,
    g2-standard-24 = 2, g2-standard-48 = 4, g2-standard-96 = 8).
  EOT
  type        = string
  default     = "g2-standard-8"
}

variable "gpu_accelerator_type" {
  description = "GPU model attached to each GPU node."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_accelerator_count" {
  description = "GPUs per node — must match what gpu_machine_type carries."
  type        = number
  default     = 1
}

variable "gpu_driver_version" {
  description = "GKE-managed NVIDIA driver install: LATEST, DEFAULT, or INSTALLATION_DISABLED (then you install the driver yourself)."
  type        = string
  default     = "LATEST"

  validation {
    condition     = contains(["LATEST", "DEFAULT", "INSTALLATION_DISABLED"], var.gpu_driver_version)
    error_message = "gpu_driver_version must be LATEST, DEFAULT or INSTALLATION_DISABLED."
  }
}

variable "gpu_node_count" {
  type    = number
  default = 1
}

variable "gpu_node_min" {
  description = "Minimum GPU nodes. 0 lets the GKE cluster autoscaler scale the pool to nothing when idle."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  type    = number
  default = 2
}

variable "gpu_node_disk_gb" {
  description = "Boot disk for GPU nodes — the vLLM image alone is >10 GB."
  type        = number
  default     = 200
}

variable "gpu_nodepool_label" {
  description = "Value of the `nodepool` label on GPU nodes — the Helm chart's vllm.*.nodeSelector and training.nodeSelector match on it (values-gcp.yaml)."
  type        = string
  default     = "gpu-coding"
}

variable "filestore_tier" {
  description = "Filestore tier behind the RWX StorageClass: standard (Basic HDD, 1 TiB minimum), premium (Basic SSD, 2.5 TiB minimum), zonal, enterprise. The CSI driver rounds a smaller PVC up to the tier's minimum."
  type        = string
  default     = "standard"

  validation {
    condition     = contains(["standard", "premium", "zonal", "enterprise"], var.filestore_tier)
    error_message = "filestore_tier must be standard, premium, zonal or enterprise."
  }
}

variable "deletion_protection" {
  description = "Refuse `destroy` on the cluster. Off for a pilot so `keystone deploy cloud --destroy` works; turn on for production."
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels applied to the cluster and node VMs."
  type        = map(string)
  default     = {}
}
