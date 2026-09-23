# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "project_id" {
  description = "GCP project (`gcloud config get-value project`). Set via terraform.tfvars or TF_VAR_project_id."
  type        = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "location" {
  description = "A zone for a zonal cluster (default) or a region for a regional one. L4 availability: `gcloud compute accelerator-types list --filter=\"name=nvidia-l4\"`."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  type    = string
  default = "keystone-pilot"
}

variable "system_machine_type" {
  type    = string
  default = "e2-standard-4"
}

variable "system_node_count" {
  type    = number
  default = 2
}

variable "gpu_machine_type" {
  description = "One NVIDIA L4 (24 GB) per node on g2-standard-8."
  type        = string
  default     = "g2-standard-8"
}

variable "gpu_accelerator_count" {
  description = "Must match the GPU count fixed by gpu_machine_type (1 for g2-standard-8)."
  type        = number
  default     = 1
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

variable "labels" {
  type = map(string)
  default = {
    project     = "keystone"
    environment = "pilot"
  }
}
