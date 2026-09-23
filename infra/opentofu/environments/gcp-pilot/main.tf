# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Keystone GPU pilot on Google Cloud: one GKE Standard cluster with a CPU
# system pool and one L4 GPU node, from ../../modules/gcp-gke-gpu. Pair
# with helm/keystone/values-gcp.yaml (which names the StorageClasses and
# node labels this module creates) — or run `keystone deploy cloud --cloud
# gcp --env pilot --apply`, which wraps exactly the tofu commands below.
#
# State is LOCAL by design for a pilot (terraform.tfstate next to this
# file, git-ignored). For a shared/production deployment move it to a GCS
# bucket before the first apply by uncommenting the backend block and
# running `tofu init -migrate-state`.
#
# Verified: `tofu fmt -check`, `tofu init -backend=false`, `tofu validate`
# in CI. NOT yet applied against a real GCP project — see
# docs/deployment/verification-log.md before treating any of this as
# battle-tested.

terraform {
  required_version = ">= 1.6.0" # OpenTofu 1.6+ or Terraform 1.6+ — no provider-specific syntax below

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
  }

  # backend "gcs" {
  #   bucket = "<your-state-bucket>"
  #   prefix = "keystone/gcp-pilot"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "gke" {
  source = "../../modules/gcp-gke-gpu"

  project_id   = var.project_id
  cluster_name = var.cluster_name
  region       = var.region
  location     = var.location

  system_machine_type = var.system_machine_type
  system_node_count   = var.system_node_count

  gpu_machine_type      = var.gpu_machine_type
  gpu_accelerator_count = var.gpu_accelerator_count
  gpu_node_count        = var.gpu_node_count
  gpu_node_min          = var.gpu_node_min
  gpu_node_max          = var.gpu_node_max

  labels = var.labels
}

# The kubernetes provider authenticates with the OAuth token of whoever runs
# `tofu apply` (`gcloud auth application-default login`) — no kubeconfig
# needed during apply. kubectl/helm afterwards use the kubeconfig_command
# output.
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${module.gke.cluster_endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.cluster_ca_certificate)
}
