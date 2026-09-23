# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Keystone GPU pilot on Azure: one AKS cluster with a CPU system pool and
# one A10 GPU node, from ../../modules/azure-aks-gpu. Pair with
# helm/keystone/values-azure.yaml (which names the StorageClasses and node
# labels this module creates) — or run `keystone deploy cloud --cloud azure
# --env pilot --apply`, which wraps exactly the tofu commands below.
#
# State is LOCAL by design for a pilot (terraform.tfstate next to this
# file, git-ignored). For a shared/production deployment move it to an
# Azure Storage container before the first apply by uncommenting the
# backend block and running `tofu init -migrate-state`.
#
# Verified: `tofu fmt -check`, `tofu init -backend=false`, `tofu validate`
# in CI. NOT yet applied against a real Azure subscription — see
# docs/deployment/verification-log.md before treating any of this as
# battle-tested.

terraform {
  required_version = ">= 1.6.0" # OpenTofu 1.6+ or Terraform 1.6+ — no provider-specific syntax below

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.17"
    }
  }

  # backend "azurerm" {
  #   resource_group_name  = "<state-rg>"
  #   storage_account_name = "<statestorageaccount>"
  #   container_name       = "tfstate"
  #   key                  = "keystone/azure-pilot/terraform.tfstate"
  #   use_azuread_auth     = true
  # }
}

provider "azurerm" {
  subscription_id = var.subscription_id

  features {
    resource_group {
      prevent_deletion_if_contains_resources = false # a pilot must be destroyable in one command
    }
  }
}

module "aks" {
  source = "../../modules/azure-aks-gpu"

  cluster_name       = var.cluster_name
  location           = var.location
  kubernetes_version = var.kubernetes_version
  sku_tier           = var.sku_tier

  system_vm_size    = var.system_vm_size
  system_node_count = var.system_node_count

  gpu_vm_size    = var.gpu_vm_size
  gpu_node_count = var.gpu_node_count
  gpu_node_min   = var.gpu_node_min
  gpu_node_max   = var.gpu_node_max

  tags = var.tags
}

# The kubernetes/helm providers are configured from the admin kubeconfig
# the module returns (local accounts stay enabled on this pilot cluster;
# for Entra-ID-only clusters switch these to an `exec` block running
# kubelogin).
provider "kubernetes" {
  host                   = module.aks.cluster_endpoint
  client_certificate     = base64decode(module.aks.client_certificate)
  client_key             = base64decode(module.aks.client_key)
  cluster_ca_certificate = base64decode(module.aks.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = module.aks.cluster_endpoint
    client_certificate     = base64decode(module.aks.client_certificate)
    client_key             = base64decode(module.aks.client_key)
    cluster_ca_certificate = base64decode(module.aks.cluster_ca_certificate)
  }
}
