# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Keystone GPU pilot on AWS: one EKS cluster with a CPU system pool and one
# L4 GPU node, from ../../modules/aws-eks-gpu. Pair with
# helm/keystone/values-aws.yaml (which names the StorageClasses and node
# labels this module creates) — or run `keystone deploy cloud --cloud aws
# --env pilot --apply`, which wraps exactly the tofu commands below.
#
# State is LOCAL by design for a pilot (terraform.tfstate next to this
# file, git-ignored). For a shared/production deployment move it to S3
# before the first apply by uncommenting the backend block and running
# `tofu init -migrate-state`.
#
# Verified: `tofu fmt -check`, `tofu init -backend=false`, `tofu validate`
# in CI. NOT yet applied against a real AWS account — see
# docs/deployment/verification-log.md before treating any of this as
# battle-tested.

terraform {
  required_version = ">= 1.6.0" # OpenTofu 1.6+ or Terraform 1.6+ — no provider-specific syntax below

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
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

  # backend "s3" {
  #   bucket       = "<your-state-bucket>"
  #   key          = "keystone/aws-pilot/terraform.tfstate"
  #   region       = "us-east-1"
  #   use_lockfile = true # S3-native state locking (no DynamoDB table needed)
  #   encrypt      = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.tags
  }
}

module "eks" {
  source = "../../modules/aws-eks-gpu"

  cluster_name       = var.cluster_name
  kubernetes_version = var.kubernetes_version

  system_instance_type = var.system_instance_type
  system_node_count    = var.system_node_count

  gpu_instance_type = var.gpu_instance_type
  gpu_node_count    = var.gpu_node_count
  gpu_node_min      = var.gpu_node_min
  gpu_node_max      = var.gpu_node_max

  tags = var.tags
}

# The kubernetes/helm providers are configured from the cluster the module
# creates — the standard EKS pattern. Authentication goes through the AWS
# CLI (`aws eks get-token`), so whoever runs `tofu apply` needs the AWS CLI
# installed and the same credentials that created the cluster (EKS grants
# the creator cluster-admin via bootstrap_cluster_creator_admin_permissions).
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_ca_certificate)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name, "--region", var.aws_region]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_ca_certificate)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", var.cluster_name, "--region", var.aws_region]
    }
  }
}
