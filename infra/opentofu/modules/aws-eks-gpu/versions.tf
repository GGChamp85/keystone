# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

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
}
