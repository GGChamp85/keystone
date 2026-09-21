# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Provisions RunPod GPU pods for the dev/test tier — this is the concrete,
# buildable part of the infra plan (RunPod has a real OpenTofu-compatible
# Terraform provider, registry.terraform.io/runpod/runpod); the client-VPC
# production module lives in ../client-vpc-network and is intentionally
# left provider-agnostic pending the client's actual cloud confirmation.
#
# Attribute names below were checked against the provider's actual
# generated schema (github.com/runpod/terraform-provider-runpod,
# internal/provider/resource_pod/pod_resource_gen.go) as of v1.0.9 — pulled
# from source and diffed by hand, not guessed. Full `terraform validate`
# against v1.0.9 currently fails due to an unrelated upstream bug in that
# release (a malformed schema on the unused `runpod_endpoint_workers` data
# source breaks provider schema loading entirely — see
# github.com/runpod/terraform-provider-runpod issues). This resource block
# was validated instead against v1.0.0 (predates that data source) with the
# `type` argument temporarily removed, then `type` was restored by hand
# once the rest was confirmed — re-run that check if bumping the pinned
# version, and prefer a release where the upstream bug is fixed once one
# ships.

terraform {
  required_providers {
    runpod = {
      source  = "runpod/runpod"
      version = "~> 1.0"
    }
  }
}

resource "runpod_pod" "gpu" {
  count = var.pod_count

  name       = "${var.pool_name}-${count.index}"
  type       = var.pod_type # v2 API required field, added after v1.0.0 — not present in the version this was hand-validated against
  image_name = var.image_name

  gpu_type_id = var.gpu_type_id
  gpu_count   = var.gpu_count
  cloud_type  = var.cloud_type

  container_disk_in_gb = var.container_disk_gb
  volume_in_gb         = var.volume_gb
  volume_mount_path    = var.volume_mount_path

  ports = var.ports
  env   = var.env

  docker_args = var.docker_args
}
