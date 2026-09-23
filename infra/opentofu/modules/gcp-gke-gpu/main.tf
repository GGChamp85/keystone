# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# One GKE Standard cluster sized for a Keystone GPU pilot: a VPC-native
# network with Cloud NAT for private nodes, a CPU system pool for the
# stateful stack, one NVIDIA-L4 GPU pool for the vLLM coding role (GKE
# taints it nvidia.com/gpu=present:NoSchedule automatically), the Compute
# Engine persistent-disk and Filestore CSI drivers (GKE managed add-ons), a
# Filestore StorageClass for the shared RWX model cache and a pd-balanced
# StorageClass for Postgres/Redis/Qdrant/OpenBao.
#
# Drivers: with gpu_driver_installation_config set, GKE installs the NVIDIA
# driver on every GPU node AND runs its own device plugin DaemonSet
# (nvidia-gpu-device-plugin in kube-system) — so unlike the AWS and Azure
# modules this one installs no device plugin of its own.
#
# Verified so far: `tofu fmt -check`, `tofu init -backend=false`,
# `tofu validate` (CI job iac-validate). No real `apply` has been run on a
# GCP project yet — see docs/deployment/verification-log.md.
#
# The kubernetes provider is configured by the calling root module from
# this module's cluster outputs (see environments/gcp-pilot/main.tf).

locals {
  labels = merge(var.labels, {
    "keystone-io-cluster" = var.cluster_name # GCP labels allow only lowercase letters, digits, _ and -
    "managed-by"          = "opentofu"
  })
  gpu_taint_key    = "nvidia.com/gpu"
  gpu_taint_value  = "present"
  gpu_taint_effect = "NoSchedule"
}

# ---------------------------------------------------------------------------
# APIs the cluster, Filestore and NAT need (idempotent; left enabled on destroy).
# ---------------------------------------------------------------------------

resource "google_project_service" "required" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "file.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Network: custom VPC, one subnet with pod/service secondary ranges, Cloud
# NAT so private nodes can pull images and weights.
# ---------------------------------------------------------------------------

resource "google_compute_network" "this" {
  project                 = var.project_id
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.required]
}

resource "google_compute_subnetwork" "nodes" {
  project                  = var.project_id
  name                     = "${var.cluster_name}-nodes"
  region                   = var.region
  network                  = google_compute_network.this.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }
}

resource "google_compute_router" "this" {
  project = var.project_id
  name    = "${var.cluster_name}-router"
  region  = var.region
  network = google_compute_network.this.id
}

resource "google_compute_router_nat" "this" {
  project                            = var.project_id
  name                               = "${var.cluster_name}-nat"
  region                             = var.region
  router                             = google_compute_router.this.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ---------------------------------------------------------------------------
# Node service account (least privilege: logs, metrics, registry pulls).
# ---------------------------------------------------------------------------

resource "google_service_account" "nodes" {
  project      = var.project_id
  account_id   = "${var.cluster_name}-nodes"
  display_name = "GKE node service account for ${var.cluster_name}"
}

resource "google_project_iam_member" "nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/artifactregistry.reader",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.nodes.email}"
}

# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------

resource "google_container_cluster" "this" {
  project  = var.project_id
  name     = var.cluster_name
  location = var.location

  network    = google_compute_network.this.id
  subnetwork = google_compute_subnetwork.nodes.id

  # The default pool is deleted right after creation; every real node lives
  # in the explicit pools below so they can be sized and tainted separately.
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = var.deletion_protection

  networking_mode   = "VPC_NATIVE"
  datapath_provider = "ADVANCED_DATAPATH" # GKE Dataplane V2: enforces the chart's NetworkPolicy objects without Calico

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false # public API endpoint so `keystone deploy cloud` works from a laptop
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
    http_load_balancing {
      disabled = false
    }
  }

  resource_labels = local.labels

  depends_on = [google_project_service.required]
}

# ---------------------------------------------------------------------------
# Node pools
# ---------------------------------------------------------------------------

resource "google_container_node_pool" "system" {
  project    = var.project_id
  name       = "system"
  cluster    = google_container_cluster.this.id
  location   = var.location
  node_count = var.system_node_count

  autoscaling {
    min_node_count = var.system_node_min
    max_node_count = var.system_node_max
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.system_machine_type
    disk_size_gb    = 100
    disk_type       = "pd-balanced"
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = merge(local.labels, { nodepool = "system" })

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  lifecycle {
    ignore_changes = [node_count] # the cluster autoscaler owns this
  }
}

resource "google_container_node_pool" "gpu" {
  project    = var.project_id
  name       = "gpu"
  cluster    = google_container_cluster.this.id
  location   = var.location
  node_count = var.gpu_node_count

  autoscaling {
    min_node_count = var.gpu_node_min
    max_node_count = var.gpu_node_max
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.gpu_machine_type
    disk_size_gb    = var.gpu_node_disk_gb
    disk_type       = "pd-balanced"
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = merge(local.labels, {
      nodepool                 = var.gpu_nodepool_label
      "nvidia.com/gpu.present" = "true"
    })

    guest_accelerator {
      type  = var.gpu_accelerator_type
      count = var.gpu_accelerator_count

      gpu_driver_installation_config {
        gpu_driver_version = var.gpu_driver_version
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # GKE adds the nvidia.com/gpu=present:NoSchedule taint to GPU pools by
    # itself; it is declared here too so the plan shows it and so a pool
    # edit never silently drops it. The chart's vLLM/training pods carry
    # the matching toleration.
    taint {
      key    = local.gpu_taint_key
      value  = local.gpu_taint_value
      effect = "NO_SCHEDULE"
    }
  }

  lifecycle {
    ignore_changes = [node_count]
  }
}

# ---------------------------------------------------------------------------
# In-cluster: StorageClasses. (No device plugin — GKE runs its own, see the
# file-level comment.)
# ---------------------------------------------------------------------------

resource "kubernetes_storage_class_v1" "filestore_rwx" {
  metadata {
    name = "keystone-filestore-rwx"
  }

  storage_provisioner    = "filestore.csi.storage.gke.io"
  reclaim_policy         = "Delete" # a pilot's model cache is re-downloadable; switch to Retain for production weights
  volume_binding_mode    = "Immediate"
  allow_volume_expansion = true

  # One Filestore instance per PVC on this cluster's VPC. `standard` is
  # Basic HDD with a 1 TiB minimum: the chart's 500Gi modelCache request is
  # rounded up to 1 TiB by the driver.
  parameters = {
    tier    = var.filestore_tier
    network = google_compute_network.this.name
  }

  depends_on = [google_container_node_pool.system]
}

resource "kubernetes_storage_class_v1" "pd_balanced" {
  metadata {
    name = "keystone-pd-balanced"
  }

  storage_provisioner    = "pd.csi.storage.gke.io"
  reclaim_policy         = "Delete"
  volume_binding_mode    = "WaitForFirstConsumer" # bind in the zone the pod lands in
  allow_volume_expansion = true

  parameters = {
    type = "pd-balanced"
  }

  depends_on = [google_container_node_pool.system]
}
