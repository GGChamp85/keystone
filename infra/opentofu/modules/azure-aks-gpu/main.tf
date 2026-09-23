# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# One AKS cluster sized for a Keystone GPU pilot: a resource group, a VNet
# with one node subnet, a CPU system pool for the stateful stack, one
# NVIDIA-A10 GPU pool for the vLLM coding role (tainted so only GPU
# workloads land on it), the Azure Disk and Azure Files CSI drivers (AKS
# managed add-ons), an NFS Azure Files Premium StorageClass for the shared
# model cache, a Premium SSD StorageClass for Postgres/Redis/Qdrant/
# OpenBao, and NVIDIA's k8s-device-plugin so `nvidia.com/gpu` is
# schedulable.
#
# Drivers: AKS installs the NVIDIA driver on GPU VM sizes automatically
# (the node pool's gpu_driver defaults to Install); it does NOT install
# the device plugin, which is why this module installs the
# nvidia-device-plugin Helm chart itself.
#
# Verified so far: `tofu fmt -check`, `tofu init -backend=false`,
# `tofu validate` (CI job iac-validate). No real `apply` has been run on an
# Azure subscription yet — see docs/deployment/verification-log.md.
#
# The kubernetes and helm providers are configured by the calling root
# module from this module's kube_config outputs (see
# environments/azure-pilot/main.tf).

locals {
  resource_group_name = var.resource_group_name != "" ? var.resource_group_name : "${var.cluster_name}-rg"
  tags = merge(var.tags, {
    "keystone.io/cluster" = var.cluster_name
    "ManagedBy"           = "opentofu"
  })
  gpu_taint_key    = "nvidia.com/gpu"
  gpu_taint_value  = "present"
  gpu_taint_effect = "NoSchedule"
}

resource "azurerm_resource_group" "this" {
  name     = local.resource_group_name
  location = var.location
  tags     = local.tags
}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

resource "azurerm_virtual_network" "this" {
  name                = "${var.cluster_name}-vnet"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  address_space       = [var.vnet_cidr]
  tags                = local.tags
}

resource "azurerm_subnet" "nodes" {
  name                 = "nodes"
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [var.nodes_subnet_cidr]

  # NFS Azure Files shares are reachable only from subnets with the
  # Microsoft.Storage service endpoint (or a private endpoint).
  service_endpoints = ["Microsoft.Storage"]
}

# ---------------------------------------------------------------------------
# Control plane + system pool
# ---------------------------------------------------------------------------

resource "azurerm_kubernetes_cluster" "this" {
  name                = var.cluster_name
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  dns_prefix          = var.cluster_name
  kubernetes_version  = var.kubernetes_version
  sku_tier            = var.sku_tier

  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  identity {
    type = "SystemAssigned"
  }

  default_node_pool {
    name                        = "system"
    vm_size                     = var.system_vm_size
    node_count                  = var.system_node_count
    min_count                   = var.system_node_min
    max_count                   = var.system_node_max
    auto_scaling_enabled        = true
    vnet_subnet_id              = azurerm_subnet.nodes.id
    os_disk_size_gb             = 100
    os_sku                      = "Ubuntu"
    temporary_name_for_rotation = "systemtmp"

    node_labels = {
      nodepool = "system"
    }

    upgrade_settings {
      max_surge = "10%"
    }
  }

  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    network_data_plane  = "cilium"
    network_policy      = "cilium" # the chart's NetworkPolicy objects are enforced by Cilium
    pod_cidr            = var.pod_cidr
    service_cidr        = var.service_cidr
    dns_service_ip      = var.dns_service_ip
  }

  storage_profile {
    disk_driver_enabled         = true
    file_driver_enabled         = true
    blob_driver_enabled         = false
    snapshot_controller_enabled = true
  }

  tags = local.tags

  lifecycle {
    ignore_changes = [default_node_pool[0].node_count] # the cluster autoscaler owns this
  }
}

# ---------------------------------------------------------------------------
# GPU pool
# ---------------------------------------------------------------------------

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.this.id
  vm_size               = var.gpu_vm_size
  node_count            = var.gpu_node_count
  min_count             = var.gpu_node_min
  max_count             = var.gpu_node_max
  auto_scaling_enabled  = true
  vnet_subnet_id        = azurerm_subnet.nodes.id
  os_disk_size_gb       = var.gpu_node_disk_gb
  os_sku                = "Ubuntu"
  priority              = "Regular"

  node_labels = {
    nodepool                 = var.gpu_nodepool_label
    "nvidia.com/gpu.present" = "true"
  }

  # Keeps every non-GPU pod off the expensive nodes. The chart's vLLM and
  # training pods carry the matching toleration, as does the device plugin.
  node_taints = ["${local.gpu_taint_key}=${local.gpu_taint_value}:${local.gpu_taint_effect}"]

  tags = local.tags

  lifecycle {
    ignore_changes = [node_count]
  }
}

# ---------------------------------------------------------------------------
# In-cluster: StorageClasses and the NVIDIA device plugin.
# ---------------------------------------------------------------------------

resource "kubernetes_storage_class_v1" "azurefile_nfs" {
  metadata {
    name = "keystone-azurefile-nfs"
  }

  storage_provisioner    = "file.csi.azure.com"
  reclaim_policy         = "Delete" # a pilot's model cache is re-downloadable; switch to Retain for production weights
  volume_binding_mode    = "Immediate"
  allow_volume_expansion = true

  # Premium (SSD-backed) file share over NFS 4.1 — the RWX model cache.
  # The driver creates a Premium FileStorage account in the node resource
  # group on the first PVC.
  parameters = {
    skuName  = "Premium_LRS"
    protocol = "nfs"
  }

  mount_options = ["nconnect=4", "noresvport", "actimeo=30"]

  depends_on = [azurerm_kubernetes_cluster.this]
}

resource "kubernetes_storage_class_v1" "managed_premium" {
  metadata {
    name = "keystone-managed-premium"
  }

  storage_provisioner    = "disk.csi.azure.com"
  reclaim_policy         = "Delete"
  volume_binding_mode    = "WaitForFirstConsumer" # bind in the zone the pod lands in
  allow_volume_expansion = true

  parameters = {
    skuName = "Premium_LRS"
  }

  depends_on = [azurerm_kubernetes_cluster.this]
}

resource "helm_release" "nvidia_device_plugin" {
  count = var.install_nvidia_device_plugin ? 1 : 0

  name             = "nvidia-device-plugin"
  repository       = "https://nvidia.github.io/k8s-device-plugin"
  chart            = "nvidia-device-plugin"
  version          = var.nvidia_device_plugin_version
  namespace        = "nvidia-device-plugin"
  create_namespace = true
  wait             = true
  timeout          = 600

  # Only run the plugin DaemonSet on GPU nodes; the chart's default
  # tolerations already cover the nvidia.com/gpu NoSchedule taint.
  set {
    name  = "nodeSelector.nodepool"
    value = var.gpu_nodepool_label
  }

  depends_on = [azurerm_kubernetes_cluster_node_pool.gpu]
}
