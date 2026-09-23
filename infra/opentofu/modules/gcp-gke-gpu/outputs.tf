# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "cluster_name" {
  value = google_container_cluster.this.name
}

output "cluster_id" {
  value = google_container_cluster.this.id
}

output "location" {
  value = google_container_cluster.this.location
}

output "project_id" {
  value = var.project_id
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint (no scheme) — feed `https://<this>` to the kubernetes/helm providers in the root module."
  value       = google_container_cluster.this.endpoint
}

output "cluster_ca_certificate" {
  description = "Base64-encoded cluster CA — `base64decode()` it for the kubernetes/helm providers."
  value       = google_container_cluster.this.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "kubeconfig_command" {
  description = "Run this once to point kubectl/helm at the cluster (uses your current `gcloud auth login`; needs the gke-gcloud-auth-plugin component)."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.this.name} --location ${google_container_cluster.this.location} --project ${var.project_id}"
}

output "network_name" {
  value = google_compute_network.this.name
}

output "rwx_storage_class" {
  description = "ReadWriteMany StorageClass for modelCache.storageClassName / training.storageClassName."
  value       = kubernetes_storage_class_v1.filestore_rwx.metadata[0].name
}

output "block_storage_class" {
  description = "Block StorageClass for postgres/redis/qdrant/openbao storageClassName."
  value       = kubernetes_storage_class_v1.pd_balanced.metadata[0].name
}

output "gpu_node_selector" {
  description = "nodeSelector the Helm chart's GPU roles must use (values-gcp.yaml already does)."
  value       = { nodepool = var.gpu_nodepool_label }
}

output "gpu_toleration" {
  description = "Taint on the GPU pool; the chart's vLLM/training pods tolerate `nvidia.com/gpu` with operator Exists, which covers it."
  value = {
    key      = local.gpu_taint_key
    value    = local.gpu_taint_value
    effect   = local.gpu_taint_effect
    operator = "Equal"
  }
}

output "gpu_machine_type" {
  value = var.gpu_machine_type
}

output "helm_install_command" {
  description = "The Helm install that matches this cluster's StorageClasses and node labels."
  value       = "helm upgrade --install keystone helm/keystone --namespace keystone --create-namespace -f helm/keystone/values-client-vpc.yaml -f helm/keystone/values-gcp.yaml --wait --timeout 15m"
}
