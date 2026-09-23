# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "cluster_name" {
  value = module.gke.cluster_name
}

output "location" {
  value = module.gke.location
}

output "kubeconfig_command" {
  description = "Step 1 after apply: point kubectl/helm at the cluster."
  value       = module.gke.kubeconfig_command
}

output "helm_install_command" {
  description = "Step 2 after apply: install the chart with the overlay that matches this cluster."
  value       = module.gke.helm_install_command
}

output "rwx_storage_class" {
  value = module.gke.rwx_storage_class
}

output "block_storage_class" {
  value = module.gke.block_storage_class
}

output "gpu_node_selector" {
  value = module.gke.gpu_node_selector
}

output "gpu_toleration" {
  value = module.gke.gpu_toleration
}
