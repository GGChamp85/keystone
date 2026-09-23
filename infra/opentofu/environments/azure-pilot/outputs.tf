# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "cluster_name" {
  value = module.aks.cluster_name
}

output "resource_group_name" {
  value = module.aks.resource_group_name
}

output "kubeconfig_command" {
  description = "Step 1 after apply: point kubectl/helm at the cluster."
  value       = module.aks.kubeconfig_command
}

output "helm_install_command" {
  description = "Step 2 after apply: install the chart with the overlay that matches this cluster."
  value       = module.aks.helm_install_command
}

output "rwx_storage_class" {
  value = module.aks.rwx_storage_class
}

output "block_storage_class" {
  value = module.aks.block_storage_class
}

output "gpu_node_selector" {
  value = module.aks.gpu_node_selector
}

output "gpu_toleration" {
  value = module.aks.gpu_toleration
}
