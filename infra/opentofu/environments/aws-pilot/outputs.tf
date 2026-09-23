# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "kubeconfig_command" {
  description = "Step 1 after apply: point kubectl/helm at the cluster."
  value       = module.eks.kubeconfig_command
}

output "helm_install_command" {
  description = "Step 2 after apply: install the chart with the overlay that matches this cluster."
  value       = module.eks.helm_install_command
}

output "rwx_storage_class" {
  value = module.eks.rwx_storage_class
}

output "block_storage_class" {
  value = module.eks.block_storage_class
}

output "gpu_node_selector" {
  value = module.eks.gpu_node_selector
}

output "gpu_toleration" {
  value = module.eks.gpu_toleration
}

output "efs_file_system_id" {
  value = module.eks.efs_file_system_id
}
