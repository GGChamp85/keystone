# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "cluster_name" {
  value = azurerm_kubernetes_cluster.this.name
}

output "cluster_id" {
  value = azurerm_kubernetes_cluster.this.id
}

output "resource_group_name" {
  value = azurerm_resource_group.this.name
}

output "node_resource_group_name" {
  description = "The AKS-managed resource group holding VMs, disks and the dynamically created Azure Files storage account."
  value       = azurerm_kubernetes_cluster.this.node_resource_group
}

output "location" {
  value = azurerm_resource_group.this.location
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint — feed to the kubernetes/helm providers in the root module. Marked sensitive because azurerm exposes it through the sensitive kube_config block."
  value       = azurerm_kubernetes_cluster.this.kube_config[0].host
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Base64-encoded cluster CA — `base64decode()` it for the kubernetes/helm providers."
  value       = azurerm_kubernetes_cluster.this.kube_config[0].cluster_ca_certificate
  sensitive   = true
}

output "client_certificate" {
  description = "Base64-encoded admin client certificate (local accounts are enabled on this pilot cluster)."
  value       = azurerm_kubernetes_cluster.this.kube_config[0].client_certificate
  sensitive   = true
}

output "client_key" {
  description = "Base64-encoded admin client key."
  value       = azurerm_kubernetes_cluster.this.kube_config[0].client_key
  sensitive   = true
}

output "kubeconfig_command" {
  description = "Run this once to point kubectl/helm at the cluster (uses your current `az login`)."
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.this.name} --name ${azurerm_kubernetes_cluster.this.name} --overwrite-existing"
}

output "rwx_storage_class" {
  description = "ReadWriteMany StorageClass for modelCache.storageClassName / training.storageClassName."
  value       = kubernetes_storage_class_v1.azurefile_nfs.metadata[0].name
}

output "block_storage_class" {
  description = "Block StorageClass for postgres/redis/qdrant/openbao storageClassName."
  value       = kubernetes_storage_class_v1.managed_premium.metadata[0].name
}

output "gpu_node_selector" {
  description = "nodeSelector the Helm chart's GPU roles must use (values-azure.yaml already does)."
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

output "gpu_vm_size" {
  value = var.gpu_vm_size
}

output "helm_install_command" {
  description = "The Helm install that matches this cluster's StorageClasses and node labels."
  value       = "helm upgrade --install keystone helm/keystone --namespace keystone --create-namespace -f helm/keystone/values-client-vpc.yaml -f helm/keystone/values-azure.yaml --wait --timeout 15m"
}
