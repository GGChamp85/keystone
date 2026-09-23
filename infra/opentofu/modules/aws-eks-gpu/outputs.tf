# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "cluster_name" {
  value = aws_eks_cluster.this.name
}

output "cluster_arn" {
  value = aws_eks_cluster.this.arn
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint — feed to the kubernetes/helm providers in the root module."
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_ca_certificate" {
  description = "Base64-encoded cluster CA — `base64decode()` it for the kubernetes/helm providers."
  value       = aws_eks_cluster.this.certificate_authority[0].data
}

output "region" {
  value = data.aws_region.current.region
}

output "kubeconfig_command" {
  description = "Run this once to point kubectl/helm at the cluster (uses your current AWS CLI credentials)."
  value       = "aws eks update-kubeconfig --region ${data.aws_region.current.region} --name ${aws_eks_cluster.this.name}"
}

output "vpc_id" {
  value = aws_vpc.this.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "efs_file_system_id" {
  description = "EFS file system backing the RWX model-cache StorageClass."
  value       = aws_efs_file_system.model_cache.id
}

output "rwx_storage_class" {
  description = "ReadWriteMany StorageClass for modelCache.storageClassName / training.storageClassName."
  value       = kubernetes_storage_class_v1.efs_rwx.metadata[0].name
}

output "block_storage_class" {
  description = "Block StorageClass for postgres/redis/qdrant/openbao storageClassName."
  value       = kubernetes_storage_class_v1.gp3.metadata[0].name
}

output "gpu_node_selector" {
  description = "nodeSelector the Helm chart's GPU roles must use (values-aws.yaml already does)."
  value       = { nodepool = var.gpu_nodepool_label }
}

output "gpu_toleration" {
  description = "Taint on the GPU node group; the chart's vLLM/training pods tolerate `nvidia.com/gpu` with operator Exists, which covers it."
  value = {
    key      = local.gpu_taint_key
    value    = local.gpu_taint_value
    effect   = local.gpu_taint_effect
    operator = "Equal"
  }
}

output "gpu_instance_type" {
  value = var.gpu_instance_type
}

output "helm_install_command" {
  description = "The Helm install that matches this cluster's StorageClasses and node labels."
  value       = "helm upgrade --install keystone helm/keystone --namespace keystone --create-namespace -f helm/keystone/values-client-vpc.yaml -f helm/keystone/values-aws.yaml --wait --timeout 15m"
}
