# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

output "pod_ids" {
  value = runpod_pod.gpu[*].id
}

output "cluster_ips" {
  description = "RunPod internal cluster IPs (runpod_pod's `cluster_ip` attribute) — usable for pod-to-pod traffic within the same RunPod network, not necessarily reachable from outside it."
  value       = runpod_pod.gpu[*].cluster_ip
}

output "proxy_urls" {
  description = <<-EOT
    RunPod's documented HTTP proxy URL pattern for pods with an exposed
    http port: https://{pod_id}-{port}.proxy.runpod.net — this, not
    cluster_ip, is normally how Keystone Inference's VLLM_*_URL settings
    should reach a RunPod-hosted vLLM pod from outside RunPod's network.
    Only valid for ports declared with the "/http" suffix in var.ports.
  EOT
  value       = [for id in runpod_pod.gpu[*].id : "https://${id}-8000.proxy.runpod.net"]
}
