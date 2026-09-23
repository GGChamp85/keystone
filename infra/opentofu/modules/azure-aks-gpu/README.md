# azure-aks-gpu

One AKS cluster sized for a Keystone GPU pilot. Used by `environments/azure-pilot` and paired with `helm/keystone/values-azure.yaml`.

## What it creates

| Resource | Detail |
|---|---|
| Resource group | `<cluster_name>-rg` (or `resource_group_name`) in `location` (default eastus) |
| Network | One VNet (`vnet_cidr`, default 10.41.0.0/16) with a `nodes` subnet (10.41.0.0/20) carrying the `Microsoft.Storage` service endpoint that NFS Azure Files requires |
| AKS cluster | `sku_tier` Free by default, system-assigned identity, OIDC issuer + workload identity on, Azure CNI **overlay** with the Cilium data plane and Cilium network policy (the chart's `NetworkPolicy` objects are enforced), Azure Disk + Azure Files CSI drivers enabled |
| System pool | `system_vm_size` (default Standard_D4s_v5) × `system_node_count` (default 2, autoscaler 2–4), label `nodepool=system` |
| GPU pool | `gpu_vm_size` (default **Standard_NV36ads_A10_v5** — one full A10 24 GB, 36 vCPU, 440 GiB) × `gpu_node_count` (default 1, min 0, max 2), label `nodepool=gpu-coding`, taint `nvidia.com/gpu=present:NoSchedule`, 200 GB OS disk |
| StorageClass `keystone-azurefile-nfs` | `file.csi.azure.com`, `Premium_LRS` + `protocol=nfs` — ReadWriteMany for `modelCache` and `training`; the driver creates a Premium FileStorage account in the node resource group on the first PVC |
| StorageClass `keystone-managed-premium` | `disk.csi.azure.com`, `Premium_LRS`, `WaitForFirstConsumer` — Postgres, Redis, Qdrant, OpenBao |
| NVIDIA device plugin | Helm chart `nvidia-device-plugin` 0.17.1 in namespace `nvidia-device-plugin`, pinned to the GPU nodes |

**Drivers:** AKS installs the NVIDIA driver on GPU VM sizes itself (the node pool's GPU driver setting defaults to install). It does not install the device plugin, so this module installs it (`install_nvidia_device_plugin = true`).

**Why NV36ads_A10_v5:** it is the smallest NVadsA10_v5 size that exposes a whole GPU — NV6/NV12/NV18 are fractional A10 slices that vLLM cannot use as a 24 GB device. `Standard_NC24ads_A100_v4` (one 80 GB A100) is the next step up.

## Outputs the Helm overlay depends on

| Output | Value | Used by |
|---|---|---|
| `rwx_storage_class` | `keystone-azurefile-nfs` | `modelCache.storageClassName`, `training.storageClassName` |
| `block_storage_class` | `keystone-managed-premium` | `postgres/redis/qdrant/openbao.storageClassName` |
| `gpu_node_selector` | `{ nodepool = "gpu-coding" }` | `vllm.*.nodeSelector`, `training.nodeSelector` |
| `gpu_toleration` | `nvidia.com/gpu=present:NoSchedule` | already tolerated by `templates/vllm.yaml` and `templates/training-runtime.yaml` (`operator: Exists`) |
| `kubeconfig_command` | `az aks get-credentials --resource-group … --name …` | step 1 after apply |
| `helm_install_command` | the `helm upgrade --install … -f values-client-vpc.yaml -f values-azure.yaml` line | step 2 after apply |

`cluster_endpoint`, `cluster_ca_certificate`, `client_certificate`, `client_key` are marked sensitive (azurerm exposes them through the sensitive `kube_config` block) and feed the root module's kubernetes/helm providers.

## Providers

`hashicorp/azurerm ~> 4.0` (needs `subscription_id` and `features {}` in the root provider block), `hashicorp/kubernetes ~> 2.35`, `hashicorp/helm ~> 2.17`.

## Verification status

- Verified: `tofu fmt -check`, `tofu init -backend=false`, `tofu validate` against azurerm 4.81 / kubernetes 2.38 / helm 2.17 (CI job `iac-validate`).
- **Not verified:** no `tofu apply` has been run against a real Azure subscription. Regional availability of NVadsA10_v5 quota, the NFS share's first mount, and the Cilium data-plane option's rollout on the chosen region have not been exercised. Record the first real apply in `docs/deployment/verification-log.md`.
