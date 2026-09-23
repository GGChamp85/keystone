# aws-eks-gpu

One EKS cluster sized for a Keystone GPU pilot. Used by `environments/aws-pilot` and paired with `helm/keystone/values-aws.yaml`.

## What it creates

| Resource | Detail |
|---|---|
| VPC | `vpc_cidr` (default 10.40.0.0/16), two AZs, two public + two private /20 subnets, one internet gateway, **one** NAT gateway (pilot cost trade-off), subnet tags for EKS load balancers |
| IAM | Cluster role (`AmazonEKSClusterPolicy`), node role (`AmazonEKSWorkerNodePolicy`, `AmazonEKS_CNI_Policy`, `AmazonEC2ContainerRegistryReadOnly`), two Pod Identity roles for the EBS and EFS CSI drivers |
| EKS cluster | `kubernetes_version` (default 1.33), public + private API endpoint, `API_AND_CONFIG_MAP` auth, the creator gets cluster-admin |
| System node group | `system_instance_type` (default m6i.xlarge) × `system_node_count` (default 2), AL2023, label `nodepool=system` |
| GPU node group | `gpu_instance_type` (default **g6.2xlarge** — one L4 24 GB, 8 vCPU, 32 GiB) × `gpu_node_count` (default 1, min 0, max 2), `AL2023_x86_64_NVIDIA` AMI, label `nodepool=gpu-coding`, taint `nvidia.com/gpu=present:NoSchedule`, 200 GB root disk |
| Add-ons | `vpc-cni`, `kube-proxy`, `coredns`, `eks-pod-identity-agent`, `aws-ebs-csi-driver`, `aws-efs-csi-driver` (both CSI drivers bound to their IAM roles through Pod Identity) |
| EFS | One encrypted file system (`elastic` throughput), mount targets in both private subnets, security group allowing NFS from the VPC |
| StorageClass `keystone-efs-rwx` | `efs.csi.aws.com`, dynamic access points — ReadWriteMany for `modelCache` and `training` |
| StorageClass `keystone-gp3` | `ebs.csi.aws.com`, gp3, encrypted, `WaitForFirstConsumer` — Postgres, Redis, Qdrant, OpenBao |
| NVIDIA device plugin | Helm chart `nvidia-device-plugin` 0.17.1 in namespace `nvidia-device-plugin`, pinned to the GPU nodes |

**Drivers:** the EKS AL2023 NVIDIA AMI ships the NVIDIA kernel driver and container toolkit. It does not ship the device plugin, so this module installs it (`install_nvidia_device_plugin = true`).

**Why g6.2xlarge and not g6.xlarge:** both carry one L4, but g6.xlarge has 16 GiB of host RAM — too tight for vLLM to load the ~15 GB of Qwen2.5-Coder-7B weights alongside the kubelet and system pods. Set `gpu_instance_type = "g6.xlarge"` for models under ~4B parameters.

## Outputs the Helm overlay depends on

| Output | Value | Used by |
|---|---|---|
| `rwx_storage_class` | `keystone-efs-rwx` | `modelCache.storageClassName`, `training.storageClassName` |
| `block_storage_class` | `keystone-gp3` | `postgres/redis/qdrant/openbao.storageClassName` |
| `gpu_node_selector` | `{ nodepool = "gpu-coding" }` | `vllm.*.nodeSelector`, `training.nodeSelector` |
| `gpu_toleration` | `nvidia.com/gpu=present:NoSchedule` | already tolerated by `templates/vllm.yaml` and `templates/kuberay.yaml` (`operator: Exists`) |
| `kubeconfig_command` | `aws eks update-kubeconfig --region … --name …` | step 1 after apply |
| `helm_install_command` | the `helm upgrade --install … -f values-client-vpc.yaml -f values-aws.yaml` line | step 2 after apply |

## Providers

`hashicorp/aws ~> 6.0`, `hashicorp/kubernetes ~> 2.35`, `hashicorp/helm ~> 2.17`. The kubernetes/helm providers must be configured by the root module from `cluster_endpoint` / `cluster_ca_certificate` (see `environments/aws-pilot/main.tf`) — a provider cannot be configured inside the module that creates the cluster it points at.

## Verification status

- Verified: `tofu fmt -check`, `tofu init -backend=false`, `tofu validate` against aws 6.66 / kubernetes 2.38 / helm 2.17 (CI job `iac-validate`).
- **Not verified:** no `tofu apply` has been run against a real AWS account. Attribute names were checked against the provider schema by `validate`; runtime behaviour (IAM propagation timing, add-on ordering, EFS mount targets reaching `available` before the StorageClass is used) has not. Record the first real apply in `docs/deployment/verification-log.md`.
