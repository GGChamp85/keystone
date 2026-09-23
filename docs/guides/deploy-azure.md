# Deploy a GPU pilot on Azure (AKS)

One command stands up an AKS cluster with a CPU pool for the platform and one NVIDIA A10 GPU node serving Qwen2.5-Coder-7B-Instruct through vLLM; a second installs Keystone on it. The same Helm chart and OpenTofu layout as every other tier — `infra/opentofu/environments/azure-pilot` wires `infra/opentofu/modules/azure-aks-gpu`, and `helm/keystone/values-azure.yaml` fills in the StorageClass and node-label placeholders that `values-client-vpc.yaml` leaves open.

**Status (2026-09-22): fmt / init / validate / helm template verified in CI. No real `apply` has been run on an Azure subscription.** See [What is verified](#what-is-verified-and-what-is-not) before budgeting anything on this page.

## What you get

| | |
|---|---|
| Cluster | AKS (region default), Free tier control plane, Azure CNI overlay with the Cilium data plane and network policy, public API endpoint |
| Platform nodes | 2 × Standard_D4s_v5 (4 vCPU / 16 GiB each) — app, worker, Postgres, Redis, Qdrant, Temporal, OpenBao, observability |
| GPU node | 1 × Standard_NV36ads_A10_v5 — one full NVIDIA A10 (24 GB), 36 vCPU / 440 GiB; autoscaler min 0 |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct`, 16k context, tool calling (`hermes` parser), one GPU |
| Shared model cache | Azure Files Premium over NFS 4.1 (ReadWriteMany) — `keystone-azurefile-nfs` |
| Block storage | Premium SSD managed disks — `keystone-managed-premium` |
| GPU scheduling | NVIDIA device plugin 0.17.1; AKS installs the driver itself on N-series sizes |
| Sandbox | gVisor (no nested virtualisation on these sizes) |
| Not installed | An ingress controller, the OpenBao secrets CSI path, the KubeRay operator for multi-GPU training — the overlay disables the first two; see `docs/deployment/KUBERNETES_CLIENT_VPC.md` for adding them |

## What it costs

Approximate **pay-as-you-go list prices, East US, written 2026-09-22** from the published price lists — not pulled live; confirm with the Azure pricing calculator before committing budget.

| Item | Rate | Per hour |
|---|---|---|
| Standard_NV36ads_A10_v5 (1 × A10) | $3.20/h | $3.20 |
| 2 × Standard_D4s_v5 | $0.192/h each | $0.38 |
| AKS control plane (Free tier) | $0 | $0.00 |
| Azure Files Premium, 100 GiB provisioned | ≈ $0.16/GiB-month | ≈ $0.02 |
| Premium SSD, ~150 GiB across the stateful set | ≈ $0.12/GiB-month | ≈ $0.03 |
| Outbound data | first 100 GB/month free, then per GB | data |
| **Total, everything running** | | **≈ $3.65/h ≈ $2,660/month** |
| GPU pool scaled to 0 | | ≈ $0.45/h |

The A10 VM is the whole bill. The NVadsA10_v5 family is the cheapest Azure size with a full 24 GB GPU; the fractional NV6/NV12/NV18 sizes are cheaper but expose only a slice, which vLLM cannot use. `gpu_node_min = 0` lets the AKS autoscaler remove it when no vLLM pod is scheduled.

## The five commands

Prerequisites: OpenTofu 1.6+ (`brew install opentofu`) or Terraform 1.6+, the Azure CLI logged in (`az login`) to a subscription with Contributor rights and NVadsA10_v5 quota in the region, `kubectl`, `helm`.

```bash
# 1. Set the subscription (the only value without a default) and sizes; the file is git-ignored
cp infra/opentofu/environments/azure-pilot/terraform.tfvars.example infra/opentofu/environments/azure-pilot/terraform.tfvars
#    subscription_id = "$(az account show --query id -o tsv)"

# 2. See exactly what will be created — changes nothing
keystone deploy cloud --cloud azure --env pilot --plan

# 3. Create it (asks for the usual "yes"; ~10 minutes). Prints the kubeconfig and Helm commands at the end.
keystone deploy cloud --cloud azure --env pilot --apply

# 4. Point kubectl at it, then install Keystone with the Azure overlay
az aks get-credentials --resource-group keystone-pilot-rg --name keystone-pilot --overwrite-existing
keystone deploy cloud --cloud azure --helm-install

# 5. Reach the API (no ingress controller on the pilot), then create a key and check it
kubectl -n keystone port-forward svc/keystone-app 8080:8080 &
KEYSTONE_INFERENCE_URL=http://localhost:8080 keystone doctor
```

Tear everything down — the resource group and everything in it:

```bash
keystone deploy cloud --cloud azure --env pilot --destroy
```

`keystone deploy cloud` is a thin wrapper: it prints and streams `tofu -chdir=infra/opentofu/environments/azure-pilot init|plan|apply|destroy` and `helm upgrade --install keystone helm/keystone -f values-client-vpc.yaml -f values-azure.yaml`. Run those by hand if you prefer; the argv is asserted in `tests/test_cli_ops.py`.

## What is verified, and what is not

Verified, in CI on every push (`.github/workflows/ci.yml`, jobs `iac-validate` and `helm`):

- `tofu fmt -check`, `tofu init -backend=false`, `tofu validate` for `modules/azure-aks-gpu` and `environments/azure-pilot` against azurerm 4.81 / kubernetes 2.38 / helm 2.17 — every attribute name exists in the provider schema.
- `helm lint` and `helm template` with `values-client-vpc.yaml` + `values-azure.yaml`, plus a check that the render names the module's StorageClasses and no longer carries the client-VPC `sandbox-capable` placeholder.
- `tests/test_cli_ops.py`: the exact argv `keystone deploy cloud` runs, and a real `init -backend=false` + `validate` through that argv.

**Not verified — nobody has run `apply` on Azure yet:**

- That the cluster and both pools come up, and that the Cilium data plane and NFS Azure Files options are accepted in the chosen region.
- That the NV36ads_A10_v5 node exposes `nvidia.com/gpu: 1` through the device plugin (the A10 needs the GRID driver AKS installs on NV sizes) and that vLLM loads the 7B model on it.
- Quotas: a fresh subscription usually has 0 vCPU quota for the NVadsA10v5 family — request it before step 3.
- Anything about cost beyond the list prices above.

The first real apply, with its date, region, timings and what broke, goes in `docs/deployment/verification-log.md`.
