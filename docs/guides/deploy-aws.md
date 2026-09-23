# Deploy a GPU pilot on AWS (EKS)

One command stands up an EKS cluster with a CPU pool for the platform and one NVIDIA L4 GPU node serving Qwen2.5-Coder-7B-Instruct through vLLM; a second installs Keystone on it. The same Helm chart and OpenTofu layout as every other tier — `infra/opentofu/environments/aws-pilot` wires `infra/opentofu/modules/aws-eks-gpu`, and `helm/keystone/values-aws.yaml` fills in the StorageClass and node-label placeholders that `values-client-vpc.yaml` leaves open.

**Status (2026-09-22): fmt / init / validate / helm template verified in CI. No real `apply` has been run on an AWS account.** See [What is verified](#what-is-verified-and-what-is-not) before budgeting anything on this page.

## What you get

| | |
|---|---|
| Cluster | EKS 1.33, private nodes in a new VPC, one NAT gateway, public API endpoint |
| Platform nodes | 2 × m6i.xlarge (4 vCPU / 16 GiB each) — app, worker, Postgres, Redis, Qdrant, Temporal, OpenBao, observability |
| GPU node | 1 × g6.2xlarge — one NVIDIA L4 (24 GB), 8 vCPU / 32 GiB; scales to 0 with a cluster autoscaler |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct`, 16k context, tool calling (`hermes` parser), one GPU |
| Shared model cache | EFS (ReadWriteMany) through the EFS CSI driver — `keystone-efs-rwx` |
| Block storage | gp3, encrypted — `keystone-gp3` |
| GPU scheduling | NVIDIA device plugin 0.17.1 on the EKS NVIDIA AMI (driver pre-installed) |
| Sandbox | gVisor (EKS managed nodes have no `/dev/kvm`) |
| Not installed | An ingress controller, the OpenBao secrets CSI path, the KubeRay operator for multi-GPU training — the overlay disables the first two; see `docs/deployment/KUBERNETES_CLIENT_VPC.md` for adding them |

## What it costs

Approximate **on-demand list prices, us-east-1, written 2026-09-22** from the published price lists — not pulled live; confirm with the AWS pricing calculator before committing budget.

| Item | Rate | Per hour |
|---|---|---|
| g6.2xlarge (1 × L4) | $0.9776/h | $0.98 |
| 2 × m6i.xlarge | $0.192/h each | $0.38 |
| EKS control plane | $0.10/h | $0.10 |
| NAT gateway | $0.045/h + $0.045/GB processed | $0.05 + data |
| EFS Standard, ~20 GB of weights | $0.30/GB-month | ≈ $0.01 |
| gp3, ~150 GiB across the stateful set | $0.08/GB-month | ≈ $0.02 |
| **Total, everything running** | | **≈ $1.55/h ≈ $1,130/month** |
| GPU pool scaled to 0 | | ≈ $0.57/h |

The single biggest lever is the GPU node: `gpu_node_min = 0` plus a cluster autoscaler (not installed by the module) lets it disappear when no vLLM pod is scheduled.

## The five commands

Prerequisites: OpenTofu 1.6+ (`brew install opentofu`) or Terraform 1.6+, the AWS CLI logged in to an account with permission to create VPCs, IAM roles and EKS clusters, `kubectl`, `helm`.

```bash
# 1. Pick region / sizes (every value has a default; the file is git-ignored)
cp infra/opentofu/environments/aws-pilot/terraform.tfvars.example infra/opentofu/environments/aws-pilot/terraform.tfvars

# 2. See exactly what will be created — changes nothing
keystone deploy cloud --cloud aws --env pilot --plan

# 3. Create it (asks for the usual "yes"; ~15 minutes). Prints the kubeconfig and Helm commands at the end.
keystone deploy cloud --cloud aws --env pilot --apply

# 4. Point kubectl at it, then install Keystone with the AWS overlay
aws eks update-kubeconfig --region us-east-1 --name keystone-pilot
keystone deploy cloud --cloud aws --helm-install

# 5. Reach the API (no ingress controller on the pilot), then create a key and check it
kubectl -n keystone port-forward svc/keystone-app 8080:8080 &
KEYSTONE_INFERENCE_URL=http://localhost:8080 keystone doctor
```

Tear everything down — billing stops with the last resource:

```bash
keystone deploy cloud --cloud aws --env pilot --destroy
```

`keystone deploy cloud` is a thin wrapper: it prints and streams `tofu -chdir=infra/opentofu/environments/aws-pilot init|plan|apply|destroy` and `helm upgrade --install keystone helm/keystone -f values-client-vpc.yaml -f values-aws.yaml`. Run those by hand if you prefer; the argv is asserted in `tests/test_cli_ops.py`.

## What is verified, and what is not

Verified, in CI on every push (`.github/workflows/ci.yml`, jobs `iac-validate` and `helm`):

- `tofu fmt -check`, `tofu init -backend=false`, `tofu validate` for `modules/aws-eks-gpu` and `environments/aws-pilot` against aws 6.66 / kubernetes 2.38 / helm 2.17 — every attribute name exists in the provider schema.
- `helm lint` and `helm template` with `values-client-vpc.yaml` + `values-aws.yaml`, plus a check that the render names the module's StorageClasses and no longer carries the client-VPC `sandbox-capable` placeholder.
- `tests/test_cli_ops.py`: the exact argv `keystone deploy cloud` runs, and a real `init -backend=false` + `validate` through that argv.

**Not verified — nobody has run `apply` on AWS yet:**

- That the cluster, node groups, add-ons and EFS come up in the order the dependency graph says, and how long it takes.
- That a g6.2xlarge node with the AL2023 NVIDIA AMI exposes `nvidia.com/gpu: 1` through the device plugin and that vLLM loads the 7B model on it.
- Quotas: a fresh account often has a 0 vCPU quota for G instances — request it before step 3.
- Anything about cost beyond the list prices above.

The first real apply, with its date, region, timings and what broke, goes in `docs/deployment/verification-log.md`.
