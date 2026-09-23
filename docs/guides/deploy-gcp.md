# Deploy a GPU pilot on Google Cloud (GKE)

One command stands up a GKE Standard cluster with a CPU pool for the platform and one NVIDIA L4 GPU node serving Qwen2.5-Coder-7B-Instruct through vLLM; a second installs Keystone on it. The same Helm chart and OpenTofu layout as every other tier — `infra/opentofu/environments/gcp-pilot` wires `infra/opentofu/modules/gcp-gke-gpu`, and `helm/keystone/values-gcp.yaml` fills in the StorageClass and node-label placeholders that `values-client-vpc.yaml` leaves open.

**Status (2026-09-22): fmt / init / validate / helm template verified in CI. No real `apply` has been run on a GCP project.** See [What is verified](#what-is-verified-and-what-is-not) before budgeting anything on this page.

## What you get

| | |
|---|---|
| Cluster | GKE Standard, zonal (us-central1-a by default), REGULAR release channel, private nodes behind Cloud NAT, public API endpoint, Dataplane V2 (network policy enforced) |
| Platform nodes | 2 × e2-standard-4 (4 vCPU / 16 GiB each) — app, worker, Postgres, Redis, Qdrant, Temporal, OpenBao, observability |
| GPU node | 1 × g2-standard-8 — one NVIDIA L4 (24 GB), 8 vCPU / 32 GiB; autoscaler min 0 |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct`, 16k context, tool calling (`hermes` parser), one GPU |
| Shared model cache | Filestore Basic HDD (ReadWriteMany) through the Filestore CSI driver — `keystone-filestore-rwx` (**1 TiB minimum instance**) |
| Block storage | pd-balanced — `keystone-pd-balanced` |
| GPU scheduling | GKE installs the NVIDIA driver and runs its own device plugin — nothing extra to install |
| Sandbox | gVisor (what GKE Sandbox itself uses) |
| Not installed | An ingress controller, the OpenBao secrets CSI path, the KubeRay operator for multi-GPU training — the overlay disables the first two; see `docs/deployment/KUBERNETES_CLIENT_VPC.md` for adding them |

## What it costs

Approximate **on-demand list prices, us-central1, written 2026-09-22** from the published price lists — not pulled live; confirm with the Google Cloud pricing calculator before committing budget.

| Item | Rate | Per hour |
|---|---|---|
| g2-standard-8 (includes 1 × L4) | $0.8541/h | $0.85 |
| 2 × e2-standard-4 | $0.134/h each | $0.27 |
| GKE Standard cluster management | $0.10/h (one zonal cluster per billing account is free) | $0.10 or $0 |
| Cloud NAT | $0.044/h per gateway + $0.045/GB | $0.04 + data |
| Filestore Basic HDD, **1 TiB minimum** | $0.20/GiB-month | ≈ $0.28 |
| pd-balanced, ~150 GiB across the stateful set | $0.10/GiB-month | ≈ $0.02 |
| **Total, everything running** | | **≈ $1.56/h ≈ $1,140/month** |
| GPU pool scaled to 0 | | ≈ $0.71/h |

Two things worth knowing: the Filestore minimum means the model cache costs ~$205/month even holding 15 GB of weights (a `zonal`-tier instance has the same 1 TiB floor; there is no smaller RWX option with the CSI driver), and the L4's price is bundled into the g2 machine type rather than billed separately.

## The five commands

Prerequisites: OpenTofu 1.6+ (`brew install opentofu`) or Terraform 1.6+, `gcloud` logged in (`gcloud auth login` and `gcloud auth application-default login`) to a project with billing and Owner/Editor rights, the `gke-gcloud-auth-plugin` component, `kubectl`, `helm`.

```bash
# 1. Set the project (the only value without a default) and sizes; the file is git-ignored
cp infra/opentofu/environments/gcp-pilot/terraform.tfvars.example infra/opentofu/environments/gcp-pilot/terraform.tfvars
#    project_id = "$(gcloud config get-value project)"

# 2. See exactly what will be created — changes nothing
keystone deploy cloud --cloud gcp --env pilot --plan

# 3. Create it (asks for the usual "yes"; ~10 minutes). Prints the kubeconfig and Helm commands at the end.
keystone deploy cloud --cloud gcp --env pilot --apply

# 4. Point kubectl at it, then install Keystone with the GCP overlay
gcloud container clusters get-credentials keystone-pilot --location us-central1-a --project "$(gcloud config get-value project)"
keystone deploy cloud --cloud gcp --helm-install

# 5. Reach the API (no ingress controller on the pilot), then create a key and check it
kubectl -n keystone port-forward svc/keystone-app 8080:8080 &
KEYSTONE_INFERENCE_URL=http://localhost:8080 keystone doctor
```

Tear everything down — billing stops with the last resource (the Filestore instance is deleted with its PVC):

```bash
keystone deploy cloud --cloud gcp --env pilot --destroy
```

`keystone deploy cloud` is a thin wrapper: it prints and streams `tofu -chdir=infra/opentofu/environments/gcp-pilot init|plan|apply|destroy` and `helm upgrade --install keystone helm/keystone -f values-client-vpc.yaml -f values-gcp.yaml`. Run those by hand if you prefer; the argv is asserted in `tests/test_cli_ops.py`.

## What is verified, and what is not

Verified, in CI on every push (`.github/workflows/ci.yml`, jobs `iac-validate` and `helm`):

- `tofu fmt -check`, `tofu init -backend=false`, `tofu validate` for `modules/gcp-gke-gpu` and `environments/gcp-pilot` against google 6.50 / kubernetes 2.38 — every attribute name exists in the provider schema.
- `helm lint` and `helm template` with `values-client-vpc.yaml` + `values-gcp.yaml`, plus a check that the render names the module's StorageClasses and no longer carries the client-VPC `sandbox-capable` placeholder.
- `tests/test_cli_ops.py`: the exact argv `keystone deploy cloud` runs, and a real `init -backend=false` + `validate` through that argv.

**Not verified — nobody has run `apply` on GCP yet:**

- That the cluster and both pools come up, and that the three APIs are enabled early enough for the network resources that follow.
- That the g2-standard-8 node exposes `nvidia.com/gpu: 1` through GKE's device plugin and that vLLM loads the 7B model on it.
- That the Filestore instance (several minutes to create) is ready before the vLLM pod's PVC needs it — a first-start delay is expected, not a failure.
- Quotas: a fresh project has 0 `NVIDIA_L4_GPUS` quota — request it in the chosen zone before step 3.
- Anything about cost beyond the list prices above.

The first real apply, with its date, zone, timings and what broke, goes in `docs/deployment/verification-log.md`.
