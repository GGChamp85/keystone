# gcp-gke-gpu

One GKE Standard cluster sized for a Keystone GPU pilot. Used by `environments/gcp-pilot` and paired with `helm/keystone/values-gcp.yaml`.

## What it creates

| Resource | Detail |
|---|---|
| APIs | `compute`, `container`, `file` enabled on the project (left enabled on destroy) |
| Network | Custom VPC, one subnet in `region` (nodes 10.42.0.0/20, secondary ranges `pods` 10.44.0.0/14 and `services` 10.48.0.0/20), Private Google Access, Cloud Router + Cloud NAT so private nodes can pull images and weights |
| Node service account | `<cluster_name>-nodes` with log writer, metric writer, monitoring viewer, resource-metadata writer and Artifact Registry reader |
| GKE cluster | `location` (default zone us-central1-a → zonal; pass a region for regional), VPC-native, **Dataplane V2** (enforces the chart's `NetworkPolicy` objects), private nodes with a public API endpoint, `release_channel` REGULAR, workload identity, Persistent Disk + Filestore CSI drivers enabled, default pool removed, `deletion_protection` off (pilot) |
| System pool | `system_machine_type` (default e2-standard-4) × `system_node_count` (default 2, autoscaler 2–4), shielded VMs, label `nodepool=system` |
| GPU pool | `gpu_machine_type` (default **g2-standard-8** — one L4 24 GB, 8 vCPU, 32 GiB) × `gpu_node_count` (default 1, min 0, max 2), `nvidia-l4` × `gpu_accelerator_count` (1), GKE-managed driver install (`LATEST`), label `nodepool=gpu-coding`, taint `nvidia.com/gpu=present:NoSchedule`, 200 GB boot disk |
| StorageClass `keystone-filestore-rwx` | `filestore.csi.storage.gke.io`, `tier=standard` (Basic HDD), `network=<this VPC>` — ReadWriteMany for `modelCache` and `training`; one Filestore instance per PVC, rounded up to the tier minimum (**1 TiB**) |
| StorageClass `keystone-pd-balanced` | `pd.csi.storage.gke.io`, `pd-balanced`, `WaitForFirstConsumer` — Postgres, Redis, Qdrant, OpenBao |

**Drivers:** with `gpu_driver_installation_config` set, GKE installs the NVIDIA driver on each GPU node **and runs its own device plugin** (`nvidia-gpu-device-plugin` in `kube-system`). Unlike the AWS and Azure modules, this one installs no device plugin and needs no helm provider.

**Why g2-standard-8:** G2 machine types have a fixed GPU count; g2-standard-8 is the smallest with a full L4 and 32 GiB of host RAM. `gpu_accelerator_count` must match the machine type (g2-standard-24 = 2 L4, -48 = 4, -96 = 8).

## Outputs the Helm overlay depends on

| Output | Value | Used by |
|---|---|---|
| `rwx_storage_class` | `keystone-filestore-rwx` | `modelCache.storageClassName`, `training.storageClassName` |
| `block_storage_class` | `keystone-pd-balanced` | `postgres/redis/qdrant/openbao.storageClassName` |
| `gpu_node_selector` | `{ nodepool = "gpu-coding" }` | `vllm.*.nodeSelector`, `training.nodeSelector` |
| `gpu_toleration` | `nvidia.com/gpu=present:NoSchedule` | already tolerated by `templates/vllm.yaml` and `templates/training-runtime.yaml` (`operator: Exists`) |
| `kubeconfig_command` | `gcloud container clusters get-credentials … --location … --project …` | step 1 after apply (needs the `gke-gcloud-auth-plugin` gcloud component) |
| `helm_install_command` | the `helm upgrade --install … -f values-client-vpc.yaml -f values-gcp.yaml` line | step 2 after apply |

## Providers

`hashicorp/google ~> 6.0`, `hashicorp/kubernetes ~> 2.35`. The root module authenticates the kubernetes provider with `data.google_client_config.default.access_token` (see `environments/gcp-pilot/main.tf`), so no kubeconfig is needed during apply.

## Verification status

- Verified: `tofu fmt -check`, `tofu init -backend=false`, `tofu validate` against google 6.50 / kubernetes 2.38 (CI job `iac-validate`).
- **Not verified:** no `tofu apply` has been run against a real GCP project. L4 quota in the chosen zone, the Filestore instance's creation time (several minutes) before the model-cache PVC binds, and the API-enablement ordering have not been exercised. Record the first real apply in `docs/deployment/verification-log.md`.
