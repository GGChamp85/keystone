# Deployment verification log

A plain record of what each deployment path has actually been run against, so nobody mistakes "it validates" for "it works". Add a dated entry every time a tier is exercised for real; never edit an old entry to make it sound better.

## GPU pilots on AWS / Azure / GCP (`infra/opentofu/modules/{aws-eks-gpu,azure-aks-gpu,gcp-gke-gpu}`, `environments/{aws,azure,gcp}-pilot`, `helm/keystone/values-{aws,azure,gcp}.yaml`, `keystone deploy cloud`)

### 2026-09-22 — static verification only

Verified, locally and in CI (`.github/workflows/ci.yml`, jobs `iac-validate` and `helm`):

| Check | Tool / version | Result |
|---|---|---|
| `tofu fmt -check -recursive infra/opentofu` | OpenTofu 1.12.6 | clean |
| `tofu init -backend=false` + `tofu validate`, `modules/aws-eks-gpu` and `environments/aws-pilot` | aws 6.66.0, kubernetes 2.38.0, helm 2.17.0 | valid |
| `tofu init -backend=false` + `tofu validate`, `modules/azure-aks-gpu` and `environments/azure-pilot` | azurerm 4.81.0, kubernetes 2.38.0, helm 2.17.0 | valid |
| `tofu init -backend=false` + `tofu validate`, `modules/gcp-gke-gpu` and `environments/gcp-pilot` | google 6.50.0, kubernetes 2.38.0 | valid |
| `helm lint` + `helm template keystone helm/keystone -f values-client-vpc.yaml -f values-<cloud>.yaml`, all three clouds | Helm 4.2.3 locally, `azure/setup-helm@v4` in CI | renders; the render names the module's StorageClasses (`keystone-efs-rwx`/`keystone-gp3`, `keystone-azurefile-nfs`/`keystone-managed-premium`, `keystone-filestore-rwx`/`keystone-pd-balanced`), the coding role is a single-GPU Qwen2.5-Coder-7B StatefulSet, and the client-VPC `keystone.io/sandbox-capable` placeholder is gone |
| `python -m pytest tests/test_cli_ops.py` | pytest | 22 passed — including a real `init -backend=false` + `validate` through the exact argv `keystone deploy cloud` builds |
| `keystone deploy cloud --cloud aws --plan` with an empty PATH | CLI | refuses with the install instruction instead of a traceback |

**No real `apply` has been run on any cloud.** Nothing on this page says a cluster came up, a GPU was scheduled, a model was served, or that the cost figures in `docs/guides/deploy-*.md` matched a bill. `tofu validate` proves every attribute exists in the provider schema; it does not prove IAM propagation timing, add-on ordering, quota availability, driver/device-plugin behaviour on the chosen instance type, or the RWX filesystem's first mount.

Known gap in the static check itself: `environments/runpod-test` is not in the `iac-validate` job because its provider ships a malformed data-source schema that breaks `validate` before any HCL is read (see the comment in `modules/runpod-gpu-pool/main.tf`).

### How to record the first real apply

Copy the block below under this section, fill it in the same day, and commit it with the fixes it produced. One block per cloud per attempt — a failed attempt is worth recording too.

```markdown
### <YYYY-MM-DD> — <aws|azure|gcp> pilot, first real apply

- Who / account: <initials>, <account or subscription or project id — no secrets>
- Region / zone: <…>; sizes: <gpu instance>, <system instance> × <n>
- Versions: OpenTofu <x.y.z>; providers <aws|azurerm|google> <x.y.z>, kubernetes <x.y.z>, helm <x.y.z>; Helm <x.y.z>
- `keystone deploy cloud --cloud <cloud> --env pilot --apply`: <succeeded in NN min | failed at <resource> with <one-line error>>
- Quota requests needed first: <none | which family, how long it took>
- `kubectl get nodes -L nodepool` shows the GPU node with `nvidia.com/gpu: 1` allocatable: <yes | no — why>
- `keystone deploy cloud --cloud <cloud> --helm-install`: <all pods Ready in NN min | which pod did not, and why>
- Model cache PVC bound on `<rwx class>`: <yes in NN s | no — why>; first vLLM start (model download + load): <NN min>
- `keystone doctor` against the port-forwarded API: <OK | which check failed>
- One real completion through `/v1/chat/completions`: <tokens/s observed | not attempted>
- `--destroy`: <clean in NN min | what was left behind and how it was removed>
- Actual cost for the session: <$X for N hours from the billing console — or "not yet visible">
- Fixes committed as a result: <commit hashes>
```

Then update the "Status" line at the top of the matching `docs/guides/deploy-<cloud>.md` and the module README's "Verification status" section to point at the entry.
