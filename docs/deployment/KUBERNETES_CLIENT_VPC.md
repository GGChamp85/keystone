# Kubernetes deployment (client VPC / production tier)

## Prerequisites

- A Kubernetes cluster inside the client's own VPC (1.28+). Cloud provider
  is intentionally unconfirmed at the chart level — see
  `helm/keystone/values-client-vpc.yaml`'s `TODO` markers for the
  StorageClass/nodeSelector values that need filling in once known.
- An NVIDIA GPU device plugin installed (`nvidia.com/gpu` resources must be
  schedulable) if `vllm.*.enabled=true` or `training.enabled=true`.
- An RWX-capable StorageClass (NFS/EFS/Filestore or equivalent) for
  `modelCache` and `training.storage` — model weights and training data are
  shared across multiple GPU nodes, which plain block storage can't do.
- (Optional, for `training.enabled=true`) The Kubeflow Trainer controller
  installed cluster-wide — **not** part of this chart, since it's shared
  cluster infrastructure, not something a single tenant/release should own:

  ```bash
  # Real, current install command for the project's v2.3.0 release —
  # verify against github.com/kubeflow/training-operator for newer tags.
  kubectl apply --server-side -k \
    "github.com/kubeflow/training-operator/manifests/base?ref=v2.3.0"
  ```

  This chart's `templates/training-runtime.yaml` only defines Keystone's
  own `ClusterTrainingRuntime` (our fine-tuning image + GPU placement) on
  top of that shared controller — it does not install the controller
  itself.
- (Optional, for any `vllm.*.autoscaling.enabled=true`) [KEDA](https://keda.sh)
  installed cluster-wide — same shared-infrastructure reasoning as the
  Kubeflow Trainer controller above, so this chart's `ScaledObject`
  resources (`templates/vllm.yaml`) assume it's already there rather than
  installing it themselves:

  ```bash
  helm repo add kedacore https://kedacore.github.io/charts
  helm install keda kedacore/keda --namespace keda --create-namespace
  ```

## Install

```bash
helm install keystone helm/keystone \
  --namespace keystone --create-namespace \
  -f helm/keystone/values-client-vpc.yaml \
  --set global.imageRegistry=registry.internal.<client-domain> \
  --wait --timeout 10m
```

## Verification

```bash
kubectl get pods -n keystone
kubectl logs -n keystone -l app.kubernetes.io/component=app --tail=50
curl -k https://<ingress-host>/health
```

## Production posture

Four things the chart does for a production install that the dev tier does not need:

- **Secrets from OpenBao, not a Kubernetes Secret at rest** (`secrets.useOpenBao=true`): `templates/openbao-secretproviderclass.yaml` declares a Secrets Store CSI `SecretProviderClass` (Vault provider; OpenBao speaks the same API) reading `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `QDRANT_API_KEY`, `KEYSTONE_ROOT_ADMIN_TOKEN`, `VS_SECRET_KEY` and `GRAFANA_ADMIN_PASSWORD` from `secrets.openbao.secretsPath` and syncing them into `<release>-secrets` — the same Secret name every template already references, so nothing else changes. The app and worker pods mount the CSI volume (that is what triggers the sync). Cluster prerequisites, installed once: `secrets-store-csi-driver` with `syncSecret.enabled=true`, the Vault CSI provider, and a Kubernetes auth role in OpenBao (`secrets.openbao.role`) bound to the chart's ServiceAccount. `VS_SECRET_KEY` must be a stable value: API keys are hashed with it.
- **Migrations as a release hook** (`migrations.enabled`, default on): `templates/migrate-job.yaml` runs `alembic upgrade head` once per install/upgrade as a `pre-install,pre-upgrade` hook Job, waiting for Postgres first. A failed migration fails the release before any new pod takes traffic. The app Deployment rolls with `maxUnavailable: 0`, a `preStop` pause and a 60 s grace period so streamed completions finish.
- **Nightly database backups** (`postgres.backup.enabled`, on in `values-client-vpc.yaml`): a CronJob writes `pg_dump --format=custom` into its own PVC and prunes dumps older than `retentionDays`. Copy that volume off-cluster with your organisation's tooling. **Prove restores work**: `scripts/restore_drill.sh --from-url <db url>` restores a dump into a throwaway Postgres and compares Alembic revision and row counts; CI runs it on every push against its Postgres service.
- **Egress allowlisting** (`networkPolicy.egress.enabled`, on in `values-client-vpc.yaml`): default-deny egress for every pod, then DNS, in-cluster peers, and the CIDRs you list in `networkPolicy.egress.allowedCIDRs` (the git host, package mirrors, a hosted model endpoint). vLLM gets internet only with `vllmInternet: true`; an air-gapped site keeps it false and serves weights from the model cache.

All four render in `helm lint`/`helm template` in CI with every option enabled; the OpenBao sync and the hook ordering have not yet been exercised on a real cluster with the CSI driver installed — the verification table below says so.

## What's been actually verified vs. what's real-but-unexercised

Verified end-to-end against a real (if small: `kind`) Kubernetes cluster
during development — not just `helm lint`:

- Full non-GPU stack deploys and reaches Ready: app, worker, postgres,
  redis, qdrant, Temporal, Temporal UI, OpenBao, sandbox-daemon
  (gVisor backend), Prometheus, Grafana, Alertmanager, Loki, Promtail.
- The Temporal `DB` driver value bug, the image-reference Helm helper
  losing `.Values.global` context, a Go-YAML parser quirk in OpenBao's
  readinessProbe, and a `runAsNonRoot` failure from a named (non-numeric)
  container USER were all found and fixed this way.
- Prometheus/Grafana (`templates/observability.yaml`): deployed on a real
  `kind` cluster, confirmed `keystone-*-prometheus-0` and
  `keystone-*-grafana-0` both reach `1/1 Running`, Prometheus reports its
  own scrape target `up`, and all three dashboards
  (`helm/keystone/dashboards/*.json`, embedded into a ConfigMap via
  `.Files.Glob`) are listed by Grafana's own API after fetching the
  cluster-generated `GRAFANA_ADMIN_PASSWORD` secret. Every PromQL
  expression in all three dashboards was also syntax-checked against a
  live Prometheus instance (29/29 valid) and every vLLM metric name used
  was verified against the real `vllm/engine/metrics.py` source for the
  pinned v0.6.3 release, not assumed.
- **Real `NetworkPolicy` bug found and fixed this way**: the chart's
  default-deny-all-ingress policy (`templates/networkpolicies.yaml`) had
  no exception for the observability tier at all — Prometheus's scrape
  targets and Loki's push endpoint were both silently unreachable
  (`context deadline exceeded`, not a clean refusal — NetworkPolicy drops
  packets rather than rejecting them, which is why this looked at first
  like a slow/unready service rather than a firewall rule). Confirmed
  kind's default CNI (kindnet) does enforce `NetworkPolicy` with a real
  positive/negative control (removed the fix rule, watched the same
  connection hang again; restored it, watched it succeed) before
  concluding this was the actual cause rather than a coincidence. Fixed
  by adding `allow-prometheus-to-scrape-targets` and
  `allow-observability-internal` rules. **This also means `app`'s and
  `sandbox-daemon`'s scrape targets showing `down` in earlier passes of
  this project may have been partly this bug, not purely the
  no-local-image limitation noted below** — re-verify once a real image
  is built.
- Alertmanager/Loki/Promtail: all three reach `1/1 Running` (Promtail as
  a DaemonSet using `kubernetes_sd_configs` + a `hostPath` mount of
  `/var/log/pods` — **not** `docker_sd_configs` against a Docker socket,
  because kind nodes run containerd with no `/var/run/docker.sock` at
  all, confirmed directly on the test cluster). Verified a real Postgres
  pod's log line reached Loki through the full path (Promtail discovery →
  NetworkPolicy → Loki ingest → Loki query API), and that a real
  in-cluster Grafana pod's Prometheus and Loki datasource health checks
  both return healthy under NetworkPolicy enforcement. Grafana's
  "Alertmanager" datasource type returns `plugin.unavailable` on this
  Grafana version's health check — a Grafana-side gap, not a config
  error; alerting itself doesn't depend on it (see
  `docs/LICENSES_AND_COMPLIANCE.md`).
- Prometheus's alert rules (`observability/prometheus/alerts.yml`, also
  embedded in the Helm chart) load with zero errors (6/6 rules `ok`) and
  Prometheus correctly discovers Alertmanager via in-cluster DNS. A real
  alert (`KeystoneTargetDown`, since the un-built app/sandbox-daemon
  images have no running pod to scrape) was watched transition from
  pending to firing after its real 2-minute `for:` window and confirmed
  received by Alertmanager — the whole rule → fire → route → receive
  pipeline, not just "the YAML parsed."

Real but **not** exercised against an actual cluster in this pass (no GPU
hardware or multi-node cluster available in the dev environment this was
built in) — the manifests are believed correct (checked against the real
upstream schemas, not guessed) but should be smoke-tested before a client
handoff:

- `vllm.*.enabled=true` StatefulSets (need real GPU nodes to schedule),
  including the newer `nodeCount > 1` multi-node Ray bootstrap and the
  `autoscaling.enabled=true` KEDA `ScaledObject` — all three
  (single-node, multi-node, autoscaling) render correctly through `helm
  lint`/`helm template` in CI, but none has been scheduled onto a real
  GPU node or multi-node cluster.
- `training.enabled=true` (`ClusterTrainingRuntime` + submitting a real
  `TrainJob` — needs both GPU nodes and the Trainer controller installed).
- Multi-node NCCL/Infiniband networking for the training node pool.
