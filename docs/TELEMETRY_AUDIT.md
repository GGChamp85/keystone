# Keystone — Telemetry / Phone-Home Audit

Living checklist for the client's zero-egress requirement. "Disabled"
below means verified against the actual env var/config the component
documents for this — not assumed from a component's general reputation.
Re-run the greps in **How to re-verify** after any dependency bump.

## Disabled, verified wired into the build

| Component | Setting | Where it's set |
|---|---|---|
| vLLM (all 3 model servers) | `VLLM_NO_USAGE_STATS=1` | `docker-compose.yml` (vllm-coding/coding-fallback/reasoning), `helm/keystone/templates/vllm.yaml` |
| Hugging Face Hub (vLLM + embeddings) | `HF_HUB_DISABLE_TELEMETRY=1` | same as above, plus `helm/keystone/templates/configmap.yaml` |
| Hugging Face Hub (all model loads) | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | `helm/keystone/templates/configmap.yaml`, `helm/keystone/templates/vllm.yaml`, `helm/keystone/templates/training-runtime.yaml` (production default; `.env.example`'s dev default is `HF_HUB_OFFLINE=0` so a developer's first `docker compose up` can still pull a model without pre-seeding — flip to `1` for any air-gapped or otherwise network-restricted run) |
| Qdrant | `QDRANT__TELEMETRY_DISABLED=true` | `docker-compose.yml`, `helm/keystone/templates/qdrant.yaml` |
| Grafana | `GF_ANALYTICS_REPORTING_ENABLED=false`, `GF_ANALYTICS_CHECK_FOR_UPDATES=false` | `docker-compose.yml`, `helm/keystone/templates/observability.yaml` |
| Loki | `analytics.reporting_enabled: false` (defaults to `true`, phones home to `stats.grafana.org` — found while building this project's real Loki config, not carried over from Grafana's setting) | `observability/loki/loki-config.yaml`, embedded in `helm/keystone/templates/observability.yaml`'s Loki ConfigMap |
| Weights & Biases (finetuning extra) | `WANDB_MODE=disabled` | `.env.example` |

## Verified inert (present but never activated)

- **`sentry-sdk`** (transitive dep of `fastapi[standard]` via
  `fastapi-cloud-cli`) and **`langsmith`** (transitive dep of
  `langchain-core`): both installed, neither imported or configured
  anywhere in `src/`. Verified with:
  ```bash
  grep -rn "sentry\|SENTRY\|langsmith\|LANGCHAIN_TRACING\|LANGSMITH" src/ pyproject.toml .env.example docker-compose.yml
  # -> no matches
  ```
  Neither SDK sends anything without an explicit DSN/API key, which
  Keystone never sets. No action needed, but re-run this grep after any
  dependency bump — a future FastAPI/LangChain update could change what
  gets pulled in or auto-initialized.

- **`opencode-ai`** (Keystone Agents' interactive CLI): the published npm
  package ships `@opentelemetry/*` tracing packages, but verified against
  the actual CLI source (`sst/opencode`, `packages/opencode/src`) that the
  OTLP exporter is wired purely from the standard
  `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` environment
  variables (`control-plane/workspace.ts`) — there is no hardcoded
  default endpoint, so tracing only leaves the machine if an operator
  explicitly sets those variables. A full-source grep for hardcoded
  telemetry/analytics/stats endpoints (posthog, segment, amplitude,
  mixpanel, or any `*.opencode.ai`-style stats URL) in the CLI package
  found none.

## Known unverified — flag for follow-up, don't assume

- **OpenCode's `share` feature**: not traced through in this pass. If it
  reaches a hosted `opencode.ai` service for session sharing, that's a
  legitimate opt-in feature (not covert phone-home) but would need to be
  disabled/left unused in an air-gapped deployment, since it would simply
  fail closed (no network) rather than silently leak anything — still
  worth an explicit `opencode.json` check before a client engagement that
  enables the CLI.
- **Docker Hub / npm registry / PyPI reachability at *build* time**: not
  in scope for this audit — that's expected and handled by the air-gap
  bundle scripts (`airgap/*.sh`), which run on a build machine with
  internet access precisely so the *deployed* target needs none. This
  audit is about what a running service does at **runtime**.
- **`finetuning` extras** (`torch`, `transformers`, `wandb`, etc.): not
  re-audited here beyond `WANDB_MODE=disabled` above — re-check
  `transformers`' and `datasets`' own telemetry env vars
  (`HF_HUB_DISABLE_TELEMETRY` covers the Hub client; `transformers` itself
  has no separate telemetry channel as of the versions pinned in
  `pyproject.toml`, but re-verify on any version bump) before a client
  engagement that actually runs fine-tuning.

## How to re-verify

```bash
# Python: confirm no telemetry SDK got silently initialized
grep -rn "sentry\|SENTRY\|langsmith\|LANGCHAIN_TRACING\|LANGSMITH\|wandb\.init\|posthog\|segment\.io\|mixpanel" src/

# Container images: confirm every telemetry-disabling env var still exists
grep -n "TELEMETRY\|USAGE_STATS\|ANALYTICS\|WANDB_MODE\|HF_HUB_OFFLINE\|HF_HUB_DISABLE\|reporting_enabled" docker-compose.yml helm/keystone/templates/*.yaml observability/loki/loki-config.yaml

# Full end-to-end proof (the one that actually matters): bring the stack
# up on a host with egress blocked and confirm nothing errors trying to
# reach the internet.
sudo iptables -A OUTPUT -m owner --uid-owner "$(id -u)" -d 0.0.0.0/0 ! -d <internal-subnet> -j REJECT
docker compose up -d
# then check for any connection-refused/DNS-failure noise in logs that
# points at an *external* host, not an internal service name.
```

The `iptables`/`tcpdump` no-outbound-connections check described in the
project plan's Verification section has not been run in this pass (needs
a network-isolated test host, not available in this dev environment) —
the checks above are static/source-level verification, not yet a live
zero-egress test. Run that live test before a client handoff.
