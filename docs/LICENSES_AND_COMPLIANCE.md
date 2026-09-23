# Keystone — Open-Source License Inventory & Compliance Notes

For the client's legal review. Every entry below was checked against a
real source (a package's installed `LICENSE`/metadata, or the upstream
project's GitHub-reported SPDX license) during this pass — not assumed
from memory — and real issues were found and fixed as a result of that
checking, not just documented. See **Findings that changed the build**
below before anything else. (The NetworkPolicy bug found while deploying
the Loki/Alertmanager/Promtail stack to a real cluster — see
`docs/deployment/KUBERNETES_CLIENT_VPC.md` — isn't a licensing finding,
but was caught the same way: by actually running the thing rather than
trusting the config.)

## Findings that changed the build

### 1. Redis → Valkey (fixed)

`redis:7-alpine` on Docker Hub currently resolves to **Redis 7.4.11**.
Redis Inc. relicensed Redis away from BSD-3-Clause to a dual
**RSALv2/SSPLv1** license as of the 7.4 release (announced March 2024) —
neither license is OSI-approved open source, and SSPL in particular is
widely treated as incompatible with "open source" for exactly the kind of
service Keystone is. This is the same category of problem that already
motivated this project's Terraform → OpenTofu and Vault → OpenBao swaps,
just not caught for Redis at the time.

**Fixed**: switched to **Valkey** (`valkey/valkey:7.2-alpine`), the Linux
Foundation's BSD-3-Clause fork of pre-relicense Redis, created by the
original Redis maintainers specifically in response to this change.
Verified as a real drop-in replacement, not just license-swapped: Valkey
speaks the same RESP wire protocol, so the existing `redis` Python client
(`REDIS_URL=redis://...`) connects, authenticates, and does GET/SET/PING
against it with zero application code changes — confirmed with a live
container during this pass. Updated in `docker-compose.yml`,
`helm/keystone/templates/redis.yaml`, and `airgap/build_image_bundle.sh`.

### 2. Grafana is AGPL-3.0, not a permissive license (documented, not swapped)

Verified against `grafana/grafana`'s repository license (GitHub API,
`license.spdx_id`): **AGPL-3.0**, not Apache-2.0/MIT as the original
project plan assumed for the observability stack. AGPL-3.0 is a real,
OSI-approved open-source license (so it doesn't fail the client's
"fully open source" requirement the way Redis's RSALv2/SSPLv1 does), but
it carries network copyleft: if Keystone ever **modifies Grafana's own
source** and offers access to that modified version over a network,
those modifications must be made available under AGPL too.

Keystone's actual usage — running the stock `grafana/grafana:11.2.0`
image, configured only via its supported environment variables and
file-based provisioning (`observability/grafana/provisioning/`,
`helm/keystone/dashboards/*.json`) — does **not** modify Grafana's source,
so this triggers no obligation today. Flagged here so it's a known,
deliberate fact for legal review rather than something discovered later:
if a future engagement ever forks/patches Grafana itself, re-check this.

### 3. Codestral-22B-v0.1 was considered and rejected — non-production license (removed)

The original plan included Codestral 22B as a "fast coding" model role.
Checked against the real license (`mistral.ai/licences/MNPL-0.1.md`,
linked from the model's own HF `README.md` `license_link` field, fetched
and read in full during this pass, not assumed from the `license: other`
tag alone): **Mistral AI Non-Production License (MNPL)**. Section 3.2 is
explicit and leaves no ambiguity:

> You shall only use the Mistral Models and Derivatives ... for testing,
> research, Personal, or evaluation purposes in Non-Production
> Environments; ... You shall not supply the Mistral Models or
> Derivatives in the course of a commercial activity ... including but
> not limited to through a hosted or managed service (e.g. SaaS, cloud
> instances, etc.), or behind a software layer.

Serving Codestral through Keystone — a platform a client deploys
commercially, behind an API layer — is exactly the use this license
prohibits. **Removed entirely** rather than documented-and-accepted (the
path taken for Grafana's AGPL-3.0, finding #2): unlike AGPL's network
copyleft, which Keystone's actual usage pattern doesn't trigger, MNPL's
restriction is triggered by the mere act of serving the model in
production, which is the platform's entire purpose. No config flag or
usage caveat makes this safe to ship enabled-by-default; the `codestral`
model role, its `vllm-codestral` service/StatefulSet, and every reference
to it were removed from `docker-compose.yml`, `helm/keystone/`,
`src/config.py`, `src/inference/`, `src/api/`, `src/db/models.py` (and
its Alembic baseline enum), `airgap/`, `cli/opencode.config.json`, and
`web/`. Coding capacity is unaffected — GLM-5.3-Flash (primary) and
Qwen2.5-Coder-32B-Instruct (fallback) cover the same ground.

## Self-hosted infrastructure images

| Component | Image | License (verified) |
|---|---|---|
| Postgres | `postgres:16-alpine` | PostgreSQL License (permissive, OSI-approved) |
| Cache/queue | `valkey/valkey:7.2-alpine` | BSD-3-Clause |
| Vector DB | `qdrant/qdrant:v1.11.0` | Apache-2.0 |
| LLM serving | `vllm/vllm-openai:v0.29.0` (`glm53-flash` tag for the coding role — same vLLM project, a model-specific pre-built image) | Apache-2.0 |
| Durable execution | `temporalio/auto-setup:1.24`, `temporalio/ui:2.27.0` | MIT |
| Reverse proxy | `nginx:1.27-alpine` | BSD-2-Clause (nginx license) |
| Secrets | `openbao/openbao:2.1` | MPL-2.0 (Apache-2.0-era Vault fork, but OpenBao itself ships under MPL-2.0 — verified against the real upstream LICENSE, correcting an earlier assumption in this project's own notes that it was Apache-2.0) |
| Metrics | `prom/prometheus:v2.55.0` | Apache-2.0 |
| Alerting | `prom/alertmanager:v0.27.0` | Apache-2.0 |
| Dashboards | `grafana/grafana:11.2.0` | AGPL-3.0 — see finding #2 above |
| Log aggregation | `grafana/loki:3.2.0`, `grafana/promtail:3.2.0` | AGPL-3.0 (same Grafana Labs monorepo/license as Grafana core — same "don't modify and redistribute" caveat as finding #2; unmodified stock images, same low-risk treatment) |
| Sandbox (fallback) | gVisor (`runsc`) | Apache-2.0 |
| Sandbox (primary) | Firecracker | Apache-2.0 |
| Multi-GPU training (optional) | KubeRay operator (`quay.io/kuberay/operator:v1.7.1`) + Ray (`ray[train]`, the training image) | Apache-2.0 (both; verified against the upstream `LICENSE` files at github.com/ray-project/kuberay and github.com/ray-project/ray) |
| Adapter export to GGUF | llama.cpp's `convert_hf_to_gguf.py`, `conversion/` and `gguf-py/`, vendored into the training image at a pinned commit (`docker/training.Dockerfile`); the `gguf` and `sentencepiece` packages | MIT (llama.cpp and gguf-py, `Copyright (c) 2023 Georgi Gerganov`, the `gguf-py/LICENSE` vendored with it); Apache-2.0 (sentencepiece, verified against the upstream `LICENSE` at github.com/google/sentencepiece — its PyPI metadata carries no classifier) |
| Adapter export to AWQ (optional, GPU only) | `autoawq` — not installed by default, not part of any image | MIT |
| Autoscaling (optional, vLLM roles) | KEDA | Apache-2.0 (verified against the real upstream `LICENSE` at github.com/kedacore/keda) |
| Infra tooling | OpenTofu | MPL-2.0 |
| Infra tooling | Helm | Apache-2.0 |
| Local dev cluster | kind | Apache-2.0 |

No AGPL/SSPL/BSL/proprietary component sits in Keystone's own image or
package builds except Grafana and Loki/Promtail (finding #2,
accepted/documented, same reasoning applies to both) — nothing else in
this table required a substitution beyond the Redis→Valkey swap.

## Model weights

A separate concern from the serving software above (vLLM is Apache-2.0
regardless of which weights it's pointed at) — the license on the weights
themselves. Every entry checked against the model's real HuggingFace
`LICENSE` file (or, where the repo only links out, the linked license
text itself fetched and read in full — see finding #3 above for why that
mattered for Codestral).

| Role | Model | License (verified) |
|---|---|---|
| Coding (primary) | `zai-org/GLM-5.3-Flash` | MIT |
| Coding (fallback) | `Qwen/Qwen2.5-Coder-32B-Instruct` | Apache-2.0 |
| Reasoning (critic) | `deepseek-ai/DeepSeek-R1` | MIT |
| Embeddings | `BAAI/bge-large-en-v1.5` | MIT |

All four are genuinely permissive with no usage restriction incompatible
with commercial self-hosting — unlike Codestral-22B-v0.1 (finding #3),
none of these carry a non-production/non-commercial clause. Note the
flagship `zai-org/GLM-5.3` (not the `-Flash` variant Keystone actually
serves) ships under a *different*, custom "GLM-5.3 License" — permissive
and MIT-like, but with an added clause requiring a Z.AI security review
only for a licensee that itself operates a competing "Model as a Service"
business earning US$10B+/year in aggregate revenue. That clause doesn't
apply to Keystone's own customers (running Keystone for their own
internal use, not reselling GLM-5.3 access as a competing MaaS product at
that revenue scale) — noted here only because it's a real difference from
the `-Flash` variant actually deployed, in case a future engagement
considers the flagship model instead.

## Python dependencies (`pyproject.toml`, core + dev)

Full inventory generated with `pip-licenses` against a clean venv install
of `.[dev]` (136 packages, including transitive dependencies) —
`/tmp/keystone-py-licenses.csv`-style output, summarized here; regenerate
with:

```bash
python3 -m venv /tmp/audit-venv && source /tmp/audit-venv/bin/activate
pip install -q pip-licenses && pip install -q -e ".[dev]"
pip-licenses --format=markdown --order=license
```

**Result: zero GPL/AGPL/SSPL/BSL/proprietary packages found** in the core
+ dev dependency tree — every package resolves to MIT, BSD (2- or
3-clause), Apache-2.0, ISC, MPL-2.0, PSF-2.0, 0BSD, or public-domain
(Unlicense). License families seen, by rough package count:

- MIT / MIT-style: the majority (FastAPI, Pydantic, SQLAlchemy, Alembic,
  LangGraph, LangChain-core, Anthropic SDK, OpenAI SDK, Temporal SDK,
  Redis client, ruff, pytest, uvicorn, ...)
- Apache-2.0 / Apache-2.0-family: transformers, huggingface-hub,
  sentence-transformers, tokenizers, safetensors, asyncpg, docker,
  qdrant-client, requests, tenacity
- BSD (2/3-clause): httpx, GitPython, click, Jinja2, scikit-learn, scipy,
  numpy (BSD-3-Clause plus small 0BSD/MIT/Zlib/CC0 sub-components in
  vendored code), psutil, uvicorn's transport deps
- MPL-2.0: certifi, orjson (dual MPL-2.0/Apache-2.0-or-MIT), tqdm
- Other permissive: ISC (dnspython, shellingham), 0BSD (detect-installer),
  Unlicense (email-validator), PSF-2.0 (typing-extensions)

**finetuning extra** (`torch`, `peft`, `trl`, `bitsandbytes`, `accelerate`,
`wandb`) not installed/audited in this pass (large, GPU-oriented,
optional) — `torch` itself reports a composite
`Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND
BSD-3-Clause AND BSL-1.0 AND MIT` (multiple vendored components, all
individually permissive; BSL-1.0 here is the unrelated **Boost Software
License 1.0**, not Business Source License — a common naming collision,
confirmed by checking which sub-component it tags). Re-run the audit
command above with `-e ".[finetuning]"` before a client engagement that
enables fine-tuning, to get the same level of verification.

**Present as a transitive dependency but never used**: `sentry-sdk`
(pulled in by `fastapi[standard]`'s optional `fastapi-cloud-cli`) and
`langsmith` (pulled in by `langchain-core`) are both installed but never
imported, initialized, or configured anywhere in `src/` — verified by
grep (`sentry`, `SENTRY`, `langsmith`, `LANGCHAIN_TRACING`, `LANGSMITH`
all return no matches outside dependency metadata). Neither can phone
home without an explicit DSN/API key Keystone never sets. See
`TELEMETRY_AUDIT.md` for the full phone-home audit.

## npm (Keystone Agents interactive CLI)

- **`opencode-ai`**: MIT (verified against the real published package,
  `npm view opencode-ai license` and the package's own `package.json`,
  during the `airgap/build_npm_mirror.sh` build — that script now asserts
  this at build time and warns loudly if a future version's license
  changes, rather than assuming it stays MIT forever).

## Log aggregation + alerting (Loki, Promtail, Alertmanager)

Prometheus + Grafana + Loki + Promtail + Alertmanager are all built and
deployed (`docker-compose.yml`'s `observability` profile,
`helm/keystone/templates/observability.yaml`) — see
`KUBERNETES_CLIENT_VPC.md` for what's been verified end-to-end on a real
cluster.

### 3. Loki has its own default-on telemetry (fixed)

Separate from the Grafana core finding above: Loki's `analytics.
reporting_enabled` setting defaults to `true` and phones home to
`stats.grafana.org` — found while building the real Loki config for this
project, not something carried over from finding #2. Disabled explicitly
in both `observability/loki/loki-config.yaml` and the Helm chart's
embedded Loki config. See `docs/TELEMETRY_AUDIT.md`.
