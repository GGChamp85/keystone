# Open weights: risk and support

For the person who has to sign off. What "self-hosted open-weight models" means for your organisation, what can go wrong, and who is responsible for what.

## Why open weights

| Concern | With a hosted frontier API | With Keystone and open weights |
|---|---|---|
| Where code and prompts go | the vendor's cloud, every request | your network; optional frontier path is explicit and can redact personal data |
| Cost as usage grows | per token, set by the vendor | GPU time you own or rent; fixed at volume; a ledger shows the dollars per tenant and model |
| Model changes | at the vendor's schedule | at yours; a model is a file you keep and can pin forever |
| Improvement on your code | prompting only | fine-tuning on your own history, gated on measurable improvement |
| Vendor dependency | total | none required; every component is open source with its license verified |

## What can go wrong, and what Keystone does about it

| Risk | Mitigation built in | What remains yours |
|---|---|---|
| **Quality below a frontier model** | the same pipeline can front a frontier model for the tasks that need it; the benchmark harness measures your models on your tasks, not vendor claims | deciding which workloads run on which model |
| **An agent changes code incorrectly** | every task ends in a pull request; nothing merges without a human; quality gates and tests are enforced by the platform, not requested of the model; a fresh clone re-verifies the pushed branch | your review process and branch protection |
| **Malicious or compromised repository content** | tool results are wrapped and injection phrasings flagged; the sandbox's egress is restricted to allowed hosts; the agent only clones from hosts you list | the allowlist and the repositories you point it at |
| **Personal data leaving the network** | nothing leaves by default; the one frontier path redacts common personal-data patterns when enabled | classifying what may use the frontier path |
| **Model licensing** | defaults are MIT and Apache-2.0 models; every third-party license is inventoried with the finding that changed the build | legal review of any model you add |
| **GPU cost surprises** | no hidden metering; the ledger records every token with your own price; hardware sizing is documented per model; RunPod Serverless scales to zero | choosing hardware and prices |
| **Operational failure** | health and readiness endpoints, a per-role circuit breaker with visible state, Prometheus alerts, durable task execution that survives restarts | running the cluster, backups, on-call |
| **Air-gap drift** | the same bundle scripts package every image, wheel, package and weight; the runbook states what is verified offline | your transfer and registry process |

## Who supports it

- **The open-source project**: issues, discussions and pull requests on GitHub, best effort, as described in `SUPPORT.md`. Every release runs the full test suite against real services in CI, and the documentation site's claims are checked mechanically against those tests.
- **Your platform team**: deployment, upgrades, backups, GPU capacity and the git server, using the runbooks under `docs/deployment` and `docs/airgap` and `keystone doctor` for diagnosis.
- **Commercial support and services**: not bundled with the software. Organisations that need a support agreement or hands-on deployment work should open a GitHub Discussion or contact the maintainers through the repository; no third party is authorised to represent the project.

## What to ask before adopting

1. Which workloads must never leave the network, and which may use a frontier model through the proxy?
2. Which GPU tier serves the model you want (see [Hardware sizing](../getting-started/hardware-sizing.md)), and who owns it?
3. Which git server will the agent use, and which repositories will it be allowed to touch?
4. What is the review policy for agent pull requests?
5. Who is on call for the platform, and what is the backup and restore drill for Postgres and the model cache?
