# Keystone

A self-hosted, air-gap-capable platform with two pillars: an **OpenAI- and Anthropic-compatible gateway** in front of open-weight models served by vLLM, and a **deterministic coding agent** that clones your real repository into a sandbox, edits it with real tools, runs your real tests and opens a real pull request.

Every page here says what has been verified and by which test. A claim with no test is not made (`docs/claims.yaml`, checked in CI).

## Start here

| I want to… | Read |
|---|---|
| run it on a laptop in minutes, then on one GPU | [Quickstart](getting-started/quickstart.md) |
| know what GPU a model needs | [Hardware sizing](getting-started/hardware-sizing.md) |
| call it from my apps and tools | [The gateway API](guides/gateway-api.md) |
| hand a task to the coding agent | [The coding agent](guides/coding-agent.md) |
| see what models are up and try one | [Model Library and Playground](guides/model-library-and-playground.md) |
| fine-tune a small model on my repositories | [Fine-tune an SLM on your repo](guides/fine-tune-slm-on-your-repo.md) |
| use it from VS Code | [Use it from VS Code](guides/use-from-vscode.md) |
| deploy on Kubernetes, RunPod, or with no internet | [Deployment](deployment/KUBERNETES_CLIENT_VPC.md), [RunPod](deployment/RUNPOD_SETUP.md), [Air-gapped](airgap/OFFLINE_INSTALL_RUNBOOK.md) |
| stand up a GPU pilot on AWS, Azure or Google Cloud | [AWS](guides/deploy-aws.md), [Azure](guides/deploy-azure.md), [Google Cloud](guides/deploy-gcp.md) — validated, not yet applied ([verification log](deployment/verification-log.md)) |
| decide whether open weights fit my organisation | [Open weights: risk and support](enterprise/open-weights-risk-and-support.md) |
| see how quality and cost are measured | [Benchmarks](benchmarks/README.md) |
| fix an error message | [Troubleshooting](troubleshooting.md) |
| look up a setting, a command, or a route | [Configuration](reference/configuration.md), [CLI](reference/cli.md), [API](reference/api.md) |
| understand why it is built this way | [Architecture](architecture/SANDBOX_ARCHITECTURE.md), [Decision records](architecture/adr/README.md) |

## What is real today

- Gateway: `/v1/chat/completions` (tools, structured output, streaming with real usage), `/v1/messages` (Anthropic Messages API: text, tools, streaming), `/v1/models`, `/v1/keystone/models` (live states), a dollar ledger per tenant and role.
- Agent: repository map, line-ranged reads, fuzzy patches, dependency install, related-then-full tests, quality gates that fail closed, review that fails closed, re-planning, a live step-level trace, a real PR on a real Gitea.
- Fine-tuning: describe → plan & cost → approve → train with live loss → verdict against the base model → promote → served without a restart.
- Deployment: docker-compose, Helm (multi-node vLLM, KEDA autoscaling, LoRA serving), RunPod Serverless, an offline bundle for air-gapped sites.

What is not yet real is stated on the page that would otherwise claim it, and tracked in the repository's `ROADMAP.md`.
