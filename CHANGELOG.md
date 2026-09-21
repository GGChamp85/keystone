# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Platform foundation — self-hosted, air-gapped LLM inference gateway and autonomous coding agent platform:

### Added
- **Keystone Inference**: OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/models`), API key management, rate limiting, token budgets, subscription tiers, multi-model routing (coding/coding_fallback/reasoning) with health-aware fallback.
- **Keystone Agents**: LangGraph Plan→Code→Review→Test→Fix background agent with Temporal-backed durable execution; a live plan/execution-trace web UI (`web/`, React SPA served at `/app`) streaming progress over Server-Sent Events; the OpenCode interactive CLI pre-configured against Keystone Inference.
- **Self-hosted sandboxing**: Firecracker microVMs (primary) with an automatic gVisor fallback where KVM isn't available; host-side-only network egress enforcement.
- **Air-gap packaging**: offline bundle scripts for container images, Python wheels, npm packages, and model weights; a real internal CA (`pki/`); verified zero-egress runtime for the core stateful stack.
- **Observability**: Prometheus, Grafana (3 real dashboards), Loki, Promtail, and Alertmanager with 6 real alert rules — deployed and verified on both docker-compose and a live Kubernetes cluster.
- **Infrastructure**: Kubernetes/Helm chart (client-VPC production tier) and OpenTofu modules (RunPod dev/test tier); NetworkPolicy-segmented tiers; OpenBao-backed secrets with a real init/unseal/seed lifecycle.
- **Fine-tuning scaffolding**: LoRA/SFT/DPO training scripts, an RL rollout coordinator reusing the sandbox pool, and a Kubeflow Trainer `ClusterTrainingRuntime` for multi-node runs.
- **Compliance**: full third-party license inventory (`docs/LICENSES_AND_COMPLIANCE.md`) and telemetry audit (`docs/TELEMETRY_AUDIT.md`), including catching and fixing a Redis relicensing issue (switched to Valkey) before it shipped.

### Known limitations (tracked, not hidden)
- Model-quality behavior (the actual LLM output) is unverified in this codebase's development environment, which has no GPU — see `docs/deployment/KUBERNETES_CLIENT_VPC.md` for what's verified vs. GPU-gated.
- See `ROADMAP.md` for the agent-loop, fine-tuning, and enterprise-readiness work planned next.

[Unreleased]: https://github.com/GGChamp85/keystone/commits/main
