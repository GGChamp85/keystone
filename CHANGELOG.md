# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Platform foundation — self-hosted, air-gapped LLM inference gateway and autonomous coding agent platform:

### Added
- **Keystone Inference**: OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/models`), API key management, rate limiting, token budgets, subscription tiers, multi-model routing (coding/coding_fallback/reasoning) with health-aware fallback and per-tenant adapter resolution.
- **Keystone Agents**: LangGraph Plan→Code→Quality→Review→Test→Fix background agent with real tool use (read/grep/run/patch, not full-file rewrites), a real git workflow (clone → branch → commit → push → PR against a real Gitea), Temporal-backed durable execution, and a live plan/execution-trace web UI (`web/`, React SPA served at `/app`) streaming progress over Server-Sent Events.
- **Multi-developer + scale**: user/role identity (`admin`/`lead`/`developer`) on every task and memory action; a Redis-backed per-tenant/per-user concurrency limiter; automatic PR-status polling into `task_feedback`; incremental, AST-aware, per-tenant repo indexing for large repos/monorepos.
- **Memory system**: a per-repo/per-tenant memory the agent learns from and a human can inspect, pin, approve, and forget — exposed via `keystone memory ...`, the web UI's Memory panel, and an OpenCode plugin/MCP server.
- **Fine-tuning, end to end**: real training-data sources (git history, accepted-task trajectories, repo pretraining text), a stratified train/holdout manifest with real content hashes, eval-aware LoRA/SFT/DPO trainers, an adapter registry with promote/rollback, LoRA-serving vLLM flags, adapter-aware routing through every real inference call site, a Temporal-backed job runner with an asyncio fallback, a full `/v1/finetune` REST API, `keystone finetune ...` CLI commands, and a 3-step web wizard (pick data → start → verdict).
- **Distributed model serving**: per-role horizontal replica scaling behind a load-balanced Service, real multi-node pipeline-parallel serving via vLLM's Ray executor for models too large for one GPU node, and KEDA-based autoscaling on the real `vllm:num_requests_waiting` metric — all three paths rendered and verified in CI via `helm lint`/`helm template`.
- **Real, non-mocked benchmark harness** (`benchmarks/`): scores a model's completion by actually executing it in a real sandbox (gVisor) — no LLM-as-judge — with a real token-cost comparison against gated frontier clients (Anthropic/OpenAI, only constructed when a real API key is present). Run for real against `claude-opus`/`claude-sonnet` in this repo's own development environment; see the README's benchmark section for the actual numbers.
- **Self-hosted sandboxing**: Firecracker microVMs (primary) with an automatic gVisor fallback where KVM isn't available; host-side-only network egress enforcement.
- **Air-gap packaging**: offline bundle scripts for container images, Python wheels, npm packages, and model weights; a real internal CA (`pki/`); verified zero-egress runtime for the core stateful stack.
- **Observability**: Prometheus, Grafana (3 real dashboards), Loki, Promtail, and Alertmanager with 6 real alert rules — deployed and verified on both docker-compose and a live Kubernetes cluster.
- **Infrastructure**: Kubernetes/Helm chart (client-VPC production tier) and OpenTofu modules (RunPod dev/test tier); NetworkPolicy-segmented tiers; OpenBao-backed secrets with a real init/unseal/seed lifecycle.
- **Compliance**: full third-party license inventory (`docs/LICENSES_AND_COMPLIANCE.md`) and telemetry audit (`docs/TELEMETRY_AUDIT.md`), including catching and fixing a Redis relicensing issue (switched to Valkey) before it shipped.

### Fixed
- **The sandbox's egress firewall never actually allowed a deployment's real configured `GIT_ALLOWED_HOSTS`** — only the hardcoded placeholder `*.internal.keystone.local` hostnames, regardless of what an operator set. A custom git host correctly passed the SSRF allowlist check (`_validate_repo_url`) and then had its real `git clone` silently blocked by egress DENY, completely breaking the README's own "Connect your own git server" instructions for anyone who followed them. Found and fixed by actually running `tests/test_git_workflow_integration.py` end to end against a live Gitea for the first time — `build_egress_policy()` now wires the real configured git/mirror hosts into real ALLOW rules.
- **`docker-compose.yml`'s sandbox network had no pinned name**, so Compose silently prefixed it with the project name; every network-enabled sandbox (any git operation) then failed outright with a real 404 from the Docker API. Same discovery pass as above.

### Known limitations (tracked, not hidden)
- Model-quality behavior on Keystone's own self-hosted models (the actual LLM output) is unverified in this codebase's development environment, which has no GPU — see `docs/deployment/KUBERNETES_CLIENT_VPC.md` for what's verified vs. GPU-gated.
- No adapter has been fine-tuned on real GPU hardware yet, and fine-tuning's "only promote if it beats base+RAG" auto-verdict gate isn't wired up — promotion is a manual decision today.
- The full multi-file-repo benchmark suite (SWE-bench-lite-style, run through the complete agent loop) doesn't exist yet — only a small 3-task single-function harness does.
- See `ROADMAP.md` for the full phase-by-phase status, including what's checked and what each checked phase's known gap still is.

[Unreleased]: https://github.com/GGChamp85/keystone/commits/main
