# Configuration reference

Every setting Keystone reads, generated from `src/config.py` by `scripts/gen_config_reference.py` (CI fails when this page is stale). Set them in `.env` (which `keystone init` writes) or as environment variables; names are case-insensitive. Every limit is `0 = unlimited` by default: capacity is bounded by what you deploy, not by a policy number.

## Platform

| Variable | Type | Default | What it does |
|---|---|---|---|
| `VS_ENV` | `Environment` | `'production'` | development \| staging \| production — production refuses weak secrets and unsandboxed runs |
| `VS_DEBUG` | `bool` | `False` | verbose errors in API responses; never in production |
| `VS_SECRET_KEY` | `secret` | `(generated at startup)` | signs session/CSRF material; set a stable value in production (a restart otherwise rotates it) |
| `KEYSTONE_ROOT_ADMIN_TOKEN` | `secret | None` | `unset` | bootstrap token for the /v1/admin routes and `keystone tenants\|users\|keys-create` |
| `VS_ADMIN_EMAIL` | `str` | `'admin@keystone.local'` | contact shown in operator-facing messages |
| `VS_PLATFORM_NAME` | `str` | `'Keystone'` | display name in logs and the web UI |
| `CORS_ALLOWED_ORIGINS` | `list[str]` | `['https://keystone.local']` | browser origins allowed to call the API (CORS); JSON list |
| `MCP_ALLOWED_HOSTS` | `list[str]` | `['127.0.0.1:*', 'localhost:*', '[::1]:*']` | MCP server (src/api/routes/mcp.py) DNS-rebinding-protection allowlist — defaults match the SDK's own localhost-only default; a real deployment reachable at a real hostname (not just 127.0.0.1 behind a reverse proxy) must override these to that hostname, or every MCP request will be correctly rejected the same way an arbitrary Host header is today. |
| `MCP_ALLOWED_ORIGINS` | `list[str]` | `['http://127.0.0.1:*', 'http://localhost:*', 'http://[::1]:*']` | Origin header allowlist for the MCP server (same rule as MCP_ALLOWED_HOSTS) |

## PostgreSQL

| Variable | Type | Default | What it does |
|---|---|---|---|
| `POSTGRES_HOST` | `str` | `'postgres'` | Postgres host (the compose service name by default) |
| `POSTGRES_PORT` | `int` | `5432` | Postgres port |
| `POSTGRES_DB` | `str` | `'keystone'` | database name |
| `POSTGRES_USER` | `str` | `'keystone'` | database role |
| `POSTGRES_PASSWORD` | `secret` | `(secret; set it)` | database password — `keystone init` generates one; the placeholder is refused |
| `DATABASE_URL` | `str | None` | `unset` | full SQLAlchemy URL; assembled from the POSTGRES_* values when unset |

## Redis

| Variable | Type | Default | What it does |
|---|---|---|---|
| `REDIS_URL` | `str` | `'redis://redis:6379/0'` | Redis/Valkey URL for events, rate limits and the concurrency limiter |
| `REDIS_PASSWORD` | `secret | None` | `unset` | Redis/Valkey password, embedded into REDIS_URL by `keystone init` |

## Keystone Inference — vLLM endpoints

| Variable | Type | Default | What it does |
|---|---|---|---|
| `VLLM_CODING_URL` | `str` | `'http://vllm-coding:8000/v1'` | OpenAI-compatible base URL serving the `coding` role (vLLM, llama.cpp, RunPod, the frontier proxy) |
| `VLLM_CODING_FALLBACK_URL` | `str` | `'http://vllm-coding-fallback:8000/v1'` | base URL serving the `coding_fallback` role |
| `VLLM_REASONING_URL` | `str` | `'http://vllm-reasoning:8000/v1'` | base URL serving the `reasoning` role |
| `VLLM_CODING_API_KEY` | `secret | None` | `unset` | bearer token for the coding endpoint, if it needs one |
| `VLLM_CODING_FALLBACK_API_KEY` | `secret | None` | `unset` | bearer token for the coding_fallback endpoint |
| `VLLM_REASONING_API_KEY` | `secret | None` | `unset` | bearer token for the reasoning endpoint |

## RunPod Serverless (see docs/deployment/RUNPOD_SETUP.md)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `RUNPOD_API_KEY` | `secret | None` | `unset` | RunPod API key (`keystone deploy runpod-serverless`, `keystone doctor`) |
| `RUNPOD_ENDPOINT_URL` | `str | None` | `unset` | the RunPod Serverless endpoint's OpenAI-compatible base URL, once created |
| `CODING_MODEL_ID` | `str` | `'zai-org/GLM-5.3-Flash'` | GLM-5.3-Flash (zai-org, MIT) — primary coding model as of the Phase 0 model refresh; Qwen2.5-Coder-32B-Instruct moved to coding_fallback_model_id (src/inference/model_router.py's fallback chain), used when the primary endpoint is unhealthy. See src/inference/config.py's coding_model_config() docstring for the real verified footprint/tool-parser details. HF id of the model behind `coding` — metadata for /v1/models and the Model Library, not routing |
| `CODING_FALLBACK_MODEL_ID` | `str` | `'Qwen/Qwen2.5-Coder-32B-Instruct'` | HF id of the model behind `coding_fallback` |
| `REASONING_MODEL_ID` | `str` | `'deepseek-ai/DeepSeek-R1'` | HF id of the model behind `reasoning` |
| `TASK_COMPLEXITY_ROUTING_ENABLED` | `bool` | `False` | When a task is submitted with model="auto" and classify_task_to_role resolves it to "coding", this additionally checks task complexity (src/inference/model_router.py's classify_task_complexity) and routes a "simple" task to coding_fallback instead of the primary model. Default off: this changes which model actually serves an "auto" task, so it's an explicit opt-in rather than a silent behavior change. |

## Qdrant

| Variable | Type | Default | What it does |
|---|---|---|---|
| `QDRANT_HOST` | `str` | `'qdrant'` | Qdrant host for the code/memory vector store |
| `QDRANT_PORT` | `int` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | `secret | None` | `unset` | Qdrant API key |
| `QDRANT_COLLECTION` | `str` | `'keystone_code_memory'` | collection name for code chunks and memories |
| `QDRANT_USE_TLS` | `bool` | `False` | internal cluster traffic; flip on once mTLS is wired up (see OpenBao workstream) |

## Embedding

| Variable | Type | Default | What it does |
|---|---|---|---|
| `EMBEDDING_MODEL_NAME` | `str` | `'BAAI/bge-large-en-v1.5'` | sentence-embedding model for retrieval |
| `EMBEDDING_MODEL_PATH` | `str | None` | `unset` | local path override for air-gapped deployments |
| `EMBEDDING_DIMENSION` | `int` | `1024` | vector size of the embedding model |

## Sandbox (self-hosted — src/sandbox/)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `SANDBOX_BACKEND` | `str` | `'gvisor'` | "gvisor" \| "firecracker" |
| `SANDBOX_DAEMON_URL` | `str` | `'http://sandbox-daemon:9000'` | the sandbox daemon every agent command runs through |
| `SANDBOX_TIMEOUT_SECONDS` | `int` | `300` | default wall clock for one sandbox command |
| `SANDBOX_MAX_CONCURRENT` | `int` | `0` | per-tenant concurrent sandboxes; 0 = no cap (host capacity is the limit) |
| `ALLOW_UNSANDBOXED_DEV` | `bool` | `False` | run agent commands on the host without a sandbox (development only; refused in production) |

## Git (internal Gitea in air-gapped mode)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `GIT_ALLOWED_HOSTS` | `list[str]` | `['gitea.internal.keystone.local']` | the only hosts the agent may clone from or push to (SSRF guard); JSON list |
| `GIT_HOST_KIND` | `str` | `'gitea'` | selects the GitHost implementation (src/git/host.py) — "gitea" only, for now |
| `GIT_HOST_API_URL` | `str` | `'http://gitea.internal.keystone.local:3000/api/v1'` | the git host's REST API base |
| `GIT_HOST_TOKEN` | `secret | None` | `unset` | bot account token used to open PRs on the agent's behalf |
| `GIT_HOST_COMMIT_AUTHOR_NAME` | `str` | `'Keystone Agents'` | author on the agent's commits |
| `GIT_HOST_COMMIT_AUTHOR_EMAIL` | `str` | `'keystone-agents@keystone.local'` | author email on the agent's commits |
| `PR_POLL_INTERVAL_SECONDS` | `int` | `300` | src/orchestrator/pr_polling.py's periodic merged/rejected feedback check |
| `SCHEDULE_POLL_INTERVAL_SECONDS` | `int` | `60` | src/orchestrator/schedules.py's periodic due-cron-schedule check |

## Package mirrors (internal, air-gap-safe — src/orchestrator/repo_profile.py)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `PIP_INDEX_URL` | `str | None` | `unset` | e.g. https://pypi.internal.keystone.local/simple |
| `NPM_REGISTRY_URL` | `str | None` | `unset` | e.g. https://npm.internal.keystone.local |
| `GO_PROXY_URL` | `str | None` | `unset` | e.g. https://goproxy.internal.keystone.local |

## Token Budgets

| Variable | Type | Default | What it does |
|---|---|---|---|
| `DEFAULT_DAILY_TOKEN_LIMIT` | `int` | `0` | tokens/day per new tenant (0 = unlimited) |
| `DEFAULT_MONTHLY_TOKEN_LIMIT` | `int` | `0` | tokens/month per new tenant (0 = unlimited) |
| `DEFAULT_REQUESTS_PER_MINUTE` | `int` | `0` | gateway request rate per API key (0 = unlimited) |
| `MAX_AGENT_ITERATIONS` | `int` | `15` | the one per-task SAFETY bound (a runaway fix/test loop); per-task override, no ceiling |
| `MAX_TOKENS_PER_TASK` | `int` | `0` | tokens one agent task may spend (0 = unlimited) |
| `MAX_TOKENS_PER_REQUEST` | `int` | `0` | ceiling on a completion request's max_tokens (0 = the model's own limit) |

## Agent time bounds (SAFETY, not capacity — generous, configurable, per-task)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `AGENT_TEST_TIMEOUT_SECONDS` | `int` | `3600` | the repo's full test suite |
| `AGENT_INSTALL_TIMEOUT_SECONDS` | `int` | `3600` | the repo's dependency install |
| `AGENT_QUALITY_TIMEOUT_SECONDS` | `int` | `1800` | each lint/typecheck/security tool |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `int` | `1800` | ceiling for run_command's own timeout argument |
| `AGENT_MAX_WALL_CLOCK_SECONDS` | `int` | `0` | whole-task wall clock (0 = none; max_iterations is the bound) |

## Inference endpoint health (src/inference/health.py)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `INFERENCE_HEALTH_CACHE_SECONDS` | `int` | `10` | reuse a /models probe result this long |
| `INFERENCE_BREAKER_FAILURE_THRESHOLD` | `int` | `3` | consecutive failures that open a role's breaker (0 = never) |
| `INFERENCE_BREAKER_OPEN_SECONDS` | `int` | `30` | how long an open breaker skips the role before one retry |

## Usage ledger pricing (src/billing/ledger.py)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `MODEL_PRICES_PER_MILLION` | `dict[str, float]` | `{}` | USD per 1,000,000 tokens per model role, from your own GPU economics (see benchmarks/cost_model.py for the calculator). JSON, e.g. MODEL_PRICES_PER_MILLION='{"coding": 0.40, "coding_fallback": 0.20, "reasoning": 1.20}' Unpriced roles are recorded with cost 0 and the usage API says pricing is not configured. |

## Coding agent behaviour (src/orchestrator/nodes/)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `QUALITY_GATE_BLOCKING_TOOLS` | `list[str]` | `['bandit', 'mypy']` | Which quality-gate tools send a task back to fixing on a finding (nodes/quality.py). bandit (high/medium) and mypy by default; add "ruff", "eslint", "tsc", "go", "cargo" to fail on lint too. Per-task override: AgentTaskRequest.quality_blocking_tools. |
| `AGENT_MAX_CONTEXT_TOKENS` | `int` | `24000` | Token budget for the coding loop's own conversation (context.py's trimming/summarising); keep headroom under the serving model's max_model_len for the 8192-token completion. |
| `AGENT_BEST_OF_N` | `int` | `1` | Best-of-N coding: N independent attempts from the same base, each scored by the repo's own quality commands and related tests, the winner applied — multiplies model spend by N. 1 = off. Per-task override: AgentTaskRequest.best_of_n (no ceiling). |
| `AGENT_STREAM_TURNS` | `bool` | `True` | Stream each coding turn (tool calls + text accumulated from deltas) instead of one blocking request — live progress in the trace and no idle read-timeout on long turns. Set false for a backend that cannot stream tool calls. |

## Gateway behaviour

| Variable | Type | Default | What it does |
|---|---|---|---|
| `LOG_PROMPTS` | `bool` | `False` | Log every gateway prompt and reply (through PII redaction) — off by default: prompts are your users' data. |
| `GATEWAY_COST_ROUTING` | `bool` | `False` | model="auto" on the gateway routes a simple last message to coding_fallback (cheaper) instead of coding; the decision is returned in X-VS-Route-Decision. Off by default: a wrong downgrade costs quality. |

## Temporal

| Variable | Type | Default | What it does |
|---|---|---|---|
| `TEMPORAL_HOST` | `str` | `'temporal:7233'` | Temporal server address for durable agent/fine-tune workflows |
| `TEMPORAL_NAMESPACE` | `str` | `'keystone'` | Temporal namespace |
| `TEMPORAL_TASK_QUEUE` | `str` | `'keystone-agent-tasks'` | task queue the Keystone worker polls |

## TLS

| Variable | Type | Default | What it does |
|---|---|---|---|
| `TLS_CERT_PATH` | `str | None` | `unset` | TLS certificate for the API (`make certs` / `make certs-ca`) |
| `TLS_KEY_PATH` | `str | None` | `unset` | TLS private key |

## Observability

| Variable | Type | Default | What it does |
|---|---|---|---|
| `PROMETHEUS_PORT` | `int` | `9090` | port the /metrics scrape target listens on |
| `LOG_LEVEL` | `str` | `'INFO'` | DEBUG \| INFO \| WARNING \| ERROR |
| `LOG_FORMAT` | `str` | `'json'` | json (machine-readable) or console |

## GitHub

| Variable | Type | Default | What it does |
|---|---|---|---|
| `GITHUB_APP_ID` | `str | None` | `unset` | GitHub App id (GitHub host support; Gitea is the verified path today) |
| `GITHUB_PRIVATE_KEY_PATH` | `str | None` | `unset` | GitHub App private key file |
| `GITHUB_WEBHOOK_SECRET` | `str | None` | `unset` | GitHub webhook secret |

## Fine-Tuning

| Variable | Type | Default | What it does |
|---|---|---|---|
| `WANDB_API_KEY` | `str | None` | `unset` | Weights & Biases key for training-run logging (optional) |
| `FINETUNING_OUTPUT_DIR` | `str` | `'/data/finetuning/output'` | where adapters and the lora_modules.json manifest are written; mount it into the serving pods |
| `FINETUNING_DATA_DIR` | `str` | `'/data/finetuning/data'` | where guided fine-tunes write their datasets and manifests |
| `HF_TOKEN` | `str | None` | `unset` | Hugging Face token for gated model downloads |
| `FINETUNE_BACKEND` | `str` | `'inprocess'` | Where a job's trainer runs (src/finetuning/backends/): "inprocess" = this process, on the host's own GPUs; "ray" = a Ray Train job submitted to the KubeRay cluster at RAY_ADDRESS (ADR 0002); "runpod_pod" = an on-demand RunPod GPU pod running the training image, the adapter synced to a network volume (ADR 0005) |

## Fine-Tuning: Ray Train backend (src/finetuning/backends/ray_train.py)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `RAY_ADDRESS` | `str` | `'http://keystone-training-head-svc:8265'` | the Ray dashboard URL jobs are submitted to — KubeRay names it <cluster>-head-svc; the chart sets this |
| `RAY_TRAIN_NUM_WORKERS` | `int` | `0` | Ray Train workers, one GPU each (0 = the plan's gpu_count, else 1) |
| `RAY_TRAIN_USE_GPU` | `bool` | `True` | ask Ray for a GPU per worker (false only for a CPU smoke on a CPU cluster) |
| `RAY_JOB_POLL_INTERVAL_SECONDS` | `float` | `5.0` | how often the runner asks the Ray Jobs API for the job's status |
| `RAY_JOB_TIMEOUT_SECONDS` | `int` | `0` | give up on a Ray job after this long (0 = wait as long as it runs) |

## Fine-Tuning: RunPod pod backend (src/finetuning/backends/runpod_pod.py)

| Variable | Type | Default | What it does |
|---|---|---|---|
| `RUNPOD_TRAINING_IMAGE` | `str` | `'keystone-training:latest'` | docker/training.Dockerfile's image, as RunPod can pull it (a registry RunPod reaches, not a local tag) |
| `RUNPOD_TRAINING_GPU_TYPE_IDS` | `list[str]` | `['NVIDIA L4', 'NVIDIA RTX A5000', 'NVIDIA GeForce RTX 4090']` | RunPod `gpuTypeIds` tried in order for the training pod (exact enum values from RunPod's OpenAPI document) |
| `RUNPOD_TRAINING_GPU_COUNT` | `int` | `1` | GPUs on the training pod |
| `RUNPOD_TRAINING_CONTAINER_DISK_GB` | `int` | `50` | the pod's container disk (image + model cache) |
| `RUNPOD_TRAINING_CLOUD_TYPE` | `str` | `'SECURE'` | RunPod cloud tier for the pod: SECURE or COMMUNITY |
| `RUNPOD_NETWORK_VOLUME_ID` | `str | None` | `unset` | the RunPod network volume the adapter is written to (mounted at /runpod-volume); a serverless endpoint created with the same volume serves it — required for this backend, a pod's own disk dies with it |
| `RUNPOD_POD_POLL_INTERVAL_SECONDS` | `float` | `15.0` | how often the runner polls the pod and its status endpoint |
| `RUNPOD_POD_TIMEOUT_SECONDS` | `int` | `0` | stop and delete the pod after this long (0 = as long as training runs) |
| `RUNPOD_REST_BASE_URL` | `str` | `'https://rest.runpod.io/v1'` | RunPod's REST API (override only to test locally) |
| `RUNPOD_PROXY_URL_TEMPLATE` | `str` | `'https://{pod_id}-{port}.proxy.runpod.net'` | RunPod's HTTP proxy for a pod's exposed port — how the runner reads the pod's live status endpoint |
| `RUNPOD_POD_STATUS_PORT` | `int` | `8000` | the port the training pod's status endpoint listens on |
