"""
Keystone — Central configuration loaded from environment.
Every module imports settings from here.
"""

from __future__ import annotations

import secrets
from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_WEAK_SECRET_SENTINELS = {
    "",
    "changeme",
    "change_me",
    "generate-a-64-char-hex-with-openssl-rand-hex-32",
    "change_me_strong_pg_password",
    "change_me_redis_password",
    "change_me_qdrant_key",
    "change_me_generate_with_openssl_rand_hex_32",
}


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Platform ──────────────────────────────────────────────
    vs_env: Environment = (
        Environment.PRODUCTION
    )  # development | staging | production — production refuses weak secrets and unsandboxed runs
    vs_debug: bool = False  # verbose errors in API responses; never in production
    # signs session/CSRF material; set a stable value in production (a restart otherwise rotates it)
    vs_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_hex(32)))
    keystone_root_admin_token: SecretStr | None = (
        None  # bootstrap token for the /v1/admin routes and `keystone tenants|users|keys-create`
    )
    vs_admin_email: str = "admin@keystone.local"  # contact shown in operator-facing messages
    vs_platform_name: str = "Keystone"  # display name in logs and the web UI
    # browser origins allowed to call the API (CORS); JSON list
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["https://keystone.local"])

    # MCP server (src/api/routes/mcp.py) DNS-rebinding-protection allowlist —
    # defaults match the SDK's own localhost-only default; a real deployment
    # reachable at a real hostname (not just 127.0.0.1 behind a reverse
    # proxy) must override these to that hostname, or every MCP request will
    # be correctly rejected the same way an arbitrary Host header is today.
    mcp_allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1:*", "localhost:*", "[::1]:*"])
    # Origin header allowlist for the MCP server (same rule as MCP_ALLOWED_HOSTS)
    mcp_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
    )

    # ── PostgreSQL ────────────────────────────────────────────
    postgres_host: str = "postgres"  # Postgres host (the compose service name by default)
    postgres_port: int = 5432  # Postgres port
    postgres_db: str = "keystone"  # database name
    postgres_user: str = "keystone"  # database role
    postgres_password: SecretStr = SecretStr(
        "changeme"
    )  # database password — `keystone init` generates one; the placeholder is refused
    database_url: str | None = None  # full SQLAlchemy URL; assembled from the POSTGRES_* values when unset

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_url(cls, v: str | None, info) -> str:
        if v:
            return v
        d = info.data
        password = d.get("postgres_password", "changeme")
        if isinstance(password, SecretStr):
            password = password.get_secret_value()
        return (
            f"postgresql+asyncpg://{d.get('postgres_user', 'keystone')}"
            f":{password}"
            f"@{d.get('postgres_host', 'postgres')}"
            f":{d.get('postgres_port', 5432)}"
            f"/{d.get('postgres_db', 'keystone')}"
        )

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"  # Redis/Valkey URL for events, rate limits and the concurrency limiter
    redis_password: SecretStr | None = None  # Redis/Valkey password, embedded into REDIS_URL by `keystone init`

    # ── Keystone Inference — vLLM endpoints ─────────────────────────
    # OpenAI-compatible base URL serving the `coding` role (vLLM, llama.cpp, RunPod, the frontier proxy)
    vllm_coding_url: str = "http://vllm-coding:8000/v1"
    vllm_coding_fallback_url: str = "http://vllm-coding-fallback:8000/v1"  # base URL serving the `coding_fallback` role
    vllm_reasoning_url: str = "http://vllm-reasoning:8000/v1"  # base URL serving the `reasoning` role

    # Optional per-role bearer token, sent as `Authorization: Bearer ...` on
    # every request to that role's endpoint. A self-hosted in-cluster vLLM
    # needs none; a hosted OpenAI-compatible endpoint (RunPod Serverless's
    # worker-vllm, for instance) requires one.
    vllm_coding_api_key: SecretStr | None = None  # bearer token for the coding endpoint, if it needs one
    vllm_coding_fallback_api_key: SecretStr | None = None  # bearer token for the coding_fallback endpoint
    vllm_reasoning_api_key: SecretStr | None = None  # bearer token for the reasoning endpoint

    # ── RunPod Serverless (see docs/deployment/RUNPOD_SETUP.md) ───────
    # RUNPOD_API_KEY authenticates both the endpoint's OpenAI-compatible
    # base URL (https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1) and its
    # management API. Point a role at the endpoint by setting that role's
    # VLLM_*_URL to the base URL and VLLM_*_API_KEY to this same key.
    runpod_api_key: SecretStr | None = None  # RunPod API key (`keystone deploy runpod-serverless`, `keystone doctor`)
    runpod_endpoint_url: str | None = None  # the RunPod Serverless endpoint's OpenAI-compatible base URL, once created

    # GLM-5.3-Flash (zai-org, MIT) — primary coding model as of the Phase 0
    # model refresh; Qwen2.5-Coder-32B-Instruct moved to coding_fallback_model_id
    # (src/inference/model_router.py's fallback chain), used when the primary
    # endpoint is unhealthy. See src/inference/config.py's coding_model_config()
    # docstring for the real verified footprint/tool-parser details.
    # HF id of the model behind `coding` — metadata for /v1/models and the Model Library, not routing
    coding_model_id: str = "zai-org/GLM-5.3-Flash"
    coding_fallback_model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"  # HF id of the model behind `coding_fallback`
    reasoning_model_id: str = "deepseek-ai/DeepSeek-R1"  # HF id of the model behind `reasoning`

    # When a task is submitted with model="auto" and classify_task_to_role
    # resolves it to "coding", this additionally checks task complexity
    # (src/inference/model_router.py's classify_task_complexity) and routes
    # a "simple" task to coding_fallback instead of the primary model.
    # Default off: this changes which model actually serves an "auto" task,
    # so it's an explicit opt-in rather than a silent behavior change.
    task_complexity_routing_enabled: bool = False

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_host: str = "qdrant"  # Qdrant host for the code/memory vector store
    qdrant_port: int = 6333  # Qdrant port
    qdrant_api_key: SecretStr | None = None  # Qdrant API key
    qdrant_collection: str = "keystone_code_memory"  # collection name for code chunks and memories
    qdrant_use_tls: bool = False  # internal cluster traffic; flip on once mTLS is wired up (see OpenBao workstream)

    # ── Embedding ─────────────────────────────────────────────
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"  # sentence-embedding model for retrieval
    embedding_model_path: str | None = None  # local path override for air-gapped deployments
    embedding_dimension: int = 1024  # vector size of the embedding model

    # ── Sandbox (self-hosted — src/sandbox/) ───────────────────
    sandbox_backend: str = "gvisor"  # "gvisor" | "firecracker"
    sandbox_daemon_url: str = "http://sandbox-daemon:9000"  # the sandbox daemon every agent command runs through
    sandbox_timeout_seconds: int = 300  # default wall clock for one sandbox command
    sandbox_max_concurrent: int = 0  # per-tenant concurrent sandboxes; 0 = no cap (host capacity is the limit)
    allow_unsandboxed_dev: bool = (
        False  # run agent commands on the host without a sandbox (development only; refused in production)
    )

    # ── Git (internal Gitea in air-gapped mode) ────────────────
    # the only hosts the agent may clone from or push to (SSRF guard); JSON list
    git_allowed_hosts: list[str] = Field(default_factory=lambda: ["gitea.internal.keystone.local"])
    git_host_kind: str = "gitea"  # selects the GitHost implementation (src/git/host.py) — "gitea" only, for now
    git_host_api_url: str = "http://gitea.internal.keystone.local:3000/api/v1"  # the git host's REST API base
    git_host_token: SecretStr | None = None  # bot account token used to open PRs on the agent's behalf
    git_host_commit_author_name: str = "Keystone Agents"  # author on the agent's commits
    git_host_commit_author_email: str = "keystone-agents@keystone.local"  # author email on the agent's commits
    pr_poll_interval_seconds: int = 300  # src/orchestrator/pr_polling.py's periodic merged/rejected feedback check

    # ── Package mirrors (internal, air-gap-safe — src/orchestrator/repo_profile.py) ──
    pip_index_url: str | None = None  # e.g. https://pypi.internal.keystone.local/simple
    npm_registry_url: str | None = None  # e.g. https://npm.internal.keystone.local
    go_proxy_url: str | None = None  # e.g. https://goproxy.internal.keystone.local

    # ── Token Budgets ─────────────────────────────────────────
    # Every budget here is 0 = no limit by default. Capacity is bounded by what you
    # deploy (GPUs, workers, sandboxes), not by a policy number; set a positive value
    # only to impose a real spend or safety cap for a tenant, a task, or a request.
    default_daily_token_limit: int = 0  # tokens/day per new tenant (0 = unlimited)
    default_monthly_token_limit: int = 0  # tokens/month per new tenant (0 = unlimited)
    default_requests_per_minute: int = 0  # gateway request rate per API key (0 = unlimited)
    max_agent_iterations: int = (
        15  # the one per-task SAFETY bound (a runaway fix/test loop); per-task override, no ceiling
    )
    max_tokens_per_task: int = 0  # tokens one agent task may spend (0 = unlimited)
    max_tokens_per_request: int = 0  # ceiling on a completion request's max_tokens (0 = the model's own limit)

    # ── Agent time bounds (SAFETY, not capacity — generous, configurable, per-task) ──
    agent_test_timeout_seconds: int = 3_600  # the repo's full test suite
    agent_install_timeout_seconds: int = 3_600  # the repo's dependency install
    agent_quality_timeout_seconds: int = 1_800  # each lint/typecheck/security tool
    agent_tool_timeout_seconds: int = 1_800  # ceiling for run_command's own timeout argument
    agent_max_wall_clock_seconds: int = 0  # whole-task wall clock (0 = none; max_iterations is the bound)

    # ── Inference endpoint health (src/inference/health.py) ──────────
    inference_health_cache_seconds: int = 10  # reuse a /models probe result this long
    inference_breaker_failure_threshold: int = 3  # consecutive failures that open a role's breaker (0 = never)
    inference_breaker_open_seconds: int = 30  # how long an open breaker skips the role before one retry

    # ── Usage ledger pricing (src/billing/ledger.py) ─────────────────
    # USD per 1,000,000 tokens per model role, from your own GPU economics (see
    # benchmarks/cost_model.py for the calculator). JSON, e.g.
    #   MODEL_PRICES_PER_MILLION='{"coding": 0.40, "coding_fallback": 0.20, "reasoning": 1.20}'
    # Unpriced roles are recorded with cost 0 and the usage API says pricing is not configured.
    model_prices_per_million: dict[str, float] = Field(default_factory=dict)

    # ── Coding agent behaviour (src/orchestrator/nodes/) ──────────
    # Which quality-gate tools send a task back to fixing on a finding (nodes/quality.py).
    # bandit (high/medium) and mypy by default; add "ruff", "eslint", "tsc", "go", "cargo"
    # to fail on lint too. Per-task override: AgentTaskRequest.quality_blocking_tools.
    quality_gate_blocking_tools: list[str] = Field(default_factory=lambda: ["bandit", "mypy"])
    # Token budget for the coding loop's own conversation (context.py's trimming/summarising);
    # keep headroom under the serving model's max_model_len for the 8192-token completion.
    agent_max_context_tokens: int = 24_000
    # Stream each coding turn (tool calls + text accumulated from deltas) instead of one
    # blocking request — live progress in the trace and no idle read-timeout on long turns.
    # Set false for a backend that cannot stream tool calls.
    agent_stream_turns: bool = True

    # ── Gateway behaviour ───────────────────────────────────────
    # Log every gateway prompt and reply (through PII redaction) — off by default: prompts are your users' data.
    log_prompts: bool = False
    # model="auto" on the gateway routes a simple last message to coding_fallback (cheaper) instead of coding;
    # the decision is returned in X-VS-Route-Decision. Off by default: a wrong downgrade costs quality.
    gateway_cost_routing: bool = False

    # ── Temporal ──────────────────────────────────────────────
    temporal_host: str = "temporal:7233"  # Temporal server address for durable agent/fine-tune workflows
    temporal_namespace: str = "keystone"  # Temporal namespace
    temporal_task_queue: str = "keystone-agent-tasks"  # task queue the Keystone worker polls

    # ── TLS ────────────────────────────────────────────────────
    tls_cert_path: str | None = None  # TLS certificate for the API (`make certs` / `make certs-ca`)
    tls_key_path: str | None = None  # TLS private key

    # ── Observability ─────────────────────────────────────────
    prometheus_port: int = 9090  # port the /metrics scrape target listens on
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_format: str = "json"  # json (machine-readable) or console

    # ── GitHub ────────────────────────────────────────────────
    github_app_id: str | None = None  # GitHub App id (GitHub host support; Gitea is the verified path today)
    github_private_key_path: str | None = None  # GitHub App private key file
    github_webhook_secret: str | None = None  # GitHub webhook secret

    # ── Fine-Tuning ───────────────────────────────────────────
    wandb_api_key: str | None = None  # Weights & Biases key for training-run logging (optional)
    # where adapters and the lora_modules.json manifest are written; mount it into the serving pods
    finetuning_output_dir: str = "/data/finetuning/output"
    finetuning_data_dir: str = "/data/finetuning/data"  # where guided fine-tunes write their datasets and manifests
    hf_token: str | None = None  # Hugging Face token for gated model downloads

    # ── Derived helpers ───────────────────────────────────────
    @property
    def secret_key_is_ephemeral(self) -> bool:
        """True when VS_SECRET_KEY was not supplied and a random one was generated for this process —
        API-key hashes (ADR 0004) would then be unverifiable after a restart."""
        return "vs_secret_key" not in self.model_fields_set

    @property
    def is_production(self) -> bool:
        return self.vs_env == Environment.PRODUCTION

    @property
    def model_endpoint_map(self) -> dict[str, str]:
        """Map model role → vLLM base URL."""
        return {
            "coding": self.vllm_coding_url,
            "coding_fallback": self.vllm_coding_fallback_url,
            "reasoning": self.vllm_reasoning_url,
        }

    @property
    def model_id_map(self) -> dict[str, str]:
        """Map model role → HuggingFace model identifier."""
        return {
            "coding": self.coding_model_id,
            "coding_fallback": self.coding_fallback_model_id,
            "reasoning": self.reasoning_model_id,
        }

    @property
    def model_api_key_map(self) -> dict[str, str | None]:
        """Map model role → bearer token for that role's endpoint (None when the endpoint needs none)."""

        def _plain(value: SecretStr | None) -> str | None:
            return value.get_secret_value() if value is not None else None

        return {
            "coding": _plain(self.vllm_coding_api_key),
            "coding_fallback": _plain(self.vllm_coding_fallback_api_key),
            "reasoning": _plain(self.vllm_reasoning_api_key),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
