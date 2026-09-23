"""
Keystone — API request schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ── Inference Requests ────────────────────────────────────────


class ChatMessage(BaseModel):
    """One OpenAI chat message. `content` is a string, OpenAI content parts, or null (an assistant turn that
    only made tool calls); `tool_calls` on an assistant turn and `tool_call_id` on a `tool` turn carry a
    tool-use round trip through the gateway unchanged."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


def _normalize_model(v: str) -> str:
    shortcuts = {"code": "coding", "reason": "reasoning", "qwen": "coding_fallback"}
    return shortcuts.get(v.lower(), v.lower())


class CompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request."""

    model: str = Field(
        ...,
        description="Model role: 'coding', 'coding_fallback', 'reasoning', or a full model id",
    )
    messages: list[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    stream_options: dict[str, Any] | None = Field(
        default=None,
        description="OpenAI's {'include_usage': true} — when set, the final streamed chunk carries real token usage.",
    )
    stop: list[str] | None = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="OpenAI function-calling tools, passed to the backend unchanged."
    )
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = Field(
        default=None,
        description="{'type': 'json_schema', ...} or {'type': 'json_object'} (structured output); "
        "not combinable with tools in one request.",
    )

    @field_validator("model")
    @classmethod
    def normalize_model(cls, v: str) -> str:
        return _normalize_model(v)


class MessagesRequest(BaseModel):
    """Anthropic Messages API request (`POST /v1/messages`) — text, tools and streaming; see
    src/inference/anthropic_compat.py for what is translated and what is refused."""

    model: str = Field(..., description="Model role: 'coding', 'coding_fallback', 'reasoning', or a full model id")
    messages: list[dict[str, Any]]
    max_tokens: int = Field(..., ge=1, le=131072)
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("model")
    @classmethod
    def normalize_model(cls, v: str) -> str:
        return _normalize_model(v)


class CountTokensRequest(BaseModel):
    """`POST /v1/messages/count_tokens`."""

    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


# ── API Key Requests ──────────────────────────────────────────


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    tier: Literal["free", "pro", "enterprise"] = "free"
    daily_token_limit: int | None = None
    monthly_token_limit: int | None = None
    monthly_budget_usd: float = Field(default=0, ge=0, description="USD per calendar month; 0 (default) = no budget")
    max_concurrent_agents: int = Field(
        default=0,
        ge=0,
        description="Cap on concurrently running agent tasks for this tenant. 0 (default) = no cap — "
        "throughput is bounded by deployed worker/sandbox/GPU capacity. A positive value is enforced "
        "with per-user fairness (no one user may hold more than half of it).",
    )


class SetTenantLimitsRequest(BaseModel):
    """`POST /v1/admin/tenants/{id}/limits` — every field optional; 0 = no limit. Only the fields sent change."""

    daily_token_limit: int | None = Field(default=None, ge=0)
    monthly_token_limit: int | None = Field(default=None, ge=0)
    max_concurrent_agents: int | None = Field(default=None, ge=0)
    monthly_budget_usd: float | None = Field(
        default=None, ge=0, description="USD per calendar month from the priced ledger; 0 = no budget"
    )


class RotateAPIKeyRequest(BaseModel):
    """`POST /v1/admin/tenants/{id}/keys/{prefix}/rotate` — the old key keeps working for `grace_hours`."""

    grace_hours: int = Field(default=24, ge=0, le=24 * 30, description="0 = revoke the old key immediately")


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: list[str] = Field(default=["inference", "agent"])
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)
    daily_token_limit_override: int | None = None
    rate_limit_override: int | None = Field(
        default=None, ge=0, description="Requests per minute for this key; 0 = unlimited"
    )


class RevokeAPIKeyRequest(BaseModel):
    key_prefix: str = Field(..., description="The vs-xxxx prefix shown at creation time")


class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    role: Literal["admin", "lead", "developer"] = "developer"


class LinkAPIKeyToUserRequest(BaseModel):
    user_id: str = Field(..., description="A user id from POST /v1/admin/tenants/{id}/users")


# ── Agent Task Requests ───────────────────────────────────────


class AgentTaskRequest(BaseModel):
    """Submit a coding task to Keystone Agents."""

    task: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="Natural language description of the coding task",
    )
    repository_url: str | None = Field(
        default=None,
        description="Git repository URL to work on",
    )
    branch: str = Field(default="main")
    file_paths: list[str] = Field(
        default=[],
        description="Specific files the agent should focus on",
    )
    model: Literal["coding", "coding_fallback", "reasoning", "auto"] = Field(
        default="coding",
        description="Which model to use as primary coder — 'auto' classifies the task "
        "description (src/inference/model_router.py's classify_task_to_role/"
        "classify_task_complexity) and resolves to a concrete role before the "
        "task is persisted, rather than staying a live routing decision",
    )
    max_iterations: int = Field(
        default=15,
        ge=1,
        description="Safety bound on plan/code/test/fix iterations for this task (a runaway loop stops here). "
        "No ceiling — set what the task needs.",
    )
    enable_reasoning_review: bool = Field(
        default=True,
        description="Use DeepSeek-R1 as a critic to review code before finalization",
    )
    enable_sandbox_testing: bool = Field(
        default=True,
        description="Run code in self-hosted sandbox to verify correctness",
    )
    context_files: dict[str, str] | None = Field(
        default=None,
        description="Extra context files as {filename: content}",
    )
    quality_blocking_tools: list[str] | None = Field(
        default=None,
        description="Quality-gate tools whose findings send the task back to fixing. Default: the server's "
        "QUALITY_GATE_BLOCKING_TOOLS (bandit, mypy). Add 'ruff', 'eslint', 'tsc', 'go', 'cargo' to be stricter.",
    )
    best_of_n: int | None = Field(
        default=None,
        ge=1,
        description="Run N independent coding attempts and keep the one that passes the repo's own lint and "
        "related tests best (multiplies model spend by N). Default: the server's AGENT_BEST_OF_N (1 = off).",
    )


class TaskCostEstimateRequest(BaseModel):
    """`POST /v1/keystone/tasks/estimate` — the same shape a real submission would use, no task created."""

    task: str = Field(..., min_length=10, max_length=50000)
    repository_url: str | None = Field(
        default=None, description="Only affects the token floor, not the estimate source"
    )
    model: Literal["coding", "coding_fallback", "reasoning", "auto"] = Field(default="coding")
    best_of_n: int | None = Field(default=None, ge=1)


# ── Fine-Tuning Requests ─────────────────────────────────────


class GPUSpec(BaseModel):
    name: str
    vram_gb: float = Field(..., gt=0)


class GuidedPlanRequest(BaseModel):
    """Describe a fine-tune in plain terms; the server builds the dataset, picks the method and returns a plan."""

    repositories: list[str] = Field(..., min_length=1, description="Allow-listed git URLs to learn from")
    goal: str = Field(..., min_length=10, max_length=2000, description="What the adapter should get better at")
    base_model: str = Field(default="auto", description="A catalog model id, or 'auto' for the default SLM")
    holdout_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    epochs: int = Field(default=1, ge=1)
    lora_r: int | None = Field(default=None, ge=1)
    gpus: list[GPUSpec] | None = Field(default=None, description="Target hardware; omitted = detect on this host")
    gpu_hourly_cost_usd: float | None = Field(default=None, ge=0, description="Your GPU price, for the cost line")


class ApprovePlanRequest(BaseModel):
    force: bool = Field(default=False, description="Start even if the planner says it does not fit")


class StartFineTuneRequest(BaseModel):
    base_model: str = Field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="HuggingFace model to fine-tune (the catalog default SLM; see GET /v1/finetune/catalog)",
    )
    job_type: Literal["lora", "sft", "dpo"] = "lora"
    training_data_path: str = Field(..., description="Path to training data within the mounted volume")
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.03,
            "max_seq_length": 8192,
            "fp16": True,
        }
    )


# ── Memory ────────────────────────────────────────────────────


class CreateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    kind: Literal["convention", "preference", "fact", "avoid"] = "fact"
    repository: str | None = Field(
        default=None, description="Repo clone URL for a repo-scope memory; omit for tenant-scope"
    )
    pinned: bool = False


class RecallMemoryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    repository: str | None = None
    budget_tokens: int = Field(default=2000, ge=1, le=32_000)


class SubmitTaskFeedbackRequest(BaseModel):
    verdict: Literal["accepted", "rejected", "merged", "reverted"]
    reason: str | None = None


# ── Codebase Ingestion ────────────────────────────────────────


class IngestRepositoryRequest(BaseModel):
    repository_url: str = Field(..., description="Git clone URL")
    branch: str = "main"
    file_extensions: list[str] = Field(
        default=[".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"],
        description="File extensions to index",
    )
    max_file_size_kb: int = Field(default=500, description="Skip files larger than this")
    chunk_size: int = Field(default=1500, description="Characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
