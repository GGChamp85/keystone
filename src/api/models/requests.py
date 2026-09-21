"""
Keystone — API request schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ── Inference Requests ────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


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
    stop: list[str] | None = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    @field_validator("model")
    @classmethod
    def normalize_model(cls, v: str) -> str:
        shortcuts = {"code": "coding", "reason": "reasoning", "qwen": "coding_fallback"}
        return shortcuts.get(v.lower(), v.lower())


# ── API Key Requests ──────────────────────────────────────────


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    tier: Literal["free", "pro", "enterprise"] = "free"
    daily_token_limit: int | None = None
    monthly_token_limit: int | None = None


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: list[str] = Field(default=["inference", "agent"])
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)
    daily_token_limit_override: int | None = None
    rate_limit_override: int | None = Field(default=None, ge=1, le=10000, description="Requests per minute")


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
    model: Literal["coding", "coding_fallback", "reasoning"] = Field(
        default="coding",
        description="Which model to use as primary coder",
    )
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum agent loop iterations (circuit breaker)",
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


# ── Fine-Tuning Requests ─────────────────────────────────────


class StartFineTuneRequest(BaseModel):
    base_model: str = Field(
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        description="HuggingFace model to fine-tune",
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
