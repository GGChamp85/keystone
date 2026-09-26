"""
Keystone — API response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ── Inference Responses ───────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int = 0
    message: dict[str, str]
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    """OpenAI-compatible response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class StreamDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class ModelInfo(BaseModel):
    """OpenAI-compatible model entry. `id` is the role clients pass as
    `model` in /v1/chat/completions (coding/coding_fallback/reasoning);
    `root` is the underlying HF model identifier being served, for
    operator visibility."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "keystone"
    root: str


class ModelListResponse(BaseModel):
    """OpenAI-compatible GET /v1/models response."""

    object: str = "list"
    data: list[ModelInfo]


# ── API Key Responses ─────────────────────────────────────────


class TenantResponse(BaseModel):
    id: UUID
    name: str
    email: str
    tier: str
    daily_token_limit: int
    monthly_token_limit: int
    max_concurrent_agents: int
    monthly_budget_usd: float = 0
    is_active: bool
    created_at: datetime


class APIKeyCreatedResponse(BaseModel):
    """Returned only once — the full key is never shown again."""

    id: UUID
    name: str
    key: str = Field(..., description="Full API key — save this, it will not be shown again")
    key_prefix: str
    scopes: list[str]
    expires_at: datetime | None
    created_at: datetime


class APIKeyInfoResponse(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    status: str
    scopes: list[str]
    user_id: UUID | None = None
    last_used_at: datetime | None
    expires_at: datetime | None
    created_at: datetime


class UserResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


# ── Usage Responses ───────────────────────────────────────────


class TokenBudgetResponse(BaseModel):
    tenant_id: UUID
    daily_limit: int
    daily_used: int
    daily_remaining: int | None  # None = unlimited
    monthly_limit: int
    monthly_used: int
    monthly_remaining: int | None  # None = unlimited
    percent_daily_used: float
    percent_monthly_used: float


class UsageRecordResponse(BaseModel):
    date: datetime
    model_role: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int
    estimated_cost_usd: float


# ── Agent Task Responses ──────────────────────────────────────


class AgentTaskResponse(BaseModel):
    id: UUID
    user_id: UUID | None = None
    status: str
    task_description: str
    current_step: int
    max_steps: int
    model_role: str
    iteration_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    result_summary: str | None
    output_diff: str | None
    output_files: list[str]
    error_message: str | None
    execution_trace: list[dict[str, Any]]
    cost_breakdown: dict[str, Any]
    sandbox_id: str | None
    temporal_workflow_id: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class AgentTaskSummaryResponse(BaseModel):
    """Lean per-row shape for GET /v1/keystone/tasks — the team task list.
    Omits execution_trace/output_diff (can be large) that AgentTaskResponse
    carries for a single task's detail view."""

    id: UUID
    user_id: UUID | None
    status: str
    task_description: str
    repository_url: str | None
    branch: str
    branch_name: str | None
    pr_url: str | None
    pr_number: int | None
    model_role: str
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class AgentTaskSubmittedResponse(BaseModel):
    task_id: UUID
    status: str = "pending"
    message: str = "Task submitted to Keystone Agents"
    temporal_workflow_id: str | None = None


class TaskFeedbackResponse(BaseModel):
    id: UUID
    task_id: UUID
    user_id: str | None
    verdict: str
    reason: str | None
    created_at: datetime


# ── Health ────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    platform: str = "Keystone"
    products: list[str] = ["Keystone Inference", "Keystone Agents"]
    version: str = "1.0.0"
    components: dict[str, str] = {}


# ── Memory ────────────────────────────────────────────────────


class MemoryResponse(BaseModel):
    id: UUID
    repository: str | None
    scope: str
    kind: str
    content: str
    source: str
    status: str
    pinned: bool
    confidence: float
    hit_count: int
    created_by: str | None


class SkillResponse(BaseModel):
    id: UUID
    name: str
    content: str
    trigger_keywords: list[str]
    enabled: bool
    created_by: str | None


# ── Fine-Tuning ──────────────────────────────────────────────


class FineTuneJobResponse(BaseModel):
    id: UUID
    base_model: str
    job_type: str
    status: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    output_model_path: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    error_message: str | None
