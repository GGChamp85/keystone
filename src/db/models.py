"""
Keystone — Database models for Keystone Inference & VS Code Agent.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────


class TenantTier(enum.StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(enum.StrEnum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class APIKeyStatus(enum.StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class TaskStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ModelRole(enum.StrEnum):
    CODING = "coding"
    CODING_FALLBACK = "coding_fallback"
    REASONING = "reasoning"


class MemoryScope(enum.StrEnum):
    REPO = "repo"
    TENANT = "tenant"


class MemoryKind(enum.StrEnum):
    CONVENTION = "convention"
    PREFERENCE = "preference"
    FACT = "fact"
    AVOID = "avoid"


class MemorySource(enum.StrEnum):
    AUTO = "auto"
    USER = "user"


class MemoryStatus(enum.StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    FORGOTTEN = "forgotten"


class FeedbackVerdict(enum.StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MERGED = "merged"
    REVERTED = "reverted"


class UserRole(enum.StrEnum):
    ADMIN = "admin"
    LEAD = "lead"
    DEVELOPER = "developer"


class AdapterStatus(enum.StrEnum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    RETIRED = "retired"


# ── Tenant ────────────────────────────────────────────────────


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    tier: Mapped[TenantTier] = mapped_column(SAEnum(TenantTier), default=TenantTier.FREE, nullable=False)
    # Token budgets: 0 = unlimited (the default). A positive value is enforced by the gateway
    # (src/api/middleware/rate_limiter.py) and the agent circuit breaker.
    daily_token_limit: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    monthly_token_limit: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    # 0 = no cap (the default): concurrency is bounded by deployed capacity, not by policy.
    # A positive value is enforced by src/orchestrator/concurrency.py with per-user fairness.
    max_concurrent_agents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # USD this tenant may spend per calendar month across gateway requests and agent tasks, from the ledger's
    # priced cost (MODEL_PRICES_PER_MILLION); 0 = no budget.
    monthly_budget_usd: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False, default=0)
    subscription_status: Mapped[SubscriptionStatus] = mapped_column(
        SAEnum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE, nullable=False
    )
    subscription_renews_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    api_keys: Mapped[list[APIKey]] = relationship("APIKey", back_populates="tenant", cascade="all, delete-orphan")
    usage_records: Mapped[list[UsageRecord]] = relationship(
        "UsageRecord", back_populates="tenant", cascade="all, delete-orphan"
    )
    agent_tasks: Mapped[list[AgentTask]] = relationship(
        "AgentTask", back_populates="tenant", cascade="all, delete-orphan"
    )
    users: Mapped[list[User]] = relationship("User", back_populates="tenant", cascade="all, delete-orphan")


# ── User ──────────────────────────────────────────────────────


class User(Base):
    """A real person on a tenant's team — distinct from an APIKey, which is a
    credential a user (or a service) authenticates with. Role gates
    tenant-scope memory approval, adapter promotion, and admin routes
    (src/api/middleware/auth.py's require_role)."""

    __tablename__ = "users"
    __table_args__ = (Index("ix_users_tenant", "tenant_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole), default=UserRole.DEVELOPER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="users")


# ── API Key ───────────────────────────────────────────────────


class APIKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash"),
        Index("ix_api_keys_prefix", "key_prefix"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)  # "ks-xxxx" shown to user
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    status: Mapped[APIKeyStatus] = mapped_column(SAEnum(APIKeyStatus), default=APIKeyStatus.ACTIVE, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        JSONB, default=lambda: ["inference", "agent"], nullable=False
    )  # what this key can access
    rate_limit_override: Mapped[int | None] = mapped_column(Integer, nullable=True)  # req/min override
    daily_token_limit_override: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="api_keys")
    user: Mapped[User | None] = relationship("User")


# ── Usage Tracking ────────────────────────────────────────────


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_tenant_date", "tenant_id", "date"),
        Index("ix_usage_key_date", "api_key_id", "date"),
        # The ledger's upsert bucket (src/billing/ledger.py): NULL api_key_id is one bucket, not many.
        Index(
            "uq_usage_bucket",
            "tenant_id",
            "api_key_id",
            "date",
            "model_role",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_role: Mapped[ModelRole] = mapped_column(SAEnum(ModelRole), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, default=0.0)

    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="usage_records")


# ── Agent Task ────────────────────────────────────────────────


class AgentTask(Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (Index("ix_agent_tasks_tenant_status", "tenant_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Task definition
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    repository_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    branch: Mapped[str | None] = mapped_column(String(255), default="main")
    file_paths: Mapped[list[Any] | None] = mapped_column(JSONB, default=list)  # specific files to work on
    images: Mapped[list[Any] | None] = mapped_column(JSONB, default=list)  # [{media_type, data (base64)}, ...]

    # Execution state
    status: Mapped[TaskStatus] = mapped_column(SAEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    current_step: Mapped[int | None] = mapped_column(Integer, default=0)
    max_steps: Mapped[int | None] = mapped_column(Integer, default=15)
    model_role: Mapped[ModelRole | None] = mapped_column(SAEnum(ModelRole), default=ModelRole.CODING)

    # Results
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_files: Mapped[list[Any] | None] = mapped_column(JSONB, default=list)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Token accounting
    total_prompt_tokens: Mapped[int | None] = mapped_column(BigInteger, default=0)
    total_completion_tokens: Mapped[int | None] = mapped_column(BigInteger, default=0)
    iteration_count: Mapped[int | None] = mapped_column(Integer, default=0)

    # Sandbox
    sandbox_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Git workflow (src/orchestrator/workspace.py, src/git/) — the branch
    # the agent pushed its work to and the PR/MR opened against it, if any.
    # NULL until the task actually reaches the point of committing/pushing.
    branch_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pr_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Temporal workflow
    temporal_workflow_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    temporal_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    # Full execution trace
    execution_trace: Mapped[list[Any] | None] = mapped_column(JSONB, default=list)

    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="agent_tasks")
    user: Mapped[User | None] = relationship("User")


# ── Codebase Index ────────────────────────────────────────────


class CodebaseIndex(Base):
    """Tracks which repositories/files have been ingested into Qdrant."""

    __tablename__ = "codebase_indexes"
    __table_args__ = (UniqueConstraint("tenant_id", "repository_url", "file_path", name="uq_codebase_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    repository_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256 of file content
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


# ── Audit Log ─────────────────────────────────────────────────


class CodeChunk(Base):
    """The text of one indexed chunk — the same id as its Qdrant point — with a generated tsvector, so
    retrieval can rank by full text as well as by embedding (src/memory/hybrid_search.py)."""

    __tablename__ = "code_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    repository_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tsv: Mapped[Any] = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    __table_args__ = (Index("ix_code_chunks_tenant_repo", "tenant_id", "repository_url"),)


class AuditLog(Base):
    """Records every root-admin action (tenant/key create/revoke, etc.)."""

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)  # "root_admin" or a tenant/key identifier
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "tenant.create", "api_key.revoke"
    target_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # "tenant" | "api_key"
    target_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=dict)
    source_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


# ── Fine-Tune Job ─────────────────────────────────────────────


class FineTuneJob(Base):
    __tablename__ = "finetune_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "lora", "sft", "dpo"
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict)
    output_model_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ── Model Adapter ─────────────────────────────────────────────


class ModelAdapter(Base):
    """A trained LoRA/SFT/DPO adapter, registered so src/inference/model_router.py
    can route a tenant's requests to it once promoted. `is_default` marks the
    one adapter (per tenant + base_model_id) that routing actually uses —
    application code's responsibility to keep at most one true per group,
    same as this project's other soft invariants (no DB-level partial unique
    index for it, matching e.g. AgentMemory's pinned flag)."""

    __tablename__ = "model_adapters"
    __table_args__ = (Index("ix_model_adapters_tenant_base_model", "tenant_id", "base_model_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    base_model_id: Mapped[str] = mapped_column(String(255), nullable=False)  # e.g. "zai-org/GLM-5.3-Flash"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )  # filesystem/volume path vLLM's --lora-modules points at
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "lora" | "sft" | "dpo"
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("finetune_jobs.id", ondelete="SET NULL"), nullable=True
    )
    # The benchmark run (below) that produced this adapter's verdict, if any.
    benchmark_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("benchmark_runs.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[AdapterStatus] = mapped_column(
        SAEnum(AdapterStatus), default=AdapterStatus.CANDIDATE, nullable=False
    )
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict
    )  # eval_loss, benchmark pass rate, etc. — whatever produced the verdict
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    tenant: Mapped[Tenant] = relationship("Tenant")
    job: Mapped[FineTuneJob | None] = relationship("FineTuneJob")


# ── Benchmark Run ─────────────────────────────────────────────


class BenchmarkRun(Base):
    """One repo-task benchmark result (benchmarks/agent_runner.py --persist): which task, which backend
    (a label such as "coding", "frontier", "base+rag", or an adapter name), whether the held-out
    fail_to_pass/pass_to_pass tests passed on the pushed branch, and what it cost. The persisted,
    reproducible evidence behind docs/benchmarks/latest.md (benchmarks/report.py)."""

    __tablename__ = "benchmark_runs"
    __table_args__ = (Index("ix_benchmark_runs_suite_backend_created", "suite", "backend_label", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    suite: Mapped[str] = mapped_column(String(50), nullable=False)  # "repo"
    task_id: Mapped[str] = mapped_column(String(255), nullable=False)  # benchmarks/tasks/repo/<id>
    backend_label: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)  # the served model behind the label
    agent_task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_tasks.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # completed | failed | timeout | ...
    solved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fail_to_pass: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    pass_to_pass: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pr_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)  # the full runner result
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


# ── Agent Memory ──────────────────────────────────────────────


class AgentMemory(Base):
    """
    A learned fact/convention/preference/thing-to-avoid, layered repo-over-
    tenant (src/memory/store.py's `recall()` — a repo-scope memory always
    outranks a tenant-scope one on the same topic). `repository` is NULL
    for tenant-scope memories, the repo's clone URL for repo-scope ones.
    """

    __tablename__ = "agent_memories"
    __table_args__ = (
        Index("ix_agent_memories_tenant_repo", "tenant_id", "repository"),
        Index("ix_agent_memories_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    repository: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    scope: Mapped[MemoryScope] = mapped_column(SAEnum(MemoryScope), nullable=False)
    kind: Mapped[MemoryKind] = mapped_column(SAEnum(MemoryKind), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[MemorySource] = mapped_column(SAEnum(MemorySource), default=MemorySource.USER, nullable=False)
    status: Mapped[MemoryStatus] = mapped_column(SAEnum(MemoryStatus), default=MemoryStatus.APPROVED, nullable=False)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_by: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # a user identifier, or "agent" for auto-proposed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


# ── Task Feedback ─────────────────────────────────────────────


class TaskFeedback(Base):
    """A human's accept/reject/merge/revert verdict on a completed agent task — the raw signal
    src/memory/extract.py turns into proposed memories (e.g. a rejection reason -> an `avoid` memory)."""

    __tablename__ = "task_feedback"
    __table_args__ = (Index("ix_task_feedback_task", "task_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    verdict: Mapped[FeedbackVerdict] = mapped_column(SAEnum(FeedbackVerdict), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
