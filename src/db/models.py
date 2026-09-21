"""
Keystone — Database models for Keystone Inference & VS Code Agent.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


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


# ── Tenant ────────────────────────────────────────────────────


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    tier = Column(SAEnum(TenantTier), default=TenantTier.FREE, nullable=False)
    daily_token_limit = Column(BigInteger, nullable=False, default=5_000_000)
    monthly_token_limit = Column(BigInteger, nullable=False, default=100_000_000)
    max_concurrent_agents = Column(Integer, nullable=False, default=3)
    subscription_status = Column(SAEnum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE, nullable=False)
    subscription_renews_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    api_keys = relationship("APIKey", back_populates="tenant", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="tenant", cascade="all, delete-orphan")
    agent_tasks = relationship("AgentTask", back_populates="tenant", cascade="all, delete-orphan")
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")


# ── User ──────────────────────────────────────────────────────


class User(Base):
    """A real person on a tenant's team — distinct from an APIKey, which is a
    credential a user (or a service) authenticates with. Role gates
    tenant-scope memory approval, adapter promotion, and admin routes
    (src/api/middleware/auth.py's require_role)."""

    __tablename__ = "users"
    __table_args__ = (Index("ix_users_tenant", "tenant_id"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    role = Column(SAEnum(UserRole), default=UserRole.DEVELOPER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    tenant = relationship("Tenant", back_populates="users")


# ── API Key ───────────────────────────────────────────────────


class APIKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash"),
        Index("ix_api_keys_prefix", "key_prefix"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(255), nullable=False)
    key_prefix = Column(String(12), nullable=False)  # "ks-xxxx" shown to user
    key_hash = Column(String(128), nullable=False, unique=True)
    status = Column(SAEnum(APIKeyStatus), default=APIKeyStatus.ACTIVE, nullable=False)
    scopes = Column(JSONB, default=lambda: ["inference", "agent"])  # what this key can access
    rate_limit_override = Column(Integer, nullable=True)  # req/min override
    daily_token_limit_override = Column(BigInteger, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    tenant = relationship("Tenant", back_populates="api_keys")
    user = relationship("User")


# ── Usage Tracking ────────────────────────────────────────────


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_tenant_date", "tenant_id", "date"),
        Index("ix_usage_key_date", "api_key_id", "date"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True)
    date = Column(DateTime(timezone=True), nullable=False)
    model_role = Column(SAEnum(ModelRole), nullable=False)
    prompt_tokens = Column(BigInteger, default=0, nullable=False)
    completion_tokens = Column(BigInteger, default=0, nullable=False)
    total_tokens = Column(BigInteger, default=0, nullable=False)
    request_count = Column(Integer, default=0, nullable=False)
    estimated_cost_usd = Column(Float, default=0.0)

    tenant = relationship("Tenant", back_populates="usage_records")


# ── Agent Task ────────────────────────────────────────────────


class AgentTask(Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (Index("ix_agent_tasks_tenant_status", "tenant_id", "status"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Task definition
    task_description = Column(Text, nullable=False)
    repository_url = Column(String(1024), nullable=True)
    branch = Column(String(255), default="main")
    file_paths = Column(JSONB, default=list)  # specific files to work on

    # Execution state
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    current_step = Column(Integer, default=0)
    max_steps = Column(Integer, default=15)
    model_role = Column(SAEnum(ModelRole), default=ModelRole.CODING)

    # Results
    result_summary = Column(Text, nullable=True)
    output_diff = Column(Text, nullable=True)
    output_files = Column(JSONB, default=list)
    error_message = Column(Text, nullable=True)

    # Token accounting
    total_prompt_tokens = Column(BigInteger, default=0)
    total_completion_tokens = Column(BigInteger, default=0)
    iteration_count = Column(Integer, default=0)

    # Sandbox
    sandbox_id = Column(String(255), nullable=True)

    # Git workflow (src/orchestrator/workspace.py, src/git/) — the branch
    # the agent pushed its work to and the PR/MR opened against it, if any.
    # NULL until the task actually reaches the point of committing/pushing.
    branch_name = Column(String(255), nullable=True)
    commit_sha = Column(String(64), nullable=True)
    pr_url = Column(String(1024), nullable=True)
    pr_number = Column(Integer, nullable=True)

    # Temporal workflow
    temporal_workflow_id = Column(String(255), nullable=True)
    temporal_run_id = Column(String(255), nullable=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Full execution trace
    execution_trace = Column(JSONB, default=list)

    tenant = relationship("Tenant", back_populates="agent_tasks")
    user = relationship("User")


# ── Codebase Index ────────────────────────────────────────────


class CodebaseIndex(Base):
    """Tracks which repositories/files have been ingested into Qdrant."""

    __tablename__ = "codebase_indexes"
    __table_args__ = (UniqueConstraint("tenant_id", "repository_url", "file_path", name="uq_codebase_index"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    repository_url = Column(String(1024), nullable=False)
    file_path = Column(String(2048), nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256 of file content
    chunk_count = Column(Integer, default=0)
    language = Column(String(50), nullable=True)
    last_indexed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


# ── Audit Log ─────────────────────────────────────────────────


class AuditLog(Base):
    """Records every root-admin action (tenant/key create/revoke, etc.)."""

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_created_at", "created_at"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor = Column(String(255), nullable=False)  # "root_admin" or a tenant/key identifier
    action = Column(String(100), nullable=False)  # e.g. "tenant.create", "api_key.revoke"
    target_type = Column(String(50), nullable=True)  # "tenant" | "api_key"
    target_id = Column(String(255), nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict)
    source_ip = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


# ── Fine-Tune Job ─────────────────────────────────────────────


class FineTuneJob(Base):
    __tablename__ = "finetune_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    base_model = Column(String(255), nullable=False)
    job_type = Column(String(50), nullable=False)  # "lora", "sft", "dpo"
    status = Column(String(50), default="pending")
    config = Column(JSONB, default=dict)
    metrics = Column(JSONB, default=dict)
    output_model_path = Column(String(1024), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    error_message = Column(Text, nullable=True)


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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    repository = Column(String(1024), nullable=True)
    scope = Column(SAEnum(MemoryScope), nullable=False)
    kind = Column(SAEnum(MemoryKind), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(SAEnum(MemorySource), default=MemorySource.USER, nullable=False)
    status = Column(SAEnum(MemoryStatus), default=MemoryStatus.APPROVED, nullable=False)
    pinned = Column(Boolean, default=False, nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)
    hit_count = Column(Integer, default=0, nullable=False)
    created_by = Column(String(255), nullable=True)  # a user identifier, or "agent" for auto-proposed
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(255), nullable=True)
    verdict = Column(SAEnum(FeedbackVerdict), nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
