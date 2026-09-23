"""add benchmark_runs, and the real FK model_adapters.benchmark_run_id has waited for

`benchmark_runs` is the persisted, reproducible record behind every
number in docs/benchmarks/latest.md: one row per repo task per backend
label per run, with the held-out test outcomes, tokens, duration and the
full runner result. `model_adapters.benchmark_run_id` was a plain UUID
column "until the table exists"; it now references it (SET NULL on delete).

Revision ID: 8c3e5f0a2b41
Revises: 7b2d4e8f1a90
Create Date: 2026-09-22 18:10:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "8c3e5f0a2b41"
down_revision: str | None = "7b2d4e8f1a90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("suite", sa.String(50), nullable=False),
        sa.Column("task_id", sa.String(255), nullable=False),
        sa.Column("backend_label", sa.String(255), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=True),
        sa.Column(
            "agent_task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("solved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fail_to_pass", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("pass_to_pass", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("prompt_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pr_url", sa.String(1024), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_benchmark_runs_suite_backend_created", "benchmark_runs", ["suite", "backend_label", "created_at"]
    )
    op.create_foreign_key(
        "fk_model_adapters_benchmark_run_id",
        "model_adapters",
        "benchmark_runs",
        ["benchmark_run_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_model_adapters_benchmark_run_id", "model_adapters", type_="foreignkey")
    op.drop_index("ix_benchmark_runs_suite_backend_created", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
