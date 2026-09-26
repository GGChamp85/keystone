"""add task_schedules table

Revision ID: c303ddf4ad32
Revises: 1e9b934df7d6
Create Date: 2026-09-25 19:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c303ddf4ad32"
down_revision: str | None = "1e9b934df7d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "task_schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("trigger_type", sa.Enum("CRON", "WEBHOOK", name="scheduletrigger"), nullable=False),
        sa.Column("cron_expression", sa.String(length=120), nullable=True),
        sa.Column("webhook_token", sa.String(length=64), nullable=True, unique=True),
        sa.Column("task_description", sa.Text(), nullable=False),
        sa.Column("repository_url", sa.String(length=1024), nullable=True),
        sa.Column("branch", sa.String(length=255), nullable=False, server_default="main"),
        sa.Column("model_role", sa.String(length=32), nullable=False, server_default="coding"),
        sa.Column("max_iterations", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("context_files", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_task_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_task_schedules_tenant_enabled", "task_schedules", ["tenant_id", "enabled"])


def downgrade() -> None:
    op.drop_index("ix_task_schedules_tenant_enabled", table_name="task_schedules")
    op.drop_table("task_schedules")

    # Native-Postgres-enum cleanup (same gotcha as every earlier migration that added one — see
    # 891b3b43f1db's downgrade): the table drop above doesn't drop the ENUM type itself.
    bind = op.get_bind()
    sa.Enum(name="scheduletrigger").drop(bind, checkfirst=True)
