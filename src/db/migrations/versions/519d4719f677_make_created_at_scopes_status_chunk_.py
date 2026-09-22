"""make created_at, scopes, status, chunk_count not null

Every one of these columns has always had a Python-side default, and the
application code has always treated them as present (response models
declare `created_at: datetime`, the ingestion pipeline does arithmetic on
`chunk_count`, the fine-tune routes return `status: str`). Declaring them
NOT NULL makes the schema say what the code already assumes — surfaced by
converting src/db/models.py to SQLAlchemy 2.0 `Mapped[...]` types, where
a nullable column is `T | None` and mypy then flagged every place a
possibly-NULL value was handed to a non-optional field.

Each column is backfilled first so the constraint cannot fail on a row
inserted outside the ORM (the Python-side defaults never applied to those):
timestamps get now(), `scopes` gets the same default the model uses,
`status` gets "pending", `chunk_count` gets 0.

Revision ID: 519d4719f677
Revises: 52e74272c1a4
Create Date: 2026-09-22 15:32:02.977335

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "519d4719f677"
down_revision: str | None = "52e74272c1a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = postgresql.TIMESTAMP(timezone=True)

_CREATED_AT_TABLES = (
    "agent_memories",
    "agent_tasks",
    "api_keys",
    "audit_logs",
    "finetune_jobs",
    "model_adapters",
    "task_feedback",
    "tenants",
    "users",
)


def upgrade() -> None:
    for table in _CREATED_AT_TABLES:
        created_at = sa.column("created_at", _TIMESTAMP)
        op.execute(sa.update(sa.table(table, created_at)).where(created_at.is_(None)).values(created_at=sa.func.now()))
        op.alter_column(table, "created_at", existing_type=_TIMESTAMP, nullable=False)

    op.execute("""UPDATE api_keys SET scopes = '["inference", "agent"]'::jsonb WHERE scopes IS NULL""")
    op.alter_column("api_keys", "scopes", existing_type=postgresql.JSONB(astext_type=sa.Text()), nullable=False)

    op.execute("UPDATE codebase_indexes SET chunk_count = 0 WHERE chunk_count IS NULL")
    op.alter_column("codebase_indexes", "chunk_count", existing_type=sa.INTEGER(), nullable=False)

    op.execute("UPDATE finetune_jobs SET status = 'pending' WHERE status IS NULL")
    op.alter_column("finetune_jobs", "status", existing_type=sa.VARCHAR(length=50), nullable=False)


def downgrade() -> None:
    op.alter_column("finetune_jobs", "status", existing_type=sa.VARCHAR(length=50), nullable=True)
    op.alter_column("codebase_indexes", "chunk_count", existing_type=sa.INTEGER(), nullable=True)
    op.alter_column("api_keys", "scopes", existing_type=postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    for table in reversed(_CREATED_AT_TABLES):
        op.alter_column(table, "created_at", existing_type=_TIMESTAMP, nullable=True)
