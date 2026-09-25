"""add images column to agent_tasks

Revision ID: b1797962365e
Revises: b3c7d9e1f2a4
Create Date: 2026-09-24 08:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b1797962365e"
down_revision: str | None = "b3c7d9e1f2a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_tasks",
        sa.Column("images", postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("agent_tasks", "images")
