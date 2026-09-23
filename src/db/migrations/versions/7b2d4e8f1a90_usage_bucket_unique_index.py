"""usage_records: unique bucket index for atomic upserts

The ledger (src/billing/ledger.py) accumulates one row per
(tenant, api key, model role, hour) with `INSERT ... ON CONFLICT DO UPDATE`,
which needs a unique constraint on that bucket. `api_key_id` is nullable
(agent tasks and keyless internal calls), so the index is declared
NULLS NOT DISTINCT (PostgreSQL 15+) — two NULL keys are the same bucket.

Revision ID: 7b2d4e8f1a90
Revises: 6a1c2f9e0b3d
Create Date: 2026-09-22 17:40:00

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b2d4e8f1a90"
down_revision: str | None = "6a1c2f9e0b3d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_usage_bucket",
        "usage_records",
        ["tenant_id", "api_key_id", "date", "model_role"],
        unique=True,
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.drop_index("uq_usage_bucket", table_name="usage_records")
