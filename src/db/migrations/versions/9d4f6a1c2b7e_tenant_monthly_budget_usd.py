# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Tenant.monthly_budget_usd — a dollar budget per calendar month, enforced from the ledger's priced cost.
0 = no budget (the default: capacity, not policy, bounds spend unless an admin sets one).

Revision ID: 9d4f6a1c2b7e
Revises: 8c3e5f0a2b41
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "9d4f6a1c2b7e"
down_revision: str | None = "8c3e5f0a2b41"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("monthly_budget_usd", sa.Numeric(12, 4), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("tenants", "monthly_budget_usd")
