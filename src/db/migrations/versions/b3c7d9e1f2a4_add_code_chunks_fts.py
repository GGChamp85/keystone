# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""code_chunks: the text of every indexed chunk in Postgres with a generated tsvector and a GIN index, so
retrieval can combine Qdrant's vector similarity with full-text ranking (src/memory/hybrid_search.py).
One row per Qdrant point (same id), written and deleted by the ingestion pipeline alongside the point.

Revision ID: b3c7d9e1f2a4
Revises: 9d4f6a1c2b7e
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b3c7d9e1f2a4"
down_revision: str | None = "9d4f6a1c2b7e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "code_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("repository_url", sa.String(1024), nullable=False),
        sa.Column("file_path", sa.String(2048), nullable=False),
        sa.Column("language", sa.String(50), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_code_chunks_tenant_repo", "code_chunks", ["tenant_id", "repository_url"])
    op.create_index("ix_code_chunks_tsv", "code_chunks", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_code_chunks_tsv", table_name="code_chunks")
    op.drop_index("ix_code_chunks_tenant_repo", table_name="code_chunks")
    op.drop_table("code_chunks")
