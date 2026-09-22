"""uncap tenant concurrency and token budgets by default

`tenants.max_concurrent_agents` used to default to 3 with no API to change
it, so every tenant was silently limited to three concurrent agent tasks
and the fourth was rejected with HTTP 429; `daily_token_limit` (5,000,000)
and `monthly_token_limit` (100,000,000) likewise applied to every tenant
that did not override them — policy numbers standing in for capacity.
The meaning of all three columns changes: 0 = no limit (now the default);
a positive value is a deliberate admin choice.

Existing rows still holding exactly the old default become 0. Any other
value was chosen on purpose and is left alone. No DDL.

Revision ID: 6a1c2f9e0b3d
Revises: 519d4719f677
Create Date: 2026-09-22 16:20:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6a1c2f9e0b3d"
down_revision: str | None = "519d4719f677"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_tenants = sa.table(
    "tenants",
    sa.column("max_concurrent_agents", sa.Integer()),
    sa.column("daily_token_limit", sa.BigInteger()),
    sa.column("monthly_token_limit", sa.BigInteger()),
)

# (column, old default that meant "nobody chose this", new meaning: 0 = unlimited)
_OLD_DEFAULTS = (
    ("max_concurrent_agents", 3),
    ("daily_token_limit", 5_000_000),
    ("monthly_token_limit", 100_000_000),
)


def upgrade() -> None:
    for column, old_default in _OLD_DEFAULTS:
        col = _tenants.c[column]
        op.execute(sa.update(_tenants).where(col == old_default).values({column: 0}))


def downgrade() -> None:
    for column, old_default in _OLD_DEFAULTS:
        col = _tenants.c[column]
        op.execute(sa.update(_tenants).where(col == 0).values({column: old_default}))
