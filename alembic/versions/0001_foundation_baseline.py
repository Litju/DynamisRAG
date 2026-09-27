"""RES-130: establish the migration baseline on an empty database.

This revision is intentionally empty. RES-130 delivers the runtime foundation
and must not introduce the scientific schema, which is owned by RES-131. Its one
observable effect is the creation of Alembic's ``alembic_version`` bookkeeping
table, and that is exactly the evidence needed: ``alembic upgrade head``
succeeds against a database that contains no application objects.

Every later revision is hand-written, because ``alembic/env.py`` registers no
``MetaData`` while no SQLAlchemy models exist. That constraint is re-stated on
the migration that first adds models, so the omission cannot be mistaken for an
oversight.

Revision ID: 0001_foundation_baseline
Revises:
Create Date: 2026-09-27T03:36:25.897125Z
"""

from __future__ import annotations

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0001_foundation_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    No application objects exist yet; see the module docstring.
    """


def downgrade() -> None:
    """Revert this revision.

    Alembic drops ``alembic_version`` itself once the last revision is undone.
    """
