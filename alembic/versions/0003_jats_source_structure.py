"""RES-133: JATS source structure — canonical paragraph + citation source anchor.

Extends the sealed RES-131 canonical model (0002, untouched) with the two
schema additions JATS source-structure preservation requires:

* ``paragraph`` — the immutable source-text unit. Identity is the
  deterministic ``paragraph_key`` (SHA-256 of the document version semantic
  key plus the stable source anchor); ``(document_version_id, source_anchor)``
  and ``(document_version_id, ordinal)`` uniqueness prevent duplicate
  canonical paragraphs; the composite foreign key forces the owning section
  to belong to the same ``document_version``; the region CHECK restricts the
  column to the source-derived JATS regions; the generic immutability trigger
  makes the table append-only like every other canonical table.
* ``citation.source_anchor`` — the stable source location of the reference
  inside the source XML. The RES-131 citation contract already carried
  ``source_reference_id`` (the source's own ``@id`` when present) but no
  deterministic location for references whose source lacks an ``@id``.

No RES-131 table, constraint or column is modified: 0002 stays exactly as
sealed, and downgrading to it removes only what 0003 added.

Revision ID: 0003_jats_source_structure
Revises: 0002_canonical_document_model
Create Date: 2026-09-28T21:20:00.000000Z
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_jats_source_structure"
down_revision: str | None = "0002_canonical_document_model"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_HEX64 = "~ '^[0-9a-f]{64}$'"

_PARAGRAPH_TABLE = "paragraph"
"""The one canonical table this revision adds."""

_ROW_CREATED_AT = sa.Column(
    "row_created_at",
    sa.DateTime(timezone=True),
    nullable=False,
    server_default=sa.func.now(),
)


def upgrade() -> None:
    """Apply this revision: add the paragraph table and the citation
    source-anchor column."""

    op.create_table(
        _PARAGRAPH_TABLE,
        sa.Column("id", sa.Uuid, nullable=False),
        sa.Column("document_version_id", sa.Uuid, nullable=False),
        sa.Column("section_id", sa.Uuid, nullable=True),
        sa.Column("section_document_version_id", sa.Uuid, nullable=True),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("region", sa.Text, nullable=False),
        sa.Column("source_anchor", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("content_sha256", sa.Text, nullable=False),
        sa.Column("paragraph_key", sa.Text, nullable=False),
        _ROW_CREATED_AT,
        sa.PrimaryKeyConstraint("id", name="pk_paragraph"),
        sa.UniqueConstraint("paragraph_key", name="uq_paragraph_paragraph_key"),
        sa.UniqueConstraint(
            "document_version_id",
            "source_anchor",
            name="uq_paragraph_document_version_source_anchor",
        ),
        sa.UniqueConstraint(
            "document_version_id", "ordinal", name="uq_paragraph_document_version_ordinal"
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_paragraph_document_version",
        ),
        sa.ForeignKeyConstraint(
            ["section_id", "section_document_version_id"],
            ["section.id", "section.document_version_id"],
            name="fk_paragraph_section",
        ),
        sa.CheckConstraint(
            "(section_id IS NULL) = (section_document_version_id IS NULL)",
            name="ck_paragraph_section_pair",
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_paragraph_ordinal_nonnegative"),
        sa.CheckConstraint(
            "region IN ('front', 'body', 'back')", name="ck_paragraph_region_source_derived"
        ),
        sa.CheckConstraint(f"content_sha256 {_HEX64}", name="ck_paragraph_content_sha256_hex"),
    )
    op.create_index("ix_paragraph_document_version_id", _PARAGRAPH_TABLE, ["document_version_id"])
    op.create_index("ix_paragraph_section_id", _PARAGRAPH_TABLE, ["section_id"])

    op.add_column("citation", sa.Column("source_anchor", sa.Text, nullable=True))

    op.execute(
        f"""
        CREATE TRIGGER trg_{_PARAGRAPH_TABLE}_immutable
        BEFORE UPDATE OR DELETE ON {_PARAGRAPH_TABLE}
        FOR EACH ROW
        EXECUTE FUNCTION dynamisrag_enforce_immutable();
        """
    )


def downgrade() -> None:
    """Revert this revision: remove the paragraph table, its trigger and the
    citation source-anchor column, leaving the sealed 0002 model intact."""
    op.execute(f"DROP TRIGGER IF EXISTS trg_{_PARAGRAPH_TABLE}_immutable ON {_PARAGRAPH_TABLE}")
    op.drop_table(_PARAGRAPH_TABLE)
    op.drop_column("citation", "source_anchor")
