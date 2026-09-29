"""RES-134: passage source spans — exact immutable Passage -> Paragraph lineage.

Extends the sealed RES-133 canonical model (0001-0003, untouched) with the
smallest provenance structure that makes passage lineage exact:

* ``passage_source_span`` — the authoritative ordered mapping from a Passage
  to the exact ``[start_char, end_char)`` ranges of the canonical normalized
  ``Paragraph.text`` it was derived from. Identity is the deterministic
  ``span_key`` (SHA-256 of the passage key plus the span's order within the
  passage); ``span_key`` and ``(passage_id, source_order)`` uniqueness make a
  repeated identical span collide instead of duplicating. Composite foreign
  keys force the referenced passage and paragraph to belong to the span's own
  ``document_version``; CHECK constraints enforce non-negative ordering and
  positive, ordered offsets; a trigger verifies ``end_char`` against the
  referenced paragraph's persisted text (a CHECK cannot express a
  cross-table bound); and the generic immutability trigger makes the table
  append-only like every other canonical table.
* ``uq_passage_id_document_version_id`` / ``uq_paragraph_id_document_version_id``
  — the unique pairs the composite span foreign keys reference. These are
  the narrow Passage/paragraph-side constraints real chunking needs; the
  sealed tables are not modified, only extended with a unique constraint.
* ``ck_passage_text_nonempty`` — chunking always produces non-empty passage
  text, so the persistence boundary enforces it for any writer.

No sealed migration is modified: downgrading to 0003 removes only what this
revision added.

Revision ID: 0004_passage_source_spans
Revises: 0003_jats_source_structure
Create Date: 2026-09-29T10:00:00.000000Z
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_passage_source_spans"
down_revision: str | None = "0003_jats_source_structure"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SPAN_TABLE = "passage_source_span"
"""The one canonical table this revision adds."""

_OFFSET_CHECK_FUNCTION = """
CREATE OR REPLACE FUNCTION dynamisrag_check_passage_source_span()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    v_passage_version_id uuid;
    v_paragraph_version_id uuid;
    v_paragraph_text text;
BEGIN
    SELECT document_version_id INTO v_passage_version_id
    FROM passage WHERE id = NEW.passage_id;
    IF NOT FOUND OR v_passage_version_id <> NEW.document_version_id THEN
        RAISE EXCEPTION
            'dynamisrag: passage_source_span % does not belong to the same document version
            as its passage',
            NEW.span_key
        USING ERRCODE = 'foreign_key_violation';
    END IF;
    SELECT document_version_id, text INTO v_paragraph_version_id, v_paragraph_text
    FROM paragraph WHERE id = NEW.paragraph_id;
    IF NOT FOUND OR v_paragraph_version_id <> NEW.document_version_id THEN
        RAISE EXCEPTION
            'dynamisrag: passage_source_span % does not belong to the same document version
            as its paragraph',
            NEW.span_key
        USING ERRCODE = 'foreign_key_violation';
    END IF;
    IF NEW.end_char > length(v_paragraph_text) THEN
        RAISE EXCEPTION
            'dynamisrag: passage_source_span % end_char % exceeds paragraph text length %',
            NEW.span_key, NEW.end_char, length(v_paragraph_text)
        USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END;
$$;
"""
"""Trigger function enforcing the cross-table invariants a CHECK cannot.

The composite foreign keys already bind passage and paragraph to the span's
document version structurally; the trigger re-states the same-version checks
self-contained and adds the offset-integrity rule ``end_char <=
length(paragraph.text)`` that no single-table constraint can express.
"""

_ROW_CREATED_AT = sa.Column(
    "row_created_at",
    sa.DateTime(timezone=True),
    nullable=False,
    server_default=sa.func.now(),
)


def upgrade() -> None:
    """Apply this revision: add the passage source-span table, its offset
    trigger, and the unique/constraint pairs the composite foreign keys
    reference."""

    # The composite span foreign keys reference these unique pairs, so they
    # must exist before the span table is created.
    op.create_unique_constraint(
        "uq_passage_id_document_version_id", "passage", ["id", "document_version_id"]
    )
    op.create_unique_constraint(
        "uq_paragraph_id_document_version_id", "paragraph", ["id", "document_version_id"]
    )
    op.create_check_constraint("ck_passage_text_nonempty", "passage", "length(text) > 0")

    op.create_table(
        _SPAN_TABLE,
        sa.Column("id", sa.Uuid, nullable=False),
        sa.Column("document_version_id", sa.Uuid, nullable=False),
        sa.Column("passage_id", sa.Uuid, nullable=False),
        sa.Column("passage_document_version_id", sa.Uuid, nullable=False),
        sa.Column("paragraph_id", sa.Uuid, nullable=False),
        sa.Column("paragraph_document_version_id", sa.Uuid, nullable=False),
        sa.Column("source_order", sa.Integer, nullable=False),
        sa.Column("start_char", sa.Integer, nullable=False),
        sa.Column("end_char", sa.Integer, nullable=False),
        sa.Column("span_key", sa.Text, nullable=False),
        _ROW_CREATED_AT,
        sa.PrimaryKeyConstraint("id", name="pk_passage_source_span"),
        sa.UniqueConstraint("span_key", name="uq_passage_source_span_span_key"),
        sa.UniqueConstraint(
            "passage_id", "source_order", name="uq_passage_source_span_passage_order"
        ),
        sa.ForeignKeyConstraint(
            ["passage_id", "passage_document_version_id"],
            ["passage.id", "passage.document_version_id"],
            name="fk_passage_source_span_passage",
        ),
        sa.ForeignKeyConstraint(
            ["paragraph_id", "paragraph_document_version_id"],
            ["paragraph.id", "paragraph.document_version_id"],
            name="fk_passage_source_span_paragraph",
        ),
        sa.CheckConstraint("source_order >= 0", name="ck_passage_source_span_order_nonnegative"),
        sa.CheckConstraint("start_char >= 0", name="ck_passage_source_span_start_nonnegative"),
        sa.CheckConstraint("end_char > 0", name="ck_passage_source_span_end_positive"),
        sa.CheckConstraint("end_char > start_char", name="ck_passage_source_span_end_after_start"),
    )
    op.create_index("ix_passage_source_span_passage_id", _SPAN_TABLE, ["passage_id"])
    op.create_index("ix_passage_source_span_paragraph_id", _SPAN_TABLE, ["paragraph_id"])

    op.execute(_OFFSET_CHECK_FUNCTION)
    op.execute(
        f"""
        CREATE TRIGGER trg_{_SPAN_TABLE}_check
        BEFORE INSERT OR UPDATE ON {_SPAN_TABLE}
        FOR EACH ROW
        EXECUTE FUNCTION dynamisrag_check_passage_source_span();
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_{_SPAN_TABLE}_immutable
        BEFORE UPDATE OR DELETE ON {_SPAN_TABLE}
        FOR EACH ROW
        EXECUTE FUNCTION dynamisrag_enforce_immutable();
        """
    )


def downgrade() -> None:
    """Revert this revision: remove the span table, its triggers and function,
    and the constraints added to the sealed passage/paragraph tables, leaving
    the 0003 model intact."""
    op.execute(f"DROP TRIGGER IF EXISTS trg_{_SPAN_TABLE}_check ON {_SPAN_TABLE}")
    op.execute(f"DROP TRIGGER IF EXISTS trg_{_SPAN_TABLE}_immutable ON {_SPAN_TABLE}")
    op.drop_table(_SPAN_TABLE)
    op.execute("DROP FUNCTION IF EXISTS dynamisrag_check_passage_source_span()")
    op.drop_constraint("ck_passage_text_nonempty", "passage", type_="check")
    op.drop_constraint("uq_paragraph_id_document_version_id", "paragraph", type_="unique")
    op.drop_constraint("uq_passage_id_document_version_id", "passage", type_="unique")
