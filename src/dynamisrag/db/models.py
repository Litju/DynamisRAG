"""SQLAlchemy 2.x records for the canonical scientific document schema.

These classes mirror the DDL applied by ``0002_canonical_document_model``
and are the persistence-side representation of the canonical contracts in
:mod:`dynamisrag.domain.contracts`. They carry no domain behaviour: the
contracts own the meanings, these rows own the columns. Constraint names
match the migration exactly so the two views of the schema never drift
apart silently.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "CitationRecord",
    "CitationResolutionRecord",
    "DocumentIdentifierRecord",
    "DocumentRecord",
    "DocumentTableRecord",
    "DocumentVersionRecord",
    "FigureRecord",
    "ParagraphRecord",
    "PassageRecord",
    "SectionRecord",
    "SourceArtifactRecord",
]

_DOI_VALUE_FORMAT = r"^10\.[0-9]{4,9}/\S+$"
_PMID_VALUE_FORMAT = r"^[0-9]{1,10}$"
_PMCID_VALUE_FORMAT = r"^PMC[0-9]{1,12}$"
_IDENTIFIER_NAMESPACE_VALUE_FORMAT = (
    "(namespace = 'doi' AND normalized_value ~ '" + _DOI_VALUE_FORMAT + "')"
    " OR (namespace = 'pmid' AND normalized_value ~ '" + _PMID_VALUE_FORMAT + "')"
    " OR (namespace = 'pmcid' AND normalized_value ~ '" + _PMCID_VALUE_FORMAT + "')"
)
"""The document_identifier CHECK mirroring the domain namespace/value pairs."""


class Base(DeclarativeBase):
    """Declarative base for the canonical schema records."""


class SourceArtifactRecord(Base):
    __tablename__ = "source_artifact"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_source_artifact"),
        UniqueConstraint("artifact_key", name="uq_source_artifact_artifact_key"),
        CheckConstraint("byte_size >= 0", name="ck_source_artifact_byte_size_nonnegative"),
        CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_source_artifact_content_sha256_hex",
        ),
        Index("ix_source_artifact_source_lookup", "source_system", "source_external_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    source_system: Mapped[str] = mapped_column(Text, nullable=False)
    source_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    license_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    license_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    versions: Mapped[list[DocumentVersionRecord]] = relationship(back_populates="source_artifact")


class DocumentRecord(Base):
    __tablename__ = "document"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_document"),
        UniqueConstraint("canonical_key", name="uq_document_canonical_key"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    canonical_key: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    versions: Mapped[list[DocumentVersionRecord]] = relationship(back_populates="document")
    identifiers: Mapped[list[DocumentIdentifierRecord]] = relationship(back_populates="document")
    resolutions: Mapped[list[CitationResolutionRecord]] = relationship(
        back_populates="resolved_document"
    )


class DocumentIdentifierRecord(Base):
    __tablename__ = "document_identifier"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_document_identifier"),
        UniqueConstraint(
            "namespace", "normalized_value", name="uq_document_identifier_namespace_value"
        ),
        ForeignKeyConstraint(
            ["document_id"], ["document.id"], name="fk_document_identifier_document"
        ),
        CheckConstraint(
            "namespace IN ('doi', 'pmid', 'pmcid')", name="ck_document_identifier_namespace"
        ),
        CheckConstraint(
            _IDENTIFIER_NAMESPACE_VALUE_FORMAT, name="ck_document_identifier_value_format"
        ),
        Index("ix_document_identifier_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[DocumentRecord] = relationship(back_populates="identifiers")


class DocumentVersionRecord(Base):
    __tablename__ = "document_version"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_document_version"),
        UniqueConstraint("version_key", name="uq_document_version_version_key"),
        ForeignKeyConstraint(["document_id"], ["document.id"], name="fk_document_version_document"),
        ForeignKeyConstraint(
            ["source_artifact_id"],
            ["source_artifact.id"],
            name="fk_document_version_source_artifact",
        ),
        CheckConstraint(
            "content_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_document_version_content_fingerprint_hex",
        ),
        CheckConstraint("language ~ '^[a-z]{2,3}$'", name="ck_document_version_language_code"),
        Index("ix_document_version_document_id", "document_id"),
        Index("ix_document_version_source_artifact_id", "source_artifact_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    source_artifact_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    parser_revision: Mapped[str] = mapped_column(Text, nullable=False)
    normalizer_revision: Mapped[str] = mapped_column(Text, nullable=False)
    content_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    versioned_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[DocumentRecord] = relationship(back_populates="versions")
    source_artifact: Mapped[SourceArtifactRecord] = relationship(back_populates="versions")
    sections: Mapped[list[SectionRecord]] = relationship(back_populates="document_version")
    passages: Mapped[list[PassageRecord]] = relationship(back_populates="document_version")
    paragraphs: Mapped[list[ParagraphRecord]] = relationship(back_populates="document_version")
    citations: Mapped[list[CitationRecord]] = relationship(back_populates="document_version")
    tables: Mapped[list[DocumentTableRecord]] = relationship(back_populates="document_version")
    figures: Mapped[list[FigureRecord]] = relationship(back_populates="document_version")


class SectionRecord(Base):
    __tablename__ = "section"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_section"),
        UniqueConstraint("id", "document_version_id", name="uq_section_id_document_version_id"),
        UniqueConstraint(
            "document_version_id", "section_key", name="uq_section_document_version_section_key"
        ),
        ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_section_document_version",
        ),
        ForeignKeyConstraint(
            ["parent_section_id", "parent_document_version_id"],
            ["section.id", "section.document_version_id"],
            name="fk_section_parent",
        ),
        CheckConstraint(
            "(parent_section_id IS NULL) = (parent_document_version_id IS NULL)",
            name="ck_section_parent_pair",
        ),
        CheckConstraint("ordinal >= 0", name="ck_section_ordinal_nonnegative"),
        CheckConstraint("depth >= 0", name="ck_section_depth_nonnegative"),
        CheckConstraint(
            "structural_path ~ '^[0-9]+(\\.[0-9]+)*$'",
            name="ck_section_structural_path_canonical",
        ),
        CheckConstraint(
            "content_fingerprint IS NULL OR content_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_section_content_fingerprint_hex",
        ),
        Index("ix_section_document_version_id", "document_version_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    parent_section_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    parent_document_version_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    semantic_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_anchor: Mapped[str | None] = mapped_column(Text, nullable=True)
    structural_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document_version: Mapped[DocumentVersionRecord] = relationship(back_populates="sections")


class PassageRecord(Base):
    __tablename__ = "passage"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_passage"),
        UniqueConstraint(
            "document_version_id",
            "chunker_revision",
            "ordinal",
            name="uq_passage_version_chunker_ordinal",
        ),
        ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_passage_document_version",
        ),
        ForeignKeyConstraint(
            ["section_id", "section_document_version_id"],
            ["section.id", "section.document_version_id"],
            name="fk_passage_section",
        ),
        CheckConstraint(
            "(section_id IS NULL) = (section_document_version_id IS NULL)",
            name="ck_passage_section_pair",
        ),
        CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'", name="ck_passage_content_sha256_hex"),
        CheckConstraint("ordinal >= 0", name="ck_passage_ordinal_nonnegative"),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_passage_token_count_nonnegative",
        ),
        Index("ix_passage_document_version_id", "document_version_id"),
        Index("ix_passage_section_id", "section_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    section_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    section_document_version_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    chunker_revision: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    source_anchor: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passage_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document_version: Mapped[DocumentVersionRecord] = relationship(back_populates="passages")


class ParagraphRecord(Base):
    __tablename__ = "paragraph"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_paragraph"),
        UniqueConstraint("paragraph_key", name="uq_paragraph_paragraph_key"),
        UniqueConstraint(
            "document_version_id", "source_anchor", name="uq_paragraph_document_version_source_anchor"
        ),
        UniqueConstraint(
            "document_version_id", "ordinal", name="uq_paragraph_document_version_ordinal"
        ),
        ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_paragraph_document_version",
        ),
        ForeignKeyConstraint(
            ["section_id", "section_document_version_id"],
            ["section.id", "section.document_version_id"],
            name="fk_paragraph_section",
        ),
        CheckConstraint(
            "(section_id IS NULL) = (section_document_version_id IS NULL)",
            name="ck_paragraph_section_pair",
        ),
        CheckConstraint("ordinal >= 0", name="ck_paragraph_ordinal_nonnegative"),
        CheckConstraint(
            "region IN ('front', 'body', 'back')", name="ck_paragraph_region_source_derived"
        ),
        CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'", name="ck_paragraph_content_sha256_hex"),
        Index("ix_paragraph_document_version_id", "document_version_id"),
        Index("ix_paragraph_section_id", "section_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    section_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    section_document_version_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    region: Mapped[str] = mapped_column(Text, nullable=False)
    source_anchor: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    paragraph_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document_version: Mapped[DocumentVersionRecord] = relationship(back_populates="paragraphs")


class CitationRecord(Base):
    __tablename__ = "citation"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_citation"),
        UniqueConstraint(
            "document_version_id", "ordinal", name="uq_citation_document_version_ordinal"
        ),
        UniqueConstraint(
            "document_version_id",
            "citation_key",
            name="uq_citation_document_version_key",
        ),
        ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_citation_document_version",
        ),
        CheckConstraint("ordinal >= 0", name="ck_citation_ordinal_nonnegative"),
        CheckConstraint(
            "year IS NULL OR (year >= 1000 AND year <= 2200)",
            name="ck_citation_year_plausible",
        ),
        CheckConstraint(
            "doi IS NULL OR doi ~ '" + _DOI_VALUE_FORMAT + "'", name="ck_citation_doi_format"
        ),
        CheckConstraint(
            "pmid IS NULL OR pmid ~ '" + _PMID_VALUE_FORMAT + "'", name="ck_citation_pmid_format"
        ),
        CheckConstraint(
            "pmcid IS NULL OR pmcid ~ '" + _PMCID_VALUE_FORMAT + "'",
            name="ck_citation_pmcid_format",
        ),
        Index("ix_citation_document_version_id", "document_version_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_reference_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_anchor: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(Text, nullable=True)
    pmid: Mapped[str | None] = mapped_column(Text, nullable=True)
    pmcid: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_reference_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document_version: Mapped[DocumentVersionRecord] = relationship(back_populates="citations")
    resolutions: Mapped[list[CitationResolutionRecord]] = relationship(back_populates="citation")


class CitationResolutionRecord(Base):
    __tablename__ = "citation_resolution"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_citation_resolution"),
        UniqueConstraint("resolution_key", name="uq_citation_resolution_resolution_key"),
        ForeignKeyConstraint(
            ["citation_id"], ["citation.id"], name="fk_citation_resolution_citation"
        ),
        ForeignKeyConstraint(
            ["resolved_document_id"],
            ["document.id"],
            name="fk_citation_resolution_resolved_document",
        ),
        Index("ix_citation_resolution_citation_id", "citation_id"),
        Index("ix_citation_resolution_resolved_document_id", "resolved_document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    citation_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    resolved_document_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    resolver_revision: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolution_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    citation: Mapped[CitationRecord] = relationship(back_populates="resolutions")
    resolved_document: Mapped[DocumentRecord] = relationship(back_populates="resolutions")


class DocumentTableRecord(Base):
    __tablename__ = "document_table"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_document_table"),
        UniqueConstraint(
            "document_version_id",
            "document_table_key",
            name="uq_document_table_document_version_key",
        ),
        ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_document_table_document_version",
        ),
        ForeignKeyConstraint(
            ["section_id", "section_document_version_id"],
            ["section.id", "section.document_version_id"],
            name="fk_document_table_section",
        ),
        CheckConstraint(
            "(section_id IS NULL) = (section_document_version_id IS NULL)",
            name="ck_document_table_section_pair",
        ),
        CheckConstraint("ordinal >= 0", name="ck_document_table_ordinal_nonnegative"),
        CheckConstraint(
            "content_fingerprint IS NULL OR content_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_document_table_content_fingerprint_hex",
        ),
        Index("ix_document_table_document_version_id", "document_version_id"),
        Index("ix_document_table_section_id", "section_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    section_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    section_document_version_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_anchor: Mapped[str | None] = mapped_column(Text, nullable=True)
    structured_representation: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    content_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_table_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document_version: Mapped[DocumentVersionRecord] = relationship(back_populates="tables")


class FigureRecord(Base):
    __tablename__ = "figure"

    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_figure"),
        UniqueConstraint(
            "document_version_id", "figure_key", name="uq_figure_document_version_key"
        ),
        ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name="fk_figure_document_version",
        ),
        ForeignKeyConstraint(
            ["section_id", "section_document_version_id"],
            ["section.id", "section.document_version_id"],
            name="fk_figure_section",
        ),
        CheckConstraint(
            "(section_id IS NULL) = (section_document_version_id IS NULL)",
            name="ck_figure_section_pair",
        ),
        CheckConstraint("ordinal >= 0", name="ck_figure_ordinal_nonnegative"),
        CheckConstraint(
            "content_fingerprint IS NULL OR content_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_figure_content_fingerprint_hex",
        ),
        Index("ix_figure_document_version_id", "document_version_id"),
        Index("ix_figure_section_id", "section_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    section_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    section_document_version_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_anchor: Mapped[str | None] = mapped_column(Text, nullable=True)
    asset_locator: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    figure_key: Mapped[str] = mapped_column(Text, nullable=False)
    row_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document_version: Mapped[DocumentVersionRecord] = relationship(back_populates="figures")
