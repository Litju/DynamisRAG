"""Persistence primitives for the canonical document model.

Thin, explicit functions over the ORM records in :mod:`dynamisrag.db.models`
— deliberately not a repository or service abstraction. Each ``insert_*``
maps one canonical contract to its row and flushes, so database constraints
are enforced inside the caller's transaction; each ``get_*``/``list_*``
reads rows back. The contracts in :mod:`dynamisrag.domain.contracts` remain
the only place the meanings are defined.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from dynamisrag.db.models import (
    CitationRecord,
    CitationResolutionRecord,
    DocumentIdentifierRecord,
    DocumentRecord,
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    PassageRecord,
    SectionRecord,
    SourceArtifactRecord,
)
from dynamisrag.domain.contracts import (
    Citation,
    CitationResolution,
    Document,
    DocumentIdentifier,
    DocumentTable,
    DocumentVersion,
    Figure,
    Passage,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.values import IdentifierNamespace

__all__ = [
    "get_citation_resolutions",
    "get_document",
    "get_document_identifiers",
    "get_document_version",
    "get_section",
    "get_source_artifact",
    "insert_citation",
    "insert_citation_resolution",
    "insert_document",
    "insert_document_identifier",
    "insert_document_table",
    "insert_document_version",
    "insert_figure",
    "insert_passage",
    "insert_section",
    "insert_source_artifact",
    "list_citations",
    "list_document_tables",
    "list_document_versions",
    "list_figures",
    "list_passages",
    "list_sections",
]


def insert_source_artifact(session: Session, artifact: SourceArtifact) -> SourceArtifactRecord:
    """Persist one exact acquired artifact and flush to enforce constraints."""
    record = SourceArtifactRecord(
        id=artifact.id,
        source_system=artifact.source_system,
        source_external_id=artifact.source_external_id,
        source_uri=artifact.source_uri,
        media_type=artifact.media_type,
        content_sha256=artifact.content_sha256,
        byte_size=artifact.byte_size,
        retrieved_at=artifact.retrieved_at,
        storage_uri=artifact.storage_uri,
        license_name=artifact.license_name,
        license_uri=artifact.license_uri,
        artifact_key=artifact.artifact_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_document(session: Session, document: Document) -> DocumentRecord:
    """Persist one logical scientific work and flush to enforce constraints.

    The strong identifiers known at creation (``doi``/``pmid``/``pmcid``) are
    persisted alongside the row as :class:`DocumentIdentifier` aliases, so the
    alias table is the complete record of every identifier the work has been
    known by. They fixed the document's canonical identity at creation and
    are never recomputed; later identifiers arrive through
    :func:`insert_document_identifier` and only ever add aliases.
    """
    record = DocumentRecord(
        id=document.id,
        canonical_key=document.canonical_key,
        document_type=document.document_type.value,
        title=document.title,
    )
    session.add(record)
    for namespace, value in (
        (IdentifierNamespace.DOI, document.doi),
        (IdentifierNamespace.PMID, document.pmid),
        (IdentifierNamespace.PMCID, document.pmcid),
    ):
        if value is not None:
            session.add(
                DocumentIdentifierRecord(
                    id=uuid4(),
                    document_id=document.id,
                    namespace=namespace.value,
                    normalized_value=value,
                )
            )
    session.flush()
    return record


def insert_document_identifier(
    session: Session, identifier: DocumentIdentifier
) -> DocumentIdentifierRecord:
    """Attach one identifier alias to a document and flush to enforce
    constraints.

    ``(namespace, normalized_value)`` is globally unique, so an alias that is
    already attached to any document — this one included — is rejected rather
    than duplicated. Attaching an alias never mutates the document's canonical
    identity.
    """
    record = DocumentIdentifierRecord(
        id=identifier.id,
        document_id=identifier.document_id,
        namespace=identifier.namespace.value,
        normalized_value=identifier.normalized_value,
    )
    session.add(record)
    session.flush()
    return record


def insert_document_version(session: Session, version: DocumentVersion) -> DocumentVersionRecord:
    """Persist one immutable canonical version and flush to enforce constraints."""
    record = DocumentVersionRecord(
        id=version.id,
        document_id=version.document_id,
        source_artifact_id=version.source_artifact_id,
        parser_revision=version.parser_revision,
        normalizer_revision=version.normalizer_revision,
        content_fingerprint=version.content_fingerprint,
        title=version.title,
        language=version.language,
        versioned_metadata=version.versioned_metadata,
        created_at=version.created_at,
        version_key=version.version_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_section(session: Session, section: Section) -> SectionRecord:
    """Persist one structural node and flush to enforce constraints.

    The composite parent foreign key needs the parent's document version
    alongside the parent id, so the redundant ``parent_document_version_id``
    column is filled from the section's own version — the database then
    rejects any parent that belongs to a different document version.
    """
    record = SectionRecord(
        id=section.id,
        document_version_id=section.document_version_id,
        parent_section_id=section.parent_section_id,
        parent_document_version_id=(
            section.document_version_id if section.parent_section_id is not None else None
        ),
        ordinal=section.ordinal,
        depth=section.depth,
        title=section.title,
        semantic_type=section.semantic_type,
        source_anchor=section.source_anchor,
        structural_path=section.structural_path,
        content_fingerprint=section.content_fingerprint,
        section_key=section.section_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_passage(session: Session, passage: Passage) -> PassageRecord:
    """Persist one retrieval unit and flush to enforce constraints.

    The composite section foreign key needs the owning section's document
    version alongside the section id, so ``section_document_version_id`` is
    filled from the passage's own version — the database then rejects any
    section that belongs to a different document version.
    """
    record = PassageRecord(
        id=passage.id,
        document_version_id=passage.document_version_id,
        section_id=passage.section_id,
        section_document_version_id=(
            passage.document_version_id if passage.section_id is not None else None
        ),
        chunker_revision=passage.chunker_revision,
        ordinal=passage.ordinal,
        text=passage.text,
        content_sha256=passage.content_sha256,
        source_anchor=passage.source_anchor,
        token_count=passage.token_count,
        passage_key=passage.passage_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_citation(session: Session, citation: Citation) -> CitationRecord:
    """Persist one immutable bibliographic reference and flush to enforce
    constraints.

    The canonical citation carries resolution content only. Linking a
    citation to the document it resolved to is append-only state living on
    ``citation_resolution`` records, so persisting an unresolved citation
    never forecloses later resolution.
    """
    record = CitationRecord(
        id=citation.id,
        document_version_id=citation.document_version_id,
        ordinal=citation.ordinal,
        source_reference_id=citation.source_reference_id,
        doi=citation.doi,
        pmid=citation.pmid,
        pmcid=citation.pmcid,
        title=citation.title,
        year=citation.year,
        raw_reference_text=citation.raw_reference_text,
        citation_key=citation.citation_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_citation_resolution(
    session: Session, resolution: CitationResolution
) -> CitationResolutionRecord:
    """Append one citation resolution and flush to enforce constraints.

    The resolution references the immutable canonical citation and the
    resolved document by foreign key; its deterministic identity collides
    on a repeated identical resolution instead of duplicating. The canonical
    citation row is never touched.
    """
    record = CitationResolutionRecord(
        id=resolution.id,
        citation_id=resolution.citation_id,
        resolved_document_id=resolution.resolved_document_id,
        resolver_revision=resolution.resolver_revision,
        resolved_at=resolution.resolved_at,
        resolution_key=resolution.resolution_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_document_table(session: Session, table: DocumentTable) -> DocumentTableRecord:
    """Persist one scientific table and flush to enforce constraints."""
    record = DocumentTableRecord(
        id=table.id,
        document_version_id=table.document_version_id,
        section_id=table.section_id,
        section_document_version_id=(
            table.document_version_id if table.section_id is not None else None
        ),
        ordinal=table.ordinal,
        label=table.label,
        caption=table.caption,
        source_anchor=table.source_anchor,
        structured_representation=table.structured_representation,
        content_fingerprint=table.content_fingerprint,
        document_table_key=table.document_table_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_figure(session: Session, figure: Figure) -> FigureRecord:
    """Persist one scientific figure and flush to enforce constraints."""
    record = FigureRecord(
        id=figure.id,
        document_version_id=figure.document_version_id,
        section_id=figure.section_id,
        section_document_version_id=(
            figure.document_version_id if figure.section_id is not None else None
        ),
        ordinal=figure.ordinal,
        label=figure.label,
        caption=figure.caption,
        source_anchor=figure.source_anchor,
        asset_locator=figure.asset_locator,
        content_fingerprint=figure.content_fingerprint,
        figure_key=figure.figure_key,
    )
    session.add(record)
    session.flush()
    return record


def get_source_artifact(session: Session, artifact_id: UUID) -> SourceArtifactRecord | None:
    return session.get(SourceArtifactRecord, artifact_id)


def get_document(session: Session, document_id: UUID) -> DocumentRecord | None:
    return session.get(DocumentRecord, document_id)


def get_document_identifiers(
    session: Session, document_id: UUID
) -> Sequence[DocumentIdentifierRecord]:
    return list(
        session.scalars(
            select(DocumentIdentifierRecord)
            .where(DocumentIdentifierRecord.document_id == document_id)
            .order_by(DocumentIdentifierRecord.namespace, DocumentIdentifierRecord.normalized_value)
        )
    )


def get_document_version(session: Session, version_id: UUID) -> DocumentVersionRecord | None:
    return session.get(DocumentVersionRecord, version_id)


def get_section(session: Session, section_id: UUID) -> SectionRecord | None:
    return session.get(SectionRecord, section_id)


def list_document_versions(session: Session, document_id: UUID) -> Sequence[DocumentVersionRecord]:
    return list(
        session.scalars(
            select(DocumentVersionRecord)
            .where(DocumentVersionRecord.document_id == document_id)
            .order_by(DocumentVersionRecord.created_at, DocumentVersionRecord.id)
        )
    )


def list_sections(session: Session, document_version_id: UUID) -> Sequence[SectionRecord]:
    return list(
        session.scalars(
            select(SectionRecord)
            .where(SectionRecord.document_version_id == document_version_id)
            .order_by(SectionRecord.structural_path, SectionRecord.ordinal)
        )
    )


def list_passages(
    session: Session, document_version_id: UUID, chunker_revision: str
) -> Sequence[PassageRecord]:
    return list(
        session.scalars(
            select(PassageRecord)
            .where(PassageRecord.document_version_id == document_version_id)
            .where(PassageRecord.chunker_revision == chunker_revision)
            .order_by(PassageRecord.ordinal)
        )
    )


def list_citations(session: Session, document_version_id: UUID) -> Sequence[CitationRecord]:
    return list(
        session.scalars(
            select(CitationRecord)
            .where(CitationRecord.document_version_id == document_version_id)
            .order_by(CitationRecord.ordinal)
        )
    )


def get_citation_resolutions(
    session: Session, citation_id: UUID
) -> Sequence[CitationResolutionRecord]:
    return list(
        session.scalars(
            select(CitationResolutionRecord)
            .where(CitationResolutionRecord.citation_id == citation_id)
            .order_by(
                CitationResolutionRecord.resolver_revision,
                CitationResolutionRecord.resolved_at,
            )
        )
    )


def list_document_tables(
    session: Session, document_version_id: UUID
) -> Sequence[DocumentTableRecord]:
    return list(
        session.scalars(
            select(DocumentTableRecord)
            .where(DocumentTableRecord.document_version_id == document_version_id)
            .order_by(DocumentTableRecord.ordinal)
        )
    )


def list_figures(session: Session, document_version_id: UUID) -> Sequence[FigureRecord]:
    return list(
        session.scalars(
            select(FigureRecord)
            .where(FigureRecord.document_version_id == document_version_id)
            .order_by(FigureRecord.ordinal)
        )
    )
