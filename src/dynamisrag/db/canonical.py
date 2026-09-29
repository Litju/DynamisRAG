"""Persistence primitives for the canonical document model.

Thin, explicit functions over the ORM records in :mod:`dynamisrag.db.models`
— deliberately not a repository or service abstraction. Each ``insert_*``
maps one canonical contract to its row and flushes, so database constraints
are enforced inside the caller's transaction; each ``get_*``/``list_*``
reads rows back. The contracts in :mod:`dynamisrag.domain.contracts` remain
the only place the meanings are defined.

Every ``insert_*`` for a child whose identity includes a semantic parent key
first verifies that key against the canonical key of the referenced persisted
parent (see the ``_require_*_key`` helpers). A child may therefore never
persist a surrogate foreign key to parent A while carrying parent B's
semantic key — a graph PostgreSQL would otherwise accept because the
foreign keys alone are valid. A missing referenced parent is not rejected
here; unknown ids surface through the database's own foreign-key semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from dynamisrag.chunking.errors import PassageSourceSpanError
from dynamisrag.db.errors import SemanticParentKeyError
from dynamisrag.db.models import (
    CitationRecord,
    CitationResolutionRecord,
    DocumentIdentifierRecord,
    DocumentRecord,
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
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
    Paragraph,
    Passage,
    PassageSourceSpan,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.values import IdentifierNamespace

__all__ = [
    "get_citation_resolutions",
    "get_document",
    "get_document_by_canonical_key",
    "get_document_by_identifier",
    "get_document_identifiers",
    "get_document_version",
    "get_document_version_by_key",
    "get_paragraph_by_key",
    "get_section",
    "get_source_artifact",
    "get_source_artifact_by_key",
    "insert_citation",
    "insert_citation_resolution",
    "insert_document",
    "insert_document_identifier",
    "insert_document_table",
    "insert_document_version",
    "insert_figure",
    "insert_paragraph",
    "insert_passage",
    "insert_passage_source_span",
    "insert_section",
    "insert_source_artifact",
    "list_citations",
    "list_document_tables",
    "list_document_versions",
    "list_figures",
    "list_paragraphs",
    "list_passage_source_spans",
    "list_passages",
    "list_sections",
]


def _require_document_key(
    session: Session, relationship: str, document_id: UUID, expected_key: str
) -> None:
    """Reject a supplied ``document_canonical_key`` that is not the referenced
    document's persisted canonical key."""
    document = get_document(session, document_id)
    if document is not None and document.canonical_key != expected_key:
        raise SemanticParentKeyError(
            relationship=relationship,
            parent_id=document_id,
            expected_key=document.canonical_key,
            received_key=expected_key,
        )


def _require_source_artifact_key(
    session: Session, relationship: str, artifact_id: UUID, expected_key: str
) -> None:
    """Reject a supplied ``source_artifact_key`` that is not the referenced
    artifact's persisted key."""
    artifact = get_source_artifact(session, artifact_id)
    if artifact is not None and artifact.artifact_key != expected_key:
        raise SemanticParentKeyError(
            relationship=relationship,
            parent_id=artifact_id,
            expected_key=artifact.artifact_key,
            received_key=expected_key,
        )


def _require_version_key(
    session: Session, relationship: str, version_id: UUID, expected_key: str
) -> None:
    """Reject a supplied ``version_key`` that is not the referenced document
    version's persisted key."""
    version = get_document_version(session, version_id)
    if version is not None and version.version_key != expected_key:
        raise SemanticParentKeyError(
            relationship=relationship,
            parent_id=version_id,
            expected_key=version.version_key,
            received_key=expected_key,
        )


def _require_citation_key(
    session: Session, relationship: str, citation_id: UUID, expected_key: str
) -> None:
    """Reject a supplied ``citation_key`` that is not the referenced
    citation's persisted key."""
    citation = session.get(CitationRecord, citation_id)
    if citation is not None and citation.citation_key != expected_key:
        raise SemanticParentKeyError(
            relationship=relationship,
            parent_id=citation_id,
            expected_key=citation.citation_key,
            received_key=expected_key,
        )


def _require_passage_key(
    session: Session, relationship: str, passage_id: UUID, expected_key: str
) -> None:
    """Reject a supplied ``passage_key`` that is not the referenced passage's
    persisted key."""
    passage = session.get(PassageRecord, passage_id)
    if passage is not None and passage.passage_key != expected_key:
        raise SemanticParentKeyError(
            relationship=relationship,
            parent_id=passage_id,
            expected_key=passage.passage_key,
            received_key=expected_key,
        )


def _require_paragraph_key(
    session: Session, relationship: str, paragraph_id: UUID, expected_key: str
) -> None:
    """Reject a supplied ``paragraph_key`` that is not the referenced
    paragraph's persisted key."""
    paragraph = session.get(ParagraphRecord, paragraph_id)
    if paragraph is not None and paragraph.paragraph_key != expected_key:
        raise SemanticParentKeyError(
            relationship=relationship,
            parent_id=paragraph_id,
            expected_key=paragraph.paragraph_key,
            received_key=expected_key,
        )


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
    """Persist one immutable canonical version and flush to enforce constraints.

    Both semantic parent keys are bound to their referenced parents first: the
    version may not point at one document/artifact by surrogate id while
    carrying another's canonical key.
    """
    _require_document_key(
        session,
        "document_version.document_id -> document.canonical_key",
        version.document_id,
        version.document_canonical_key,
    )
    _require_source_artifact_key(
        session,
        "document_version.source_artifact_id -> source_artifact.artifact_key",
        version.source_artifact_id,
        version.source_artifact_key,
    )
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
    rejects any parent that belongs to a different document version. The
    section's ``version_key`` is bound to the referenced version first, so a
    section can never claim a version it does not belong to.
    """
    _require_version_key(
        session,
        "section.document_version_id -> document_version.version_key",
        section.document_version_id,
        section.version_key,
    )
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
    section that belongs to a different document version. The passage's
    ``version_key`` is bound to the referenced version first.
    """
    _require_version_key(
        session,
        "passage.document_version_id -> document_version.version_key",
        passage.document_version_id,
        passage.version_key,
    )
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


def insert_paragraph(session: Session, paragraph: Paragraph) -> ParagraphRecord:
    """Persist one immutable source paragraph and flush to enforce
    constraints.

    The composite section foreign key needs the owning section's document
    version alongside the section id, so ``section_document_version_id`` is
    filled from the paragraph's own version — the database then rejects any
    section that belongs to a different document version. The paragraph's
    ``version_key`` is bound to the referenced version first.
    """
    _require_version_key(
        session,
        "paragraph.document_version_id -> document_version.version_key",
        paragraph.document_version_id,
        paragraph.version_key,
    )
    record = ParagraphRecord(
        id=paragraph.id,
        document_version_id=paragraph.document_version_id,
        section_id=paragraph.section_id,
        section_document_version_id=(
            paragraph.document_version_id if paragraph.section_id is not None else None
        ),
        ordinal=paragraph.ordinal,
        region=paragraph.region.value,
        source_anchor=paragraph.source_anchor,
        text=paragraph.text,
        content_sha256=paragraph.content_sha256,
        paragraph_key=paragraph.paragraph_key,
    )
    session.add(record)
    session.flush()
    return record


def insert_passage_source_span(
    session: Session, span: PassageSourceSpan
) -> PassageSourceSpanRecord:
    """Persist one exact passage source span and flush to enforce constraints.

    The span's semantic parent keys are bound to their referenced parents
    first: a span may not point at one passage/paragraph by surrogate id
    while carrying another's canonical key. The offsets must fit the
    referenced paragraph's persisted text — the database's offset trigger
    enforces the same rule for any writer, and the composite foreign keys
    bind passage and paragraph to the span's document version.
    """
    _require_passage_key(
        session,
        "passage_source_span.passage_id -> passage.passage_key",
        span.passage_id,
        span.passage_key,
    )
    _require_paragraph_key(
        session,
        "passage_source_span.paragraph_id -> paragraph.paragraph_key",
        span.paragraph_id,
        span.paragraph_key,
    )
    paragraph = session.get(ParagraphRecord, span.paragraph_id)
    if paragraph is not None and span.end_char > len(paragraph.text):
        raise PassageSourceSpanError(
            f"passage_source_span {span.span_key!r} end_char {span.end_char} exceeds "
            f"paragraph {span.paragraph_id} text length {len(paragraph.text)}"
        )
    record = PassageSourceSpanRecord(
        id=span.id,
        document_version_id=span.document_version_id,
        passage_id=span.passage_id,
        passage_document_version_id=span.document_version_id,
        paragraph_id=span.paragraph_id,
        paragraph_document_version_id=span.document_version_id,
        source_order=span.source_order,
        start_char=span.start_char,
        end_char=span.end_char,
        span_key=span.span_key,
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
    never forecloses later resolution. The citation's ``version_key`` is bound
    to the referenced version first.
    """
    _require_version_key(
        session,
        "citation.document_version_id -> document_version.version_key",
        citation.document_version_id,
        citation.version_key,
    )
    record = CitationRecord(
        id=citation.id,
        document_version_id=citation.document_version_id,
        ordinal=citation.ordinal,
        source_reference_id=citation.source_reference_id,
        source_anchor=citation.source_anchor,
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
    citation row is never touched. Both semantic parent keys are bound to
    their referenced parents first.
    """
    _require_citation_key(
        session,
        "citation_resolution.citation_id -> citation.citation_key",
        resolution.citation_id,
        resolution.citation_key,
    )
    _require_document_key(
        session,
        "citation_resolution.resolved_document_id -> document.canonical_key",
        resolution.resolved_document_id,
        resolution.resolved_document_canonical_key,
    )
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
    """Persist one scientific table and flush to enforce constraints.

    The table's ``version_key`` is bound to the referenced version first.
    """
    _require_version_key(
        session,
        "document_table.document_version_id -> document_version.version_key",
        table.document_version_id,
        table.version_key,
    )
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
    """Persist one scientific figure and flush to enforce constraints.

    The figure's ``version_key`` is bound to the referenced version first.
    """
    _require_version_key(
        session,
        "figure.document_version_id -> document_version.version_key",
        figure.document_version_id,
        figure.version_key,
    )
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


def get_source_artifact_by_key(session: Session, artifact_key: str) -> SourceArtifactRecord | None:
    """Resolve an artifact by its deterministic canonical key, or ``None``.

    This is the lookup that makes idempotent acquisition possible: the
    ``artifact_key`` is a pure function of source system, external id and
    content digest, so re-acquiring identical bytes resolves to the same row
    instead of inserting a duplicate.
    """
    return session.scalars(
        select(SourceArtifactRecord).where(SourceArtifactRecord.artifact_key == artifact_key)
    ).first()


def get_document(session: Session, document_id: UUID) -> DocumentRecord | None:
    return session.get(DocumentRecord, document_id)


def get_document_by_identifier(
    session: Session, namespace: IdentifierNamespace, normalized_value: str
) -> DocumentRecord | None:
    """Resolve the Document carrying one identifier alias, or ``None``.

    This is the lookup that makes logical-document resolution possible: the
    alias table is the complete record of every identifier a work has been
    known by, so a parsed DOI/PMID/PMCID finds the existing logical Document
    instead of creating a duplicate.
    """
    return session.scalars(
        select(DocumentRecord)
        .join(
            DocumentIdentifierRecord,
            DocumentIdentifierRecord.document_id == DocumentRecord.id,
        )
        .where(DocumentIdentifierRecord.namespace == namespace.value)
        .where(DocumentIdentifierRecord.normalized_value == normalized_value)
    ).first()


def get_document_by_canonical_key(session: Session, canonical_key: str) -> DocumentRecord | None:
    """Resolve a Document by its fixed canonical key, or ``None``."""
    return session.scalars(
        select(DocumentRecord).where(DocumentRecord.canonical_key == canonical_key)
    ).first()


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


def get_document_version_by_key(session: Session, version_key: str) -> DocumentVersionRecord | None:
    """Resolve a DocumentVersion by its deterministic key, or ``None``.

    This is the lookup that makes canonical materialization idempotent: the
    same SourceArtifact parsed under the same parser/normalizer revisions
    yields the same ``version_key``, so a reparse finds the persisted version
    instead of duplicating the canonical graph.
    """
    return session.scalars(
        select(DocumentVersionRecord).where(DocumentVersionRecord.version_key == version_key)
    ).first()


def get_section(session: Session, section_id: UUID) -> SectionRecord | None:
    return session.get(SectionRecord, section_id)


def get_paragraph_by_key(session: Session, paragraph_key: str) -> ParagraphRecord | None:
    """Resolve a Paragraph by its deterministic key, or ``None``.

    The lookup that makes span materialization resolve semantic paragraph
    identities to their persisted rows: the manifest carries paragraph
    keys, never surrogate ids.
    """
    return session.scalars(
        select(ParagraphRecord).where(ParagraphRecord.paragraph_key == paragraph_key)
    ).first()


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


def list_paragraphs(session: Session, document_version_id: UUID) -> Sequence[ParagraphRecord]:
    return list(
        session.scalars(
            select(ParagraphRecord)
            .where(ParagraphRecord.document_version_id == document_version_id)
            .order_by(ParagraphRecord.ordinal)
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


def list_passage_source_spans(
    session: Session, passage_ids: Sequence[UUID]
) -> Sequence[PassageSourceSpanRecord]:
    """Read the ordered source spans of the given passages.

    Ordered by ``(passage_id, source_order)`` so each passage's spans come
    back in their deterministic provenance order.
    """
    if not passage_ids:
        return []
    return list(
        session.scalars(
            select(PassageSourceSpanRecord)
            .where(PassageSourceSpanRecord.passage_id.in_(passage_ids))
            .order_by(
                PassageSourceSpanRecord.passage_id,
                PassageSourceSpanRecord.source_order,
            )
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
