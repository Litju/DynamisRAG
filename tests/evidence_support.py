"""Clearly synthetic evidence graph sources, shared only by tests."""

from datetime import UTC, datetime
from hashlib import sha256

from sqlalchemy.orm import Session

from dynamisrag.db.canonical import (
    insert_document,
    insert_document_version,
    insert_paragraph,
    insert_source_artifact,
)
from dynamisrag.domain.contracts import (
    Document,
    DocumentVersion,
    Paragraph,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.values import DocumentType, ParagraphRegion


def seed_source(session: Session) -> tuple[DocumentVersion, list[Section], list[Paragraph]]:
    now = datetime(2026, 10, 31, tzinfo=UTC)
    artifact = SourceArtifact(
        source_system="synthetic",
        source_external_id="test-fixture",
        source_uri="urn:synthetic:test",
        media_type="application/xml",
        content_sha256="a" * 64,
        byte_size=0,
        retrieved_at=now,
        storage_uri="urn:synthetic:unavailable",
    )
    document = Document(document_type=DocumentType.JOURNAL_ARTICLE, title="Synthetic test source")
    version = DocumentVersion(
        document_id=document.id,
        document_canonical_key=document.canonical_key,
        source_artifact_id=artifact.id,
        source_artifact_key=artifact.artifact_key,
        parser_revision="synthetic-v1",
        normalizer_revision="synthetic-v1",
        content_fingerprint="b" * 64,
        title="Synthetic test source",
        language="en",
        created_at=now,
    )
    insert_source_artifact(session, artifact)
    insert_document(session, document)
    insert_document_version(session, version)
    paragraphs: list[Paragraph] = []
    # Different sectionless paragraphs intentionally produce independent passages.
    for ordinal, value in enumerate(
        (
            "Synthetic methods paragraph has several words. " * 5,
            "Different synthetic paragraph also has several words. " * 5,
        )
    ):
        paragraph = Paragraph(
            document_version_id=version.id,
            version_key=version.version_key,
            ordinal=ordinal,
            region=ParagraphRegion.BODY,
            source_anchor=f"synthetic:/p[{ordinal}]",
            text=value,
            content_sha256=sha256(value.encode()).hexdigest(),
        )
        insert_paragraph(session, paragraph)
        paragraphs.append(paragraph)
    return version, [], paragraphs
