"""Adapt registered BEIR SciFact abstracts to the existing passage projection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.db.canonical import PassageProjectionRecords, PassageSourceSpanLineage
from dynamisrag.db.models import (
    DocumentRecord,
    DocumentVersionRecord,
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
    SourceArtifactRecord,
)
from dynamisrag.domain.contracts import (
    Document,
    DocumentVersion,
    Paragraph,
    Passage,
    PassageSourceSpan,
    SourceArtifact,
)
from dynamisrag.domain.values import DocumentType, ParagraphRegion

__all__ = [
    "SCIFACT_PROJECTION_PARSER_REVISION",
    "SciFactCorpusDocument",
    "scifact_projection_records",
]

SCIFACT_PROJECTION_PARSER_REVISION = "beir-scifact-corpus-v1"
_SCIFACT_PROJECTION_NORMALIZER_REVISION = "beir-scifact-text-v1"


@dataclass(frozen=True, slots=True)
class SciFactCorpusDocument:
    """One exact corpus.jsonl row and its source-record provenance."""

    document_id: str
    title: str
    text: str
    source_artifact: SourceArtifact

    def __post_init__(self) -> None:
        if not self.document_id or not self.title.strip() or not self.text.strip():
            raise ValueError("SciFact corpus rows need an ID, title and nonblank abstract")
        if (
            self.source_artifact.source_system != "beir_scifact"
            or self.source_artifact.source_external_id != self.document_id
        ):
            raise ValueError("SciFact row and source artifact identities differ")


def _content_fingerprint(row: SciFactCorpusDocument) -> str:
    payload = {
        "source_id": "beir.scifact",
        "document_id": row.document_id,
        "title": row.title,
        "text": row.text,
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _projection_records_for_document(
    row: SciFactCorpusDocument,
    *,
    logical_title: str,
    chunker: StructureAwareChunker,
) -> tuple[PassageProjectionRecords, ...]:
    document = Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        title=logical_title,
    )
    version = DocumentVersion(
        document_id=document.id,
        document_canonical_key=document.canonical_key,
        source_artifact_id=row.source_artifact.id,
        source_artifact_key=row.source_artifact.artifact_key,
        parser_revision=SCIFACT_PROJECTION_PARSER_REVISION,
        normalizer_revision=_SCIFACT_PROJECTION_NORMALIZER_REVISION,
        content_fingerprint=_content_fingerprint(row),
        title=row.title,
        language="en",
        versioned_metadata={
            "dataset_source_id": "beir.scifact",
            "beir_document_id": row.document_id,
        },
        created_at=datetime.now(UTC),
    )
    anchor = f"beir/{row.document_id}/abstract"
    paragraph = Paragraph(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=0,
        region=ParagraphRegion.BODY,
        source_anchor=anchor,
        text=row.text,
        content_sha256=hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
    )
    manifest = chunker.plan(version, sections=(), paragraphs=(paragraph,))

    artifact_record = SourceArtifactRecord(
        id=row.source_artifact.id,
        source_system=row.source_artifact.source_system,
        source_external_id=row.source_artifact.source_external_id,
        source_uri=row.source_artifact.source_uri,
        media_type=row.source_artifact.media_type,
        content_sha256=row.source_artifact.content_sha256,
        byte_size=row.source_artifact.byte_size,
        retrieved_at=row.source_artifact.retrieved_at,
        storage_uri=row.source_artifact.storage_uri,
        license_name=row.source_artifact.license_name,
        license_uri=row.source_artifact.license_uri,
        artifact_key=row.source_artifact.artifact_key,
    )
    document_record = DocumentRecord(
        id=document.id,
        canonical_key=document.canonical_key,
        document_type=document.document_type.value,
        title=document.title,
    )
    version_record = DocumentVersionRecord(
        id=version.id,
        document_id=version.document_id,
        source_artifact_id=version.source_artifact_id,
        parser_revision=version.parser_revision,
        normalizer_revision=version.normalizer_revision,
        content_fingerprint=version.content_fingerprint,
        title=version.title,
        language=version.language,
        versioned_metadata=dict(version.versioned_metadata),
        created_at=version.created_at,
        version_key=version.version_key,
    )
    paragraph_record = ParagraphRecord(
        id=paragraph.id,
        document_version_id=paragraph.document_version_id,
        section_id=None,
        section_document_version_id=None,
        ordinal=paragraph.ordinal,
        region=paragraph.region.value,
        source_anchor=paragraph.source_anchor,
        text=paragraph.text,
        content_sha256=paragraph.content_sha256,
        paragraph_key=paragraph.paragraph_key,
    )

    records: list[PassageProjectionRecords] = []
    for planned in manifest.passages:
        passage = Passage(
            document_version_id=version.id,
            version_key=version.version_key,
            chunker_revision=manifest.chunker_revision,
            ordinal=planned.ordinal,
            text=planned.text,
            content_sha256=planned.content_sha256,
            source_anchor=planned.primary_source_anchor,
            token_count=planned.token_count,
        )
        if passage.passage_key != planned.passage_key:
            raise ValueError("SciFact passage identity differs from the production chunker output")
        passage_record = PassageRecord(
            id=passage.id,
            document_version_id=passage.document_version_id,
            section_id=None,
            section_document_version_id=None,
            chunker_revision=passage.chunker_revision,
            ordinal=passage.ordinal,
            text=passage.text,
            content_sha256=passage.content_sha256,
            source_anchor=passage.source_anchor,
            token_count=passage.token_count,
            passage_key=passage.passage_key,
        )
        lineages: list[PassageSourceSpanLineage] = []
        for source_span in planned.source_spans:
            span = PassageSourceSpan(
                document_version_id=version.id,
                passage_id=passage.id,
                paragraph_id=paragraph.id,
                passage_key=passage.passage_key,
                paragraph_key=paragraph.paragraph_key,
                source_order=source_span.source_order,
                start_char=source_span.start_char,
                end_char=source_span.end_char,
            )
            span_record = PassageSourceSpanRecord(
                id=span.id,
                document_version_id=span.document_version_id,
                passage_id=span.passage_id,
                passage_document_version_id=span.document_version_id,
                paragraph_id=span.paragraph_id,
                paragraph_document_version_id=paragraph.document_version_id,
                source_order=span.source_order,
                start_char=span.start_char,
                end_char=span.end_char,
                span_key=span.span_key,
            )
            lineages.append(PassageSourceSpanLineage(span=span_record, paragraph=paragraph_record))
        records.append(
            PassageProjectionRecords(
                passage=passage_record,
                version=version_record,
                document=document_record,
                artifact=artifact_record,
                section=None,
                source_spans=tuple(lineages),
                identifiers=(),
            )
        )
    return tuple(records)


def scifact_projection_records(
    documents: Sequence[SciFactCorpusDocument],
    *,
    chunker: StructureAwareChunker | None = None,
) -> tuple[PassageProjectionRecords, ...]:
    """Build production projection records for the pinned BEIR abstract corpus."""
    rows = tuple(documents)
    ids = tuple(row.document_id for row in rows)
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("SciFact projection requires unique, non-empty corpus rows")
    title_counts = Counter(row.title.strip() for row in rows)
    planner = chunker if chunker is not None else StructureAwareChunker()
    records: list[PassageProjectionRecords] = []
    for row in rows:
        logical_title = row.title.strip()
        if title_counts[logical_title] > 1:
            logical_title = f"{logical_title} [BEIR SciFact ID {row.document_id}]"
        records.extend(
            _projection_records_for_document(row, logical_title=logical_title, chunker=planner)
        )
    return tuple(records)
