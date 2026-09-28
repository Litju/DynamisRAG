"""Canonical materialization of parsed JATS articles (RES-133).

``JatsCanonicalImporter`` converts a :class:`~dynamisrag.jats.parser.ParsedJatsArticle`
plus its :class:`~dynamisrag.domain.contracts.SourceArtifact` into the
canonical domain records and persists them:

    verify source integrity (SHA-256 + byte size against the artifact)
    -> pure parse (JatsParser)
    -> resolve/create the logical Document through identifier aliases
    -> build the deterministic DocumentVersion identity
    -> idempotent reparse check (same version_key -> created=False)
    -> append newly discovered identifier aliases
    -> persist version, sections, paragraphs, citations, tables, figures

The importer never commits: the caller owns the transaction, so one article
canonicalization is atomic — a failure halfway through materialization
leaves no partial canonical graph. No Passage rows are created: paragraphs
are source structure, chunking is RES-134.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from dynamisrag.db.canonical import (
    get_document_by_identifier,
    get_document_identifiers,
    get_document_version_by_key,
    insert_citation,
    insert_document,
    insert_document_identifier,
    insert_document_table,
    insert_document_version,
    insert_figure,
    insert_paragraph,
    insert_section,
    list_citations,
    list_document_tables,
    list_figures,
    list_paragraphs,
    list_sections,
)
from dynamisrag.db.models import DocumentRecord, DocumentVersionRecord, SectionRecord
from dynamisrag.domain.contracts import (
    Citation,
    Document,
    DocumentIdentifier,
    DocumentTable,
    DocumentVersion,
    Figure,
    Paragraph,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.values import DocumentType, IdentifierNamespace
from dynamisrag.jats.errors import JatsDocumentIdentityConflict, JatsSourceIntegrityError
from dynamisrag.jats.parser import (
    JATS_NORMALIZER_REVISION,
    JATS_PARSER_REVISION,
    JatsParser,
    ParsedJatsArticle,
)

__all__ = ["JatsCanonicalImporter", "JatsImportCounts", "JatsImportResult"]

_PMCID_VALUE_FORMAT = r"PMC[0-9]{1,12}"
"""The canonical PMCID value format, mirrored from the domain contract."""


@dataclass(frozen=True)
class JatsImportCounts:
    """How many canonical child records one materialization produced."""

    sections: int
    paragraphs: int
    citations: int
    tables: int
    figures: int


@dataclass(frozen=True)
class JatsImportResult:
    """The outcome of one materialization: the logical Document, the
    canonical DocumentVersion, whether this call created it, and the child
    record counts."""

    document: Document
    version: DocumentVersion
    created: bool
    counts: JatsImportCounts


class JatsCanonicalImporter:
    """Materializes parsed JATS articles into the canonical document model."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def import_artifact(self, artifact: SourceArtifact, xml_bytes: bytes) -> JatsImportResult:
        """Materialize one acquired JATS artifact into the canonical model.

        Idempotent by deterministic identity: re-parsing the same artifact
        under the same parser/normalizer revisions returns the persisted
        version with ``created=False`` and no duplicate rows. Raises
        :class:`JatsSourceIntegrityError` when the bytes do not match the
        artifact's recorded provenance and
        :class:`JatsDocumentIdentityConflict` when the parsed identifiers
        resolve to different existing Documents.
        """
        parsed = self.parse_verified(artifact, xml_bytes)
        candidates = candidate_identifiers(artifact, parsed)
        document = self._resolve_document(artifact, parsed, candidates)
        version = self._build_version(document, artifact, parsed)
        existing = get_document_version_by_key(self._session, version.version_key)
        if existing is not None:
            return JatsImportResult(
                document=document,
                version=self._version_from_record(document, artifact, existing),
                created=False,
                counts=self._counts(existing.id),
            )
        self._attach_missing_aliases(document, candidates)
        insert_document_version(self._session, version)
        section_records = self._insert_sections(version, parsed)
        self._insert_paragraphs(version, parsed, section_records)
        self._insert_citations(version, parsed)
        self._insert_tables(version, parsed, section_records)
        self._insert_figures(version, parsed, section_records)
        return JatsImportResult(
            document=document,
            version=version,
            created=True,
            counts=self._counts(version.id),
        )

    # ------------------------------------------------------------------
    # Source integrity and parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_verified(artifact: SourceArtifact, xml_bytes: bytes) -> ParsedJatsArticle:
        """Prove the supplied bytes are the artifact's exact bytes before
        parsing; never parse bytes under the wrong artifact provenance."""
        content_sha256 = hashlib.sha256(xml_bytes).hexdigest()
        if content_sha256 != artifact.content_sha256 or len(xml_bytes) != artifact.byte_size:
            raise JatsSourceIntegrityError(
                f"source bytes do not match SourceArtifact {artifact.artifact_key!r}: "
                f"computed sha256 {content_sha256} over {len(xml_bytes)} bytes, "
                f"artifact records {artifact.content_sha256} over {artifact.byte_size} bytes"
            )
        return JatsParser().parse(xml_bytes)

    # ------------------------------------------------------------------
    # Logical document resolution
    # ------------------------------------------------------------------

    def _resolve_document(
        self,
        artifact: SourceArtifact,
        parsed: ParsedJatsArticle,
        candidates: Sequence[tuple[IdentifierNamespace, str]],
    ) -> Document:
        """Resolve the logical Document through identifier aliases.

        Zero matches -> create a new Document (strongest identifier fixes
        its canonical key at creation); one match -> reuse it; multiple
        different matches -> explicit identity conflict. Documents are
        never merged heuristically by title.
        """
        matches: dict[UUID, DocumentRecord] = {}
        match_details: list[str] = []
        for namespace, value in candidates:
            record = get_document_by_identifier(self._session, namespace, value)
            if record is not None:
                matches[record.id] = record
                match_details.append(f"{namespace.value}:{value} -> {record.canonical_key}")
        if len(matches) > 1:
            raise JatsDocumentIdentityConflict(
                "parsed identifiers resolve to different documents and documents are "
                "never merged heuristically: " + "; ".join(sorted(match_details))
            )
        if len(matches) == 1:
            return self._document_from_record(next(iter(matches.values())))
        return self._create_document(parsed, artifact, candidates)

    def _create_document(
        self,
        parsed: ParsedJatsArticle,
        artifact: SourceArtifact,
        candidates: Sequence[tuple[IdentifierNamespace, str]],
    ) -> Document:
        """Create the logical Document for a new article.

        Identity follows the canonical contract's strongest-known-identifier
        precedence (DOI -> PMID -> PMCID -> title); the acquired Europe PMC
        PMCID is itself a known identifier even when the XML does not repeat
        it. The source article-type is preserved in version metadata, not by
        expanding the global DocumentType enum.
        """
        pmcid = parsed.pmcid
        if pmcid is None and artifact.source_system == "europe_pmc":
            candidate = artifact.source_external_id.strip()
            if re.fullmatch(_PMCID_VALUE_FORMAT, candidate):
                pmcid = candidate
        document = Document(
            document_type=DocumentType.JOURNAL_ARTICLE,
            doi=parsed.doi,
            pmid=parsed.pmid,
            pmcid=pmcid,
            title=parsed.title,
        )
        insert_document(self._session, document)
        self._attach_missing_aliases(document, candidates)
        return document

    def _attach_missing_aliases(
        self, document: Document, candidates: Sequence[tuple[IdentifierNamespace, str]]
    ) -> None:
        """Append identifier aliases that are not attached yet.

        Enrichment only ever adds aliases: an existing Document's canonical
        key is never recomputed or replaced.
        """
        existing = get_document_identifiers(self._session, document.id)
        existing_pairs = {(alias.namespace, alias.normalized_value) for alias in existing}
        for namespace, value in candidates:
            if (namespace.value, value) in existing_pairs:
                continue
            insert_document_identifier(
                self._session,
                DocumentIdentifier(
                    document_id=document.id,
                    namespace=namespace,
                    normalized_value=value,
                ),
            )
            existing_pairs.add((namespace.value, value))

    def _document_from_record(self, record: DocumentRecord) -> Document:
        """Rebuild the domain Document from a persisted record and its
        aliases, reproducing the canonical key fixed at creation."""
        doi: str | None = None
        pmid: str | None = None
        pmcid: str | None = None
        for alias in get_document_identifiers(self._session, record.id):
            if alias.namespace == IdentifierNamespace.DOI.value:
                doi = alias.normalized_value
            elif alias.namespace == IdentifierNamespace.PMID.value:
                pmid = alias.normalized_value
            elif alias.namespace == IdentifierNamespace.PMCID.value:
                pmcid = alias.normalized_value
        return Document(
            id=record.id,
            document_type=DocumentType(record.document_type),
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            title=record.title,
        )

    # ------------------------------------------------------------------
    # Version identity and idempotent reparse
    # ------------------------------------------------------------------

    @staticmethod
    def _build_version(
        document: Document, artifact: SourceArtifact, parsed: ParsedJatsArticle
    ) -> DocumentVersion:
        return DocumentVersion(
            document_id=document.id,
            document_canonical_key=document.canonical_key,
            source_artifact_id=artifact.id,
            source_artifact_key=artifact.artifact_key,
            parser_revision=JATS_PARSER_REVISION,
            normalizer_revision=JATS_NORMALIZER_REVISION,
            content_fingerprint=parsed.content_fingerprint,
            title=parsed.title,
            language=parsed.language,
            versioned_metadata=dict(parsed.metadata),
            created_at=datetime.now(UTC),
        )

    def _version_from_record(
        self, document: Document, artifact: SourceArtifact, record: DocumentVersionRecord
    ) -> DocumentVersion:
        """Rebuild the exact persisted version contract (same version_key)
        for the idempotent reparse result."""
        return DocumentVersion(
            id=record.id,
            document_id=document.id,
            document_canonical_key=document.canonical_key,
            source_artifact_id=artifact.id,
            source_artifact_key=artifact.artifact_key,
            parser_revision=record.parser_revision,
            normalizer_revision=record.normalizer_revision,
            content_fingerprint=record.content_fingerprint,
            title=record.title,
            language=record.language,
            versioned_metadata=record.versioned_metadata,
            created_at=record.created_at,
        )

    # ------------------------------------------------------------------
    # Child materialization
    # ------------------------------------------------------------------

    def _insert_sections(
        self, version: DocumentVersion, parsed: ParsedJatsArticle
    ) -> dict[str, SectionRecord]:
        """Persist sections in document order, mapping each source anchor to
        its persisted record so children can be wired to surrogate ids while
        keeping deterministic semantic keys."""
        records: dict[str, SectionRecord] = {}
        for parsed_section in parsed.sections:
            parent = (
                records.get(parsed_section.parent_anchor)
                if parsed_section.parent_anchor is not None
                else None
            )
            section = Section(
                document_version_id=version.id,
                version_key=version.version_key,
                ordinal=parsed_section.ordinal,
                depth=parsed_section.depth,
                title=parsed_section.title,
                semantic_type=parsed_section.semantic_type,
                source_anchor=parsed_section.source_anchor,
                structural_path=parsed_section.structural_path,
                content_fingerprint=parsed_section.content_fingerprint,
                parent_section_id=parent.id if parent is not None else None,
            )
            record = insert_section(self._session, section)
            records[parsed_section.source_anchor] = record
        return records

    def _insert_paragraphs(
        self,
        version: DocumentVersion,
        parsed: ParsedJatsArticle,
        section_records: Mapping[str, SectionRecord],
    ) -> None:
        for parsed_paragraph in parsed.paragraphs:
            section = (
                section_records.get(parsed_paragraph.section_anchor)
                if parsed_paragraph.section_anchor is not None
                else None
            )
            insert_paragraph(
                self._session,
                Paragraph(
                    document_version_id=version.id,
                    version_key=version.version_key,
                    section_id=section.id if section is not None else None,
                    ordinal=parsed_paragraph.ordinal,
                    region=parsed_paragraph.region,
                    source_anchor=parsed_paragraph.source_anchor,
                    text=parsed_paragraph.text,
                    content_sha256=parsed_paragraph.content_sha256,
                ),
            )

    def _insert_citations(self, version: DocumentVersion, parsed: ParsedJatsArticle) -> None:
        for parsed_citation in parsed.citations:
            insert_citation(
                self._session,
                Citation(
                    document_version_id=version.id,
                    version_key=version.version_key,
                    ordinal=parsed_citation.ordinal,
                    source_reference_id=parsed_citation.source_reference_id,
                    source_anchor=parsed_citation.source_anchor,
                    doi=parsed_citation.doi,
                    pmid=parsed_citation.pmid,
                    pmcid=parsed_citation.pmcid,
                    title=parsed_citation.title,
                    year=parsed_citation.year,
                    raw_reference_text=parsed_citation.raw_reference_text,
                ),
            )

    def _insert_tables(
        self,
        version: DocumentVersion,
        parsed: ParsedJatsArticle,
        section_records: Mapping[str, SectionRecord],
    ) -> None:
        for parsed_table in parsed.tables:
            section = (
                section_records.get(parsed_table.section_anchor)
                if parsed_table.section_anchor is not None
                else None
            )
            insert_document_table(
                self._session,
                DocumentTable(
                    document_version_id=version.id,
                    version_key=version.version_key,
                    section_id=section.id if section is not None else None,
                    ordinal=parsed_table.ordinal,
                    label=parsed_table.label,
                    caption=parsed_table.caption,
                    source_anchor=parsed_table.source_anchor,
                    structured_representation=dict(parsed_table.structured_representation),
                    content_fingerprint=parsed_table.content_fingerprint,
                ),
            )

    def _insert_figures(
        self,
        version: DocumentVersion,
        parsed: ParsedJatsArticle,
        section_records: Mapping[str, SectionRecord],
    ) -> None:
        for parsed_figure in parsed.figures:
            section = (
                section_records.get(parsed_figure.section_anchor)
                if parsed_figure.section_anchor is not None
                else None
            )
            insert_figure(
                self._session,
                Figure(
                    document_version_id=version.id,
                    version_key=version.version_key,
                    section_id=section.id if section is not None else None,
                    ordinal=parsed_figure.ordinal,
                    label=parsed_figure.label,
                    caption=parsed_figure.caption,
                    source_anchor=parsed_figure.source_anchor,
                    asset_locator=parsed_figure.asset_locator,
                    content_fingerprint=parsed_figure.content_fingerprint,
                ),
            )

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _counts(self, version_id: UUID) -> JatsImportCounts:
        return JatsImportCounts(
            sections=len(list_sections(self._session, version_id)),
            paragraphs=len(list_paragraphs(self._session, version_id)),
            citations=len(list_citations(self._session, version_id)),
            tables=len(list_document_tables(self._session, version_id)),
            figures=len(list_figures(self._session, version_id)),
        )


def candidate_identifiers(
    artifact: SourceArtifact, parsed: ParsedJatsArticle
) -> list[tuple[IdentifierNamespace, str]]:
    """Every strong identifier this materialization can assert: the parsed
    article identifiers plus, for a Europe PMC artifact, the acquired PMCID
    itself — which is strong provenance even when the XML does not repeat
    it."""
    candidates: list[tuple[IdentifierNamespace, str]] = []
    if parsed.doi is not None:
        candidates.append((IdentifierNamespace.DOI, parsed.doi))
    if parsed.pmid is not None:
        candidates.append((IdentifierNamespace.PMID, parsed.pmid))
    if parsed.pmcid is not None:
        candidates.append((IdentifierNamespace.PMCID, parsed.pmcid))
    if artifact.source_system == "europe_pmc":
        pmcid = artifact.source_external_id.strip()
        if (
            re.fullmatch(_PMCID_VALUE_FORMAT, pmcid)
            and (IdentifierNamespace.PMCID, pmcid) not in candidates
        ):
            candidates.append((IdentifierNamespace.PMCID, pmcid))
    return candidates
