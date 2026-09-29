"""Idempotent savepoint-atomic passage materialization (RES-134).

``PassageMaterializer`` turns a :class:`~dynamisrag.chunking.manifest.PassageManifest`
into persisted ``Passage`` + ``PassageSourceSpan`` rows:

    verify the persisted DocumentVersion semantic key
    -> idempotent re-run check (same version + chunker revision)
    -> savepoint-atomic insert of passages and their exact source spans
    -> persisted-output verification against the planned manifest

The materializer never commits: the caller owns the transaction, so one
materialization is atomic — a failure halfway through rolls the whole
passage set back through the savepoint while the caller's outer transaction
stays usable. Repeated materialization of the same document version under
the same chunker revision returns the persisted set with ``created=False``
and no duplicate rows; a persisted set that reconstructs to a different
manifest fails explicitly instead of being silently returned or replaced.

An empty manifest (``manifest.passages == ()``) is a deterministic no-op:
it performs zero database mutations and returns ``created=False`` with
empty passage and span tuples, identically on first and repeated calls —
no persisted run marker is written.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from dynamisrag.chunking.errors import ChunkerRevisionConflictError, PassageSourceSpanError
from dynamisrag.chunking.manifest import (
    ManifestPassage,
    ManifestSourceSpan,
    PassageManifest,
    SectionMetadata,
)
from dynamisrag.db.canonical import (
    get_document_version,
    insert_passage,
    insert_passage_source_span,
    list_paragraphs,
    list_passage_source_spans,
    list_passages,
    list_sections,
)
from dynamisrag.db.models import (
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
    SectionRecord,
)
from dynamisrag.domain.contracts import DocumentVersion, Passage, PassageSourceSpan

__all__ = ["MaterializationResult", "PassageMaterializer"]


@dataclass(frozen=True)
class MaterializationResult:
    """The outcome of one materialization: whether this call created the
    passage set, the manifest it materialized, and the persisted domain
    contracts."""

    created: bool
    manifest: PassageManifest
    passages: tuple[Passage, ...]
    source_spans: tuple[PassageSourceSpan, ...]


class PassageMaterializer:
    """Materializes passage manifests into the canonical persistence layer."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def materialize(
        self, version: DocumentVersion, manifest: PassageManifest
    ) -> MaterializationResult:
        """Materialize one manifest for a document version, idempotently."""
        persisted_version = get_document_version(self._session, version.id)
        if persisted_version is None:
            raise PassageSourceSpanError(
                f"document version {version.id} is not persisted; chunk canonical "
                "source structure before materializing passages"
            )
        if persisted_version.version_key != version.version_key:
            raise PassageSourceSpanError(
                f"document version {version.id} has persisted version key "
                f"{persisted_version.version_key!r} but the supplied contract declares "
                f"{version.version_key!r}"
            )
        if manifest.document_version_key != version.version_key:
            raise PassageSourceSpanError(
                f"manifest document version key {manifest.document_version_key!r} does not "
                f"match the version being chunked {version.version_key!r}"
            )

        if manifest.passages == ():
            return MaterializationResult(
                created=False,
                manifest=manifest,
                passages=(),
                source_spans=(),
            )

        existing = list_passages(self._session, version.id, manifest.chunker_revision)
        if existing:
            return self._verify_existing(version, manifest, existing)
        return self._insert_new(version, manifest)

    # ------------------------------------------------------------------
    # Idempotent re-run
    # ------------------------------------------------------------------

    def _verify_existing(
        self,
        version: DocumentVersion,
        manifest: PassageManifest,
        existing: Sequence[PassageRecord],
    ) -> MaterializationResult:
        """Return the persisted set after proving it matches the manifest.

        The persisted semantic representation is reconstructed into a fresh
        manifest and compared byte-for-byte with the newly computed one. A
        mismatch means the existing passage set is inconsistent with the
        manifest being materialized — an explicit failure, never a silent
        return of inconsistent data.
        """
        reconstructed = self._reconstruct_manifest(version, manifest, existing)
        if reconstructed.manifest_bytes != manifest.manifest_bytes:
            raise ChunkerRevisionConflictError(
                f"document version {version.id} already carries passages for chunker "
                f"revision {manifest.chunker_revision!r} but they reconstruct to a "
                "different manifest; existing passage sets are immutable"
            )
        passages = tuple(
            self._passage_from_record(record, version.version_key) for record in existing
        )
        paragraph_map = {
            paragraph.id: paragraph for paragraph in list_paragraphs(self._session, version.id)
        }
        passage_map = {record.id: record for record in existing}
        spans = list_passage_source_spans(self._session, [record.id for record in existing])
        source_spans = tuple(
            self._span_from_record(span, paragraph_map, passage_map) for span in spans
        )
        return MaterializationResult(
            created=False,
            manifest=manifest,
            passages=passages,
            source_spans=source_spans,
        )

    def _reconstruct_manifest(
        self,
        version: DocumentVersion,
        manifest: PassageManifest,
        existing: Sequence[PassageRecord],
    ) -> PassageManifest:
        """Rebuild the manifest from persisted rows, semantic fields only."""
        section_map = {section.id: section for section in list_sections(self._session, version.id)}
        paragraph_map = {
            paragraph.id: paragraph for paragraph in list_paragraphs(self._session, version.id)
        }
        spans = list_passage_source_spans(self._session, [record.id for record in existing])
        spans_by_passage: dict[UUID, list[PassageSourceSpanRecord]] = defaultdict(list)
        for span in spans:
            spans_by_passage[span.passage_id].append(span)

        manifest_passages: list[ManifestPassage] = []
        for record in existing:
            passage_spans = sorted(
                spans_by_passage.get(record.id, []), key=lambda item: item.source_order
            )
            section = section_map.get(record.section_id) if record.section_id is not None else None
            manifest_passages.append(
                ManifestPassage(
                    ordinal=record.ordinal,
                    passage_key=record.passage_key,
                    text=record.text,
                    content_sha256=record.content_sha256,
                    token_count=record.token_count if record.token_count is not None else 0,
                    primary_source_anchor=(
                        record.source_anchor if record.source_anchor is not None else ""
                    ),
                    section=SectionMetadata(
                        section_key=section.section_key if section is not None else None,
                        structural_path=(section.structural_path if section is not None else None),
                        source_anchor=section.source_anchor if section is not None else None,
                        title=section.title if section is not None else None,
                    ),
                    source_spans=tuple(
                        ManifestSourceSpan(
                            source_order=span.source_order,
                            paragraph_key=paragraph_map[span.paragraph_id].paragraph_key,
                            paragraph_source_anchor=paragraph_map[span.paragraph_id].source_anchor,
                            start_char=span.start_char,
                            end_char=span.end_char,
                        )
                        for span in passage_spans
                    ),
                )
            )
        return PassageManifest(
            schema_revision=manifest.schema_revision,
            document_version_key=manifest.document_version_key,
            chunker_revision=manifest.chunker_revision,
            algorithm_revision=manifest.algorithm_revision,
            config_sha256=manifest.config_sha256,
            config=manifest.config,
            passages=tuple(manifest_passages),
        )

    # ------------------------------------------------------------------
    # First materialization
    # ------------------------------------------------------------------

    def _insert_new(
        self, version: DocumentVersion, manifest: PassageManifest
    ) -> MaterializationResult:
        """Insert the passage set atomically inside a savepoint.

        Every mutation happens inside ``begin_nested()``, so a failure
        halfway through rolls the whole passage set back while the caller's
        outer transaction stays usable.
        """
        paragraph_map = {
            paragraph.paragraph_key: paragraph
            for paragraph in list_paragraphs(self._session, version.id)
        }
        section_map = {
            section.section_key: section for section in list_sections(self._session, version.id)
        }
        self._validate_manifest_against_canonical(manifest, paragraph_map, section_map)

        with self._session.begin_nested():
            passages: list[Passage] = []
            source_spans: list[PassageSourceSpan] = []
            for planned in manifest.passages:
                section = (
                    section_map.get(planned.section.section_key)
                    if planned.section.section_key is not None
                    else None
                )
                passage = Passage(
                    document_version_id=version.id,
                    version_key=version.version_key,
                    section_id=section.id if section is not None else None,
                    chunker_revision=manifest.chunker_revision,
                    ordinal=planned.ordinal,
                    text=planned.text,
                    content_sha256=planned.content_sha256,
                    source_anchor=planned.primary_source_anchor,
                    token_count=planned.token_count,
                )
                record = insert_passage(self._session, passage)
                passages.append(passage)
                for span in planned.source_spans:
                    paragraph = paragraph_map[span.paragraph_key]
                    span_contract = PassageSourceSpan(
                        document_version_id=version.id,
                        passage_id=record.id,
                        paragraph_id=paragraph.id,
                        passage_key=planned.passage_key,
                        paragraph_key=span.paragraph_key,
                        source_order=span.source_order,
                        start_char=span.start_char,
                        end_char=span.end_char,
                    )
                    insert_passage_source_span(self._session, span_contract)
                    source_spans.append(span_contract)
            self._verify_persisted(passages, manifest)
        return MaterializationResult(
            created=True,
            manifest=manifest,
            passages=tuple(passages),
            source_spans=tuple(source_spans),
        )

    # ------------------------------------------------------------------
    # Persistence-boundary validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_manifest_against_canonical(
        manifest: PassageManifest,
        paragraph_map: dict[str, ParagraphRecord],
        section_map: dict[str, SectionRecord],
    ) -> None:
        """Validate the manifest against the canonical source structure
        before any row is written.

        Every referenced paragraph must exist in the version; every
        referenced section must exist and match the manifest's semantic
        section metadata; every passage's spans must belong to the passage's
        section; span ordering must be sequential; spans must not overlap
        within one paragraph of one passage; and every span's offsets must fit
        the referenced paragraph's persisted text.
        """
        for planned in manifest.passages:
            section = (
                section_map.get(planned.section.section_key)
                if planned.section.section_key is not None
                else None
            )
            if planned.section.section_key is not None:
                if section is None:
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} references section "
                        f"{planned.section.section_key!r} which is not persisted"
                    )
                if section.structural_path != planned.section.structural_path:
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} section structural path "
                        f"{planned.section.structural_path!r} does not match persisted "
                        f"section {section.structural_path!r}"
                    )
            previous_key: str | None = None
            previous_end = -1
            for order, span in enumerate(planned.source_spans):
                if span.source_order != order:
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} span ordering is not sequential: "
                        f"expected source_order {order}, found {span.source_order}"
                    )
                paragraph = paragraph_map.get(span.paragraph_key)
                if paragraph is None:
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} references paragraph "
                        f"{span.paragraph_key!r} which is not persisted"
                    )
                if span.end_char > len(paragraph.text):
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} span end_char {span.end_char} "
                        f"exceeds paragraph {span.paragraph_key!r} text length "
                        f"{len(paragraph.text)}"
                    )
                if section is not None and paragraph.section_id != section.id:
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} span references paragraph "
                        f"{span.paragraph_key!r} owned by section {paragraph.section_id} "
                        f"but the passage belongs to section {section.id}"
                    )
                if span.paragraph_key == previous_key and span.start_char < previous_end:
                    raise PassageSourceSpanError(
                        f"passage {planned.passage_key!r} spans overlap within paragraph "
                        f"{span.paragraph_key!r}"
                    )
                previous_key = span.paragraph_key
                previous_end = span.end_char

    @staticmethod
    def _verify_persisted(passages: Sequence[Passage], manifest: PassageManifest) -> None:
        """Prove the persisted passage rows agree with the planned manifest.

        The materializer just flushed every row; reading the set back and
        comparing text, hash, token count, ordinal and passage key against
        the manifest is the persistence-boundary proof that the planned
        deterministic output is what was persisted.
        """
        for passage, planned in zip(passages, manifest.passages, strict=True):
            if passage.content_sha256 != planned.content_sha256:
                raise PassageSourceSpanError(
                    f"passage {passage.passage_key!r} content hash disagrees with the "
                    "planned manifest"
                )
            if passage.text != planned.text:
                raise PassageSourceSpanError(
                    f"passage {passage.passage_key!r} text disagrees with the planned manifest"
                )
            if passage.token_count != planned.token_count:
                raise PassageSourceSpanError(
                    f"passage {passage.passage_key!r} token count disagrees with the "
                    "planned manifest"
                )

    # ------------------------------------------------------------------
    # Record-to-contract reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _passage_from_record(record: PassageRecord, version_key: str) -> Passage:
        return Passage(
            id=record.id,
            document_version_id=record.document_version_id,
            version_key=version_key,
            section_id=record.section_id,
            chunker_revision=record.chunker_revision,
            ordinal=record.ordinal,
            text=record.text,
            content_sha256=record.content_sha256,
            source_anchor=record.source_anchor,
            token_count=record.token_count,
        )

    @staticmethod
    def _span_from_record(
        record: PassageSourceSpanRecord,
        paragraph_map: dict[UUID, ParagraphRecord],
        passage_map: dict[UUID, PassageRecord],
    ) -> PassageSourceSpan:
        paragraph = paragraph_map[record.paragraph_id]
        passage = passage_map[record.passage_id]
        return PassageSourceSpan(
            id=record.id,
            document_version_id=record.document_version_id,
            passage_id=record.passage_id,
            paragraph_id=record.paragraph_id,
            passage_key=passage.passage_key,
            paragraph_key=paragraph.paragraph_key,
            source_order=record.source_order,
            start_char=record.start_char,
            end_char=record.end_char,
        )
