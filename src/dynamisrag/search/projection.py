"""Deterministic versioned OpenSearch passage projection (RES-135).

    canonical PostgreSQL state (explicit chunker revision)
        -> PassageProjectionManifest (pure, frozen, byte-reproducible)
        -> physical OpenSearch index
        -> bulk documents
        -> verify
        -> atomic alias cutover

Two halves, deliberately separated:

*Pure* — :func:`build_projection_manifest` turns canonical rows into frozen
:class:`PassageProjectionDocument` values and a :class:`PassageProjectionManifest`
with a canonical byte serialization and SHA-256. No JATS is reparsed, no object
store is read, no identity is generated, no clock is consulted. The same
canonical state, chunker revision and projection schema therefore produce the
same ``projection_sha256`` in any database, even when every surrogate UUID
differs.

*Effectful* — :class:`PassageProjector` reads canonical rows, prepares a
:class:`~dynamisrag.search.publication.PublicationPlan` and hands it to
:class:`~dynamisrag.search.publication.FailClosedAliasPublisher`, which drives
OpenSearch. The projector never mutates PostgreSQL: the projection is a
disposable cache that can be deleted and rebuilt from canonical state alone.

**This module decides *what* is projected; it does not decide *how* it is
published.** The cutover ordering and the fail-closed rules — build, bulk,
verify, *then* switch atomically, and never destroy an index the stable alias
currently targets — live in :mod:`dynamisrag.search.publication`, extracted from
the RES-135 projector unchanged so that every later schema revision publishes
through one implementation rather than a second copy of the safety argument.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from sqlalchemy.orm import Session

from dynamisrag.db.canonical import (
    PassageProjectionRecords,
    list_passage_chunker_revisions,
    list_passage_projection_records,
)
from dynamisrag.domain.values import IdentifierNamespace
from dynamisrag.search.client import JsonValue, OpenSearchClient, canonical_json_line
from dynamisrag.search.errors import ProjectionError
from dynamisrag.search.publication import (
    DEFAULT_BULK_BATCH_SIZE,
    FailClosedAliasPublisher,
    PublicationPlan,
    PublicationResult,
)
from dynamisrag.search.schema import (
    PASSAGE_INDEX_SCHEMA_REVISION,
    index_mappings,
    index_meta,
    index_settings,
    physical_index_name,
)

__all__ = [
    "DEFAULT_BULK_BATCH_SIZE",
    "PassageProjectionDocument",
    "PassageProjectionManifest",
    "PassageProjector",
    "ProjectionResult",
    "ProjectionSourceSpan",
    "build_projection_manifest",
]


@dataclass(frozen=True)
class ProjectionSourceSpan:
    """One exact semantic source span, as projected.

    Semantic values only — the paragraph is addressed by its deterministic key
    and its stable source anchor, never by a surrogate UUID. This is the
    canonical ``PassageSourceSpan`` representation reduced to what makes a hit
    auditable back to exact source characters.
    """

    source_order: int
    paragraph_key: str
    paragraph_source_anchor: str
    start_char: int
    end_char: int

    def payload(self) -> Mapping[str, JsonValue]:
        return {
            "source_order": self.source_order,
            "paragraph_key": self.paragraph_key,
            "paragraph_source_anchor": self.paragraph_source_anchor,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True)
class PassageProjectionDocument:
    """One projected passage: identity, searchable text, structure, provenance.

    Frozen and semantic. The field set is exactly the passage index mapping:
    anything absent here could not be indexed, and anything present but not
    here would be rejected by ``dynamic: strict``.
    """

    passage_key: str
    document_canonical_key: str
    document_version_key: str
    chunker_revision: str
    passage_ordinal: int
    content_sha256: str
    text: str
    title: str
    language: str
    document_type: str
    token_count: int
    source_system: str
    source_external_id: str
    doi: str | None
    pmid: str | None
    pmcid: str | None
    section_key: str | None
    section_path: str | None
    section_title: str | None
    section_source_anchor: str | None
    primary_source_anchor: str | None
    source_spans: tuple[ProjectionSourceSpan, ...]

    def payload(self) -> Mapping[str, JsonValue]:
        """The semantic document, without projection-level provenance.

        The manifest hashes exactly this. The two projection provenance fields
        depend on the manifest digest, so they are added afterwards by
        :meth:`to_source`.
        """
        return {
            "passage_key": self.passage_key,
            "document_canonical_key": self.document_canonical_key,
            "document_version_key": self.document_version_key,
            "chunker_revision": self.chunker_revision,
            "passage_ordinal": self.passage_ordinal,
            "content_sha256": self.content_sha256,
            "text": self.text,
            "title": self.title,
            "language": self.language,
            "document_type": self.document_type,
            "token_count": self.token_count,
            "source_system": self.source_system,
            "source_external_id": self.source_external_id,
            "doi": self.doi,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "section_key": self.section_key,
            "section_path": self.section_path,
            "section_title": self.section_title,
            "section_source_anchor": self.section_source_anchor,
            "primary_source_anchor": self.primary_source_anchor,
            "source_spans": [span.payload() for span in self.source_spans],
        }

    def to_source(self, *, projection_sha256: str) -> Mapping[str, JsonValue]:
        """The exact ``_source`` document, including projection provenance.

        Every indexed document carries the projection schema revision, the
        projection digest and the chunker revision that produced it, so a stored
        document states the semantics under which it was written.
        """
        return {
            "projection_schema_revision": PASSAGE_INDEX_SCHEMA_REVISION,
            "projection_sha256": projection_sha256,
            **self.payload(),
        }


@dataclass(frozen=True)
class PassageProjectionManifest:
    """The frozen, byte-reproducible projection of one canonical snapshot."""

    schema_revision: str
    chunker_revision: str
    documents: tuple[PassageProjectionDocument, ...]

    def __post_init__(self) -> None:
        keys = [document.passage_key for document in self.documents]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError(
                "projection documents must be sorted by passage_key and unique; a manifest "
                "whose order is not canonical cannot produce reproducible bytes"
            )

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def projection_bytes(self) -> bytes:
        """Canonical serialization of the manifest, UTF-8 encoded."""
        return canonical_json_line(self._payload()).encode("utf-8")

    @property
    def projection_sha256(self) -> str:
        """SHA-256 of :attr:`projection_bytes`."""
        return hashlib.sha256(self.projection_bytes).hexdigest()

    def index_name(self, *, alias: str) -> str:
        """Deterministic physical index name for this snapshot."""
        return physical_index_name(alias=alias, projection_sha256=self.projection_sha256)

    def expected_meta(self) -> Mapping[str, JsonValue]:
        """The mapping ``_meta`` this manifest must produce."""
        return index_meta(
            projection_sha256=self.projection_sha256, chunker_revision=self.chunker_revision
        )

    def publication_plan(self, *, alias: str) -> PublicationPlan:
        """The publication description of this snapshot, for one alias.

        The single point where the lexical schema and the fail-closed publisher
        meet: the schema revision and the mapping come from
        :mod:`dynamisrag.search.schema`, the documents and the expected
        ``_meta`` come from this manifest, and the resulting
        :class:`~dynamisrag.search.publication.PublicationPlan` carries no
        knowledge of either back into the publisher.

        Pure, so the exact bytes OpenSearch would receive are assertable without
        a node.
        """
        digest = self.projection_sha256
        return PublicationPlan(
            index_name=self.index_name(alias=alias),
            projection_sha256=digest,
            settings=index_settings(),
            mappings=index_mappings(
                projection_sha256=digest, chunker_revision=self.chunker_revision
            ),
            documents=self.source_documents(),
            expected_meta=self.expected_meta(),
        )

    def source_documents(self) -> tuple[tuple[str, Mapping[str, JsonValue]], ...]:
        """``(passage_key, _source)`` pairs in canonical order.

        The document id is the passage key, so a rebuild overwrites rather than
        duplicates and the identity is never an OpenSearch-generated value.
        """
        digest = self.projection_sha256
        return tuple(
            (document.passage_key, document.to_source(projection_sha256=digest))
            for document in self.documents
        )

    def _payload(self) -> Mapping[str, JsonValue]:
        return {
            "schema_revision": self.schema_revision,
            "chunker_revision": self.chunker_revision,
            "document_count": self.document_count,
            "documents": [document.payload() for document in self.documents],
        }


def build_projection_manifest(
    records: Sequence[PassageProjectionRecords], *, chunker_revision: str
) -> PassageProjectionManifest:
    """Build the deterministic manifest from canonical rows.

    Pure: the input is a materialised sequence of records, so this is
    reproducible in a unit test with no database, no node and no clock. Output
    order is by ``passage_key``, independent of the order rows were read in.
    """
    documents = sorted(
        (_document_from_record(record, chunker_revision=chunker_revision) for record in records),
        key=lambda document: document.passage_key,
    )
    return PassageProjectionManifest(
        schema_revision=PASSAGE_INDEX_SCHEMA_REVISION,
        chunker_revision=chunker_revision,
        documents=tuple(documents),
    )


def _document_from_record(
    record: PassageProjectionRecords, *, chunker_revision: str
) -> PassageProjectionDocument:
    """Interpret one passage's canonical rows as a projection document."""
    identifiers = {
        identifier.namespace: identifier.normalized_value for identifier in record.identifiers
    }
    section = record.section
    return PassageProjectionDocument(
        passage_key=record.passage.passage_key,
        document_canonical_key=record.document.canonical_key,
        document_version_key=record.version.version_key,
        chunker_revision=chunker_revision,
        passage_ordinal=record.passage.ordinal,
        content_sha256=record.passage.content_sha256,
        text=record.passage.text,
        title=record.version.title,
        language=record.version.language,
        document_type=record.document.document_type,
        token_count=(record.passage.token_count if record.passage.token_count is not None else 0),
        source_system=record.artifact.source_system,
        source_external_id=record.artifact.source_external_id,
        doi=identifiers.get(IdentifierNamespace.DOI.value),
        pmid=identifiers.get(IdentifierNamespace.PMID.value),
        pmcid=identifiers.get(IdentifierNamespace.PMCID.value),
        section_key=section.section_key if section is not None else None,
        section_path=section.structural_path if section is not None else None,
        section_title=section.title if section is not None else None,
        section_source_anchor=section.source_anchor if section is not None else None,
        primary_source_anchor=record.passage.source_anchor,
        source_spans=tuple(
            ProjectionSourceSpan(
                source_order=lineage.span.source_order,
                paragraph_key=lineage.paragraph.paragraph_key,
                paragraph_source_anchor=lineage.paragraph.source_anchor,
                start_char=lineage.span.start_char,
                end_char=lineage.span.end_char,
            )
            for lineage in record.source_spans
        ),
    )


@dataclass(frozen=True)
class ProjectionResult:
    """The outcome of one projection run, safe to render as JSON."""

    created: bool
    chunker_revision: str
    projection_schema_revision: str
    projection_sha256: str
    document_count: int
    index_name: str
    alias: str
    removed_index_names: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "created": self.created,
            "chunker_revision": self.chunker_revision,
            "projection_schema_revision": self.projection_schema_revision,
            "projection_sha256": self.projection_sha256,
            "document_count": self.document_count,
            "index_name": self.index_name,
            "alias": self.alias,
            "removed_index_names": list(self.removed_index_names),
        }


class PassageProjector:
    """Rebuilds the OpenSearch passage projection from canonical PostgreSQL."""

    __slots__ = ("_publisher", "_session")

    def __init__(
        self,
        session: Session,
        client: OpenSearchClient,
        *,
        alias: str,
        batch_size: int = DEFAULT_BULK_BATCH_SIZE,
    ) -> None:
        """Bind a projector to one read session and one OpenSearch client.

        ``session`` is used for reading only; the projector never writes to
        PostgreSQL. Every OpenSearch operation is delegated to the shared
        fail-closed publisher, which is bound here to this client's alias and
        batch size.
        """
        self._session: Final[Session] = session
        self._publisher: Final[FailClosedAliasPublisher] = FailClosedAliasPublisher(
            client, alias=alias, batch_size=batch_size
        )

    def project(self, *, chunker_revision: str) -> ProjectionResult:
        """Project exactly ``chunker_revision``'s passages and make them live.

        The revision is an explicit, mandatory selection: PostgreSQL may hold
        several immutable passage sets for one document version, and indexing
        more than one would duplicate retrieval content under different
        identities. Timestamps are never consulted to pick a "latest" revision.
        """
        manifest = self._manifest(chunker_revision=chunker_revision)
        publication = self._publisher.publish(
            manifest.publication_plan(alias=self._publisher.alias)
        )
        return self._result(manifest, publication)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _manifest(self, *, chunker_revision: str) -> PassageProjectionManifest:
        """Read canonical rows and build the manifest, failing loudly on an
        empty selection.

        An empty projection would silently replace a live alias with an index
        holding nothing, so it is never produced. When the requested revision
        has no passages the revisions that *do* exist are named, because the
        actionable cause is almost always a mismatched revision string.
        """
        records = list_passage_projection_records(self._session, chunker_revision=chunker_revision)
        if not records:
            raise self._no_passages_error(chunker_revision)
        return build_projection_manifest(records, chunker_revision=chunker_revision)

    def _no_passages_error(self, chunker_revision: str) -> ProjectionError:
        available = list_passage_chunker_revisions(self._session)
        if not available:
            detail = (
                f"projection of chunker revision {chunker_revision!r} found no passages: the "
                "canonical corpus contains no passages at all. Chunk and materialize the "
                "canonical source structure before projecting."
            )
        else:
            detail = (
                f"projection of chunker revision {chunker_revision!r} found no passages; the "
                f"canonical corpus contains passages only for {list(available)}. Project the "
                "revision that actually exists instead of guessing one."
            )
        return ProjectionError(detail, operation="project")

    @staticmethod
    def _result(
        manifest: PassageProjectionManifest, publication: PublicationResult
    ) -> ProjectionResult:
        """Add the lexical projection's semantics to the schema-agnostic outcome.

        The publisher reports what happened to the index and the alias; only
        this module knows which projection was published, so only this module
        can state its schema and chunker revisions. Kept as a separate step so
        the published bytes and the reported bytes stay independent
        descriptions that a test can hold against each other.
        """
        return ProjectionResult(
            created=publication.created,
            chunker_revision=manifest.chunker_revision,
            projection_schema_revision=manifest.schema_revision,
            projection_sha256=manifest.projection_sha256,
            document_count=publication.document_count,
            index_name=publication.index_name,
            alias=publication.alias,
            removed_index_names=publication.removed_index_names,
        )
