"""The first complete vertical slice, end to end.

    PMCID
      -> EuropePmcAcquisition (mocked Europe PMC HTTP)
      -> SourceArtifact (temporary filesystem object store)
      -> JatsCanonicalImporter (canonical Document/Version/Sections/Paragraphs)
      -> StructureAwareChunker + PassageMaterializer
      -> PassageProjector (live OpenSearch 3.8.0)
      -> Bm25SearchService
      -> expected search result

Everything real except the one thing that must not be real: the Europe PMC
network. A mocked transport serves a synthetic JATS article, so CI never
depends on a public service, while the live PostgreSQL 18 and OpenSearch 3.8.0
containers prove the whole chain against the infrastructure that will run it.

The closing assertions trace a returned hit back through the projection to
canonical rows: ``passage_key``, ``document_version_key``,
``document_canonical_key``, ``primary_source_anchor`` and ``source_spans`` all
resolve to the same canonical facts, and each source span's character range
addresses exactly the persisted ``Paragraph.text``.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx2
import pytest
from sqlalchemy.orm import Session

from dynamisrag.chunking import ChunkerConfig, PassageMaterializer, StructureAwareChunker
from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import (
    list_paragraphs,
    list_passages,
    list_sections,
    paragraph_from_record,
    section_from_record,
)
from dynamisrag.domain.contracts import Document, DocumentVersion, SourceArtifact
from dynamisrag.ingestion import EuropePmcAcquisition, EuropePmcClient
from dynamisrag.jats import JatsCanonicalImporter
from dynamisrag.search import Bm25SearchService, OpenSearchClient, PassageProjector
from dynamisrag.search.bm25 import SearchHit
from dynamisrag.search.projection import ProjectionResult
from dynamisrag.storage import FileSystemObjectStore
from tests._support import JATS_FULL_ARTICLE

pytestmark = pytest.mark.integration

_PMCID: Final[str] = "PMC123456"
_DOI: Final[str] = "10.1371/journal.pone.03089012"
_TITLE: Final[str] = "A synthetic study of things and numbers"
_QUERY: Final[str] = "synthetic"
_EXPECTED_SECTION: Final[str] = "Methods"
_SOURCE_URI: Final[str] = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{_PMCID}/fullTextXML"

# The article is already deterministic; the digest pins the artifact identity
# the whole slice is traced back to.
_ARTICLE_SHA256: Final[str] = hashlib.sha256(JATS_FULL_ARTICLE).hexdigest()


# ---------------------------------------------------------------------------
# The ingestion half: everything upstream of the projection
# ---------------------------------------------------------------------------


class _EuropePmcStub(httpx2.BaseTransport):
    """Serves the synthetic article for any Europe PMC request.

    One transport, no network: the slice is deterministic and CI-safe.
    """

    def __init__(self, content: bytes) -> None:
        self._content = content
        self.requests: list[str] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(str(request.url))
        return httpx2.Response(
            200,
            content=self._content,
            headers={"content-type": "application/xml"},
            request=request,
        )


@dataclass(frozen=True)
class _Ingested:
    """The canonical state one vertical-slice run produced."""

    artifact: SourceArtifact
    document: Document
    version: DocumentVersion
    chunker_revision: str
    requested_urls: tuple[str, ...]
    stored_object: Path
    passage_count: int


def _ingest(session: Session, artifacts_root: Path) -> _Ingested:
    """Acquire, canonicalize, chunk and materialize — the whole ingestion half."""
    transport = _EuropePmcStub(JATS_FULL_ARTICLE)
    client = EuropePmcClient(transport=transport)
    try:
        acquired = EuropePmcAcquisition(client, FileSystemObjectStore(artifacts_root)).acquire(
            session, _PMCID
        )
    finally:
        client.close()

    imported = JatsCanonicalImporter(session).import_artifact(acquired.artifact, JATS_FULL_ARTICLE)
    sections = [
        section_from_record(record, version_key=imported.version.version_key)
        for record in list_sections(session, imported.version.id)
    ]
    paragraphs = [
        paragraph_from_record(record, version_key=imported.version.version_key)
        for record in list_paragraphs(session, imported.version.id)
    ]
    chunker = StructureAwareChunker(ChunkerConfig())
    materialized = PassageMaterializer(session).materialize(
        imported.version, chunker.plan(imported.version, sections, paragraphs)
    )
    return _Ingested(
        artifact=acquired.artifact,
        document=imported.document,
        version=imported.version,
        chunker_revision=chunker.chunker_revision,
        requested_urls=tuple(transport.requests),
        stored_object=artifacts_root / "sha256" / _ARTICLE_SHA256[:2] / f"{_ARTICLE_SHA256}.xml",
        passage_count=len(materialized.passages),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session(live_settings: Settings) -> Iterator[Session]:
    """A session whose transaction is always rolled back, never DELETE."""
    engine = create_database_engine(live_settings)
    session = Session(bind=engine)
    transaction = session.begin()
    try:
        yield session
    finally:
        transaction.rollback()
        session.close()
        engine.dispose()


@pytest.fixture
def artifacts_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


class _Namespace:
    """One isolated OpenSearch alias, removed in teardown."""

    def __init__(self, alias: str, settings: Settings) -> None:
        self.alias = alias
        self.client = OpenSearchClient(settings)
        self.indexes: set[str] = set()

    def project(self, session: Session, *, chunker_revision: str) -> ProjectionResult:
        result = PassageProjector(session, self.client, alias=self.alias).project(
            chunker_revision=chunker_revision
        )
        self.indexes.add(result.index_name)
        return result

    def service(self) -> Bm25SearchService:
        return Bm25SearchService(self.client, alias=self.alias)

    def drop_projection(self) -> None:
        for index in self.client.alias_targets(self.alias):
            self.client.delete_index(index)
        self.indexes.clear()

    def cleanup(self) -> None:
        for index in sorted(self.indexes | set(self.client.alias_targets(self.alias))):
            with contextlib.suppress(Exception):
                self.client.delete_index(index)
        self.client.close()


@pytest.fixture
def node(live_settings: Settings) -> Iterator[_Namespace]:
    namespace = _Namespace(f"dynamisrag-e2e-{uuid.uuid4().hex[:12]}", live_settings)
    try:
        yield namespace
    finally:
        namespace.cleanup()


# ---------------------------------------------------------------------------
# The slice
# ---------------------------------------------------------------------------


def test_the_full_vertical_slice_reaches_an_auditable_search_hit(
    db_session: Session, artifacts_root: Path, node: _Namespace
) -> None:
    """Europe PMC -> JATS -> chunking -> OpenSearch -> BM25, in one test."""
    # --- acquisition -----------------------------------------------------
    ingested = _ingest(db_session, artifacts_root)
    artifact = ingested.artifact

    assert ingested.requested_urls == (_SOURCE_URI,)
    assert artifact.content_sha256 == _ARTICLE_SHA256
    assert artifact.byte_size == len(JATS_FULL_ARTICLE)
    assert artifact.source_system == "europe_pmc"
    assert artifact.source_external_id == _PMCID
    assert artifact.storage_uri.startswith("file:///")
    assert ingested.stored_object.exists()

    # --- canonical JATS import ------------------------------------------
    assert ingested.document.canonical_key == f"doi:{_DOI}"
    assert ingested.version.title == _TITLE
    assert ingested.version.language == "en"
    assert ingested.passage_count > 0

    # --- projection into OpenSearch --------------------------------------
    projection = node.project(db_session, chunker_revision=ingested.chunker_revision)

    assert projection.created is True
    assert projection.chunker_revision == ingested.chunker_revision
    assert projection.document_count == ingested.passage_count
    assert node.client.count(projection.index_name) == projection.document_count
    assert node.client.alias_targets(node.alias) == (projection.index_name,)

    # --- BM25 search -----------------------------------------------------
    response = node.service().search(_QUERY, limit=5)

    assert response.total > 0
    assert response.query_revision == "bm25-v1"
    assert response.chunker_revision == ingested.chunker_revision
    assert response.index_schema_revision == projection.projection_schema_revision
    assert response.projection_sha256 == projection.projection_sha256
    _assert_provenance(db_session, ingested, response.hits)
    # The section structure survived the whole slice, not just the flat text.
    assert _EXPECTED_SECTION in {hit.section_title for hit in response.hits}


def _assert_provenance(session: Session, ingested: _Ingested, hits: tuple[SearchHit, ...]) -> None:
    """Every returned hit resolves back to the canonical rows it came from."""
    canonical = {
        record.passage_key: record
        for record in list_passages(session, ingested.version.id, ingested.chunker_revision)
    }
    paragraph_text = {
        record.paragraph_key: record.text
        for record in list_paragraphs(session, ingested.version.id)
    }

    assert hits
    assert {hit.passage_key for hit in hits} <= set(canonical)
    for hit in hits:
        record = canonical[hit.passage_key]
        assert hit.text == record.text
        assert hit.content_sha256 == record.content_sha256
        assert hit.passage_ordinal == record.ordinal
        assert hit.token_count == record.token_count
        assert hit.document_version_key == ingested.version.version_key
        assert hit.document_canonical_key == ingested.document.canonical_key
        assert hit.primary_source_anchor == record.source_anchor
        assert hit.source_system == "europe_pmc"
        assert hit.source_external_id == _PMCID
        assert hit.doi == _DOI
        assert hit.pmcid == _PMCID
        assert hit.title == _TITLE
        assert hit.source_spans
        for span in hit.source_spans:
            text = paragraph_text[span.paragraph_key]
            assert 0 <= span.start_char < span.end_char <= len(text)


def test_the_vertical_slice_rebuilds_identically_from_postgresql(
    db_session: Session, artifacts_root: Path, node: _Namespace
) -> None:
    """Delete the projection completely and rebuild it: the same SHA, the same
    documents and the same ranking, from canonical state alone."""
    ingested = _ingest(db_session, artifacts_root)
    first = node.project(db_session, chunker_revision=ingested.chunker_revision)
    service = node.service()
    before = service.search(_QUERY, limit=10)

    node.drop_projection()
    assert node.client.alias_targets(node.alias) == ()

    rebuilt = node.project(db_session, chunker_revision=ingested.chunker_revision)
    after = service.search(_QUERY, limit=10)

    assert rebuilt.index_name == first.index_name
    assert rebuilt.projection_sha256 == first.projection_sha256
    assert rebuilt.document_count == first.document_count
    assert [hit.passage_key for hit in after.hits] == [hit.passage_key for hit in before.hits]
    assert [hit.score for hit in after.hits] == [hit.score for hit in before.hits]
