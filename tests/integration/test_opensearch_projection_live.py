"""The passage projection and BM25 search against the live OpenSearch 3.8.0 node.

These tests use the real container, so they prove what a mocked transport
cannot: that the exact settings and mapping are accepted, that a
``refresh=wait_for`` bulk index is immediately visible, that BM25 with the
declared similarity actually ranks, and — above all — that the alias cutover
and the rebuild-from-PostgreSQL path behave as documented against a real index.

Isolation: every test gets its own alias, and the physical index name embeds
that alias, so parallel runs never collide on a shared node. Each test deletes
every index it observed in teardown, leaving the node as it found it.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any, Final

import pytest
from sqlalchemy.orm import Session

from dynamisrag.chunking import ChunkerConfig, PassageMaterializer, StructureAwareChunker
from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.canonical import (
    insert_document,
    insert_document_version,
    insert_paragraph,
    insert_section,
    insert_source_artifact,
    list_paragraphs,
    list_passages,
    list_sections,
    paragraph_from_record,
    section_from_record,
)
from dynamisrag.domain.contracts import (
    Document,
    DocumentVersion,
    Paragraph,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.values import DocumentType, ParagraphRegion
from dynamisrag.search import (
    BM25_QUERY_REVISION,
    PASSAGE_INDEX_SCHEMA_REVISION,
    Bm25SearchService,
    OpenSearchClient,
    OpenSearchError,
    PassageProjector,
)
from dynamisrag.search.errors import OpenSearchTransportError
from dynamisrag.search.projection import DEFAULT_BULK_BATCH_SIZE, ProjectionResult
from tests._support import build_settings

pytestmark = pytest.mark.integration

_NOW: Final[datetime] = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
_EXPECTED_OPENSEARCH_MAJOR: Final[str] = "3."
# The document title is indexed on *every* passage and is the most boosted
# field, so it deliberately contains none of the body terms the ranking
# assertions use: a title match can therefore never be mistaken for a body
# match, and a term's hit count is exactly the number of passages whose text
# contains it.
_TITLE: Final[str] = "A two-arm controlled trial of dietary supplements"
_PMCID: Final[str] = "PMC90000001"
_DOI: Final[str] = "10.1371/journal.pone.03089999"

_CONFIG: Final[ChunkerConfig] = ChunkerConfig()
_REVISION: Final[str] = StructureAwareChunker(_CONFIG).chunker_revision
"""The canonical chunker revision this slice projects."""

_ALTERNATE_CONFIG: Final[ChunkerConfig] = ChunkerConfig(
    target_tokens=120, max_tokens=160, min_tokens=40
)
_ALTERNATE_REVISION: Final[str] = StructureAwareChunker(_ALTERNATE_CONFIG).chunker_revision
"""A second, coexisting immutable passage set for the same document version."""

# Four controlled sections whose passage text repeats a body term a different
# number of times, so BM25 ranking is inspectable without depending on any
# undocumented internal ordering.
_CORPUS: Final[tuple[tuple[str, str, str], ...]] = (
    ("Introduction", "1", "Jumping jumping jumping improved after the intervention."),
    ("Methods", "2", "Jumping once per week was the only recorded activity."),
    ("Results", "3", "The control diet produced no measurable performance difference."),
    ("Discussion", "4", "The scoring rubric was applied twice by two blinded observers."),
)

# Two sections with an identical title and two identical paragraphs: identical
# scored input, so identical BM25. Any order between them can only come from
# the declared tie-break.
_TIED_TEXT: Final[str] = "Colonic lesions were scored by two blinded observers."
_TIED: Final[tuple[tuple[str, str], ...]] = (("1", _TIED_TEXT), ("2", _TIED_TEXT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session(live_settings: Settings) -> Iterator[Session]:
    """A session whose transaction is always rolled back (never DELETE)."""
    engine = create_database_engine(live_settings)
    session = Session(bind=engine)
    transaction = session.begin()
    try:
        yield session
    finally:
        transaction.rollback()
        session.close()
        engine.dispose()


class _Namespace:
    """One isolated alias on the live node, cleaned up in teardown."""

    def __init__(self, alias: str, settings: Settings) -> None:
        self.alias = alias
        self.client = OpenSearchClient(settings)
        self.indexes: set[str] = set()

    # -- operations ------------------------------------------------------
    def project(self, session: Session, *, chunker_revision: str) -> ProjectionResult:
        result = PassageProjector(
            session, self.client, alias=self.alias, batch_size=DEFAULT_BULK_BATCH_SIZE
        ).project(chunker_revision=chunker_revision)
        self.indexes.add(result.index_name)
        self.indexes.update(result.removed_index_names)
        return result

    def service(self) -> Bm25SearchService:
        return Bm25SearchService(self.client, alias=self.alias)

    def targets(self) -> tuple[str, ...]:
        return self.client.alias_targets(self.alias)

    def exists(self, index: str) -> bool:
        return self.client.index_exists(index)

    def count(self, index: str) -> int:
        return self.client.count(index)

    def delete_projection(self) -> None:
        """Remove the projection entirely, as an accidental data loss would."""
        for index in (*self.targets(), *sorted(self.indexes)):
            self.client.delete_index(index)
            self.indexes.discard(index)

    def cleanup(self) -> None:
        for index in sorted(self.indexes | set(self.targets())):
            with contextlib.suppress(OpenSearchError):
                self.client.delete_index(index)
        self.client.close()


@pytest.fixture
def node(live_settings: Settings) -> Iterator[_Namespace]:
    """A unique alias/index namespace per test, removed afterwards."""
    namespace = _Namespace(f"dynamisrag-it-{uuid.uuid4().hex[:12]}", live_settings)
    try:
        yield namespace
    finally:
        namespace.cleanup()


# ---------------------------------------------------------------------------
# Seeding canonical PostgreSQL state
# ---------------------------------------------------------------------------


def _sections(version: DocumentVersion, corpus: Sequence[tuple[str, str, str]]) -> list[Section]:
    return [
        Section(
            document_version_id=version.id,
            version_key=version.version_key,
            ordinal=index,
            depth=0,
            title=title,
            semantic_type="sec",
            source_anchor=f"jats:#sec-{index}",
            structural_path=path,
        )
        for index, (title, path, _) in enumerate(corpus)
    ]


def _paragraphs(
    version: DocumentVersion,
    corpus: Sequence[tuple[str, str, str]],
    sections: Sequence[Section],
) -> list[Paragraph]:
    return [
        Paragraph(
            document_version_id=version.id,
            version_key=version.version_key,
            section_id=sections[index].id,
            ordinal=index,
            region=ParagraphRegion.BODY,
            source_anchor=f"jats:/body[1]/sec[{index + 1}]/p[1]",
            text=text,
            content_sha256="c" * 64,
        )
        for index, (_, _, text) in enumerate(corpus)
    ]


def _persist(
    session: Session,
    corpus: Sequence[tuple[str, str, str]],
    *,
    pmcid: str = _PMCID,
    chunker: StructureAwareChunker | None = None,
) -> DocumentVersion:
    """Persist artifact, document, version, sections, paragraphs and passages.

    Only canonical rows are written: the projection is derived from them, never
    the other way round.
    """
    artifact = SourceArtifact(
        source_system="europe_pmc",
        source_external_id=pmcid,
        source_uri=f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
        media_type="application/xml",
        content_sha256="a" * 64,
        byte_size=2048,
        retrieved_at=_NOW,
        storage_uri=f"file:///artifacts/{pmcid}.xml",
    )
    insert_source_artifact(session, artifact)
    document = Document(
        document_type=DocumentType.JOURNAL_ARTICLE, doi=_DOI, pmcid=_PMCID, title=_TITLE
    )
    insert_document(session, document)
    version = DocumentVersion(
        document_id=document.id,
        document_canonical_key=document.canonical_key,
        source_artifact_id=artifact.id,
        source_artifact_key=artifact.artifact_key,
        parser_revision="jats-1.0",
        normalizer_revision="norm-1.0",
        content_fingerprint="b" * 64,
        title=_TITLE,
        language="en",
        created_at=_NOW,
    )
    insert_document_version(session, version)

    sections = _sections(version, corpus)
    for section in sections:
        insert_section(session, section)
    paragraphs = _paragraphs(version, corpus, sections)
    for paragraph in paragraphs:
        insert_paragraph(session, paragraph)

    planner = chunker if chunker is not None else StructureAwareChunker(_CONFIG)
    manifest = planner.plan(version, sections, paragraphs)
    result = PassageMaterializer(session).materialize(version, manifest)
    assert result.created is True
    return version


def _corpus(session: Session, **kwargs: Any) -> DocumentVersion:
    return _persist(session, _CORPUS, **kwargs)


def _coexist_second_revision(session: Session, version: DocumentVersion) -> None:
    """Materialize a second, immutable passage set for the same version.

    Sections and paragraphs are *source structure* and are shared: only the
    passages differ, because a different chunker configuration packs the same
    paragraphs differently. PostgreSQL then holds two passage sets for one
    version, and the projection must select one explicitly.
    """
    sections = [
        section_from_record(record, version_key=version.version_key)
        for record in list_sections(session, version.id)
    ]
    paragraphs = [
        paragraph_from_record(record, version_key=version.version_key)
        for record in list_paragraphs(session, version.id)
    ]
    planner = StructureAwareChunker(_ALTERNATE_CONFIG)
    manifest = planner.plan(version, sections, paragraphs)
    PassageMaterializer(session).materialize(version, manifest)


# ---------------------------------------------------------------------------
# The node is the pinned OpenSearch 3.x
# ---------------------------------------------------------------------------


def test_the_node_is_the_pinned_opensearch_3(node: _Namespace) -> None:
    version = node.client.node_root()["version"]

    assert isinstance(version, dict)
    assert str(version["number"]).startswith(_EXPECTED_OPENSEARCH_MAJOR)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_projection_builds_a_verified_index_and_moves_the_alias(
    node: _Namespace, db_session: Session
) -> None:
    """Canonical PostgreSQL passages -> deterministic manifest -> physical
    index -> bulk -> stable alias, with the exact document count."""
    _corpus(db_session)

    result = node.project(db_session, chunker_revision=_REVISION)

    assert result.created is True
    assert result.projection_schema_revision == PASSAGE_INDEX_SCHEMA_REVISION
    assert result.chunker_revision == _REVISION
    assert result.document_count == 4
    assert len(result.projection_sha256) == 64
    assert result.index_name.endswith(f"-{result.projection_sha256[:12]}")
    assert result.index_name.startswith(node.alias)
    assert node.targets() == (result.index_name,)
    assert node.exists(result.index_name)
    assert node.count(result.index_name) == 4

    meta = node.client.index_meta(result.index_name)
    assert meta["schema_revision"] == PASSAGE_INDEX_SCHEMA_REVISION
    assert meta["projection_sha256"] == result.projection_sha256
    assert meta["chunker_revision"] == _REVISION
    assert meta["bm25_similarity_revision"] == "dynamis_bm25_v1"


def test_the_strict_mapping_rejects_an_unmodelled_field(
    node: _Namespace, db_session: Session
) -> None:
    """``dynamic: strict`` is a real constraint on a real index, not a
    declaration: an unmodelled field is refused."""
    _corpus(db_session)
    result = node.project(db_session, chunker_revision=_REVISION)
    documents = node.service().search("jumping", limit=1).hits
    assert documents

    with pytest.raises(OpenSearchError):
        node.client.bulk_index(
            result.index_name,
            [(documents[0].passage_key, {"passage_key": documents[0].passage_key, "nope": 1})],
            batch_size=10,
        )


def test_only_the_requested_chunker_revision_is_projected(
    node: _Namespace, db_session: Session
) -> None:
    version = _corpus(db_session)
    _coexist_second_revision(db_session, version)

    result = node.project(db_session, chunker_revision=_REVISION)
    hits = node.service().search("jumping", limit=50).hits

    assert result.chunker_revision == _REVISION
    assert result.document_count == 4
    assert {hit.chunker_revision for hit in hits} == {_REVISION}
    assert _ALTERNATE_REVISION != _REVISION


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_a_known_term_ranks_the_relevant_passages_first(
    node: _Namespace, db_session: Session
) -> None:
    _corpus(db_session)
    node.project(db_session, chunker_revision=_REVISION)

    response = node.service().search("jumping")

    assert response.query_revision == BM25_QUERY_REVISION
    assert response.index_schema_revision == PASSAGE_INDEX_SCHEMA_REVISION
    assert response.projection_sha256
    assert response.chunker_revision == _REVISION
    assert response.took_ms >= 0
    assert response.total == 2

    top = response.hits[0]
    assert top.rank == 1
    assert top.score > 0
    assert top.section_title == "Introduction"
    assert top.text.lower().count("jumping") == 3
    assert top.title == _TITLE
    assert top.language == "en"
    assert top.chunker_revision == _REVISION
    assert top.section_path == "1"
    assert top.document_canonical_key == f"doi:{_DOI}"
    assert top.document_version_key
    assert top.source_system == "europe_pmc"
    assert top.source_external_id == _PMCID
    assert top.doi == _DOI
    # A PMID was never recorded for this document, and is never inferred from
    # the artifact, the canonical key or the title.
    assert top.pmid is None
    assert top.pmcid == _PMCID
    assert top.primary_source_anchor
    assert len(top.source_spans) == 1
    assert top.source_spans[0].start_char == 0
    assert top.source_spans[0].end_char == len(top.text)


def test_a_repeated_query_term_outranks_a_single_occurrence(
    node: _Namespace, db_session: Session
) -> None:
    """BM25 term frequency is what separates the two passages containing
    ``jumping``: three occurrences outrank one."""
    _corpus(db_session)
    node.project(db_session, chunker_revision=_REVISION)

    response = node.service().search("jumping")

    assert [hit.section_title for hit in response.hits] == ["Introduction", "Methods"]
    assert response.hits[0].score > response.hits[1].score


def test_the_ranking_is_stable_across_repeated_queries(
    node: _Namespace, db_session: Session
) -> None:
    _corpus(db_session)
    node.project(db_session, chunker_revision=_REVISION)
    service = node.service()

    first = service.search("jumping", limit=4)
    second = service.search("jumping", limit=4)

    assert [hit.passage_key for hit in first.hits] == [hit.passage_key for hit in second.hits]
    assert [hit.score for hit in first.hits] == [hit.score for hit in second.hits]


def test_a_term_present_nowhere_returns_no_hits(node: _Namespace, db_session: Session) -> None:
    _corpus(db_session)
    node.project(db_session, chunker_revision=_REVISION)

    response = node.service().search("helicobacter pylori")

    assert response.total == 0
    assert response.hits == ()


def test_the_limit_bounds_the_result_set(node: _Namespace, db_session: Session) -> None:
    _corpus(db_session)
    node.project(db_session, chunker_revision=_REVISION)

    assert len(node.service().search("jumping", limit=1).hits) == 1
    assert len(node.service().search("jumping", limit=2).hits) == 2


def test_every_hit_is_auditable_back_to_its_exact_source_span(
    node: _Namespace, db_session: Session
) -> None:
    """A returned hit names the exact paragraph characters it came from, so a
    reader can verify it against canonical PostgreSQL without the projection."""
    version = _corpus(db_session)
    result = node.project(db_session, chunker_revision=_REVISION)
    canonical = {
        record.passage_key: record for record in list_passages(db_session, version.id, _REVISION)
    }
    titles = {title for title, _, _ in _CORPUS}

    hits = node.service().search("jumping performance observers intervention", limit=50).hits
    by_key = {hit.passage_key: hit for hit in hits}

    assert set(by_key) == set(canonical)
    assert len(hits) == result.document_count
    for key, hit in by_key.items():
        record = canonical[key]
        assert hit.text == record.text
        assert hit.content_sha256 == record.content_sha256
        assert hit.token_count == record.token_count
        assert hit.passage_ordinal == record.ordinal
        assert hit.primary_source_anchor == record.source_anchor
        assert hit.section_title in titles
        assert hit.section_path in {"1", "2", "3", "4"}
        assert hit.section_key
        assert hit.document_version_key == version.version_key
        span = hit.source_spans[0]
        assert (span.start_char, span.end_char) == (0, len(record.text))
        assert span.paragraph_source_anchor == record.source_anchor


# ---------------------------------------------------------------------------
# Stable tie-break
# ---------------------------------------------------------------------------


def test_equal_scoring_documents_are_ordered_by_passage_key(
    node: _Namespace, db_session: Session
) -> None:
    """Two passages with identical scored input tie exactly; the declared
    tie-break is ``passage_key`` ascending, so the order is a property of the
    query and not of Lucene's internal document order."""
    _persist(db_session, tuple(("Shared", path, text) for path, text in _TIED), pmcid="PMC90000002")
    node.project(db_session, chunker_revision=_REVISION)

    response = node.service().search("colonic lesions")

    assert response.total == 2
    scores = [hit.score for hit in response.hits]
    assert scores[0] == scores[1]
    keys = [hit.passage_key for hit in response.hits]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_projecting_the_same_canonical_state_twice_is_a_no_op(
    node: _Namespace, db_session: Session
) -> None:
    _corpus(db_session)

    first = node.project(db_session, chunker_revision=_REVISION)
    before = node.service().search("jumping", limit=4)
    second = node.project(db_session, chunker_revision=_REVISION)
    after = node.service().search("jumping", limit=4)

    assert first.created is True
    assert second.created is False
    assert second.projection_sha256 == first.projection_sha256
    assert second.index_name == first.index_name
    assert node.targets() == (first.index_name,)
    assert node.count(first.index_name) == first.document_count
    assert [hit.passage_key for hit in after.hits] == [hit.passage_key for hit in before.hits]


# ---------------------------------------------------------------------------
# Alias cutover
# ---------------------------------------------------------------------------


def test_selecting_another_immutable_revision_cuts_the_alias_over(
    node: _Namespace, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alias moves to a newly built, fully verified index, and the previous
    target is removed only after the cutover."""
    version = _corpus(db_session)
    first = node.project(db_session, chunker_revision=_REVISION)
    first_hits = [hit.passage_key for hit in node.service().search("jumping").hits]
    _coexist_second_revision(db_session, version)

    # Record what the alias pointed at immediately before each index removal.
    observed: list[tuple[str, tuple[str, ...]]] = []
    original_delete = OpenSearchClient.delete_index

    def recording_delete(self: OpenSearchClient, index: str) -> None:
        observed.append((index, node.targets()))
        original_delete(self, index)

    monkeypatch.setattr(OpenSearchClient, "delete_index", recording_delete)
    second = node.project(db_session, chunker_revision=_ALTERNATE_REVISION)

    assert first_hits
    assert second.created is True
    assert second.index_name != first.index_name
    assert second.projection_sha256 != first.projection_sha256
    assert node.targets() == (second.index_name,)
    assert not node.exists(first.index_name)
    # The old target was still present when the new one became active.
    removal = next(entry for entry in observed if entry[0] == first.index_name)
    assert removal[1] == (second.index_name,)
    # And search now sees the new projection.
    new_hits = node.service().search("jumping", limit=50).hits
    assert {hit.chunker_revision for hit in new_hits} == {_ALTERNATE_REVISION}


# ---------------------------------------------------------------------------
# Rebuildability: the projection is disposable
# ---------------------------------------------------------------------------


def test_deleting_the_projection_and_rebuilding_restores_it_exactly(
    node: _Namespace, db_session: Session
) -> None:
    """The proof that OpenSearch is a cache: delete it entirely, rebuild from
    PostgreSQL alone, and get exactly the same projection back."""
    _corpus(db_session)
    first = node.project(db_session, chunker_revision=_REVISION)
    before = node.service().search("jumping", limit=4)
    before_ids = [hit.passage_key for hit in before.hits]

    node.delete_projection()
    assert node.targets() == ()
    assert not node.exists(first.index_name)
    with pytest.raises(OpenSearchError):
        node.service().search("jumping")

    rebuilt = node.project(db_session, chunker_revision=_REVISION)
    after = node.service().search("jumping", limit=4)

    assert rebuilt.created is True
    assert rebuilt.index_name == first.index_name
    assert rebuilt.projection_sha256 == first.projection_sha256
    assert rebuilt.document_count == first.document_count
    assert node.count(rebuilt.index_name) == first.document_count
    assert [hit.passage_key for hit in after.hits] == before_ids
    assert [hit.score for hit in after.hits] == [hit.score for hit in before.hits]


# ---------------------------------------------------------------------------
# Backend failure
# ---------------------------------------------------------------------------


def test_a_search_with_no_projection_fails_cleanly(node: _Namespace, db_session: Session) -> None:
    """A search with nothing behind the alias is a backend error — never a
    silent empty result, and never a leaked body or credential."""
    _corpus(db_session)
    node.project(db_session, chunker_revision=_REVISION)
    for index in node.targets():
        node.client.delete_index(index)

    with pytest.raises(OpenSearchError) as caught:
        node.service().search("jumping")

    message = str(caught.value)
    assert "Traceback" not in message
    assert "Basic " not in message
    assert "password" not in message.lower()


def test_an_unreachable_node_is_a_transport_error() -> None:
    """A dead node is a typed transport error, not a hang and not a traceback."""
    client = OpenSearchClient(build_settings(opensearch_url="https://unreachable.invalid:9200"))
    try:
        with pytest.raises(OpenSearchTransportError, match="TransportError"):
            Bm25SearchService(client, alias="dynamisrag-it-unreachable").search("jumping")
    finally:
        client.close()
