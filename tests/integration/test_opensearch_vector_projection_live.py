"""The vector-capable projection against the live OpenSearch 3.8.0 node.

Everything else about the vector contract can be asserted with no node, and most
of it is. This file proves the parts that only a real index can answer:

* the ``passage-index-v2`` mapping, settings and ``_meta`` are **accepted** by
  OpenSearch 3.8.0 — a mapping the node refuses is not a mapping, however well
  this project reasons about it;
* documents indexed through it are immediately searchable both ways: a
  **raw test-only k-NN query** ranks the expected nearest neighbour first, and
  the existing ``bm25-v1`` query ranks the expected passages first through the
  very same alias;
* a ``SearchResponse`` from a v2 index reports ``passage-index-v2``;
* the embedding is stored on the document and is nevertheless **absent from every
  BM25 hit**, proved by fetching the raw document and comparing;
* a lexical ``passage-index-v1`` index is replaced by a verified v2 index through
  one atomic alias cutover, with the previously served target still present until
  the new one has been verified;
* the whole thing is idempotent, rebuildable and disposable — publishing twice is
  a no-op, changing one vector is a new index and a safe cutover, and deleting
  everything rebuilds to the same digest, name, document ids and neighbour ranking
  from canonical PostgreSQL plus the same explicit vector set;
* production dense retrieval and RRF run over deterministic 512-dimensional test
  vectors, including an alias cutover after the physical target is captured;
* the mechanics hold at PMC scale: the real PMC2731074 article, 19 passages, 19
  vectors, 19 documents.

Isolation: every test gets its own alias and the physical index name embeds that
alias, so parallel runs never collide on a shared node. Teardown deletes every
index the test observed.

**SYNTHETIC TEST VECTORS — NOT EMBEDDINGS.** Every vector in this file is a value
chosen by the test, not produced by any model. The embedding identity on
:class:`_CONFIG` is therefore a *label* for the synthetic generator, deliberately
named so that no reader can mistake it for weights. These tests prove index
mechanics — mapping acceptance, ANN ranking, BM25 coexistence, cutover, idempotency
— and make **no claim whatsoever about semantic quality**, because the vectors
carry no semantics. The Qwen 512 query profile exercised by the hybrid tests is
provisional and does not establish formal RES-138 Stage-B qualification.

The raw k-NN query below remains a test-only projection check. The RES-139 tests
also exercise the production dense query builder and hybrid service against the
same live OpenSearch node, using synthetic vectors and a fake query embedder.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx2
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
from dynamisrag.ingestion import EuropePmcAcquisition, EuropePmcClient
from dynamisrag.jats import JatsCanonicalImporter
from dynamisrag.search import (
    BM25_QUERY_REVISION,
    BM25_SIMILARITY_REVISION,
    DENSE_DIMENSION,
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    Bm25SearchService,
    HybridRetrievalService,
    OpenSearchClient,
    OpenSearchError,
    PassageProjector,
    VectorPassageProjector,
)
from dynamisrag.search.schema import VECTOR_FIELD
from dynamisrag.search.vector import (
    VECTOR_SPACE_COSINESIMIL,
    EmbeddingModelIdentity,
    VectorIndexConfig,
)
from dynamisrag.search.vector_projection import PassageVector, VectorProjectionResult
from dynamisrag.storage import FileSystemObjectStore
from tests._support import JATS_PMC2731074, PMC2731074_ARTICLE_SHA256

pytestmark = pytest.mark.integration

_NOW: Final[datetime] = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
_EXPECTED_OPENSEARCH_MAJOR: Final[str] = "3."

# SYNTHETIC TEST VECTORS — NOT EMBEDDINGS.
_SYNTHETIC_LABEL: Final[str] = "SYNTHETIC TEST VECTORS - NOT EMBEDDINGS"

_SYNTHETIC_MODEL: Final[EmbeddingModelIdentity] = EmbeddingModelIdentity(
    # Named for what it is, so no reader can mistake the provenance recorded in
    # every v2 index for real model weights.
    model_id="synthetic/res136-test-vectors",
    model_revision="unit-basis-v1",
    embedding_config_sha256=hashlib.sha256(_SYNTHETIC_LABEL.encode("utf-8")).hexdigest(),
)

_DIMENSION: Final[int] = 3
_CONFIG: Final[VectorIndexConfig] = VectorIndexConfig(
    dimension=_DIMENSION,
    space=VECTOR_SPACE_COSINESIMIL,
    embedding_model=_SYNTHETIC_MODEL,
)

_HYBRID_MODEL: Final[EmbeddingModelIdentity] = EmbeddingModelIdentity(
    model_id=DENSE_MODEL_ID,
    model_revision=DENSE_MODEL_REVISION,
    embedding_config_sha256=hashlib.sha256(b"synthetic document generation semantics").hexdigest(),
)
_HYBRID_CONFIG: Final[VectorIndexConfig] = VectorIndexConfig(
    dimension=DENSE_DIMENSION,
    space=VECTOR_SPACE_COSINESIMIL,
    embedding_model=_HYBRID_MODEL,
)

# Three orthogonal unit vectors and one diagonal, so a cosine query equal to one
# of them has exactly one nearest neighbour at similarity 1.0 and the rest are
# unambiguous behind it.
_UNIT_A: Final[tuple[float, ...]] = (1.0, 0.0, 0.0)
_UNIT_B: Final[tuple[float, ...]] = (0.0, 1.0, 0.0)
_UNIT_C: Final[tuple[float, ...]] = (0.0, 0.0, 1.0)
_DIAGONAL: Final[tuple[float, ...]] = (0.5773502691896258, 0.5773502691896258, 0.5773502691896258)
_QUERY: Final[tuple[float, ...]] = _UNIT_A

_CHUNKER_CONFIG: Final[ChunkerConfig] = ChunkerConfig()
_REVISION: Final[str] = StructureAwareChunker(_CHUNKER_CONFIG).chunker_revision

_TITLE: Final[str] = "A two-arm controlled trial of dietary supplements"
_PMCID: Final[str] = "PMC90000001"
_DOI: Final[str] = "10.1371/journal.pone.03089999"

# The document title is indexed on every passage and is the most boosted field,
# so it deliberately contains none of the body terms the ranking assertions use.
_CORPUS: Final[tuple[tuple[str, str, str], ...]] = (
    ("Introduction", "1", "Jumping jumping jumping improved after the intervention."),
    ("Methods", "2", "Jumping once per week was the only recorded activity."),
    ("Results", "3", "The control diet produced no measurable performance difference."),
    ("Discussion", "4", "The scoring rubric was applied twice by two blinded observers."),
)

_PMC_PMCID: Final[str] = "PMC2731074"
_PMC_PASSAGE_COUNT: Final[int] = 19


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


@dataclass(frozen=True)
class _Neighbour:
    """One hit of a raw test-only k-NN query."""

    passage_key: str
    score: float

    @property
    def embedding(self) -> tuple[float, ...] | None:
        return None


def _knn_body(vector: Sequence[float], *, k: int) -> Mapping[str, Any]:
    """A raw k-NN request body, written by this test and nowhere in production.

    Under Lucene's ``cosinesimil`` OpenSearch scores ``(1 + cosine_similarity) / 2``,
    so a vector's own unit direction scores exactly ``1.0`` and an orthogonal one
    ``0.5``. Those are the numbers the assertions below expect, which is why the
    test's vectors are orthogonal unit vectors and a diagonal rather than arbitrary
    floats.
    """
    return {
        "size": k,
        "_source": ["passage_key"],
        "query": {"knn": {VECTOR_FIELD: {"vector": list(vector), "k": k}}},
    }


def _raw_document_body(passage_key: str) -> Mapping[str, Any]:
    """Fetch a stored document whole, embedding included.

    Used only to prove the negative: that the embedding *is* in ``_source`` and is
    still absent from every BM25 :class:`~dynamisrag.search.bm25.SearchHit`.
    """
    return {
        "size": 1,
        "query": {"term": {"passage_key": passage_key}},
    }


class _Namespace:
    """One isolated alias on the live node, cleaned up in teardown."""

    def __init__(self, alias: str, settings: Settings) -> None:
        self.alias = alias
        self.client = OpenSearchClient(settings)
        self.indexes: set[str] = set()

    # -- projection --------------------------------------------------------
    def project_lexical(self, session: Session, *, chunker_revision: str) -> Any:
        result = PassageProjector(session, self.client, alias=self.alias).project(
            chunker_revision=chunker_revision
        )
        self.indexes.add(result.index_name)
        self.indexes.update(result.removed_index_names)
        return result

    def project_vector(
        self,
        session: Session,
        *,
        chunker_revision: str,
        vectors: Sequence[PassageVector],
        vector_config: VectorIndexConfig = _CONFIG,
    ) -> VectorProjectionResult:
        result = VectorPassageProjector(
            session, self.client, alias=self.alias, vector_config=vector_config
        ).project(chunker_revision=chunker_revision, vectors=vectors)
        self.indexes.add(result.index_name)
        self.indexes.update(result.removed_index_names)
        return result

    # -- reads -------------------------------------------------------------
    def service(self) -> Bm25SearchService:
        return Bm25SearchService(self.client, alias=self.alias)

    def targets(self) -> tuple[str, ...]:
        return self.client.alias_targets(self.alias)

    def exists(self, index: str) -> bool:
        return self.client.index_exists(index)

    def count(self, index: str) -> int:
        return self.client.count(index)

    def neighbours(self, target: str, vector: Sequence[float], *, k: int) -> tuple[_Neighbour, ...]:
        """Run the raw test-only k-NN query and read back the ranking.

        Written against the alias or a physical index alike, so the same
        assertion covers both the live alias and a specific index.
        """
        payload = self.client.search(target, _knn_body(vector, k=k))
        hits = payload.get("hits")
        assert isinstance(hits, Mapping)
        raw_hits = hits.get("hits")
        assert isinstance(raw_hits, list)
        neighbours: list[_Neighbour] = []
        for raw in raw_hits:
            assert isinstance(raw, Mapping)
            document_id = raw.get("_id")
            score = raw.get("_score")
            assert isinstance(document_id, str)
            assert isinstance(score, (int, float)) and not isinstance(score, bool)
            neighbours.append(_Neighbour(passage_key=document_id, score=float(score)))
        return tuple(neighbours)

    def raw_document(self, index: str, passage_key: str) -> Mapping[str, Any]:
        payload = self.client.search(index, _raw_document_body(passage_key))
        hits = payload.get("hits")
        assert isinstance(hits, Mapping)
        raw_hits = hits.get("hits")
        assert isinstance(raw_hits, list)
        assert raw_hits, f"document {passage_key} is not in {index}"
        first = raw_hits[0]
        assert isinstance(first, Mapping)
        source = first.get("_source")
        assert isinstance(source, Mapping)
        return source

    def delete_everything(self) -> None:
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
    namespace = _Namespace(f"dynamisrag-vec-{uuid.uuid4().hex[:12]}", live_settings)
    try:
        yield namespace
    finally:
        namespace.cleanup()


# ---------------------------------------------------------------------------
# Seeding canonical PostgreSQL state
# ---------------------------------------------------------------------------


def _seed(session: Session, *, pmcid: str = _PMCID) -> DocumentVersion:
    """Persist artifact, document, version, sections, paragraphs and passages."""
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
        document_type=DocumentType.JOURNAL_ARTICLE, doi=_DOI, pmcid=pmcid, title=_TITLE
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

    sections = [
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
        for index, (title, path, _) in enumerate(_CORPUS)
    ]
    for section in sections:
        insert_section(session, section)
    paragraphs = [
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
        for index, (_, _, text) in enumerate(_CORPUS)
    ]
    for paragraph in paragraphs:
        insert_paragraph(session, paragraph)

    planner = StructureAwareChunker(_CHUNKER_CONFIG)
    manifest = planner.plan(version, sections, paragraphs)
    result = PassageMaterializer(session).materialize(version, manifest)
    assert result.created is True
    return version


def _passage_keys(session: Session, version: DocumentVersion) -> list[str]:
    from dynamisrag.db.canonical import list_passages

    return [
        record.passage_key
        for record in sorted(list_passages(session, version.id, _REVISION), key=lambda r: r.ordinal)
    ]


def _synthetic_vectors(
    keys: Sequence[str], *, values: Sequence[Sequence[float]]
) -> list[PassageVector]:
    """SYNTHETIC TEST VECTORS — NOT EMBEDDINGS.

    One supplied vector per passage key, in the caller's order. The relation is
    positional only because these tests supply them in ``passage_key`` order; the
    builder itself is order-insensitive, which the unit suite proves.
    """
    assert len(keys) == len(values), "one vector per passage"
    return [
        PassageVector(passage_key=key, values=tuple(value))
        for key, value in zip(keys, values, strict=True)
    ]


def _corpus_vectors(keys: Sequence[str]) -> list[PassageVector]:
    return _synthetic_vectors(keys, values=(_UNIT_A, _UNIT_B, _UNIT_C, _DIAGONAL))


# The node is the pinned OpenSearch 3.x
# ---------------------------------------------------------------------------


def test_the_node_is_the_pinned_opensearch_3(node: _Namespace) -> None:
    version = node.client.node_root()["version"]

    assert isinstance(version, Mapping)
    assert str(version["number"]).startswith(_EXPECTED_OPENSEARCH_MAJOR)


# ---------------------------------------------------------------------------
# The v2 mapping is accepted, and BM25 works through the same alias
# ---------------------------------------------------------------------------


def test_a_v2_index_is_accepted_and_both_modalities_answer(
    node: _Namespace, db_session: Session
) -> None:
    """Mapping accepted, documents indexed, k-NN ranks A first, BM25 still ranks.

    One test because the claim is a single one: *the same alias* serves a dense
    and a lexical query, which is the whole reason a v2 index keeps v1's text
    mapping instead of declaring a new one.
    """
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)

    result = node.project_vector(
        db_session, chunker_revision=_REVISION, vectors=_corpus_vectors(keys)
    )

    # --- the mapping and settings the node actually accepted ----------------
    assert result.created is True
    assert result.projection_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert result.vector_config_sha256 == _CONFIG.config_sha256
    assert result.index_name == (f"{node.alias}-passage-index-v2-{result.projection_sha256[:12]}")
    assert node.targets() == (result.index_name,)
    assert node.count(result.index_name) == len(keys) == 4

    meta = node.client.index_meta(result.index_name)
    assert meta["schema_revision"] == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert meta["vector_engine"] == "lucene"
    assert meta["vector_method"] == "hnsw"
    assert meta["vector_data_type"] == "float"
    assert meta["vector_space_type"] == VECTOR_SPACE_COSINESIMIL
    assert meta["vector_dimension"] == _DIMENSION
    assert meta["hnsw_m"] == 16
    assert meta["hnsw_ef_construction"] == 100
    assert meta["embedding_model_id"] == _SYNTHETIC_MODEL.model_id
    assert meta["embedding_model_revision"] == _SYNTHETIC_MODEL.model_revision
    assert meta["embedding_config_sha256"] == _SYNTHETIC_MODEL.embedding_config_sha256
    assert meta["bm25_similarity_revision"] == BM25_SIMILARITY_REVISION

    # --- raw test-only k-NN: A first at 1.0, the diagonal second, B/C tied ---
    # Under Lucene's `cosinesimil` OpenSearch scores `(1 + cosine_similarity) / 2`,
    # so the query's own direction scores exactly 1.0, an orthogonal vector 0.5,
    # and the diagonal (cosine 1/sqrt(3) with A) (1 + 1/sqrt(3)) / 2.
    neighbours = node.neighbours(node.alias, _QUERY, k=4)

    assert neighbours[0].passage_key == keys[0]
    assert neighbours[0].score == pytest.approx(1.0)
    assert neighbours[1].passage_key == keys[3]
    assert neighbours[1].score == pytest.approx((1.0 + 1.0 / 3.0**0.5) / 2.0)
    # B and C are both orthogonal to the query, so they tie exactly at 0.5 and
    # their order between them is not a property of the data.
    assert {hit.passage_key for hit in neighbours[2:]} == {keys[1], keys[2]}
    assert all(hit.score == pytest.approx(0.5) for hit in neighbours[2:])

    # --- the unchanged bm25-v1 query, through the same alias ----------------
    response = node.service().search("jumping")

    assert response.query_revision == BM25_QUERY_REVISION
    assert response.index_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert response.projection_sha256 == result.projection_sha256
    assert response.chunker_revision == _REVISION
    assert response.total == 2
    assert [hit.section_title for hit in response.hits] == ["Introduction", "Methods"]
    assert response.hits[0].score > response.hits[1].score


def test_the_embedding_is_stored_and_still_never_reaches_a_lexical_hit(
    node: _Namespace, db_session: Session
) -> None:
    """The negative, proved against the stored bytes rather than assumed.

    Fetching the raw document shows the embedding really is in ``_source``, and
    the typed BM25 hit for the very same passage does not carry it. That is the
    whole design: indexed for ANN, never selected for a lexical response.
    """
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    node.project_vector(db_session, chunker_revision=_REVISION, vectors=_corpus_vectors(keys))

    raw = node.raw_document(node.alias, keys[0])

    assert raw[VECTOR_FIELD] == pytest.approx(list(_UNIT_A))
    assert raw["projection_schema_revision"] == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION

    hits = node.service().search("jumping", limit=50).hits

    assert hits
    assert keys[0] in {hit.passage_key for hit in hits}
    for hit in hits:
        assert VECTOR_FIELD not in hit.model_dump()
        assert all(
            not isinstance(value, list) or value == [] for value in hit.model_dump().values()
        )


def test_a_query_equal_to_another_vectors_finds_that_passage(
    node: _Namespace, db_session: Session
) -> None:
    """Each of the supplied vectors is individually the nearest neighbour of itself.

    A property of the indexed vectors rather than of one lucky ranking, so it
    would catch a mapping that stored the wrong values.
    """
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    node.project_vector(db_session, chunker_revision=_REVISION, vectors=_corpus_vectors(keys))

    for key, vector in zip(keys, (_UNIT_A, _UNIT_B, _UNIT_C, _DIAGONAL), strict=True):
        nearest = node.neighbours(node.alias, vector, k=1)
        assert [hit.passage_key for hit in nearest] == [key]


# ---------------------------------------------------------------------------
# v1 -> v2 cutover
# ---------------------------------------------------------------------------


def test_a_lexical_index_is_replaced_by_a_verified_v2_index_through_one_cutover(
    node: _Namespace, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alias moves from a live v1 index to a live v2 index, once.

    The old target must remain until the new index is complete and verified: a
    cutover that deletes first would leave a search outage for the length of a
    bulk load, which is exactly the failure the RES-135 ordering forbids.
    """
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)

    lexical = node.project_lexical(db_session, chunker_revision=_REVISION)
    before = node.service().search("jumping")

    assert lexical.projection_schema_revision == PASSAGE_INDEX_SCHEMA_REVISION
    assert before.index_schema_revision == PASSAGE_INDEX_SCHEMA_REVISION
    assert node.targets() == (lexical.index_name,)

    observed: list[tuple[str, tuple[str, ...]]] = []
    original_delete = OpenSearchClient.delete_index

    def recording_delete(self: OpenSearchClient, index: str) -> None:
        observed.append((index, node.targets()))
        original_delete(self, index)

    monkeypatch.setattr(OpenSearchClient, "delete_index", recording_delete)
    vector = node.project_vector(
        db_session, chunker_revision=_REVISION, vectors=_corpus_vectors(keys)
    )

    # A different revision is a different physical index, never a mutation.
    assert vector.index_name != lexical.index_name
    assert vector.projection_sha256 != lexical.projection_sha256
    assert node.targets() == (vector.index_name,)

    # The v1 index was still present, and still the alias target, at the moment
    # it was removed -- i.e. the switch had already happened and been verified.
    removal = next(entry for entry in observed if entry[0] == lexical.index_name)
    assert removal[1] == (vector.index_name,)

    # BM25 answered before, and answers after, off the same alias.
    after = node.service().search("jumping")

    assert after.index_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert after.total == before.total
    assert [hit.passage_key for hit in after.hits] == [hit.passage_key for hit in before.hits]
    assert [hit.score for hit in after.hits] == [hit.score for hit in before.hits]
    assert [hit.text for hit in after.hits] == [hit.text for hit in before.hits]

    # And the dense modality now answers off the same alias.
    assert [hit.passage_key for hit in node.neighbours(node.alias, _QUERY, k=1)] == [keys[0]]
    assert not node.exists(lexical.index_name)


# ---------------------------------------------------------------------------
# Idempotency and rebuildability
# ---------------------------------------------------------------------------


def test_publishing_the_same_passages_and_vectors_twice_is_a_no_op(
    node: _Namespace, db_session: Session
) -> None:
    """Second run: no rebuild, no bulk traffic, no alias churn."""
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    vectors = _corpus_vectors(keys)
    projector = VectorPassageProjector(
        db_session, node.client, alias=node.alias, vector_config=_CONFIG
    )

    first = node.project_vector(db_session, chunker_revision=_REVISION, vectors=vectors)
    before = [hit.passage_key for hit in node.neighbours(node.alias, _QUERY, k=1)]

    second = projector.project(chunker_revision=_REVISION, vectors=vectors)

    assert second.created is False
    assert second.projection_sha256 == first.projection_sha256
    assert second.index_name == first.index_name
    assert second.removed_index_names == ()
    assert node.targets() == (first.index_name,)
    assert node.count(first.index_name) == len(keys)
    assert [hit.passage_key for hit in node.neighbours(node.alias, _QUERY, k=1)] == before


def test_one_changed_vector_is_a_new_index_and_a_safe_cutover(
    node: _Namespace, db_session: Session
) -> None:
    """A single component changes a distance, so it must change the identity.

    Moving one passage's vector from A to C moves its nearest-neighbour identity
    with it, which is the observable consequence of the digest binding exact
    vector values rather than just their shape.
    """
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    first = node.project_vector(
        db_session, chunker_revision=_REVISION, vectors=_corpus_vectors(keys)
    )
    assert [hit.passage_key for hit in node.neighbours(node.alias, _QUERY, k=1)] == [keys[0]]

    moved = _synthetic_vectors(keys, values=(_UNIT_C, _UNIT_B, _UNIT_A, _DIAGONAL))
    second = node.project_vector(db_session, chunker_revision=_REVISION, vectors=moved)

    assert second.created is True
    assert second.index_name != first.index_name
    assert second.projection_sha256 != first.projection_sha256
    assert second.removed_index_names == (first.index_name,)
    assert node.targets() == (second.index_name,)
    assert [hit.passage_key for hit in node.neighbours(node.alias, _QUERY, k=1)] == [keys[2]]
    assert node.service().search("jumping").total == 2


def test_deleting_the_v2_index_and_rebuilding_restores_it_exactly(
    node: _Namespace, db_session: Session
) -> None:
    """The proof that OpenSearch stays a cache for the vectorized index too.

    Delete the index entirely and rebuild from canonical PostgreSQL passages plus
    the same explicit vector set plus the same configuration: the same digest, the
    same index name, the same document ids and the same k-NN ranking.
    """
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    vectors = _corpus_vectors(keys)
    first = node.project_vector(db_session, chunker_revision=_REVISION, vectors=vectors)
    before = node.neighbours(node.alias, _QUERY, k=3)
    before_meta = dict(node.client.index_meta(first.index_name))

    node.delete_everything()

    assert node.targets() == ()
    assert not node.exists(first.index_name)
    with pytest.raises(OpenSearchError):
        node.service().search("jumping")

    rebuilt = node.project_vector(db_session, chunker_revision=_REVISION, vectors=vectors)

    assert rebuilt.created is True
    assert rebuilt.index_name == first.index_name
    assert rebuilt.projection_sha256 == first.projection_sha256
    assert rebuilt.document_count == first.document_count
    assert node.count(rebuilt.index_name) == len(keys)
    assert dict(node.client.index_meta(rebuilt.index_name)) == before_meta
    assert node.neighbours(node.alias, _QUERY, k=3) == before
    assert node.service().search("jumping").index_schema_revision == (
        VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    )


# ---------------------------------------------------------------------------
# PMC-scale mechanics
# ---------------------------------------------------------------------------


class _EuropePmcStub(httpx2.BaseTransport):
    """Serves the captured PMC2731074 full text. No network, fully deterministic."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(str(request.url))
        return httpx2.Response(
            200,
            content=JATS_PMC2731074,
            headers={"content-type": "application/xml"},
            request=request,
        )


@dataclass(frozen=True)
class _Ingested:
    version: DocumentVersion
    chunker_revision: str
    passage_count: int


def _ingest(session: Session, artifacts_root: Path) -> _Ingested:
    """Acquire, canonicalize, chunk and materialize the real article."""
    transport = _EuropePmcStub()
    client = EuropePmcClient(transport=transport)
    try:
        acquired = EuropePmcAcquisition(client, FileSystemObjectStore(artifacts_root)).acquire(
            session, _PMC_PMCID
        )
    finally:
        client.close()

    imported = JatsCanonicalImporter(session).import_artifact(acquired.artifact, JATS_PMC2731074)
    sections = [
        section_from_record(record, version_key=imported.version.version_key)
        for record in list_sections(session, imported.version.id)
    ]
    paragraphs = [
        paragraph_from_record(record, version_key=imported.version.version_key)
        for record in list_paragraphs(session, imported.version.id)
    ]
    chunker = StructureAwareChunker(_CHUNKER_CONFIG)
    materialized = PassageMaterializer(session).materialize(
        imported.version, chunker.plan(imported.version, sections, paragraphs)
    )
    assert acquired.artifact.content_sha256 == PMC2731074_ARTICLE_SHA256
    return _Ingested(
        version=imported.version,
        chunker_revision=chunker.chunker_revision,
        passage_count=len(materialized.passages),
    )


def _one_hot(index: int, dimension: int) -> tuple[float, ...]:
    """SYNTHETIC TEST VECTORS — NOT EMBEDDINGS.

    A one-hot basis vector of a ``dimension``-dimensional space. Each passage gets
    a distinct basis vector, so the query equal to one of them has that passage as
    its unique nearest neighbour at cosine similarity ``1.0`` and every other
    passage at ``0.0``. The ranking is then a fact about the indexed values rather
    than about the HNSW approximation, which is what makes this a mechanics proof.

    These vectors carry no semantics, so nothing here supports any claim about
    retrieval quality, similarity meaning, or model behaviour.
    """
    return tuple(1.0 if axis == index else 0.0 for axis in range(dimension))


def test_the_mechanics_hold_for_a_real_article_with_nineteen_passages(
    node: _Namespace, db_session: Session, tmp_path: Path
) -> None:
    """PMC scale: 19 passages, 19 vectors, 19 documents, both modalities, rebuild.

    The point of using a real full-text article is cardinality and structure, not
    content: 19 passages exercise a multi-batch-shaped corpus, real section
    nesting and real identifiers, which a four-passage fixture cannot. The vectors
    remain synthetic and the k-NN assertion remains mechanical.
    """
    ingested = _ingest(db_session, tmp_path / "artifacts")

    assert ingested.passage_count == _PMC_PASSAGE_COUNT
    keys = _passage_keys(db_session, ingested.version)
    assert len(keys) == _PMC_PASSAGE_COUNT

    dimension = _PMC_PASSAGE_COUNT
    config = VectorIndexConfig(
        dimension=dimension,
        space=VECTOR_SPACE_COSINESIMIL,
        embedding_model=_SYNTHETIC_MODEL,
    )
    vectors = _synthetic_vectors(
        keys, values=[_one_hot(axis, dimension) for axis in range(dimension)]
    )
    assert len(vectors) == _PMC_PASSAGE_COUNT

    first = node.project_vector(
        db_session,
        chunker_revision=ingested.chunker_revision,
        vectors=vectors,
        vector_config=config,
    )

    # --- 19 passages, 19 vectors, 19 documents, one v2 index ---------------
    assert first.created is True
    assert first.document_count == _PMC_PASSAGE_COUNT
    assert first.projection_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert first.chunker_revision == ingested.chunker_revision
    assert first.index_name.endswith(f"-passage-index-v2-{first.projection_sha256[:12]}")
    assert node.count(first.index_name) == _PMC_PASSAGE_COUNT

    # --- BM25 over the real passages ---------------------------------------
    response = node.service().search("probiotic soy exercise", limit=5)

    assert response.index_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert response.total > 0
    assert response.hits
    assert all(hit.pmcid == _PMC_PMCID for hit in response.hits)
    assert all(hit.source_spans for hit in response.hits)

    # --- test-only k-NN: every passage is the neighbour of its own vector ---
    for axis in (0, 6, _PMC_PASSAGE_COUNT - 1):
        nearest = node.neighbours(node.alias, _one_hot(axis, dimension), k=1)
        assert [hit.passage_key for hit in nearest] == [keys[axis]]
        assert nearest[0].score == pytest.approx(1.0)

    # --- idempotent rebuild: the same digest, name, ids and neighbour -------
    second = node.project_vector(
        db_session,
        chunker_revision=ingested.chunker_revision,
        vectors=vectors,
        vector_config=config,
    )
    assert second.created is False
    assert second.projection_sha256 == first.projection_sha256
    assert second.index_name == first.index_name
    assert [
        hit.passage_key for hit in node.neighbours(node.alias, _one_hot(6, dimension), k=1)
    ] == [keys[6]]

    # --- disposable: delete everything and rebuild from canonical state -----
    node.delete_everything()
    assert node.targets() == ()

    rebuilt = node.project_vector(
        db_session,
        chunker_revision=ingested.chunker_revision,
        vectors=vectors,
        vector_config=config,
    )
    assert rebuilt.index_name == first.index_name
    assert rebuilt.projection_sha256 == first.projection_sha256
    assert node.count(rebuilt.index_name) == _PMC_PASSAGE_COUNT
    assert [
        hit.passage_key for hit in node.neighbours(node.alias, _one_hot(6, dimension), k=1)
    ] == [keys[6]]
    assert node.service().search("probiotic soy exercise", limit=5).total == response.total


def _hybrid_unit(position: int) -> tuple[float, ...]:
    return tuple(1.0 if index == position else 0.0 for index in range(DENSE_DIMENSION))


def _hybrid_vectors(keys: Sequence[str], *, offset: int = 0) -> list[PassageVector]:
    return [
        PassageVector(passage_key=key, values=_hybrid_unit((index + offset) % len(keys)))
        for index, key in enumerate(keys)
    ]


class _FakeHybridQueryEmbedder:
    """A deterministic unit query vector; no TEI or model is involved."""

    def embed_query(self, query: str) -> tuple[float, ...]:
        assert query == "jumping"
        vector = [0.0] * DENSE_DIMENSION
        vector[0] = 0.6
        vector[2] = 0.8
        return tuple(vector)


def test_production_hybrid_search_runs_over_one_live_512d_projection(
    node: _Namespace, db_session: Session
) -> None:
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    projection = node.project_vector(
        db_session,
        chunker_revision=_REVISION,
        vectors=_hybrid_vectors(keys),
        vector_config=_HYBRID_CONFIG,
    )
    service = HybridRetrievalService(
        node.client, alias=node.alias, query_embedder=_FakeHybridQueryEmbedder()
    )

    first = service.retrieve("jumping", limit=4)
    second = service.retrieve("jumping", limit=4)

    assert first.physical_index == projection.index_name
    assert first.lexical.query_revision == BM25_QUERY_REVISION
    assert first.lexical.index_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert [hit.section_title for hit in first.lexical.hits] == ["Introduction", "Methods"]
    assert first.dense.query_revision == "dense-knn-v1"
    assert first.dense.projection_sha256 == projection.projection_sha256
    assert first.dense.candidates[0].passage_key == keys[2]
    assert first.dense.candidates[1].passage_key == keys[0]
    assert first.dense.candidates[0].provenance.source_spans
    assert first.fusion.revision == "rrf-v1"
    assert first.fusion.hits[0].passage_key == keys[0]
    assert first.fusion.hits[0].lexical.present
    assert first.fusion.hits[0].dense.present
    assert (
        first.fusion.hits[0].provenance.source_spans[0].paragraph_source_anchor.startswith("jats:")
    )
    assert [hit.passage_key for hit in first.fusion.hits] == [
        hit.passage_key for hit in second.fusion.hits
    ]
    assert [candidate.passage_key for candidate in first.dense.candidates] == [
        candidate.passage_key for candidate in second.dense.candidates
    ]
    assert all("embedding" not in candidate.model_dump() for candidate in first.dense.candidates)


def test_alias_cutover_after_snapshot_keeps_both_lanes_on_the_captured_index(
    node: _Namespace,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = _seed(db_session)
    keys = _passage_keys(db_session, version)
    first = node.project_vector(
        db_session,
        chunker_revision=_REVISION,
        vectors=_hybrid_vectors(keys),
        vector_config=_HYBRID_CONFIG,
    )
    alternate_alias = f"{node.alias}-alternate"
    alternate = VectorPassageProjector(
        db_session,
        node.client,
        alias=alternate_alias,
        vector_config=_HYBRID_CONFIG,
    ).project(chunker_revision=_REVISION, vectors=_hybrid_vectors(keys, offset=1))
    node.indexes.add(alternate.index_name)
    original_targets = node.client.alias_targets
    original_search = node.client.search
    search_targets: list[str] = []
    switched = False

    def switch_after_snapshot(alias: str) -> tuple[str, ...]:
        nonlocal switched
        targets = original_targets(alias)
        if alias == node.alias and targets == (first.index_name,) and not switched:
            node.client.switch_alias(alias=node.alias, index=alternate.index_name, remove=targets)
            switched = True
        return targets

    def record_search(target: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        search_targets.append(target)
        return original_search(target, body)

    monkeypatch.setattr(node.client, "alias_targets", switch_after_snapshot)
    monkeypatch.setattr(node.client, "search", record_search)
    response = HybridRetrievalService(
        node.client, alias=node.alias, query_embedder=_FakeHybridQueryEmbedder()
    ).retrieve("jumping", limit=4)

    assert switched is True
    assert node.targets() == (alternate.index_name,)
    assert response.physical_index == first.index_name
    assert response.lexical.projection_sha256 == first.projection_sha256
    assert response.dense.projection_sha256 == first.projection_sha256
    assert search_targets == [first.index_name, first.index_name]
