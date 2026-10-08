"""Offline contracts for dense retrieval, alias snapshots and rank-only RRF."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dynamisrag.embedding.contracts import (
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingProviderIdentity,
    TruncationDirection,
)
from dynamisrag.embedding.errors import EmbeddingProviderError
from dynamisrag.embedding.tei import (
    REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
    TEI_HTTP_PROTOCOL_REVISION,
    TEI_PROVIDER_NAME,
)
from dynamisrag.search import retrieval as retrieval_module
from dynamisrag.search.bm25 import SOURCE_FIELDS, SearchHit, SearchResponse, SearchSourceSpan
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import OpenSearchTransportError, SearchBackendError
from dynamisrag.search.retrieval import (
    CANDIDATE_WINDOW,
    DENSE_DIMENSION,
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    DENSE_QUERY_REVISION,
    DENSE_SCHEMA_REVISION,
    DENSE_SOURCE_FIELDS,
    DENSE_SPACE,
    DENSE_TIE_ORDER,
    QUERY_GENERATION_CONFIG,
    RRF_K,
    DenseCandidate,
    DenseSearchResponse,
    HybridRetrievalResponse,
    HybridRetrievalService,
    PassageProvenance,
    QueryEmbedder,
    QueryEmbeddingService,
    QueryEmbeddingUnavailable,
    build_dense_knn_request,
    reciprocal_rank_fusion,
)
from dynamisrag.search.retrieval_router import (
    DENSE_RETRIEVAL_UNAVAILABLE_DETAIL,
    RETRIEVE_PATH,
    build_retrieval_router,
)
from dynamisrag.search.schema import (
    PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    vector_index_meta,
)
from dynamisrag.search.vector import (
    VECTOR_FIELD,
    EmbeddingModelIdentity,
    VectorIndexConfig,
)

_PHYSICAL_INDEX: Final[str] = "dynamisrag-passages-passage-index-v2-123456789abc"
_PROJECTION_SHA: Final[str] = "c" * 64
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_DOC_CONFIG_SHA: Final[str] = hashlib.sha256(b"document prompt config").hexdigest()
_MODEL = EmbeddingModelIdentity(DENSE_MODEL_ID, DENSE_MODEL_REVISION, _DOC_CONFIG_SHA)
_VECTOR_CONFIG = VectorIndexConfig(DENSE_DIMENSION, DENSE_SPACE, _MODEL)
_META = vector_index_meta(
    projection_sha256=_PROJECTION_SHA,
    chunker_revision=_CHUNKER_REVISION,
    vector_config=_VECTOR_CONFIG,
)
_ALIAS = "dynamisrag-passages"


def _source(
    key: str,
    *,
    text: str | None = None,
    projection_sha: str = _PROJECTION_SHA,
    revision: str = VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
) -> dict[str, Any]:
    return {
        "projection_schema_revision": revision,
        "projection_sha256": projection_sha,
        "passage_key": key,
        "document_canonical_key": "doi:10.5555/test.1",
        "document_version_key": "v" * 64,
        "chunker_revision": _CHUNKER_REVISION,
        "passage_ordinal": 0,
        "content_sha256": "5" * 64,
        "text": text or f"Passage {key[:4]} has auditable source text.",
        "title": "A synthetic retrieval article",
        "language": "en",
        "document_type": "journal_article",
        "section_key": "2" * 64,
        "section_path": "1.2",
        "section_title": "Results",
        "primary_source_anchor": "jats:/body[1]/sec[1]/p[1]",
        "source_system": "europe_pmc",
        "source_external_id": "PMC123456",
        "doi": "10.5555/test.1",
        "pmid": "12345678",
        "pmcid": "PMC123456",
        "token_count": 7,
        "source_spans": [
            {
                "source_order": 0,
                "paragraph_key": "4" * 64,
                "paragraph_source_anchor": "jats:/body[1]/sec[1]/p[1]",
                "start_char": 0,
                "end_char": 17,
            }
        ],
    }


def _raw_hit(
    key: str,
    score: float,
    *,
    physical_index: str = _PHYSICAL_INDEX,
    source: dict[str, Any] | None = None,
    include_vector: bool = False,
) -> dict[str, Any]:
    document = _source(key) if source is None else source
    if include_vector:
        document[VECTOR_FIELD] = [1.0] * DENSE_DIMENSION
    return {"_index": physical_index, "_id": key, "_score": score, "_source": document}


def _search_payload(hits: Sequence[Mapping[str, Any]], *, took: int = 3) -> dict[str, Any]:
    return {
        "took": took,
        "hits": {
            "total": {"value": len(hits), "relation": "eq"},
            "hits": list(hits),
        },
    }


class _FakeClient:
    def __init__(
        self,
        *,
        targets: tuple[str, ...] = (_PHYSICAL_INDEX,),
        meta: Mapping[str, Any] = _META,
        lexical_hits: Sequence[Mapping[str, Any]] | None = None,
        dense_hits: Sequence[Mapping[str, Any]] | None = None,
        fail_lane: str | None = None,
    ) -> None:
        self.targets = targets
        self.meta = meta
        self.lexical_hits = (
            [_raw_hit("a" * 64, 3.0), _raw_hit("b" * 64, 2.0)]
            if lexical_hits is None
            else lexical_hits
        )
        self.dense_hits = (
            [_raw_hit("c" * 64, 0.9, include_vector=True), _raw_hit("a" * 64, 0.8)]
            if dense_hits is None
            else dense_hits
        )
        self.fail_lane = fail_lane
        self.alias_calls: list[str] = []
        self.meta_calls: list[str] = []
        self.searches: list[tuple[str, Mapping[str, Any]]] = []

    def alias_targets(self, alias: str) -> tuple[str, ...]:
        self.alias_calls.append(alias)
        return self.targets

    def index_meta(self, index: str) -> Mapping[str, Any]:
        self.meta_calls.append(index)
        return self.meta

    def search(self, index: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.searches.append((index, body))
        is_lexical = "multi_match" in str(body.get("query"))
        lane = "lexical" if is_lexical else "dense"
        if lane == self.fail_lane:
            raise OpenSearchTransportError("untrusted server detail", operation=lane)
        hits = self.lexical_hits if is_lexical else self.dense_hits
        return _search_payload(hits)


class _FakeQueryEmbedder:
    def __init__(self, vector: Sequence[float] | None = None) -> None:
        self.vector = tuple(vector or _unit_vector(0))
        self.calls: list[str] = []

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.calls.append(query)
        return self.vector


def _unit_vector(position: int) -> tuple[float, ...]:
    return tuple(1.0 if index == position else 0.0 for index in range(DENSE_DIMENSION))


def _service(client: _FakeClient, embedder: QueryEmbedder | None = None) -> HybridRetrievalService:
    return HybridRetrievalService(
        cast(OpenSearchClient, client),
        alias=_ALIAS,
        query_embedder=embedder or _FakeQueryEmbedder(),
    )


def _hit(key: str, rank: int, score: float, *, text: str | None = None) -> SearchHit:
    source = _source(key, text=text)
    return SearchHit(
        rank=rank,
        score=score,
        passage_key=key,
        text=source["text"],
        document_canonical_key=source["document_canonical_key"],
        document_version_key=source["document_version_key"],
        title=source["title"],
        language=source["language"],
        chunker_revision=source["chunker_revision"],
        passage_ordinal=source["passage_ordinal"],
        token_count=source["token_count"],
        document_type=source["document_type"],
        content_sha256=source["content_sha256"],
        section_key=source["section_key"],
        section_path=source["section_path"],
        section_title=source["section_title"],
        primary_source_anchor=source["primary_source_anchor"],
        source_spans=(SearchSourceSpan(**source["source_spans"][0]),),
        doi=source["doi"],
        pmid=source["pmid"],
        pmcid=source["pmcid"],
        source_system=source["source_system"],
        source_external_id=source["source_external_id"],
    )


def _source_provenance(key: str, *, text: str | None = None) -> PassageProvenance:
    source = _source(key, text=text)
    source.pop("projection_schema_revision")
    source.pop("projection_sha256")
    return PassageProvenance.model_validate(source)


def _dense(
    key: str, rank: int, score: float, *, provenance: PassageProvenance | None = None
) -> DenseCandidate:
    return DenseCandidate(
        rank=rank,
        raw_score=score,
        passage_key=key,
        provenance=provenance if provenance is not None else _source_provenance(key),
    )


# ---------------------------------------------------------------------------
# Query embedding adapter
# ---------------------------------------------------------------------------


def _provider_identity(
    *, model_id: str = DENSE_MODEL_ID, model_sha: str = DENSE_MODEL_REVISION
) -> EmbeddingProviderIdentity:
    return EmbeddingProviderIdentity(
        provider=TEI_PROVIDER_NAME,
        protocol_revision=TEI_HTTP_PROTOCOL_REVISION,
        runtime_version="1.9.0",
        runtime_sha="e" * 40,
        runtime_docker_label="sha-e80ef22",
        model_id=model_id,
        model_sha=model_sha,
        model_dtype="float32",
        model_pooling="mean",
        max_input_length=32768,
        max_client_batch_size=8,
        max_batch_tokens=8192,
        max_batch_requests=8,
        deployment=REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
    )


class _FakeProvider:
    generation_config = QUERY_GENERATION_CONFIG
    batch_size = 1

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        identity: EmbeddingProviderIdentity | None = None,
    ) -> None:
        self.vectors = tuple(tuple(vector) for vector in vectors)
        self.identity = identity or _provider_identity()
        self.inputs: tuple[EmbeddingInput, ...] = ()
        self.describe_count = 0

    def describe(self) -> EmbeddingProviderIdentity:
        self.describe_count += 1
        return self.identity

    def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
        self.inputs = tuple(inputs)
        return self.vectors


def test_query_generation_semantics_are_explicit_and_content_addressed() -> None:
    provider = _FakeProvider((_unit_vector(7),))

    vector = QueryEmbeddingService(cast(Any, provider)).embed_query("  exact query bytes  ")

    normalized = "exact query bytes"
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    assert (
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=True,
            truncation_direction=TruncationDirection.RIGHT,
            prompt_name="query",
            dimensions=DENSE_DIMENSION,
        )
        == QUERY_GENERATION_CONFIG
    )
    assert vector == _unit_vector(7)
    assert provider.inputs[0].passage_key == f"query:{digest}"
    assert provider.inputs[0].content_sha256 == digest
    assert provider.inputs[0].text == normalized
    assert provider.describe_count == 2


@pytest.mark.parametrize(
    "vectors",
    [(), (_unit_vector(0), _unit_vector(1)), ((1.0,) * (DENSE_DIMENSION - 1),)],
)
def test_query_adapter_rejects_wrong_vector_count_or_dimension(
    vectors: Sequence[Sequence[float]],
) -> None:
    with pytest.raises(QueryEmbeddingUnavailable):
        QueryEmbeddingService(cast(Any, _FakeProvider(vectors))).embed_query("query")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_query_adapter_rejects_non_finite_components(value: float) -> None:
    vector = list(_unit_vector(0))
    vector[0] = value

    with pytest.raises(QueryEmbeddingUnavailable):
        QueryEmbeddingService(cast(Any, _FakeProvider((vector,)))).embed_query("query")


def test_query_adapter_refuses_a_different_model_revision() -> None:
    identity = _provider_identity(model_sha="a" * 40)

    with pytest.raises(QueryEmbeddingUnavailable):
        QueryEmbeddingService(
            cast(Any, _FakeProvider((_unit_vector(0),), identity=identity))
        ).embed_query("query")


def test_query_adapter_requires_the_explicit_normalized_query_generation_config() -> None:
    class _UnnormalizedProvider(_FakeProvider):
        generation_config = EmbeddingGenerationConfig(
            normalize=False,
            truncate=True,
            truncation_direction=TruncationDirection.RIGHT,
            prompt_name="query",
            dimensions=DENSE_DIMENSION,
        )

    with pytest.raises(QueryEmbeddingUnavailable):
        QueryEmbeddingService(cast(Any, _UnnormalizedProvider((_unit_vector(0),))))


def test_query_and_document_generation_digests_are_not_required_to_match() -> None:
    query_digest = (
        _provider_identity()
        .embedding_model_identity(QUERY_GENERATION_CONFIG)
        .embedding_config_sha256
    )

    response = _service(_FakeClient()).retrieve("query")

    assert query_digest != _DOC_CONFIG_SHA
    assert response.dense.document_embedding_config_sha256 == _DOC_CONFIG_SHA


def test_query_adapter_rejects_blank_and_overlong_queries_before_provider_calls() -> None:
    provider = _FakeProvider((_unit_vector(0),))
    adapter = QueryEmbeddingService(cast(Any, provider))

    with pytest.raises(ValueError):
        adapter.embed_query("  ")
    with pytest.raises(ValueError):
        adapter.embed_query("q" * 513)
    assert provider.describe_count == 0


def test_optional_tei_provider_uses_explicit_query_config_when_fully_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests._support import build_settings

    provider = object()
    seen: dict[str, Any] = {}

    def build(settings: Any, *, generation_config: EmbeddingGenerationConfig) -> Any:
        seen["settings"] = settings
        seen["generation_config"] = generation_config
        return provider

    monkeypatch.setattr(retrieval_module, "tei_provider_from_settings", build)
    unconfigured = build_settings(tei_url="http://tei.invalid")
    configured = unconfigured.model_copy(
        update={
            "tei_expected_model_id": DENSE_MODEL_ID,
            "tei_expected_model_sha": DENSE_MODEL_REVISION,
        }
    )

    assert retrieval_module.create_query_embedding_provider(unconfigured) is None
    assert retrieval_module.create_query_embedding_provider(configured) is provider
    assert seen["settings"] is configured
    assert seen["generation_config"] == QUERY_GENERATION_CONFIG


def test_query_adapter_sanitizes_provider_failures_and_keeps_the_query_key_private() -> None:
    class _FailingProvider(_FakeProvider):
        def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
            raise EmbeddingProviderError(
                f"server returned PRIVATE_QUERY_SENTINEL for {inputs[0].passage_key}",
                operation="embed",
                passage_key=inputs[0].passage_key,
            )

    with pytest.raises(QueryEmbeddingUnavailable) as caught:
        QueryEmbeddingService(cast(Any, _FailingProvider(()))).embed_query("query")

    assert "PRIVATE_QUERY_SENTINEL" not in caught.value.safe_summary()
    assert "query:" not in caught.value.safe_summary()


# ---------------------------------------------------------------------------
# Dense request and physical-index snapshot
# ---------------------------------------------------------------------------


def test_dense_knn_request_is_exact_and_never_selects_the_vector_source() -> None:
    vector = _unit_vector(3)

    request = build_dense_knn_request(vector=vector)

    assert request == {
        "size": CANDIDATE_WINDOW,
        "_source": list(DENSE_SOURCE_FIELDS),
        "query": {"knn": {VECTOR_FIELD: {"vector": list(vector), "k": CANDIDATE_WINDOW}}},
    }
    source_fields = request["_source"]
    assert isinstance(source_fields, list)
    assert VECTOR_FIELD not in source_fields
    assert DENSE_SCHEMA_REVISION == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert DENSE_SOURCE_FIELDS == SOURCE_FIELDS
    assert DENSE_TIE_ORDER == ("raw_score descending", "passage_key ascending")


def test_dense_results_sort_ties_by_passage_key_and_hide_returned_vectors() -> None:
    keys = ["b" * 64, "a" * 64]
    client = _FakeClient(
        lexical_hits=[],
        dense_hits=[
            _raw_hit(keys[0], 0.75, include_vector=True),
            _raw_hit(keys[1], 0.75, include_vector=True),
        ],
    )

    response = _service(client).retrieve("query")

    assert [candidate.passage_key for candidate in response.dense.candidates] == [keys[1], keys[0]]
    assert [candidate.rank for candidate in response.dense.candidates] == [1, 2]
    assert all("embedding" not in candidate.model_dump() for candidate in response.dense.candidates)
    dense_request = next(body for _, body in client.searches if "knn" in body["query"])
    assert VECTOR_FIELD not in dense_request["_source"]
    assert response.dense.query_revision == DENSE_QUERY_REVISION
    assert response.dense.document_embedding_config_sha256 == _DOC_CONFIG_SHA


@pytest.mark.parametrize("targets", [(), (_PHYSICAL_INDEX, "other-passage-index-v2")])
def test_hybrid_requires_exactly_one_alias_target(targets: tuple[str, ...]) -> None:
    client = _FakeClient(targets=targets)

    with pytest.raises(SearchBackendError):
        _service(client).retrieve("query")

    assert client.meta_calls == []
    assert client.searches == []


@pytest.mark.parametrize(
    "change",
    [
        {"schema_revision": PASSAGE_INDEX_SCHEMA_REVISION},
        {"embedding_model_id": "other/model"},
        {"embedding_model_revision": "a" * 40},
        {"vector_dimension": 384},
        {"vector_space_type": "l2"},
    ],
)
def test_v1_or_incompatible_vector_metadata_is_refused(change: Mapping[str, Any]) -> None:
    meta = dict(_META)
    meta.update(change)

    with pytest.raises(SearchBackendError):
        _service(_FakeClient(meta=meta)).retrieve("query")


def test_both_branches_use_one_resolved_physical_index_and_one_meta_read() -> None:
    client = _FakeClient()

    response = _service(client).retrieve("query")

    assert response.physical_index == _PHYSICAL_INDEX
    assert client.alias_calls == [_ALIAS]
    assert client.meta_calls == [_PHYSICAL_INDEX]
    assert [index for index, _ in client.searches] == [_PHYSICAL_INDEX, _PHYSICAL_INDEX]
    assert response.lexical.query_revision == "bm25-v1"
    assert response.dense.query_revision == "dense-knn-v1"


@pytest.mark.parametrize("query", ["  ", "q" * 513])
def test_hybrid_query_validation_precedes_backend_work(query: str) -> None:
    client = _FakeClient()

    with pytest.raises(ValueError):
        _service(client).retrieve(query)

    assert client.alias_calls == []
    assert client.meta_calls == []
    assert client.searches == []


@pytest.mark.parametrize("fail_lane", ["lexical", "dense"])
def test_failure_in_either_parallel_branch_fails_the_hybrid_request(fail_lane: str) -> None:
    client = _FakeClient(fail_lane=fail_lane)

    with pytest.raises(OpenSearchTransportError):
        _service(client).retrieve("query")

    assert {"multi_match" in str(body["query"]) for _, body in client.searches} == {True, False}


def test_dense_hits_must_match_the_captured_projection_and_physical_index() -> None:
    mismatch = _raw_hit("c" * 64, 0.9, source=_source("c" * 64, projection_sha="d" * 64))
    client = _FakeClient(lexical_hits=[], dense_hits=[mismatch])

    with pytest.raises(SearchBackendError):
        _service(client).retrieve("query")

    wrong_index = _raw_hit("c" * 64, 0.9, physical_index="another-physical-index")
    client = _FakeClient(lexical_hits=[], dense_hits=[wrong_index])
    with pytest.raises(SearchBackendError):
        _service(client).retrieve("query")


# ---------------------------------------------------------------------------
# RRF semantics
# ---------------------------------------------------------------------------


def test_rrf_handles_lexical_only_dense_only_and_both_with_exact_contributions() -> None:
    shared_key = "a" * 64
    lexical_only = "b" * 64
    dense_only = "c" * 64
    lexical = (_hit(shared_key, 1, 9.0), _hit(lexical_only, 2, 5000.0))
    dense = (_dense(shared_key, 2, 0.1), _dense(dense_only, 1, 9999.0))

    response = reciprocal_rank_fusion(lexical=lexical, dense=dense, limit=3)

    by_key = {candidate.passage_key: candidate for candidate in response.hits}
    assert by_key[shared_key].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert by_key[shared_key].lexical.rrf_contribution == pytest.approx(1 / 61)
    assert by_key[shared_key].dense.rrf_contribution == pytest.approx(1 / 62)
    assert by_key[lexical_only].dense.present is False
    assert by_key[dense_only].lexical.present is False
    assert RRF_K == 60


def test_rrf_ties_are_broken_by_passage_key() -> None:
    response = reciprocal_rank_fusion(
        lexical=(_hit("b" * 64, 1, 999),),
        dense=(_dense("a" * 64, 1, -500),),
        limit=2,
    )

    assert [candidate.passage_key for candidate in response.hits] == ["a" * 64, "b" * 64]


def test_rrf_ignores_raw_scores_and_honours_the_candidate_window() -> None:
    lexical = tuple(
        _hit(f"{index:064x}", index + 1, score=float(100 - index)) for index in range(51)
    )
    baseline = reciprocal_rank_fusion(lexical=lexical, dense=(), limit=50)
    changed_scores = tuple(hit.model_copy(update={"score": -hit.score}) for hit in lexical)
    changed = reciprocal_rank_fusion(lexical=changed_scores, dense=(), limit=50)

    assert [hit.passage_key for hit in baseline.hits] == [hit.passage_key for hit in changed.hits]
    assert len(baseline.hits) == CANDIDATE_WINDOW
    assert f"{50:064x}" not in {hit.passage_key for hit in baseline.hits}


def test_rrf_refuses_duplicates_and_source_provenance_disagreement() -> None:
    duplicate = _hit("a" * 64, 1, 1.0)
    with pytest.raises(SearchBackendError, match="duplicate"):
        reciprocal_rank_fusion(lexical=(duplicate, duplicate), dense=(), limit=5)

    lexical = _hit("a" * 64, 1, 1.0)
    dense = _dense(
        "a" * 64,
        1,
        1.0,
        provenance=_source_provenance("a" * 64, text="changed"),
    )
    with pytest.raises(SearchBackendError, match="provenance"):
        reciprocal_rank_fusion(lexical=(lexical,), dense=(dense,), limit=5)

    with pytest.raises(SearchBackendError, match="non-finite raw score"):
        reciprocal_rank_fusion(lexical=(_hit("d" * 64, 1, float("nan")),), dense=(), limit=5)


# ---------------------------------------------------------------------------
# HTTP surface and safe failures
# ---------------------------------------------------------------------------


def _empty_response(query: str = "query") -> HybridRetrievalResponse:
    lexical = SearchResponse(
        query=query,
        query_revision="bm25-v1",
        index_schema_revision=VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        total=0,
        took_ms=1,
        hits=(),
    )
    dense = DenseSearchResponse(
        query=query,
        query_revision="dense-knn-v1",
        physical_index=_PHYSICAL_INDEX,
        projection_schema_revision="passage-index-v2",
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        vector_config_sha256=_VECTOR_CONFIG.config_sha256,
        model_id=DENSE_MODEL_ID,
        model_revision=DENSE_MODEL_REVISION,
        dimension=512,
        vector_space="cosinesimil",
        document_embedding_config_sha256=_DOC_CONFIG_SHA,
        embedding_profile="res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1",
        embedding_profile_status="provisional",
        candidate_window=50,
        took_ms=1,
        candidates=(),
    )
    return HybridRetrievalResponse(
        query=query,
        retrieval_revision="hybrid-rrf-v1",
        physical_index=_PHYSICAL_INDEX,
        embedding_profile="res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1",
        embedding_profile_status="provisional",
        lexical=lexical,
        dense=dense,
        fusion=reciprocal_rank_fusion(lexical=(), dense=(), limit=10),
    )


class _StubRetrieval:
    def retrieve(self, query: str, *, limit: int = 10) -> HybridRetrievalResponse:
        return _empty_response(query)


def _http(service: object, **params: Any) -> httpx2.Response:
    app = FastAPI()
    app.include_router(build_retrieval_router(retrieval=cast(HybridRetrievalService, service)))
    return TestClient(app).get(RETRIEVE_PATH, params=params or None)


def test_retrieve_http_success_has_three_traces_and_no_store() -> None:
    response = _http(_StubRetrieval(), q="query")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["lexical"]["query_revision"] == "bm25-v1"
    assert body["dense"]["query_revision"] == "dense-knn-v1"
    assert body["fusion"]["revision"] == "rrf-v1"


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": " "},
        {"q": "q" * 513},
        {"q": "query", "limit": 0},
        {"q": "query", "limit": 51},
        {"q": "query", "limit": "many"},
    ],
)
def test_retrieve_http_validation_is_422_no_store(params: Mapping[str, str | int]) -> None:
    response = _http(_StubRetrieval(), **params)

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"


def test_retrieve_http_returns_fixed_safe_backend_errors() -> None:
    class _BackendFailure:
        def retrieve(self, query: str, *, limit: int = 10) -> HybridRetrievalResponse:
            raise OpenSearchTransportError("SECRET_BACKEND_REASON", operation="search")

    class _EmbeddingFailure:
        def retrieve(self, query: str, *, limit: int = 10) -> HybridRetrievalResponse:
            raise EmbeddingProviderError(
                "SECRET_TEI_PROSE", operation="embed", cause="RuntimeError"
            )

    backend = _http(_BackendFailure(), q="query")
    embed = _http(_EmbeddingFailure(), q="query")

    assert backend.status_code == embed.status_code == 503
    assert backend.json() == {"detail": "search is temporarily unavailable"}
    assert embed.json() == {"detail": DENSE_RETRIEVAL_UNAVAILABLE_DETAIL}
    assert "SECRET_" not in backend.text + embed.text
    assert backend.headers["cache-control"] == embed.headers["cache-control"] == "no-store"


def test_retrieve_without_a_query_provider_fails_safely() -> None:
    client = _FakeClient()
    retrieval = HybridRetrievalService(
        cast(OpenSearchClient, client), alias=_ALIAS, query_embedder=None
    )

    response = _http(retrieval, q="query")

    assert response.status_code == 503
    assert response.json() == {"detail": DENSE_RETRIEVAL_UNAVAILABLE_DETAIL}
    assert response.headers["cache-control"] == "no-store"
    assert client.alias_calls == []


def test_serialized_hybrid_response_has_no_raw_vector() -> None:
    response = _service(_FakeClient()).retrieve("query")
    payload = response.model_dump(mode="json")
    serialized = json.dumps(payload)

    assert "embedding" not in payload["dense"]["candidates"][0]
    assert '"vector":' not in serialized
    assert response.dense.document_embedding_config_sha256 == _DOC_CONFIG_SHA
    assert response.dense.dimension == 512
    assert response.dense.vector_space == "cosinesimil"
