"""The versioned ``bm25-v1`` query contract, exercised through a mocked node.

The exact request body is the contract: a ranking is only comparable with
another ranking when the query that produced it was identical. These tests
assert the body field by field — fields, boosts, match type, operator,
tie-breaker, sort, ``track_total_hits``, size and ``_source`` — and then the
response side: that a tied ranking keeps the documented ``passage_key`` order,
that a malformed hit is rejected, that a hit whose ``_id`` disagrees with its
``passage_key`` is treated as a projection integrity failure, and that no raw
backend error ever escapes the service.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx2
import pytest

from dynamisrag.search.bm25 import (
    BM25_FIELDS,
    BM25_MATCH_TYPE,
    BM25_OPERATOR,
    BM25_QUERY_REVISION,
    BM25_TIE_BREAKER,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    MIN_LIMIT,
    SOURCE_FIELDS,
    Bm25SearchService,
    SearchHit,
    SearchResponse,
    build_bm25_request,
)
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import (
    OpenSearchTransportError,
    OpenSearchUnexpectedResponse,
    SearchBackendError,
)
from dynamisrag.search.schema import BM25_SIMILARITY_REVISION, PASSAGE_INDEX_SCHEMA_REVISION
from tests._support import UNIT_TEST_PASSWORD, build_settings

_ALIAS: Final[str] = "dynamisrag-passages-test"
_PROJECTION_SHA: Final[str] = "c" * 64
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"

_META: Final[dict[str, str]] = {
    "schema_revision": PASSAGE_INDEX_SCHEMA_REVISION,
    "projection_sha256": _PROJECTION_SHA,
    "chunker_revision": _CHUNKER_REVISION,
    "bm25_similarity_revision": BM25_SIMILARITY_REVISION,
}

_SETTINGS = build_settings(opensearch_url="https://search.internal:9200")


def _source(
    passage_key: str,
    *,
    text: str = "A probiotic soy diet reduced colon lesions in jumping rats.",
    section_title: str | None = "Results",
    section_key: str | None = "2" * 64,
    section_path: str | None = "2",
    primary_source_anchor: str | None = "jats:/body[1]/sec[1]/p[1]",
    doi: str | None = "10.1371/journal.pone.03089012",
    pmid: str | None = "38888888",
    pmcid: str | None = "PMC2731074",
) -> dict[str, Any]:
    """One fully valid projected document, as the index would store it."""
    return {
        "projection_schema_revision": PASSAGE_INDEX_SCHEMA_REVISION,
        "projection_sha256": _PROJECTION_SHA,
        "passage_key": passage_key,
        "document_canonical_key": "doi:10.1371/journal.pone.03089012",
        "document_version_key": "v" * 64,
        "chunker_revision": _CHUNKER_REVISION,
        "passage_ordinal": 0,
        "content_sha256": "5" * 64,
        "text": text,
        "title": "Probiotic soy and colon lesions in jumping rats",
        "language": "en",
        "document_type": "journal_article",
        "section_key": section_key,
        "section_path": section_path,
        "section_title": section_title,
        "primary_source_anchor": primary_source_anchor,
        "source_system": "europe_pmc",
        "source_external_id": "PMC2731074",
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "token_count": 12,
        "source_spans": [
            {
                "source_order": 0,
                "paragraph_key": "4" * 64,
                "paragraph_source_anchor": "jats:/body[1]/sec[1]/p[1]",
                "start_char": 0,
                "end_char": 59,
            }
        ],
    }


def _search_payload(
    hits: list[dict[str, Any]], *, total: int | None = None, took: int = 3
) -> dict[str, Any]:
    return {
        "took": took,
        "timed_out": False,
        "hits": {
            "total": {"value": len(hits) if total is None else total, "relation": "eq"},
            "max_score": 1.0,
            "hits": hits,
        },
    }


def _hit(passage_key: str, score: float, **overrides: Any) -> dict[str, Any]:
    return {
        "_index": _ALIAS,
        "_id": passage_key,
        "_score": score,
        "_source": _source(passage_key, **overrides),
    }


class _Node:
    """Answers the mapping lookup and the search, recording both requests."""

    def __init__(
        self,
        *,
        meta: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.meta = _META if meta is None else meta
        self.payload = _search_payload([]) if payload is None else payload
        self.raises = raises
        self.requests: list[httpx2.Request] = []

    def service(self) -> Bm25SearchService:
        def answer(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            if self.raises is not None:
                raise self.raises
            if request.url.path.endswith("/_mapping"):
                return httpx2.Response(200, json={_ALIAS: {"mappings": {"_meta": self.meta}}})
            return httpx2.Response(200, json=self.payload)

        client = OpenSearchClient(_SETTINGS, transport=httpx2.MockTransport(answer))
        return Bm25SearchService(client, alias=_ALIAS)

    def search_request(self) -> httpx2.Request:
        return next(request for request in self.requests if request.url.path.endswith("/_search"))


# ---------------------------------------------------------------------------
# The exact bm25-v1 request
# ---------------------------------------------------------------------------


def test_the_query_revision_and_boosts_are_explicit_constants() -> None:
    assert BM25_QUERY_REVISION == "bm25-v1"
    assert BM25_FIELDS == (("title", 2.0), ("section_title", 1.5), ("text", 1.0))
    assert BM25_MATCH_TYPE == "best_fields"
    assert BM25_OPERATOR == "or"
    assert BM25_TIE_BREAKER == 0.1


def test_the_request_body_is_exact() -> None:
    assert build_bm25_request(query="probiotic exercise", limit=5) == {
        "query": {
            "multi_match": {
                "query": "probiotic exercise",
                "fields": ["title^2.0", "section_title^1.5", "text^1.0"],
                "type": "best_fields",
                "operator": "or",
                "tie_breaker": 0.1,
            }
        },
        "sort": [{"_score": "desc"}, {"passage_key": "asc"}],
        "track_total_hits": True,
        "size": 5,
        "_source": list(SOURCE_FIELDS),
    }


def test_the_query_carries_no_fuzziness_synonyms_or_reranking() -> None:
    """Those belong to a later evaluation issue; adding any of them under
    ``bm25-v1`` would silently invalidate every comparison made so far."""
    body = json.dumps(build_bm25_request(query="probiotic", limit=10))

    for forbidden in ("fuzziness", "synonym", "rescore", "rerank", "boosting", "query_string"):
        assert forbidden not in body


def test_the_source_selection_is_exactly_the_response_contract() -> None:
    """Selecting fields explicitly is what stops a raw backend document from
    leaking into the typed response, and what keeps the response body
    proportional to the contract rather than to the index mapping."""
    hit_fields = set(SearchHit.model_fields)
    assert hit_fields <= set(SOURCE_FIELDS) | {"rank", "score"}
    assert "source_uri" not in SOURCE_FIELDS
    assert "versioned_metadata" not in SOURCE_FIELDS
    assert len(SOURCE_FIELDS) == len(set(SOURCE_FIELDS))


def test_the_service_sends_the_exact_body_to_the_stable_alias() -> None:
    node = _Node(payload=_search_payload([_hit("a" * 64, 1.0)]))

    node.service().search("probiotic soy", limit=3)

    request = node.search_request()
    assert str(request.url) == f"https://search.internal:9200/{_ALIAS}/_search"
    assert json.loads(request.content) == build_bm25_request(query="probiotic soy", limit=3)


def test_track_total_hits_is_enabled_for_an_exact_count() -> None:
    assert build_bm25_request(query="x", limit=10)["track_total_hits"] is True


def test_the_sort_is_score_then_passage_key() -> None:
    """BM25 scores tie, and a stable order is what makes a ranking assertable."""
    assert build_bm25_request(query="x", limit=10)["sort"] == [
        {"_score": "desc"},
        {"passage_key": "asc"},
    ]


def test_the_default_limit_is_ten() -> None:
    assert (MIN_LIMIT, DEFAULT_LIMIT, MAX_LIMIT) == (1, 10, 50)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected_before_any_request(query: str) -> None:
    node = _Node()

    with pytest.raises(ValueError, match="non-whitespace"):
        node.service().search(query)

    assert node.requests == []


def test_a_query_is_trimmed_before_it_is_sent() -> None:
    node = _Node(payload=_search_payload([]))

    node.service().search("  probiotic  ")

    assert json.loads(node.search_request().content)["query"]["multi_match"]["query"] == "probiotic"


def test_an_over_long_query_is_rejected() -> None:
    node = _Node()

    with pytest.raises(ValueError, match=f"at most {MAX_QUERY_LENGTH}"):
        node.service().search("x" * (MAX_QUERY_LENGTH + 1))

    assert node.requests == []


@pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1, 1000])
def test_an_out_of_range_limit_is_rejected_before_any_request(limit: int) -> None:
    node = _Node()

    with pytest.raises(ValueError, match="between 1 and 50"):
        node.service().search("probiotic", limit=limit)

    assert node.requests == []


def test_a_boolean_limit_is_rejected() -> None:
    node = _Node()

    with pytest.raises(ValueError, match="must be an integer"):
        node.service().search("probiotic", limit=True)

    assert node.requests == []


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------


def test_a_response_carries_every_provenance_revision() -> None:
    node = _Node(payload=_search_payload([_hit("a" * 64, 2.5)], took=7))

    response = node.service().search("probiotic")

    assert response.query == "probiotic"
    assert response.query_revision == "bm25-v1"
    assert response.index_schema_revision == PASSAGE_INDEX_SCHEMA_REVISION
    assert response.projection_sha256 == _PROJECTION_SHA
    assert response.chunker_revision == _CHUNKER_REVISION
    assert response.total == 1
    assert response.took_ms == 7
    assert response.hits[0].rank == 1
    assert response.hits[0].score == 2.5


def test_a_hit_exposes_the_whole_audit_contract() -> None:
    node = _Node(payload=_search_payload([_hit("a" * 64, 1.0)]))

    hit = node.service().search("probiotic").hits[0]

    assert hit.passage_key == "a" * 64
    assert hit.document_canonical_key == "doi:10.1371/journal.pone.03089012"
    assert hit.document_version_key == "v" * 64
    assert hit.title == "Probiotic soy and colon lesions in jumping rats"
    assert hit.language == "en"
    assert hit.chunker_revision == _CHUNKER_REVISION
    assert hit.passage_ordinal == 0
    assert hit.token_count == 12
    assert hit.section_key == "2" * 64
    assert hit.section_path == "2"
    assert hit.section_title == "Results"
    assert hit.primary_source_anchor == "jats:/body[1]/sec[1]/p[1]"
    assert (hit.doi, hit.pmid, hit.pmcid) == (
        "10.1371/journal.pone.03089012",
        "38888888",
        "PMC2731074",
    )
    assert (hit.source_system, hit.source_external_id) == ("europe_pmc", "PMC2731074")
    assert hit.source_spans[0].paragraph_key == "4" * 64
    assert hit.source_spans[0].start_char == 0
    assert hit.source_spans[0].end_char == 59


def test_optional_structural_metadata_may_be_absent() -> None:
    payload = _search_payload(
        [_hit("a" * 64, 1.0, section_title=None, section_key=None, section_path=None, doi=None)]
    )
    node = _Node(payload=payload)

    hit = node.service().search("probiotic").hits[0]

    assert hit.section_title is None
    assert hit.section_key is None
    assert hit.section_path is None
    assert hit.doi is None


def test_ranks_are_one_based_and_dense() -> None:
    payload = _search_payload([_hit("a" * 64, 3.0), _hit("b" * 64, 1.0), _hit("c" * 64, 0.5)])
    node = _Node(payload=payload)

    response = node.service().search("probiotic")

    assert [hit.rank for hit in response.hits] == [1, 2, 3]


def test_tied_scores_retain_the_passage_key_order_the_backend_returned() -> None:
    """The query sorts by ``_score`` then ``passage_key``; the service must not
    reorder, so a tied result set keeps the stable order it was given."""
    payload = _search_payload([_hit("a" * 64, 1.5), _hit("b" * 64, 1.5), _hit("c" * 64, 1.5)])
    node = _Node(payload=payload)

    response = node.service().search("probiotic")

    assert [hit.passage_key for hit in response.hits] == ["a" * 64, "b" * 64, "c" * 64]
    assert [hit.score for hit in response.hits] == [1.5, 1.5, 1.5]


def test_an_empty_result_is_a_valid_response() -> None:
    node = _Node(payload=_search_payload([], total=0))

    response = node.service().search("nothing-matches-this")

    assert response.hits == ()
    assert response.total == 0


def test_the_exact_total_is_reported_even_beyond_the_returned_hits() -> None:
    node = _Node(payload=_search_payload([_hit("a" * 64, 1.0)], total=1234))

    response = node.service().search("probiotic", limit=1)

    assert response.total == 1234
    assert len(response.hits) == 1


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def test_a_malformed_hit_is_rejected() -> None:
    node = _Node(payload=_search_payload([{"_id": "a" * 64, "_score": 1.0}]))

    with pytest.raises(SearchBackendError, match="carries no _source"):
        node.service().search("probiotic")


def test_a_hit_that_is_not_an_object_is_rejected() -> None:
    node = _Node(payload=_search_payload(["not-an-object"]))  # type: ignore[list-item]

    with pytest.raises(SearchBackendError, match="not an object"):
        node.service().search("probiotic")


def test_an_id_and_passage_key_mismatch_is_a_projection_integrity_error() -> None:
    """The index is written with ``_id = passage_key``; a disagreement means the
    projection contradicts its own identity rule, not that the query was odd."""
    payload = _search_payload([{"_id": "b" * 64, "_score": 1.0, "_source": _source("a" * 64)}])
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="ProjectionIntegrity"):
        node.service().search("probiotic")


def test_a_missing_id_is_rejected() -> None:
    payload = _search_payload([{"_score": 1.0, "_source": _source("a" * 64)}])
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="ProjectionIntegrity"):
        node.service().search("probiotic")


def test_a_hit_indexed_under_different_projection_semantics_is_rejected() -> None:
    payload = _search_payload([_hit("a" * 64, 1.0)])
    payload["hits"]["hits"][0]["_source"]["projection_sha256"] = "9" * 64
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="ProjectionIntegrity"):
        node.service().search("probiotic")


def test_a_hit_without_a_score_is_rejected() -> None:
    payload = _search_payload([{"_id": "a" * 64, "_source": _source("a" * 64)}])
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="no numeric _score"):
        node.service().search("probiotic")


def test_a_missing_required_source_field_is_rejected() -> None:
    payload = _search_payload([_hit("a" * 64, 1.0)])
    del payload["hits"]["hits"][0]["_source"]["title"]
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="no non-empty 'title'"):
        node.service().search("probiotic")


def test_a_hit_with_a_non_integer_ordinal_is_rejected() -> None:
    payload = _search_payload([_hit("a" * 64, 1.0)])
    payload["hits"]["hits"][0]["_source"]["passage_ordinal"] = "zero"
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="no integer 'passage_ordinal'"):
        node.service().search("probiotic")


def test_a_hit_without_source_spans_is_rejected() -> None:
    """A hit that cannot be audited back to exact source characters is not a
    valid result of this contract."""
    payload = _search_payload([_hit("a" * 64, 1.0)])
    payload["hits"]["hits"][0]["_source"]["source_spans"] = None
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="no source_spans list"):
        node.service().search("probiotic")


def test_a_source_span_that_does_not_satisfy_the_contract_is_rejected() -> None:
    payload = _search_payload([_hit("a" * 64, 1.0)])
    payload["hits"]["hits"][0]["_source"]["source_spans"][0]["end_char"] = -1
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="source span"):
        node.service().search("probiotic")


def test_a_response_without_a_total_is_rejected() -> None:
    payload = _search_payload([])
    del payload["hits"]["total"]
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="no total hit count"):
        node.service().search("probiotic")


def test_more_hits_than_the_limit_is_rejected() -> None:
    payload = _search_payload([_hit("a" * 64, 1.0), _hit("b" * 64, 1.0)])
    node = _Node(payload=payload)

    with pytest.raises(SearchBackendError, match="2 hits for a limit of 1"):
        node.service().search("probiotic", limit=1)


def test_an_index_written_by_another_similarity_revision_is_rejected() -> None:
    node = _Node(
        meta={**_META, "bm25_similarity_revision": "something-else"}, payload=_search_payload([])
    )

    with pytest.raises(SearchBackendError, match="similarity revision"):
        node.service().search("probiotic")


def test_an_index_written_by_another_schema_revision_is_rejected() -> None:
    node = _Node(meta={**_META, "schema_revision": "passage-index-v0"}, payload=_search_payload([]))

    with pytest.raises(SearchBackendError, match="schema revision"):
        node.service().search("probiotic")


def test_an_index_without_mapping_meta_is_rejected() -> None:
    node = _Node(meta={}, payload=_search_payload([]))

    with pytest.raises(SearchBackendError, match="no 'schema_revision'"):
        node.service().search("probiotic")


# ---------------------------------------------------------------------------
# The backend error surface
# ---------------------------------------------------------------------------


def test_a_transport_failure_surfaces_as_a_typed_error_without_credentials() -> None:
    node = _Node(raises=httpx2.ConnectError("connection refused"))

    with pytest.raises(OpenSearchTransportError) as caught:
        node.service().search("probiotic")

    assert "ConnectError" in str(caught.value)
    assert UNIT_TEST_PASSWORD not in str(caught.value)


def test_an_unexpected_status_surfaces_as_a_typed_error() -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/_mapping"):
            return httpx2.Response(200, json={_ALIAS: {"mappings": {"_meta": _META}}})
        return httpx2.Response(
            503,
            json={"error": {"type": "search_phase_execution_exception", "reason": "shard down"}},
        )

    client = OpenSearchClient(_SETTINGS, transport=httpx2.MockTransport(answer))

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        Bm25SearchService(client, alias=_ALIAS).search("probiotic")

    assert "UnexpectedStatus: HTTP 503" in str(caught.value)
    assert "search_phase_execution_exception" in str(caught.value)


def test_a_raw_backend_response_object_never_escapes_the_service() -> None:
    node = _Node(payload=_search_payload([_hit("a" * 64, 1.0)]))

    response = node.service().search("probiotic")

    assert isinstance(response, SearchResponse)
    assert set(response.model_dump()) == {
        "query",
        "query_revision",
        "index_schema_revision",
        "projection_sha256",
        "chunker_revision",
        "total",
        "took_ms",
        "hits",
    }
    assert all(isinstance(hit, SearchHit) for hit in response.hits)
