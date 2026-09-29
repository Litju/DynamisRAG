"""``GET /search`` against a mocked search backend.

The service is replaced, never the network: these tests are about the HTTP
contract — typed responses, the ``no-store`` header, ``422`` for an invalid
request and ``503`` for a backend outage — and about the fact that a failure
never discloses operational detail to the caller.

The existing health surface must keep working unchanged: the endpoint is
additive, and the application-level tests below prove the real wiring, the
shared client and the ready/unready behaviour all still hold.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from dynamisrag.application import API_DESCRIPTION, create_app
from dynamisrag.config import Settings
from dynamisrag.health.router import LIVENESS_PATH, READINESS_PATH
from dynamisrag.search.bm25 import (
    DEFAULT_LIMIT,
    Bm25SearchService,
    SearchHit,
    SearchResponse,
    SearchSourceSpan,
)
from dynamisrag.search.errors import OpenSearchTransportError, SearchBackendError
from dynamisrag.search.router import SEARCH_PATH, SEARCH_UNAVAILABLE_DETAIL, build_search_router
from tests._support import UNIT_TEST_PASSWORD

_PROJECTION_SHA: Final[str] = "c" * 64
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_PASSAGE_KEY: Final[str] = "a" * 64
_UNAVAILABLE: Final[str] = "search is temporarily unavailable"


def _hit(passage_key: str = _PASSAGE_KEY, score: float = 3.5) -> SearchHit:
    return SearchHit(
        rank=1,
        score=score,
        passage_key=passage_key,
        text="A probiotic soy diet reduced colon lesions in jumping rats.",
        document_canonical_key="doi:10.1371/journal.pone.03089012",
        document_version_key="v" * 64,
        title="Probiotic soy and colon lesions in jumping rats",
        language="en",
        chunker_revision=_CHUNKER_REVISION,
        passage_ordinal=0,
        token_count=12,
        document_type="journal_article",
        content_sha256="5" * 64,
        section_key="2" * 64,
        section_path="2",
        section_title="Results",
        primary_source_anchor="jats:/body[1]/sec[1]/p[1]",
        source_spans=(
            SearchSourceSpan(
                source_order=0,
                paragraph_key="4" * 64,
                paragraph_source_anchor="jats:/body[1]/sec[1]/p[1]",
                start_char=0,
                end_char=59,
            ),
        ),
        doi="10.1371/journal.pone.03089012",
        pmid="38888888",
        pmcid="PMC2731074",
        source_system="europe_pmc",
        source_external_id="PMC2731074",
    )


def _response(
    query: str, *, total: int = 1, hits: tuple[SearchHit, ...] | None = None
) -> SearchResponse:
    return SearchResponse(
        query=query,
        query_revision="bm25-v1",
        index_schema_revision="passage-index-v1",
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        total=total,
        took_ms=4,
        hits=(_hit(),) if hits is None else hits,
    )


class _StubService:
    """Stands in for the BM25 service, recording what the endpoint asked for."""

    def __init__(self, response: SearchResponse | None = None) -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = DEFAULT_LIMIT) -> SearchResponse:
        self.calls.append((query, limit))
        if self.response is None:
            raise SearchBackendError("UnexpectedPayload: boom", operation="search")
        return self.response


class _Rejecting:
    """A service that rejects at the service layer rather than in validation."""

    def search(self, query: str, *, limit: int = DEFAULT_LIMIT) -> SearchResponse:
        raise ValueError("limit must be between 1 and 50, got 0")


class _Leaky:
    """A service whose error message carries operational detail and a secret."""

    def search(self, query: str, *, limit: int = DEFAULT_LIMIT) -> SearchResponse:
        raise OpenSearchTransportError(
            "TransportError: ConnectError: connection refused to "
            f"dynamisrag-passages-passage-index-v1-0a1b2c3d4e5f using {UNIT_TEST_PASSWORD}",
            operation="search",
        )


def _search(service: object, **params: Any) -> httpx2.Response:
    """Issue one ``GET /search`` against a router bound to ``service``.

    A minimal application carrying only the search router, so the HTTP contract
    is tested without the rest of the application in the way.
    """
    app = FastAPI()
    app.include_router(build_search_router(search=cast("Bm25SearchService", service)))
    with StarletteTestClient(app) as client:
        return client.get(SEARCH_PATH, params=params or None)


def _stub(response: SearchResponse | None = None) -> _StubService:
    return _StubService(response)


# ---------------------------------------------------------------------------
# Successful search
# ---------------------------------------------------------------------------


def test_search_returns_a_typed_result() -> None:
    response = _search(_stub(_response("probiotic exercise")), q="probiotic exercise")

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "probiotic exercise"
    assert body["query_revision"] == "bm25-v1"
    assert body["index_schema_revision"] == "passage-index-v1"
    assert body["projection_sha256"] == _PROJECTION_SHA
    assert body["chunker_revision"] == _CHUNKER_REVISION
    assert body["total"] == 1
    assert body["took_ms"] == 4
    hit = body["hits"][0]
    assert hit["passage_key"] == _PASSAGE_KEY
    assert hit["rank"] == 1
    assert hit["score"] == 3.5
    assert hit["source_spans"][0]["paragraph_key"] == "4" * 64
    assert hit["primary_source_anchor"] == "jats:/body[1]/sec[1]/p[1]"


def test_the_response_model_is_the_typed_contract() -> None:
    response = _search(_stub(_response("probiotic")), q="probiotic")

    parsed = SearchResponse.model_validate(response.json())

    assert isinstance(parsed.hits[0], SearchHit)
    assert isinstance(parsed.hits[0].source_spans[0], SearchSourceSpan)


def test_search_is_never_cacheable() -> None:
    response = _search(_stub(_response("probiotic")), q="probiotic")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_the_default_limit_is_ten_and_an_explicit_limit_is_forwarded() -> None:
    service = _stub(_response("probiotic"))

    _search(service, q="probiotic")
    _search(service, q="probiotic", limit=5)

    assert service.calls == [("probiotic", 10), ("probiotic", 5)]


def test_the_query_is_trimmed_before_it_reaches_the_service() -> None:
    service = _stub(_response("probiotic"))

    _search(service, q="  probiotic  ")

    assert service.calls == [("probiotic", 10)]


def test_an_empty_result_is_a_successful_response() -> None:
    response = _search(_stub(_response("nothing", total=0, hits=())), q="nothing")

    assert response.status_code == 200
    assert response.json()["hits"] == []


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "   "}, {"q": "\t"}])
def test_a_missing_or_blank_query_is_a_validation_error(params: Mapping[str, str]) -> None:
    service = _stub(_response("probiotic"))

    response = _search(service, **params)

    assert response.status_code == 422, response.text
    assert service.calls == []


@pytest.mark.parametrize("limit", [0, -5, 51, 10_000])
def test_an_out_of_range_limit_is_a_validation_error(limit: int) -> None:
    service = _stub(_response("probiotic"))

    response = _search(service, q="probiotic", limit=limit)

    assert response.status_code == 422, response.text
    assert service.calls == []


def test_a_non_numeric_limit_is_a_validation_error() -> None:
    response = _search(_stub(_response("probiotic")), q="probiotic", limit="many")

    assert response.status_code == 422


def test_a_service_level_validation_failure_still_maps_to_422() -> None:
    """The endpoint does not rely on FastAPI having caught everything."""

    response = _search(_Rejecting(), q="probiotic")

    assert response.status_code == 422
    assert "between 1 and 50" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Backend outage
# ---------------------------------------------------------------------------


def test_a_backend_outage_is_a_service_unavailable() -> None:
    response = _search(_stub(), q="probiotic")

    assert response.status_code == 503
    assert response.json() == {"detail": _UNAVAILABLE}
    assert SEARCH_UNAVAILABLE_DETAIL == _UNAVAILABLE


def test_an_error_response_discloses_no_operational_detail() -> None:
    response = _search(_Leaky(), q="probiotic")

    assert response.status_code == 503
    assert UNIT_TEST_PASSWORD not in response.text
    assert "passage-index-v1" not in response.text
    assert "connection refused" not in response.text
    assert response.json() == {"detail": _UNAVAILABLE}


def test_an_error_response_is_never_cacheable() -> None:
    response = _search(_stub(), q="probiotic")

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# The real application
# ---------------------------------------------------------------------------


def test_the_application_searches_through_its_own_shared_client(
    offline_settings: Settings,
) -> None:
    """The endpoint is wired into the application, and its only backend is the
    process-wide OpenSearch client — so an unreachable node is a clean 503."""
    with TestClient(create_app(offline_settings)) as client:
        response = client.get(SEARCH_PATH, params={"q": "probiotic"})

        assert response.status_code == 503
        assert response.json() == {"detail": _UNAVAILABLE}
        assert response.headers["cache-control"] == "no-store"


def test_health_endpoints_still_work_alongside_search(offline_settings: Settings) -> None:
    with TestClient(create_app(offline_settings)) as client:
        assert client.get(LIVENESS_PATH).status_code == 200
        # Every dependency is unreachable, so readiness is a clean 503.
        assert client.get(READINESS_PATH).status_code == 503


def test_an_unknown_route_is_still_a_404(offline_settings: Settings) -> None:
    with TestClient(create_app(offline_settings)) as client:
        assert client.get("/not-a-route").status_code == 404


def test_the_openapi_document_publishes_the_search_endpoint() -> None:
    app = FastAPI()
    service = cast("Bm25SearchService", _stub(_response("probiotic")))
    app.include_router(build_search_router(search=service))

    with TestClient(app) as client:
        document: dict[str, Any] = client.get("/openapi.json").json()

    operation: dict[str, Any] = document["paths"][SEARCH_PATH]["get"]
    assert "search" in operation["tags"]
    assert operation["summary"] == "Search passages with BM25"
    assert "422" in operation["responses"]
    assert "503" in operation["responses"]


def test_the_api_description_no_longer_claims_retrieval_is_absent() -> None:
    """The description must not tell a reader that retrieval is missing."""
    assert "Retrieval, ranking" not in API_DESCRIPTION
    assert "not part of this slice" in API_DESCRIPTION
    assert "OpenSearch" in API_DESCRIPTION
    assert "rebuildable" in API_DESCRIPTION
