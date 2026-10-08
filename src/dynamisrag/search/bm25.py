"""Versioned BM25 query contract and search service (RES-135).

Every value that can change *what a query means* lives here as an explicit
constant tied to :data:`BM25_QUERY_REVISION`, never as a literal scattered
through request-building code. A change to the queried fields, their boosts,
the match type, the operator or the tie-breaker is a change of query semantics
and must bump that revision; the revision is returned in every
:class:`SearchResponse` so a score is always attributable to the query shape
that produced it.

``bm25-v1`` is deliberately the plain lexical baseline:

* ``multi_match`` over ``best_fields`` with ``operator=or`` and
  ``tie_breaker=0.1``,
* explicit field boosts (title, then section title, then passage text),
* no fuzziness, no synonyms, no query expansion, no reranking, no rescoring.

Those belong to a later evaluation issue, and adding any of them under
``bm25-v1`` would silently invalidate every comparison already made against
it. Sorting is ``_score`` descending with ``passage_key`` ascending as the
tie-break: BM25 scores tie, and a stable result order is what makes a ranking
assertable and a response reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dynamisrag.search.client import JsonValue, OpenSearchClient
from dynamisrag.search.errors import SearchBackendError
from dynamisrag.search.schema import (
    BM25_COMPATIBLE_INDEX_SCHEMA_REVISIONS,
    BM25_SIMILARITY_REVISION,
)

__all__ = [
    "BM25_FIELDS",
    "BM25_MATCH_TYPE",
    "BM25_OPERATOR",
    "BM25_QUERY_REVISION",
    "BM25_TIE_BREAKER",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MAX_QUERY_LENGTH",
    "MIN_LIMIT",
    "SOURCE_FIELDS",
    "Bm25SearchService",
    "SearchHit",
    "SearchResponse",
    "SearchSourceSpan",
    "build_bm25_request",
    "parse_search_hit",
    "validate_limit",
    "validate_query",
]

BM25_QUERY_REVISION: Final[str] = "bm25-v1"
"""Revision of the BM25 *query* semantics. Bump on any field, boost, match
type, operator or tie-breaker change."""

BM25_FIELDS: Final[tuple[tuple[str, float], ...]] = (
    ("title", 2.0),
    ("section_title", 1.5),
    ("text", 1.0),
)
"""Ranked fields and their boosts, highest first.

A document title is the strongest evidence that a passage is on-topic; the
section title is local context; the passage text is the evidence itself.
"""

BM25_MATCH_TYPE: Final[str] = "best_fields"
"""Only the single best field contributes; no cross-field term fusion."""

BM25_OPERATOR: Final[str] = "or"
"""Any query term may match. ``and`` would silently make a two-word query
return nothing for a corpus where the words never co-occur."""

BM25_TIE_BREAKER: Final[float] = 0.1
"""How much the non-best fields contribute when several fields match."""

MIN_LIMIT: Final[int] = 1
MAX_LIMIT: Final[int] = 50
DEFAULT_LIMIT: Final[int] = 10
"""Bounded result window. There is no pagination in this slice, so an
unbounded ``size`` would be an unbounded response body."""

MAX_QUERY_LENGTH: Final[int] = 512
"""Upper bound on the raw query string, so an obviously abusive request is
rejected by validation instead of consuming backend work."""

SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "passage_key",
    "document_canonical_key",
    "document_version_key",
    "chunker_revision",
    "passage_ordinal",
    "token_count",
    "text",
    "title",
    "language",
    "document_type",
    "section_key",
    "section_path",
    "section_title",
    "primary_source_anchor",
    "content_sha256",
    "source_system",
    "source_external_id",
    "doi",
    "pmid",
    "pmcid",
    "source_spans",
    "projection_schema_revision",
    "projection_sha256",
)
"""Exactly the ``_source`` fields a :class:`SearchHit` needs — no more.

Selecting fields explicitly is what keeps a raw backend document from leaking
into the typed response and keeps the response body proportional to the
contract rather than to the index mapping.
"""


class SearchSourceSpan(BaseModel):
    """One exact source span of a hit, as projected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_order: int = Field(ge=0)
    paragraph_key: str
    paragraph_source_anchor: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)


class SearchHit(BaseModel):
    """One ranked passage with everything needed to audit it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float
    passage_key: str
    text: str
    document_canonical_key: str
    document_version_key: str
    title: str
    language: str
    chunker_revision: str
    passage_ordinal: int = Field(ge=0)
    token_count: int = Field(ge=0)
    document_type: str
    content_sha256: str
    section_key: str | None = None
    section_path: str | None = None
    section_title: str | None = None
    primary_source_anchor: str | None = None
    source_spans: tuple[SearchSourceSpan, ...] = ()
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    source_system: str
    source_external_id: str


class SearchResponse(BaseModel):
    """The complete result of one BM25 query.

    Carries the revisions that produced it — query revision, index schema
    revision, projection digest and chunker revision — so a score is never
    anonymous, and carries no raw OpenSearch object.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    query_revision: str
    index_schema_revision: str
    projection_sha256: str
    chunker_revision: str
    total: int = Field(ge=0)
    took_ms: int = Field(ge=0)
    hits: tuple[SearchHit, ...] = ()


def _boosted_fields() -> list[str]:
    return [f"{field}^{boost}" for field, boost in BM25_FIELDS]


def build_bm25_request(*, query: str, limit: int) -> Mapping[str, JsonValue]:
    """The exact ``bm25-v1`` search request body.

    Pure and total: same inputs, same body, every time. The sort places
    ``passage_key`` ascending behind ``_score`` so tied scores have a stable,
    documented order instead of whatever order the shards happened to return.
    """
    body: Mapping[str, JsonValue] = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": _boosted_fields(),
                "type": BM25_MATCH_TYPE,
                "operator": BM25_OPERATOR,
                "tie_breaker": BM25_TIE_BREAKER,
            }
        },
        "sort": [{"_score": "desc"}, {"passage_key": "asc"}],
        "track_total_hits": True,
        "size": limit,
        "_source": list(SOURCE_FIELDS),
    }
    return body


class Bm25SearchService:
    """Deterministic BM25 retrieval over the versioned passage projection.

    Search reads the disposable OpenSearch projection and nothing else: no
    PostgreSQL round trip, no document rehydration, no source reparsing. That
    separation is deliberate — the projection already carries every field a hit
    needs, so a query cannot silently disagree with what was indexed.
    """

    __slots__ = ("_alias", "_client")

    def __init__(self, client: OpenSearchClient, *, alias: str) -> None:
        self._client: Final[OpenSearchClient] = client
        self._alias: Final[str] = alias

    def search(self, query: str, *, limit: int = DEFAULT_LIMIT) -> SearchResponse:
        """Run one ``bm25-v1`` query against the stable alias.

        Raises :class:`ValueError` for an invalid query or limit, and
        :class:`~dynamisrag.search.errors.OpenSearchError` for a backend or
        response-validation failure. Raw OpenSearch objects never escape.
        """
        normalized = validate_query(query)
        validate_limit(limit)
        meta = self._client.index_meta(self._alias)
        response = self._client.search(
            self._alias, build_bm25_request(query=normalized, limit=limit)
        )
        return self._to_response(query=normalized, limit=limit, meta=meta, payload=response)

    def search_resolved(
        self,
        query: str,
        *,
        limit: int,
        physical_index: str,
        meta: Mapping[str, JsonValue],
    ) -> SearchResponse:
        """Run unchanged ``bm25-v1`` against a physical index and a captured ``_meta``.

        Hybrid retrieval resolves the alias once before either lane runs. This
        entry point keeps the lexical parser and request contract shared while
        ensuring that BM25 cannot independently resolve the alias again.
        """
        normalized = validate_query(query)
        validate_limit(limit)
        payload = self._client.search(
            physical_index, build_bm25_request(query=normalized, limit=limit)
        )
        return self._to_response(query=normalized, limit=limit, meta=meta, payload=payload)

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    def _to_response(
        self,
        *,
        query: str,
        limit: int,
        meta: Mapping[str, JsonValue],
        payload: Mapping[str, JsonValue],
    ) -> SearchResponse:
        index_schema_revision = _require_meta_str(meta, "schema_revision")
        projection_sha256 = _require_meta_str(meta, "projection_sha256")
        chunker_revision = _require_meta_str(meta, "chunker_revision")
        similarity_revision = _require_meta_str(meta, "bm25_similarity_revision")
        if similarity_revision != BM25_SIMILARITY_REVISION:
            raise SearchBackendError(
                f"UnexpectedPayload: the active index declares BM25 similarity revision "
                f"{similarity_revision!r} but this build searches with "
                f"{BM25_SIMILARITY_REVISION!r}",
                operation="search",
            )
        if index_schema_revision not in BM25_COMPATIBLE_INDEX_SCHEMA_REVISIONS:
            # Compatibility is a property of the lexical mapping, not a version
            # number, so the accepted set is declared in `schema` and membership
            # is checked here. An unknown revision is refused rather than searched:
            # the query would be answered by analysis this build does not
            # implement, which returns a confident ranking nobody chose.
            raise SearchBackendError(
                f"UnexpectedPayload: the active index declares projection schema revision "
                f"{index_schema_revision!r}, which this build cannot serve. Query revision "
                f"{BM25_QUERY_REVISION} is defined against "
                f"{sorted(BM25_COMPATIBLE_INDEX_SCHEMA_REVISIONS)}",
                operation="search",
            )

        hits_block = payload.get("hits")
        if not isinstance(hits_block, dict):
            raise SearchBackendError(
                "UnexpectedPayload: search response carries no hits object", operation="search"
            )
        raw_hits = hits_block.get("hits")
        if not isinstance(raw_hits, list):
            raise SearchBackendError(
                "UnexpectedPayload: search response carries no hits list", operation="search"
            )
        if len(raw_hits) > limit:
            raise SearchBackendError(
                f"UnexpectedPayload: search returned {len(raw_hits)} hits for a limit of {limit}",
                operation="search",
            )

        hits = tuple(
            parse_search_hit(
                position=position,
                raw=raw,
                chunker_revision=chunker_revision,
                index_schema_revision=index_schema_revision,
                projection_sha256=projection_sha256,
            )
            for position, raw in enumerate(raw_hits, start=1)
        )
        return SearchResponse(
            query=query,
            query_revision=BM25_QUERY_REVISION,
            index_schema_revision=index_schema_revision,
            projection_sha256=projection_sha256,
            chunker_revision=chunker_revision,
            total=_require_total(hits_block),
            took_ms=_require_int(payload, "took", "search"),
            hits=hits,
        )


def parse_search_hit(
    *,
    position: int,
    raw: JsonValue,
    chunker_revision: str,
    index_schema_revision: str,
    projection_sha256: str,
    physical_index: str | None = None,
) -> SearchHit:
    """Parse one lexical or dense hit through the shared provenance contract."""
    if not isinstance(raw, dict):
        raise SearchBackendError(
            f"UnexpectedPayload: hit {position} is a {type(raw).__name__}, not an object",
            operation="search",
        )
    if physical_index is not None and raw.get("_index") != physical_index:
        raise SearchBackendError(
            f"ProjectionIntegrity: hit {position} came from a different physical index",
            operation="search",
        )
    source = raw.get("_source")
    if not isinstance(source, dict):
        raise SearchBackendError(
            f"UnexpectedPayload: hit {position} carries no _source", operation="search"
        )
    document_id = raw.get("_id")
    if not isinstance(document_id, str) or document_id != source.get("passage_key"):
        raise SearchBackendError(
            f"ProjectionIntegrity: hit {position} has _id {document_id!r} but "
            f"_source.passage_key {source.get('passage_key')!r}; the projection is "
            "inconsistent with its own identity rule",
            operation="search",
        )
    if (
        source.get("chunker_revision") != chunker_revision
        or source.get("projection_schema_revision") != index_schema_revision
        or source.get("projection_sha256") != projection_sha256
    ):
        raise SearchBackendError(
            f"ProjectionIntegrity: hit {position} was indexed under different projection "
            "semantics than the active index declares",
            operation="search",
        )
    score = raw.get("_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SearchBackendError(
            f"UnexpectedPayload: hit {position} carries no numeric _score", operation="search"
        )
    try:
        return SearchHit(
            rank=position,
            score=float(score),
            passage_key=_require_str(source, "passage_key", position),
            text=_require_str(source, "text", position),
            document_canonical_key=_require_str(source, "document_canonical_key", position),
            document_version_key=_require_str(source, "document_version_key", position),
            title=_require_str(source, "title", position),
            language=_require_str(source, "language", position),
            chunker_revision=_require_str(source, "chunker_revision", position),
            passage_ordinal=_require_int(source, "passage_ordinal", f"hit {position}"),
            token_count=_require_int(source, "token_count", f"hit {position}"),
            document_type=_require_str(source, "document_type", position),
            content_sha256=_require_str(source, "content_sha256", position),
            section_key=_optional_str(source, "section_key"),
            section_path=_optional_str(source, "section_path"),
            section_title=_optional_str(source, "section_title"),
            primary_source_anchor=_optional_str(source, "primary_source_anchor"),
            source_spans=_source_spans(source, position),
            doi=_optional_str(source, "doi"),
            pmid=_optional_str(source, "pmid"),
            pmcid=_optional_str(source, "pmcid"),
            source_system=_require_str(source, "source_system", position),
            source_external_id=_require_str(source, "source_external_id", position),
        )
    except ValidationError as error:
        raise SearchBackendError(
            f"UnexpectedPayload: hit {position} does not satisfy the search contract: "
            f"{_flatten(error)}",
            operation="search",
        ) from error


def validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must contain non-whitespace text")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")
    return normalized


def validate_limit(limit: int) -> None:
    # `bool` is a subclass of `int`; `True` is not a meaningful result window.
    if isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}, got {limit}")


def _require_meta_str(meta: Mapping[str, JsonValue], key: str) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value:
        raise SearchBackendError(
            f"UnexpectedPayload: the active index mapping _meta carries no {key!r} string",
            operation="search",
        )
    return value


def _require_total(hits_block: Mapping[str, JsonValue]) -> int:
    total = hits_block.get("total")
    if isinstance(total, int) and not isinstance(total, bool):
        # `track_total_hits=true` always yields the object form; the bare integer
        # is accepted only as a defensive fallback.
        return total
    if isinstance(total, dict):
        return _require_int(total, "value", "search hits.total")
    raise SearchBackendError(
        "UnexpectedPayload: search response carries no total hit count", operation="search"
    )


def _require_str(source: Mapping[str, JsonValue], key: str, position: int) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise SearchBackendError(
            f"UnexpectedPayload: hit {position} carries no non-empty {key!r} string",
            operation="search",
        )
    return value


def _require_int(source: Mapping[str, JsonValue], key: str, context: str) -> int:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SearchBackendError(
            f"UnexpectedPayload: {context} carries no integer {key!r}", operation="search"
        )
    return value


def _optional_str(source: Mapping[str, JsonValue], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SearchBackendError(
            f"UnexpectedPayload: field {key!r} is neither a string nor null", operation="search"
        )
    return value


def _source_spans(source: Mapping[str, JsonValue], position: int) -> tuple[SearchSourceSpan, ...]:
    raw = source.get("source_spans")
    if not isinstance(raw, list):
        raise SearchBackendError(
            f"UnexpectedPayload: hit {position} carries no source_spans list", operation="search"
        )
    spans: list[SearchSourceSpan] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise SearchBackendError(
                f"UnexpectedPayload: hit {position} carries a non-object source span",
                operation="search",
            )
        try:
            spans.append(
                SearchSourceSpan(
                    source_order=_require_int(item, "source_order", f"hit {position} source span"),
                    paragraph_key=_require_str(item, "paragraph_key", position),
                    paragraph_source_anchor=_require_str(item, "paragraph_source_anchor", position),
                    start_char=_require_int(item, "start_char", f"hit {position} source span"),
                    end_char=_require_int(item, "end_char", f"hit {position} source span"),
                )
            )
        except ValidationError as error:
            raise SearchBackendError(
                f"UnexpectedPayload: hit {position} source span does not satisfy the search "
                f"contract: {_flatten(error)}",
                operation="search",
            ) from error
    return tuple(sorted(spans, key=lambda span: span.source_order))


def _flatten(error: ValidationError) -> str:
    return "; ".join(str(item["msg"]) for item in error.errors())
