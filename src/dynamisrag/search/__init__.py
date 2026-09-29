"""OpenSearch access layer: projection, BM25 retrieval and connectivity (RES-135).

    canonical PostgreSQL passages
        -> PassageProjector (versioned, disposable physical index)
        -> stable alias
        -> Bm25SearchService (versioned BM25 over title/section/passages)

PostgreSQL stays the only authority. Everything in this package is derived from
canonical state and can be deleted and rebuilt from it; no canonical fact is
ever read from or written to OpenSearch.
"""

from __future__ import annotations

from dynamisrag.search.bm25 import (
    BM25_FIELDS,
    BM25_MATCH_TYPE,
    BM25_OPERATOR,
    BM25_QUERY_REVISION,
    BM25_TIE_BREAKER,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MIN_LIMIT,
    Bm25SearchService,
    SearchHit,
    SearchResponse,
    SearchSourceSpan,
    build_bm25_request,
)
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import (
    OpenSearchBulkError,
    OpenSearchError,
    OpenSearchTransportError,
    OpenSearchUnexpectedResponse,
    ProjectionConflictError,
    ProjectionError,
    SearchBackendError,
)
from dynamisrag.search.opensearch import OPENSEARCH_DEPENDENCY_NAME, OpenSearchProbe
from dynamisrag.search.projection import (
    PassageProjectionDocument,
    PassageProjectionManifest,
    PassageProjector,
    ProjectionResult,
    build_projection_manifest,
)
from dynamisrag.search.schema import (
    BM25_SIMILARITY_NAME,
    BM25_SIMILARITY_PARAMS,
    BM25_SIMILARITY_REVISION,
    PASSAGE_INDEX_SCHEMA_REVISION,
    index_mappings,
    index_settings,
    physical_index_name,
)

__all__ = [
    "BM25_FIELDS",
    "BM25_MATCH_TYPE",
    "BM25_OPERATOR",
    "BM25_QUERY_REVISION",
    "BM25_SIMILARITY_NAME",
    "BM25_SIMILARITY_PARAMS",
    "BM25_SIMILARITY_REVISION",
    "BM25_TIE_BREAKER",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MIN_LIMIT",
    "OPENSEARCH_DEPENDENCY_NAME",
    "PASSAGE_INDEX_SCHEMA_REVISION",
    "Bm25SearchService",
    "OpenSearchBulkError",
    "OpenSearchClient",
    "OpenSearchError",
    "OpenSearchProbe",
    "OpenSearchTransportError",
    "OpenSearchUnexpectedResponse",
    "PassageProjectionDocument",
    "PassageProjectionManifest",
    "PassageProjector",
    "ProjectionConflictError",
    "ProjectionError",
    "ProjectionResult",
    "SearchBackendError",
    "SearchHit",
    "SearchResponse",
    "SearchSourceSpan",
    "build_bm25_request",
    "build_projection_manifest",
    "index_mappings",
    "index_settings",
    "physical_index_name",
]
