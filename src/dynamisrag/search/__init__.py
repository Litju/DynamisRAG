"""OpenSearch access layer: projection, BM25 retrieval and connectivity (RES-135).

    canonical PostgreSQL passages
        -> PassageProjector (versioned, disposable physical index)
        -> stable alias
        -> Bm25SearchService (versioned BM25 over title/section/passages)

    canonical PostgreSQL passages
        + a caller-supplied Sequence[PassageVector]  (RES-137 produces these)
        -> VectorPassageProjector (passage-index-v2, disposable physical index)
        -> the same stable alias

PostgreSQL stays the only authority. Everything in this package is derived from
canonical state and can be deleted and rebuilt from it; no canonical fact is
ever read from or written to OpenSearch.

The publication protocol itself — build, verify, then move the alias
atomically, never destroying a served index — lives in
:mod:`dynamisrag.search.publication` and is shared by every schema revision.
The dense-vector index contract (RES-136) is declared in
:mod:`dynamisrag.search.vector`, rendered into a ``passage-index-v2`` mapping by
:mod:`dynamisrag.search.schema` and projected in
:mod:`dynamisrag.search.vector_projection`; RES-139 adds the versioned dense
query contract and rank-only BM25+dense fusion while leaving model qualification
explicitly provisional.
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
    VectorContractError,
)
from dynamisrag.search.opensearch import OPENSEARCH_DEPENDENCY_NAME, OpenSearchProbe
from dynamisrag.search.projection import (
    PassageProjectionDocument,
    PassageProjectionManifest,
    PassageProjector,
    ProjectionResult,
    build_projection_manifest,
)
from dynamisrag.search.publication import (
    FailClosedAliasPublisher,
    PublicationPlan,
    PublicationResult,
)
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
    DOCUMENT_GENERATION_CONFIG,
    FUSION_REVISION,
    HYBRID_RETRIEVAL_REVISION,
    PROVISIONAL_EMBEDDING_PROFILE,
    RRF_K,
    DenseCandidate,
    DenseSearchResponse,
    FusedHit,
    HybridRetrievalResponse,
    HybridRetrievalService,
    QueryEmbeddingService,
    RrfLaneTrace,
    RrfTrace,
    build_dense_knn_request,
    reciprocal_rank_fusion,
)
from dynamisrag.search.schema import (
    BM25_SIMILARITY_NAME,
    BM25_SIMILARITY_PARAMS,
    BM25_SIMILARITY_REVISION,
    INDEX_KNN_ENABLED,
    PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    index_mappings,
    index_settings,
    physical_index_name,
    physical_vector_index_name,
    vector_index_mappings,
    vector_index_meta,
    vector_index_settings,
)
from dynamisrag.search.vector import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    SUPPORTED_VECTOR_SPACES,
    VECTOR_ENGINE,
    VECTOR_FIELD,
    VECTOR_INDEX_METHOD,
    VECTOR_INDEX_TYPE,
    VECTOR_SPACE_COSINESIMIL,
    VECTOR_SPACE_INNER_PRODUCT,
    VECTOR_SPACE_L2,
    EmbeddingModelIdentity,
    VectorIndexConfig,
    validate_vector_set,
)
from dynamisrag.search.vector_projection import (
    PassageVector,
    VectorPassageProjectionDocument,
    VectorPassageProjectionManifest,
    VectorPassageProjector,
    VectorProjectionResult,
    build_vector_projection_manifest,
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
    "CANDIDATE_WINDOW",
    "DEFAULT_LIMIT",
    "DENSE_DIMENSION",
    "DENSE_MODEL_ID",
    "DENSE_MODEL_REVISION",
    "DENSE_QUERY_REVISION",
    "DENSE_SCHEMA_REVISION",
    "DENSE_SOURCE_FIELDS",
    "DENSE_SPACE",
    "DENSE_TIE_ORDER",
    "DOCUMENT_GENERATION_CONFIG",
    "FUSION_REVISION",
    "HNSW_EF_CONSTRUCTION",
    "HNSW_M",
    "HYBRID_RETRIEVAL_REVISION",
    "INDEX_KNN_ENABLED",
    "MAX_LIMIT",
    "MIN_LIMIT",
    "OPENSEARCH_DEPENDENCY_NAME",
    "PASSAGE_INDEX_SCHEMA_REVISION",
    "PROVISIONAL_EMBEDDING_PROFILE",
    "RRF_K",
    "SUPPORTED_VECTOR_SPACES",
    "VECTOR_ENGINE",
    "VECTOR_FIELD",
    "VECTOR_INDEX_METHOD",
    "VECTOR_INDEX_TYPE",
    "VECTOR_PASSAGE_INDEX_SCHEMA_REVISION",
    "VECTOR_SPACE_COSINESIMIL",
    "VECTOR_SPACE_INNER_PRODUCT",
    "VECTOR_SPACE_L2",
    "Bm25SearchService",
    "DenseCandidate",
    "DenseSearchResponse",
    "EmbeddingModelIdentity",
    "FailClosedAliasPublisher",
    "FusedHit",
    "HybridRetrievalResponse",
    "HybridRetrievalService",
    "OpenSearchBulkError",
    "OpenSearchClient",
    "OpenSearchError",
    "OpenSearchProbe",
    "OpenSearchTransportError",
    "OpenSearchUnexpectedResponse",
    "PassageProjectionDocument",
    "PassageProjectionManifest",
    "PassageProjector",
    "PassageVector",
    "ProjectionConflictError",
    "ProjectionError",
    "ProjectionResult",
    "PublicationPlan",
    "PublicationResult",
    "QueryEmbeddingService",
    "RrfLaneTrace",
    "RrfTrace",
    "SearchBackendError",
    "SearchHit",
    "SearchResponse",
    "SearchSourceSpan",
    "VectorContractError",
    "VectorIndexConfig",
    "VectorPassageProjectionDocument",
    "VectorPassageProjectionManifest",
    "VectorPassageProjector",
    "VectorProjectionResult",
    "build_bm25_request",
    "build_dense_knn_request",
    "build_projection_manifest",
    "build_vector_projection_manifest",
    "index_mappings",
    "index_settings",
    "physical_index_name",
    "physical_vector_index_name",
    "reciprocal_rank_fusion",
    "validate_vector_set",
    "vector_index_mappings",
    "vector_index_meta",
    "vector_index_settings",
]
