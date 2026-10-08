"""Versioned dense ANN retrieval and deterministic BM25+dense RRF fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisrag.config import Settings
from dynamisrag.embedding.contracts import (
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingProvider,
    TruncationDirection,
    passage_content_sha256,
)
from dynamisrag.embedding.errors import EmbeddingProviderError
from dynamisrag.embedding.identity import require_sha256_hex
from dynamisrag.embedding.tei import TeiEmbeddingProvider, tei_provider_from_settings
from dynamisrag.search.bm25 import (
    SOURCE_FIELDS,
    Bm25SearchService,
    SearchHit,
    SearchResponse,
    SearchSourceSpan,
    parse_search_hit,
    validate_limit,
    validate_query,
)
from dynamisrag.search.client import JsonValue, OpenSearchClient
from dynamisrag.search.errors import SearchBackendError
from dynamisrag.search.schema import (
    BM25_SIMILARITY_REVISION,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PROJECTION_META_KEYS,
)
from dynamisrag.search.vector import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_ENGINE,
    VECTOR_FIELD,
    VECTOR_INDEX_METHOD,
    VECTOR_INDEX_TYPE,
    VECTOR_SPACE_COSINESIMIL,
    EmbeddingModelIdentity,
    VectorIndexConfig,
    validate_vector_set,
)
from dynamisrag.search.vector_projection import PassageVector

__all__ = [
    "CANDIDATE_WINDOW",
    "DENSE_DIMENSION",
    "DENSE_MODEL_ID",
    "DENSE_MODEL_REVISION",
    "DENSE_QUERY_REVISION",
    "DENSE_SCHEMA_REVISION",
    "DENSE_SOURCE_FIELDS",
    "DENSE_SPACE",
    "DENSE_TIE_ORDER",
    "FUSION_REVISION",
    "HYBRID_RETRIEVAL_REVISION",
    "PROVISIONAL_EMBEDDING_PROFILE",
    "QUERY_GENERATION_CONFIG",
    "RRF_K",
    "DenseCandidate",
    "DenseSearchResponse",
    "FusedHit",
    "HybridRetrievalResponse",
    "HybridRetrievalService",
    "PassageProvenance",
    "QueryEmbedder",
    "QueryEmbeddingService",
    "QueryEmbeddingUnavailable",
    "RrfLaneTrace",
    "RrfTrace",
    "build_dense_knn_request",
    "create_query_embedding_provider",
    "reciprocal_rank_fusion",
]

DENSE_QUERY_REVISION: Final[str] = "dense-knn-v1"
"""The query semantics for passage-index-v2 Lucene HNSW retrieval."""
DENSE_SCHEMA_REVISION: Final[str] = VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
DENSE_SOURCE_FIELDS: Final[tuple[str, ...]] = SOURCE_FIELDS
DENSE_TIE_ORDER: Final[tuple[str, str]] = ("raw_score descending", "passage_key ascending")

HYBRID_RETRIEVAL_REVISION: Final[str] = "hybrid-rrf-v1"
FUSION_REVISION: Final[str] = "rrf-v1"
RRF_K: Final[int] = 60
CANDIDATE_WINDOW: Final[int] = 50

DENSE_MODEL_ID: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
DENSE_MODEL_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DENSE_DIMENSION: Final[int] = 512
DENSE_SPACE: Final[str] = VECTOR_SPACE_COSINESIMIL
PROVISIONAL_EMBEDDING_PROFILE: Final[str] = "res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1"
"""Provisional engineering default; RES-138 Stage-B qualification is deferred."""

QUERY_GENERATION_CONFIG: Final[EmbeddingGenerationConfig] = EmbeddingGenerationConfig(
    normalize=True,
    truncate=True,
    truncation_direction=TruncationDirection.RIGHT,
    prompt_name="query",
    dimensions=DENSE_DIMENSION,
)
"""Query-side Qwen semantics. Truncation is explicit and matches the RES-138 input policy."""


class QueryEmbeddingUnavailable(EmbeddingProviderError):
    """A provider could not safely produce the provisional query vector."""

    _CATEGORY = "QueryEmbeddingUnavailable"


class QueryEmbedder(Protocol):
    """Narrow query-only seam used by the dense retrieval service and tests."""

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Return one validated query vector, without exposing its surrogate key."""
        ...


class QueryEmbeddingService:
    """Adapt the existing content-addressed EmbeddingProvider for one query."""

    __slots__ = ("_provider",)

    def __init__(self, provider: EmbeddingProvider) -> None:
        if provider.generation_config != QUERY_GENERATION_CONFIG:
            raise QueryEmbeddingUnavailable(
                "the query provider does not use the frozen provisional generation semantics",
                operation="embed_query",
            )
        self._provider: Final[EmbeddingProvider] = provider

    def embed_query(self, query: str) -> tuple[float, ...]:
        normalized = validate_query(query)
        digest = passage_content_sha256(normalized)
        surrogate_key = f"query:{digest}"
        item = EmbeddingInput(
            passage_key=surrogate_key,
            content_sha256=digest,
            text=normalized,
        )
        try:
            before = self._provider.describe()
            self._require_model(before.model_id, before.model_sha)
            vectors = self._provider.embed((item,))
            after = self._provider.describe()
            before.require_same_semantic_runtime(after, operation="embed_query")
            self._require_model(after.model_id, after.model_sha)
        except Exception as error:
            if isinstance(error, QueryEmbeddingUnavailable):
                raise
            # Provider details can contain server prose or the private query key.
            raise QueryEmbeddingUnavailable(
                "the query embedding provider could not complete the request",
                operation="embed_query",
                cause=type(error).__name__,
            ) from error

        if len(vectors) != 1:
            raise QueryEmbeddingUnavailable(
                "the query embedding provider did not return exactly one vector",
                operation="embed_query",
            )
        values = vectors[0]
        try:
            identity = after.embedding_model_identity(QUERY_GENERATION_CONFIG)
            config = VectorIndexConfig(
                dimension=DENSE_DIMENSION,
                space=DENSE_SPACE,
                embedding_model=identity,
            )
            vector = PassageVector(passage_key=surrogate_key, values=values)
            validate_vector_set(
                config=config,
                expected_keys=(surrogate_key,),
                vectors={surrogate_key: vector.values},
            )
        except Exception as error:
            raise QueryEmbeddingUnavailable(
                "the query embedding provider returned a vector that violates the dense contract",
                operation="embed_query",
                cause=type(error).__name__,
            ) from error
        if not math.isclose(math.hypot(*vector.values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise QueryEmbeddingUnavailable(
                "the query embedding provider did not honour required normalization",
                operation="embed_query",
            )
        return vector.values

    @staticmethod
    def _require_model(model_id: str, model_revision: str) -> None:
        if (model_id, model_revision) != (DENSE_MODEL_ID, DENSE_MODEL_REVISION):
            raise QueryEmbeddingUnavailable(
                "the query embedding provider does not serve the provisional model revision",
                operation="embed_query",
            )


def create_query_embedding_provider(settings: Settings) -> TeiEmbeddingProvider | None:
    """Construct the optional TEI query provider only for a complete identity."""
    if (
        settings.tei_url is None
        or settings.tei_expected_model_id is None
        or settings.tei_expected_model_sha is None
    ):
        return None
    return tei_provider_from_settings(settings, generation_config=QUERY_GENERATION_CONFIG)


def build_dense_knn_request(*, vector: Sequence[float]) -> Mapping[str, JsonValue]:
    """Build the frozen ``dense-knn-v1`` request body, with vectors excluded from source."""
    return {
        "size": CANDIDATE_WINDOW,
        "_source": list(DENSE_SOURCE_FIELDS),
        "query": {
            "knn": {
                VECTOR_FIELD: {
                    "vector": list(vector),
                    "k": CANDIDATE_WINDOW,
                }
            }
        },
    }


class PassageProvenance(BaseModel):
    """The existing auditable passage source fields, without a lane score or rank."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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


class DenseCandidate(BaseModel):
    """One ranked ANN result with its raw score and full source provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    raw_score: float = Field(allow_inf_nan=False)
    passage_key: str
    provenance: PassageProvenance

    @model_validator(mode="after")
    def _same_passage(self) -> DenseCandidate:
        if self.passage_key != self.provenance.passage_key:
            raise ValueError("dense candidate and provenance passage keys must agree")
        return self


class DenseSearchResponse(BaseModel):
    """An independently inspectable dense candidate run; no vector is serialized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    query_revision: Literal["dense-knn-v1"]
    physical_index: str
    projection_schema_revision: Literal["passage-index-v2"]
    projection_sha256: str
    chunker_revision: str
    vector_config_sha256: str
    model_id: str
    model_revision: str
    dimension: Literal[512]
    vector_space: Literal["cosinesimil"]
    document_embedding_config_sha256: str
    embedding_profile: Literal["res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1"]
    embedding_profile_status: Literal["provisional"]
    candidate_window: Literal[50]
    took_ms: int = Field(ge=0)
    candidates: tuple[DenseCandidate, ...] = ()


class RrfLaneTrace(BaseModel):
    """One lane's raw rank and its explicit RRF contribution for a fused passage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    present: bool
    rank: int | None = Field(default=None, ge=1)
    raw_score: float | None = Field(default=None, allow_inf_nan=False)
    rrf_contribution: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _presence_matches_values(self) -> RrfLaneTrace:
        has_values = self.rank is not None and self.raw_score is not None
        if self.present != has_values or (self.present != (self.rrf_contribution is not None)):
            raise ValueError("RRF lane presence must agree with its rank, score and contribution")
        return self


class FusedHit(BaseModel):
    """One deterministic fused result with independently auditable lane evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_key: str
    final_rank: int = Field(ge=1)
    rrf_score: float = Field(ge=0.0, allow_inf_nan=False)
    lexical: RrfLaneTrace
    dense: RrfLaneTrace
    provenance: PassageProvenance

    @model_validator(mode="after")
    def _same_passage(self) -> FusedHit:
        if self.passage_key != self.provenance.passage_key:
            raise ValueError("fused hit and provenance passage keys must agree")
        if not self.lexical.present and not self.dense.present:
            raise ValueError("a fused hit must occur in at least one candidate lane")
        return self


class RrfTrace(BaseModel):
    """Fusion parameters and the final, limited ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: Literal["rrf-v1"]
    k: Literal[60]
    candidate_window: Literal[50]
    hits: tuple[FusedHit, ...] = ()


class HybridRetrievalResponse(BaseModel):
    """One request's lexical, dense and fused views over one physical projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    retrieval_revision: Literal["hybrid-rrf-v1"]
    physical_index: str
    embedding_profile: Literal["res138-stage-a-provisional-qwen3-embedding-0.6b-512-v1"]
    embedding_profile_status: Literal["provisional"]
    lexical: SearchResponse
    dense: DenseSearchResponse
    fusion: RrfTrace


def _provenance(hit: SearchHit) -> PassageProvenance:
    return PassageProvenance.model_validate(hit.model_dump(exclude={"rank", "score"}))


def reciprocal_rank_fusion(
    *,
    lexical: Sequence[SearchHit],
    dense: Sequence[DenseCandidate],
    limit: int,
) -> RrfTrace:
    """Pure, rank-only RRF: score magnitudes never affect fusion or tie order."""
    validate_limit(limit)
    lexical_window = sorted(lexical, key=lambda hit: hit.rank)[:CANDIDATE_WINDOW]
    dense_window = sorted(dense, key=lambda hit: hit.rank)[:CANDIDATE_WINDOW]
    if any(not math.isfinite(hit.score) for hit in lexical_window):
        raise SearchBackendError("lexical candidate has a non-finite raw score", operation="rrf")
    lexical_by_key = {hit.passage_key: hit for hit in lexical_window}
    dense_by_key = {hit.passage_key: hit for hit in dense_window}
    if len(lexical_by_key) != len(lexical_window) or len(dense_by_key) != len(dense_window):
        raise SearchBackendError("duplicate passage keys in a candidate lane", operation="rrf")

    candidates: list[
        tuple[float, str, PassageProvenance, SearchHit | None, DenseCandidate | None]
    ] = []
    for key in sorted(lexical_by_key.keys() | dense_by_key.keys()):
        lexical_hit = lexical_by_key.get(key)
        dense_hit = dense_by_key.get(key)
        if lexical_hit is not None and dense_hit is not None:
            lexical_source = _provenance(lexical_hit)
            if lexical_source != dense_hit.provenance:
                raise SearchBackendError(
                    "lexical and dense candidates disagree on passage source provenance",
                    operation="rrf",
                )
            source = lexical_source
        elif lexical_hit is not None:
            source = _provenance(lexical_hit)
        elif dense_hit is not None:
            source = dense_hit.provenance
        else:
            raise SearchBackendError("empty candidate in RRF union", operation="rrf")
        score = (1.0 / (RRF_K + lexical_hit.rank) if lexical_hit is not None else 0.0) + (
            1.0 / (RRF_K + dense_hit.rank) if dense_hit is not None else 0.0
        )
        candidates.append((score, key, source, lexical_hit, dense_hit))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    fused = tuple(
        FusedHit(
            passage_key=key,
            final_rank=rank,
            rrf_score=score,
            lexical=(
                RrfLaneTrace(
                    present=True,
                    rank=lexical_hit.rank,
                    raw_score=lexical_hit.score,
                    rrf_contribution=1.0 / (RRF_K + lexical_hit.rank),
                )
                if lexical_hit is not None
                else RrfLaneTrace(present=False)
            ),
            dense=(
                RrfLaneTrace(
                    present=True,
                    rank=dense_hit.rank,
                    raw_score=dense_hit.raw_score,
                    rrf_contribution=1.0 / (RRF_K + dense_hit.rank),
                )
                if dense_hit is not None
                else RrfLaneTrace(present=False)
            ),
            provenance=source,
        )
        for rank, (score, key, source, lexical_hit, dense_hit) in enumerate(
            candidates[:limit], start=1
        )
    )
    return RrfTrace(
        revision=FUSION_REVISION,
        k=RRF_K,
        candidate_window=CANDIDATE_WINDOW,
        hits=fused,
    )


class _DenseIndexIdentity(BaseModel):
    """Validated v2 index metadata captured before either retrieval lane starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    physical_index: str
    projection_schema_revision: Literal["passage-index-v2"]
    projection_sha256: str
    chunker_revision: str
    vector_config_sha256: str
    model_id: Literal["Qwen/Qwen3-Embedding-0.6B"]
    model_revision: Literal["97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"]
    dimension: Literal[512]
    vector_space: Literal["cosinesimil"]
    document_embedding_config_sha256: str


def _dense_index_identity(
    physical_index: str, meta: Mapping[str, JsonValue]
) -> _DenseIndexIdentity:
    if set(meta) != set(VECTOR_PROJECTION_META_KEYS):
        raise SearchBackendError(
            "v2 index metadata does not satisfy its closed contract", operation="retrieve"
        )
    if (
        meta.get("schema_revision") != DENSE_SCHEMA_REVISION
        or meta.get("bm25_similarity_revision") != BM25_SIMILARITY_REVISION
        or meta.get("vector_engine") != VECTOR_ENGINE
        or meta.get("vector_method") != VECTOR_INDEX_METHOD
        or meta.get("vector_data_type") != VECTOR_INDEX_TYPE
        or meta.get("vector_space_type") != DENSE_SPACE
        or meta.get("hnsw_m") != HNSW_M
        or meta.get("hnsw_ef_construction") != HNSW_EF_CONSTRUCTION
    ):
        raise SearchBackendError(
            "v2 index metadata does not match the dense query contract", operation="retrieve"
        )
    dimension = meta.get("vector_dimension")
    model_id = meta.get("embedding_model_id")
    model_revision = meta.get("embedding_model_revision")
    document_config_sha = meta.get("embedding_config_sha256")
    vector_config_sha = meta.get("vector_config_sha256")
    projection_sha = meta.get("projection_sha256")
    chunker_revision = meta.get("chunker_revision")
    if (
        isinstance(dimension, bool)
        or dimension != DENSE_DIMENSION
        or not isinstance(model_id, str)
        or not isinstance(model_revision, str)
        or not isinstance(document_config_sha, str)
        or not isinstance(vector_config_sha, str)
        or not isinstance(projection_sha, str)
        or not isinstance(chunker_revision, str)
        or not chunker_revision
    ):
        raise SearchBackendError(
            "v2 index metadata is incomplete for dense retrieval", operation="retrieve"
        )
    try:
        projection_sha = require_sha256_hex(
            projection_sha, kind="projection digest", operation="retrieve"
        )
        model = EmbeddingModelIdentity(
            model_id=model_id,
            model_revision=model_revision,
            embedding_config_sha256=document_config_sha,
        )
        vector_config = VectorIndexConfig(
            dimension=DENSE_DIMENSION,
            space=DENSE_SPACE,
            embedding_model=model,
        )
    except Exception as error:
        raise SearchBackendError("v2 vector identity is invalid", operation="retrieve") from error
    if (
        model_id != DENSE_MODEL_ID
        or model_revision != DENSE_MODEL_REVISION
        or vector_config_sha != vector_config.config_sha256
    ):
        raise SearchBackendError(
            "v2 vector identity is incompatible with the provisional profile", operation="retrieve"
        )
    return _DenseIndexIdentity(
        physical_index=physical_index,
        projection_schema_revision="passage-index-v2",
        projection_sha256=projection_sha,
        chunker_revision=chunker_revision,
        vector_config_sha256=vector_config_sha,
        model_id=DENSE_MODEL_ID,
        model_revision=DENSE_MODEL_REVISION,
        dimension=512,
        vector_space="cosinesimil",
        document_embedding_config_sha256=document_config_sha,
    )


class _DenseSearchService:
    """Dense lane bound to the application query embedder and shared client."""

    __slots__ = ("_client", "_query_embedder")

    def __init__(self, client: OpenSearchClient, query_embedder: QueryEmbedder | None) -> None:
        self._client: Final[OpenSearchClient] = client
        self._query_embedder: Final[QueryEmbedder | None] = query_embedder

    def search_resolved(
        self,
        query: str,
        *,
        physical_index: str,
        identity: _DenseIndexIdentity,
    ) -> DenseSearchResponse:
        if self._query_embedder is None:
            raise QueryEmbeddingUnavailable(
                "no query embedding provider is configured", operation="embed_query"
            )
        vector = self._query_embedder.embed_query(query)
        if len(vector) != DENSE_DIMENSION or any(
            not _is_finite_component(value) for value in vector
        ):
            raise QueryEmbeddingUnavailable(
                "the query embedder returned a vector outside the frozen dense contract",
                operation="embed_query",
            )
        if not math.isclose(
            math.hypot(*(float(value) for value in vector)), 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise QueryEmbeddingUnavailable(
                "the query embedder did not honour required normalization",
                operation="embed_query",
            )
        payload = self._client.search(physical_index, build_dense_knn_request(vector=vector))
        hits_block = payload.get("hits")
        raw_hits = hits_block.get("hits") if isinstance(hits_block, Mapping) else None
        if not isinstance(raw_hits, list) or len(raw_hits) > CANDIDATE_WINDOW:
            raise SearchBackendError(
                "dense response carries an invalid candidate list", operation="dense_search"
            )

        parsed: list[SearchHit] = []
        for position, raw in enumerate(raw_hits, start=1):
            hit = parse_search_hit(
                position=position,
                raw=raw,
                chunker_revision=identity.chunker_revision,
                index_schema_revision=identity.projection_schema_revision,
                projection_sha256=identity.projection_sha256,
                physical_index=physical_index,
            )
            if not math.isfinite(hit.score):
                raise SearchBackendError(
                    "dense response contains a non-finite score", operation="dense_search"
                )
            parsed.append(hit)
        parsed.sort(key=lambda hit: (-hit.score, hit.passage_key))
        candidates = tuple(
            DenseCandidate(
                rank=rank,
                raw_score=hit.score,
                passage_key=hit.passage_key,
                provenance=_provenance(hit),
            )
            for rank, hit in enumerate(parsed, start=1)
        )
        return DenseSearchResponse(
            query=query,
            query_revision=DENSE_QUERY_REVISION,
            physical_index=physical_index,
            projection_schema_revision=identity.projection_schema_revision,
            projection_sha256=identity.projection_sha256,
            chunker_revision=identity.chunker_revision,
            vector_config_sha256=identity.vector_config_sha256,
            model_id=identity.model_id,
            model_revision=identity.model_revision,
            dimension=identity.dimension,
            vector_space=identity.vector_space,
            document_embedding_config_sha256=identity.document_embedding_config_sha256,
            embedding_profile=PROVISIONAL_EMBEDDING_PROFILE,
            embedding_profile_status="provisional",
            candidate_window=CANDIDATE_WINDOW,
            took_ms=_require_nonnegative_int(payload, "took", "dense_search"),
            candidates=candidates,
        )


class HybridRetrievalService:
    """Resolve one v2 physical target, then run BM25 and dense lanes in parallel."""

    __slots__ = ("_alias", "_bm25", "_client", "_dense", "_query_embedder")

    def __init__(
        self,
        client: OpenSearchClient,
        *,
        alias: str,
        query_embedder: QueryEmbedder | None,
    ) -> None:
        self._client: Final[OpenSearchClient] = client
        self._alias: Final[str] = alias
        self._query_embedder: Final[QueryEmbedder | None] = query_embedder
        self._bm25: Final[Bm25SearchService] = Bm25SearchService(client, alias=alias)
        self._dense: Final[_DenseSearchService] = _DenseSearchService(client, query_embedder)

    def retrieve(self, query: str, *, limit: int = 10) -> HybridRetrievalResponse:
        normalized = validate_query(query)
        validate_limit(limit)
        if self._query_embedder is None:
            raise QueryEmbeddingUnavailable(
                "no query embedding provider is configured", operation="embed_query"
            )
        targets = self._client.alias_targets(self._alias)
        if len(targets) != 1:
            raise SearchBackendError(
                "hybrid retrieval requires the configured alias to resolve to exactly one index",
                operation="resolve_retrieval_index",
            )
        physical_index = targets[0]
        meta = self._client.index_meta(physical_index)
        identity = _dense_index_identity(physical_index, meta)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dynamisrag-retrieve") as pool:
            lexical_future = pool.submit(
                self._bm25.search_resolved,
                normalized,
                limit=CANDIDATE_WINDOW,
                physical_index=physical_index,
                meta=meta,
            )
            dense_future = pool.submit(
                self._dense.search_resolved,
                normalized,
                physical_index=physical_index,
                identity=identity,
            )
            lexical = lexical_future.result()
            dense = dense_future.result()

        fusion = reciprocal_rank_fusion(lexical=lexical.hits, dense=dense.candidates, limit=limit)
        return HybridRetrievalResponse(
            query=normalized,
            retrieval_revision=HYBRID_RETRIEVAL_REVISION,
            physical_index=physical_index,
            embedding_profile=PROVISIONAL_EMBEDDING_PROFILE,
            embedding_profile_status="provisional",
            lexical=lexical,
            dense=dense,
            fusion=fusion,
        )


def _require_nonnegative_int(payload: Mapping[str, JsonValue], key: str, operation: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SearchBackendError(
            f"UnexpectedPayload: {operation} response carries no non-negative integer {key!r}",
            operation=operation,
        )
    return value


def _is_finite_component(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
