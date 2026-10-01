"""The model-agnostic embedding boundary (RES-137).

    canonical passages
        -> EmbeddingInput              (frozen, content-addressed)
        -> EmbeddingProvider           (a port that knows no vendor)
        -> TEI adapter                 (one HTTP protocol, revision tei-http-v1)
        -> exact vectors
        -> PassageEmbeddingManifest    (deterministic, byte-reproducible)
        -> PassageVector / EmbeddingModelIdentity  (the RES-136 boundary)

PostgreSQL stays the only authority for what a passage *is*. Everything here is
derived from it and can be deleted and rebuilt; nothing here reads or writes a
vector store, and no embedding table exists. The dense-vector index contract
(RES-136) is declared in :mod:`dynamisrag.search.vector`, which re-exports
:class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity` from this package.

Deliberately absent, because they belong to later issues: which model is
*best* (RES-138), and how vectors are searched (RES-139). Nothing in this package
ranks models, measures retrieval quality or scores a query.
"""

from __future__ import annotations

from dynamisrag.embedding.contracts import (
    MIN_EMBEDDING_DIMENSION,
    TRUNCATION_DIRECTIONS,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingJsonValue,
    EmbeddingProvider,
    EmbeddingProviderIdentity,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    TruncationDirection,
    canonical_json,
)
from dynamisrag.embedding.errors import (
    MAX_SAFE_DETAIL_LENGTH,
    EmbeddingContractError,
    EmbeddingManifestError,
    EmbeddingProviderError,
    EmbeddingResponseError,
    TeiIdentityError,
    TeiTransportError,
    TeiUnexpectedResponse,
)
from dynamisrag.embedding.identity import EmbeddingModelIdentity
from dynamisrag.embedding.manifest import (
    PASSAGE_EMBEDDING_MANIFEST_REVISION,
    PassageEmbeddingEntry,
    PassageEmbeddingManifest,
    build_passage_embedding_manifest,
    canonical_embedding_inputs,
    embed_passages,
)
from dynamisrag.embedding.tei import (
    NON_RETRYABLE_TEI_STATUS_CODES,
    TEI_EMBED_PATH,
    TEI_EMBEDDING_MODEL_TYPE,
    TEI_HTTP_PROTOCOL_REVISION,
    TEI_INFO_PATH,
    TEI_PROVIDER_NAME,
    TRANSIENT_TEI_STATUS_CODES,
    ExpectedTeiModel,
    TeiEmbeddingProvider,
    TeiServingInfo,
    tei_embed_request_body,
    tei_provider_from_settings,
)

__all__ = [
    "MAX_SAFE_DETAIL_LENGTH",
    "MIN_EMBEDDING_DIMENSION",
    "NON_RETRYABLE_TEI_STATUS_CODES",
    "PASSAGE_EMBEDDING_MANIFEST_REVISION",
    "TEI_EMBEDDING_MODEL_TYPE",
    "TEI_EMBED_PATH",
    "TEI_HTTP_PROTOCOL_REVISION",
    "TEI_INFO_PATH",
    "TEI_PROVIDER_NAME",
    "TRANSIENT_TEI_STATUS_CODES",
    "TRUNCATION_DIRECTIONS",
    "EmbeddingContractError",
    "EmbeddingGenerationConfig",
    "EmbeddingInput",
    "EmbeddingJsonValue",
    "EmbeddingManifestError",
    "EmbeddingModelIdentity",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingProviderIdentity",
    "EmbeddingResponseError",
    "EmbeddingRetryPolicy",
    "EmbeddingRuntimeConfig",
    "ExpectedTeiModel",
    "PassageEmbeddingEntry",
    "PassageEmbeddingManifest",
    "TeiEmbeddingProvider",
    "TeiIdentityError",
    "TeiServingInfo",
    "TeiTransportError",
    "TeiUnexpectedResponse",
    "TruncationDirection",
    "build_passage_embedding_manifest",
    "canonical_embedding_inputs",
    "canonical_json",
    "embed_passages",
    "tei_embed_request_body",
    "tei_provider_from_settings",
]
