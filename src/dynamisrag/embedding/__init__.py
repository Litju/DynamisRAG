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
)

__all__ = [
    "MAX_SAFE_DETAIL_LENGTH",
    "MIN_EMBEDDING_DIMENSION",
    "PASSAGE_EMBEDDING_MANIFEST_REVISION",
    "TRUNCATION_DIRECTIONS",
    "EmbeddingContractError",
    "EmbeddingGenerationConfig",
    "EmbeddingInput",
    "EmbeddingManifestError",
    "EmbeddingModelIdentity",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingProviderIdentity",
    "EmbeddingResponseError",
    "EmbeddingRetryPolicy",
    "EmbeddingRuntimeConfig",
    "PassageEmbeddingEntry",
    "PassageEmbeddingManifest",
    "TeiIdentityError",
    "TeiTransportError",
    "TeiUnexpectedResponse",
    "TruncationDirection",
    "build_passage_embedding_manifest",
    "canonical_embedding_inputs",
    "canonical_json",
]
