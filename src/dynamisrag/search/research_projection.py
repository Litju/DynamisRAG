"""Dense retrieval over the research workspace's *growing*, versioned Qdrant projection.

Unlike the fixed RES-138 receipt service, the research corpus changes when the workspace ingests
new open-access papers. Each change publishes a new content-addressed collection and moves one
alias. This service follows that alias: whenever it targets a collection other than the one in
memory, the projection is rebuilt from its retained embedding manifest plus fresh PostgreSQL and
re-verified point by point before any query uses it. A query never mixes two projections.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import Engine

from dynamisrag.corpus.research import load_published_projection
from dynamisrag.embedding import (
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    TeiEmbeddingProvider,
)
from dynamisrag.embedding.errors import EmbeddingProviderError, TeiIdentityError
from dynamisrag.embedding.identity import EmbeddingModelIdentity
from dynamisrag.embedding.scientific_cpu import (
    MODEL_ID,
    MODEL_REVISION,
    QUERY_CONFIG,
    QUERY_FINGERPRINT,
    CpuDeployment,
)
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.qdrant import QdrantClient, QdrantError, QdrantRetrieval
from dynamisrag.search.scientific import (
    ScientificReadiness,
    ScientificResponse,
    ScientificRetrievalService,
    ScientificUnavailable,
    require_cpu_identity,
)

__all__ = ["ResearchCorpusRetrieval"]


class ResearchCorpusRetrieval(ScientificRetrievalService):
    """Hybrid BM25 + dense retrieval pinned, per request, to the alias's current collection."""

    def __init__(
        self,
        engine: Engine,
        *,
        qdrant: QdrantClient,
        tei_url: str,
        alias: str,
        store_root: Path,
        chunker_revision: str,
    ) -> None:
        # Deliberately not calling the receipt-bound parent constructor.
        self.engine = engine
        self.directory = store_root
        self.client = qdrant
        self.alias = alias
        self.chunker_revision = chunker_revision
        self.provider = TeiEmbeddingProvider(
            base_url=tei_url.rstrip("/"),
            expected_model=ExpectedTeiModel(MODEL_ID, MODEL_REVISION),
            deployment_semantics=CpuDeployment(),
            generation_config=QUERY_CONFIG,
            runtime_config=EmbeddingRuntimeConfig(
                batch_size=1, timeout_seconds=120, retry=EmbeddingRetryPolicy(max_attempts=1)
            ),
            verify_tls=False,
        )
        self.adapter = None
        self.projection = None
        self.unavailable_code = "corpus_empty"
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Load the alias's projection, or stay empty (``corpus_empty``) when none is published."""
        with self._lock:
            self._load()

    def _load(self) -> None:
        try:
            name = self.client.aliases().get(self.alias)
        except QdrantError:
            self.adapter, self.projection, self.unavailable_code = None, None, "qdrant_unavailable"
            raise ScientificUnavailable("qdrant_unavailable") from None
        if name is None:
            self.adapter, self.projection, self.unavailable_code = None, None, "corpus_empty"
            return
        if self.projection is not None and self.projection.collection_name == name:
            return
        try:
            projection = load_published_projection(
                self.engine, self.directory, name, chunker_revision=self.chunker_revision
            )
        except (OSError, ValueError, QdrantError):
            self.adapter, self.projection = None, None
            self.unavailable_code = "canonical_inconsistent"
            raise ScientificUnavailable("canonical_inconsistent") from None
        try:
            require_cpu_identity(self.provider.describe())
        except TeiIdentityError:
            raise ScientificUnavailable("embedding_identity_mismatch") from None
        except EmbeddingProviderError:
            raise ScientificUnavailable("embedding_unavailable") from None
        try:
            adapter = QdrantRetrieval(
                self.client,
                alias=self.alias,
                projection=projection,
                query_identity=EmbeddingModelIdentity(MODEL_ID, MODEL_REVISION, QUERY_FINGERPRINT),
            )
            adapter.resolve()
        except (QdrantError, ValueError, ValidationError):
            raise ScientificUnavailable("qdrant_unavailable") from None
        self.projection, self.adapter, self.unavailable_code = projection, adapter, "ready"
        logging.getLogger(APP_LOGGER_NAME).info(
            "research projection loaded: collection=%s passages=%d",
            name,
            projection.passages.document_count,
        )

    def refresh(self) -> None:
        """Follow the alias after an acquisition (or another worker's) publication."""
        with self._lock:
            self._load()

    def readiness(self) -> ScientificReadiness:
        self.refresh()
        return super().readiness()

    def retrieve(self, question: str, limit: int, filters: Mapping[str, str]) -> ScientificResponse:
        self.refresh()
        return super().retrieve(question, limit, filters)
