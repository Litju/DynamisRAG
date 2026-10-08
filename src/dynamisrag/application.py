"""FastAPI application factory.

``create_app`` is the single entry point used by the local runtime
(``python -m dynamisrag``), by the ASGI server factory
(``uvicorn --factory dynamisrag.application:create_app``) and by the test
suite. It performs no I/O: the SQLAlchemy engine connects lazily and the
OpenSearch client opens no connection until an operation executes, so creating
an application never depends on infrastructure being up.

OpenSearch ownership is deliberately single: one
:class:`~dynamisrag.search.client.OpenSearchClient` is created here, shared by
the readiness probe and the BM25 search service, and closed exactly once when
the lifespan ends. A new client is never constructed per request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI

from dynamisrag import __version__
from dynamisrag.config import Settings, load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.embedding.errors import EmbeddingProviderError
from dynamisrag.health.router import LIVENESS_PATH, READINESS_PATH, build_health_router
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.bm25 import Bm25SearchService
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.opensearch import OpenSearchProbe
from dynamisrag.search.retrieval import (
    HybridRetrievalService,
    QueryEmbeddingService,
    create_query_embedding_provider,
)
from dynamisrag.search.retrieval_router import RETRIEVE_PATH, build_retrieval_router
from dynamisrag.search.router import SEARCH_PATH, build_search_router

__all__ = ["API_DESCRIPTION", "API_TITLE", "create_app"]

API_TITLE: Final[str] = "DynamisRAG"
API_DESCRIPTION: Final[str] = (
    "Evaluation-first RAG platform for auditable retrieval and evidence-grounded AI. "
    "This build projects the canonical PostgreSQL passage model into a versioned, "
    "disposable OpenSearch projection and serves deterministic BM25 retrieval plus "
    "provisional BM25+dense RRF retrieval over it. Dense query embeddings use an "
    "optional configured TEI provider. The projection is a rebuildable cache, "
    "never an authority: no canonical state depends on it."
)

_logger: Final[logging.Logger] = logging.getLogger(APP_LOGGER_NAME)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a ready-to-serve FastAPI application.

    ``settings`` is injectable so tests and alternative entry points can supply
    an explicit, already-validated configuration. When omitted, configuration is
    read once from the environment and ``.env``.
    """
    resolved: Settings = settings if settings is not None else load_settings()
    engine = create_database_engine(resolved)
    opensearch_client = OpenSearchClient(resolved)
    probe = OpenSearchProbe(opensearch_client)
    search = Bm25SearchService(opensearch_client, alias=resolved.opensearch_index_alias)
    tei_provider = None
    query_embedder = None
    try:
        tei_provider = create_query_embedding_provider(resolved)
        query_embedder = QueryEmbeddingService(tei_provider) if tei_provider is not None else None
    except EmbeddingProviderError as error:
        _logger.warning(
            "query embedding provider unavailable: %s (%s)",
            type(error).__name__,
            error.safe_summary(),
        )
        if tei_provider is not None:
            tei_provider.close()
            tei_provider = None
    retrieval = HybridRetrievalService(
        opensearch_client,
        alias=resolved.opensearch_index_alias,
        query_embedder=query_embedder,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        _logger.info("starting: version=%s environment=%s", __version__, resolved.environment.value)
        try:
            yield
        finally:
            try:
                if tei_provider is not None:
                    tei_provider.close()
            finally:
                opensearch_client.close()
                engine.dispose()
            _logger.info("stopped: version=%s", __version__)

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(build_health_router(settings=resolved, engine=engine, opensearch=probe))
    app.include_router(build_search_router(search=search))
    app.include_router(build_retrieval_router(retrieval=retrieval))

    _logger.debug(
        "application created: health=%s,%s search=%s retrieve=%s",
        LIVENESS_PATH,
        READINESS_PATH,
        SEARCH_PATH,
        RETRIEVE_PATH,
    )
    return app
