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
from contextlib import ExitStack, asynccontextmanager
from typing import Final

from fastapi import FastAPI

from dynamisrag import __version__
from dynamisrag.agent.auth import IdentityVerifier
from dynamisrag.agent.observability import configure_privacy_safe_langsmith
from dynamisrag.agent.router import build_agent_v2_router
from dynamisrag.agent.services import build_agent_services
from dynamisrag.config import Settings, load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.embedding.errors import EmbeddingProviderError
from dynamisrag.generation.ollama import OllamaClient
from dynamisrag.generation.router import build_answer_router
from dynamisrag.generation.scientific import ScientificAnswerService
from dynamisrag.health.router import LIVENESS_PATH, READINESS_PATH, build_health_router
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.research_ui import build_research_ui_router
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
from dynamisrag.search.scientific import ScientificRetrievalService
from dynamisrag.search.scientific_router import build_scientific_router

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


def _build_identity_verifier(resolved: Settings) -> IdentityVerifier | None:
    if not resolved.agent_enabled or resolved.agent_identity_secret is None:
        return None
    secrets = [resolved.agent_identity_secret.get_secret_value()]
    if resolved.agent_identity_previous_secret is not None:
        secrets.append(resolved.agent_identity_previous_secret.get_secret_value())
    return IdentityVerifier(
        secrets=tuple(secrets),
        issuer=resolved.agent_identity_issuer,
        audience=resolved.agent_identity_audience,
    )


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
    scientific = (
        ScientificRetrievalService(engine, resolved) if resolved.scientific_qdrant_enabled else None
    )
    answers = (
        ScientificAnswerService(scientific, OllamaClient(str(resolved.scientific_ollama_url)))
        if resolved.scientific_answers_enabled and scientific is not None
        else None
    )
    configure_privacy_safe_langsmith(
        tracing_enabled=resolved.agent_langsmith_tracing_enabled,
        project_name=resolved.agent_langsmith_project,
    )
    agent = build_agent_services(
        settings=resolved,
        engine=engine,
        scientific=scientific,
        search=search,
        retrieval=retrieval,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        _logger.info("starting: version=%s environment=%s", __version__, resolved.environment.value)
        with ExitStack() as cleanup:
            cleanup.callback(engine.dispose)
            cleanup.callback(opensearch_client.close)
            if agent is not None:
                cleanup.callback(agent.close)
                agent.start_workers()
            if tei_provider is not None:
                cleanup.callback(tei_provider.close)
            if scientific is not None:
                cleanup.callback(scientific.close)
            if answers is not None:
                cleanup.callback(answers.close)
            if scientific is not None:
                scientific.start()
            if answers is not None:
                answers.start()
            yield
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
    app.include_router(build_scientific_router(scientific))
    app.include_router(build_answer_router(answers))
    app.include_router(
        build_agent_v2_router(
            orchestrator=agent.orchestrator if agent else None,
            store=agent.store if agent else None,
            tool_registry=agent.registry if agent else None,
            on_run_queued=agent.pool.wake if agent and agent.pool else None,
            identity_verifier=_build_identity_verifier(resolved),
            allow_dev_header=resolved.agent_dev_identity_header_enabled,
            rate_limit_per_minute=resolved.agent_rate_limit_per_minute,
        )
    )
    app.include_router(build_research_ui_router())

    _logger.debug(
        "application created: health=%s,%s search=%s retrieve=%s",
        LIVENESS_PATH,
        READINESS_PATH,
        SEARCH_PATH,
        RETRIEVE_PATH,
    )
    return app
