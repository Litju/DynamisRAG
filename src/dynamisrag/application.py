"""FastAPI application factory.

``create_app`` is the single entry point used by the local runtime
(``python -m dynamisrag``), by the ASGI server factory
(``uvicorn --factory dynamisrag.application:create_app``) and by the test
suite. It performs no I/O: the SQLAlchemy engine connects lazily and the
OpenSearch client opens no connection until the first probe, so creating an
application never depends on infrastructure being up.
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
from dynamisrag.health.router import LIVENESS_PATH, READINESS_PATH, build_health_router
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.opensearch import OpenSearchProbe

__all__ = ["API_DESCRIPTION", "API_TITLE", "create_app"]

API_TITLE: Final[str] = "DynamisRAG"
API_DESCRIPTION: Final[str] = (
    "Evaluation-first RAG platform for auditable retrieval and evidence-grounded AI. "
    "This build is the RES-130 foundation: configuration, health surface and "
    "infrastructure connectivity. Retrieval, ranking, embeddings and generation are "
    "not part of this slice."
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
    opensearch = OpenSearchProbe(resolved)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        _logger.info("starting: version=%s environment=%s", __version__, resolved.environment.value)
        try:
            yield
        finally:
            opensearch.close()
            engine.dispose()
            _logger.info("stopped: version=%s", __version__)

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(build_health_router(settings=resolved, engine=engine, opensearch=opensearch))
    _logger.debug("application created: health=%s,%s", LIVENESS_PATH, READINESS_PATH)
    return app
