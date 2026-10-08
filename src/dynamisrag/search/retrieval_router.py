"""HTTP surface for the additive BM25+dense hybrid retrieval service."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, Final

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import StringConstraints

from dynamisrag.embedding.errors import EmbeddingProviderError
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.bm25 import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    MIN_LIMIT,
)
from dynamisrag.search.errors import OpenSearchError
from dynamisrag.search.retrieval import HybridRetrievalResponse, HybridRetrievalService
from dynamisrag.search.router import NO_STORE, SERVICE_UNAVAILABLE, UNPROCESSABLE

__all__ = [
    "DENSE_RETRIEVAL_UNAVAILABLE_DETAIL",
    "RETRIEVE_PATH",
    "build_retrieval_router",
]

RETRIEVE_PATH: Final[str] = "/retrieve"
DENSE_RETRIEVAL_UNAVAILABLE_DETAIL: Final[str] = "dense retrieval is temporarily unavailable"
_NO_STORE_HEADERS: Final[dict[str, str]] = {"Cache-Control": NO_STORE}
_logger: Final[logging.Logger] = logging.getLogger(APP_LOGGER_NAME)


class _NoStoreRoute(APIRoute):
    """Keep FastAPI's normal 422 body while disabling storage on validation errors."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                response = await original(request)
            except RequestValidationError as error:
                response = await request_validation_exception_handler(request, error)
            response.headers["Cache-Control"] = NO_STORE
            return response

        return handler


def build_retrieval_router(*, retrieval: HybridRetrievalService) -> APIRouter:
    """Create the new hybrid route bound to one application's retrieval service."""
    router = APIRouter(tags=["search"], route_class=_NoStoreRoute)

    @router.get(
        RETRIEVE_PATH,
        response_model=HybridRetrievalResponse,
        summary="Retrieve passages with BM25, dense ANN and RRF",
        description=(
            "Runs the bm25-v1 and dense-knn-v1 candidate lanes over one resolved "
            "passage-index-v2 physical index, then fuses their ranks with rrf-v1. "
            "Lexical, dense and fused traces retain independent scores, ranks and "
            "passage source provenance. The Qwen 512 profile is provisional; "
            "formal RES-138 Stage-B qualification is deferred."
        ),
        responses={
            UNPROCESSABLE: {"description": "Invalid query or limit."},
            SERVICE_UNAVAILABLE: {"description": "OpenSearch or dense provider unavailable."},
        },
    )
    def hybrid_retrieve(
        response: Response,
        q: Annotated[
            str,
            Query(
                min_length=1,
                max_length=MAX_QUERY_LENGTH,
                description="Non-whitespace search text.",
            ),
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
        limit: Annotated[
            int,
            Query(ge=MIN_LIMIT, le=MAX_LIMIT, description="Maximum number of fused hits."),
        ] = DEFAULT_LIMIT,
    ) -> HybridRetrievalResponse:
        response.headers["Cache-Control"] = NO_STORE
        try:
            return retrieval.retrieve(q, limit=limit)
        except ValueError as error:
            _logger.info("retrieval request rejected: %s", error)
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=str(error), headers=dict(_NO_STORE_HEADERS)
            ) from error
        except OpenSearchError as error:
            _logger.warning(
                "retrieval backend unavailable: %s (%s)",
                type(error).__name__,
                error.safe_summary(),
            )
            raise HTTPException(
                status_code=SERVICE_UNAVAILABLE,
                detail="search is temporarily unavailable",
                headers=dict(_NO_STORE_HEADERS),
            ) from error
        except EmbeddingProviderError as error:
            _logger.warning(
                "retrieval embedding unavailable: %s (%s)",
                type(error).__name__,
                error.safe_summary(),
            )
            raise HTTPException(
                status_code=SERVICE_UNAVAILABLE,
                detail=DENSE_RETRIEVAL_UNAVAILABLE_DETAIL,
                headers=dict(_NO_STORE_HEADERS),
            ) from error

    return router
