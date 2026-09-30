"""The BM25 passage search endpoint.

The router is built by a factory so each application instance closes over its
own :class:`~dynamisrag.search.bm25.Bm25SearchService`; there is no
module-level mutable state, so several differently-configured apps can coexist
in one process.

Error mapping is deliberately blunt and safe:

* an invalid query or limit is a client error and returns ``422``;
* an unreachable or untrustworthy search backend returns ``503`` with a fixed
  public message, while only structured safe values go to the log.

The public body of a ``503`` says only that search is unavailable, so no index
name, revision or query fragment is disclosed. The log line is assembled from
the exception class and
:meth:`~dynamisrag.search.errors.OpenSearchError.safe_summary` — the exception
category, the operation, the HTTP status and OpenSearch's ``error.type`` — and
never from ``error.detail``, so an OpenSearch ``error.reason`` cannot be
relayed into log aggregation even if a future caller puts one there.
"""

from __future__ import annotations

import logging
from typing import Annotated, Final

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import StringConstraints

from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.bm25 import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    MIN_LIMIT,
    Bm25SearchService,
    SearchResponse,
)
from dynamisrag.search.errors import OpenSearchError

__all__ = [
    "NO_STORE",
    "SEARCH_PATH",
    "SEARCH_UNAVAILABLE_DETAIL",
    "SERVICE_UNAVAILABLE",
    "UNPROCESSABLE",
    "build_search_router",
]

SEARCH_PATH: Final[str] = "/search"

SEARCH_UNAVAILABLE_DETAIL: Final[str] = "search is temporarily unavailable"
"""The only thing a caller learns when the search backend cannot answer."""

UNPROCESSABLE: Final[int] = 422
"""HTTP 422: the query or the limit is not acceptable."""

SERVICE_UNAVAILABLE: Final[int] = 503
"""HTTP 503: the search backend cannot serve the request."""

NO_STORE: Final[str] = "no-store"
"""A ranked result set changes with every projection: caching it would serve a
stale projection digest and stale provenance. Applied to search responses,
successful and failed alike."""

_NO_STORE_HEADERS: Final[dict[str, str]] = {"Cache-Control": NO_STORE}
"""The same header, for the error path.

A ``503`` is a moving target — it disappears the moment the backend answers —
so a proxy or a browser must not retain it either. Carried on the raised
``HTTPException`` rather than through an application-wide handler, so the rule
stays local to the endpoint that needs it.
"""

_logger: Final[logging.Logger] = logging.getLogger(APP_LOGGER_NAME)


def build_search_router(*, search: Bm25SearchService) -> APIRouter:
    """Create a router bound to one application's search service."""
    router = APIRouter(tags=["search"])

    @router.get(
        SEARCH_PATH,
        response_model=SearchResponse,
        summary="Search passages with BM25",
        description=(
            "Ranks the versioned OpenSearch passage projection with an explicit, "
            "versioned BM25 query. Results are ordered by descending score with the "
            "passage key as a stable tie-break, and every hit carries the document, "
            "section and exact source-span provenance needed to audit it. No "
            "PostgreSQL is read: the projection is a rebuildable cache of canonical "
            "state."
        ),
        responses={
            UNPROCESSABLE: {"description": "Invalid query or limit."},
            SERVICE_UNAVAILABLE: {"description": "Search backend unavailable."},
        },
    )
    def passage_search(
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
            Query(ge=MIN_LIMIT, le=MAX_LIMIT, description="Maximum number of hits."),
        ] = DEFAULT_LIMIT,
    ) -> SearchResponse:
        response.headers["Cache-Control"] = NO_STORE
        try:
            return search.search(q, limit=limit)
        except ValueError as error:
            _logger.info("search request rejected: %s", error)
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=str(error), headers=dict(_NO_STORE_HEADERS)
            ) from error
        except OpenSearchError as error:
            # Structured safe values only: the exception class and the fields
            # safe_summary() builds from. Never str(error) / error.detail.
            _logger.warning(
                "search backend unavailable: %s (%s)",
                type(error).__name__,
                error.safe_summary(),
            )
            raise HTTPException(
                status_code=SERVICE_UNAVAILABLE,
                detail=SEARCH_UNAVAILABLE_DETAIL,
                headers=dict(_NO_STORE_HEADERS),
            ) from error

    return router
