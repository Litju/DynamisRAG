"""Bounded, opt-in Qdrant HTTP API; errors contain no user or article content."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.requests import ClientDisconnect
from starlette.types import Message

from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.scientific import (
    ScientificResponse,
    ScientificRetrievalService,
    ScientificUnavailable,
)

PATH = "/retrieve/qdrant"
MAX_BODY_BYTES = 8192
_logger = logging.getLogger(APP_LOGGER_NAME)
_HEADERS = {"Cache-Control": "no-store"}
FilterKey = Literal[
    "passage_key",
    "document_canonical_key",
    "document_version_key",
    "language",
    "source_system",
    "source_external_id",
    "chunker_revision",
]
FilterValue = Annotated[
    str, StringConstraints(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$")
]


class ScientificRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=512,
            pattern=r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$",
        ),
    ]
    limit: int = Field(default=5, ge=1, le=20)
    filters: dict[FilterKey, FilterValue] = Field(
        default_factory=dict[FilterKey, str], max_length=7
    )


def error_response(code: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"code": code}}, headers=_HEADERS)


class _BoundedRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            body = bytearray()
            try:
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        return error_response("request_too_large", 413)
                    body.extend(chunk)
            except ClientDisconnect:
                return error_response("invalid_request", 400)

            async def receive() -> Message:
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            try:
                response = await original(Request(request.scope, receive))
            except RequestValidationError:
                return error_response("invalid_request", 422)
            response.headers.update(_HEADERS)
            return response

        return handler


def build_scientific_router(service: ScientificRetrievalService | None) -> APIRouter:
    router = APIRouter(tags=["scientific retrieval"], route_class=_BoundedRoute)

    @router.post(
        PATH,
        response_model=ScientificResponse,
        summary="Retrieve verified scientific passages with live Qwen CPU embeddings",
        responses={
            413: {"description": "Body exceeds 8192 bytes"},
            422: {"description": "Invalid request or question exceeds model capacity"},
            503: {"description": "Opt-in service disabled or dependency/identity unavailable"},
        },
    )
    def retrieve(body: ScientificRequest, response: Response) -> ScientificResponse | JSONResponse:
        response.headers.update(_HEADERS)
        if service is None:
            return error_response("qdrant_disabled", 503)
        try:
            return service.retrieve(
                body.question, body.limit, {str(k): v for k, v in body.filters.items()}
            )
        except ScientificUnavailable as error:
            _logger.warning("scientific retrieval unavailable: code=%s", error.code)
            return error_response(error.code, 422 if error.code == "query_rejected" else 503)
        except Exception:
            # No exception prose/tracebacks: dependencies may echo rejected content.
            _logger.error("scientific retrieval failed: code=internal_error")
            return error_response("internal_error", 503)

    return router
