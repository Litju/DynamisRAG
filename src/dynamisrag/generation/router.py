"""Opt-in local scientific answer endpoint with content-free error logging."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from dynamisrag.domain.values import Sha256Hex
from dynamisrag.generation.ollama import GenerationUnavailableError
from dynamisrag.generation.scientific import (
    AuthorizedPaper,
    ScientificAnswer,
    ScientificAnswerService,
)
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.scientific_router import (
    BoundedScientificRoute,
    ScientificQuestion,
    error_response,
)

PATH = "/answer/scientific"
PAPERS_PATH = "/answer/scientific/papers"
_logger = logging.getLogger(APP_LOGGER_NAME)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: ScientificQuestion
    document_version_key: Sha256Hex | None = None


def build_answer_router(service: ScientificAnswerService | None) -> APIRouter:
    router = APIRouter(tags=["scientific answers"], route_class=BoundedScientificRoute)

    @router.get(
        PAPERS_PATH,
        response_model=tuple[AuthorizedPaper, ...],
        summary="List papers in the verified local scientific projection",
        responses={503: {"description": "Local scientific corpus unavailable"}},
    )
    def papers() -> tuple[AuthorizedPaper, ...] | JSONResponse:
        if service is None:
            return error_response("answer_disabled", 503)
        try:
            return service.available_papers()
        except GenerationUnavailableError as error:
            _logger.warning("scientific paper list unavailable: code=%s", error.code)
            return error_response(error.code, 503)
        except Exception:
            _logger.error("scientific paper list failed: code=internal_error")
            return error_response("internal_error", 503)

    @router.post(
        PATH,
        response_model=ScientificAnswer,
        summary="Generate extractive scientific claims with verified canonical citations",
        responses={
            422: {"description": "Invalid question"},
            503: {"description": "Local generation or retrieval unavailable"},
        },
    )
    def answer(body: AnswerRequest) -> ScientificAnswer | JSONResponse:
        if service is None:
            return error_response("answer_disabled", 503)
        try:
            return service.answer(body.question, body.document_version_key)
        except GenerationUnavailableError as error:
            _logger.warning("scientific answer unavailable: code=%s", error.code)
            return error_response(error.code, 422 if error.code == "invalid_request" else 503)
        except Exception:
            _logger.error("scientific answer failed: code=internal_error")
            return error_response("internal_error", 503)

    return router
