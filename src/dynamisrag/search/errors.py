"""Typed error surface for the OpenSearch boundary (RES-135).

Deliberately small: a base :class:`OpenSearchError` for everything the search
backend can do wrong, plus four narrow specializations the caller actually
branches on. Every message is written to be *safe to log and to render*: it
carries the operation, the index or alias it addressed and — where OpenSearch
reported one — the error type and reason. It never carries an authorization
header, a password, a complete response body or indexed article text.

    OpenSearchError
      OpenSearchTransportError       the request never produced a response
      OpenSearchUnexpectedResponse   a response arrived but cannot be trusted
        OpenSearchBulkError          the bulk API reported item-level failures
      ProjectionError                the projection could not be built or verified
        ProjectionConflictError      an index exists whose state contradicts the projection
      SearchBackendError             a search response could not be validated

The nesting is meaningful: ``ProjectionConflictError`` *is* a
``ProjectionError``, and both are ``OpenSearchError``, so a caller that only
wants "the backend misbehaved" catches the base type, while the projector can
still single out the one condition that means "an index exists but must not be
trusted".
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "MAX_SAFE_DETAIL_LENGTH",
    "OpenSearchBulkError",
    "OpenSearchError",
    "OpenSearchTransportError",
    "OpenSearchUnexpectedResponse",
    "ProjectionConflictError",
    "ProjectionError",
    "SearchBackendError",
]

MAX_SAFE_DETAIL_LENGTH: Final[int] = 240
"""Longest safe detail an OpenSearch error message may carry.

A backend error reason is operator-facing context, not a payload: truncating
keeps a log line readable and makes it structurally impossible for an
unexpectedly verbose reason to smuggle a large chunk of a response body (and
therefore indexed article text) into an exception message or an API error.
"""


class OpenSearchError(Exception):
    """Base class for every failure at the OpenSearch boundary.

    The message is already a safe, loggable detail string. Subclasses refine
    *which* boundary condition failed, never *what* may be said about it.
    """

    def __init__(self, detail: str, *, operation: str) -> None:
        super().__init__(detail)
        self.detail: Final[str] = detail
        self.operation: Final[str] = operation

    def __str__(self) -> str:
        return self.detail


class OpenSearchTransportError(OpenSearchError):
    """The request never produced a response (connect, TLS, timeout, ...)."""


class OpenSearchUnexpectedResponse(OpenSearchError):
    """A response arrived but its status or payload cannot be trusted."""


class OpenSearchBulkError(OpenSearchError):
    """The bulk API reported item-level failures.

    A top-level HTTP 200 is not success: OpenSearch answers ``200`` with
    ``{"errors": true}`` when individual documents were rejected. The message
    reports how many items failed and the first safe status/error-type/reason —
    never the rejected document, which is indexed article text.
    """


class ProjectionError(OpenSearchError):
    """The passage projection could not be built, verified or completed."""


class ProjectionConflictError(ProjectionError):
    """An index of the deterministic name exists in a state that contradicts
    the projection, so it must be deleted and rebuilt rather than trusted."""


class SearchBackendError(OpenSearchError):
    """A search response could not be validated into a :class:`SearchResponse`."""
