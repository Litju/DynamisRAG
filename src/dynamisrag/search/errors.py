"""Typed error surface for the OpenSearch boundary (RES-135).

Deliberately small: a base :class:`OpenSearchError` for everything the search
backend can do wrong, plus four narrow specializations the caller actually
branches on.

    OpenSearchError
      OpenSearchTransportError       the request never produced a response
      OpenSearchUnexpectedResponse   a response arrived but cannot be trusted
        OpenSearchBulkError          the bulk API reported item-level failures
      ProjectionError                the projection could not be built or verified
        ProjectionConflictError      a live index contradicts the projection and was left alone
      SearchBackendError             a search response could not be validated
      VectorContractError            a vector index configuration or vector set is not usable

The nesting is meaningful: ``ProjectionConflictError`` *is* a
``ProjectionError``, and both are ``OpenSearchError``, so a caller that only
wants "the backend misbehaved" catches the base type, while the projector can
still single out the one condition that means "an index exists whose live state
this projection cannot account for".

**The untrusted-field policy, stated once**

Everything OpenSearch sends in a failure envelope is treated as untrusted
except one field. ``error.reason`` — and the ``reason`` of every
``caused_by`` — is prose written by the node, and it routinely restates the
thing that went wrong: a rejected field value, the query, the rejected
document, or an internal detail. For this projection the rejected document *is*
indexed article text, so relaying a reason would republish scientific content
through a log line, a readiness payload or a terminal. It is never read,
therefore never logged, rendered or raised — truncating it would be length
control, not redaction.

The single exception is ``error.type``: a machine-generated exception class
name that says what went wrong without restating the data that caused it. It
is captured, bounded, and made available as structured context.

That is why the base class carries *structured fields* alongside ``detail``.
:meth:`OpenSearchError.safe_summary` is assembled from those fields and never
from ``detail``, so a log, a readiness payload or a CLI line can be produced
from it structurally. ``detail`` remains the long-form prose for an operator
reading a traceback, and stays safe only because the code that raises it writes
it itself.

Never carried: an ``Authorization`` header, a password, a request body, a
complete response body, a rejected document, or indexed article text.
"""

from __future__ import annotations

from typing import ClassVar, Final

__all__ = [
    "MAX_SAFE_DETAIL_LENGTH",
    "OpenSearchBulkError",
    "OpenSearchError",
    "OpenSearchTransportError",
    "OpenSearchUnexpectedResponse",
    "ProjectionConflictError",
    "ProjectionError",
    "SearchBackendError",
    "VectorContractError",
]

MAX_SAFE_DETAIL_LENGTH: Final[int] = 240
"""Longest single fragment an exception message may carry.

A length bound on an *application-authored* line, so a long projection
identifier cannot push the actionable sentence off the end of a log line. It
is a formatting rule, not a security control: nothing untrusted is ever
shortened into safety by it, because nothing untrusted is ever admitted.
"""


class OpenSearchError(Exception):
    """Base class for every failure at the OpenSearch boundary.

    Two separate views of the same failure:

    ``detail``
        Long-form prose for a human reading a traceback. Written by this
        codebase, so it names the operation and the target and nothing else —
        but it is a free-form string, and a future caller could point it at
        anything.

    :meth:`safe_summary`
        One line assembled from this exception's class and its structured
        fields only. ``detail`` is deliberately excluded, which is what makes
        it safe to hand to a log, a readiness payload or a CLI without any
        further review.
    """

    _CATEGORY: ClassVar[str] = "OpenSearchError"
    """Application-authored label for this failure class.

    Set per class so the default summary reads in domain terms
    (``UnexpectedStatus``, ``BulkFailure``) rather than in exception-class
    terms, and overridable per instance where one class covers several
    distinct conditions — an authentication rejection and an unexpected status
    are both :class:`OpenSearchUnexpectedResponse`.
    """

    def __init__(
        self,
        detail: str,
        *,
        operation: str,
        category: str | None = None,
        cause: str | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        target: str | None = None,
    ) -> None:
        """Record the prose ``detail`` plus the structured safe context.

        ``cause`` is the *class name* of an underlying exception — never its
        message. ``status_code`` is the HTTP status, ``error_type`` is
        OpenSearch's ``error.type`` and ``target`` is the index or alias the
        operation addressed. All four are either written by this process or
        machine-generated by the node, which is what makes them safe to
        surface and useless as a channel for backend content.
        """
        super().__init__(detail)
        self.detail: Final[str] = detail
        self.operation: Final[str] = operation
        self.category: Final[str] = category if category is not None else type(self)._CATEGORY
        self.cause: Final[str | None] = cause
        self.status_code: Final[int | None] = status_code
        self.error_type: Final[str | None] = error_type
        self.target: Final[str | None] = target

    def __str__(self) -> str:
        return self.detail

    def safe_summary(self) -> str:
        """A one-line summary built only from controlled values.

        Never derived from ``detail``. The result is safe to log, to return in
        a readiness payload and to print on a terminal, and it keeps the facts
        an operator actually needs: the failure category, the operation, the
        HTTP status, OpenSearch's ``error.type`` and the target addressed.
        """
        parts: list[str] = [self.category]
        if self.cause is not None:
            parts.append(f"cause={self.cause}")
        parts.append(f"operation={self.operation}")
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.error_type is not None:
            parts.append(f"error.type={self.error_type}")
        if self.target is not None:
            parts.append(f"target={self.target}")
        return " ".join(parts)


class OpenSearchTransportError(OpenSearchError):
    """The request never produced a response (connect, TLS, timeout, ...).

    No response means no backend text is available and none is wanted, so the
    summary carries only the exception class name and the operation. The
    transport exception's own message is dropped: it is not written by this
    process, and it can quote whatever the URL or the peer put into it.
    """

    _CATEGORY: ClassVar[str] = "TransportError"


class OpenSearchUnexpectedResponse(OpenSearchError):
    """A response arrived but its status or payload cannot be trusted.

    The default category is :attr:`SearchBackendError`'s, because the usual
    cause is a payload whose shape cannot be trusted. The two status-based
    conditions override it per instance, so an authentication rejection and an
    unexpected status stay legible without parsing prose.
    """

    _CATEGORY: ClassVar[str] = "UnexpectedPayload"


class OpenSearchBulkError(OpenSearchError):
    """The bulk API reported item-level failures.

    A top-level HTTP 200 is not success: OpenSearch answers ``200`` with
    ``{"errors": true}`` when individual documents were rejected. The message
    reports how many items were rejected out of how many were sent and the
    first rejection's status and ``error.type``.

    Never the rejection ``reason``, the ``_source`` it refers to, or the
    document: for this projection the document is article text, and a mapping
    rejection quotes the value it could not parse.
    """

    _CATEGORY: ClassVar[str] = "BulkFailure"


class ProjectionError(OpenSearchError):
    """The passage projection could not be built, verified or completed.

    The messages here are written by the projector from canonical state — a
    requested chunker revision, the revisions that do exist, the projection
    digest — and are safe to show a user verbatim, because they are the
    actionable part of a projection failure.
    """

    _CATEGORY: ClassVar[str] = "ProjectionFailed"


class ProjectionConflictError(ProjectionError):
    """A completed verification proved a live index contradicts the projection.

    Raised only when verification *ran to completion* and the active index's
    document count or mapping ``_meta`` did not match the deterministic
    manifest. It is never raised because verification could not be performed:
    an unreadable index is an ``OpenSearchError`` of its own kind, and
    collapsing the two would claim knowledge of the index's contents that a
    failed read never provided.

    Such an index is left exactly as it is. It may be the only complete copy of
    a projection, and silently deleting or replacing what a live alias serves
    converts a visible inconsistency into an outage.
    """

    _CATEGORY: ClassVar[str] = "ProjectionConflict"


class SearchBackendError(OpenSearchError):
    """A search response could not be validated into a :class:`SearchResponse`."""

    _CATEGORY: ClassVar[str] = "UnexpectedPayload"


class VectorContractError(OpenSearchError):
    """A vector index configuration or vector set cannot be trusted.

    Raised locally, before anything is sent, when the dense-vector contract is
    violated: an unsupported or unstated distance function, a dimension outside
    the accepted range, an embedding identity that is mutable or absent, a search
    time parameter in an index mapping, or a ``passage_key`` ↔ vector set that is
    not exactly one-to-one.

    It is a :class:`OpenSearchError` so a caller that only needs to know "the
    search boundary refused to proceed" catches one type, while an operator can
    still single this condition out. Every value it reports is configuration or a
    key this process was handed; a vector *component* is never echoed, because a
    vector is derived from indexed article text.
    """

    _CATEGORY: ClassVar[str] = "VectorContractInvalid"
