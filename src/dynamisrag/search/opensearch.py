"""OpenSearch connectivity probe backing the readiness endpoint.

The probe owns no transport of its own: it reuses the process-wide
:class:`~dynamisrag.search.client.OpenSearchClient` that also serves projection
and search, so authentication, TLS and timeouts are constructed once and
readiness cannot drift from the operations it describes. What this module
adds is the *readiness* interpretation — a verdict that never raises, suitable
for a health endpoint.

Because ``/readyz`` is a public surface, a down verdict is written here rather
than relayed: the probe renders the exception's safe summary, never its detail,
so nothing an OpenSearch node said can reach the response body.

    canonical PostgreSQL state
        -> PassageProjector (disposable OpenSearch projection)
        -> Bm25SearchService (BM25 over the stable alias)
"""

from __future__ import annotations

from time import perf_counter
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from dynamisrag.health.models import CheckStatus, DependencyCheck
from dynamisrag.search.client import OpenSearchClient, flatten_validation_error
from dynamisrag.search.errors import OpenSearchError

__all__ = ["OPENSEARCH_DEPENDENCY_NAME", "OpenSearchProbe"]

OPENSEARCH_DEPENDENCY_NAME: Final[str] = "opensearch"
"""Stable dependency key used in the readiness payload."""


class _VersionInfo(BaseModel):
    """Version block of the OpenSearch node root document."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    number: str


class _NodeInfo(BaseModel):
    """Subset of the OpenSearch node root document that the probe relies on.

    ``extra="ignore"`` keeps the probe forward compatible with additional
    fields introduced by future OpenSearch releases.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    cluster_name: str
    version: _VersionInfo


class OpenSearchProbe:
    """Reusable probe of one OpenSearch node, over a shared client."""

    __slots__ = ("_client",)

    def __init__(self, client: OpenSearchClient) -> None:
        """Probe through ``client``.

        The client is *not* owned here: the application lifespan closes it
        once, after the last operation that could still need it.
        """
        self._client: Final[OpenSearchClient] = client

    def check(self) -> DependencyCheck:
        """Return the current dependency verdict without raising.

        A down verdict carries :meth:`~dynamisrag.search.errors.OpenSearchError.safe_summary`
        rather than the exception text. Readiness is a *public* endpoint, so its
        detail is assembled from the exception's class and its structured fields
        only. It keeps everything an operator needs to act — the failure
        category, the operation, the HTTP status, OpenSearch's ``error.type`` —
        and cannot relay a backend reason, a response body or article text even
        if some future caller puts one into ``detail``.
        """
        started_at = perf_counter()
        try:
            payload = self._client.node_root()
        except OpenSearchError as error:
            return self._down(started_at, error.safe_summary())

        try:
            node_info = _NodeInfo.model_validate(payload)
        except ValidationError as error:
            return self._down(
                started_at,
                f"UnexpectedPayload: node_root returned a body that does not match the node "
                f"root document ({flatten_validation_error(error)})",
            )

        return DependencyCheck(
            name=OPENSEARCH_DEPENDENCY_NAME,
            status=CheckStatus.UP,
            latency_ms=_elapsed_ms(started_at),
            version=node_info.version.number,
        )

    def _down(self, started_at: float, detail: str) -> DependencyCheck:
        return DependencyCheck(
            name=OPENSEARCH_DEPENDENCY_NAME,
            status=CheckStatus.DOWN,
            latency_ms=_elapsed_ms(started_at),
            detail=detail,
        )


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)
