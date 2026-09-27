"""OpenSearch connectivity probe backing the readiness endpoint.

RES-130 covers runtime connectivity only. Index mappings, BM25 retrieval,
vectors, ANN and hybrid search belong to later Linear issues and are
deliberately absent here.

The probe issues a single authenticated ``GET /`` against the node root, which
is the cheapest OpenSearch endpoint that both proves the HTTP/TLS path works and
returns the running version.
"""

from __future__ import annotations

from time import perf_counter
from typing import Final

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

from dynamisrag.config import Settings
from dynamisrag.health.models import CheckStatus, DependencyCheck

__all__ = ["OPENSEARCH_DEPENDENCY_NAME", "OpenSearchProbe"]

OPENSEARCH_DEPENDENCY_NAME: Final[str] = "opensearch"
"""Stable dependency key used in the readiness payload."""

_HTTP_OK: Final[int] = 200
_MAX_DETAIL_LENGTH: Final[int] = 240
_AUTHENTICATION_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})


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
    """Long-lived, reusable client for probing one OpenSearch node."""

    __slots__ = ("_client", "_root_url")

    def __init__(
        self, settings: Settings, *, transport: httpx2.BaseTransport | None = None
    ) -> None:
        """Build a probe for the node described by ``settings``.

        ``transport`` is a seam for tests: passing
        :class:`httpx2.MockTransport` exercises every response branch of
        :meth:`check` without opening a socket. ``None`` keeps the default
        network transport.
        """
        self._root_url: Final[str] = f"{str(settings.opensearch_url).rstrip('/')}/"
        self._client: Final[httpx2.Client] = httpx2.Client(
            auth=httpx2.BasicAuth(
                settings.opensearch_username,
                settings.opensearch_password.get_secret_value(),
            ),
            verify=settings.opensearch_verify_tls,
            timeout=httpx2.Timeout(settings.dependency_timeout_seconds),
            follow_redirects=False,
            # Probes must address the configured node directly. Inheriting
            # HTTP_PROXY/HTTPS_PROXY from the developer or CI environment would
            # make readiness depend on ambient configuration.
            trust_env=False,
            transport=transport,
        )

    def check(self) -> DependencyCheck:
        """Return the current dependency verdict without raising."""
        started_at = perf_counter()
        try:
            response = self._client.get(self._root_url)
        except httpx2.HTTPError as error:
            return self._down(started_at, _describe_transport_error(error))

        if response.status_code != _HTTP_OK:
            return self._down(started_at, _describe_status(response.status_code))

        try:
            node_info = _NodeInfo.model_validate(response.json())
        except ValidationError as error:
            return self._down(started_at, f"UnexpectedPayload: {_flatten(error)}")

        return DependencyCheck(
            name=OPENSEARCH_DEPENDENCY_NAME,
            status=CheckStatus.UP,
            latency_ms=_elapsed_ms(started_at),
            version=node_info.version.number,
        )

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def _down(self, started_at: float, detail: str) -> DependencyCheck:
        return DependencyCheck(
            name=OPENSEARCH_DEPENDENCY_NAME,
            status=CheckStatus.DOWN,
            latency_ms=_elapsed_ms(started_at),
            detail=detail,
        )


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _truncate(value: str) -> str:
    return f"{value[:_MAX_DETAIL_LENGTH]}..." if len(value) > _MAX_DETAIL_LENGTH else value


def _flatten(error: ValidationError) -> str:
    return "; ".join(_truncate(str(item["msg"])) for item in error.errors())


def _describe_transport_error(error: httpx2.HTTPError) -> str:
    """Render a transport failure without echoing request headers or secrets."""
    message = _truncate(" ".join(str(error).split()))
    detail = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return f"TransportError: {detail}"


def _describe_status(status_code: int) -> str:
    if status_code in _AUTHENTICATION_STATUS_CODES:
        return (
            f"AuthenticationFailed: HTTP {status_code}; "
            "check opensearch_username and opensearch_password"
        )
    return f"UnexpectedStatus: HTTP {status_code}"
