"""Reusable authenticated OpenSearch HTTP client (RES-135).

One client per process, shared by the readiness probe, the passage projector
and the BM25 search service, so authentication and TLS are constructed exactly
once and can never drift apart between three independent call sites. The
construction is pure configuration work: no connection is opened until an
operation executes, and :meth:`close` releases the pool once, from the
application lifespan.

The transport is the repository's already-tested ``httpx2`` boundary:

* Basic Auth from the frozen settings (never a raw password literal),
* TLS verification exactly as configured,
* explicit timeouts on every request,
* ``follow_redirects=False`` — a redirect is not a successful index operation,
* ``trust_env=False`` — an ambient ``HTTPS_PROXY`` must never decide where the
  search backend lives.

Every operation funnels through :meth:`_request`, so transport failures,
unexpected statuses and untrustworthy payloads are turned into the typed
errors in :mod:`dynamisrag.search.errors` with the same safe wording
everywhere. Nothing in this module logs or raises request headers, credentials,
complete response bodies or indexed article text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final, cast

import httpx2
from pydantic import ValidationError

from dynamisrag.config import Settings
from dynamisrag.search.errors import (
    MAX_SAFE_DETAIL_LENGTH,
    OpenSearchBulkError,
    OpenSearchTransportError,
    OpenSearchUnexpectedResponse,
)

__all__ = [
    "JsonValue",
    "OpenSearchClient",
    "canonical_json_line",
    "flatten_validation_error",
    "truncate_detail",
    "validate_resource_name",
]

type JsonValue = str | int | float | bool | Sequence[JsonValue] | Mapping[str, JsonValue] | None
"""The JSON value domain, stated explicitly instead of ``Any``.

Every payload crossing the OpenSearch boundary is validated structurally by the
functions below, so the type is a recursive description of what a decoded JSON
document can contain. Naming it keeps a decoded response *checked* through
validation rather than silently trusted: an unexpected shape is caught by
``isinstance`` and reported, instead of propagating as an untyped value into a
projection or a search response.

The object and array members are the covariant ``Mapping``/``Sequence``
protocols rather than the concrete ``dict``/``list``, so a narrower concrete
value — a freshly built request body, for instance — is still a ``JsonValue``
without a cast.
"""

_MAX_NAME_LENGTH: Final[int] = 255
"""OpenSearch refuses index and alias names longer than this, in bytes."""

_RESOURCE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
"""OpenSearch index/alias naming restriction.

Lowercase only, must not start with ``-``, ``_`` or ``+``, and may contain only
letters, digits, dots, hyphens and underscores. Every name this client accepts
is built from semantic, configured values, so the pattern is a guard rail
against an unvalidated caller rather than a source of valid inputs.
"""

_AUTHENTICATION_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})
_SUCCESS_STATUS_CODES: Final[frozenset[int]] = frozenset({200, 201})

_NDJSON_CONTENT_TYPE: Final[str] = "application/x-ndjson"
_JSON_CONTENT_TYPE: Final[str] = "application/json"

_ProjectionDocument = tuple[str, Mapping[str, JsonValue]]
"""``(document_id, source)`` — the id becomes the OpenSearch ``_id``."""


def truncate_detail(value: str) -> str:
    """Shorten a backend-supplied detail to a single safe log line."""
    return f"{value[:MAX_SAFE_DETAIL_LENGTH]}..." if len(value) > MAX_SAFE_DETAIL_LENGTH else value


def flatten_validation_error(error: ValidationError) -> str:
    """Render a payload-validation failure as one safe, truncated line."""
    return "; ".join(truncate_detail(str(item["msg"])) for item in error.errors())


def validate_resource_name(name: str, *, kind: str) -> str:
    """Return ``name`` when it satisfies OpenSearch naming rules, else raise.

    Validated *before* a request is issued so an invalid name is a local,
    explicit failure rather than an opaque HTTP 400 from the node.
    """
    if len(name.encode("utf-8")) > _MAX_NAME_LENGTH:
        raise ValueError(
            f"{kind} name is {len(name.encode('utf-8'))} bytes long; OpenSearch allows "
            f"at most {_MAX_NAME_LENGTH}"
        )
    if name in {".", ".."} or _RESOURCE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"{kind} name {name!r} violates the OpenSearch naming restriction: it must be "
            "lowercase and start with a letter or digit, containing only letters, digits, "
            "'.', '_' and '-'"
        )
    return name


def canonical_json_line(payload: object) -> str:
    """Deterministic JSON rendering used for bulk actions and bulk documents.

    Sorted keys, compact separators and ``ensure_ascii=False`` make the NDJSON
    body byte-stable for a given projection, which is what lets the bulk
    request be asserted exactly in tests.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class OpenSearchClient:
    """Thin, typed wrapper over the OpenSearch REST API.

    Each public method is one OpenSearch operation with one meaning, so the
    request that leaves the process is fully determined by its arguments and
    is assertable without a live node. Unknown response shapes raise rather
    than being coerced.
    """

    __slots__ = ("_base_url", "_client")

    def __init__(
        self, settings: Settings, *, transport: httpx2.BaseTransport | None = None
    ) -> None:
        """Build a client for the node described by ``settings``.

        ``transport`` is a test seam: passing :class:`httpx2.MockTransport`
        exercises every branch without opening a socket. ``None`` keeps the
        default network transport. Construction performs no I/O.
        """
        self._base_url: Final[str] = str(settings.opensearch_url).rstrip("/")
        self._client: Final[httpx2.Client] = httpx2.Client(
            auth=httpx2.BasicAuth(
                settings.opensearch_username,
                settings.opensearch_password.get_secret_value(),
            ),
            verify=settings.opensearch_verify_tls,
            timeout=httpx2.Timeout(settings.dependency_timeout_seconds),
            follow_redirects=False,
            # Every OpenSearch call must address the configured node directly.
            # Inheriting HTTP_PROXY/HTTPS_PROXY from the developer or CI
            # environment would make search depend on ambient configuration.
            trust_env=False,
            headers={"Accept": _JSON_CONTENT_TYPE},
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        """Configured base URL without a trailing slash; the URL prefix is
        preserved exactly as configured."""
        return self._base_url

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def node_root(self) -> Mapping[str, JsonValue]:
        """``GET /`` — the cheapest operation that proves the HTTP/TLS path
        works and returns the running version."""
        return self._json("GET", "/", operation="node_root")

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def index_exists(self, index: str) -> bool:
        """``HEAD /{index}`` — whether a physical index is present."""
        validate_resource_name(index, kind="index")
        response = self._request("HEAD", f"/{index}", operation="index_exists")
        if response.status_code == 404:
            return False
        self._require_ok(response, "index_exists", index)
        return True

    def create_index(
        self, index: str, *, settings: Mapping[str, JsonValue], mappings: Mapping[str, JsonValue]
    ) -> None:
        """``PUT /{index}`` with the exact settings and mappings supplied."""
        validate_resource_name(index, kind="index")
        payload: Mapping[str, JsonValue] = {"settings": settings, "mappings": mappings}
        self._require_ok(
            self._request(
                "PUT",
                f"/{index}",
                operation="create_index",
                payload=payload,
            ),
            "create_index",
            index,
        )

    def delete_index(self, index: str) -> None:
        """``DELETE /{index}``.

        A missing index counts as success: the caller wants the name free, and
        an index that vanished between the existence check and the delete is
        exactly the state being asked for.
        """
        validate_resource_name(index, kind="index")
        response = self._request("DELETE", f"/{index}", operation="delete_index")
        if response.status_code == 404:
            return
        self._require_ok(response, "delete_index", index)

    def count(self, index: str) -> int:
        """``GET /{index}/_count`` — the number of indexed documents."""
        validate_resource_name(index, kind="index")
        payload = self._json("GET", f"/{index}/_count", operation="count", target=index)
        return _require_int(payload, "count", "count")

    def index_meta(self, index: str) -> Mapping[str, JsonValue]:
        """``GET /{index}/_mapping`` — the mapping ``_meta`` block.

        When ``index`` is an alias, OpenSearch resolves it, so this reads the
        provenance of whichever physical index the query target currently
        points at.
        """
        validate_resource_name(index, kind="index")
        payload = self._json("GET", f"/{index}/_mapping", operation="index_meta", target=index)
        entries = list(payload.values())
        if len(entries) != 1:
            raise OpenSearchUnexpectedResponse(
                f"UnexpectedPayload: index_meta for {index} returned {len(entries)} mappings; "
                "exactly one index must be resolved",
                operation="index_meta",
            )
        entry = entries[0]
        if not isinstance(entry, Mapping):
            raise OpenSearchUnexpectedResponse(
                f"UnexpectedPayload: index_meta for {index} returned a non-object mapping entry",
                operation="index_meta",
            )
        mappings = entry.get("mappings")
        if not isinstance(mappings, Mapping):
            raise OpenSearchUnexpectedResponse(
                f"UnexpectedPayload: index {index} has no mappings block",
                operation="index_meta",
            )
        meta = mappings.get("_meta")
        if not isinstance(meta, Mapping):
            raise OpenSearchUnexpectedResponse(
                f"UnexpectedPayload: index {index} mapping carries no _meta block",
                operation="index_meta",
            )
        return meta

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------

    def alias_targets(self, alias: str) -> tuple[str, ...]:
        """``GET /_alias/{alias}`` — the physical indexes the alias points at.

        A missing alias is an empty target set, not an error: that is the
        normal state before the first projection.
        """
        validate_resource_name(alias, kind="alias")
        response = self._request("GET", f"/_alias/{alias}", operation="alias_targets")
        if response.status_code == 404:
            return ()
        self._require_ok(response, "alias_targets", alias)
        payload = _decode_json(response, "alias_targets", alias)
        return tuple(sorted(str(key) for key in payload))

    def switch_alias(self, alias: str, *, index: str, remove: Sequence[str]) -> None:
        """``POST /_aliases`` — atomically repoint ``alias`` at ``index``.

        Every ``remove`` action is derived from a target the alias is known to
        have, and the ``add`` action for the new index is in the same atomic
        request, so the alias is never momentarily pointing at nothing and
        never points at a partially built index.
        """
        validate_resource_name(alias, kind="alias")
        validate_resource_name(index, kind="index")
        actions: list[dict[str, JsonValue]] = [
            {"remove": {"index": validate_resource_name(target, kind="index"), "alias": alias}}
            for target in sorted(set(remove))
            if target != index
        ]
        actions.append({"add": {"index": index, "alias": alias}})
        payload: Mapping[str, JsonValue] = {"actions": actions}
        self._require_ok(
            self._request(
                "POST",
                "/_aliases",
                operation="switch_alias",
                payload=payload,
            ),
            "switch_alias",
            alias,
        )

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def bulk_index(
        self,
        index: str,
        documents: Sequence[_ProjectionDocument],
        *,
        batch_size: int,
    ) -> int:
        """``POST /_bulk`` with ``refresh=wait_for``; returns the item count.

        Every document is sent with an explicit ``index`` action whose ``_id``
        is the caller's deterministic document id — never a generated one — so
        a rebuild overwrites rather than duplicates. A top-level HTTP 200 is
        explicitly *not* treated as success: ``errors == true`` fails the call.
        """
        validate_resource_name(index, kind="index")
        if batch_size < 1:
            raise ValueError(f"bulk batch size must be >= 1, got {batch_size}")
        indexed = 0
        for batch in _batches(documents, batch_size):
            body = b"".join(_bulk_line(index, document_id, source) for document_id, source in batch)
            payload = self._json(
                "POST",
                "/_bulk",
                operation="bulk_index",
                target=index,
                content=body,
                content_type=_NDJSON_CONTENT_TYPE,
                params={"refresh": "wait_for"},
            )
            _require_no_bulk_errors(payload, index=index, items=len(batch))
            indexed += len(batch)
        return indexed

    def search(self, index: str, body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """``POST /{index}/_search`` with the exact request body supplied."""
        validate_resource_name(index, kind="index")
        return self._json(
            "POST", f"/{index}/_search", operation="search", target=index, payload=body
        )

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        payload: Mapping[str, JsonValue] | None = None,
        content: bytes | None = None,
        content_type: str | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx2.Response:
        """Issue one request, converting any transport failure into a typed
        error that names the operation but never the request headers."""
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": content_type} if content_type is not None else None
        try:
            if content is not None:
                return self._client.request(
                    method, url, content=content, headers=headers, params=params
                )
            return self._client.request(method, url, json=payload, headers=headers, params=params)
        except httpx2.HTTPError as error:
            raise OpenSearchTransportError(
                _describe_transport_error(error), operation=operation
            ) from error

    def _json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        target: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
        content: bytes | None = None,
        content_type: str | None = None,
        params: Mapping[str, str] | None = None,
    ) -> Mapping[str, JsonValue]:
        """Issue one request and decode a JSON object response body."""
        response = self._request(
            method,
            path,
            operation=operation,
            payload=payload,
            content=content,
            content_type=content_type,
            params=params,
        )
        self._require_ok(response, operation, target)
        return _decode_json(response, operation, target)

    def _require_ok(self, response: httpx2.Response, operation: str, target: str | None) -> None:
        if response.status_code in _SUCCESS_STATUS_CODES:
            return
        if response.status_code in _AUTHENTICATION_STATUS_CODES:
            raise OpenSearchUnexpectedResponse(
                "AuthenticationFailed: "
                f"HTTP {response.status_code} while performing {operation}"
                f"{_addressed(target)}; check opensearch_username and opensearch_password",
                operation=operation,
            )
        raise OpenSearchUnexpectedResponse(
            f"UnexpectedStatus: HTTP {response.status_code} while performing {operation}"
            f"{_addressed(target)}{_backend_error_suffix(response)}",
            operation=operation,
        )


def _addressed(target: str | None) -> str:
    return f" on {target}" if target is not None else ""


def _describe_transport_error(error: httpx2.HTTPError) -> str:
    """Render a transport failure without echoing request headers or secrets."""
    message = truncate_detail(" ".join(str(error).split()))
    detail = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return f"TransportError: {detail} while performing the request"


def _decode_json(
    response: httpx2.Response, operation: str, target: str | None
) -> Mapping[str, JsonValue]:
    try:
        parsed: Any = response.json()
    except ValueError:
        raise OpenSearchUnexpectedResponse(
            f"UnexpectedPayload: {operation}{_addressed(target)} returned a body that is not "
            "valid JSON",
            operation=operation,
        ) from None
    if not isinstance(parsed, dict):
        raise OpenSearchUnexpectedResponse(
            f"UnexpectedPayload: {operation}{_addressed(target)} returned a JSON "
            f"{type(parsed).__name__} where an object was required",
            operation=operation,
        )
    return cast("Mapping[str, JsonValue]", parsed)


def _backend_error_suffix(response: httpx2.Response) -> str:
    """Extract the safe part of an OpenSearch error envelope.

    Only ``error.type`` and ``error.reason`` are surfaced, both truncated. The
    full body — which may echo a rejected request or a shard failure with
    document content — is deliberately never included.
    """
    try:
        parsed: Any = response.json()
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    envelope = cast("Mapping[str, JsonValue]", parsed)
    error = envelope.get("error")
    if not isinstance(error, dict):
        return ""
    detail = cast("Mapping[str, JsonValue]", error)
    parts: list[str] = []
    for label, key in (("type", "type"), ("reason", "reason")):
        value = detail.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{label}={truncate_detail(value)}")
    return f" ({'; '.join(parts)})" if parts else ""


def _require_int(payload: Mapping[str, JsonValue], key: str, operation: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenSearchUnexpectedResponse(
            f"UnexpectedPayload: {operation} response carries no integer {key!r} field",
            operation=operation,
        )
    return value


def _batches(
    documents: Sequence[_ProjectionDocument], batch_size: int
) -> Iterator[Sequence[_ProjectionDocument]]:
    """Split documents into fixed-size batches.

    The size is configured rather than derived, so a rebuild sends the same
    request boundaries for the same projection.
    """
    for start in range(0, len(documents), batch_size):
        yield documents[start : start + batch_size]


def _bulk_line(index: str, document_id: str, source: Mapping[str, JsonValue]) -> bytes:
    """One action line plus exactly one document line, newline terminated."""
    action = canonical_json_line({"index": {"_index": index, "_id": document_id}})
    document = canonical_json_line(source)
    return f"{action}\n{document}\n".encode()


def _require_no_bulk_errors(payload: Mapping[str, JsonValue], *, index: str, items: int) -> None:
    """Reject a bulk response that reports item-level failures.

    OpenSearch answers a partially failed bulk with HTTP 200 and
    ``"errors": true``. Treating that as success would serve a partial index,
    so the first item's status and safe error type/reason are reported and the
    caller fails the whole projection.
    """
    if payload.get("errors") is not True:
        return
    failed = _first_bulk_failure(payload)
    detail = (
        f"; first failure status={failed[0]} type={failed[1]} reason={failed[2]}" if failed else ""
    )
    raise OpenSearchBulkError(
        f"bulk_index on {index} reported errors=true after indexing into the index: "
        f"HTTP 200 with at least one failed item out of {items}{detail}",
        operation="bulk_index",
    )


def _first_bulk_failure(payload: Mapping[str, JsonValue]) -> tuple[str, str, str] | None:
    """Return ``(status, type, reason)`` of the first failed bulk item."""
    response_items = payload.get("items")
    if not isinstance(response_items, Sequence):
        return None
    for entry in response_items:
        if not isinstance(entry, Mapping):
            continue
        for result in entry.values():
            if not isinstance(result, Mapping) or result.get("error") is None:
                continue
            error = result["error"]
            status = str(result.get("status", "unknown"))
            if not isinstance(error, Mapping):
                return (status, type(error).__name__, "")
            error_type = error.get("type")
            reason = error.get("reason")
            return (
                status,
                truncate_detail(error_type) if isinstance(error_type, str) else "unknown",
                truncate_detail(reason) if isinstance(reason, str) else "",
            )
    return None
