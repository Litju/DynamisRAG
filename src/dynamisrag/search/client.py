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

**Backend failure text is untrusted.** OpenSearch's ``error.reason`` and every
``caused_by.reason`` are the node's own prose about what it rejected, and for
this projection what it rejected is article text, a query or a field value. It
is never read here, so it cannot reach an exception, a log, a readiness
payload or a terminal. Only ``error.type`` is captured, as structured context
on the raised error; :func:`_backend_error_type` is the only function in this
module that looks inside a failure envelope.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
    """Bound one fragment of a message to a single readable line.

    Length control on a value this module already decided is safe to carry —
    it keeps a long identifier or a machine-generated type name from pushing the
    actionable sentence off the end of a log line. It is never how untrusted
    text is made safe; untrusted text is not admitted in the first place.
    """
    return f"{value[:MAX_SAFE_DETAIL_LENGTH]}..." if len(value) > MAX_SAFE_DETAIL_LENGTH else value


def flatten_validation_error(error: ValidationError) -> str:
    """Render a payload-validation failure as one bounded line.

    Only pydantic's own ``msg`` strings are read, never the ``input`` that
    caused them, so a value lifted out of a backend response — article text,
    a query — cannot re-enter the message through the validation report.
    """
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
            # Only the exception *class* crosses the boundary. Its message is
            # assembled from the URL, the peer and the socket layer, none of
            # which this process controls, so it is not echoed.
            raise OpenSearchTransportError(
                f"TransportError: {type(error).__name__} while performing {operation}",
                operation=operation,
                cause=type(error).__name__,
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
        """Reject any status the operation cannot use.

        Reports only what this process wrote — the HTTP status, the operation,
        the target, the credential hint for a rejection — plus the node's
        ``error.type``. The failure body itself is never rendered: see
        :func:`_backend_error_type` for the one field that is read and why.
        """
        if response.status_code in _SUCCESS_STATUS_CODES:
            return
        error_type = _backend_error_type(response)
        if response.status_code in _AUTHENTICATION_STATUS_CODES:
            raise OpenSearchUnexpectedResponse(
                "AuthenticationFailed: "
                f"HTTP {response.status_code} while performing {operation}"
                f"{_addressed(target)}; check opensearch_username and opensearch_password",
                operation=operation,
                category="AuthenticationFailed",
                status_code=response.status_code,
                error_type=error_type,
                target=target,
            )
        raise OpenSearchUnexpectedResponse(
            f"UnexpectedStatus: HTTP {response.status_code} while performing {operation}"
            f"{_addressed(target)}{_error_type_suffix(error_type)}",
            operation=operation,
            category="UnexpectedStatus",
            status_code=response.status_code,
            error_type=error_type,
            target=target,
        )


def _addressed(target: str | None) -> str:
    return f" on {target}" if target is not None else ""


def _error_type_suffix(error_type: str | None) -> str:
    return f" (type={error_type})" if error_type is not None else ""


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


def _backend_error_type(response: httpx2.Response) -> str | None:
    """Read OpenSearch's ``error.type`` from a failed response.

    ``error.type`` is a machine-generated exception class name: it says *what
    went wrong* without restating the data that caused it, so it is the one
    field of the failure envelope that can be carried. It is bounded by
    :func:`truncate_detail` so a hostile value cannot dominate a log line.

    Nothing else in the envelope is read. ``error.reason`` and the ``reason`` of
    every ``caused_by`` are the node's own prose, and OpenSearch routinely fills
    them with the value that was rejected, the query, the document that failed
    to index, or an internal detail. For this projection the rejected document
    is indexed article text, so reading a reason at all would make it eligible
    for republication through a log line, a readiness payload or a terminal.
    Truncation would not help: a short reason still leaks.
    """
    try:
        parsed: Any = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    envelope = cast("Mapping[str, JsonValue]", parsed)
    error = envelope.get("error")
    if not isinstance(error, Mapping):
        return None
    error_type = error.get("type")
    if isinstance(error_type, str) and error_type:
        return truncate_detail(error_type)
    return None


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


@dataclass(frozen=True)
class _BulkFailure:
    """Everything about a rejected bulk request that is safe to report.

    Counts and machine-generated names only. The rejected ``_source``, the
    rejection ``reason`` and any ``caused_by`` are deliberately absent: a
    mapping rejection quotes the value it could not parse, and the value here
    is an indexed passage.
    """

    failed_items: int
    status_code: int | None
    error_type: str | None


def _require_no_bulk_errors(payload: Mapping[str, JsonValue], *, index: str, items: int) -> None:
    """Reject a bulk response that reports item-level failures.

    OpenSearch answers a partially failed bulk with HTTP 200 and
    ``"errors": true``. Treating that as success would serve a partial index, so
    the call fails and the caller abandons the whole projection.

    The message says only what this process knows: the index addressed, how many
    items were rejected out of how many were sent, and the first rejection's
    status and ``error.type``.
    """
    if payload.get("errors") is not True:
        return
    failure = _bulk_failure(payload)
    raise OpenSearchBulkError(
        f"bulk_index on {index} failed: {failure.failed_items} of {items} items rejected"
        f"{_bulk_failure_suffix(failure)}",
        operation="bulk_index",
        status_code=failure.status_code,
        error_type=failure.error_type,
        target=index,
    )


def _bulk_failure_suffix(failure: _BulkFailure) -> str:
    if failure.status_code is None and failure.error_type is None:
        return "; the response reported errors=true without naming a failed item"
    return f"; first failure status={failure.status_code} type={failure.error_type}"


def _bulk_failure(payload: Mapping[str, JsonValue]) -> _BulkFailure:
    """Count the rejected items and read the first one's status and type.

    An item counts as rejected when its result carries a non-null ``error``,
    which is OpenSearch's own signal. Only ``status`` and ``error.type`` are
    read from that item; the rest of the item is the rejected document.
    """
    response_items = payload.get("items")
    if not isinstance(response_items, Sequence) or isinstance(response_items, (str, bytes)):
        return _BulkFailure(failed_items=0, status_code=None, error_type=None)

    failed_items = 0
    first_status: int | None = None
    first_type: str | None = None
    for entry in response_items:
        if not isinstance(entry, Mapping):
            continue
        for result in entry.values():
            if not isinstance(result, Mapping) or result.get("error") is None:
                continue
            failed_items += 1
            if first_status is not None or first_type is not None:
                continue
            status = result.get("status")
            if isinstance(status, int) and not isinstance(status, bool):
                first_status = status
            error = result.get("error")
            if isinstance(error, Mapping):
                error_type = error.get("type")
                if isinstance(error_type, str) and error_type:
                    first_type = truncate_detail(error_type)
    return _BulkFailure(failed_items=failed_items, status_code=first_status, error_type=first_type)
