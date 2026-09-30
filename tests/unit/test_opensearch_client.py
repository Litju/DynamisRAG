"""The OpenSearch HTTP client, exercised entirely through a mocked transport.

No test here opens a socket. The suite pins the two things that make the client
trustworthy at a glance:

* **Construction** — authentication, TLS verification, timeouts, redirect
  policy and the absence of ambient proxy inheritance are asserted from the
  exact arguments handed to ``httpx2.Client``, so a later refactor cannot
  quietly relax one of them.
* **Requests** — every operation's method, path, query string, content type
  and body is asserted byte-for-byte, because the bulk body and the alias
  cutover request are protocol, not implementation detail.

Failures are asserted to be *safe*: a typed error that names the exception
class, the operation, the HTTP status and OpenSearch's ``error.type``, and
never the credentials, the response body, an ``error.reason`` or indexed
article text.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx2
import pytest

from dynamisrag.config import Settings
from dynamisrag.search.client import OpenSearchClient, validate_resource_name
from dynamisrag.search.errors import (
    OpenSearchBulkError,
    OpenSearchError,
    OpenSearchTransportError,
    OpenSearchUnexpectedResponse,
)
from tests._support import UNIT_TEST_PASSWORD, build_settings

_OPENSEARCH_URL: Final[str] = "https://search.internal:9200"
_TIMEOUT_SECONDS: Final[float] = 2.0
_INDEX: Final[str] = "dynamisrag-passages-passage-index-v1-0a1b2c3d4e5f"
_ALIAS: Final[str] = "dynamisrag-passages"

_SETTINGS: Final[Settings] = build_settings(opensearch_url=_OPENSEARCH_URL)

_MAPPING_SETTINGS: Final[Mapping[str, Any]] = {"index": {"number_of_shards": 1}}
_MAPPING_PROPERTIES: Final[Mapping[str, Any]] = {
    "dynamic": "strict",
    "properties": {"passage_key": {"type": "keyword"}},
}

_BULK_OK: Final[Mapping[str, Any]] = {"errors": False, "items": [{"index": {"status": 201}}]}


# ---------------------------------------------------------------------------
# Mock transport plumbing
# ---------------------------------------------------------------------------


class _Recorder:
    """A scripted transport that records every request it answers."""

    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses: list[httpx2.Response] = list(responses)
        self.requests: list[httpx2.Request] = []

    def transport(self) -> httpx2.MockTransport:
        def answer(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            if not self.responses:
                raise AssertionError(f"unexpected extra request: {request.method} {request.url}")
            response = self.responses.pop(0)
            response.request = request
            return response

        return httpx2.MockTransport(answer)


def _json_response(payload: object, status_code: int = 200) -> httpx2.Response:
    return httpx2.Response(status_code, content=json.dumps(payload).encode("utf-8"))


def _client(recorder: _Recorder) -> OpenSearchClient:
    return OpenSearchClient(_SETTINGS, transport=recorder.transport())


def _presented_credentials(request: httpx2.Request) -> str:
    """Decode the credential pair the transport actually put on the wire.

    The test asserts the real header rather than the constructor arguments, so
    a change in how the credentials are assembled is still caught. The decoded
    value is a throwaway test credential, never a real secret.
    """
    header = request.headers["authorization"]
    assert header.startswith("Basic ")
    return base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_preserves_every_transport_safety_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authentication, TLS, timeouts, redirects and proxy isolation are passed
    explicitly, so none of them can be inherited from the environment."""
    recorded: dict[str, Any] = {}

    class _Recording:
        def __init__(self, **kwargs: Any) -> None:
            recorded.update(kwargs)

        def request(self, *_: Any, **__: Any) -> httpx2.Response:  # pragma: no cover
            raise AssertionError("no request is issued during construction")

        def close(self) -> None:
            recorded["closed"] = True

    monkeypatch.setattr(httpx2, "Client", _Recording)
    client = OpenSearchClient(_SETTINGS)

    assert isinstance(recorded["auth"], httpx2.BasicAuth)
    assert recorded["verify"] is False
    assert recorded["follow_redirects"] is False
    assert recorded["trust_env"] is False
    assert recorded["timeout"] == httpx2.Timeout(_TIMEOUT_SECONDS)
    assert recorded["transport"] is None
    assert client.base_url == _OPENSEARCH_URL


def test_a_url_prefix_is_preserved_exactly() -> None:
    recorder = _Recorder(_json_response({"version": {"number": "3.8.0"}}))
    client = OpenSearchClient(
        build_settings(opensearch_url="https://gateway.internal/opensearch"),
        transport=recorder.transport(),
    )

    client.node_root()

    assert str(recorder.requests[0].url) == "https://gateway.internal/opensearch/"
    assert client.base_url == "https://gateway.internal/opensearch"


def test_a_trailing_slash_on_the_configured_url_is_normalised() -> None:
    client = OpenSearchClient(
        build_settings(opensearch_url="https://search.internal:9200/"),
        transport=_Recorder().transport(),
    )

    assert client.base_url == "https://search.internal:9200"


def test_every_request_carries_basic_auth() -> None:
    recorder = _Recorder(_json_response({"hits": {}}))
    client = _client(recorder)

    client.search(_ALIAS, {"query": {"match_all": {}}})

    assert _presented_credentials(recorder.requests[0]) == f"admin:{UNIT_TEST_PASSWORD}"


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


def test_create_index_sends_the_exact_settings_and_mappings() -> None:
    recorder = _Recorder(_json_response({"acknowledged": True}))
    client = _client(recorder)

    client.create_index(_INDEX, settings=_MAPPING_SETTINGS, mappings=_MAPPING_PROPERTIES)

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert str(request.url) == f"{_OPENSEARCH_URL}/{_INDEX}"
    assert json.loads(request.content) == {
        "settings": _MAPPING_SETTINGS,
        "mappings": _MAPPING_PROPERTIES,
    }


def test_index_exists_is_a_head_request() -> None:
    recorder = _Recorder(httpx2.Response(200))
    client = _client(recorder)

    assert client.index_exists(_INDEX) is True
    assert recorder.requests[0].method == "HEAD"


def test_a_missing_index_is_reported_as_absent_rather_than_an_error() -> None:
    recorder = _Recorder(httpx2.Response(404, content=b'{"error":{"type":"index_not_found"}}'))
    client = _client(recorder)

    assert client.index_exists(_INDEX) is False


def test_deleting_an_absent_index_is_success() -> None:
    recorder = _Recorder(httpx2.Response(404, content=b"{}"))
    client = _client(recorder)

    client.delete_index(_INDEX)


def test_count_returns_the_exact_document_count() -> None:
    recorder = _Recorder(_json_response({"count": 42, "_shards": {"total": 1}}))
    client = _client(recorder)

    assert client.count(_INDEX) == 42
    assert str(recorder.requests[0].url) == f"{_OPENSEARCH_URL}/{_INDEX}/_count"


def test_a_count_response_without_an_integer_count_is_rejected() -> None:
    recorder = _Recorder(_json_response({"count": "many"}))
    client = _client(recorder)

    with pytest.raises(OpenSearchUnexpectedResponse, match="no integer 'count'"):
        client.count(_INDEX)


def test_index_meta_returns_the_meta_block() -> None:
    meta = {
        "schema_revision": "passage-index-v1",
        "projection_sha256": "a" * 64,
        "chunker_revision": "structure-v1.1.b19e0939b5de",
        "bm25_similarity_revision": "dynamis_bm25_v1",
    }
    recorder = _Recorder(_json_response({_INDEX: {"mappings": {"_meta": meta, "properties": {}}}}))
    client = _client(recorder)

    assert client.index_meta(_INDEX) == meta
    assert str(recorder.requests[0].url) == f"{_OPENSEARCH_URL}/{_INDEX}/_mapping"


def test_an_index_without_meta_is_rejected() -> None:
    recorder = _Recorder(_json_response({_INDEX: {"mappings": {"properties": {}}}}))
    client = _client(recorder)

    with pytest.raises(OpenSearchUnexpectedResponse, match="no _meta block"):
        client.index_meta(_INDEX)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_alias_lookup_returns_the_sorted_targets() -> None:
    recorder = _Recorder(_json_response({"index-b": {"aliases": {}}, "index-a": {"aliases": {}}}))
    client = _client(recorder)

    assert client.alias_targets(_ALIAS) == ("index-a", "index-b")
    assert str(recorder.requests[0].url) == f"{_OPENSEARCH_URL}/_alias/{_ALIAS}"


def test_a_missing_alias_is_an_empty_target_set() -> None:
    recorder = _Recorder(httpx2.Response(404, content=b'{"error":{"type":"alias_not_found"}}'))
    client = _client(recorder)

    assert client.alias_targets(_ALIAS) == ()


def test_the_alias_switch_is_one_atomic_request() -> None:
    recorder = _Recorder(_json_response({"acknowledged": True}))
    client = _client(recorder)

    client.switch_alias(_ALIAS, index=_INDEX, remove=["old-a", "old-b", _INDEX])

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{_OPENSEARCH_URL}/_aliases"
    # One request carries the removals and the addition, so the alias is never
    # momentarily absent. The already-correct target is not removed.
    assert json.loads(request.content) == {
        "actions": [
            {"remove": {"index": "old-a", "alias": _ALIAS}},
            {"remove": {"index": "old-b", "alias": _ALIAS}},
            {"add": {"index": _INDEX, "alias": _ALIAS}},
        ]
    }


# ---------------------------------------------------------------------------
# Bulk indexing
# ---------------------------------------------------------------------------


def _documents(count: int) -> Sequence[tuple[str, Mapping[str, Any]]]:
    return tuple(
        (f"passage-{index}", {"passage_key": f"passage-{index}", "text": f"body {index}"})
        for index in range(count)
    )


def test_the_bulk_body_is_exact_ndjson_with_the_deterministic_document_id() -> None:
    recorder = _Recorder(_json_response(_BULK_OK))
    client = _client(recorder)

    indexed = client.bulk_index(_INDEX, _documents(2), batch_size=500)

    request = recorder.requests[0]
    assert indexed == 2
    assert str(request.url) == f"{_OPENSEARCH_URL}/_bulk?refresh=wait_for"
    assert request.headers["content-type"] == "application/x-ndjson"
    assert request.content == (
        b'{"index":{"_id":"passage-0","_index":"dynamisrag-passages-passage-index-v1-0a1b2c3d4e5f"}}\n'
        b'{"passage_key":"passage-0","text":"body 0"}\n'
        b'{"index":{"_id":"passage-1","_index":"dynamisrag-passages-passage-index-v1-0a1b2c3d4e5f"}}\n'
        b'{"passage_key":"passage-1","text":"body 1"}\n'
    )
    assert request.content.endswith(b"\n")


def test_batching_is_deterministic_and_configured() -> None:
    recorder = _Recorder(_json_response(_BULK_OK), _json_response(_BULK_OK))
    client = _client(recorder)

    indexed = client.bulk_index(_INDEX, _documents(3), batch_size=2)

    assert indexed == 3
    assert len(recorder.requests) == 2
    first, second = (request.content for request in recorder.requests)
    assert first.count(b"\n") == 4  # two action/document pairs
    assert second.count(b"\n") == 2  # the remaining pair
    assert first.decode("utf-8").startswith('{"index":{"_id":"passage-0"')


def test_an_empty_projection_sends_no_bulk_request_at_all() -> None:
    recorder = _Recorder()
    client = _client(recorder)

    assert client.bulk_index(_INDEX, (), batch_size=500) == 0
    assert recorder.requests == []


def test_a_bulk_response_reporting_errors_is_rejected_even_though_http_is_200() -> None:
    failure = {
        "errors": True,
        "items": [
            {"index": {"status": 201}},
            {
                "index": {
                    "_index": _INDEX,
                    "status": 400,
                    "error": {
                        "type": "mapper_parsing_exception",
                        "reason": "failed to parse field [text] of type [text]",
                        "caused_by": {"type": "x", "reason": "internal detail"},
                    },
                }
            },
        ],
    }
    recorder = _Recorder(_json_response(failure))
    client = _client(recorder)

    with pytest.raises(OpenSearchBulkError) as caught:
        client.bulk_index(_INDEX, _documents(2), batch_size=500)

    message = str(caught.value)
    assert f"bulk_index on {_INDEX} failed" in message
    assert "1 of 2 items rejected" in message
    assert "status=400" in message
    assert "type=mapper_parsing_exception" in message
    # Neither the rejection reason nor the rejected document may appear: for
    # this projection the document is article text.
    assert "failed to parse field" not in message
    assert "internal detail" not in message
    assert "body 1" not in message
    assert UNIT_TEST_PASSWORD not in message
    # The structured context survives the sanitisation.
    assert caught.value.status_code == 400
    assert caught.value.error_type == "mapper_parsing_exception"
    assert caught.value.target == _INDEX
    assert caught.value.operation == "bulk_index"


def test_a_bulk_batch_size_below_one_is_rejected_before_any_request() -> None:
    recorder = _Recorder()
    client = _client(recorder)

    with pytest.raises(ValueError, match="batch size must be >= 1"):
        client.bulk_index(_INDEX, _documents(1), batch_size=0)

    assert recorder.requests == []


# ---------------------------------------------------------------------------
# Search transport
# ---------------------------------------------------------------------------


def test_search_sends_the_exact_body_to_the_named_target() -> None:
    recorder = _Recorder(_json_response({"hits": {"hits": [], "total": {"value": 0}}, "took": 1}))
    client = _client(recorder)
    body: Mapping[str, Any] = {"query": {"match_all": {}}, "size": 3}

    client.search(_ALIAS, body)

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{_OPENSEARCH_URL}/{_ALIAS}/_search"
    assert json.loads(request.content) == body


# ---------------------------------------------------------------------------
# Typed, safe errors
# ---------------------------------------------------------------------------


def test_a_transport_failure_becomes_a_typed_error_without_credentials() -> None:
    def failing(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused")

    client = OpenSearchClient(_SETTINGS, transport=httpx2.MockTransport(failing))

    with pytest.raises(OpenSearchTransportError) as caught:
        client.node_root()

    assert "TransportError: ConnectError" in str(caught.value)
    assert UNIT_TEST_PASSWORD not in str(caught.value)
    assert caught.value.operation == "node_root"
    assert caught.value.cause == "ConnectError"


def test_a_transport_failure_never_relays_the_exceptions_own_message() -> None:
    """Only the exception *class* crosses the boundary.

    A transport message is assembled from the URL, the peer and the socket
    layer, none of which this process wrote, so it is dropped rather than
    truncated.
    """

    def failing(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(
            f"all connection attempts failed for {UNIT_TEST_PASSWORD} at _INDEX"
        )

    client = OpenSearchClient(_SETTINGS, transport=httpx2.MockTransport(failing))

    with pytest.raises(OpenSearchTransportError) as caught:
        client.node_root()

    assert "ConnectError" in caught.value.safe_summary()
    assert "connection attempts" not in str(caught.value)
    assert UNIT_TEST_PASSWORD not in str(caught.value)
    assert UNIT_TEST_PASSWORD not in caught.value.safe_summary()


def test_a_timeout_becomes_a_transport_error() -> None:
    def timing_out(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out")

    client = OpenSearchClient(_SETTINGS, transport=httpx2.MockTransport(timing_out))

    with pytest.raises(OpenSearchTransportError, match="ReadTimeout"):
        client.node_root()


def test_a_malformed_json_body_is_rejected() -> None:
    recorder = _Recorder(httpx2.Response(200, content=b"<html>not json</html>"))
    client = _client(recorder)

    with pytest.raises(OpenSearchUnexpectedResponse, match="not valid JSON"):
        client.node_root()


def test_a_json_array_where_an_object_is_required_is_rejected() -> None:
    recorder = _Recorder(_json_response([1, 2, 3]))
    client = _client(recorder)

    with pytest.raises(OpenSearchUnexpectedResponse, match="where an object was required"):
        client.node_root()


def test_an_unexpected_status_reports_the_error_type_but_never_the_reason() -> None:
    """``error.type`` is carried; ``error.reason`` is untrusted and is not.

    A reason states what the node rejected, so it can echo a credential, a
    field value, the query or the document. There is no way to sanitise it
    after the fact, so it is never read.
    """
    body = json.dumps(
        {
            "error": {
                "type": "index_not_found_exception",
                "reason": f"no such index [{_INDEX}], with {UNIT_TEST_PASSWORD} leaked",
                "caused_by": {"type": "inner", "reason": "very long internal stack"},
            }
        }
    ).encode()
    recorder = _Recorder(httpx2.Response(404, content=body))
    client = _client(recorder)

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        client.count(_INDEX)

    message = str(caught.value)
    assert "UnexpectedStatus: HTTP 404" in message
    assert f"while performing count on {_INDEX}" in message
    assert "type=index_not_found_exception" in message
    assert "very long internal stack" not in message
    assert "no such index" not in message
    assert UNIT_TEST_PASSWORD not in message
    assert "Authorization" not in message
    # The facts an operator needs are structured, not prose.
    assert caught.value.status_code == 404
    assert caught.value.error_type == "index_not_found_exception"
    assert caught.value.operation == "count"
    assert caught.value.target == _INDEX
    assert "UnexpectedStatus" in caught.value.safe_summary()
    assert "HTTP 404" in caught.value.safe_summary()
    assert UNIT_TEST_PASSWORD not in caught.value.safe_summary()


def test_rejected_credentials_are_named_as_such() -> None:
    recorder = _Recorder(httpx2.Response(401, content=b"{}"))
    client = _client(recorder)

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        client.node_root()

    assert str(caught.value).startswith("AuthenticationFailed: HTTP 401")
    assert "opensearch_username" in str(caught.value)
    assert UNIT_TEST_PASSWORD not in str(caught.value)


def test_every_typed_error_is_catchable_as_the_base_error() -> None:
    """The hierarchy is the point: a caller that only wants "the backend
    misbehaved" catches one type."""
    for error in (
        OpenSearchTransportError("t", operation="o"),
        OpenSearchUnexpectedResponse("u", operation="o"),
        OpenSearchBulkError("b", operation="o"),
    ):
        assert isinstance(error, OpenSearchError)
        assert error.operation == "o"


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["dynamisrag-passages", "a", "a.b_c-d", "0abc"],
)
def test_valid_opensearch_names_are_accepted(name: str) -> None:
    assert validate_resource_name(name, kind="index") == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "-leading-dash",
        "_leading-underscore",
        "+leading-plus",
        "UPPERCASE",
        "has space",
        "has/slash",
        "..",
        ".",
        "a" * 256,
    ],
)
def test_invalid_opensearch_names_are_rejected_before_any_request(name: str) -> None:
    with pytest.raises(ValueError):
        validate_resource_name(name, kind="index")


def test_an_invalid_index_name_never_reaches_the_node() -> None:
    recorder = _Recorder()
    client = _client(recorder)

    with pytest.raises(ValueError, match="naming restriction"):
        client.create_index("Not Valid", settings={}, mappings={})

    assert recorder.requests == []
