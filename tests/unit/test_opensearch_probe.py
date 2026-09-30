"""OpenSearch probe behaviour, exercised through a mocked transport.

Every branch of the probe runs here without a live node: success, both
authentication rejections, an unexpected status, a malformed payload and a
transport-level failure. The integration suite then proves the same code path
against the real OpenSearch 3.x container.

The probe owns no transport: it is built over the shared
:class:`~dynamisrag.search.client.OpenSearchClient`, so these tests also pin the
summary the probe renders from that client's typed errors — the string that
reaches ``/readyz`` as ``dependencies.opensearch.detail``.
"""

from __future__ import annotations

import json
from typing import Final

import httpx2
import pytest
from pydantic import SecretStr

from dynamisrag.config import Settings
from dynamisrag.health.models import CheckStatus
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.opensearch import OPENSEARCH_DEPENDENCY_NAME, OpenSearchProbe
from tests._support import (
    UNIT_TEST_PASSWORD,
    build_settings,
    opensearch_root_document,
    stub_transport,
)

_EXPECTED_VERSION: Final[str] = "3.8.0"
_OPENSEARCH_URL: Final[str] = "https://search.internal:9200"


def _body(version: str = _EXPECTED_VERSION) -> bytes:
    return json.dumps(opensearch_root_document(version)).encode()


def _probe(transport: httpx2.MockTransport, url: str = _OPENSEARCH_URL) -> OpenSearchProbe:
    return OpenSearchProbe(
        OpenSearchClient(build_settings(opensearch_url=url), transport=transport)
    )


def test_valid_node_document_reports_up_with_the_running_version() -> None:
    probe = _probe(stub_transport(status_code=200, body=_body()))

    check = probe.check()

    assert check.name == OPENSEARCH_DEPENDENCY_NAME
    assert check.status is CheckStatus.UP
    assert check.version == _EXPECTED_VERSION
    assert check.detail is None
    assert check.latency_ms >= 0


def test_probe_requests_the_configured_root_url_with_basic_auth() -> None:
    seen: list[httpx2.Request] = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, content=_body())

    probe = _probe(httpx2.MockTransport(capture))
    probe.check()

    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == f"{_OPENSEARCH_URL}/"
    assert request.method == "GET"
    assert request.headers["authorization"].startswith("Basic ")


def test_a_url_prefix_is_preserved() -> None:
    seen: list[str] = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        return httpx2.Response(200, content=_body())

    probe = _probe(httpx2.MockTransport(capture), url="https://gateway.internal/opensearch")
    probe.check()

    assert seen == ["https://gateway.internal/opensearch/"]


@pytest.mark.parametrize("status_code", [401, 403])
def test_rejected_credentials_report_an_authentication_failure(status_code: int) -> None:
    probe = _probe(stub_transport(status_code=status_code, body=b"{}"))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("AuthenticationFailed")
    assert str(status_code) in check.detail


def test_unexpected_status_is_reported_as_a_safe_summary() -> None:
    probe = _probe(stub_transport(status_code=503, body=b"{}"))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("UnexpectedStatus")
    assert "HTTP 503" in check.detail
    assert "node_root" in check.detail


def test_an_unexpected_status_reports_the_error_type_but_never_the_reason() -> None:
    """``/readyz`` is a public endpoint.

    OpenSearch's ``error.type`` is a machine-generated class name and is safe
    to carry. Its ``error.reason`` and the ``reason`` of a ``caused_by`` are
    the node's own prose about what it rejected, and are never relayed.
    """
    body = json.dumps(
        {
            "error": {
                "type": "illegal_argument_exception",
                "reason": f"request [/index] is missing, password {UNIT_TEST_PASSWORD}",
                "caused_by": {"type": "nope", "reason": "internal detail"},
            },
            "status": 400,
        }
    ).encode()
    probe = _probe(stub_transport(status_code=400, body=body))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("UnexpectedStatus")
    assert "HTTP 400" in check.detail
    assert "error.type=illegal_argument_exception" in check.detail
    assert "node_root" in check.detail
    assert "request [/index] is missing" not in check.detail
    assert "internal detail" not in check.detail
    assert UNIT_TEST_PASSWORD not in check.detail


def test_malformed_payload_is_reported_as_an_unexpected_payload() -> None:
    probe = _probe(stub_transport(status_code=200, body=b'{"cluster_name": "c"}'))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("UnexpectedPayload")


def test_a_body_that_is_not_json_is_reported_as_an_unexpected_payload() -> None:
    probe = _probe(stub_transport(status_code=200, body=b"<html>gateway</html>"))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail is not None
    assert check.detail.startswith("UnexpectedPayload")


def test_transport_failure_is_reported_without_raising() -> None:
    probe = _probe(stub_transport(raises=httpx2.ConnectError("connection refused")))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("TransportError")
    # The exception class is kept — it distinguishes refused from timed out —
    # but its own message, which is not written by this process, is not.
    assert "cause=ConnectError" in check.detail
    assert "operation=node_root" in check.detail
    assert "connection refused" not in check.detail


def test_a_transport_failure_never_echoes_the_password() -> None:
    """A connect failure is the case where an implementation is most tempted to
    dump the request; the shared client must not."""
    settings: Settings = build_settings(opensearch_url=_OPENSEARCH_URL)
    probe = OpenSearchProbe(
        OpenSearchClient(
            settings, transport=stub_transport(raises=httpx2.ConnectError("connection refused"))
        )
    )

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert UNIT_TEST_PASSWORD not in check.detail
    assert "Basic" not in check.detail


def test_check_is_idempotent_and_reusable() -> None:
    probe = _probe(stub_transport(status_code=200, body=_body()))

    assert [probe.check().status for _ in range(3)] == [CheckStatus.UP] * 3


def test_a_shared_client_is_reused_rather_than_recreated() -> None:
    """The probe never owns its transport, so closing the client once is
    enough: there is no second pool to leak."""
    client = OpenSearchClient(
        build_settings(opensearch_url=_OPENSEARCH_URL),
        transport=stub_transport(status_code=200, body=_body()),
    )
    probe = OpenSearchProbe(client)

    assert probe.check().status is CheckStatus.UP
    assert probe.check().status is CheckStatus.UP
    client.close()


def test_wrong_credentials_produce_an_authentication_failure_not_a_crash() -> None:
    settings = build_settings(opensearch_url=_OPENSEARCH_URL).model_copy(
        update={"opensearch_password": SecretStr("definitely-not-the-password-1A")}
    )
    probe = OpenSearchProbe(
        OpenSearchClient(settings, transport=stub_transport(status_code=401, body=b"{}"))
    )

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("AuthenticationFailed")
    assert "definitely-not-the-password-1A" not in check.detail
