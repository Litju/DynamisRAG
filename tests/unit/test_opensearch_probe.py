"""OpenSearch probe behaviour, exercised through a mocked transport.

Every branch of the probe runs here without a live node: success, both
authentication rejections, an unexpected status, a malformed payload and a
transport-level failure. The integration suite then proves the same code path
against the real OpenSearch 3.x container.
"""

from __future__ import annotations

import json
from typing import Final

import httpx2
import pytest

from dynamisrag.health.models import CheckStatus
from dynamisrag.search.opensearch import OPENSEARCH_DEPENDENCY_NAME, OpenSearchProbe
from tests._support import build_settings, opensearch_root_document, stub_transport

_EXPECTED_VERSION: Final[str] = "3.8.0"
_OPENSEARCH_URL: Final[str] = "https://search.internal:9200"


def _body(version: str = _EXPECTED_VERSION) -> bytes:
    return json.dumps(opensearch_root_document(version)).encode()


def _probe(transport: httpx2.MockTransport, url: str = _OPENSEARCH_URL) -> OpenSearchProbe:
    return OpenSearchProbe(build_settings(opensearch_url=url), transport=transport)


def test_valid_node_document_reports_up_with_the_running_version() -> None:
    probe = _probe(stub_transport(status_code=200, body=_body()))

    check = probe.check()

    assert check.name == OPENSEARCH_DEPENDENCY_NAME
    assert check.status is CheckStatus.UP
    assert check.version == _EXPECTED_VERSION
    assert check.detail is None
    assert check.latency_ms >= 0
    probe.close()


def test_probe_requests_the_configured_root_url_with_basic_auth() -> None:
    seen: list[httpx2.Request] = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, content=_body())

    probe = _probe(httpx2.MockTransport(capture))
    probe.check()
    probe.close()

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
    probe.close()

    assert seen == ["https://gateway.internal/opensearch/"]


@pytest.mark.parametrize("status_code", [401, 403])
def test_rejected_credentials_report_an_authentication_failure(status_code: int) -> None:
    probe = _probe(stub_transport(status_code=status_code, body=b"{}"))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("AuthenticationFailed")
    assert str(status_code) in check.detail
    probe.close()


def test_unexpected_status_is_reported_verbatim() -> None:
    probe = _probe(stub_transport(status_code=503, body=b"{}"))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail == "UnexpectedStatus: HTTP 503"
    probe.close()


def test_malformed_payload_is_reported_as_an_unexpected_payload() -> None:
    probe = _probe(stub_transport(status_code=200, body=b'{"cluster_name": "c"}'))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("UnexpectedPayload")
    probe.close()


def test_transport_failure_is_reported_without_raising() -> None:
    probe = _probe(stub_transport(raises=httpx2.ConnectError("connection refused")))

    check = probe.check()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("TransportError: ConnectError")
    assert "connection refused" in check.detail
    probe.close()


def test_check_is_idempotent_and_reusable() -> None:
    probe = _probe(stub_transport(status_code=200, body=_body()))

    assert [probe.check().status for _ in range(3)] == [CheckStatus.UP] * 3
    probe.close()
