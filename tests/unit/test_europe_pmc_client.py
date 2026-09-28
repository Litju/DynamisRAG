"""Unit tests for the Europe PMC full-text acquisition client.

Every HTTP behaviour is proven against a deterministic recording transport —
no socket is ever opened. The tests cover PMCID validation before HTTP, the
exact request issued, byte-exact payload preservation, the error surface for
unavailable/unexpected responses, transport failures, and the narrow license
extraction over minimal synthetic JATS snippets (never a copyrighted article).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Final

import httpx2
import pytest

from dynamisrag.ingestion.europe_pmc import (
    EuropePmcClient,
    EuropePmcError,
    EuropePmcInvalidPmcid,
    EuropePmcNotAvailable,
    EuropePmcUnexpectedResponse,
)

_PMCID: Final[str] = "PMC3253803"
_BASE_URL: Final[str] = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_FULLTEXT_URL: Final[str] = f"{_BASE_URL}/{_PMCID}/fullTextXML"

_JATS_WITH_EXPLICIT_LICENSE: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <permissions>
        <license license-type="CC BY" xlink:href="http://creativecommons.org/licenses/by/4.0/">
          <license-p>Distributed under the Creative Commons Attribution 4.0 License.</license-p>
        </license>
      </permissions>
    </article-meta>
  </front>
</article>
"""

_JATS_WITH_NESTED_EXTR_LINK: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <permissions>
        <license>
          <ext-link ext-link-type="uri" xlink:href="http://creativecommons.org/licenses/by-nc/4.0/">
            http://creativecommons.org/licenses/by-nc/4.0/
          </ext-link>
        </license>
      </permissions>
    </article-meta>
  </front>
</article>
"""

_JATS_WITHOUT_EXPLICIT_LICENSE: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <permissions>
        <copyright-statement>Example et al.</copyright-statement>
        <license>
          <license-p>This is an open-access article.</license-p>
        </license>
      </permissions>
    </article-meta>
  </front>
</article>
"""

_WHITESPACE_SENSITIVE_PAYLOAD: Final[bytes] = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n<article>\n  <body>\n'
    b"    <p>  Indentation and\n    newlines are part of the entity.  </p>\n"
    b"  </body>\n</article>\n"
)
"""A payload whose exact bytes — whitespace and newlines included — must be
preserved so any accidental normalization changes the artifact identity."""

_EXPECTED_SHA256: Final[str] = hashlib.sha256(_WHITESPACE_SENSITIVE_PAYLOAD).hexdigest()


class _RecordingTransport(httpx2.BaseTransport):
    """A deterministic transport seam that records every request."""

    def __init__(
        self,
        responder: Callable[[httpx2.Request], httpx2.Response] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self._responder = responder
        self._raises = raises
        self.requests: list[httpx2.Request] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self._raises is not None:
            raise self._raises
        assert self._responder is not None
        return self._responder(request)


def _client(transport: _RecordingTransport) -> EuropePmcClient:
    return EuropePmcClient(transport=transport)


def _responding(
    body: bytes,
    *,
    status_code: int = 200,
    content_type: str | None = "application/xml",
) -> Callable[[httpx2.Request], httpx2.Response]:
    def responder(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        headers = {"content-type": content_type} if content_type is not None else {}
        return httpx2.Response(status_code, content=body, headers=headers, request=request)

    return responder


# ---------------------------------------------------------------------------
# PMCID validation
# ---------------------------------------------------------------------------


def test_valid_canonical_pmcid_is_accepted() -> None:
    transport = _RecordingTransport(_responding(_JATS_WITH_EXPLICIT_LICENSE))

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert payload.pmcid == _PMCID
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "malformed",
    ["3253803", "pmc3253803", "PMC", "PMC3253803extra", "PMC 3253803", "PMC-3253803", ""],
)
def test_malformed_pmcid_is_rejected_before_any_http(malformed: str) -> None:
    transport = _RecordingTransport(_responding(_JATS_WITH_EXPLICIT_LICENSE))

    with pytest.raises(EuropePmcInvalidPmcid, match="invalid PMCID"):
        _client(transport).fetch_fulltext(malformed)

    assert transport.requests == []


# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------


def test_fulltext_request_targets_the_canonical_operation() -> None:
    transport = _RecordingTransport(_responding(_JATS_WITH_EXPLICIT_LICENSE))

    payload = _client(transport).fetch_fulltext(_PMCID)

    request = transport.requests[0]
    assert str(request.url) == _FULLTEXT_URL
    assert payload.source_uri == _FULLTEXT_URL
    assert request.method == "GET"


def test_request_sends_the_expected_headers() -> None:
    transport = _RecordingTransport(_responding(_JATS_WITH_EXPLICIT_LICENSE))

    _client(transport).fetch_fulltext(_PMCID)

    headers = transport.requests[0].headers
    assert headers["accept"] == "application/xml"
    assert headers["accept-encoding"] == "identity"
    assert headers["user-agent"].startswith("DynamisRAG/")


def test_exact_response_bytes_are_preserved() -> None:
    transport = _RecordingTransport(_responding(_WHITESPACE_SENSITIVE_PAYLOAD))

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert payload.content == _WHITESPACE_SENSITIVE_PAYLOAD


def test_known_bytes_hash_and_size_match_exactly() -> None:
    """Given known bytes the acquired content's SHA-256 and byte count must
    match exactly — a payload where whitespace and newlines matter, so any
    accidental normalization would fail this test."""
    transport = _RecordingTransport(_responding(_WHITESPACE_SENSITIVE_PAYLOAD))

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert hashlib.sha256(payload.content).hexdigest() == _EXPECTED_SHA256
    assert len(payload.content) == len(_WHITESPACE_SENSITIVE_PAYLOAD)


def test_xml_content_type_is_normalized() -> None:
    transport = _RecordingTransport(
        _responding(_JATS_WITH_EXPLICIT_LICENSE, content_type="application/xml; charset=utf-8")
    )

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert payload.media_type == "application/xml"


@pytest.mark.parametrize("status_code", [400, 403, 500, 503])
def test_non_2xx_responses_are_handled(status_code: int) -> None:
    transport = _RecordingTransport(
        _responding(_JATS_WITH_EXPLICIT_LICENSE, status_code=status_code)
    )

    with pytest.raises(EuropePmcUnexpectedResponse, match=f"HTTP {status_code}"):
        _client(transport).fetch_fulltext(_PMCID)


def test_article_not_in_oa_subset_is_not_available() -> None:
    transport = _RecordingTransport(_responding(b"", status_code=404))

    with pytest.raises(EuropePmcNotAvailable, match="not available"):
        _client(transport).fetch_fulltext(_PMCID)


@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ConnectTimeout("connect timed out"),
        httpx2.ReadTimeout("read timed out"),
        httpx2.ConnectError("connection refused"),
        httpx2.NetworkError("network unreachable"),
    ],
)
def test_timeout_and_network_failures_become_acquisition_errors(
    failure: Exception,
) -> None:
    transport = _RecordingTransport(raises=failure)

    with pytest.raises(EuropePmcError, match="PMC3253803"):
        _client(transport).fetch_fulltext(_PMCID)


def test_empty_payload_is_rejected() -> None:
    transport = _RecordingTransport(_responding(b""))

    with pytest.raises(EuropePmcUnexpectedResponse, match="empty body"):
        _client(transport).fetch_fulltext(_PMCID)


def test_non_xml_content_type_is_rejected() -> None:
    transport = _RecordingTransport(
        _responding(b"<html>error</html>", content_type="text/html; charset=utf-8")
    )

    with pytest.raises(EuropePmcUnexpectedResponse, match="non-XML content type"):
        _client(transport).fetch_fulltext(_PMCID)


def test_missing_content_type_is_rejected() -> None:
    """HTTP 200 with well-formed XML but no Content-Type header must fail
    explicitly: the source media type is required provenance and is never
    invented by acquisition."""
    transport = _RecordingTransport(_responding(_JATS_WITH_EXPLICIT_LICENSE, content_type=None))

    with pytest.raises(EuropePmcUnexpectedResponse, match="no Content-Type"):
        _client(transport).fetch_fulltext(_PMCID)


def test_malformed_xml_body_is_rejected() -> None:
    transport = _RecordingTransport(_responding(b"<article><unclosed>"))

    with pytest.raises(EuropePmcUnexpectedResponse, match="not well-formed XML"):
        _client(transport).fetch_fulltext(_PMCID)


# ---------------------------------------------------------------------------
# License extraction
# ---------------------------------------------------------------------------


def test_explicit_license_uri_and_type_are_captured() -> None:
    transport = _RecordingTransport(_responding(_JATS_WITH_EXPLICIT_LICENSE))

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert payload.license_uri == "http://creativecommons.org/licenses/by/4.0/"
    assert payload.license_name == "CC BY"


def test_nested_ext_link_uri_is_captured() -> None:
    transport = _RecordingTransport(_responding(_JATS_WITH_NESTED_EXTR_LINK))

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert payload.license_uri == "http://creativecommons.org/licenses/by-nc/4.0/"
    assert payload.license_name is None


def test_article_without_explicit_license_is_legitimate_none_provenance() -> None:
    transport = _RecordingTransport(_responding(_JATS_WITHOUT_EXPLICIT_LICENSE))

    payload = _client(transport).fetch_fulltext(_PMCID)

    assert payload.license_name is None
    assert payload.license_uri is None
