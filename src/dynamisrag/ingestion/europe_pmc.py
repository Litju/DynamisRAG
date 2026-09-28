"""Europe PMC full-text acquisition client (RES-132).

Fetches one open-access article's JATS XML from the official Europe PMC REST
``/{id}/fullTextXML`` operation and returns the exact response bytes plus the
narrow provenance metadata acquisition needs: the normalized XML media type
and any *explicit* license metadata in the returned JATS.

The client is deliberately narrow. It validates the PMCID against the
canonical value contract before any HTTP, requests an identity (uncompressed)
response so the acquired bytes are the exact wire bytes, and never parses
scientific content — JATS parsing is RES-133. The only XML inspection here is
the minimum required to identify explicit license metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from xml.etree.ElementTree import Element, ParseError

import httpx2
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring as safe_fromstring
from pydantic import TypeAdapter, ValidationError

from dynamisrag.domain.values import Pmcid

__all__ = [
    "AcquiredFulltext",
    "EuropePmcClient",
    "EuropePmcError",
    "EuropePmcInvalidPmcid",
    "EuropePmcNotAvailable",
    "EuropePmcUnexpectedResponse",
]

_BASE_URL: Final[str] = "https://www.ebi.ac.uk/europepmc/webservices/rest"
"""Official Europe PMC production REST service."""

_USER_AGENT: Final[str] = "DynamisRAG/0.1 (+https://github.com/Litju/DynamisRAG)"
"""Stable User-Agent identifying DynamisRAG."""

_XML_MEDIA_TYPES: Final[frozenset[str]] = frozenset({"application/xml", "text/xml"})
"""Media types that unambiguously declare an XML payload."""

_XML_SUFFIX: Final[str] = "+xml"
"""Suffix marking XML-based media types such as ``application/atom+xml``."""

_XLINK_HREF: Final[str] = "{http://www.w3.org/1999/xlink}href"
"""The ``xlink:href`` attribute in its expanded ElementTree form."""

_PMCID_ADAPTER: Final[TypeAdapter[Pmcid]] = TypeAdapter(Pmcid)
"""The canonical PMCID value contract, reused for pre-HTTP validation."""


class EuropePmcError(Exception):
    """Base error for Europe PMC acquisition failures."""


class EuropePmcInvalidPmcid(EuropePmcError):
    """A malformed PMCID was rejected before any HTTP request was made."""


class EuropePmcNotAvailable(EuropePmcError):
    """The article is not available through the OA full-text service (HTTP 404)."""


class EuropePmcUnexpectedResponse(EuropePmcError):
    """The response cannot be trusted as the article's exact JATS bytes."""


@dataclass(frozen=True)
class AcquiredFulltext:
    """The exact acquired bytes plus the narrow provenance metadata.

    ``content`` is the exact HTTP body — never parsed, reserialized or
    normalized — so the acquisition hash represents the actual acquired
    entity. ``license_name``/``license_uri`` are ``None`` unless the returned
    JATS explicitly carries them; that absence is legitimate provenance.
    """

    pmcid: str
    source_uri: str
    content: bytes
    media_type: str
    license_name: str | None
    license_uri: str | None


class EuropePmcClient:
    """Narrow client for the Europe PMC ``fullTextXML`` operation."""

    def __init__(
        self,
        *,
        transport: httpx2.BaseTransport | None = None,
        timeout_seconds: float = 30.0,
        base_url: str = _BASE_URL,
    ) -> None:
        """Build a client; ``transport`` is a seam for deterministic tests."""
        self._base_url = base_url
        self._client = httpx2.Client(
            transport=transport,
            timeout=httpx2.Timeout(timeout_seconds, connect=timeout_seconds),
            follow_redirects=True,
            trust_env=False,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/xml",
                "Accept-Encoding": "identity",
            },
        )

    def fetch_fulltext(self, pmcid: str) -> AcquiredFulltext:
        """Fetch one article's exact JATS bytes by PMCID.

        Raises :class:`EuropePmcInvalidPmcid` before any HTTP when the
        identifier is malformed, :class:`EuropePmcNotAvailable` when the
        article is not in the OA full-text subset, and
        :class:`EuropePmcUnexpectedResponse` for any response that cannot be
        trusted as the article's exact JATS bytes.
        """
        canonical = self._validate_pmcid(pmcid)
        url = f"{self._base_url}/{canonical}/fullTextXML"
        response = self._get(url, canonical)
        content = self._exact_bytes(response, canonical)
        media_type = self._media_type(response, canonical)
        root = self._parse_xml(content, canonical)
        license_name, license_uri = _extract_license(root)
        return AcquiredFulltext(
            pmcid=canonical,
            source_uri=url,
            content=content,
            media_type=media_type,
            license_name=license_name,
            license_uri=license_uri,
        )

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def _validate_pmcid(self, pmcid: str) -> str:
        try:
            return str(_PMCID_ADAPTER.validate_python(pmcid))
        except ValidationError as error:
            raise EuropePmcInvalidPmcid(
                f"invalid PMCID {pmcid!r}: {error.error_count()} validation error(s)"
            ) from error

    def _get(self, url: str, canonical: str) -> httpx2.Response:
        try:
            return self._client.get(url)
        except httpx2.TimeoutException as error:
            raise EuropePmcError(
                f"Europe PMC request for {canonical} timed out: {type(error).__name__}"
            ) from error
        except httpx2.TransportError as error:
            raise EuropePmcError(
                f"Europe PMC request for {canonical} failed: {type(error).__name__}"
            ) from error

    def _exact_bytes(self, response: httpx2.Response, canonical: str) -> bytes:
        if response.status_code == 404:
            raise EuropePmcNotAvailable(
                f"article {canonical} is not available through the Europe PMC "
                "OA full-text service (HTTP 404)"
            )
        if response.status_code != 200:
            raise EuropePmcUnexpectedResponse(
                f"Europe PMC returned HTTP {response.status_code} for {canonical}"
            )
        content = response.content
        if not content:
            raise EuropePmcUnexpectedResponse(f"Europe PMC returned an empty body for {canonical}")
        return content

    def _media_type(self, response: httpx2.Response, canonical: str) -> str:
        declared = response.headers.get("content-type")
        if declared is None:
            raise EuropePmcUnexpectedResponse(
                f"Europe PMC returned no Content-Type for {canonical}; the source media "
                "type is required provenance and is never invented"
            )
        media_type = declared.split(";", 1)[0].strip().lower()
        if media_type not in _XML_MEDIA_TYPES and not media_type.endswith(_XML_SUFFIX):
            raise EuropePmcUnexpectedResponse(
                f"Europe PMC returned non-XML content type {declared!r} for {canonical}"
            )
        return media_type

    def _parse_xml(self, content: bytes, canonical: str) -> Element:
        try:
            return safe_fromstring(content)
        except ParseError as error:
            raise EuropePmcUnexpectedResponse(
                f"Europe PMC response for {canonical} is not well-formed XML: {error}"
            ) from error
        except DefusedXmlException as error:
            raise EuropePmcUnexpectedResponse(
                f"Europe PMC response for {canonical} contains forbidden XML constructs "
                f"({type(error).__name__})"
            ) from error


def _extract_license(root: Element) -> tuple[str | None, str | None]:
    """Extract only the explicit license metadata present in the JATS.

    Looks under ``permissions``/``license`` structures and captures
    ``license/@xlink:href`` (or a descendant ``ext-link/@xlink:href``) as the
    license URI and ``license/@license-type`` as the license name. Nothing is
    inferred from the fact that the OA endpoint succeeded; when no explicit
    license is present both values are ``None``.
    """
    for permissions in root.iter():
        if _local_name(permissions.tag) != "permissions":
            continue
        for candidate in permissions.iter():
            if _local_name(candidate.tag) != "license":
                continue
            return _license_metadata(candidate)
    return None, None


def _license_metadata(license_element: Element) -> tuple[str | None, str | None]:
    uri = license_element.get(_XLINK_HREF)
    if uri is None:
        uri = _ext_link_uri(license_element)
    return license_element.get("license-type") or None, uri


def _ext_link_uri(license_element: Element) -> str | None:
    for descendant in license_element.iter():
        if _local_name(descendant.tag) == "ext-link":
            return descendant.get(_XLINK_HREF)
    return None


def _local_name(tag: str) -> str:
    """Return an ElementTree tag's local name, stripping any namespace."""
    return tag.rsplit("}", 1)[-1]
