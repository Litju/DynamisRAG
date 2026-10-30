"""Bounded official discovery APIs and deterministic strong-identifier reconciliation."""

from __future__ import annotations

import base64
import hashlib
import time
from typing import Annotated, Any, Literal, cast

import httpx2
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError

from dynamisrag.domain.values import NormalizedDoi, Pmcid, Sha256Hex

type Provider = Literal["europe_pmc", "openalex", "crossref"]
PROVIDERS: tuple[Provider, ...] = ("europe_pmc", "openalex", "crossref")
ENDPOINTS = {
    "europe_pmc": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
    "openalex": "https://api.openalex.org/works",
    "crossref": "https://api.crossref.org/works",
}


class DiscoveryPlan(BaseModel):
    """Finite request budget, recorded verbatim in every discovery receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    topics: tuple[Annotated[str, StringConstraints(pattern=r"^[\w -]{3,120}$")], ...] = Field(
        min_length=1, max_length=10
    )
    pages_per_topic: int = Field(default=2, ge=1, le=10)
    page_size: int = Field(default=8, ge=1, le=100)


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    provider: Provider
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    doi: NormalizedDoi | None = None
    pmcid: Pmcid | None = None
    page_sha256: Sha256Hex
    metadata: dict[str, Any]


class BoundedHttp:
    """Three attempts, finite timeout, retry only transient HTTP/transport errors."""

    def __init__(self, client: httpx2.Client, *, attempts: int = 3) -> None:
        if not 1 <= attempts <= 5:
            raise ValueError("attempts must be in [1, 5]")
        self.client = client
        self.attempts = attempts

    def get(self, url: str, params: dict[str, str | int]) -> httpx2.Response:
        for attempt in range(self.attempts):
            try:
                response = self.client.get(url, params=params, timeout=40)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    return response
                response.raise_for_status()
            except (httpx2.TransportError, httpx2.HTTPStatusError) as error:
                transient = not isinstance(error, httpx2.HTTPStatusError) or (
                    error.response.status_code in {429, 500, 502, 503, 504}
                )
                if not transient or attempt + 1 == self.attempts:
                    raise
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError("unreachable retry state")


def normalize_record(provider: Provider, raw: dict[str, Any], page_sha256: str) -> Candidate:
    """Reject malformed metadata; no title-based identity or invented identifiers."""
    if provider == "europe_pmc":
        source_id, title = raw.get("id"), raw.get("title")
        doi, pmcid = raw.get("doi"), raw.get("pmcid")
    elif provider == "openalex":
        source_id, title = raw.get("id"), raw.get("display_name")
        ids = TypeAdapter(dict[str, Any]).validate_python(raw.get("ids") or {})
        doi, pmcid = ids.get("doi"), ids.get("pmcid")
        if isinstance(pmcid, str):
            pmcid = pmcid.rstrip("/").rsplit("/", 1)[-1]
            if pmcid.isdecimal():
                pmcid = f"PMC{pmcid}"
    else:
        source_id, doi, pmcid = raw.get("DOI"), raw.get("DOI"), None
        titles = TypeAdapter(list[str]).validate_python(raw.get("title", []))
        title = titles[0] if titles else None
    candidate = Candidate.model_validate(
        {
            "provider": provider,
            "source_id": source_id,
            "title": title,
            "doi": doi,
            "pmcid": pmcid,
            "page_sha256": page_sha256,
            "metadata": raw,
        }
    )
    if candidate.doi is None and candidate.pmcid is None:
        raise ValueError("no strong identifier")
    return candidate


def discover(http: BoundedHttp, plan: DiscoveryPlan) -> dict[str, Any]:
    """Freeze exact pages, queries and record-level rejections for offline replay.

    Provider failures are explicit and do not erase successful providers. A
    malformed page is a provider failure, never an empty successful search.
    """
    pages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for topic in plan.topics:
            cursor = "*"
            seen: set[str] = set()
            for page in range(plan.pages_per_topic):
                params = _params(provider, topic, plan.page_size, cursor, page)
                try:
                    response = http.get(ENDPOINTS[provider], params)
                    payload: dict[str, Any] = response.json()
                    items, next_cursor = parse_page(provider, payload)
                    if len(items) > plan.page_size:
                        raise ValueError("invalid or oversized result page")
                    digest = hashlib.sha256(response.content).hexdigest()
                    pages.append(
                        {
                            "provider": provider,
                            "topic": topic,
                            "page": page + 1,
                            "url": str(response.request.url),
                            "sha256": digest,
                            "body_base64": base64.b64encode(response.content).decode("ascii"),
                        }
                    )
                    for raw in items:
                        try:
                            if not isinstance(raw, dict):
                                raise ValueError("record is not an object")
                            candidate = normalize_record(
                                provider, cast(dict[str, Any], raw), digest
                            )
                            records.append(candidate.model_dump(mode="json"))
                        except (ValidationError, ValueError, TypeError, AttributeError) as error:
                            rejected.append(
                                {
                                    "provider": provider,
                                    "page_sha256": digest,
                                    "record": raw,
                                    "reason": str(error),
                                }
                            )
                    if provider == "openalex":
                        next_cursor = str(page + 2)
                    if (
                        len(items) < plan.page_size
                        or not next_cursor
                        or next_cursor == cursor
                        or next_cursor in seen
                    ):
                        break
                    seen.add(cursor)
                    cursor = next_cursor
                except (httpx2.HTTPError, ValueError, KeyError, TypeError) as error:
                    failures.append(
                        {
                            "provider": provider,
                            "topic": topic,
                            "page": page + 1,
                            "url": page_request_url(provider, topic, plan.page_size, cursor, page),
                            "error": str(error),
                        }
                    )
                    break
    return {
        "schema_revision": "discovery-2",
        "plan": plan.model_dump(mode="json"),
        "pages": pages,
        "records": records,
        "rejected": rejected,
        "failures": failures,
    }


def _params(
    provider: Provider, topic: str, size: int, cursor: str, page: int
) -> dict[str, str | int]:
    if provider == "europe_pmc":
        return {
            "query": f'TITLE_ABS:"{topic}" AND OPEN_ACCESS:Y',
            "format": "json",
            "resultType": "core",
            "pageSize": size,
            "cursorMark": cursor,
        }
    if provider == "openalex":
        # Bounded basic pagination works without following API-key-only cursors.
        return {"search": topic, "per_page": size, "page": page + 1}
    return {"query": topic, "rows": size, "cursor": cursor, "filter": "type:journal-article"}


def page_request_url(provider: Provider, topic: str, size: int, cursor: str, page: int) -> str:
    """Derive the exact URL issued by the bounded discovery request."""
    return str(
        httpx2.Request(
            "GET", ENDPOINTS[provider], params=_params(provider, topic, size, cursor, page)
        ).url
    )


def parse_page(provider: Provider, payload: dict[str, Any]) -> tuple[list[Any], str | None]:
    if provider == "europe_pmc":
        items, cursor = payload["resultList"]["result"], payload.get("nextCursorMark")
    elif provider == "openalex":
        items, cursor = payload["results"], None
    else:
        items, cursor = payload["message"]["items"], payload["message"].get("next-cursor")
    if not isinstance(items, list) or (cursor is not None and not isinstance(cursor, str)):
        raise ValueError("malformed result list or cursor")
    return cast(list[Any], items), cursor


def reconcile(records: list[Candidate]) -> tuple[list[list[Candidate]], list[list[Candidate]]]:
    """Connected DOI/PMCID components. Quarantine the entire conflicting component."""
    groups: list[list[Candidate]] = []
    for candidate in sorted(records, key=lambda c: (c.provider, c.source_id, c.page_sha256)):
        keys = _keys(candidate)
        matches = [g for g in groups if any(keys & _keys(c) for c in g)]
        merged = [candidate]
        for group in matches:
            merged.extend(group)
            groups.remove(group)
        groups.append(merged)
    accepted: list[list[Candidate]] = []
    conflicts: list[list[Candidate]] = []
    for group in groups:
        dois = {c.doi for c in group if c.doi}
        pmcids = {c.pmcid for c in group if c.pmcid}
        (conflicts if len(dois) > 1 or len(pmcids) > 1 else accepted).append(group)
    return sorted(accepted, key=lambda g: sorted(k for c in g for k in _keys(c))), conflicts


def _keys(candidate: Candidate) -> set[str]:
    return {
        f"{kind}:{value}"
        for kind, value in (("doi", candidate.doi), ("pmcid", candidate.pmcid))
        if value
    }
