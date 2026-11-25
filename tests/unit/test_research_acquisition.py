"""Live literature acquisition (RES-638): query rules, identity linking, HTTP bounds, ingestion.

HTTP is replayed through ``httpx2.MockTransport``; the full text is a packaged CC BY 4.0 JATS
article, so rights, scope and the canonical importer run on real bytes. The Qdrant/TEI
projection is exercised live (see ``docs/evidence/res638-research-workflow``), not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from dynamisrag.corpus.research import (
    AcquisitionError,
    AcquisitionLimits,
    DiscoveredRecord,
    LiteratureAcquisition,
    RateLimitedHttp,
    build_query_ladder,
    link_identities,
)
from dynamisrag.db.models import DocumentVersionRecord, PassageRecord
from dynamisrag.ingestion import EuropePmcClient
from dynamisrag.search.scientific import CHUNKER
from tests.unit.test_agent_tools import (
    _create_sqlite_schema,  # pyright: ignore[reportPrivateUsage]
)

SAMPLE = Path(__file__).resolve().parents[2] / "data/research-edition-sample-v1/sources"
QUESTION = (
    "What does the evidence say about the validity, reliability, practical applications and "
    "limitations of force–velocity profiling using linear position transducers (LPT) in "  # noqa: RUF001
    "resistance training?"
)


def test_query_ladder_anchors_on_the_defined_abbreviation_and_relaxes_in_order() -> None:
    ladder = build_query_ladder(QUESTION)
    anchor = '(TITLE_ABS:"linear position transducer" OR TITLE_ABS:"LPT")'
    assert [q.europe_pmc() for q in ladder] == [
        f'{anchor} AND (TITLE_ABS:"force-velocity") AND '
        '(TITLE_ABS:"validity" OR TITLE_ABS:"reliability") AND OPEN_ACCESS:Y',
        f'{anchor} AND (TITLE_ABS:"validity" OR TITLE_ABS:"reliability") AND OPEN_ACCESS:Y',
        f'{anchor} AND (TITLE_ABS:"force-velocity") AND OPEN_ACCESS:Y',
        f'{anchor} AND (TITLE_ABS:"resistance training") AND OPEN_ACCESS:Y',
        f"{anchor} AND OPEN_ACCESS:Y",
    ]


def test_follow_up_reuses_an_abbreviation_defined_in_an_earlier_turn() -> None:
    ladder = build_query_ladder(
        "How do LPT-derived force-velocity measures compare with force plates?",
        context=QUESTION,
    )
    assert ladder[0].groups[0] == ("linear position transducer", "LPT")
    assert ("force plates",) in {g for q in ladder for g in q.groups}
    assert build_query_ladder("What is it?") == ()


def _record(provider: str, rank: int, **ids: Any) -> DiscoveredRecord:
    return DiscoveredRecord(
        provider=provider,
        source_id=f"{provider}:{rank}",
        title="t",
        query_level=1,
        rank=rank,
        **ids,
    )


def test_identities_link_across_providers_and_quarantine_conflicts() -> None:
    accepted, conflicts = link_identities(
        [
            _record("openalex", 1, doi="10.1000/AA", pmid="11"),
            _record("europe_pmc", 2, doi="10.1000/aa", pmcid="PMC1"),  # same DOI, case-folded
            _record("europe_pmc", 1, pmid="22", pmcid="PMC2"),
            _record("openalex", 2, pmid="22", pmcid="PMC3"),  # same PMID, different PMCID
            _record("europe_pmc", 3),  # no strong identifier: never kept
        ]
    )
    assert [sorted({r.provider for r in g}) for g in accepted] == [["europe_pmc", "openalex"]]
    assert {r.pmcid for r in accepted[0]} == {"PMC1", None}
    assert len(conflicts) == 1 and {r.pmcid for r in conflicts[0]} == {"PMC2", "PMC3"}


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def test_rate_limited_http_enforces_allowlist_spacing_budget_and_bounded_retry_after() -> None:
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(str(request.url))
        if "openalex" in str(request.url) and len(calls) == 2:
            return httpx2.Response(429, json={"retryAfter": 3})
        return httpx2.Response(200, json={"ok": True})

    clock = _Clock()
    http = RateLimitedHttp(
        httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_requests=4,
        min_interval_seconds=1.0,
        max_retry_after_seconds=5.0,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(AcquisitionError, match="host_not_allowlisted"):
        http.get_json("https://example.org/search", {})
    with pytest.raises(AcquisitionError, match="host_not_allowlisted"):
        http.get_json("http://api.openalex.org/works", {})
    assert http.get_json("https://api.openalex.org/works", {"search": "x"})[0] == {"ok": True}
    payload, _ = http.get_json("https://api.openalex.org/works", {"search": "y"})
    assert payload == {"ok": True}
    assert http.requests[1]["status"] == 429 and http.requests[1]["retry_after_seconds"] == 3
    assert 3.0 in clock.sleeps and 1.0 in clock.sleeps  # Retry-After honoured, spacing kept
    http.get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {})
    with pytest.raises(AcquisitionError, match="request_budget_exhausted"):
        http.get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {})
    assert all("sha256" in r for r in http.requests)


def _epmc(pmcid: str, doi: str, license_name: str, pmid: str) -> dict[str, Any]:
    return {
        "id": pmid,
        "source": "MED",
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": f"Study {pmcid}",
        "pubYear": "2024",
        "license": license_name,
        "isOpenAccess": "Y",
        "inEPMC": "Y",
    }


def test_acquisition_discovers_deduplicates_gates_rights_and_imports_exact_jats(
    tmp_path: Path,
) -> None:
    content = (SAMPLE / "PMC10780924.xml").read_bytes()
    search_queries: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if url.startswith("https://www.ebi.ac.uk/europepmc/webservices/rest/search"):
            search_queries.append(request.url.params["query"])
            return httpx2.Response(
                200,
                json={
                    "resultList": {
                        "result": [
                            _epmc("PMC10780924", "10.3390/s24010044", "cc by", "38202906"),
                            _epmc("PMC9999999", "10.1000/nd", "cc by-nd", "1"),
                        ]
                    }
                },
            )
        if url.startswith("https://api.openalex.org/works"):
            return httpx2.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "display_name": "Same work seen by OpenAlex",
                            "doi": "https://doi.org/10.3390/s24010044",
                            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/38202906"},
                            "open_access": {"is_oa": True},
                        }
                    ]
                },
            )
        if url.endswith("/PMC10780924/fullTextXML"):
            return httpx2.Response(
                200, content=content, headers={"content-type": "application/xml"}
            )
        return httpx2.Response(404)

    transport = httpx2.MockTransport(handler)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_sqlite_schema(engine)
    acquisition = LiteratureAcquisition(
        ingest_engine=engine,
        store_root=tmp_path,
        qdrant=None,  # type: ignore[arg-type]  # projection is replaced below
        alias="unused",
        document_provider=lambda: None,  # type: ignore[arg-type,return-value]
        chunker_revision=CHUNKER,
        limits=AcquisitionLimits(target_candidates=1),
        http_factory=lambda: httpx2.Client(transport=transport),
        fulltext_factory=lambda: EuropePmcClient(transport=transport),
        sleep=lambda _s: None,
    )
    acquisition._project = lambda _started: {"collection": "c"}  # type: ignore[method-assign]
    events: list[str] = []
    receipt = acquisition.acquire(QUESTION, progress=lambda stage, _payload: events.append(stage))

    assert receipt["status"] == "completed"
    assert receipt["model_requests"] == 0
    assert len(search_queries) == 1  # the first level met the candidate target
    assert receipt["discovered_records"] == 3 and receipt["unique_works"] == 2
    merged = next(w for w in receipt["works"] if w["pmcid"] == "PMC10780924")
    assert merged["providers"] == ["europe_pmc", "openalex"] and merged["pmid"] == "38202906"
    [paper] = receipt["ingested"]
    assert paper["doi"] == "10.3390/s24010044"
    assert paper["license_uri"] == "https://creativecommons.org/licenses/by/4.0/"
    assert paper["passages"] > 0
    assert [r["reason"] for r in receipt["rejected"]] == [
        "repository rights absent, restrictive or conflicting"
    ]
    assert events == ["literature.discovered", "literature.ingested", "corpus.projected"]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(DocumentVersionRecord)) == 1
        assert session.scalar(select(func.count()).select_from(PassageRecord)) == paper["passages"]
    stored = list((tmp_path / "sources").rglob("*"))
    assert any(p.is_file() and p.read_bytes() == content for p in stored)
    json.dumps(receipt)  # the receipt is plain JSON
