"""Live, bounded literature acquisition for the research workspace (RES-638).

One acquisition turns a research question into canonical evidence:

1. **Discovery** - a deterministic query ladder over Europe PMC (``OPEN_ACCESS:Y``) and the
   OpenAlex works API. Only two allowlisted hosts are reachable, every request is rate limited,
   counted against a fixed budget and recorded verbatim (URL, status, SHA-256 of the body).
2. **Identity** - DOI, PMID and PMCID are normalized and connected across providers; a component
   that names two different DOIs, PMIDs or PMCIDs is quarantined, never merged by title.
3. **Rights** - repository metadata must say CC BY / CC0, open access and in Europe PMC before any
   full text is requested; the original JATS article-level licence and human-research scope are
   then re-checked on the exact bytes (``corpus.rights``). Everything else is recorded as rejected.
4. **Canonical import** - exact JATS bytes are stored content-addressed, imported, sectioned,
   chunked and materialized into PostgreSQL by the existing importer (idempotent).
5. **Projection** - every canonical passage is embedded with the pinned Qwen3-Embedding-0.6B TEI
   deployment (vectors reused only for byte-identical passages under the same fingerprint), a new
   content-addressed Qdrant collection is created, populated, verified, and the alias is moved
   from exactly the previous collection. The embedding manifest is kept so any process can
   reconstruct and verify the same projection.

Queries come from the research model when it asked for this search (``discover_external_literature``
with validated concept groups) and from fixed rules otherwise; every query actually sent is in the
receipt with its origin. No model is called here. Works already in the canonical corpus are
recorded as such and never count against the new-paper budget. Nothing here deletes a collection
or rewrites a canonical row.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast
from urllib.parse import urlsplit

import httpx2
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from dynamisrag.chunking.materializer import PassageMaterializer
from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.corpus.discovery import Candidate
from dynamisrag.corpus.rights import QualificationError, preflight, verify_article
from dynamisrag.db import list_paragraphs, list_sections, paragraph_from_record, section_from_record
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.domain.values import NormalizedDoi, Pmcid
from dynamisrag.embedding import EmbeddingInput, embed_passages
from dynamisrag.embedding.contracts import EmbeddingProvider
from dynamisrag.embedding.manifest import PassageEmbeddingEntry, PassageEmbeddingManifest
from dynamisrag.embedding.scientific_cpu import DOCUMENT_FINGERPRINT
from dynamisrag.ingestion import AcquiredFulltext, EuropePmcAcquisition, EuropePmcClient
from dynamisrag.ingestion.europe_pmc import EuropePmcError
from dynamisrag.jats import JatsCanonicalImporter
from dynamisrag.jats.errors import JatsParseError
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import QdrantClient, QdrantError, QdrantProjection, QdrantProjector
from dynamisrag.search.qdrant_artifact import load_embedding_manifest
from dynamisrag.storage import FileSystemObjectStore

__all__ = [
    "ACQUISITION_REVISION",
    "ALLOWED_HOSTS",
    "AcquisitionError",
    "AcquisitionLimits",
    "LiteratureAcquisition",
    "RateLimitedHttp",
    "build_query_ladder",
    "infer_acronyms",
    "link_identities",
    "load_published_projection",
    "model_query_levels",
]

ACQUISITION_REVISION: Final = "research-acquisition-v1"
EUROPE_PMC_SEARCH: Final = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENALEX_WORKS: Final = "https://api.openalex.org/works"
ALLOWED_HOSTS: Final = frozenset({"www.ebi.ac.uk", "api.openalex.org"})
USER_AGENT: Final = "DynamisRAG-research-workspace/0.1 (+https://github.com/Litju/DynamisRAG)"
PROJECTION_REGISTRY: Final = "projections"

# Generic evidence-quality vocabulary. Words a question asks *about* ("validity", "reliability")
# are useful search terms; words that only frame the question ("say", "evidence") are not.
_ASPECT_TERMS: Final = (
    "validity",
    "reliability",
    "agreement",
    "accuracy",
    "reproducibility",
    "sensitivity",
    "concordance",
)
_ASPECT_ALIASES: Final = {"valid": "validity", "reliable": "reliability", "accurate": "accuracy"}
# Generic measurement-science wording for a comparison against a reference instrument. It names no
# instrument or domain; the research model proposes the domain-specific reference systems.
_REFERENCE_TERMS: Final = ("criterion", "gold standard", "concurrent validity", "agreement")
_REFERENCE_WORDS: Final = frozenset({"reference", "criterion"})
_STOP_WORDS: Final = """
a an and are as at be between by can compare compared comparison could do does did evidence
for from how in into is it its of on or other say says such than that the their them these
they this those to use used using via was were what when where which who why with within
would about across against practical applications application limitations limitation
including studies study research paper papers literature systems system measures measure
derived respect versus vs current available
"""
_STOP: Final = frozenset(_STOP_WORDS.split())
_ABBREVIATION = re.compile(
    r"((?:[A-Za-z][\w-]*\s+){1,5}?[A-Za-z][\w-]*)\s*\(\s*([A-Z][A-Z0-9]{1,7})s?\s*\)"
)
_DASHES: Final = "-\u2010\u2011\u2013"  # hyphen-minus, hyphen, non-breaking hyphen, en dash
_HYPHENATED = re.compile(rf"\b([A-Za-z]{{3,}}(?:[{_DASHES}][A-Za-z]{{3,}})+)\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_ACRONYM = re.compile(r"\b([A-Z]{2,6})s?\b")
_QUERY_TERM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{0,58}[A-Za-z0-9]$")


class AcquisitionError(RuntimeError):
    """The acquisition could not run safely (budget, host allowlist, infrastructure)."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class AcquisitionLimits(BaseModel):
    """Hard bounds for one acquisition; recorded verbatim in its receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    max_search_requests: int = Field(default=6, ge=1, le=12)
    page_size: int = Field(default=20, ge=5, le=50)
    target_candidates: int = Field(default=12, ge=1, le=50)
    max_fulltext_requests: int = Field(default=8, ge=1, le=16)
    max_papers: int = Field(default=5, ge=1, le=10)
    max_passages: int = Field(default=400, ge=10, le=1500)
    min_request_interval_seconds: float = Field(default=1.0, ge=0.2, le=10.0)
    max_retry_after_seconds: float = Field(default=15.0, ge=0.0, le=60.0)
    deadline_seconds: float = Field(default=900.0, ge=30.0, le=3600.0)


class RateLimitedHttp:
    """GET-only client: allowlisted hosts, per-host spacing, a request budget, full receipts."""

    def __init__(
        self,
        client: httpx2.Client,
        *,
        max_requests: int,
        min_interval_seconds: float,
        max_retry_after_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._max_requests = max_requests
        self._interval = min_interval_seconds
        self._max_retry_after = max_retry_after_seconds
        self._clock, self._sleep = clock, sleep
        self._last: dict[str, float] = {}
        self.requests: list[dict[str, Any]] = []

    def get_json(self, url: str, params: dict[str, str | int]) -> tuple[dict[str, Any] | None, str]:
        """Return ``(payload, request_url)``; ``None`` means a recorded provider failure."""
        host = urlsplit(url).hostname or ""
        if urlsplit(url).scheme != "https" or host not in ALLOWED_HOSTS:
            raise AcquisitionError("host_not_allowlisted")
        for attempt in range(2):
            if len(self.requests) >= self._max_requests:
                raise AcquisitionError("request_budget_exhausted")
            wait = self._last.get(host, -1e9) + self._interval - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last[host] = self._clock()
            request = self._client.build_request(
                "GET", url, params=params, headers={"User-Agent": USER_AGENT}
            )
            record: dict[str, Any] = {"url": str(request.url), "attempt": attempt + 1}
            self.requests.append(record)
            try:
                response = self._client.send(request)
            except httpx2.HTTPError as error:
                record["error"] = type(error).__name__
                return None, str(request.url)
            record["status"] = response.status_code
            record["sha256"] = hashlib.sha256(response.content).hexdigest()
            record["bytes"] = len(response.content)
            if response.status_code == 429 and attempt == 0:
                retry_after = _retry_after(response)
                record["retry_after_seconds"] = retry_after
                if retry_after is not None and retry_after <= self._max_retry_after:
                    self._sleep(retry_after)
                    continue
            if response.status_code != 200:
                return None, str(request.url)
            try:
                payload = response.json()
            except ValueError:
                record["error"] = "invalid_json"
                return None, str(request.url)
            if not isinstance(payload, dict):
                record["error"] = "unexpected_shape"
                return None, str(request.url)
            return cast("dict[str, Any]", payload), str(request.url)
        return None, url


def _retry_after(response: httpx2.Response) -> float | None:
    header = response.headers.get("retry-after")
    try:
        if header is not None:
            return float(header)
        body: object = response.json()
        if isinstance(body, dict):
            value = cast("dict[str, Any]", body).get("retryAfter")
            if isinstance(value, int | float):
                return float(value)
    except ValueError:
        return None
    return None


# ---------------------------------------------------------------------------------- queries


class QueryLevel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    level: int
    groups: tuple[tuple[str, ...], ...]

    def europe_pmc(self) -> str:
        clauses = [
            "(" + " OR ".join(f'TITLE_ABS:"{term}"' for term in group) + ")"
            for group in self.groups
        ]
        return " AND ".join([*clauses, "OPEN_ACCESS:Y"])

    def openalex(self) -> str:
        return " ".join(group[0] for group in self.groups)


def _singular(phrase: str) -> str:
    words = phrase.split()
    last = words[-1]
    if len(last) > 4 and last.lower().endswith("s") and not last.lower().endswith("ss"):
        words[-1] = last[:-1]
    return " ".join(words)


def infer_acronyms(question: str, context: str = "") -> dict[str, str]:
    """Undefined acronyms in ``question`` whose letters are the initials of an earlier phrase.

    "LPT" after a turn about "linear position transducers" becomes "linear position transducer".
    Only exact initial matches of consecutive words are accepted; nothing is guessed.
    """
    known = " ".join(f"{context} {question}".split())
    words = _WORD.findall(known)
    found: dict[str, str] = {}
    for match in _ACRONYM.finditer(question):
        acronym = match.group(1)
        size = len(acronym)
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            if any(w.isupper() for w in window) or any(w.lower() in _STOP for w in window):
                continue
            if "".join(w[0] for w in window).upper() == acronym:
                found[acronym] = _singular(" ".join(w.lower() for w in window))
                break
    return found


def model_query_levels(searches: Sequence[Sequence[Sequence[str]]]) -> tuple[QueryLevel, ...]:
    """Validated research-model searches: each search is AND-ed groups of OR-synonym terms.

    Invalid terms (quotes, operators, field syntax, over-long text) are dropped, never repaired
    into query syntax; a search left without a group is discarded.
    """
    levels: list[tuple[tuple[str, ...], ...]] = []
    for search in list(searches)[:4]:
        groups: list[tuple[str, ...]] = []
        for group in list(search)[:4]:
            terms = tuple(
                dict.fromkeys(
                    " ".join(str(t).split())
                    for t in list(group)[:6]
                    if _QUERY_TERM.match(" ".join(str(t).split()))
                )
            )
            if terms:
                groups.append(terms)
        if groups and tuple(groups) not in levels:
            levels.append(tuple(groups))
    return tuple(QueryLevel(level=i + 1, groups=g) for i, g in enumerate(levels))


def build_query_ladder(  # noqa: PLR0912 - one branch per fixed query rule
    question: str, context: str = ""
) -> tuple[QueryLevel, ...]:
    """Fixed-rule search groups, most specific first. No model, no domain dictionary.

    * an abbreviation defined in the question or earlier user turns ("linear position
      transducers (LPT)") becomes ``("linear position transducer", "LPT")`` and is the anchor;
    * hyphenated compounds ("force-velocity") are topic groups;
    * evidence-quality words the question asks about form one OR group;
    * otherwise adjacent content words ("resistance training") form the topic groups.
    """
    text = " ".join(question.split())
    known = " ".join(f"{context} {question}".split())
    definitions: dict[str, str] = {}
    for match in _ABBREVIATION.finditer(known):
        words = [w for w in match.group(1).split() if w.lower() not in _STOP]
        abbreviation = match.group(2)
        phrase = " ".join(words[-len(abbreviation) :]) if len(words) >= len(abbreviation) else ""
        if phrase:
            definitions[abbreviation] = _singular(phrase)
    for abbreviation, phrase in infer_acronyms(text, context).items():
        definitions.setdefault(abbreviation, phrase)
    anchors: list[tuple[str, ...]] = []
    for abbreviation, phrase in definitions.items():
        if re.search(rf"\b{re.escape(abbreviation)}s?\b", text) or phrase.lower() in text.lower():
            anchors.append((phrase, abbreviation))
    lowered = text.lower()
    compounds: list[tuple[str, ...]] = []
    for match in _HYPHENATED.finditer(text):
        parts = re.split(f"[{_DASHES}]", match.group(1))
        if any(p.upper() in definitions or p.lower() in _STOP for p in parts):
            continue
        term = "-".join(p.lower() for p in parts)
        if (term,) not in compounds:
            compounds.append((term,))
    aspects = tuple(
        dict.fromkeys(
            _ASPECT_ALIASES.get(word, word)
            for word in _WORD.findall(lowered)
            if word in _ASPECT_TERMS or word in _ASPECT_ALIASES
        )
    )
    if _REFERENCE_WORDS & set(_WORD.findall(lowered)):
        aspects = tuple(dict.fromkeys((*aspects, *_REFERENCE_TERMS)))
    # Words of an abbreviation's expansion are already searched as one phrase.
    covered = {w.lower() for group in anchors for t in group for w in t.split()}
    words = [
        w
        for w in _WORD.findall(text)
        if w.lower() not in _STOP
        and w.lower() not in covered
        and w.lower() not in _ASPECT_TERMS
        and w.lower() not in _ASPECT_ALIASES
        and w.lower() not in _REFERENCE_WORDS
        and w.upper() not in definitions
        and w.rstrip("s").upper() not in definitions
        and len(w) > 2
    ]
    # Adjacent content words; never the tail of a hyphenated compound ("force-velocity
    # profiling" is the compound plus a modifier, not the topic "velocity profiling").
    bigrams = [
        (f"{a.lower()} {b.lower()}",)
        for a, b in itertools.pairwise(words)
        if re.search(rf"(?<![{_DASHES}\w]){re.escape(a)}\s+{re.escape(b)}\b", text)
    ]
    # Three adjacent content words are one term ("linear position transducers"), not two
    # overlapping bigrams; the bigrams inside it are dropped.
    triples = [
        (a.lower(), b.lower(), c.lower())
        for a, b, c in zip(words, words[1:], words[2:], strict=False)
        if re.search(rf"(?<![{_DASHES}\w]){re.escape(a)}\s+{re.escape(b)}\s+{re.escape(c)}\b", text)
    ][:1]
    inside = {f"{x} {y}" for t in triples for x, y in itertools.pairwise(t)}
    bigrams = [b for b in bigrams if b[0] not in inside]
    topics = [*compounds, *((_singular(" ".join(t)),) for t in triples), *bigrams[:2]]
    if not anchors and not topics:
        topics = [(w.lower(),) for w in words[:3]]
    if not anchors and not topics:
        return ()
    core = anchors or topics[:1]
    rest = [t for t in topics if t not in core]
    levels: list[tuple[tuple[str, ...], ...]] = []
    aspect_group = (aspects,) if aspects else ()
    for candidate in (
        (*core, *rest[:1], *aspect_group),
        (*core, *aspect_group),
        (*core, *rest[:1]),
        (*core, *rest[1:2]),
        tuple(core),
    ):
        if candidate and candidate not in levels:
            levels.append(candidate)
    return tuple(QueryLevel(level=i + 1, groups=g) for i, g in enumerate(levels))


# ------------------------------------------------------------------------------- identities


class DiscoveredRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    provider: str
    source_id: str
    title: str
    doi: NormalizedDoi | None = None
    pmid: str | None = None
    pmcid: Pmcid | None = None
    year: int | None = None
    license: str | None = None
    is_open_access: bool | None = None
    in_europe_pmc: bool | None = None
    query_level: int
    rank: int
    # Query groups (of the most specific search) matched by title + abstract: a broad fallback
    # hit about another topic must not displace a work on the asked topic.
    topical_score: int = 0


def topical_score(title: str, abstract: str, groups: Sequence[Sequence[str]]) -> int:
    text = " ".join(f"{title} {abstract}".lower().replace("\u2013", "-").split())
    return sum(
        1
        for group in groups
        if any(re.search(rf"\b{re.escape(term.lower())}", text) for term in group)
    )


def _pmid(value: object) -> str | None:
    if not isinstance(value, str | int):
        return None
    text = str(value).rstrip("/").rsplit("/", 1)[-1]
    return text if text.isdecimal() else None


def _pmcid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.rstrip("/").rsplit("/", 1)[-1]
    text = f"PMC{text}" if text.isdecimal() else text
    return cast(str, TypeAdapter(Pmcid).validate_python(text))


def _europe_pmc_record(raw: dict[str, Any], level: int, rank: int) -> DiscoveredRecord:
    return DiscoveredRecord(
        provider="europe_pmc",
        source_id=str(raw.get("source", "")) + ":" + str(raw.get("id", "")),
        title=re.sub(r"<[^>]+>|&lt;.*?&gt;", "", str(raw.get("title") or "")).strip() or "untitled",
        doi=raw.get("doi"),
        pmid=_pmid(raw.get("pmid")),
        pmcid=_pmcid(raw.get("pmcid")),
        year=int(raw["pubYear"]) if str(raw.get("pubYear", "")).isdecimal() else None,
        license=raw.get("license"),
        is_open_access=raw.get("isOpenAccess") == "Y",
        in_europe_pmc=raw.get("inEPMC") == "Y",
        query_level=level,
        rank=rank,
    )


def _openalex_record(raw: dict[str, Any], level: int, rank: int) -> DiscoveredRecord:
    ids = cast("dict[str, Any]", raw.get("ids") or {})
    location = cast("dict[str, Any]", raw.get("best_oa_location") or {})
    return DiscoveredRecord(
        provider="openalex",
        source_id=str(raw.get("id", "")),
        title=str(raw.get("display_name") or "untitled"),
        doi=ids.get("doi") or raw.get("doi"),
        pmid=_pmid(ids.get("pmid")),
        pmcid=_pmcid(ids.get("pmcid")),
        year=raw.get("publication_year") if isinstance(raw.get("publication_year"), int) else None,
        license=location.get("license"),
        is_open_access=bool(cast("dict[str, Any]", raw.get("open_access") or {}).get("is_oa")),
        query_level=level,
        rank=rank,
    )


def _keys(record: DiscoveredRecord) -> set[str]:
    return {
        f"{kind}:{value}"
        for kind, value in (("doi", record.doi), ("pmid", record.pmid), ("pmcid", record.pmcid))
        if value
    }


def link_identities(
    records: Sequence[DiscoveredRecord],
) -> tuple[list[list[DiscoveredRecord]], list[list[DiscoveredRecord]]]:
    """Connected DOI/PMID/PMCID components; a component with two values of a kind is quarantined.

    Records without any strong identifier are dropped by the caller. Order is preserved by the
    best (lowest) provider rank so relevance ranking survives deduplication.
    """
    groups: list[list[DiscoveredRecord]] = []
    for record in records:
        keys = _keys(record)
        if not keys:
            continue
        matches = [g for g in groups if any(keys & _keys(r) for r in g)]
        merged = [record]
        for group in matches:
            merged.extend(group)
            groups.remove(group)
        groups.append(merged)
    accepted: list[list[DiscoveredRecord]] = []
    conflicts: list[list[DiscoveredRecord]] = []
    for group in groups:
        kinds = [{getattr(r, k) for r in group if getattr(r, k)} for k in ("doi", "pmid", "pmcid")]
        (conflicts if any(len(values) > 1 for values in kinds) else accepted).append(group)

    def order(group: list[DiscoveredRecord]) -> tuple[int, int, int, int]:
        best = min(group, key=lambda r: (r.provider != "europe_pmc", r.query_level, r.rank))
        topical = max((r.topical_score for r in group), default=0)
        return (-topical, best.provider != "europe_pmc", best.query_level, best.rank)

    return sorted(accepted, key=order), conflicts


def _identity(group: list[DiscoveredRecord]) -> dict[str, Any]:
    first = min(group, key=lambda r: (r.provider != "europe_pmc", r.query_level, r.rank))

    def pick(attr: str) -> Any:  # noqa: ANN401 - identifier or None
        return next((getattr(r, attr) for r in group if getattr(r, attr)), None)

    return {
        "doi": pick("doi"),
        "pmid": pick("pmid"),
        "pmcid": pick("pmcid"),
        "title": first.title[:300],
        "year": pick("year"),
        "topical_score": max(r.topical_score for r in group),
        "providers": sorted({r.provider for r in group}),
        "europe_pmc_license": next(
            (r.license for r in group if r.provider == "europe_pmc" and r.license), None
        ),
    }


def _rights_candidates(
    group: list[DiscoveredRecord], raw: dict[str, dict[str, Any]]
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for record in group:
        if record.provider != "europe_pmc":
            continue
        metadata = raw.get(record.source_id, {})
        candidates.append(
            Candidate(
                provider="europe_pmc",
                source_id=record.source_id,
                title=record.title,
                doi=record.doi,
                pmcid=record.pmcid,
                page_sha256=hashlib.sha256(
                    json.dumps(metadata, sort_keys=True).encode()
                ).hexdigest(),
                metadata=metadata,
            )
        )
    return candidates


# ------------------------------------------------------------------------------- projection


class PublishedProjection(BaseModel):
    """What another process needs to reconstruct and verify the published collection."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    collection: str
    alias: str
    embedding_manifest_file: str
    embedding_manifest_sha256: str
    projection_sha256: str
    passage_count: int
    document_count: int
    published_at: str


def _registry(store_root: Path) -> Path:
    return store_root / PROJECTION_REGISTRY


def load_published_projection(
    engine: Engine, store_root: Path, collection: str, *, chunker_revision: str
) -> QdrantProjection:
    """Rebuild the exact projection for ``collection`` from its manifest and fresh PostgreSQL."""
    record = PublishedProjection.model_validate_json(
        (_registry(store_root) / f"{collection}.json").read_bytes()
    )
    embeddings = load_embedding_manifest(
        _registry(store_root) / record.embedding_manifest_file,
        expected_sha256=record.embedding_manifest_sha256,
    )
    keys = tuple(entry.passage_key for entry in embeddings.entries)
    with Session(engine) as session:
        records = tuple(
            list_passage_projection_records(
                session, chunker_revision=chunker_revision, passage_keys=keys
            )
        )
    passages = build_projection_manifest(records, chunker_revision=chunker_revision)
    projection = QdrantProjection(passages, embeddings, DOCUMENT_FINGERPRINT)
    if projection.collection_name != collection:
        raise QdrantError("published projection no longer matches canonical PostgreSQL")
    return projection


# ------------------------------------------------------------------------------ acquisition


_LOCK = threading.Lock()


class ClosableEmbeddingProvider(EmbeddingProvider, Protocol):
    def close(self) -> None: ...


class LiteratureAcquisition:
    """Discovery -> rights -> canonical import -> versioned dense projection, bounded."""

    def __init__(
        self,
        *,
        ingest_engine: Engine,
        store_root: Path,
        qdrant: QdrantClient,
        alias: str,
        document_provider: Callable[[], ClosableEmbeddingProvider],
        chunker_revision: str,
        limits: AcquisitionLimits | None = None,
        http_factory: Callable[[], httpx2.Client] | None = None,
        fulltext_factory: Callable[[], EuropePmcClient] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ingest_engine = ingest_engine
        self.store_root = store_root.resolve()
        self.qdrant = qdrant
        self.alias = alias
        self.document_provider = document_provider
        self.chunker_revision = chunker_revision
        self.limits = limits or AcquisitionLimits()
        self._http_factory = http_factory or (
            lambda: httpx2.Client(timeout=30.0, follow_redirects=False, trust_env=False)
        )
        self._fulltext_factory = fulltext_factory or (lambda: EuropePmcClient(timeout_seconds=40))
        self._sleep = sleep

    # -- public ---------------------------------------------------------------------------

    def acquire(
        self,
        question: str,
        *,
        context: str = "",
        should_stop: Callable[[], bool] = lambda: False,
        progress: Callable[[str, dict[str, Any]], None] = lambda _stage, _payload: None,
        model_searches: Sequence[Sequence[Sequence[str]]] = (),
    ) -> dict[str, Any]:
        """Run one bounded acquisition and return its receipt (never raises on provider data).

        ``model_searches`` are the research model's own concept groups (validated here); they are
        sent first, then the fixed-rule ladder fills any remaining search budget.
        """
        started = time.monotonic()
        limits = self.limits
        receipt: dict[str, Any] = {
            "revision": ACQUISITION_REVISION,
            "started_at": datetime.now(UTC).isoformat(),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "limits": limits.model_dump(mode="json"),
            "allowed_hosts": sorted(ALLOWED_HOSTS),
            "model_requests": 0,
        }
        proposed = model_query_levels(model_searches)
        rules = build_query_ladder(question, context)
        merged = [*proposed, *(q for q in rules if q.groups not in {p.groups for p in proposed})]
        ladder = tuple(q.model_copy(update={"level": i + 1}) for i, q in enumerate(merged))
        receipt["query_ladder"] = [
            {
                "level": q.level,
                "origin": "model_directed" if i < len(proposed) else "fixed_rule",
                "europe_pmc": q.europe_pmc(),
                "openalex": q.openalex(),
            }
            for i, q in enumerate(ladder)
        ]
        if not ladder:
            receipt["status"] = "no_searchable_terms"
            return receipt
        with self._http_factory() as client:
            http = RateLimitedHttp(
                client,
                max_requests=limits.max_search_requests,
                min_interval_seconds=limits.min_request_interval_seconds,
                max_retry_after_seconds=limits.max_retry_after_seconds,
                sleep=self._sleep,
            )
            records, raw_epmc, failures = self._discover(http, ladder, should_stop)
            receipt["search_requests"] = http.requests
        receipt["provider_failures"] = failures
        groups, conflicts = link_identities(records)
        receipt["discovered_records"] = len(records)
        receipt["unique_works"] = len(groups)
        receipt["identity_conflicts"] = [[r.model_dump(mode="json") for r in g] for g in conflicts]
        receipt["works"] = [_identity(g) for g in groups]
        progress(
            "literature.discovered",
            {
                "records": len(records),
                "unique_works": len(groups),
                "identity_conflicts": len(conflicts),
                "provider_failures": len(failures),
                "works": [
                    {k: w[k] for k in ("doi", "pmid", "pmcid", "title")}
                    for w in receipt["works"][:12]
                ],
            },
        )
        if should_stop():
            receipt["status"] = "stopped"
            return receipt
        with _LOCK:  # one canonical writer and one Qdrant publisher per process
            ingested, rejected, present = self._ingest(groups, raw_epmc, started, should_stop)
            receipt["ingested"] = ingested
            receipt["rejected"] = rejected
            receipt["already_in_corpus"] = present
            progress(
                "literature.ingested",
                {
                    "ingested": [
                        {
                            k: p[k]
                            for k in ("doi", "pmid", "pmcid", "title", "passages", "license_uri")
                        }
                        for p in ingested
                    ],
                    "rejected": len(rejected),
                    "already_in_corpus": len(present),
                },
            )
            if not ingested:
                # Discovered works that are already canonical are still evidence to retrieve.
                receipt["status"] = "already_in_corpus" if present else "nothing_ingested"
                return receipt
            receipt["projection"] = self._project(started)
            progress("corpus.projected", receipt["projection"])
        receipt["status"] = "completed"
        receipt["duration_seconds"] = round(time.monotonic() - started, 3)
        return receipt

    # -- discovery ------------------------------------------------------------------------

    def _discover(  # noqa: PLR0912 - explicit, recorded provider failure branches
        self, http: RateLimitedHttp, ladder: Sequence[QueryLevel], should_stop: Callable[[], bool]
    ) -> tuple[list[DiscoveredRecord], dict[str, dict[str, Any]], list[dict[str, Any]]]:
        limits = self.limits
        records: list[DiscoveredRecord] = []
        raw_epmc: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        seen: set[str] = set()

        def unique() -> int:
            return len({k for r in records for k in _keys(r) if k.startswith("pmcid:")})

        for level in ladder:
            if should_stop() or unique() >= limits.target_candidates:
                break
            try:
                payload, url = http.get_json(
                    EUROPE_PMC_SEARCH,
                    {
                        "query": level.europe_pmc(),
                        "format": "json",
                        "resultType": "core",
                        "pageSize": limits.page_size,
                    },
                )
            except AcquisitionError as error:
                failures.append(
                    {"provider": "europe_pmc", "level": level.level, "error": error.code}
                )
                break
            if payload is None:
                failures.append({"provider": "europe_pmc", "level": level.level, "url": url})
                continue
            try:
                items = cast("list[Any]", payload["resultList"]["result"])
            except (KeyError, TypeError):
                failures.append({"provider": "europe_pmc", "level": level.level, "error": "shape"})
                continue
            for rank, raw in enumerate(items[: limits.page_size], start=1):
                try:
                    record = _europe_pmc_record(cast("dict[str, Any]", raw), level.level, rank)
                except (ValidationError, ValueError, TypeError, AttributeError, KeyError):
                    continue
                record = record.model_copy(
                    update={
                        "topical_score": topical_score(
                            record.title,
                            str(cast("dict[str, Any]", raw).get("abstractText") or ""),
                            ladder[0].groups,
                        )
                    }
                )
                if record.source_id in seen:
                    continue
                seen.add(record.source_id)
                raw_epmc[record.source_id] = cast("dict[str, Any]", raw)
                records.append(record)
        # OpenAlex once, at the most specific level, as an independent cross-check of identity.
        if ladder and not should_stop():
            try:
                payload, url = http.get_json(
                    OPENALEX_WORKS,
                    {
                        "search": ladder[min(1, len(ladder) - 1)].openalex(),
                        "filter": "open_access.is_oa:true",
                        "per_page": limits.page_size,
                    },
                )
                if payload is None:
                    failures.append({"provider": "openalex", "url": url})
                else:
                    for rank, raw in enumerate(
                        cast("list[Any]", payload.get("results") or [])[: limits.page_size], 1
                    ):
                        try:
                            records.append(_openalex_record(cast("dict[str, Any]", raw), 2, rank))
                        except (ValidationError, ValueError, TypeError, AttributeError):
                            continue
            except AcquisitionError as error:
                failures.append({"provider": "openalex", "error": error.code})
        return records, raw_epmc, failures

    # -- ingestion ------------------------------------------------------------------------

    def _ingest(
        self,
        groups: Sequence[list[DiscoveredRecord]],
        raw_epmc: dict[str, dict[str, Any]],
        started: float,
        should_stop: Callable[[], bool],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        limits = self.limits
        ingested: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        present: list[dict[str, Any]] = []
        existing = self._existing_identifiers()
        fetches = 0
        store = FileSystemObjectStore(self.store_root / "sources")
        fulltext = self._fulltext_factory()
        try:
            for group in groups:
                identity = _identity(group)
                if any(
                    (kind, str(identity[kind]).lower()) in existing
                    for kind in ("doi", "pmid", "pmcid")
                    if identity.get(kind)
                ):
                    # Already canonical: it is evidence for this turn but not a new paper.
                    present.append(identity)
                    continue
                if len(ingested) >= limits.max_papers:
                    break
                if should_stop() or time.monotonic() - started > limits.deadline_seconds:
                    rejected.append({**identity, "reason": "stopped_or_deadline"})
                    break
                try:
                    pmcid = preflight(_rights_candidates(group, raw_epmc))
                except QualificationError as error:
                    rejected.append(
                        {**identity, "stage": "repository_rights", "reason": str(error)}
                    )
                    continue
                if fetches >= limits.max_fulltext_requests:
                    rejected.append({**identity, "reason": "fulltext_budget_exhausted"})
                    break
                fetches += 1
                if fetches > 1:
                    self._sleep(limits.min_request_interval_seconds)
                try:
                    payload = fulltext.fetch_fulltext(pmcid)
                    rights = verify_article(
                        payload.content, _rights_candidates(group, raw_epmc), pmcid
                    )
                except (EuropePmcError, QualificationError, JatsParseError, ValueError) as error:
                    rejected.append(
                        {**identity, "stage": "original_article", "reason": str(error)[:240]}
                    )
                    continue
                paper = self._import(payload, EuropePmcAcquisition(fulltext, store))
                ingested.append(
                    {
                        **identity,
                        **paper,
                        "license_uri": rights["license_uri"],
                        "license_family": rights["license_family"],
                        "source_sha256": rights["source_sha256"],
                        "article_type": rights["scientific_scope"]["article_type"],
                        "article_url": f"https://europepmc.org/article/PMC/{pmcid}",
                    }
                )
        finally:
            fulltext.close()
        return ingested, rejected, present

    def _existing_identifiers(self) -> set[tuple[str, str]]:
        with Session(self.ingest_engine) as session:
            rows = session.execute(
                text("SELECT namespace, normalized_value FROM document_identifier")
            ).all()
        return {(str(kind), str(value).lower()) for kind, value in rows}

    def _import(
        self, payload: AcquiredFulltext, acquisition: EuropePmcAcquisition
    ) -> dict[str, Any]:
        """Persist the exact verified bytes; the importer is idempotent per source digest."""
        with Session(self.ingest_engine) as session, session.begin():
            acquired = acquisition.acquire_payload(session, payload)
            result = JatsCanonicalImporter(session).import_artifact(
                acquired.artifact, payload.content
            )
            sections = [
                section_from_record(r, version_key=result.version.version_key)
                for r in list_sections(session, result.version.id)
            ]
            paragraphs = [
                paragraph_from_record(r, version_key=result.version.version_key)
                for r in list_paragraphs(session, result.version.id)
            ]
            manifest = StructureAwareChunker().plan(result.version, sections, paragraphs)
            if manifest.chunker_revision != self.chunker_revision:
                raise AcquisitionError("chunker_revision_mismatch")
            materialized = PassageMaterializer(session).materialize(result.version, manifest)
            return {
                "document_version_key": result.version.version_key,
                "canonical_title": result.version.title[:300],
                "passages": len(materialized.passages),
                "import_created": result.created,
                "passage_manifest_sha256": manifest.manifest_sha256,
            }

    # -- projection -----------------------------------------------------------------------

    def _project(self, started: float) -> dict[str, Any]:
        with Session(self.ingest_engine) as session:
            records = tuple(
                list_passage_projection_records(session, chunker_revision=self.chunker_revision)
            )
        passages = build_projection_manifest(records, chunker_revision=self.chunker_revision)
        if passages.document_count > self.limits.max_passages:
            raise AcquisitionError("passage_budget_exceeded")
        previous_name = self.qdrant.aliases().get(self.alias)
        reusable: dict[tuple[str, str], tuple[float, ...]] = {}
        if previous_name is not None:
            try:
                previous = load_published_projection(
                    self.ingest_engine,
                    self.store_root,
                    previous_name,
                    chunker_revision=self.chunker_revision,
                )
                reusable = {
                    (e.passage_key, e.content_sha256): e.values for e in previous.embeddings.entries
                }
            except (OSError, ValueError, QdrantError):
                reusable = {}
        missing = [
            EmbeddingInput(d.passage_key, d.content_sha256, d.text)
            for d in passages.documents
            if (d.passage_key, d.content_sha256) not in reusable
        ]
        provider = self.document_provider()
        embed_started = time.monotonic()
        try:
            fresh = embed_passages(provider, tuple(missing or _first(passages.documents)))
        finally:
            provider.close()
        if fresh.embedding_config_sha256 != DOCUMENT_FINGERPRINT or fresh.dimension != 512:
            raise AcquisitionError("embedding_identity_mismatch")
        entries = {e.passage_key: e for e in fresh.entries}
        merged: list[PassageEmbeddingEntry] = []
        for document in passages.documents:
            entry = entries.get(document.passage_key)
            if entry is None:
                values = reusable[(document.passage_key, document.content_sha256)]
                entry = dataclasses.replace(
                    fresh.entries[0],
                    passage_key=document.passage_key,
                    content_sha256=document.content_sha256,
                    values=values,
                )
            merged.append(entry)
        merged.sort(key=lambda e: e.passage_key)
        embeddings: PassageEmbeddingManifest = dataclasses.replace(
            fresh, entries=tuple(merged), document_count=len(merged)
        )
        projection = QdrantProjection(passages, embeddings, DOCUMENT_FINGERPRINT)
        projector = QdrantProjector(self.qdrant)
        projector.create(projection)
        projector.populate(projection)
        projector.verify(projection)
        registry = _registry(self.store_root)
        registry.mkdir(parents=True, exist_ok=True)
        manifest_file = f"{projection.collection_name}.embeddings.json"
        (registry / manifest_file).write_bytes(embeddings.manifest_bytes)
        record = PublishedProjection(
            collection=projection.collection_name,
            alias=self.alias,
            embedding_manifest_file=manifest_file,
            embedding_manifest_sha256=embeddings.manifest_sha256,
            projection_sha256=passages.projection_sha256,
            passage_count=passages.document_count,
            document_count=len({d.document_version_key for d in passages.documents}),
            published_at=datetime.now(UTC).isoformat(),
        )
        (registry / f"{projection.collection_name}.json").write_text(
            record.model_dump_json(indent=1), encoding="utf-8", newline="\n"
        )
        projector.publish(projection, alias=self.alias, expected_previous=previous_name)
        return {
            **record.model_dump(mode="json"),
            "previous_collection": previous_name,
            "embedded_now": len(missing),
            "reused_vectors": passages.document_count - len(missing),
            "embedding_seconds": round(time.monotonic() - embed_started, 3),
            "model_id": embeddings.provider.model_id,
            "model_revision": embeddings.provider.model_sha,
            "embedding_config_sha256": embeddings.embedding_config_sha256,
            "dimension": embeddings.dimension,
            "collection_identity": projection.identity,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def _first(documents: Iterable[Any]) -> tuple[EmbeddingInput, ...]:
    """Embed one passage when everything is reusable, to re-attest the live TEI identity."""
    document = next(iter(documents))
    return (EmbeddingInput(document.passage_key, document.content_sha256, document.text),)
