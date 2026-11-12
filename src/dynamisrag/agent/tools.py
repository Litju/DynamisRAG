"""Scientific tool registry and canonical read-only adapters (RES-637).

Implements the 10 required scientific research tools plus an explicitly
restricted external discovery guard:

1. `list_papers`
2. `hybrid_search`
3. `search_index`
4. `get_passage`
5. `expand_passage_context`
6. `get_study_metadata`
7. `extract_study_findings`
8. `compare_studies`
9. `calculate_statistics`
10. `verify_claims`
11. `discover_external_literature` (disabled by policy unless explicitly enabled;
    never mutates canonical state)

Every tool enforces:
- Strict Pydantic input/output contracts.
- Explicit `DataScope` authorization and allowed document-version boundaries.
- Read-only canonical PostgreSQL transactions (`REPEATABLE READ READ ONLY` with
  `statement_timeout`).
- Prompt-injection detection on untrusted source text.
- Bounded execution time and byte budgets.
- Structured `ToolExecutionEnvelope` responses with trace and output digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from dynamisrag.agent.claims import assess_numeric_provenance, assess_semantic_support
from dynamisrag.agent.contracts import (
    CalculateStatisticsInput,
    CalculateStatisticsOutput,
    CandidateClaimInput,
    CompareStudiesInput,
    CompareStudiesOutput,
    DataScope,
    DiscoverExternalLiteratureInput,
    DiscoverExternalLiteratureOutput,
    ExpandedParagraphItem,
    ExpandPassageContextInput,
    ExpandPassageContextOutput,
    ExtractStudyFindingsInput,
    ExtractStudyFindingsOutput,
    GetPassageInput,
    GetPassageOutput,
    GetStudyMetadataInput,
    GetStudyMetadataOutput,
    HybridSearchInput,
    HybridSearchOutput,
    ListPapersInput,
    ListPapersOutput,
    PaperSummary,
    PassageDetail,
    PassageSpanItem,
    ProposedStudyFinding,
    SearchHitItem,
    SearchIndexInput,
    SearchIndexOutput,
    StatisticalCalculationKind,
    StatisticalGroupInput,
    StudyComparisonDimension,
    StudyComparisonMatrix,
    StudyMetadataDetail,
    ToolErrorCode,
    ToolErrorDetail,
    ToolExecutionEnvelope,
    ToolName,
    ToolProvenance,
    VerifiedCitationItem,
    VerifyClaimsInput,
    VerifyClaimsOutput,
)
from dynamisrag.db.canonical import (
    PassageProjectionRecords,
    list_passage_projection_records,
)
from dynamisrag.db.evidence import source_fingerprint
from dynamisrag.db.models import (
    CitationRecord,
    DocumentIdentifierRecord,
    DocumentRecord,
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    ParagraphRecord,
    PassageRecord,
    SectionRecord,
    SourceArtifactRecord,
)
from dynamisrag.domain.evidence import (
    Claim,
    EvidenceGraph,
    EvidenceLink,
    EvidenceMember,
    EvidenceSet,
    InvalidProvenanceError,
)
from dynamisrag.domain.identity import digest, evidence_identity, paragraph_key
from dynamisrag.generation.scientific import scientific_terms
from dynamisrag.search.bm25 import Bm25SearchService, SearchHit
from dynamisrag.search.errors import OpenSearchError
from dynamisrag.search.retrieval import (
    HybridRetrievalService,
    PassageProvenance,
    QueryEmbeddingUnavailable,
)
from dynamisrag.search.scientific import (
    CHUNKER,
    ScientificRetrievalService,
    ScientificUnavailable,
)

_InputT = TypeVar("_InputT", bound=BaseModel)

_INJECTION_PATTERN = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|above|prior|all)\s+(?:instructions|prompts|rules)|"
    r"system\s+prompt|developer\s+message|you\s+are\s+now|"
    r"override\s+(?:tool|security|policy|permissions)|"
    r"reveal\s+.*(?:secret|api[_\s]?key|password|token)|"
    r"execute\s+(?:sql|shell|bash|command)|"
    r"<\s*(?:system|instruction|tool_call)\s*>",
    re.IGNORECASE,
)

_STUDY_DESIGN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "randomized controlled trial",
        re.compile(r"\brandomized\s+controlled\s+trial\b|\bRCT\b", re.I),
    ),
    (
        "crossover / repeated-measures design",
        re.compile(
            r"\brepeated[\s-]measures\b|\bcross[\s-]?over\b|\bwithin[\s-]subject",
            re.I,
        ),
    ),
    (
        "controlled intervention study",
        re.compile(r"\bcontrol\s+group\b|\btraining\s+group\b", re.I),
    ),
    (
        "experimental biomechanical study",
        re.compile(r"\b30°\b|\bknee\s+flexion\b|\bforce\s+platform\b", re.I),
    ),
)

_POPULATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:male|female|healthy|young|collegiate|recreationally\s+active|trained)?"
        r"\s*(?:cricketers|athletes|participants|subjects|men|women|adults|individuals|players)\b"
        r"[^.;]{0,80}",
        re.I,
    ),
)

_INTERVENTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:plyometric\s+training|complex\s+training|depth\s+jump(?:ing)?|"
        r"plyometrics\s+without\s+jumping|knee\s+flexion\s+angle)[^.;]{0,80}",
        re.I,
    ),
)

_PROTOCOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:electromyograph\w*|sEMG|EMG|ultrasound|force\s+plate\w*|motion\s+capture|"
        r"goniomet\w*|medicine\s+ball|vertical\s+jump|McGill|plank)[^.;]{0,80}",
        re.I,
    ),
)

_OUTCOME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:muscle\s+activity|peak\s+force|rate\s+of\s+force\s+development|RFD|"
        r"jump\s+height|impulse|fascicle\s+length|tendon\s+stiffness|Achilles\s+tendon|"
        r"core\s+strength|upper\s+limb\s+power|lower\s+limb\s+power|sprint|agility)[^.;]{0,60}",
        re.I,
    ),
)

_STAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:p\s*[<=>]\s*0\.\d+|F\s*=\s*\d+(?:\.\d+)?|t\s*=\s*[-+]?\d+(?:\.\d+)?|"
        r"d\s*=\s*[-+]?\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:\.\d+)?"
        r"\s*(?:%|N|m/s|cm|ms|s|kg|W|°)?|\b\d+(?:\.\d+)?\s*(?:%|N\b|cm\b|ms\b|m/s\b|°))",
        re.I,
    ),
)

_UNIT_NORMALIZATION: dict[str, str] = {
    "n": "N",
    "newton": "N",
    "newtons": "N",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "%": "%",
    "percent": "%",
    "%mvc": "%MVC",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "°": "deg",
    "w": "W",
    "watt": "W",
    "watts": "W",
    "m/s": "m/s",
    "n/s": "N/s",
}


def contains_prompt_injection(text_value: str) -> bool:
    """Detect prompt-injection or instruction-override directives inside untrusted text."""
    return bool(_INJECTION_PATTERN.search(text_value))


def normalize_unit(unit: str) -> str:
    """Canonicalize a unit string for strict compatibility checks."""
    cleaned = unit.strip()
    return _UNIT_NORMALIZATION.get(cleaned.casefold(), cleaned)


class ToolExecutionError(Exception):
    """Internal exception mapped deterministically to a `ToolErrorDetail`."""

    def __init__(self, code: ToolErrorCode, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message[:480]
        self.retryable = retryable
        super().__init__(self.message)


class ScientificToolRegistry:
    """Authoritative registry of read-only scientific tools for the LangGraph agent."""

    def __init__(
        self,
        engine: Engine,
        *,
        scientific_retrieval: ScientificRetrievalService | None = None,
        opensearch_bm25: Bm25SearchService | None = None,
        opensearch_hybrid: HybridRetrievalService | None = None,
        chunker_revision: str = CHUNKER,
    ) -> None:
        self.engine = engine
        self.scientific_retrieval = scientific_retrieval
        self.opensearch_bm25 = opensearch_bm25
        self.opensearch_hybrid = opensearch_hybrid
        self.chunker_revision = chunker_revision

    def _read_only_session(self, scope: DataScope) -> Session:
        session = Session(self.engine)
        timeout_ms = max(1000, min(int(scope.timeout_seconds * 1000), 30000))
        if self.engine.dialect.name == "postgresql":
            session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            session.execute(
                text("SELECT set_config('statement_timeout', :timeout_val, true)"),
                {"timeout_val": f"{timeout_ms}ms"},
            )
        elif self.engine.dialect.name == "sqlite":
            session.execute(text("PRAGMA query_only = ON"))
        return session

    def _corpus_identity(self, session: Session, scope: DataScope) -> str:
        if (
            self.scientific_retrieval is not None
            and self.scientific_retrieval.projection is not None
        ):
            return self.scientific_retrieval.projection.passages.projection_sha256
        stmt = select(DocumentVersionRecord.version_key).order_by(DocumentVersionRecord.version_key)
        if scope.allowed_version_keys:
            stmt = stmt.where(DocumentVersionRecord.version_key.in_(scope.allowed_version_keys))
        keys = tuple(session.scalars(stmt).all())
        return digest("canonical-corpus-v2", *keys)

    def _execute_bounded(
        self,
        tool_name: ToolName,
        scope: DataScope,
        fn: Callable[[], BaseModel],
    ) -> ToolExecutionEnvelope:
        trace_id = f"trace-{uuid4().hex[:16]}"
        started = perf_counter()
        try:
            result_model = fn()
            elapsed_ms = (perf_counter() - started) * 1000.0
            if elapsed_ms > scope.timeout_seconds * 1000.0:
                raise ToolExecutionError(
                    ToolErrorCode.TIMEOUT,
                    f"Tool {tool_name.value} exceeded timeout budget of {scope.timeout_seconds}s",
                    retryable=True,
                )
            payload = result_model.model_dump(mode="json")
            serialized = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            if len(serialized.encode("utf-8")) > scope.max_output_bytes:
                raise ToolExecutionError(
                    ToolErrorCode.BUDGET_EXCEEDED,
                    f"Tool {tool_name.value} output exceeded max_output_bytes "
                    f"({scope.max_output_bytes})",
                )
            output_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            return ToolExecutionEnvelope(
                ok=True,
                tool_name=tool_name,
                provenance=ToolProvenance(
                    tool_name=tool_name,
                    trace_id=trace_id,
                    corpus_identity=str(payload.get("corpus_identity"))
                    if "corpus_identity" in payload
                    else None,
                    canonical_verified=True,
                    duration_ms=round(elapsed_ms, 3),
                    output_sha256=output_sha,
                ),
                payload=payload,
            )
        except ToolExecutionError as error:
            elapsed_ms = (perf_counter() - started) * 1000.0
            err_detail = ToolErrorDetail(
                code=error.code,
                message=error.message,
                tool_name=tool_name,
                trace_id=trace_id,
                retryable=error.retryable,
            )
            err_sha = hashlib.sha256(err_detail.model_dump_json().encode("utf-8")).hexdigest()
            return ToolExecutionEnvelope(
                ok=False,
                tool_name=tool_name,
                provenance=ToolProvenance(
                    tool_name=tool_name,
                    trace_id=trace_id,
                    canonical_verified=False,
                    duration_ms=round(elapsed_ms, 3),
                    output_sha256=err_sha,
                ),
                error=err_detail,
            )
        except ValidationError as error:
            elapsed_ms = (perf_counter() - started) * 1000.0
            err_detail = ToolErrorDetail(
                code=ToolErrorCode.INVALID_INPUT,
                message=f"Contract validation failed: {error.error_count()} error(s)",
                tool_name=tool_name,
                trace_id=trace_id,
                retryable=False,
            )
            err_sha = hashlib.sha256(err_detail.model_dump_json().encode("utf-8")).hexdigest()
            return ToolExecutionEnvelope(
                ok=False,
                tool_name=tool_name,
                provenance=ToolProvenance(
                    tool_name=tool_name,
                    trace_id=trace_id,
                    canonical_verified=False,
                    duration_ms=round(elapsed_ms, 3),
                    output_sha256=err_sha,
                ),
                error=err_detail,
            )
        except SQLAlchemyError:
            elapsed_ms = (perf_counter() - started) * 1000.0
            err_detail = ToolErrorDetail(
                code=ToolErrorCode.BACKEND_UNAVAILABLE,
                message="Canonical PostgreSQL database is unavailable or query timed out",
                tool_name=tool_name,
                trace_id=trace_id,
                retryable=True,
            )
            err_sha = hashlib.sha256(err_detail.model_dump_json().encode("utf-8")).hexdigest()
            return ToolExecutionEnvelope(
                ok=False,
                tool_name=tool_name,
                provenance=ToolProvenance(
                    tool_name=tool_name,
                    trace_id=trace_id,
                    canonical_verified=False,
                    duration_ms=round(elapsed_ms, 3),
                    output_sha256=err_sha,
                ),
                error=err_detail,
            )
        except Exception:
            elapsed_ms = (perf_counter() - started) * 1000.0
            err_detail = ToolErrorDetail(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Unexpected failure executing {tool_name.value}",
                tool_name=tool_name,
                trace_id=trace_id,
                retryable=False,
            )
            err_sha = hashlib.sha256(err_detail.model_dump_json().encode("utf-8")).hexdigest()
            return ToolExecutionEnvelope(
                ok=False,
                tool_name=tool_name,
                provenance=ToolProvenance(
                    tool_name=tool_name,
                    trace_id=trace_id,
                    canonical_verified=False,
                    duration_ms=round(elapsed_ms, 3),
                    output_sha256=err_sha,
                ),
                error=err_detail,
            )

    # ------------------------------------------------------------------
    # 1. list_papers
    # ------------------------------------------------------------------

    def list_papers(self, params: ListPapersInput, scope: DataScope) -> ToolExecutionEnvelope:
        def _run() -> ListPapersOutput:
            if params.document_version_key is not None and not scope.allows_version(
                params.document_version_key
            ):
                raise ToolExecutionError(
                    ToolErrorCode.UNAUTHORIZED_SCOPE,
                    "Requested document_version_key is outside the authorized scope",
                )
            with self._read_only_session(scope) as session:
                stmt = (
                    select(DocumentVersionRecord, DocumentRecord)
                    .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
                    .order_by(DocumentVersionRecord.title, DocumentVersionRecord.version_key)
                )
                if scope.allowed_version_keys:
                    stmt = stmt.where(
                        DocumentVersionRecord.version_key.in_(scope.allowed_version_keys)
                    )
                if params.document_version_key is not None:
                    stmt = stmt.where(
                        DocumentVersionRecord.version_key == params.document_version_key
                    )
                rows = session.execute(stmt).all()
                summaries: list[PaperSummary] = []
                query_lower = params.query_filter.casefold() if params.query_filter else None
                for version, document in rows:
                    identifiers = {
                        ident.namespace: ident.normalized_value
                        for ident in session.scalars(
                            select(DocumentIdentifierRecord).where(
                                DocumentIdentifierRecord.document_id == document.id
                            )
                        )
                    }
                    passage_count = (
                        session.scalar(
                            select(func.count())
                            .select_from(PassageRecord)
                            .where(
                                PassageRecord.document_version_id == version.id,
                                PassageRecord.chunker_revision == self.chunker_revision,
                            )
                        )
                        or 0
                    )
                    if passage_count == 0:
                        continue
                    section_titles = tuple(
                        t
                        for t in session.scalars(
                            select(SectionRecord.title)
                            .where(
                                SectionRecord.document_version_id == version.id,
                                SectionRecord.title.is_not(None),
                            )
                            .order_by(SectionRecord.ordinal)
                        )
                        if t
                    )
                    if query_lower is not None:
                        haystack = " ".join(
                            (
                                version.title,
                                document.canonical_key,
                                identifiers.get("doi") or "",
                                identifiers.get("pmcid") or "",
                                identifiers.get("pmid") or "",
                                *section_titles,
                            )
                        ).casefold()
                        if query_lower not in haystack and not scientific_terms(
                            params.query_filter or ""
                        ).intersection(scientific_terms(haystack)):
                            continue
                    summaries.append(
                        PaperSummary(
                            document_version_key=version.version_key,
                            document_canonical_key=document.canonical_key,
                            title=version.title[:500],
                            doi=identifiers.get("doi"),
                            pmid=identifiers.get("pmid"),
                            pmcid=identifiers.get("pmcid"),
                            language=version.language,
                            passage_count=passage_count,
                            section_titles=section_titles[:15],
                        )
                    )
                corpus_id = self._corpus_identity(session, scope)
                limited = tuple(summaries[: params.limit])
                return ListPapersOutput(
                    papers=limited,
                    total_authorized=len(summaries),
                    corpus_identity=corpus_id,
                )

        return self._execute_bounded(ToolName.LIST_PAPERS, scope, _run)

    # ------------------------------------------------------------------
    # 2. hybrid_search
    # ------------------------------------------------------------------

    def hybrid_search(self, params: HybridSearchInput, scope: DataScope) -> ToolExecutionEnvelope:
        def _run() -> HybridSearchOutput:
            if contains_prompt_injection(params.query):
                raise ToolExecutionError(
                    ToolErrorCode.INJECTION_DETECTED,
                    "Search query contains disallowed instruction-override directives",
                )
            for v_key in params.document_version_keys:
                if not scope.allows_version(v_key):
                    raise ToolExecutionError(
                        ToolErrorCode.UNAUTHORIZED_SCOPE,
                        f"Document version {v_key} is outside the authorized scope",
                    )
            effective_limit = min(params.limit, scope.max_passages_per_call)

            if (
                self.scientific_retrieval is not None
                and self.scientific_retrieval.adapter is not None
                and self.scientific_retrieval.projection is not None
            ):
                return self._hybrid_search_qdrant(params, scope, effective_limit)

            return self._hybrid_search_lexical_fallback(params, scope, effective_limit)

        return self._execute_bounded(ToolName.HYBRID_SEARCH, scope, _run)

    def _hybrid_search_qdrant(
        self, params: HybridSearchInput, scope: DataScope, effective_limit: int
    ) -> HybridSearchOutput:
        assert self.scientific_retrieval is not None  # noqa: S101
        filters: dict[str, str] = {}
        if len(params.document_version_keys) == 1:
            filters["document_version_key"] = params.document_version_keys[0]
        try:
            response = self.scientific_retrieval.retrieve(
                params.query, min(20, max(effective_limit * 2, 10)), filters
            )
        except ScientificUnavailable as error:
            raise ToolExecutionError(
                ToolErrorCode.BACKEND_UNAVAILABLE,
                f"Scientific Qdrant retrieval unavailable: {error.code}",
                retryable=True,
            ) from None

        allowed_set = set(params.document_version_keys) or set(scope.allowed_version_keys)
        section_filter_lower = params.section_filter.casefold() if params.section_filter else None
        candidates = [
            (
                fused.provenance,
                fused.rrf_score,
                fused.lexical.rank,
                fused.dense.rank,
            )
            for fused in response.fusion.hits
        ]
        with self._read_only_session(scope) as session:
            hits = self._verify_projection_candidates(
                session=session,
                candidates=candidates,
                scope=scope,
                allowed_set=allowed_set,
                section_filter_lower=section_filter_lower,
                effective_limit=effective_limit,
            )
            return HybridSearchOutput(
                query=params.query,
                hits=hits,
                corpus_identity=response.projection_sha256,
                retrieval_mode="qdrant_hybrid_rrf",
            )

    def _hybrid_search_lexical_fallback(
        self, params: HybridSearchInput, scope: DataScope, effective_limit: int
    ) -> HybridSearchOutput:
        with self._read_only_session(scope) as session:
            records = list_passage_projection_records(
                session, chunker_revision=self.chunker_revision
            )
            allowed_set = set(params.document_version_keys) or set(scope.allowed_version_keys)
            section_filter_lower = (
                params.section_filter.casefold() if params.section_filter else None
            )
            query_terms = scientific_terms(params.query)
            query_lower = params.query.casefold()
            scored: list[tuple[float, PassageProjectionRecords, tuple[PassageSpanItem, ...]]] = []

            for rec in records:
                item = self._score_canonical_passage(
                    rec=rec,
                    scope=scope,
                    allowed_set=allowed_set,
                    section_filter_lower=section_filter_lower,
                    query_terms=query_terms,
                    query_lower=query_lower,
                )
                if item is not None:
                    scored.append(item)

            scored.sort(key=lambda entry: (-entry[0], entry[1].passage.passage_key))
            hits: list[SearchHitItem] = []
            for rank, (score, rec, spans) in enumerate(scored[:effective_limit], start=1):
                idents = {i.namespace: i.normalized_value for i in rec.identifiers}
                hits.append(
                    SearchHitItem(
                        passage_key=rec.passage.passage_key,
                        document_version_key=rec.version.version_key,
                        document_canonical_key=rec.document.canonical_key[:500],
                        title=rec.version.title[:500],
                        section_key=rec.section.section_key if rec.section else None,
                        section_title=rec.section.title if rec.section else None,
                        section_path=rec.section.structural_path if rec.section else None,
                        text=rec.passage.text,
                        content_sha256=rec.passage.content_sha256,
                        rank=rank,
                        score=round(score, 6),
                        bm25_rank=rank,
                        dense_rank=None,
                        doi=idents.get("doi"),
                        pmid=idents.get("pmid"),
                        pmcid=idents.get("pmcid"),
                        source_spans=spans,
                        canonical_verified=True,
                    )
                )
            return HybridSearchOutput(
                query=params.query,
                hits=tuple(hits),
                corpus_identity=self._corpus_identity(session, scope),
                retrieval_mode="canonical_lexical_fallback",
            )

    @staticmethod
    def _score_canonical_passage(
        *,
        rec: PassageProjectionRecords,
        scope: DataScope,
        allowed_set: set[str],
        section_filter_lower: str | None,
        query_terms: set[str],
        query_lower: str,
    ) -> tuple[float, PassageProjectionRecords, tuple[PassageSpanItem, ...]] | None:
        if allowed_set and rec.version.version_key not in allowed_set:
            return None
        if not scope.allows_version(rec.version.version_key):
            return None
        sec_title = rec.section.title if rec.section is not None else None
        if section_filter_lower and section_filter_lower not in (sec_title or "").casefold():
            return None
        if contains_prompt_injection(rec.passage.text):
            return None
        try:
            fp = source_fingerprint(rec)
        except InvalidProvenanceError as err:
            raise ToolExecutionError(
                ToolErrorCode.PROVENANCE_INVALID,
                "Canonical passage failed source-span reconstruction",
            ) from err

        title_terms = scientific_terms(rec.version.title)
        sec_terms = scientific_terms(sec_title or "")
        text_terms = scientific_terms(rec.passage.text)

        overlap_title = len(query_terms.intersection(title_terms))
        overlap_sec = len(query_terms.intersection(sec_terms))
        overlap_text = len(query_terms.intersection(text_terms))
        exact_phrase = 2.5 if query_lower in rec.passage.text.casefold() else 0.0
        body_bonus = 0.5 if all(s.paragraph.region == "body" for s in rec.source_spans) else 0.0

        score = (
            overlap_title * 2.0 + overlap_sec * 1.5 + overlap_text * 1.0 + exact_phrase + body_bonus
        )
        if score <= 0.0:
            return None
        spans = tuple(
            PassageSpanItem(
                span_key=s[0],
                paragraph_key=s[1],
                paragraph_source_anchor=s[5][:500],
                source_order=idx,
                start_char=s[3],
                end_char=s[4],
            )
            for idx, s in enumerate(fp[6])
        )
        return (score, rec, spans)

    def _verify_projection_candidates(
        self,
        *,
        session: Session,
        candidates: list[tuple[PassageProvenance, float, int | None, int | None]],
        scope: DataScope,
        allowed_set: set[str],
        section_filter_lower: str | None,
        effective_limit: int,
    ) -> tuple[SearchHitItem, ...]:
        hits: list[SearchHitItem] = []
        for prov, score, bm25_rank, dense_rank in candidates:
            if not scope.allows_version(prov.document_version_key):
                continue
            if allowed_set and prov.document_version_key not in allowed_set:
                continue
            if (
                section_filter_lower
                and section_filter_lower not in (prov.section_title or "").casefold()
            ):
                continue
            if contains_prompt_injection(prov.text):
                continue
            records = list_passage_projection_records(
                session,
                chunker_revision=self.chunker_revision,
                passage_keys=(prov.passage_key,),
            )
            if len(records) != 1:
                raise ToolExecutionError(
                    ToolErrorCode.PROVENANCE_INVALID,
                    "Retrieved passage could not be verified against canonical PostgreSQL",
                )
            fp = source_fingerprint(records[0])
            spans = tuple(
                PassageSpanItem(
                    span_key=s[0],
                    paragraph_key=s[1],
                    paragraph_source_anchor=s[5][:500],
                    source_order=idx,
                    start_char=s[3],
                    end_char=s[4],
                )
                for idx, s in enumerate(fp[6])
            )
            hits.append(
                SearchHitItem(
                    passage_key=prov.passage_key,
                    document_version_key=prov.document_version_key,
                    document_canonical_key=prov.document_canonical_key[:500],
                    title=prov.title[:500],
                    section_key=prov.section_key,
                    section_title=prov.section_title,
                    section_path=prov.section_path,
                    text=prov.text,
                    content_sha256=prov.content_sha256,
                    rank=len(hits) + 1,
                    score=round(float(score), 6),
                    bm25_rank=bm25_rank,
                    dense_rank=dense_rank,
                    doi=prov.doi,
                    pmid=prov.pmid,
                    pmcid=prov.pmcid,
                    source_spans=spans,
                    canonical_verified=True,
                )
            )
            if len(hits) >= effective_limit:
                break
        return tuple(hits)

    # ------------------------------------------------------------------
    # 3. search_index (OpenSearch BM25 / Hybrid)
    # ------------------------------------------------------------------

    def search_index(self, params: SearchIndexInput, scope: DataScope) -> ToolExecutionEnvelope:
        def _run() -> SearchIndexOutput:
            if not scope.allow_opensearch_projection:
                raise ToolExecutionError(
                    ToolErrorCode.UNAUTHORIZED_SCOPE,
                    "OpenSearch projection access is disabled for this scope",
                )
            if contains_prompt_injection(params.query):
                raise ToolExecutionError(
                    ToolErrorCode.INJECTION_DETECTED,
                    "Search query contains disallowed instruction-override directives",
                )
            if params.document_version_key is not None and not scope.allows_version(
                params.document_version_key
            ):
                raise ToolExecutionError(
                    ToolErrorCode.UNAUTHORIZED_SCOPE,
                    "Requested document_version_key is outside the authorized scope",
                )
            effective_limit = min(params.limit, scope.max_passages_per_call)
            physical_index, projection_sha, raw_candidates = self._execute_opensearch_query(
                params, effective_limit
            )
            allowed_set = (
                {params.document_version_key}
                if params.document_version_key is not None
                else set(scope.allowed_version_keys)
            )
            with self._read_only_session(scope) as session:
                hits = self._verify_projection_candidates(
                    session=session,
                    candidates=raw_candidates,
                    scope=scope,
                    allowed_set=allowed_set,
                    section_filter_lower=None,
                    effective_limit=effective_limit,
                )
                return SearchIndexOutput(
                    query=params.query,
                    mode=params.mode,
                    physical_index=physical_index[:500],
                    projection_sha256=projection_sha,
                    hits=hits,
                )

        return self._execute_bounded(ToolName.SEARCH_INDEX, scope, _run)

    def _execute_opensearch_query(
        self, params: SearchIndexInput, effective_limit: int
    ) -> tuple[str, str, list[tuple[PassageProvenance, float, int | None, int | None]]]:
        try:
            if params.mode == "hybrid":
                if self.opensearch_hybrid is None:
                    raise ToolExecutionError(
                        ToolErrorCode.BACKEND_UNAVAILABLE,
                        "OpenSearch hybrid retrieval service is not configured",
                    )
                hybrid_res = self.opensearch_hybrid.retrieve(
                    params.query, limit=min(50, effective_limit * 2)
                )
                return (
                    hybrid_res.physical_index,
                    hybrid_res.lexical.projection_sha256,
                    [
                        (
                            h.provenance,
                            h.rrf_score,
                            h.lexical.rank,
                            h.dense.rank,
                        )
                        for h in hybrid_res.fusion.hits
                    ],
                )
            if self.opensearch_bm25 is None:
                raise ToolExecutionError(
                    ToolErrorCode.BACKEND_UNAVAILABLE,
                    "OpenSearch BM25 service is not configured",
                )
            bm25_res = self.opensearch_bm25.search(params.query, limit=min(50, effective_limit * 2))
            return (
                bm25_res.index_schema_revision,
                bm25_res.projection_sha256,
                [(self._hit_to_prov(h), h.score, h.rank, None) for h in bm25_res.hits],
            )
        except (OpenSearchError, QueryEmbeddingUnavailable) as err:
            raise ToolExecutionError(
                ToolErrorCode.BACKEND_UNAVAILABLE,
                f"OpenSearch projection search unavailable: {type(err).__name__}",
                retryable=True,
            ) from None

    @staticmethod
    def _hit_to_prov(hit: SearchHit) -> PassageProvenance:
        return PassageProvenance.model_validate(hit.model_dump(exclude={"rank", "score"}))

    # ------------------------------------------------------------------
    # 4. get_passage
    # ------------------------------------------------------------------

    def get_passage(self, params: GetPassageInput, scope: DataScope) -> ToolExecutionEnvelope:
        def _run() -> GetPassageOutput:
            with self._read_only_session(scope) as session:
                records = list_passage_projection_records(
                    session,
                    chunker_revision=self.chunker_revision,
                    passage_keys=(params.passage_key,),
                )
                if not records:
                    raise ToolExecutionError(
                        ToolErrorCode.RESOURCE_NOT_FOUND,
                        f"Passage {params.passage_key} was not found in canonical storage",
                    )
                rec = records[0]
                if not scope.allows_version(rec.version.version_key):
                    raise ToolExecutionError(
                        ToolErrorCode.UNAUTHORIZED_SCOPE,
                        "Passage belongs to a document version outside the authorized scope",
                    )
                if (
                    params.document_version_key is not None
                    and rec.version.version_key != params.document_version_key
                ):
                    raise ToolExecutionError(
                        ToolErrorCode.PROVENANCE_INVALID,
                        "Passage document_version_key does not match expected version key",
                    )
                if contains_prompt_injection(rec.passage.text):
                    raise ToolExecutionError(
                        ToolErrorCode.INJECTION_DETECTED,
                        "Passage contains disallowed instruction-like content",
                    )
                try:
                    fp = source_fingerprint(rec)
                except InvalidProvenanceError as err:
                    raise ToolExecutionError(
                        ToolErrorCode.PROVENANCE_INVALID,
                        "Canonical passage failed source-span reconstruction",
                    ) from err

                idents = {i.namespace: i.normalized_value for i in rec.identifiers}
                spans = tuple(
                    PassageSpanItem(
                        span_key=s[0],
                        paragraph_key=s[1],
                        paragraph_source_anchor=s[5][:500],
                        source_order=idx,
                        start_char=s[3],
                        end_char=s[4],
                    )
                    for idx, s in enumerate(fp[6])
                )
                prov_sha = evidence_identity("source-v1", fp)
                return GetPassageOutput(
                    passage=PassageDetail(
                        passage_key=rec.passage.passage_key,
                        document_version_key=rec.version.version_key,
                        document_canonical_key=rec.document.canonical_key[:500],
                        title=rec.version.title[:500],
                        language=rec.version.language,
                        chunker_revision=rec.passage.chunker_revision[:500],
                        ordinal=rec.passage.ordinal,
                        token_count=rec.passage.token_count or 0,
                        section_key=rec.section.section_key if rec.section else None,
                        section_title=rec.section.title if rec.section else None,
                        section_path=rec.section.structural_path if rec.section else None,
                        text=rec.passage.text,
                        content_sha256=rec.passage.content_sha256,
                        doi=idents.get("doi"),
                        pmid=idents.get("pmid"),
                        pmcid=idents.get("pmcid"),
                        source_spans=spans,
                        provenance_sha256=prov_sha,
                        canonical_verified=True,
                    )
                )

        return self._execute_bounded(ToolName.GET_PASSAGE, scope, _run)

    # ------------------------------------------------------------------
    # 5. expand_passage_context
    # ------------------------------------------------------------------

    def expand_passage_context(
        self, params: ExpandPassageContextInput, scope: DataScope
    ) -> ToolExecutionEnvelope:
        def _run() -> ExpandPassageContextOutput:
            with self._read_only_session(scope) as session:
                records = list_passage_projection_records(
                    session,
                    chunker_revision=self.chunker_revision,
                    passage_keys=(params.passage_key,),
                )
                if not records:
                    raise ToolExecutionError(
                        ToolErrorCode.RESOURCE_NOT_FOUND,
                        f"Anchor passage {params.passage_key} was not found",
                    )
                rec = records[0]
                if not scope.allows_version(rec.version.version_key):
                    raise ToolExecutionError(
                        ToolErrorCode.UNAUTHORIZED_SCOPE,
                        "Anchor passage is outside the authorized document version scope",
                    )
                try:
                    source_fingerprint(rec)
                except InvalidProvenanceError as err:
                    raise ToolExecutionError(
                        ToolErrorCode.PROVENANCE_INVALID,
                        "Anchor passage failed canonical provenance verification",
                    ) from err

                if rec.section is None or any(
                    s.paragraph.region != "body" for s in rec.source_spans
                ):
                    return ExpandPassageContextOutput(
                        anchor_passage_key=rec.passage.passage_key,
                        document_version_key=rec.version.version_key,
                        paragraphs=(),
                        exclusions=("non_body_or_missing_section",),
                    )

                section_map: dict[Any, SectionRecord] = {rec.section.id: rec.section}
                if params.include_parent_section and rec.section.parent_section_id is not None:
                    parent = session.scalar(
                        select(SectionRecord).where(
                            SectionRecord.id == rec.section.parent_section_id,
                            SectionRecord.document_version_id == rec.version.id,
                        )
                    )
                    if parent is not None:
                        section_map[parent.id] = parent

                opened_keys = {s.paragraph.paragraph_key for s in rec.source_spans}
                ordinals = [s.paragraph.ordinal for s in rec.source_spans]
                low_ord = min(ordinals) - params.sibling_distance
                high_ord = max(ordinals) + params.sibling_distance

                candidate_paragraphs = session.scalars(
                    select(ParagraphRecord)
                    .where(
                        ParagraphRecord.document_version_id == rec.version.id,
                        ParagraphRecord.section_id.in_(tuple(section_map.keys())),
                        ParagraphRecord.ordinal >= low_ord,
                        ParagraphRecord.ordinal <= high_ord,
                    )
                    .order_by(ParagraphRecord.ordinal, ParagraphRecord.paragraph_key)
                ).all()

                items: list[ExpandedParagraphItem] = []
                exclusions: list[str] = []
                for para in candidate_paragraphs:
                    if para.paragraph_key in opened_keys:
                        continue
                    if para.region != "body":
                        exclusions.append(f"non_body:{para.paragraph_key}")
                        continue
                    if contains_prompt_injection(para.text):
                        exclusions.append(f"injection_filtered:{para.paragraph_key}")
                        continue
                    expected_pk = paragraph_key(rec.version.version_key, para.source_anchor)
                    actual_sha = hashlib.sha256(para.text.encode("utf-8")).hexdigest()
                    if para.paragraph_key != expected_pk or para.content_sha256 != actual_sha:
                        raise ToolExecutionError(
                            ToolErrorCode.PROVENANCE_INVALID,
                            "Expanded paragraph failed canonical hash or key verification",
                        )
                    if len(items) >= params.max_paragraphs:
                        exclusions.append(f"max_paragraphs_budget:{para.paragraph_key}")
                        continue
                    sec = section_map.get(para.section_id)
                    items.append(
                        ExpandedParagraphItem(
                            paragraph_key=para.paragraph_key,
                            document_version_key=rec.version.version_key,
                            section_key=sec.section_key if sec else None,
                            section_title=sec.title if sec else None,
                            ordinal=para.ordinal,
                            region="body",
                            source_anchor=para.source_anchor[:500],
                            kind="sibling" if para.section_id == rec.section.id else "parent",
                            text=para.text,
                            content_sha256=para.content_sha256,
                            start_char=0,
                            end_char=len(para.text),
                        )
                    )
                return ExpandPassageContextOutput(
                    anchor_passage_key=rec.passage.passage_key,
                    document_version_key=rec.version.version_key,
                    paragraphs=tuple(items),
                    exclusions=tuple(exclusions),
                )

        return self._execute_bounded(ToolName.EXPAND_PASSAGE_CONTEXT, scope, _run)

    # ------------------------------------------------------------------
    # 6. get_study_metadata
    # ------------------------------------------------------------------

    def get_study_metadata(
        self, params: GetStudyMetadataInput, scope: DataScope
    ) -> ToolExecutionEnvelope:
        def _run() -> GetStudyMetadataOutput:
            if not scope.allows_version(params.document_version_key):
                raise ToolExecutionError(
                    ToolErrorCode.UNAUTHORIZED_SCOPE,
                    "Requested document_version_key is outside the authorized scope",
                )
            with self._read_only_session(scope) as session:
                version = session.scalar(
                    select(DocumentVersionRecord).where(
                        DocumentVersionRecord.version_key == params.document_version_key
                    )
                )
                if version is None:
                    raise ToolExecutionError(
                        ToolErrorCode.RESOURCE_NOT_FOUND,
                        f"Study version {params.document_version_key} was not found",
                    )
                document = session.get(DocumentRecord, version.document_id)
                artifact = session.get(SourceArtifactRecord, version.source_artifact_id)
                if document is None or artifact is None:
                    raise ToolExecutionError(
                        ToolErrorCode.PROVENANCE_INVALID,
                        "Study version is missing its parent document or source artifact",
                    )
                identifiers = {
                    ident.namespace: ident.normalized_value
                    for ident in session.scalars(
                        select(DocumentIdentifierRecord).where(
                            DocumentIdentifierRecord.document_id == document.id
                        )
                    )
                }
                sections = tuple(
                    f"{sec.structural_path}: {sec.title}"
                    for sec in session.scalars(
                        select(SectionRecord)
                        .where(SectionRecord.document_version_id == version.id)
                        .order_by(SectionRecord.ordinal)
                    )
                    if sec.title
                )
                passage_count = (
                    session.scalar(
                        select(func.count())
                        .select_from(PassageRecord)
                        .where(
                            PassageRecord.document_version_id == version.id,
                            PassageRecord.chunker_revision == self.chunker_revision,
                        )
                    )
                    or 0
                )
                paragraph_count = (
                    session.scalar(
                        select(func.count())
                        .select_from(ParagraphRecord)
                        .where(ParagraphRecord.document_version_id == version.id)
                    )
                    or 0
                )
                table_count = (
                    session.scalar(
                        select(func.count())
                        .select_from(DocumentTableRecord)
                        .where(DocumentTableRecord.document_version_id == version.id)
                    )
                    or 0
                )
                figure_count = (
                    session.scalar(
                        select(func.count())
                        .select_from(FigureRecord)
                        .where(FigureRecord.document_version_id == version.id)
                    )
                    or 0
                )
                citation_count = (
                    session.scalar(
                        select(func.count())
                        .select_from(CitationRecord)
                        .where(CitationRecord.document_version_id == version.id)
                    )
                    or 0
                )
                return GetStudyMetadataOutput(
                    study=StudyMetadataDetail(
                        document_version_key=version.version_key,
                        document_canonical_key=document.canonical_key[:500],
                        title=version.title[:500],
                        document_type=document.document_type[:500],
                        language=version.language,
                        doi=identifiers.get("doi"),
                        pmid=identifiers.get("pmid"),
                        pmcid=identifiers.get("pmcid"),
                        source_system=artifact.source_system[:500],
                        source_external_id=artifact.source_external_id[:500],
                        source_artifact_sha256=artifact.content_sha256,
                        parser_revision=version.parser_revision[:500],
                        normalizer_revision=version.normalizer_revision[:500],
                        content_fingerprint=version.content_fingerprint,
                        license_uri=artifact.license_uri,
                        versioned_metadata=dict(version.versioned_metadata),
                        sections=sections[:30],
                        passage_count=passage_count,
                        paragraph_count=paragraph_count,
                        table_count=table_count,
                        figure_count=figure_count,
                        citation_count=citation_count,
                    )
                )

        return self._execute_bounded(ToolName.GET_STUDY_METADATA, scope, _run)

    # ------------------------------------------------------------------
    # 7. extract_study_findings
    # ------------------------------------------------------------------

    def extract_study_findings(
        self, params: ExtractStudyFindingsInput, scope: DataScope
    ) -> ToolExecutionEnvelope:
        def _run() -> ExtractStudyFindingsOutput:
            if params.research_focus and contains_prompt_injection(params.research_focus):
                raise ToolExecutionError(
                    ToolErrorCode.INJECTION_DETECTED,
                    "Research focus contains disallowed instruction-override directives",
                )
            with self._read_only_session(scope) as session:
                records = list_passage_projection_records(
                    session,
                    chunker_revision=self.chunker_revision,
                    passage_keys=params.passage_keys,
                )
                by_key = {r.passage.passage_key: r for r in records}
                if len(by_key) != len(set(params.passage_keys)):
                    raise ToolExecutionError(
                        ToolErrorCode.RESOURCE_NOT_FOUND,
                        "One or more requested passage_keys do not exist in canonical storage",
                    )

                focus_terms: set[str] = (
                    scientific_terms(params.research_focus) if params.research_focus else set()
                )
                findings: list[ProposedStudyFinding] = []
                excluded: list[str] = []

                for pk in params.passage_keys:
                    rec = by_key[pk]
                    rec_findings, exclusion_reason = self._extract_findings_from_record(
                        pk=pk,
                        rec=rec,
                        scope=scope,
                        focus_terms=focus_terms,
                    )
                    if exclusion_reason is not None:
                        excluded.append(exclusion_reason)
                        continue
                    for finding in rec_findings:
                        if len(findings) >= params.max_findings:
                            break
                        findings.append(finding)

                return ExtractStudyFindingsOutput(
                    findings=tuple(findings),
                    excluded_passages=tuple(excluded),
                )

        return self._execute_bounded(ToolName.EXTRACT_STUDY_FINDINGS, scope, _run)

    def _extract_findings_from_record(
        self,
        *,
        pk: str,
        rec: PassageProjectionRecords,
        scope: DataScope,
        focus_terms: set[str],
    ) -> tuple[list[ProposedStudyFinding], str | None]:
        if not scope.allows_version(rec.version.version_key):
            raise ToolExecutionError(
                ToolErrorCode.UNAUTHORIZED_SCOPE,
                f"Passage {pk} belongs to an unauthorized document version",
            )
        if contains_prompt_injection(rec.passage.text):
            return ([], f"injection_detected:{pk}")
        try:
            source_fingerprint(rec)
        except InvalidProvenanceError as err:
            raise ToolExecutionError(
                ToolErrorCode.PROVENANCE_INVALID,
                "Passage failed canonical source-span reconstruction",
            ) from err

        if any(s.paragraph.region != "body" for s in rec.source_spans):
            return ([], f"front_or_back_matter:{pk}")

        idents = {i.namespace: i.normalized_value for i in rec.identifiers}
        study_design = self._extract_design(rec.version.title + " " + rec.passage.text)
        population = self._extract_first(_POPULATION_PATTERNS, rec.passage.text)
        intervention = self._extract_first(
            _INTERVENTION_PATTERNS, rec.version.title + " " + rec.passage.text
        )
        protocol = self._extract_first(_PROTOCOL_PATTERNS, rec.passage.text)
        outcomes = self._extract_all(_OUTCOME_PATTERNS, rec.passage.text, limit=4)

        best_candidates = self._candidate_finding_sentences(rec, focus_terms, outcomes)
        if not best_candidates:
            return ([], f"no_complete_finding_sentence:{pk}")

        best_candidates.sort(key=lambda item: (-item[0], item[5]))
        record_findings: list[ProposedStudyFinding] = []
        is_method_sec = bool(
            rec.section and rec.section.title and "method" in rec.section.title.casefold()
        )
        for (
            _score,
            quote,
            span_key,
            para_key,
            anchor,
            abs_start,
            abs_end,
            stats_found,
        ) in best_candidates[:2]:
            category = (
                "QUANTITATIVE"
                if stats_found
                else ("METHODOLOGICAL" if is_method_sec else "DESCRIPTIVE")
            )
            units_detected = self._extract_units_from_text(quote)
            missing_info: list[str] = []
            if population is None:
                missing_info.append("sample_population_not_stated_in_passage")
            if not stats_found:
                missing_info.append("exact_effect_size_or_p_value_not_in_sentence")

            finding_id = evidence_identity(
                "proposed-study-finding-v2",
                {
                    "passage_key": pk,
                    "span_key": span_key,
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "verbatim_quote": quote,
                },
            )
            quote_lower = quote.casefold()
            has_comp = "vs" in quote_lower or "compared" in quote_lower or "between" in quote_lower
            record_findings.append(
                ProposedStudyFinding(
                    finding_id=finding_id,
                    document_version_key=rec.version.version_key,
                    document_title=rec.version.title[:500],
                    doi=idents.get("doi"),
                    pmcid=idents.get("pmcid"),
                    passage_key=pk,
                    span_key=span_key,
                    paragraph_key=para_key,
                    paragraph_source_anchor=anchor[:500],
                    start_char=abs_start,
                    end_char=abs_end,
                    verbatim_quote=quote,
                    category=category,
                    study_design=study_design,
                    population=population,
                    intervention=intervention,
                    comparator=(
                        "within-subject angle/condition comparison or control group"
                        if has_comp
                        else None
                    ),
                    measurement_protocol=protocol,
                    outcome_measures=outcomes,
                    units=units_detected,
                    reported_statistics=stats_found,
                    missing_information=tuple(missing_info),
                    uncertainty_notes=(
                        "MODEL_PROPOSED extraction from a single passage span; "
                        "requires full-study context and expert adjudication.",
                    ),
                )
            )
        return (record_findings, None)

    def _candidate_finding_sentences(
        self,
        rec: PassageProjectionRecords,
        focus_terms: set[str],
        outcomes: tuple[str, ...],
    ) -> list[tuple[int, str, str, str, str, int, int, tuple[str, ...]]]:
        candidates: list[tuple[int, str, str, str, str, int, int, tuple[str, ...]]] = []
        for lineage in rec.source_spans:
            span, para = lineage.span, lineage.paragraph
            span_text = para.text[span.start_char : span.end_char]
            for sentence, rel_start, rel_end in self._complete_sentences(span_text):
                if len(sentence) < 20 or len(sentence) > 650:
                    continue
                if contains_prompt_injection(sentence):
                    continue
                abs_start = span.start_char + rel_start
                abs_end = span.start_char + rel_end
                if para.text[abs_start:abs_end] != sentence:
                    continue
                sent_terms = scientific_terms(sentence)
                overlap = len(focus_terms.intersection(sent_terms)) if focus_terms else 1
                stats_found = self._extract_all(_STAT_PATTERNS, sentence, limit=5)
                score = overlap * 3 + (2 if stats_found else 0) + (1 if outcomes else 0)
                candidates.append(
                    (
                        score,
                        sentence,
                        span.span_key,
                        para.paragraph_key,
                        para.source_anchor,
                        abs_start,
                        abs_end,
                        stats_found,
                    )
                )
        return candidates

    # ------------------------------------------------------------------
    # 8. compare_studies
    # ------------------------------------------------------------------

    def compare_studies(
        self, params: CompareStudiesInput, scope: DataScope
    ) -> ToolExecutionEnvelope:
        def _run() -> CompareStudiesOutput:
            unique_versions = tuple(dict.fromkeys(params.document_version_keys))
            if len(unique_versions) < 2:
                raise ToolExecutionError(
                    ToolErrorCode.INVALID_INPUT,
                    "compare_studies requires at least two distinct document_version_keys",
                )
            for v_key in unique_versions:
                if not scope.allows_version(v_key):
                    raise ToolExecutionError(
                        ToolErrorCode.UNAUTHORIZED_SCOPE,
                        f"Study version {v_key} is outside the authorized scope",
                    )

            with self._read_only_session(scope) as session:
                by_version = self._load_comparison_records(
                    session, unique_versions, params.passage_keys
                )
                dimensions = tuple(
                    self._build_study_comparison_dimension(v_key, by_version[v_key])
                    for v_key in unique_versions
                )
                matrix = self._synthesize_comparison_matrix(
                    unique_versions=unique_versions,
                    dimensions=dimensions,
                    comparison_aspect=params.comparison_aspect,
                )
                return CompareStudiesOutput(comparison=matrix)

        return self._execute_bounded(ToolName.COMPARE_STUDIES, scope, _run)

    def _load_comparison_records(
        self,
        session: Session,
        unique_versions: tuple[str, ...],
        passage_keys: tuple[str, ...],
    ) -> dict[str, list[PassageProjectionRecords]]:
        all_records = list_passage_projection_records(
            session,
            chunker_revision=self.chunker_revision,
            passage_keys=passage_keys if passage_keys else None,
        )
        by_version: dict[str, list[PassageProjectionRecords]] = {v: [] for v in unique_versions}
        for rec in all_records:
            if rec.version.version_key in by_version:
                by_version[rec.version.version_key].append(rec)

        for v_key in unique_versions:
            if not by_version[v_key]:
                version_records = [
                    r
                    for r in list_passage_projection_records(
                        session, chunker_revision=self.chunker_revision
                    )
                    if r.version.version_key == v_key
                ]
                if not version_records:
                    raise ToolExecutionError(
                        ToolErrorCode.RESOURCE_NOT_FOUND,
                        f"Study version {v_key} was not found in canonical storage",
                    )
                by_version[v_key] = version_records[:6]
        return by_version

    def _build_study_comparison_dimension(
        self, v_key: str, v_recs: list[PassageProjectionRecords]
    ) -> StudyComparisonDimension:
        for r in v_recs:
            try:
                source_fingerprint(r)
            except InvalidProvenanceError as err:
                raise ToolExecutionError(
                    ToolErrorCode.PROVENANCE_INVALID,
                    f"Study {v_key} passage failed canonical verification",
                ) from err
        first = v_recs[0]
        idents = {i.namespace: i.normalized_value for i in first.identifiers}
        combined_text = " ".join([first.version.title, *(r.passage.text for r in v_recs[:6])])
        design = (
            self._extract_design(combined_text)
            or "Experimental sports/biomechanics study (exact design not fully specified)"
        )
        population = (
            self._extract_first(_POPULATION_PATTERNS, combined_text)
            or "Not explicitly stated in inspected passages"
        )
        intervention = (
            self._extract_first(_INTERVENTION_PATTERNS, combined_text)
            or "Plyometric / neuromuscular exercise protocol"
        )
        comparator = (
            "Randomized active comparator / control group"
            if "randomized" in combined_text.casefold()
            or "control group" in combined_text.casefold()
            else "Within-subject condition / angle / pre-post comparison"
        )
        protocol = (
            self._extract_first(_PROTOCOL_PATTERNS, combined_text)
            or "Biomechanical / performance testing protocol"
        )
        outcomes = self._extract_all(_OUTCOME_PATTERNS, combined_text, limit=6) or (
            "Reported neuromuscular / performance outcomes",
        )
        units_list = self._extract_all_units(combined_text) or ("unit_not_explicit",)
        results, supp_pks, supp_sks = self._collect_dimension_results(v_recs[:4])
        missing: list[str] = []
        if "Not explicitly stated" in population:
            missing.append("complete_demographic_inclusion_criteria")
        if units_list == ("unit_not_explicit",):
            missing.append("explicit_measurement_units_in_selected_passages")

        return StudyComparisonDimension(
            document_version_key=v_key,
            title=first.version.title[:500],
            doi=idents.get("doi"),
            pmcid=idents.get("pmcid"),
            study_design=design,
            population=population,
            intervention=intervention,
            control_or_comparator=comparator,
            measurement_protocol=protocol,
            outcomes=outcomes,
            units=units_list,
            reported_results=results or ("See supporting canonical passages.",),
            supporting_passage_keys=supp_pks,
            supporting_span_keys=supp_sks,
            missing_information=tuple(missing),
        )

    def _collect_dimension_results(
        self, recs: list[PassageProjectionRecords]
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        results: list[str] = []
        supp_pks: list[str] = []
        supp_sks: list[str] = []
        for r in recs:
            supp_pks.append(r.passage.passage_key)
            for s in r.source_spans:
                supp_sks.append(s.span.span_key)
                span_slice = s.paragraph.text[s.span.start_char : s.span.end_char]
                for sent, _, _ in self._complete_sentences(span_slice):
                    if 25 <= len(sent) <= 300 and not contains_prompt_injection(sent):
                        results.append(sent)
                        if len(results) >= 3:
                            break
                if len(results) >= 3:
                    break
        return (
            tuple(results[:3]),
            tuple(dict.fromkeys(supp_pks)),
            tuple(dict.fromkeys(supp_sks)),
        )

    @staticmethod
    def _synthesize_comparison_matrix(
        *,
        unique_versions: tuple[str, ...],
        dimensions: tuple[StudyComparisonDimension, ...],
        comparison_aspect: str | None,
    ) -> StudyComparisonMatrix:
        designs = {d.study_design for d in dimensions}
        units_sets: list[set[str]] = [set(d.units) for d in dimensions]
        shared_units: set[str] = (
            set(units_sets[0]).intersection(*units_sets[1:]) if units_sets else set()
        )
        outcome_sets: list[set[str]] = [{o.casefold() for o in d.outcomes} for d in dimensions]
        shared_outcomes: set[str] = (
            set(outcome_sets[0]).intersection(*outcome_sets[1:]) if outcome_sets else set()
        )

        if len(designs) == 1 and shared_outcomes and shared_units - {"unit_not_explicit"}:
            compat = "DIRECTLY_COMPARABLE"
            rationale = (
                "Studies share compatible study designs, overlapping outcome constructs, "
                "and overlapping measurement units, though population and protocol "
                "differences still require caution."
            )
        elif not shared_outcomes and not (shared_units - {"unit_not_explicit"}):
            compat = "INCOMPATIBLE_OUTCOMES_OR_UNITS"
            rationale = (
                "The studies share no outcome terms and no explicit measurement unit; "
                "numerical pooling or direct effect-size equivalence is not supported by the "
                "extracted evidence."
            )
        else:
            compat = "METHODOLOGICALLY_HETEROGENEOUS"
            rationale = (
                "The studies are neither clearly directly comparable nor clearly unrelated: "
                f"{len(designs)} distinct study design(s) "
                f"({', '.join(sorted(designs)) or 'not stated'}); "
                f"{len(shared_outcomes)} shared outcome term(s); "
                f"{len(shared_units - {'unit_not_explicit'})} shared measurement unit(s). "
                "Population, protocol and instrumentation differences (listed per study) must "
                "be weighed before any numerical combination."
            )

        method_diffs = tuple(
            f"{d.pmcid or d.document_version_key[:8]}: design='{d.study_design}', "
            f"population='{d.population}', protocol='{d.measurement_protocol}'"
            for d in dimensions
        )
        outcome_diffs = tuple(
            f"{d.pmcid or d.document_version_key[:8]}: outcomes={list(d.outcomes)}, "
            f"units={list(d.units)}"
            for d in dimensions
        )
        comp_id = evidence_identity(
            "study-comparison-v2",
            {
                "versions": unique_versions,
                "aspect": comparison_aspect,
                "compatibility": compat,
            },
        )
        return StudyComparisonMatrix(
            comparison_id=comp_id,
            studies=dimensions,
            evidence_compatibility=compat,
            compatibility_rationale=rationale,
            shared_constructs=tuple(sorted(shared_outcomes)),
            methodology_differences=method_diffs,
            outcome_and_unit_differences=outcome_diffs,
            uncertainty_and_gaps=(
                "Cross-paper comparisons are MODEL_PROPOSED / PROPOSED syntheses over "
                "canonical passages; heterogeneous interventions and measurement scales "
                "must not be numerically combined.",
            ),
        )

    # ------------------------------------------------------------------
    # 9. calculate_statistics
    # ------------------------------------------------------------------

    def calculate_statistics(
        self, params: CalculateStatisticsInput, scope: DataScope
    ) -> ToolExecutionEnvelope:
        def _run() -> CalculateStatisticsOutput:
            groups = params.groups
            canonical_units = {normalize_unit(g.unit) for g in groups}
            canonical_outcomes = {g.outcome_name.strip().casefold() for g in groups}
            protocols = {
                g.measurement_protocol.strip().casefold()
                for g in groups
                if g.measurement_protocol is not None
            }

            if len(canonical_units) > 1 or len(canonical_outcomes) > 1 or len(protocols) > 1:
                raise ToolExecutionError(
                    ToolErrorCode.INCOMPATIBLE_UNITS_OR_OUTCOMES,
                    "Refusing statistical calculation across incompatible units, outcomes, or "
                    f"measurement protocols: units={sorted(canonical_units)}, "
                    f"outcomes={sorted(canonical_outcomes)}",
                )

            unit = normalize_unit(groups[0].unit)
            outcome_name = groups[0].outcome_name

            if params.calculation == StatisticalCalculationKind.SUMMARY_STATS:
                return self._calc_summary_stats(groups[0], params.calculation, outcome_name, unit)

            if len(groups) != 2:
                raise ToolExecutionError(
                    ToolErrorCode.INVALID_INPUT,
                    f"{params.calculation.value} requires exactly 2 compatible groups",
                )

            g1, g2 = groups[0], groups[1]
            m1 = (
                g1.mean
                if g1.mean is not None
                else (sum(g1.values) / len(g1.values) if g1.values else None)
            )
            m2 = (
                g2.mean
                if g2.mean is not None
                else (sum(g2.values) / len(g2.values) if g2.values else None)
            )
            if m1 is None or m2 is None:
                raise ToolExecutionError(
                    ToolErrorCode.INVALID_INPUT,
                    "Both groups must provide either `mean` or non-empty `values`",
                )

            if params.calculation == StatisticalCalculationKind.MEAN_DIFFERENCE:
                diff = m2 - m1
                return CalculateStatisticsOutput(
                    calculation=params.calculation,
                    outcome_name=outcome_name,
                    unit=unit,
                    compatible=True,
                    result_metrics={
                        "group_1_mean": round(m1, 6),
                        "group_2_mean": round(m2, 6),
                        "mean_difference": round(diff, 6),
                        "absolute_difference": round(abs(diff), 6),
                    },
                    formula_used="mean_2 - mean_1",
                )

            if params.calculation == StatisticalCalculationKind.PERCENTAGE_CHANGE:
                if math.isclose(m1, 0.0, abs_tol=1e-12):
                    raise ToolExecutionError(
                        ToolErrorCode.INVALID_INPUT,
                        "Cannot compute percentage_change when baseline mean is zero",
                    )
                pct = ((m2 - m1) / abs(m1)) * 100.0
                return CalculateStatisticsOutput(
                    calculation=params.calculation,
                    outcome_name=outcome_name,
                    unit=unit,
                    compatible=True,
                    result_metrics={
                        "baseline_mean": round(m1, 6),
                        "comparison_mean": round(m2, 6),
                        "percentage_change": round(pct, 6),
                    },
                    formula_used="((mean_2 - mean_1) / |mean_1|) * 100",
                )

            # HEDGES_G_SAME_UNIT
            sd1, sd2 = g1.sd, g2.sd
            n1 = g1.sample_size or (len(g1.values) if g1.values else None)
            n2 = g2.sample_size or (len(g2.values) if g2.values else None)
            if sd1 is None or sd2 is None or n1 is None or n2 is None or n1 < 2 or n2 < 2:
                raise ToolExecutionError(
                    ToolErrorCode.INVALID_INPUT,
                    "hedges_g_same_unit requires sd and sample_size >= 2 for both groups",
                )
            df = n1 + n2 - 2
            pooled_var = (((n1 - 1) * (sd1**2)) + ((n2 - 1) * (sd2**2))) / df
            if pooled_var <= 0.0:
                raise ToolExecutionError(
                    ToolErrorCode.INVALID_INPUT,
                    "Pooled standard deviation must be positive for effect-size calculation",
                )
            pooled_sd = math.sqrt(pooled_var)
            cohens_d = (m2 - m1) / pooled_sd
            correction = 1.0 - (3.0 / (4.0 * df - 1.0))
            hedges_g = cohens_d * correction
            return CalculateStatisticsOutput(
                calculation=params.calculation,
                outcome_name=outcome_name,
                unit=unit,
                compatible=True,
                result_metrics={
                    "cohens_d": round(cohens_d, 6),
                    "hedges_g": round(hedges_g, 6),
                    "pooled_sd": round(pooled_sd, 6),
                    "degrees_of_freedom": float(df),
                },
                formula_used="((mean_2 - mean_1) / s_pooled) * (1 - 3 / (4*(n1+n2-2) - 1))",
            )

        return self._execute_bounded(ToolName.CALCULATE_STATISTICS, scope, _run)

    @staticmethod
    def _calc_summary_stats(
        group: StatisticalGroupInput,
        calculation: StatisticalCalculationKind,
        outcome_name: str,
        unit: str,
    ) -> CalculateStatisticsOutput:
        if not group.values:
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT,
                "summary_stats requires non-empty `values` on the input group",
            )
        for val in group.values:
            if not math.isfinite(val):
                raise ToolExecutionError(
                    ToolErrorCode.INVALID_INPUT, "All numerical values must be finite"
                )
        n = len(group.values)
        mean_val = sum(group.values) / n
        variance = sum((x - mean_val) ** 2 for x in group.values) / (n - 1) if n > 1 else 0.0
        sd_val = math.sqrt(variance)
        sorted_vals = sorted(group.values)
        mid = n // 2
        median_val = (
            sorted_vals[mid] if n % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        )
        return CalculateStatisticsOutput(
            calculation=calculation,
            outcome_name=outcome_name,
            unit=unit,
            compatible=True,
            result_metrics={
                "n": float(n),
                "mean": round(mean_val, 6),
                "sd": round(sd_val, 6),
                "median": round(median_val, 6),
                "min": round(sorted_vals[0], 6),
                "max": round(sorted_vals[-1], 6),
            },
            formula_used="sample_mean_and_unbiased_sample_sd",
        )

    # ------------------------------------------------------------------
    # 10. verify_claims
    # ------------------------------------------------------------------

    def verify_claims(self, params: VerifyClaimsInput, scope: DataScope) -> ToolExecutionEnvelope:
        def _run() -> VerifyClaimsOutput:
            with self._read_only_session(scope) as session:
                requested_pks = tuple(dict.fromkeys(c.passage_key for c in params.claims))
                records = list_passage_projection_records(
                    session,
                    chunker_revision=self.chunker_revision,
                    passage_keys=requested_pks,
                )
                by_key = {r.passage.passage_key: r for r in records}

                verified_items: list[VerifiedCitationItem] = []
                verified_members: dict[str, EvidenceMember] = {}
                domain_claims: list[Claim] = []
                domain_links: list[EvidenceLink] = []

                for idx, candidate in enumerate(params.claims):
                    item, member, claim_obj, link_obj = self._verify_single_claim(
                        idx=idx,
                        candidate=candidate,
                        by_key=by_key,
                        scope=scope,
                    )
                    verified_items.append(item)
                    if (
                        item.verified
                        and member is not None
                        and claim_obj is not None
                        and link_obj is not None
                    ):
                        verified_members[member.passage_key] = member
                        domain_claims.append(claim_obj)
                        domain_links.append(link_obj)

                verified_count = sum(1 for item in verified_items if item.verified)
                rejected_count = len(verified_items) - verified_count
                all_ok = verified_count == len(verified_items) and verified_count > 0

                evidence_set_key, evidence_graph_key = (
                    self._build_verified_evidence_graph(
                        session=session,
                        scope=scope,
                        question=params.question,
                        verified_members=verified_members,
                        domain_claims=domain_claims,
                        domain_links=domain_links,
                    )
                    if all_ok and verified_members
                    else (None, None)
                )

                return VerifyClaimsOutput(
                    all_verified=all_ok,
                    verified_count=verified_count,
                    rejected_count=rejected_count,
                    items=tuple(verified_items),
                    evidence_set_key=evidence_set_key,
                    evidence_graph_key=evidence_graph_key,
                )

        return self._execute_bounded(ToolName.VERIFY_CLAIMS, scope, _run)

    @staticmethod
    def _verify_single_claim(
        *,
        idx: int,
        candidate: CandidateClaimInput,
        by_key: dict[str, PassageProjectionRecords],
        scope: DataScope,
    ) -> tuple[VerifiedCitationItem, EvidenceMember | None, Claim | None, EvidenceLink | None]:
        rec = by_key.get(candidate.passage_key)
        if rec is None or not scope.allows_version(rec.version.version_key):
            reason = (
                "passage_not_found_in_canonical_db"
                if rec is None
                else "unauthorized_document_version"
            )
            return (
                VerifiedCitationItem(
                    claim_index=idx,
                    claim_text=candidate.claim_text,
                    verbatim_quote=candidate.verbatim_quote,
                    passage_key=candidate.passage_key,
                    span_key=candidate.span_key,
                    document_version_key=None if rec is None else rec.version.version_key,
                    verified=False,
                    rejection_reason=reason,
                ),
                None,
                None,
                None,
            )

        try:
            fp = source_fingerprint(rec)
        except InvalidProvenanceError:
            return (
                VerifiedCitationItem(
                    claim_index=idx,
                    claim_text=candidate.claim_text,
                    verbatim_quote=candidate.verbatim_quote,
                    passage_key=candidate.passage_key,
                    span_key=candidate.span_key,
                    document_version_key=rec.version.version_key,
                    verified=False,
                    rejection_reason="canonical_source_fingerprint_invalid",
                ),
                None,
                None,
                None,
            )

        quote = candidate.verbatim_quote.strip()
        if contains_prompt_injection(quote) or contains_prompt_injection(candidate.claim_text):
            return (
                VerifiedCitationItem(
                    claim_index=idx,
                    claim_text=candidate.claim_text,
                    verbatim_quote=candidate.verbatim_quote,
                    passage_key=candidate.passage_key,
                    span_key=candidate.span_key,
                    document_version_key=rec.version.version_key,
                    verified=False,
                    rejection_reason="injection_directive_detected",
                ),
                None,
                None,
                None,
            )

        matched_span = None
        matched_coords: tuple[int, int] | None = None
        for lineage in rec.source_spans:
            if candidate.span_key is not None and lineage.span.span_key != candidate.span_key:
                continue
            para_text = lineage.paragraph.text
            pos = para_text.find(quote, lineage.span.start_char, lineage.span.end_char)
            if pos >= lineage.span.start_char and pos + len(quote) <= lineage.span.end_char:
                matched_span = lineage
                matched_coords = (pos, pos + len(quote))
                break

        idents = {i.namespace: i.normalized_value for i in rec.identifiers}
        if matched_span is None or matched_coords is None:
            return (
                VerifiedCitationItem(
                    claim_index=idx,
                    claim_text=candidate.claim_text,
                    verbatim_quote=candidate.verbatim_quote,
                    passage_key=candidate.passage_key,
                    span_key=candidate.span_key,
                    document_version_key=rec.version.version_key,
                    document_title=rec.version.title,
                    doi=idents.get("doi"),
                    pmcid=idents.get("pmcid"),
                    verified=False,
                    rejection_reason="verbatim_quote_absent_from_canonical_span",
                ),
                None,
                None,
                None,
            )

        numeric, numeric_reasons = assess_numeric_provenance(candidate.claim_text, quote)
        semantic = assess_semantic_support(candidate.claim_text, quote)
        layer_fields: dict[str, Any] = {
            "quote_verified": True,
            "numeric_provenance": numeric,
            "semantic_support": semantic.status,
            "semantic_support_score": semantic.score,
            "semantic_support_reasons": semantic.reasons,
        }
        span_fields: dict[str, Any] = {
            "span_key": matched_span.span.span_key,
            "document_version_key": rec.version.version_key,
            "document_title": rec.version.title,
            "doi": idents.get("doi"),
            "pmcid": idents.get("pmcid"),
            "paragraph_key": matched_span.paragraph.paragraph_key,
            "paragraph_source_anchor": matched_span.paragraph.source_anchor,
            "start_char": matched_coords[0],
            "end_char": matched_coords[1],
        }
        rejection: str | None = None
        if numeric == "FAILED":
            rejection = f"unverified_numerical_value_in_claim: {'; '.join(numeric_reasons)}"
        elif semantic.status == "UNSUPPORTED":
            rejection = f"claim_not_supported_by_quoted_span: {'; '.join(semantic.reasons)}"
        if rejection is not None:
            return (
                VerifiedCitationItem(
                    claim_index=idx,
                    claim_text=candidate.claim_text,
                    verbatim_quote=candidate.verbatim_quote,
                    passage_key=candidate.passage_key,
                    verified=False,
                    rejection_reason=rejection,
                    **{**layer_fields, **span_fields},
                ),
                None,
                None,
                None,
            )

        member = EvidenceMember(
            passage_key=rec.passage.passage_key,
            version_key=rec.version.version_key,
            span_keys=tuple(s[0] for s in fp[6]),
            provenance_sha256=evidence_identity("source-v1", fp),
        )
        item = VerifiedCitationItem(
            claim_index=idx,
            claim_text=candidate.claim_text,
            verbatim_quote=quote,
            passage_key=rec.passage.passage_key,
            span_key=matched_span.span.span_key,
            document_version_key=rec.version.version_key,
            document_title=rec.version.title,
            doi=idents.get("doi"),
            pmcid=idents.get("pmcid"),
            paragraph_key=matched_span.paragraph.paragraph_key,
            paragraph_source_anchor=matched_span.paragraph.source_anchor,
            start_char=matched_coords[0],
            end_char=matched_coords[1],
            verified=True,
            rejection_reason=None,
            **layer_fields,
        )
        claim_obj = Claim(
            proposition=candidate.claim_text,
            category=candidate.category,
            origin="MODEL_PROPOSED",
            origin_reference=f"agent-run:{scope.run_id}",
            authoring_process="res637-verify-claims-v2",
        )
        link_obj = EvidenceLink(
            claim_key=claim_obj.key,
            evidence_set_key="0" * 64,
            member=member,
            span_key=matched_span.span.span_key,
            relation=candidate.relation,
            rationale=quote,
            authoring_process="res637-verify-claims-v2",
        )
        return (item, member, claim_obj, link_obj)

    def _build_verified_evidence_graph(
        self,
        *,
        session: Session,
        scope: DataScope,
        question: str,
        verified_members: dict[str, EvidenceMember],
        domain_claims: list[Claim],
        domain_links: list[EvidenceLink],
    ) -> tuple[str, str]:
        corpus_id = self._corpus_identity(session, scope)
        ev_set = EvidenceSet(
            question_id=evidence_identity("agent-question-v2", question),
            question=question,
            corpus_id=corpus_id,
            retrieval_revision="agent-hybrid-v2",
            selection_context=f"thread:{scope.thread_id}:run:{scope.run_id}",
            sufficiency="CANDIDATE_SUFFICIENT",
            members=tuple(verified_members.values()),
        )
        dedup_claims = tuple({c.key: c for c in domain_claims}.values())
        rebound_links = tuple(
            EvidenceLink(
                claim_key=lnk.claim_key,
                evidence_set_key=ev_set.key,
                member=lnk.member,
                span_key=lnk.span_key,
                relation=lnk.relation,
                rationale=lnk.rationale,
                authoring_process=lnk.authoring_process,
            )
            for lnk in domain_links
        )
        dedup_links = tuple({lnk.key: lnk for lnk in rebound_links}.values())
        ev_graph = EvidenceGraph(
            evidence_set=ev_set,
            claims=dedup_claims,
            links=dedup_links,
        )
        evidence_graph_key = evidence_identity(
            "evidence-graph-v1", ev_graph.model_dump(mode="json")
        )
        return (ev_set.key, evidence_graph_key)

    # ------------------------------------------------------------------
    # 11. discover_external_literature (restricted connector)
    # ------------------------------------------------------------------

    def discover_external_literature(
        self, params: DiscoverExternalLiteratureInput, scope: DataScope
    ) -> ToolExecutionEnvelope:
        def _run() -> DiscoverExternalLiteratureOutput:
            if not scope.allow_external_discovery:
                return DiscoverExternalLiteratureOutput(
                    enabled=False,
                    pmcid=params.pmcid,
                    status="disabled_by_policy",
                    message=(
                        "External literature discovery is disabled by default to preserve "
                        "canonical corpus authority. Only pre-ingested canonical papers may "
                        "be cited."
                    ),
                    canonical_ingested=False,
                )
            return DiscoverExternalLiteratureOutput(
                enabled=True,
                pmcid=params.pmcid,
                status="preview_only",
                message=(
                    f"External candidate {params.pmcid} noted for operator acquisition; "
                    "never silently ingested into canonical state."
                ),
                canonical_ingested=False,
            )

        return self._execute_bounded(ToolName.DISCOVER_EXTERNAL_LITERATURE, scope, _run)

    # ------------------------------------------------------------------
    # LangChain StructuredTool bindings
    # ------------------------------------------------------------------

    def build_langchain_tools(
        self,
        scope: DataScope,
        *,
        on_tool_executed: Callable[[ToolExecutionEnvelope], None] | None = None,
    ) -> tuple[BaseTool, ...]:
        """Bind all 10 scientific tools as typed LangChain `StructuredTool` objects."""

        def _wrap(
            fn: Callable[[_InputT, DataScope], ToolExecutionEnvelope],
            input_cls: type[_InputT],
        ) -> Callable[..., dict[str, Any]]:
            def _invoke(**kwargs: object) -> dict[str, Any]:
                try:
                    validated = input_cls.model_validate(kwargs)
                except ValidationError as err:
                    trace_id = f"trace-{uuid4().hex[:16]}"
                    tool_name = ToolName(fn.__name__)
                    err_detail = ToolErrorDetail(
                        code=ToolErrorCode.INVALID_INPUT,
                        message=f"Invalid tool arguments: {err.error_count()} validation error(s)",
                        tool_name=tool_name,
                        trace_id=trace_id,
                        retryable=False,
                    )
                    err_sha = hashlib.sha256(
                        err_detail.model_dump_json().encode("utf-8")
                    ).hexdigest()
                    env = ToolExecutionEnvelope(
                        ok=False,
                        tool_name=tool_name,
                        provenance=ToolProvenance(
                            tool_name=tool_name,
                            trace_id=trace_id,
                            canonical_verified=False,
                            duration_ms=0.0,
                            output_sha256=err_sha,
                        ),
                        error=err_detail,
                    )
                    if on_tool_executed is not None:
                        on_tool_executed(env)
                    return env.model_dump(mode="json")

                envelope = fn(validated, scope)
                if on_tool_executed is not None:
                    on_tool_executed(envelope)
                return envelope.model_dump(mode="json")

            return _invoke

        return (
            StructuredTool.from_function(
                func=_wrap(self.list_papers, ListPapersInput),
                name=ToolName.LIST_PAPERS.value,
                description="List authorized scientific papers in the canonical corpus.",
                args_schema=ListPapersInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.hybrid_search, HybridSearchInput),
                name=ToolName.HYBRID_SEARCH.value,
                description=(
                    "Search canonical scientific passages using hybrid RRF or "
                    "verified lexical retrieval."
                ),
                args_schema=HybridSearchInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.search_index, SearchIndexInput),
                name=ToolName.SEARCH_INDEX.value,
                description="Search the OpenSearch passage projection where configured.",
                args_schema=SearchIndexInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.get_passage, GetPassageInput),
                name=ToolName.GET_PASSAGE.value,
                description="Retrieve one canonical passage with verified paragraph source spans.",
                args_schema=GetPassageInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.expand_passage_context, ExpandPassageContextInput),
                name=ToolName.EXPAND_PASSAGE_CONTEXT.value,
                description="Retrieve neighboring body paragraphs in the same or parent section.",
                args_schema=ExpandPassageContextInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.get_study_metadata, GetStudyMetadataInput),
                name=ToolName.GET_STUDY_METADATA.value,
                description="Inspect canonical study identity, identifiers, sections, and counts.",
                args_schema=GetStudyMetadataInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.extract_study_findings, ExtractStudyFindingsInput),
                name=ToolName.EXTRACT_STUDY_FINDINGS.value,
                description=(
                    "Extract structured MODEL_PROPOSED study findings with exact span coordinates."
                ),
                args_schema=ExtractStudyFindingsInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.compare_studies, CompareStudiesInput),
                name=ToolName.COMPARE_STUDIES.value,
                description=(
                    "Compare study designs, populations, interventions, protocols, "
                    "and outcomes across papers."
                ),
                args_schema=CompareStudiesInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.calculate_statistics, CalculateStatisticsInput),
                name=ToolName.CALCULATE_STATISTICS.value,
                description=(
                    "Run deterministic statistical calculations on unit-compatible "
                    "numerical inputs."
                ),
                args_schema=CalculateStatisticsInput,
            ),
            StructuredTool.from_function(
                func=_wrap(self.verify_claims, VerifyClaimsInput),
                name=ToolName.VERIFY_CLAIMS.value,
                description=(
                    "Independently verify candidate claims and exact quotes against "
                    "canonical PostgreSQL spans."
                ),
                args_schema=VerifyClaimsInput,
            ),
        )

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _complete_sentences(text_block: str) -> list[tuple[str, int, int]]:
        sentences: list[tuple[str, int, int]] = []
        pattern = re.compile(r"[^\s.!?].*?[.!?](?=\s|$)")
        for match in pattern.finditer(text_block):
            raw = match.group(0)
            stripped = raw.strip()
            if not stripped:
                continue
            lead = raw.find(stripped)
            start = match.start() + lead
            end = start + len(stripped)
            sentences.append((stripped, start, end))
        return sentences

    @staticmethod
    def _extract_design(text_block: str) -> str | None:
        for label, pattern in _STUDY_DESIGN_PATTERNS:
            if pattern.search(text_block):
                return label
        return None

    @staticmethod
    def _extract_first(patterns: tuple[re.Pattern[str], ...], text_block: str) -> str | None:
        for pattern in patterns:
            m = pattern.search(text_block)
            if m:
                return m.group(0).strip()[:160]
        return None

    @staticmethod
    def _extract_all(
        patterns: tuple[re.Pattern[str], ...], text_block: str, *, limit: int = 5
    ) -> tuple[str, ...]:
        found: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for m in pattern.finditer(text_block):
                val = m.group(0).strip()[:120]
                key = val.casefold()
                if key not in seen:
                    seen.add(key)
                    found.append(val)
                    if len(found) >= limit:
                        return tuple(found)
        return tuple(found)

    @staticmethod
    def _extract_units_from_text(text_block: str) -> str | None:
        units = ScientificToolRegistry._extract_all_units(text_block)
        return ", ".join(units) if units else None

    @staticmethod
    def _extract_all_units(text_block: str) -> tuple[str, ...]:
        matches = re.findall(
            r"(?:\b\d+(?:\.\d+)?\s*)(%|%MVC|N/s|m/s|cm|mm|ms|s|kg|W|N|°|deg\b)",
            text_block,
        )
        normalized = [normalize_unit(u) for u in matches]
        return tuple(dict.fromkeys(normalized))
