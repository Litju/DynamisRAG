"""Live RES-146 to provisional RES-145 structured synthesis path."""

from __future__ import annotations

import hashlib
from threading import BoundedSemaphore
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.db.evidence import canonical_member, validate_evidence_set
from dynamisrag.db.models import DocumentVersionRecord, ParagraphRecord
from dynamisrag.domain.evidence import EvidenceGraph, EvidenceSet, InvalidProvenanceError, Relation
from dynamisrag.domain.identity import evidence_identity, paragraph_key
from dynamisrag.evidence.contracts import SelectionDecision, SelectionRequest, SelectionResult
from dynamisrag.evidence.service import select_live
from dynamisrag.generation.ollama import (
    DIGEST,
    MODEL,
    OPTIONS,
    PROMPT_BYTES,
    RUNTIME,
    GenerationUnavailableError,
)
from dynamisrag.generation.provider import (
    ContextEvidence,
    ContextParagraph,
    ContextSpan,
    DraftClaim,
    DraftSynthesis,
    ExecutionConditions,
    ExecutionMetrics,
    GenerationContext,
    GenerationProvider,
    GenerationSemantics,
    GenerationSettings,
    serialize_generation_context,
)
from dynamisrag.generation.scientific import scientific_terms
from dynamisrag.generation.structured import POLICY_SHA256, SYSTEM
from dynamisrag.generation.validation import VerifiedSource, VerifiedSpan, build_graph
from dynamisrag.search.qdrant import QdrantError, QdrantProjector
from dynamisrag.search.scientific import (
    CHUNKER,
    ScientificRetrievalService,
    ScientificUnavailable,
)

_LIMITATION = (
    "Structural source and quotation checks do not establish entailment, study validity,"
    " applicability, or scientific approval. All model relationships remain provisional."
)
_CONTEXT_POLICY_REVISION = "res147-context-span-first-v1"


class ContextSpanCoverage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    span_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    query_overlap_terms: int = Field(ge=0)
    text_bytes: int = Field(gt=0)
    included: bool
    inclusion_order: int | None = Field(default=None, ge=1)
    byte_cost: int | None = Field(default=None, ge=1)
    required_prompt_bytes: int | None = Field(default=None, ge=0)
    reason: (
        Literal[
            "span_byte_limit",
            "span_count_limit",
            "passage_count_limit",
            "prompt_byte_budget",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def consistent_disposition(self) -> ContextSpanCoverage:
        if self.included != (self.inclusion_order is not None and self.byte_cost is not None):
            raise ValueError("included span requires its order and byte cost")
        if self.included == (self.reason is not None):
            raise ValueError("span exclusion reason must match its disposition")
        if self.end_char <= self.start_char:
            raise ValueError("span coordinates must be a non-empty half-open range")
        return self


class ContextPassageCoverage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    passage_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_version_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    fused_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = Field(default=None, ge=0)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = Field(default=None, ge=0)
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    spans: tuple[ContextSpanCoverage, ...]


class ContextParagraphCoverage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    paragraph_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    included: bool
    inclusion_order: int | None = Field(default=None, ge=1)
    byte_cost: int | None = Field(default=None, ge=1)
    required_prompt_bytes: int | None = Field(default=None, ge=0)
    reason: (
        Literal[
            "anchor_passage_not_included",
            "paragraph_count_limit",
            "paragraph_byte_limit",
            "prompt_byte_budget",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def consistent_disposition(self) -> ContextParagraphCoverage:
        if self.included != (self.inclusion_order is not None and self.byte_cost is not None):
            raise ValueError("included paragraph requires its order and byte cost")
        if self.included == (self.reason is not None):
            raise ValueError("paragraph exclusion reason must match its disposition")
        return self


class ContextCoverageReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    policy_revision: Literal["res147-context-span-first-v1"] = _CONTEXT_POLICY_REVISION
    context_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_bytes: int = Field(ge=1, le=PROMPT_BYTES)
    base_prompt_bytes: int = Field(gt=0)
    context_bytes: int = Field(gt=0)
    selected_passages: tuple[ContextPassageCoverage, ...]
    included_span_keys: tuple[str, ...]
    excluded_span_keys: tuple[str, ...]
    expanded_context: tuple[ContextParagraphCoverage, ...] = ()

    @model_validator(mode="after")
    def consistent_inventory(self) -> ContextCoverageReceipt:
        all_spans = [span.span_key for passage in self.selected_passages for span in passage.spans]
        if (
            len(set(all_spans)) != len(all_spans)
            or set(all_spans) != set(self.included_span_keys) | set(self.excluded_span_keys)
            or set(self.included_span_keys) & set(self.excluded_span_keys)
        ):
            raise ValueError("context coverage span inventory is inconsistent")
        if self.context_bytes > self.budget_bytes:
            raise ValueError("context coverage exceeds its byte budget")
        if (
            self.base_prompt_bytes
            + sum(
                item.byte_cost or 0 for passage in self.selected_passages for item in passage.spans
            )
            + sum(item.byte_cost or 0 for item in self.expanded_context)
            != self.context_bytes
        ):
            raise ValueError("context coverage byte accounting is inconsistent")
        included_orders = [
            item.inclusion_order
            for passage in self.selected_passages
            for item in passage.spans
            if item.included and item.inclusion_order is not None
        ] + [
            item.inclusion_order
            for item in self.expanded_context
            if item.included and item.inclusion_order is not None
        ]
        if sorted(included_orders) != list(range(1, len(included_orders) + 1)):
            raise ValueError("context coverage inclusion order is inconsistent")
        return self


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    selection: SelectionRequest
    max_claims: int = Field(default=5, ge=1, le=5)
    max_context_bytes: int = Field(default=PROMPT_BYTES, ge=512, le=PROMPT_BYTES)

    @model_validator(mode="after")
    def bounded_question(self) -> SynthesisRequest:
        if len(self.selection.question) > 1000:
            raise ValueError("question exceeds generation context limit")
        empty_context = GenerationContext(question=self.selection.question, evidence=(), context=())
        if len((SYSTEM + serialize_generation_context(empty_context)).encode("utf-8")) > (
            self.max_context_bytes
        ):
            raise ValueError(
                "generation context budget cannot contain the system prompt and question"
            )
        return self


class StructuredSynthesisResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    revision: Literal["structured-synthesis-v1"] = "structured-synthesis-v1"
    status: Literal[
        "CANDIDATE_GROUNDED",
        "INSUFFICIENT_EVIDENCE",
        "UNRESOLVED_CONTRADICTION",
        "INVALID_GENERATION",
        "PROVIDER_UNAVAILABLE",
    ]
    question_identity: str
    retrieval_identity: str
    retrieval_response_identity: str
    selection_identity: str
    selection_trace: tuple[SelectionDecision, ...]
    evidence_set: EvidenceSet | None
    evidence_set_identity: str | None
    graph: EvidenceGraph | None
    graph_identity: str | None = None
    proposed_claims: tuple[DraftClaim, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    proposed_limitations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = (_LIMITATION,)
    context_bytes: int = 0
    context_exclusions: tuple[str, ...] = ()
    context_coverage: ContextCoverageReceipt | None = None
    generation_semantics: GenerationSemantics | None = None
    execution_conditions: ExecutionConditions | None = None
    execution_metrics: ExecutionMetrics | None = None
    raw_output_sha256: str | None = None
    generation_invoked: bool = False
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def graph_matches_outcome(self) -> StructuredSynthesisResult:
        if (self.graph is not None) != (
            self.status in {"CANDIDATE_GROUNDED", "UNRESOLVED_CONTRADICTION"}
        ):
            raise ValueError("only accepted grounded outcomes carry a complete graph")
        if self.graph is not None and self.graph.evidence_set.key != self.evidence_set_identity:
            raise ValueError("graph EvidenceSet identity mismatch")
        if self.graph is not None:
            expected = evidence_identity("evidence-graph-v1", self.graph.model_dump(mode="json"))
            if self.graph_identity != expected:
                raise ValueError("EvidenceGraph identity mismatch")
        elif self.graph_identity is not None:
            raise ValueError("abstention cannot carry a graph identity")
        if self.generation_invoked != (self.raw_output_sha256 is not None):
            raise ValueError("generation digest must reflect provider invocation")
        return self


def _result(
    selection: SelectionResult,
    status: Literal[
        "CANDIDATE_GROUNDED",
        "INSUFFICIENT_EVIDENCE",
        "UNRESOLVED_CONTRADICTION",
        "INVALID_GENERATION",
        "PROVIDER_UNAVAILABLE",
    ],
    *,
    graph: EvidenceGraph | None = None,
    proposed_claims: tuple[DraftClaim, ...] = (),
    missing_evidence: tuple[str, ...] | None = None,
    proposed_limitations: tuple[str, ...] = (),
    context_bytes: int = 0,
    context_exclusions: tuple[str, ...] | None = None,
    context_coverage: ContextCoverageReceipt | None = None,
    generation_semantics: GenerationSemantics | None = None,
    execution_conditions: ExecutionConditions | None = None,
    execution_metrics: ExecutionMetrics | None = None,
    raw_output_sha256: str | None = None,
    generation_invoked: bool = False,
    diagnostics: tuple[str, ...] = (),
) -> StructuredSynthesisResult:
    evidence_set = selection.evidence_set
    graph_identity = (
        evidence_identity("evidence-graph-v1", graph.model_dump(mode="json"))
        if graph is not None
        else None
    )
    return StructuredSynthesisResult(
        status=status,
        question_identity=selection.question_id,
        retrieval_identity=selection.corpus_sha256,
        retrieval_response_identity=selection.response_sha256,
        selection_identity=selection.key,
        selection_trace=selection.decisions,
        evidence_set=evidence_set,
        evidence_set_identity=evidence_set.key if evidence_set else None,
        graph=graph,
        graph_identity=graph_identity,
        proposed_claims=proposed_claims,
        missing_evidence=selection.missing if missing_evidence is None else missing_evidence,
        proposed_limitations=proposed_limitations,
        context_bytes=context_bytes,
        context_exclusions=(
            selection.context_exclusions if context_exclusions is None else context_exclusions
        ),
        context_coverage=context_coverage,
        generation_semantics=generation_semantics,
        execution_conditions=execution_conditions,
        execution_metrics=execution_metrics,
        raw_output_sha256=raw_output_sha256,
        generation_invoked=generation_invoked,
        diagnostics=diagnostics,
    )


def _read_only(engine: Engine) -> Session:
    session = Session(engine)
    session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
    return session


def materialize_sources(engine: Engine, selection: SelectionResult) -> tuple[VerifiedSource, ...]:
    evidence_set = selection.evidence_set
    if evidence_set is None:
        return ()
    with _read_only(engine) as session:
        verified = validate_evidence_set(session, evidence_set)
        records = list_passage_projection_records(
            session,
            chunker_revision=CHUNKER,
            passage_keys=tuple(m.passage_key for m in verified.members),
        )
        by_key = {record.passage.passage_key: record for record in records}
        if set(by_key) != {m.passage_key for m in verified.members}:
            raise InvalidProvenanceError("canonical passage inventory changed")
        result: list[VerifiedSource] = []
        for member in verified.members:
            actual = canonical_member(session, member.passage_key, member.version_key)
            record = by_key[member.passage_key]
            if actual != member:
                raise InvalidProvenanceError("EvidenceSet membership changed")
            spans = tuple(
                VerifiedSpan(
                    span.span.span_key,
                    span.paragraph.text[span.span.start_char : span.span.end_char],
                    span.paragraph.text,
                    span.span.start_char,
                    span.span.end_char,
                )
                for span in record.source_spans
            )
            result.append(VerifiedSource(member=member, text=record.passage.text, spans=spans))
        return tuple(result)


def _context(  # noqa: PLR0912, PLR0915 -- emits per-item audit diagnostics
    engine: Engine,
    question: str,
    sources: tuple[VerifiedSource, ...],
    selection: SelectionResult,
    max_bytes: int,
) -> tuple[GenerationContext, tuple[str, ...], int, ContextCoverageReceipt]:
    """Pack complete canonical spans by query overlap, then verified context paragraphs."""
    if selection.evidence_set is None:
        raise InvalidProvenanceError("context packing requires a selected EvidenceSet")
    members = {member.passage_key: member for member in selection.evidence_set.members}
    decisions = {
        decision.passage_key: decision
        for decision in selection.decisions
        if decision.disposition == "SELECTED"
    }
    by_key = {source.member.passage_key: source for source in sources}
    if (
        len(by_key) != len(sources)
        or set(by_key) != set(members)
        or set(decisions) != set(members)
        or any(source.member != members[key] for key, source in by_key.items())
        or any(decisions[key].version_key != member.version_key for key, member in members.items())
    ):
        raise InvalidProvenanceError("context sources differ from selected EvidenceSet")

    ordered_sources = tuple(
        sorted(
            sources,
            key=lambda source: (
                decisions[source.member.passage_key].fused_rank
                if decisions[source.member.passage_key].fused_rank is not None
                else 2**31,
                source.member.passage_key,
            ),
        )
    )
    terms = scientific_terms(question)
    candidates: list[tuple[VerifiedSource, VerifiedSpan, SelectionDecision, int, int, int]] = []
    overlap_by_span: dict[str, int] = {}
    all_span_keys: list[str] = []
    for source_order, source in enumerate(ordered_sources):
        keys = tuple(span.span_key for span in source.spans)
        if (
            len(set(keys)) != len(keys)
            or set(keys) != set(source.member.span_keys)
            or any(
                span.start_char < 0
                or span.end_char <= span.start_char
                or span.end_char > len(span.paragraph_text)
                or span.paragraph_text[span.start_char : span.end_char] != span.text
                for span in source.spans
            )
        ):
            raise InvalidProvenanceError("canonical source span inventory or coordinates changed")
        all_span_keys.extend(keys)
        decision = decisions[source.member.passage_key]
        for span_order, span in enumerate(source.spans):
            overlap = len(terms.intersection(scientific_terms(span.text)))
            overlap_by_span[span.span_key] = overlap
            candidates.append((source, span, decision, overlap, source_order, span_order))
    if len(set(all_span_keys)) != len(all_span_keys):
        raise InvalidProvenanceError("canonical source span identity is duplicated")

    def priority(
        item: tuple[VerifiedSource, VerifiedSpan, SelectionDecision, int, int, int],
    ) -> tuple[int, int, float, int, float, int, float, int, int, str]:
        _source, span, decision, overlap, source_order, span_order = item
        return (
            -overlap,
            decision.fused_rank if decision.fused_rank is not None else 2**31,
            -(decision.rrf_score if decision.rrf_score is not None else float("-inf")),
            decision.bm25_rank if decision.bm25_rank is not None else 2**31,
            -(decision.bm25_score if decision.bm25_score is not None else float("-inf")),
            decision.dense_rank if decision.dense_rank is not None else 2**31,
            -(decision.dense_score if decision.dense_score is not None else float("-inf")),
            source_order,
            span_order,
            span.span_key,
        )

    candidates.sort(key=priority)
    evidence: list[ContextEvidence] = []
    excluded: list[str] = []
    span_coverage: dict[str, ContextSpanCoverage] = {}
    included_span_keys: list[str] = []
    excluded_span_keys: list[str] = []
    context = GenerationContext(question=question, evidence=(), context=())
    encoded = serialize_generation_context(context)
    current_bytes = len((SYSTEM + encoded).encode("utf-8"))
    if current_bytes > max_bytes:
        raise ValueError("generation context budget cannot contain the system prompt and question")
    for source, span, _decision, overlap, _source_order, _span_order in candidates:
        text_bytes = len(span.text.encode("utf-8"))
        span_reason: (
            Literal[
                "span_byte_limit",
                "span_count_limit",
                "passage_count_limit",
                "prompt_byte_budget",
            ]
            | None
        ) = None
        evidence_index = next(
            (i for i, item in enumerate(evidence) if item.passage_key == source.member.passage_key),
            None,
        )
        if text_bytes > 12000 or len(span.text) > 12000:
            span_reason = "span_byte_limit"
        elif evidence_index is None and len(evidence) >= 8:
            span_reason = "passage_count_limit"
        elif evidence_index is not None and len(evidence[evidence_index].spans) >= 16:
            span_reason = "span_count_limit"

        required_prompt_bytes: int | None = None
        if span_reason is None:
            context_span = ContextSpan(span_key=span.span_key, text=span.text)
            if evidence_index is None:
                candidate_evidence = [
                    *evidence,
                    ContextEvidence(
                        passage_key=source.member.passage_key,
                        document_version_key=source.member.version_key,
                        spans=(context_span,),
                    ),
                ]
            else:
                existing = evidence[evidence_index]
                candidate_evidence = list(evidence)
                candidate_evidence[evidence_index] = ContextEvidence(
                    passage_key=existing.passage_key,
                    document_version_key=existing.document_version_key,
                    spans=(*existing.spans, context_span),
                )
            candidate = GenerationContext(
                question=question, evidence=tuple(candidate_evidence), context=()
            )
            candidate_json = serialize_generation_context(candidate)
            required_prompt_bytes = len((SYSTEM + candidate_json).encode("utf-8"))
            if required_prompt_bytes > max_bytes:
                span_reason = "prompt_byte_budget"
            else:
                evidence = candidate_evidence
                context = candidate
                incremental_cost = required_prompt_bytes - current_bytes
                current_bytes = required_prompt_bytes
                included_span_keys.append(span.span_key)
                span_coverage[span.span_key] = ContextSpanCoverage(
                    span_key=span.span_key,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    query_overlap_terms=overlap,
                    text_bytes=text_bytes,
                    included=True,
                    inclusion_order=len(included_span_keys),
                    byte_cost=incremental_cost,
                )
                continue

        excluded_span_keys.append(span.span_key)
        excluded.append(f"{span_reason}:{source.member.passage_key}:{span.span_key}")
        span_coverage[span.span_key] = ContextSpanCoverage(
            span_key=span.span_key,
            start_char=span.start_char,
            end_char=span.end_char,
            query_overlap_terms=overlap,
            text_bytes=text_bytes,
            included=False,
            required_prompt_bytes=required_prompt_bytes,
            reason=span_reason,
        )

    allowed = {item.passage_key for item in evidence}
    paragraphs: list[ContextParagraph] = []
    paragraph_coverage: list[ContextParagraphCoverage] = []
    with _read_only(engine) as session:
        for ref in selection.context:
            if ref.selected_passage_key not in allowed:
                excluded.append(f"expanded_context_anchor_not_included:{ref.paragraph_key}")
                paragraph_coverage.append(
                    ContextParagraphCoverage(
                        paragraph_key=ref.paragraph_key,
                        included=False,
                        reason="anchor_passage_not_included",
                    )
                )
                continue
            paragraph = session.scalar(
                select(ParagraphRecord).where(ParagraphRecord.paragraph_key == ref.paragraph_key)
            )
            version = (
                session.get(DocumentVersionRecord, paragraph.document_version_id)
                if paragraph is not None
                else None
            )
            if (
                paragraph is None
                or version is None
                or version.version_key != ref.version_key
                or ref.version_key
                != next(
                    s.member.version_key
                    for s in sources
                    if s.member.passage_key == ref.selected_passage_key
                )
                or paragraph.paragraph_key
                != paragraph_key(version.version_key, paragraph.source_anchor)
                or paragraph.content_sha256 != ref.paragraph_sha256
                or hashlib.sha256(paragraph.text.encode()).hexdigest() != paragraph.content_sha256
                or ref.end != len(paragraph.text)
                or ref.utf8_bytes != len(paragraph.text.encode("utf-8"))
            ):
                raise InvalidProvenanceError("expanded context identity or coordinates changed")
            item = ContextParagraph(
                paragraph_key=ref.paragraph_key,
                kind=ref.kind,
                text=paragraph.text,
            )
            paragraph_reason: (
                Literal[
                    "paragraph_count_limit",
                    "paragraph_byte_limit",
                    "prompt_byte_budget",
                ]
                | None
            ) = None
            if len(paragraphs) >= 12:
                paragraph_reason = "paragraph_count_limit"
            elif len(paragraph.text) > 12000:
                paragraph_reason = "paragraph_byte_limit"
            required_prompt_bytes: int | None = None
            if paragraph_reason is None:
                candidate = GenerationContext(
                    question=question, evidence=tuple(evidence), context=(*paragraphs, item)
                )
                candidate_json = serialize_generation_context(candidate)
                required_prompt_bytes = len((SYSTEM + candidate_json).encode("utf-8"))
                if required_prompt_bytes > max_bytes:
                    paragraph_reason = "prompt_byte_budget"
                else:
                    incremental_cost = required_prompt_bytes - current_bytes
                    current_bytes = required_prompt_bytes
                    context = candidate
                    paragraphs.append(item)
                    paragraph_coverage.append(
                        ContextParagraphCoverage(
                            paragraph_key=ref.paragraph_key,
                            included=True,
                            inclusion_order=len(included_span_keys) + len(paragraphs),
                            byte_cost=incremental_cost,
                        )
                    )
                    continue
            excluded.append(f"expanded_context_budget:{ref.paragraph_key}")
            paragraph_coverage.append(
                ContextParagraphCoverage(
                    paragraph_key=ref.paragraph_key,
                    included=False,
                    required_prompt_bytes=required_prompt_bytes,
                    reason=paragraph_reason,
                )
            )

    encoded = serialize_generation_context(context)
    size = len((SYSTEM + encoded).encode("utf-8"))
    if size > max_bytes:
        raise ValueError("generation context exceeds configured byte limit")
    passage_coverage = tuple(
        ContextPassageCoverage(
            passage_key=source.member.passage_key,
            document_version_key=source.member.version_key,
            fused_rank=decisions[source.member.passage_key].fused_rank,
            rrf_score=decisions[source.member.passage_key].rrf_score,
            bm25_rank=decisions[source.member.passage_key].bm25_rank,
            bm25_score=decisions[source.member.passage_key].bm25_score,
            dense_rank=decisions[source.member.passage_key].dense_rank,
            dense_score=decisions[source.member.passage_key].dense_score,
            spans=tuple(span_coverage[span.span_key] for span in source.spans),
        )
        for source in ordered_sources
    )
    coverage = ContextCoverageReceipt(
        context_identity=context.key,
        budget_bytes=max_bytes,
        base_prompt_bytes=len(
            (
                SYSTEM
                + serialize_generation_context(
                    GenerationContext(question=question, evidence=(), context=())
                )
            ).encode("utf-8")
        ),
        context_bytes=size,
        selected_passages=passage_coverage,
        included_span_keys=tuple(included_span_keys),
        excluded_span_keys=tuple(excluded_span_keys),
        expanded_context=tuple(paragraph_coverage),
    )
    if not included_span_keys:
        excluded.append("no_complete_source_span_fits_generation_budget")
    return context, tuple(excluded), size, coverage


class StructuredSynthesisService:
    """Owns live retrieval, canonical verification, inference, and atomic validation."""

    def __init__(
        self,
        retrieval: ScientificRetrievalService,
        provider: GenerationProvider,
    ) -> None:
        self.retrieval = retrieval
        self.engine = retrieval.engine
        self.provider = provider
        self.gate = BoundedSemaphore(1)

    def synthesize(  # noqa: PLR0911, PLR0912 -- explicit fail-closed outcomes
        self, request: SynthesisRequest
    ) -> StructuredSynthesisResult:
        request = SynthesisRequest.model_validate_json(request.model_dump_json())
        selection = SelectionResult.model_validate_json(
            select_live(self.retrieval, request.selection).model_dump_json()
        )
        if (
            selection.authenticity != "live-service-call"
            or selection.request_sha256 != request.selection.key
        ):
            raise InvalidProvenanceError("synthesis requires this service's live selection")
        if selection.evidence_set is None:
            return _result(selection, "INSUFFICIENT_EVIDENCE")
        sources = materialize_sources(self.engine, selection)
        context, exclusions, context_bytes, coverage = _context(
            self.engine, request.selection.question, sources, selection, request.max_context_bytes
        )
        if not context.evidence:
            return _result(
                selection,
                "INSUFFICIENT_EVIDENCE",
                context_exclusions=exclusions,
                context_bytes=context_bytes,
                context_coverage=coverage,
                missing_evidence=tuple(
                    dict.fromkeys((*selection.missing, "no_complete_source_fits_generation_budget"))
                ),
            )
        if not self.gate.acquire(blocking=False):
            return _result(
                selection,
                "PROVIDER_UNAVAILABLE",
                context_exclusions=exclusions,
                context_bytes=context_bytes,
                context_coverage=coverage,
                diagnostics=("generation_busy",),
            )
        try:
            try:
                attempt = self.provider.generate(context)
            except GenerationUnavailableError as error:
                return _result(
                    selection,
                    "PROVIDER_UNAVAILABLE",
                    context_exclusions=exclusions,
                    context_bytes=context_bytes,
                    context_coverage=coverage,
                    diagnostics=(error.code,),
                )
        finally:
            self.gate.release()
        raw_digest = hashlib.sha256(attempt.raw_output.encode("utf-8")).hexdigest()
        try:
            try:
                readiness = self.retrieval.readiness()
                projection = self.retrieval.projection
                if (
                    projection is None
                    or readiness.physical_collection != request.selection.corpus.physical_collection
                ):
                    raise InvalidProvenanceError("retrieval projection identity changed")
                QdrantProjector(self.retrieval.client).verify(projection)
            except (QdrantError, ScientificUnavailable):
                raise InvalidProvenanceError(
                    "retrieval projection changed during generation"
                ) from None
            current = materialize_sources(self.engine, selection)
            if current != sources:
                raise InvalidProvenanceError("canonical source changed during generation")
            current_context, _, _, current_coverage = _context(
                self.engine,
                request.selection.question,
                current,
                selection,
                request.max_context_bytes,
            )
            if current_context != context or current_coverage != coverage:
                raise InvalidProvenanceError("canonical expanded context changed during generation")
            draft = DraftSynthesis.model_validate_json(attempt.draft.model_dump_json())
            if (
                len(attempt.raw_output.encode("utf-8")) > 12000
                or DraftSynthesis.model_validate_json(attempt.raw_output) != draft
                or attempt.semantics.model.model != MODEL
                or attempt.semantics.model.digest != DIGEST
                or attempt.semantics.model.runtime_version != RUNTIME
                or attempt.semantics.model.quantization != "Q4_K_M"
                or attempt.semantics.model.parameter_size != "4.0B"
                or attempt.semantics.model.device != "cpu"
                or attempt.semantics.prompt_policy_sha256 != POLICY_SHA256
                or attempt.semantics.context_identity != context.key
                or attempt.semantics.settings != GenerationSettings.model_validate(OPTIONS)
                or attempt.conditions.endpoint != "http://127.0.0.1:11434"
                or attempt.conditions.timeout_seconds != 300
                or attempt.conditions.attempts != 1
                or attempt.conditions.concurrency_limit != 1
                or not attempt.conditions.runtime_available
            ):
                raise ValueError(
                    "generation output or runtime identity does not match pinned semantics"
                )
            if len(draft.claims) > request.max_claims:
                raise ValueError("claim count exceeds request limit")
            missing = tuple(dict.fromkeys((*selection.missing, *draft.missing_evidence)))
            if draft.status == "insufficient_evidence":
                return _result(
                    selection,
                    "INSUFFICIENT_EVIDENCE",
                    context_exclusions=exclusions,
                    context_bytes=context_bytes,
                    context_coverage=coverage,
                    missing_evidence=missing,
                    proposed_limitations=draft.limitations,
                    generation_semantics=attempt.semantics,
                    execution_conditions=attempt.conditions,
                    execution_metrics=attempt.metrics,
                    raw_output_sha256=raw_digest,
                    generation_invoked=True,
                )
            allowed = {
                evidence.passage_key: frozenset(span.span_key for span in evidence.spans)
                for evidence in context.evidence
            }
            graph = build_graph(
                selection.evidence_set,
                sources,
                draft,
                model_identity=attempt.semantics.model.digest,
                policy_sha256=POLICY_SHA256,
                allowed_spans=allowed,
            )
            contradiction = any(link.relation == Relation.CONTRADICTS for link in graph.links)
            return _result(
                selection,
                "UNRESOLVED_CONTRADICTION" if contradiction else "CANDIDATE_GROUNDED",
                graph=graph,
                proposed_claims=draft.claims,
                context_exclusions=exclusions,
                context_bytes=context_bytes,
                context_coverage=coverage,
                missing_evidence=missing,
                proposed_limitations=draft.limitations,
                generation_semantics=attempt.semantics,
                execution_conditions=attempt.conditions,
                execution_metrics=attempt.metrics,
                raw_output_sha256=raw_digest,
                generation_invoked=True,
            )
        except (ValueError, ValidationError, InvalidProvenanceError) as error:
            return _result(
                selection,
                "INVALID_GENERATION",
                context_exclusions=exclusions,
                context_bytes=context_bytes,
                context_coverage=coverage,
                proposed_claims=attempt.draft.claims,
                generation_semantics=attempt.semantics,
                execution_conditions=attempt.conditions,
                execution_metrics=attempt.metrics,
                raw_output_sha256=raw_digest,
                generation_invoked=True,
                diagnostics=(str(error)[:240],),
            )
