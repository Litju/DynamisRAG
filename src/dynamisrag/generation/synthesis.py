"""Live RES-146 to provisional RES-145 structured synthesis path."""

from __future__ import annotations

import hashlib
import json
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
)
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


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    selection: SelectionRequest
    max_claims: int = Field(default=5, ge=1, le=5)
    max_context_bytes: int = Field(default=PROMPT_BYTES, ge=512, le=PROMPT_BYTES)

    @model_validator(mode="after")
    def bounded_question(self) -> SynthesisRequest:
        if len(self.selection.question) > 1000:
            raise ValueError("question exceeds generation context limit")
        if self.max_context_bytes < len(SYSTEM.encode("utf-8")) + 2:
            raise ValueError("generation context budget is smaller than the system prompt")
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


def _context(
    engine: Engine,
    question: str,
    sources: tuple[VerifiedSource, ...],
    selection: SelectionResult,
    max_bytes: int,
) -> tuple[GenerationContext, tuple[str, ...], int]:
    """Pack only whole canonical spans and verified whole context paragraphs."""
    evidence: list[ContextEvidence] = []
    excluded: list[str] = []
    for source in sources:
        spans = tuple(
            ContextSpan(span_key=s.span_key, text=s.text)
            for s in source.spans
            if len(s.text.encode("utf-8")) <= 12000
        )
        if len(spans) != len(source.spans) or not spans or len(spans) > 16:
            excluded.append(f"passage_context_budget:{source.member.passage_key}")
            continue
        item = ContextEvidence(
            passage_key=source.member.passage_key,
            document_version_key=source.member.version_key,
            spans=spans,
        )
        if len(evidence) >= 8:
            excluded.append(f"passage_context_budget:{source.member.passage_key}")
            continue
        candidate = GenerationContext(question=question, evidence=(*evidence, item), context=())
        payload = json.dumps(
            candidate.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        if len((SYSTEM + payload).encode("utf-8")) > max_bytes:
            excluded.append(f"passage_context_budget:{source.member.passage_key}")
            continue
        evidence.append(item)
    allowed = {item.passage_key for item in evidence}
    paragraphs: list[ContextParagraph] = []
    with _read_only(engine) as session:
        for ref in selection.context:
            if ref.selected_passage_key not in allowed:
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
            if len(paragraphs) >= 12 or len(paragraph.text) > 12000:
                excluded.append(f"expanded_context_budget:{ref.paragraph_key}")
                continue
            candidate = GenerationContext(
                question=question, evidence=tuple(evidence), context=(*paragraphs, item)
            )
            payload = json.dumps(
                candidate.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
            )
            if len((SYSTEM + payload).encode("utf-8")) <= max_bytes:
                paragraphs.append(item)
            else:
                excluded.append(f"expanded_context_budget:{ref.paragraph_key}")
    context = GenerationContext(
        question=question, evidence=tuple(evidence), context=tuple(paragraphs)
    )
    encoded = json.dumps(context.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    size = len((SYSTEM + encoded).encode("utf-8"))
    if size > max_bytes:
        raise ValueError("generation context exceeds configured byte limit")
    if len(evidence) != len(sources):
        excluded.append("one_or_more_selected_passages_excluded_by_context_budget")
    return context, tuple(excluded), size


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
        context, exclusions, context_bytes = _context(
            self.engine, request.selection.question, sources, selection, request.max_context_bytes
        )
        if not context.evidence:
            return _result(
                selection,
                "INSUFFICIENT_EVIDENCE",
                context_exclusions=exclusions,
                context_bytes=context_bytes,
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
            current_context, _, _ = _context(
                self.engine,
                request.selection.question,
                current,
                selection,
                request.max_context_bytes,
            )
            if current_context != context:
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
                proposed_claims=attempt.draft.claims,
                generation_semantics=attempt.semantics,
                execution_conditions=attempt.conditions,
                execution_metrics=attempt.metrics,
                raw_output_sha256=raw_digest,
                generation_invoked=True,
                diagnostics=(str(error)[:240],),
            )
