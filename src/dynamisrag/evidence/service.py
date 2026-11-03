"""Opt-in read-only bridge from scientific retrieval to the RES-145 EvidenceSet."""

from time import monotonic
from typing import Literal

from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.db.evidence import canonical_member, validate_evidence_set
from dynamisrag.db.models import (
    DocumentIdentifierRecord,
    DocumentVersionRecord,
    ParagraphRecord,
    PassageRecord,
    PassageSourceSpanRecord,
)
from dynamisrag.domain.evidence import EvidenceMember, EvidenceSet, InvalidProvenanceError
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.evidence.context import expand_context
from dynamisrag.evidence.contracts import RetrievalIdentity, SelectionRequest, SelectionResult
from dynamisrag.evidence.retrieval import verify_response
from dynamisrag.evidence.selection import choose
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.retrieval import PassageProvenance
from dynamisrag.search.scientific import CHUNKER, ScientificResponse, ScientificRetrievalService


def _preflight(session: Session, keys: tuple[str, ...], request: SelectionRequest) -> int:
    """Bound source inventories before using RES-145's complete-inventory verifier."""
    p = request.policy
    passages = select(PassageRecord.id).where(PassageRecord.passage_key.in_(keys))
    versions = select(PassageRecord.document_version_id).where(PassageRecord.passage_key.in_(keys))
    documents = select(DocumentVersionRecord.document_id).where(
        DocumentVersionRecord.id.in_(versions)
    )
    spans = (
        session.scalar(
            select(func.count())
            .select_from(PassageSourceSpanRecord)
            .where(PassageSourceSpanRecord.passage_id.in_(passages))
        )
        or 0
    )
    identifiers = (
        session.scalar(
            select(func.count())
            .select_from(DocumentIdentifierRecord)
            .where(DocumentIdentifierRecord.document_id.in_(documents))
        )
        or 0
    )
    max_paragraph = (
        session.scalar(
            select(func.max(func.octet_length(ParagraphRecord.text))).where(
                ParagraphRecord.id.in_(
                    select(PassageSourceSpanRecord.paragraph_id).where(
                        PassageSourceSpanRecord.passage_id.in_(passages)
                    )
                )
            )
        )
        or 0
    )
    max_passage = (
        session.scalar(
            select(func.max(func.octet_length(PassageRecord.text))).where(
                PassageRecord.passage_key.in_(keys)
            )
        )
        or 0
    )
    per_passage_identifiers = (
        session.scalar(
            select(func.count())
            .select_from(PassageRecord)
            .join(
                DocumentVersionRecord, DocumentVersionRecord.id == PassageRecord.document_version_id
            )
            .join(
                DocumentIdentifierRecord,
                DocumentIdentifierRecord.document_id == DocumentVersionRecord.document_id,
            )
            .where(PassageRecord.passage_key.in_(keys))
        )
        or 0
    )
    # Reserve fetched rows for initial projection, two complete canonical reads
    # per candidate, and scalar preflight results. No corpus-wide ORM inventory.
    reserved = 10 * len(keys) + 3 * spans + identifiers + 2 * per_passage_identifiers + 6
    if reserved > p.max_db_rows or max(max_paragraph, max_passage) > p.max_row_bytes:
        raise ValueError("canonical read budget exceeded")
    return p.max_db_rows - reserved


def _select(
    engine: Engine,
    request: SelectionRequest,
    candidate: ScientificResponse,
    authenticity: Literal["live-service-call", "offline-operator-replay"],
) -> SelectionResult:
    started = monotonic()
    request = SelectionRequest.model_validate_json(request.model_dump_json(warnings=False))
    response = verify_response(request, candidate)
    sources = {
        c.provenance.passage_key: c.provenance
        for lane in (response.bm25, response.dense)
        for c in lane.candidates
    }
    with Session(engine) as session:
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        session.execute(
            text("SELECT set_config('statement_timeout', :v, true)"),
            {"v": str(request.policy.statement_timeout_ms)},
        )
        remaining_rows = _preflight(session, tuple(sources), request)
        records = tuple(
            list_passage_projection_records(
                session,
                chunker_revision=CHUNKER,
                passage_keys=tuple(sources),
            )
        )
        by_key = {r.passage.passage_key: r for r in records}
        projected = build_projection_manifest(records, chunker_revision=CHUNKER)
        verified: dict[str, EvidenceMember] = {}
        for document in projected.documents:
            if monotonic() - started > request.policy.duration_seconds:
                raise ValueError("selection duration budget exceeded")
            key = document.passage_key
            actual = PassageProvenance.model_validate(
                {k: v for k, v in document.payload().items() if k in PassageProvenance.model_fields}
            )
            if actual != sources[key]:
                continue
            try:
                # SQL provenance rejection must not poison other candidate reads.
                with session.begin_nested():
                    verified[key] = canonical_member(session, key, actual.document_version_key)
            except InvalidProvenanceError:
                continue
        selected, decisions = choose(request, response, verified)
        size = sum(len(p.text.encode()) for p in selected)
        context, exclusions = expand_context(
            session,
            request,
            tuple(by_key[p.passage_key] for p in selected),
            size,
            row_limit=remaining_rows,
        )
        docs = {p.document_canonical_key for p in selected}
        versions = {p.document_version_key for p in selected}
        missing: list[str] = []
        if request.response_requirement != "UNASSESSED":
            missing.append(f"declared_response_requirement:{request.response_requirement}")
        if not selected:
            missing.append("no_admissible_evidence")
        if len(docs) < request.minimum_documents:
            missing.append("minimum_documents")
        missing.extend(
            f"required_version:{v}" for v in request.required_versions if v not in versions
        )
        if request.required_context and not context:
            missing.append("required_context")
        evidence_set = None
        if selected:
            evidence_set = validate_evidence_set(
                session,
                EvidenceSet(
                    question_id=request.question_id,
                    question=request.question,
                    corpus_id=request.corpus.key,
                    retrieval_revision=response.api_revision,
                    selection_context=evidence_identity(
                        "selection-input-v1",
                        {
                            "request": request.key,
                            "response": evidence_identity(
                                "scientific-response-v1", response.model_dump(mode="json")
                            ),
                            "authenticity": authenticity,
                        },
                    ),
                    constraints=(
                        request.policy.key,
                        "Scientific sufficiency requires independent adjudication",
                    ),
                    sufficiency="INSUFFICIENT" if missing else "UNASSESSED",
                    members=tuple(verified[p.passage_key] for p in selected),
                ),
            )
        if monotonic() - started > request.policy.duration_seconds:
            raise ValueError("selection duration budget exceeded")
        return SelectionResult(
            request_sha256=request.key,
            question_id=request.question_id,
            corpus_sha256=request.corpus.key,
            policy_sha256=request.policy.key,
            response_sha256=evidence_identity(
                "scientific-response-v1", response.model_dump(mode="json")
            ),
            authenticity=authenticity,
            evidence_set=evidence_set,
            decisions=decisions,
            context=context,
            context_exclusions=exclusions,
            selected_bytes=size,
            expanded_bytes=sum(c.utf8_bytes for c in context),
            documents=len(docs),
            versions=len(versions),
            sections=len({(p.document_version_key, p.section_key) for p in selected}),
            missing=tuple(missing),
        )


def select_offline(
    engine: Engine,
    request: SelectionRequest,
    response: ScientificResponse,
) -> SelectionResult:
    """Operator replay only: a response cannot authenticate its originating question."""
    return _select(engine, request, response, "offline-operator-replay")


def select_live(service: ScientificRetrievalService, request: SelectionRequest) -> SelectionResult:
    """Trusted application-owned service executes the exact question; no supplied response."""
    request = SelectionRequest.model_validate_json(request.model_dump_json(warnings=False))
    projection = service.projection
    if projection is None:
        raise ValueError("retrieval service not initialized")
    expected = RetrievalIdentity(
        embedding_manifest_sha256=projection.embeddings.manifest_sha256,
        projection_sha256=projection.passages.projection_sha256,
    )
    if request.corpus != expected or projection.collection_name != expected.physical_collection:
        raise ValueError("requested corpus differs from trusted service projection")
    response = service.retrieve(request.question, request.policy.retrieval_limit, {})
    return _select(service.engine, request, response, "live-service-call")
