"""Atomic evidence publication within the caller's transaction, never a commit."""

from __future__ import annotations

from hashlib import sha256

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import PassageProjectionRecords, list_passage_projection_records
from dynamisrag.db.evidence_models import (
    ClaimRecord,
    EvidenceLinkRecord,
    EvidenceMembershipRecord,
    EvidenceSetRecord,
)
from dynamisrag.db.models import PassageRecord, PassageSourceSpanRecord
from dynamisrag.domain.evidence import (
    Claim,
    EvidenceGraph,
    EvidenceLink,
    EvidenceMember,
    EvidenceSet,
    IdentityConflictError,
    InvalidProvenanceError,
    MembershipError,
)
from dynamisrag.domain.identity import (
    document_version_key,
    evidence_identity,
    paragraph_key,
    passage_source_span_key,
    source_artifact_key,
)
from dynamisrag.domain.identity import (
    passage_key as derive_passage_key,
)

type SourceFingerprint = tuple[
    str, str, str, str, str, str, tuple[tuple[str, str, str, int, int, str], ...]
]
_SOURCE: TypeAdapter[SourceFingerprint] = TypeAdapter(SourceFingerprint)


def source_fingerprint(record: PassageProjectionRecords) -> SourceFingerprint:
    """Independent Python verification of canonical keys and paragraph coordinates."""
    p, v, a, d = record.passage, record.version, record.artifact, record.document
    if (
        p.document_version_id != v.id
        or v.document_id != d.id
        or v.source_artifact_id != a.id
        or a.artifact_key
        != source_artifact_key(a.source_system, a.source_external_id, a.content_sha256)
        or v.version_key
        != document_version_key(
            d.canonical_key,
            a.artifact_key,
            v.parser_revision,
            v.normalizer_revision,
            v.content_fingerprint,
        )
        or p.passage_key != derive_passage_key(v.version_key, p.chunker_revision, p.ordinal)
        or sha256(p.text.encode()).hexdigest() != p.content_sha256
    ):
        raise InvalidProvenanceError("invalid canonical source identity")
    spans: list[tuple[str, str, str, int, int, str]] = []
    parts: list[str] = []
    previous = None
    end = 0
    for order, lineage in enumerate(record.source_spans):
        span, paragraph = lineage.span, lineage.paragraph
        if (
            span.passage_id != p.id
            or span.paragraph_id != paragraph.id
            or span.document_version_id != v.id
            or paragraph.document_version_id != v.id
            or span.source_order != order
            or span.span_key != passage_source_span_key(p.passage_key, order)
            or paragraph.paragraph_key != paragraph_key(v.version_key, paragraph.source_anchor)
            or not 0 <= span.start_char < span.end_char <= len(paragraph.text)
            or sha256(paragraph.text.encode()).hexdigest() != paragraph.content_sha256
        ):
            raise InvalidProvenanceError("invalid canonical source span")
        if previous == paragraph.id:
            if span.start_char < end or paragraph.text[end : span.start_char].strip():
                raise InvalidProvenanceError("overlapping or gapped source spans")
            parts[-1] += paragraph.text[end : span.end_char]
        else:
            parts.append(paragraph.text[span.start_char : span.end_char])
        previous, end = paragraph.id, span.end_char
        spans.append(
            (
                span.span_key,
                paragraph.paragraph_key,
                paragraph.content_sha256,
                span.start_char,
                span.end_char,
                paragraph.source_anchor,
            )
        )
    if not spans or "\n\n".join(parts) != p.text:
        raise InvalidProvenanceError("canonical passage reconstruction failed")
    return (
        v.version_key,
        a.artifact_key,
        a.content_sha256,
        v.content_fingerprint,
        p.passage_key,
        p.content_sha256,
        tuple(spans),
    )


def canonical_member(session: Session, passage_key: str, version_key: str) -> EvidenceMember:
    """Read and reconstruct exact retained source lineage; fail closed on absent sources."""
    try:
        source = _SOURCE.validate_python(
            session.scalar(
                text("SELECT evidence_source(:pk, :vk)"), {"pk": passage_key, "vk": version_key}
            )
        )
        passage = session.scalars(
            select(PassageRecord).where(PassageRecord.passage_key == passage_key)
        ).one()
        records = list_passage_projection_records(
            session, chunker_revision=passage.chunker_revision, passage_keys=(passage_key,)
        )
        if len(records) != 1 or source_fingerprint(records[0]) != source:
            raise InvalidProvenanceError("Python and SQL canonical verification disagree")
        return EvidenceMember.model_validate(
            {
                "passage_key": passage_key,
                "version_key": version_key,
                "span_keys": tuple(s[0] for s in source[6]),
                "provenance_sha256": evidence_identity("source-v1", source),
            }
        )
    except (DBAPIError, ValidationError) as error:
        raise InvalidProvenanceError("canonical source verification failed") from error


def validate_evidence_set(session: Session, candidate: EvidenceSet) -> EvidenceSet:
    candidate = EvidenceSet.model_validate_json(candidate.model_dump_json())
    for member in candidate.members:
        if canonical_member(session, member.passage_key, member.version_key) != member:
            raise InvalidProvenanceError("member disagrees with canonical source")
    return candidate


def _insert[Row: (ClaimRecord, EvidenceLinkRecord, EvidenceSetRecord)](
    session: Session, row: Row
) -> None:
    # Unique constraints serialize concurrent inserts; compare after ON CONFLICT,
    # rather than using an unsafe SELECT-then-INSERT idempotency check.
    from sqlalchemy.dialects.postgresql import insert

    values = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    session.execute(insert(type(row)).values(**values).on_conflict_do_nothing())
    existing = session.get(type(row), row.key, populate_existing=True)
    if existing is None or existing.payload != row.payload:
        raise IdentityConflictError("persisted identity payload conflict")


def persist_graph(session: Session, candidate: EvidenceGraph) -> EvidenceGraph:
    """Validate, insert and reconstruct before releasing a savepoint. Caller commits."""
    graph = EvidenceGraph.model_validate_json(candidate.model_dump_json())
    with session.begin_nested():
        validate_evidence_set(session, graph.evidence_set)
        evidence_set = graph.evidence_set
        _insert(
            session,
            EvidenceSetRecord(key=evidence_set.key, payload=evidence_set.model_dump(mode="json")),
        )
        for ordinal, member in enumerate(evidence_set.members):
            from sqlalchemy.dialects.postgresql import insert

            passage = session.scalars(
                select(PassageRecord).where(PassageRecord.passage_key == member.passage_key)
            ).one()
            values = {
                "set_key": evidence_set.key,
                "passage_key": member.passage_key,
                "ordinal": ordinal,
                "passage_id": passage.id,
                "version_id": passage.document_version_id,
                "payload": member.model_dump(mode="json"),
            }
            session.execute(
                insert(EvidenceMembershipRecord).values(**values).on_conflict_do_nothing()
            )
            existing = session.get(
                EvidenceMembershipRecord,
                (evidence_set.key, member.passage_key),
                populate_existing=True,
            )
            if existing is None or any(getattr(existing, k) != v for k, v in values.items()):
                raise IdentityConflictError("persisted membership conflict")
        for claim in sorted(graph.claims, key=lambda c: c.key):
            _insert(session, ClaimRecord(key=claim.key, payload=claim.model_dump(mode="json")))
        for link in sorted(graph.links, key=lambda x: x.key):
            span = session.scalars(
                select(PassageSourceSpanRecord).where(
                    PassageSourceSpanRecord.span_key == link.span_key
                )
            ).one()
            _insert(
                session,
                EvidenceLinkRecord(
                    key=link.key,
                    claim_key=link.claim_key,
                    set_key=evidence_set.key,
                    passage_key=link.member.passage_key,
                    span_id=span.id,
                    payload=link.model_dump(mode="json"),
                ),
            )
        rebuilt = get_graph(session, evidence_set.key)
        if rebuilt is None:
            raise IdentityConflictError("graph disappeared before publication")
        if not {x.key for x in graph.links}.issubset({x.key for x in rebuilt.links}):
            raise IdentityConflictError("graph reconstruction mismatch")
        return rebuilt


def get_graph(session: Session, set_key: str) -> EvidenceGraph | None:
    """Revalidate domain identities and canonical lineage on every reconstruction."""
    row = session.get(EvidenceSetRecord, set_key, populate_existing=True)
    if row is None:
        return None
    evidence_set = validate_evidence_set(session, EvidenceSet.model_validate(row.payload))
    if evidence_set.key != row.key:
        raise IdentityConflictError("evidence set identity mismatch")
    memberships = session.scalars(
        select(EvidenceMembershipRecord)
        .where(EvidenceMembershipRecord.set_key == set_key)
        .order_by(EvidenceMembershipRecord.ordinal)
    ).all()
    if [m.payload for m in memberships] != [
        m.model_dump(mode="json") for m in evidence_set.members
    ]:
        raise MembershipError("persisted membership inventory mismatch")
    for membership in memberships:
        passage = session.get(PassageRecord, membership.passage_id, populate_existing=True)
        if (
            passage is None
            or passage.passage_key != membership.passage_key
            or passage.document_version_id != membership.version_id
        ):
            raise InvalidProvenanceError("persisted membership source binding mismatch")
    links: list[EvidenceLink] = []
    claims: dict[str, Claim] = {}
    for record in session.scalars(
        select(EvidenceLinkRecord)
        .where(EvidenceLinkRecord.set_key == set_key)
        .order_by(EvidenceLinkRecord.key)
    ):
        link = EvidenceLink.model_validate(record.payload)
        if link.key != record.key:
            raise IdentityConflictError("link identity mismatch")
        span = session.get(PassageSourceSpanRecord, record.span_id, populate_existing=True)
        membership = session.get(EvidenceMembershipRecord, (set_key, link.member.passage_key))
        if (
            span is None
            or membership is None
            or span.span_key != link.span_key
            or span.passage_id != membership.passage_id
            or span.document_version_id != membership.version_id
            or record.claim_key != link.claim_key
            or record.set_key != link.evidence_set_key
            or record.passage_key != link.member.passage_key
        ):
            raise InvalidProvenanceError("persisted link source binding mismatch")
        claim_row = session.get(ClaimRecord, link.claim_key, populate_existing=True)
        if claim_row is None:
            raise MembershipError("missing claim")
        claim = Claim.model_validate(claim_row.payload)
        if claim.key != claim_row.key:
            raise IdentityConflictError("claim identity mismatch")
        claims[claim.key] = claim
        links.append(link)
    return EvidenceGraph(
        evidence_set=evidence_set,
        claims=tuple(claims[k] for k in sorted(claims)),
        links=tuple(links),
    )
