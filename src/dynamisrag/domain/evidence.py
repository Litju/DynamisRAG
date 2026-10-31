"""Immutable, provisional scientific evidence contracts; no model invocation required."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dynamisrag.domain.identity import evidence_identity
from dynamisrag.domain.values import Sha256Hex

Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class EvidenceGraphError(ValueError):
    """Invalid evidence graph."""


class InvalidProvenanceError(EvidenceGraphError):
    """Source identity or lineage does not agree with canonical records."""


class IdentityConflictError(EvidenceGraphError):
    """A persisted semantic identity has a different payload."""


class MembershipError(EvidenceGraphError):
    """Duplicate, missing or inconsistent membership."""


class UnsupportedReviewError(EvidenceGraphError):
    """Authenticated review infrastructure is unavailable."""


class ReviewState(StrEnum):
    PROPOSED = "PROPOSED"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    REVIEWED = "REVIEWED"
    REJECTED = "REJECTED"


class Relation(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTEXTUAL = "CONTEXTUAL"
    INSUFFICIENT = "INSUFFICIENT"


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceMember(Contract):
    passage_key: Sha256Hex
    version_key: Sha256Hex
    span_keys: tuple[Sha256Hex, ...] = Field(min_length=1)
    provenance_sha256: Sha256Hex

    @model_validator(mode="after")
    def unique_spans(self) -> Self:
        if len(set(self.span_keys)) != len(self.span_keys):
            raise MembershipError("duplicate source spans")
        return self


class EvidenceSet(Contract):
    schema_revision: Literal["evidence-set-v1"] = "evidence-set-v1"
    question_id: Text
    question: Text
    corpus_id: Text
    retrieval_revision: Text | None = None
    selection_context: Text
    constraints: tuple[Text, ...] = ()
    sufficiency: Literal["UNASSESSED", "INSUFFICIENT", "CANDIDATE_SUFFICIENT"] = "UNASSESSED"
    members: tuple[EvidenceMember, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_members(self) -> Self:
        if len({m.passage_key for m in self.members}) != len(self.members):
            raise MembershipError("duplicate passage membership")
        return self

    @property
    def key(self) -> str:
        return evidence_identity("evidence-set-v1", self.model_dump(mode="json"))


class Provisional(Contract):
    review_state: ReviewState = ReviewState.PROPOSED

    @model_validator(mode="after")
    def refuse_unsigned_review(self) -> Self:
        if self.review_state == ReviewState.REVIEWED:
            raise UnsupportedReviewError("REVIEWED requires unavailable authenticated authority")
        return self


class Claim(Provisional):
    schema_revision: Literal["claim-v1"] = "claim-v1"
    proposition: Text
    category: Literal["DESCRIPTIVE", "METHODOLOGICAL", "QUANTITATIVE", "ASSOCIATIONAL", "CAUSAL"]
    population: Text | None = None
    condition: Text | None = None
    comparator: Text | None = None
    outcome: Text | None = None
    relationship: Text | None = None
    numerical_qualifiers: tuple[Text, ...] = ()
    units: Text | None = None
    temporal_context: Text | None = None
    applicability: tuple[Text, ...] = ()
    limitations: tuple[Text, ...] = ()
    origin: Literal["SOURCE_EXTRACTION", "HUMAN_AUTHORED", "MODEL_PROPOSED"]
    origin_reference: Text
    authoring_process: Text

    @property
    def key(self) -> str:
        return evidence_identity("claim-v1", self.model_dump(mode="json"))


class EvidenceLink(Provisional):
    schema_revision: Literal["evidence-link-v1"] = "evidence-link-v1"
    claim_key: Sha256Hex
    evidence_set_key: Sha256Hex
    member: EvidenceMember
    span_key: Sha256Hex
    relation: Relation
    rationale: Text
    authoring_process: Text

    @model_validator(mode="after")
    def span_in_member(self) -> Self:
        if self.span_key not in self.member.span_keys:
            raise MembershipError("link span is absent from member")
        return self

    @property
    def key(self) -> str:
        return evidence_identity("evidence-link-v1", self.model_dump(mode="json"))


class EvidenceGraph(Contract):
    evidence_set: EvidenceSet
    claims: tuple[Claim, ...] = Field(min_length=1)
    links: tuple[EvidenceLink, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def connected(self) -> Self:
        claims = {c.key for c in self.claims}
        if len(claims) != len(self.claims) or len({x.key for x in self.links}) != len(self.links):
            raise MembershipError("duplicate graph objects")
        members = {m.passage_key: m for m in self.evidence_set.members}
        for link in self.links:
            if (
                link.claim_key not in claims
                or link.evidence_set_key != self.evidence_set.key
                or members.get(link.member.passage_key) != link.member
            ):
                raise MembershipError("link does not resolve within this graph")
        if claims != {x.claim_key for x in self.links}:
            raise MembershipError("unlinked claim")
        return self
