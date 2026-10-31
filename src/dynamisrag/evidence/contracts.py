"""Closed selection-v1 contracts. Public traces contain references, never source text."""

from typing import Literal, Self

from pydantic import Field, model_validator

from dynamisrag.domain.evidence import Contract, EvidenceSet, Text
from dynamisrag.domain.identity import evidence_identity
from dynamisrag.domain.values import Sha256Hex
from dynamisrag.embedding.scientific_cpu import (
    DOCUMENT_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_FINGERPRINT,
)
from dynamisrag.search.scientific import ALIAS, COLLECTION


class SelectionPolicy(Contract):
    revision: Literal["selection-v1"] = "selection-v1"
    retrieval_limit: int = Field(default=20, ge=1, le=50)
    max_passages: int = Field(default=5, ge=1, le=50)
    max_documents: int = Field(default=3, ge=1, le=50)
    max_per_document: int = Field(default=5, ge=1, le=50)
    max_per_section: int = Field(default=5, ge=1, le=50)
    prefer_document_diversity: bool = True
    prefer_section_diversity: bool = True
    soft_per_document: int = Field(default=1, ge=1, le=50)
    soft_per_section: int = Field(default=1, ge=1, le=50)
    relax_preferences: bool = True
    max_bytes: int = Field(default=24000, ge=1, le=1000000)
    max_document_bytes: int = Field(default=16000, ge=1, le=1000000)
    max_expansions: int = Field(default=6, ge=0, le=100)
    max_expansion_bytes: int = Field(default=8000, ge=0, le=1000000)
    sibling_distance: int = Field(default=1, ge=0, le=5)
    parent_depth: int = Field(default=1, ge=0, le=3)
    max_document_span: int = Field(default=10, ge=0, le=100)
    max_db_rows: int = Field(default=2000, ge=1, le=10000)
    max_row_bytes: int = Field(default=100000, ge=1, le=1000000)
    statement_timeout_ms: int = Field(default=5000, ge=1, le=30000)
    duration_seconds: int = Field(default=30, ge=1, le=300)
    duplicate_scope: Literal["same-version-exact-or-overlap"] = "same-version-exact-or-overlap"
    tie_break: Literal["fused-rank-then-key"] = "fused-rank-then-key"
    expansion_boundary: Literal["body-same-version-section"] = "body-same-version-section"
    insufficient_behavior: Literal["explicit-no-sentinel"] = "explicit-no-sentinel"

    @property
    def key(self) -> str:
        return evidence_identity(self.revision, self.model_dump(mode="json"))


class RetrievalIdentity(Contract):
    api_revision: Literal["qdrant-retrieval-v1"] = "qdrant-retrieval-v1"
    alias: str = ALIAS
    physical_collection: str = COLLECTION
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dimension: Literal[512] = 512
    document_embedding_config_sha256: Sha256Hex = DOCUMENT_FINGERPRINT
    query_embedding_config_sha256: Sha256Hex = QUERY_FINGERPRINT
    embedding_manifest_sha256: Sha256Hex
    projection_sha256: Sha256Hex

    @property
    def key(self) -> str:
        return evidence_identity("selection-corpus-v1", self.model_dump(mode="json"))


class SelectionRequest(Contract):
    question: Text
    corpus: RetrievalIdentity
    policy: SelectionPolicy = SelectionPolicy()
    depth: Literal["passages", "context"] = "context"
    required_versions: tuple[Sha256Hex, ...] = ()
    allowed_versions: tuple[Sha256Hex, ...] = ()
    minimum_documents: int = Field(default=1, ge=1, le=50)
    required_context: bool = False
    response_requirement: Literal["UNASSESSED", "ABSTAIN", "CORRECT_PREMISE"] = "UNASSESSED"

    @property
    def question_id(self) -> str:
        return evidence_identity("selection-question-v1", self.question)

    @property
    def key(self) -> str:
        return evidence_identity("selection-request-v1", self.model_dump(mode="json"))


class SelectionDecision(Contract):
    passage_key: Sha256Hex
    fused_rank: int | None
    rrf_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    dense_rank: int | None
    dense_score: float | None
    version_key: Sha256Hex
    document_key: str
    section_key: str | None
    disposition: Literal[
        "SELECTED",
        "DUPLICATE",
        "DIVERSITY_DEFERRED",
        "BUDGET_EXCEEDED",
        "SOURCE_INVALID",
        "NOT_SELECTED",
    ]
    reason: Literal[
        "ranked",
        "soft_quota_relaxed",
        "same_version_redundancy",
        "hard_quota",
        "byte_budget",
        "passage_budget",
        "soft_quota",
        "outside_fused_window",
        "canonical_mismatch",
        "outside_authorized_scope",
    ]


class ContextReference(Contract):
    selected_passage_key: Sha256Hex
    version_key: Sha256Hex
    paragraph_key: Sha256Hex
    paragraph_sha256: Sha256Hex
    section_key: str | None
    coordinate_system: Literal["canonical-paragraph-python-chars-half-open"] = (
        "canonical-paragraph-python-chars-half-open"
    )
    start: Literal[0] = 0
    end: int = Field(gt=0)
    utf8_bytes: int = Field(gt=0)
    kind: Literal["parent", "sibling"]


ContextExclusion = Literal[
    "missing_section",
    "missing_sibling",
    "context_budget",
    "row_budget",
    "boundary",
]


class SelectionResult(Contract):
    revision: Literal["selection-result-v1"] = "selection-result-v1"
    request_sha256: Sha256Hex
    question_id: Sha256Hex
    corpus_sha256: Sha256Hex
    policy_sha256: Sha256Hex
    response_sha256: Sha256Hex
    authenticity: Literal["live-service-call", "offline-operator-replay"]
    evidence_set: EvidenceSet | None
    decisions: tuple[SelectionDecision, ...]
    context: tuple[ContextReference, ...]
    context_exclusions: tuple[ContextExclusion, ...]
    selected_bytes: int
    expanded_bytes: int
    documents: int
    versions: int
    sections: int
    missing: tuple[str, ...]
    scientific_sufficiency: Literal["UNASSESSED"] = "UNASSESSED"

    @model_validator(mode="after")
    def consistent_inventory(self) -> Self:
        selected = {d.passage_key for d in self.decisions if d.disposition == "SELECTED"}
        members: set[str] = (
            {m.passage_key for m in self.evidence_set.members} if self.evidence_set else set()
        )
        if selected != members or len({d.passage_key for d in self.decisions}) != len(
            self.decisions
        ):
            raise ValueError("selection disposition inventory mismatch")
        if not members and "no_admissible_evidence" not in self.missing:
            raise ValueError("empty selection requires an explicit insufficiency reason")
        if any(c.selected_passage_key not in members for c in self.context):
            raise ValueError("context has no selected evidence anchor")
        if self.expanded_bytes != sum(c.utf8_bytes for c in self.context):
            raise ValueError("context byte inventory mismatch")
        return self

    @property
    def key(self) -> str:
        return evidence_identity(self.revision, self.model_dump(mode="json"))
