"""DynamisBench v0 contracts, independent of the frozen embedding benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dynamisrag.embedding.scientific_cpu import QUERY_FINGERPRINT
from dynamisrag.search.scientific import ScientificTimings

EXECUTION: dict[str, object] = {
    "api": "qdrant-retrieval-v1",
    "limit": 20,
    "candidate_window": 50,
    "rrf_k": 60,
    "filters": {},
    "query_config": QUERY_FINGERPRINT,
}


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Category = Literal[
    "direct_factual",
    "methodological",
    "numerical",
    "comparative",
    "multi_study",
    "contradictory",
    "temporal",
    "unanswerable",
    "false_premise",
]
Answerability = Literal["answerable", "insufficient_evidence", "false_premise"]
Lane = Literal["bm25", "dense", "fused"]


class Status(StrEnum):
    CANDIDATE = "CANDIDATE"
    AWAITING_EXPERT_REVIEW = "AWAITING_EXPERT_REVIEW"
    EXPERT_APPROVED = "EXPERT_APPROVED"
    QUALIFIED_FOR_REGRESSION = "QUALIFIED_FOR_REGRESSION"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class Source(Contract):
    pmcid: Text
    version_key: Digest
    original_sha256: Digest


class Passage(Contract):
    key: Digest
    pmcid: Text
    version_key: Digest
    content_sha256: Digest
    provenance_sha256: Digest
    anchors: tuple[Text, ...] = Field(min_length=1)
    characters: int = Field(gt=0)


class Corpus(Contract):
    revision: Literal["real-science-v1"] = "real-science-v1"
    embedding_manifest_sha256: Digest
    receipt_sha256: Digest
    projection_sha256: Digest
    collection: Text
    alias: Text
    model_id: Text
    model_revision: Text
    dimension: Literal[512] = 512
    document_config_sha256: Digest
    query_config_sha256: Digest
    sources: tuple[Source, ...] = Field(min_length=3, max_length=3)
    passages: tuple[Passage, ...] = Field(min_length=75, max_length=75)

    @model_validator(mode="after")
    def inventory(self) -> Self:
        sources = {s.pmcid: s for s in self.sources}
        if len(sources) != len(self.sources) or len({p.key for p in self.passages}) != 75:
            raise ValueError("duplicate corpus inventory")
        for p in self.passages:
            if p.pmcid not in sources or sources[p.pmcid].version_key != p.version_key:
                raise ValueError("passage source identity mismatch")
        return self


class Judgment(Contract):
    passage_key: Digest
    grade: int = Field(ge=0, le=3)
    rationale: Text


class Span(Contract):
    passage_key: Digest
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: Digest
    source_anchor: Text


class Question(Contract):
    id: Text
    category: Category
    query: Text = Field(max_length=512)
    articles: tuple[Text, ...] = Field(min_length=1)
    answerability: Answerability
    judgments: tuple[Judgment, ...]
    gold_spans: tuple[Span, ...]
    expected_answer: Text
    citation_behavior: Literal["cite_evidence", "abstain", "correct_premise"]
    population_constraints: Text
    intervention_constraints: Text
    claim_restrictions: tuple[Text, ...] = Field(min_length=1)
    limitations: tuple[Text, ...] = Field(min_length=1)
    human_review_requirements: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def semantics(self) -> Self:
        keys = [j.passage_key for j in self.judgments]
        if len(set(keys)) != len(keys) or len(set(self.articles)) != len(self.articles):
            raise ValueError("duplicate question evidence")
        if self.category == "unanswerable" and self.answerability != "insufficient_evidence":
            raise ValueError("unanswerable category requires abstention")
        if self.category == "false_premise" and self.answerability != "false_premise":
            raise ValueError("false premise cannot be supported as stated")
        expected = {
            "answerable": "cite_evidence",
            "insufficient_evidence": "abstain",
            "false_premise": "correct_premise",
        }
        if self.citation_behavior != expected[self.answerability]:
            raise ValueError("inconsistent citation behavior")
        positives = {j.passage_key for j in self.judgments if j.grade > 0}
        if self.answerability == "insufficient_evidence":
            if positives or self.gold_spans:
                raise ValueError("unanswerable evidence is not a negative retrieval pool")
        elif not positives or not self.gold_spans:
            raise ValueError("answerable or corrective item needs proposed evidence")
        if any(s.passage_key not in positives for s in self.gold_spans):
            raise ValueError("gold span must have a positive judgment")
        return self


class Specification(Contract):
    revision: Literal["dynamisbench-v0"] = "dynamisbench-v0"
    status: Literal["CANDIDATE", "AWAITING_EXPERT_REVIEW"]
    corpus: Corpus
    questions: tuple[Question, ...] = Field(min_length=1)
    category_gaps: tuple[Text, ...]
    judgment_policy: Literal["pooled-provisional-0-to-3-v1"] = "pooled-provisional-0-to-3-v1"

    @property
    def query_set_sha256(self) -> str:
        return digest_json([{"id": q.id, "query": q.query} for q in self.questions])

    @model_validator(mode="after")
    def references(self) -> Self:
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("duplicate question id")
        passages = {p.key: p for p in self.corpus.passages}
        sources = {s.pmcid for s in self.corpus.sources}
        for q in self.questions:
            if not set(q.articles) <= sources:
                raise ValueError("unknown question source")
            for j in q.judgments:
                if j.passage_key not in passages or passages[j.passage_key].pmcid not in q.articles:
                    raise ValueError("dangling or incorrectly attributed judgment")
            for s in q.gold_spans:
                p = passages[s.passage_key]
                if not s.start < s.end <= p.characters or s.source_anchor not in p.anchors:
                    raise ValueError("invalid source span")
        return self


class Hit(Contract):
    passage_key: Digest
    rank: int = Field(ge=1)
    score: float
    provenance_sha256: Digest


class Ranking(Contract):
    lane: Lane
    hits: tuple[Hit, ...]

    @model_validator(mode="after")
    def order(self) -> Self:
        if [h.rank for h in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("ranking must be contiguous and ordered")
        if len({h.passage_key for h in self.hits}) != len(self.hits):
            raise ValueError("duplicate retrieved passage")
        if any(a.score < b.score for a, b in zip(self.hits, self.hits[1:], strict=False)):
            raise ValueError("scores must be descending")
        return self


class QueryResult(Contract):
    question_id: Text
    query_sha256: Digest
    response_sha256: Digest
    rankings: tuple[Ranking, ...] = Field(min_length=3, max_length=3)
    total_ms: float = Field(ge=0)
    timings: ScientificTimings

    @model_validator(mode="after")
    def lanes(self) -> Self:
        if {r.lane for r in self.rankings} != {"bm25", "dense", "fused"}:
            raise ValueError("missing retrieval lane")
        return self


class Run(Contract):
    revision: Literal["dynamisbench-run-v0"] = "dynamisbench-run-v0"
    spec_sha256: Digest
    corpus_sha256: Digest
    query_set_sha256: Digest
    git_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    executed_at: AwareDatetime
    backend: Literal["qdrant-1.20.0"] = "qdrant-1.20.0"
    execution_config_sha256: Digest
    results: tuple[QueryResult, ...]


class QualificationReceipt(Contract):
    """Operator verification receipt; this format cannot confer scientific approval."""

    revision: Literal["dynamisbench-qualification-receipt-v0"]
    implementation_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    spec_sha256: Digest
    query_set_sha256: Digest
    corpus_sha256: Digest
    run_sha256: Digest
    metrics_sha256: Digest
    original_artifact_sha256: dict[str, Digest]
    sources: tuple[Source, ...]
    jats_reparse_matches_postgresql: Literal[True]
    canonical_passages_verified: Literal[75]
    original_qdrant_vectors_verified_after_capture: Literal[75]
    alias: Text
    retrieval_execution_status: Literal["CAPTURED_AND_RAW_REPLAY_VERIFIED"]
    live_answer_execution_status: Literal["BLOCKED_GENERATION_UNAVAILABLE"]
    scientific_status: Literal["AWAITING_EXPERT_REVIEW"]
    expert_signatures: tuple[()] = ()
    performance_gate_enabled: Literal[False]
    refusal_reasons: tuple[Text, ...] = Field(min_length=1)
    private_raw_directory: Text
    candidate_system_measurements: None


def validate_run(spec: Specification, run: Run) -> None:  # noqa: PLR0912 -- independent trust-boundary checks
    """Revalidate objects too: Pydantic model_copy is deliberately not a trust boundary."""
    spec = Specification.model_validate_json(spec.canonical_bytes())
    run = Run.model_validate_json(run.canonical_bytes())
    if run.execution_config_sha256 != digest_json(EXECUTION):
        raise ValueError("execution configuration mismatch")
    if run.query_set_sha256 != spec.query_set_sha256:
        raise ValueError("query set mismatch")
    if (run.spec_sha256, run.corpus_sha256) != (spec.digest, spec.corpus.digest):
        raise ValueError("run/spec identity mismatch")
    if [r.question_id for r in run.results] != [q.id for q in spec.questions]:
        raise ValueError("changed or incomplete query inventory")
    passages = {p.key: p for p in spec.corpus.passages}
    for q, result in zip(spec.questions, run.results, strict=True):
        if result.total_ms != result.timings.total_request_ms or any(
            not math.isfinite(value) or value < 0 for value in result.timings.model_dump().values()
        ):
            raise ValueError("invalid execution timings")
        if result.query_sha256 != hashlib.sha256(q.query.encode()).hexdigest():
            raise ValueError("changed query")
        for ranking in result.rankings:
            for h in ranking.hits:
                if h.passage_key not in passages:
                    raise ValueError("unknown retrieved passage")
                if h.provenance_sha256 != passages[h.passage_key].provenance_sha256:
                    raise ValueError("retrieved provenance mismatch")
        lanes = {r.lane: r for r in result.rankings}
        if len(lanes["dense"].hits) != 50:
            raise ValueError("incomplete dense result inventory for 75-passage unfiltered corpus")
        scores: dict[str, float] = {}
        for lane in ("bm25", "dense"):
            if len(lanes[lane].hits) > 50:
                raise ValueError("candidate window exceeds 50")
            for hit in lanes[lane].hits:
                scores[hit.passage_key] = scores.get(hit.passage_key, 0.0) + 1 / (60 + hit.rank)
        expected = sorted(scores, key=lambda k: (-scores[k], k))[:20]
        if [h.passage_key for h in lanes["fused"].hits] != expected or any(
            not math.isclose(h.score, scores[h.passage_key], rel_tol=1e-12, abs_tol=1e-15)
            for h in lanes["fused"].hits
        ):
            raise ValueError("fused ranking does not reconstruct from lanes")
