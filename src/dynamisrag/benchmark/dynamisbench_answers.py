"""Structural answer audits and externally authenticated scientific review labels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import AwareDatetime, Field

from dynamisrag.benchmark.dynamisbench import Contract, Digest, Specification, Status, Text
from dynamisrag.benchmark.dynamisbench_runner import capture, digest_json, provenance_digest
from dynamisrag.generation.scientific import ScientificAnswer

Label = Literal["pass", "fail", "not_applicable"]


class ExpertLabels(Contract):
    """Unsigned labels are proposals. Only an independently trusted verifier can accept them."""

    spec_sha256: Digest
    question_id: Text
    answer_sha256: Digest
    reviewer: Text
    reviewed_at: AwareDatetime
    protocol_reference: Text
    rationale: Text
    citation_support: Label
    scientific_entailment: Label
    numerical_fidelity: Label
    population_applicability: Label
    contradiction_handling: Label
    abstention_quality: Label
    premise_correction: Label


class SignedReview(Contract):
    labels: ExpertLabels
    signature: Text
    key_id: Text


class AnswerEvaluation(Contract):
    question_id: str
    answer_sha256: Digest
    status: Literal["supported", "insufficient_evidence"]
    structural_citations_valid: bool
    proposed_relevant_citations: int = Field(ge=0)
    abstained: bool
    expected_abstention: bool
    review_status: Status
    scientific_labels: ExpertLabels | None
    scientific_pass: bool | None
    performance_gate_enabled: Literal[False] = False


def evaluate_answer(
    spec: Specification,
    question_id: str,
    answer: ScientificAnswer,
    *,
    review: SignedReview | None = None,
    verify_signature: Callable[[str, bytes, str], bool] | None = None,
) -> AnswerEvaluation:
    spec = Specification.model_validate_json(spec.canonical_bytes())
    answer = ScientificAnswer.model_validate_json(answer.model_dump_json())
    q = next(q for q in spec.questions if q.id == question_id)
    # Validate all retrieval lanes and their evidence against the complete frozen inventory.
    one = spec.model_copy(update={"questions": (q,)})
    capture(
        one, lambda _q, _n, _f: answer.retrieval, git_commit="0" * 40, verified_corpus=spec.corpus
    )
    digest = digest_json(answer.model_dump(mode="json"))
    passages = {p.key: p for p in spec.corpus.passages}
    evidence = {p.passage_key: p for p in answer.evidence}
    contexts = {p.passage_key: p for p in answer.context_excerpts}
    valid = len(evidence) == len(answer.evidence) and len(contexts) == len(answer.context_excerpts)
    retrieved = {h.passage_key for h in answer.retrieval.fusion.hits}
    valid &= set(evidence) <= retrieved and set(contexts) == set(evidence)
    for p in answer.evidence:
        valid &= (
            p.passage_key in passages
            and provenance_digest(p) == passages[p.passage_key].provenance_sha256
        )
    for claim in answer.claims:
        p = claim.source
        context = contexts.get(p.passage_key)
        valid &= (
            p.passage_key in evidence
            and p == evidence[p.passage_key]
            and 0 <= claim.excerpt_start < claim.excerpt_end <= len(p.text)
            and p.text[claim.excerpt_start : claim.excerpt_end]
            == claim.text
            == claim.supporting_excerpt
            and context is not None
            and claim.text in context.text
            and context.text == p.text[context.start : context.end]
        )
    valid &= (answer.status == "supported") == bool(answer.claims)
    valid &= answer.answer == (
        "\n\n".join(f"{c.text} [{i}]" for i, c in enumerate(answer.claims, 1))
        if answer.claims
        else None
    )
    labels = None
    scientific_pass = None
    if review is not None:
        review = SignedReview.model_validate_json(review.canonical_bytes())
        labels = review.labels
        if (
            (labels.spec_sha256, labels.question_id, labels.answer_sha256)
            != (
                spec.digest,
                q.id,
                digest,
            )
            or verify_signature is None
            or not verify_signature(
                review.key_id,
                labels.canonical_bytes(),
                review.signature,
            )
        ):
            raise ValueError("unauthenticated or incorrectly bound expert review")
        applicable = labels.model_dump(
            include={
                "citation_support",
                "scientific_entailment",
                "numerical_fidelity",
                "population_applicability",
                "contradiction_handling",
                "abstention_quality",
                "premise_correction",
            }
        )
        # The expected answer behavior comes from the question's scientific
        # protocol, not from reviewer labels alone. A false premise needs a
        # supported correction; an abstention is not a successful correction.
        expected_status = (
            "insufficient_evidence"
            if q.answerability == "insufficient_evidence"
            else "supported"
        )
        required: set[str] = set()
        if answer.status == "supported":
            required |= {"citation_support", "scientific_entailment", "population_applicability"}
        else:
            required.add("abstention_quality")
        if q.category == "numerical" and answer.status == "supported":
            required.add("numerical_fidelity")
        if q.category == "contradictory" and answer.status == "supported":
            required.add("contradiction_handling")
        if q.answerability == "false_premise" and answer.status == "supported":
            required.add("premise_correction")
        abstention_label_consistent = (
            applicable["abstention_quality"]
            == ("pass" if answer.status == "insufficient_evidence" else "not_applicable")
        )
        scientific_pass = (
            valid
            and answer.status == expected_status
            and abstention_label_consistent
            and "fail" not in applicable.values()
            and all(applicable[k] == "pass" for k in required)
        )
    relevant = {j.passage_key for j in q.judgments if j.grade > 0}
    return AnswerEvaluation(
        question_id=q.id,
        answer_sha256=digest,
        status=answer.status,
        structural_citations_valid=bool(valid),
        proposed_relevant_citations=sum(c.source.passage_key in relevant for c in answer.claims),
        abstained=answer.status == "insufficient_evidence",
        expected_abstention=q.answerability == "insufficient_evidence",
        review_status=Status.EXPERT_APPROVED if labels else Status.AWAITING_EXPERT_REVIEW,
        scientific_labels=labels,
        scientific_pass=scientific_pass,
    )


class AnswerInventory(Contract):
    spec_sha256: Digest
    results: tuple[AnswerEvaluation, ...]
    performance_gate_enabled: Literal[False] = False


def evaluate_answers(
    spec: Specification,
    answers: dict[str, ScientificAnswer],
) -> AnswerInventory:
    if list(answers) != [q.id for q in spec.questions]:
        raise ValueError("missing, duplicate or changed answer inventory")
    return AnswerInventory(
        spec_sha256=spec.digest,
        results=tuple(evaluate_answer(spec, q.id, answers[q.id]) for q in spec.questions),
    )


class AnswerDifference(Contract):
    question_id: str
    baseline_abstained: bool
    candidate_abstained: bool
    expected_abstention: bool
    unsupported_acceptance: bool
    unnecessary_refusal: bool
    scientific_review: Literal["AWAITING_EXPERT_REVIEW"] = "AWAITING_EXPERT_REVIEW"


def compare_answers(
    spec: Specification,
    baseline: dict[str, ScientificAnswer],
    candidate: dict[str, ScientificAnswer],
) -> tuple[AnswerDifference, ...]:
    left, right = evaluate_answers(spec, baseline), evaluate_answers(spec, candidate)
    return tuple(
        AnswerDifference(
            question_id=a.question_id,
            baseline_abstained=a.abstained,
            candidate_abstained=b.abstained,
            expected_abstention=b.expected_abstention,
            unsupported_acceptance=b.expected_abstention and not b.abstained,
            unnecessary_refusal=q.answerability in {"answerable", "false_premise"}
            and b.abstained,
        )
        for q, a, b in zip(spec.questions, left.results, right.results, strict=True)
    )
