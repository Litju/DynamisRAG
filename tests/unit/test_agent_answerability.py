"""Calibration and held-out regression tests for the RES-638 evidence gate.

These tests exercise the gate directly. They do not claim semantic entailment:
every pass is the module's documented lexical/structural label. The licensed
corpus run is a separate integration receipt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from dynamisrag.agent.answerability import (
    CALIBRATED_MIN_COMPARISON_COVERAGE,
    CALIBRATED_MIN_RELEVANCE,
    ClaimEvidence,
    EvidenceCandidate,
    assess_claims,
    assess_retrieval,
    build_query_profile,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "answerability"
_Verdict = Literal["ANSWERABLE", "ABSTAIN"]


class _CandidateFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_key: str
    document_version_key: str
    title: str
    text: str
    section_title: str | None = None
    canonical_verified: bool = True
    rank: int = 0


class _RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    expected_verdict: _Verdict
    expected_code: str | None = None
    authorized_document_count: int | None = None
    candidates: tuple[_CandidateFixture, ...]


class _ClaimFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_key: str
    document_version_key: str
    document_title: str
    claim_text: str
    verbatim_quote: str


class _ClaimCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    expected_verdict: _Verdict
    expected_code: str | None = None
    claims: tuple[_ClaimFixture, ...]


class _FixtureFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    fixture_schema: str = Field(alias="schema")
    note: str
    retrieval_cases: tuple[_RetrievalCase, ...]
    claim_cases: tuple[_ClaimCase, ...]


def _load(name: str) -> _FixtureFile:
    return _FixtureFile.model_validate(json.loads((_FIXTURES / name).read_text(encoding="utf-8")))


def _candidates(case: _RetrievalCase) -> tuple[EvidenceCandidate, ...]:
    return tuple(
        EvidenceCandidate(
            passage_key=c.passage_key,
            document_version_key=c.document_version_key,
            title=c.title,
            text=c.text,
            section_title=c.section_title,
            rank=c.rank,
            canonical_verified=c.canonical_verified,
        )
        for c in case.candidates
    )


@pytest.mark.parametrize("filename", ["calibration.json", "heldout.json"])
def test_retrieval_cases_match_expected_decisions(filename: str) -> None:
    fixture = _load(filename)
    failures: list[str] = []
    for case in fixture.retrieval_cases:
        decision = assess_retrieval(
            case.question,
            _candidates(case),
            authorized_document_count=case.authorized_document_count,
        )
        if decision.verdict != case.expected_verdict:
            failures.append(f"{case.id}: expected {case.expected_verdict}, got {decision.code}")
        if case.expected_code is not None and decision.code != case.expected_code:
            failures.append(f"{case.id}: expected code {case.expected_code}, got {decision.code}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("filename", ["calibration.json", "heldout.json"])
def test_claim_cases_match_expected_decisions(filename: str) -> None:
    fixture = _load(filename)
    failures: list[str] = []
    for case in fixture.claim_cases:
        decision = assess_claims(
            case.question,
            tuple(
                ClaimEvidence(
                    passage_key=c.passage_key,
                    document_version_key=c.document_version_key,
                    document_title=c.document_title,
                    claim_text=c.claim_text,
                    verbatim_quote=c.verbatim_quote,
                )
                for c in case.claims
            ),
        )
        if decision.verdict != case.expected_verdict:
            failures.append(f"{case.id}: expected {case.expected_verdict}, got {decision.code}")
        if case.expected_code is not None and decision.code != case.expected_code:
            failures.append(f"{case.id}: expected code {case.expected_code}, got {decision.code}")
    assert not failures, "\n".join(failures)


def _case_coverage(case: _RetrievalCase) -> tuple[float, bool]:
    """Best candidate coverage and whether every candidate passed the entity gate."""
    decision = assess_retrieval(
        case.question,
        _candidates(case),
        authorized_document_count=case.authorized_document_count,
    )
    candidates = [c for c in decision.assessments if c.candidate.canonical_verified]
    if not candidates:
        return 0.0, False
    best = max(c.lexical_similarity for c in candidates)
    entities_ok = all(c.entity_support >= 1.0 for c in candidates)
    return best, entities_ok


def test_calibration_threshold_separates_positive_and_negative_cases() -> None:
    fixture = _load("calibration.json")
    positives: list[float] = []
    negatives: list[float] = []
    for case in fixture.retrieval_cases:
        profile = build_query_profile(case.question)
        if profile.kind == "comparison" or profile.entities:
            continue
        coverage, entities_ok = _case_coverage(case)
        if not entities_ok:
            continue
        if case.expected_verdict == "ANSWERABLE":
            positives.append(coverage)
        elif case.expected_code == "insufficient_query_coverage":
            negatives.append(coverage)
    assert positives and negatives
    # The production threshold was selected inside the calibrated interval.
    assert max(negatives) < CALIBRATED_MIN_RELEVANCE <= min(positives), (
        f"threshold {CALIBRATED_MIN_RELEVANCE} outside ({max(negatives)}, {min(positives)}]"
    )


def test_comparison_threshold_is_below_factual_threshold_and_calibrated() -> None:
    assert 0.0 < CALIBRATED_MIN_COMPARISON_COVERAGE < CALIBRATED_MIN_RELEVANCE
    fixture = _load("calibration.json")
    comparison_positives = [
        case
        for case in fixture.retrieval_cases
        if build_query_profile(case.question).kind == "comparison"
    ]
    assert comparison_positives
    for case in comparison_positives:
        decision = assess_retrieval(
            case.question,
            _candidates(case),
            authorized_document_count=case.authorized_document_count,
        )
        if case.expected_verdict == "ANSWERABLE":
            assert decision.verdict == "ANSWERABLE"


def test_claim_threshold_is_used_for_verified_quotations() -> None:
    fixture = _load("calibration.json")
    coverages: list[float] = []
    for case in fixture.claim_cases:
        if case.expected_verdict != "ANSWERABLE":
            continue
        if build_query_profile(case.question).kind == "comparison":
            continue
        decision = assess_claims(
            case.question,
            tuple(
                ClaimEvidence(
                    passage_key=c.passage_key,
                    document_version_key=c.document_version_key,
                    document_title=c.document_title,
                    claim_text=c.claim_text,
                    verbatim_quote=c.verbatim_quote,
                )
                for c in case.claims
            ),
        )
        assert decision.answerable
        coverages.extend(c.lexical_similarity for c in decision.assessments if c.admissible)
    assert coverages
    assert min(coverages) >= CALIBRATED_MIN_RELEVANCE


def test_gate_separates_lexical_similarity_from_query_relevance() -> None:
    decision = assess_retrieval(
        "What is the orbital period of the exoplanet Kepler-22b?",
        (
            EvidenceCandidate(
                passage_key="a" * 64,
                document_version_key="b" * 64,
                title="Rest Intervals during Plyometric Training",
                text="Participants rested for a period of 24 h before the testing session.",
            ),
        ),
    )
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "unsupported_entity"
    candidate = decision.assessments[0]
    assert candidate.lexical_similarity > 0.0
    assert candidate.query_relevance < candidate.lexical_similarity
    assert candidate.entity_support == 0.0
    assert candidate.admissible is False


def test_claim_gate_excludes_an_irrelevant_quotation_within_the_same_study() -> None:
    decision = assess_claims(
        "How many weeks did those participants complete?",
        (
            ClaimEvidence(
                passage_key="a" * 64,
                document_version_key="b" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The training group completed 12 weeks of training.",
                verbatim_quote=(
                    "The training group completed plyometric training for 12 weeks (3 days/week)."
                ),
            ),
            ClaimEvidence(
                passage_key="a" * 64,
                document_version_key="b" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The subjects lay supine.",
                verbatim_quote=(
                    "The subjects lay in the supine position on the sliding bed of the "
                    "sledge apparatus."
                ),
            ),
        ),
    )
    assert decision.answerable
    assert decision.accepted_claim_indexes == (0,)
    assert decision.accepted_passage_keys == ("a" * 64,)


def test_comparison_does_not_accept_an_irrelevant_document_for_two_doc_count() -> None:
    fixture = _load("calibration.json")
    case = next(
        item for item in fixture.retrieval_cases if item.id == "cal-pos-comparison-mixed-relevance"
    )
    decision = assess_retrieval(case.question, _candidates(case))
    assert decision.answerable
    assert len(decision.accepted_passage_keys) == 1
    assert "1000000000000000000000000000000000000000000000000000000000000013" not in (
        decision.accepted_passage_keys
    )


def test_bare_followup_uses_prior_context_only_when_the_current_turn_is_empty() -> None:
    candidate = EvidenceCandidate(
        passage_key="a" * 64,
        document_version_key="b" * 64,
        title="Muscle-Tendon Behavior during Depth Jumping",
        text="Achilles tendon stiffness increased during depth jumping.",
    )
    without_context = assess_retrieval("What about it?", (candidate,))
    assert without_context.verdict == "ABSTAIN"
    assert without_context.code == "ambiguous_question"
    with_context = assess_retrieval(
        "What about it?",
        (candidate,),
        context="What happened to Achilles tendon stiffness during depth jumping?",
    )
    assert with_context.answerable


def test_ambiguous_question_without_checkable_content_abstains() -> None:
    decision = assess_retrieval(
        "What about it?",
        (
            EvidenceCandidate(
                passage_key="a" * 64,
                document_version_key="b" * 64,
                title="Plyometric Training",
                text="Jump height increased.",
            ),
        ),
    )
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "ambiguous_question"


def test_resolved_multi_turn_reference_can_be_answerable() -> None:
    resolved = (
        "How many weeks of plyometric training did those depth jumping participants complete?"
    )
    decision = assess_retrieval(
        resolved,
        (
            EvidenceCandidate(
                passage_key="a" * 64,
                document_version_key="b" * 64,
                title="Depth Jumping Study",
                text="Participants completed 12 weeks of plyometric training.",
            ),
        ),
    )
    assert decision.answerable
    assert build_query_profile(resolved).kind == "factual"


def test_retrieval_never_upgrades_noncanonical_evidence() -> None:
    decision = assess_retrieval(
        "Does plyometric training improve jump height?",
        (
            EvidenceCandidate(
                passage_key="a" * 64,
                document_version_key="b" * 64,
                title="Plyometric Training and Jump Performance",
                text="Jump height increased after plyometric training.",
                canonical_verified=False,
            ),
        ),
    )
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "source_not_canonical"


def test_payload_is_bounded_and_json_safe() -> None:
    case = _load("calibration.json").retrieval_cases[0]
    decision = assess_retrieval(case.question, _candidates(case))
    payload = decision.to_payload()
    assert payload["gate_version"] == "structured-evidence-gate-v1"
    assert payload["verdict"] == "ANSWERABLE"
    assert payload["claim_support"] == "NOT_ASSESSED"
    encoded = json.dumps(payload, sort_keys=True)
    assert len(encoded) < 20_000
    assert "text" not in encoded


def test_fixture_sets_are_disjoint() -> None:
    calibration = _load("calibration.json")
    heldout = _load("heldout.json")
    calibration_questions = {case.question for case in calibration.retrieval_cases}
    heldout_questions = {case.question for case in heldout.retrieval_cases}
    assert not calibration_questions & heldout_questions
