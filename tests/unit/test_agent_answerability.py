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
        item for item in fixture.retrieval_cases if item.id == "cal-neg-comparison-mixed-relevance"
    )
    decision = assess_retrieval(case.question, _candidates(case))
    assert not decision.answerable
    assert decision.code == "insufficient_cross_study_evidence"
    assert not decision.accepted_passage_keys


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


_DEPTH_JUMPING = EvidenceCandidate(
    passage_key="c" * 64,
    document_version_key="d" * 64,
    title="Plyometric Training Favors Optimizing Muscle-Tendon Behavior during Depth Jumping",
    section_title="Methods",
    text="Participants completed 12 weeks of depth jumping training.",
)


def test_metadata_route_rejects_an_unsupported_entity() -> None:
    decision = assess_retrieval("What is the DOI of the Kepler-22b paper?", (_DEPTH_JUMPING,))
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "unsupported_entity"


def test_metadata_route_accepts_a_supported_study_qualifier() -> None:
    decision = assess_retrieval("What is the DOI of the depth jumping study?", (_DEPTH_JUMPING,))
    assert decision.verdict == "ANSWERABLE"
    assert decision.code == "inventory_or_metadata_route"
    assert decision.accepted_passage_keys == (_DEPTH_JUMPING.passage_key,)


def test_inventory_route_rejects_an_absent_topical_qualifier() -> None:
    decision = assess_retrieval(
        "Which papers are available about exoplanet atmospheres?", (_DEPTH_JUMPING,)
    )
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "insufficient_query_coverage"


def test_inventory_route_accepts_a_plain_corpus_inventory() -> None:
    decision = assess_retrieval("Which papers are available?", (_DEPTH_JUMPING,))
    assert decision.verdict == "ANSWERABLE"
    assert decision.code == "inventory_or_metadata_route"


def test_claim_metadata_route_rejects_an_unsupported_entity_and_topical_qualifier() -> None:
    unsupported_entity = assess_claims(
        "What is the DOI of the Kepler-22b paper?",
        (
            ClaimEvidence(
                passage_key="e" * 64,
                document_version_key="f" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The study reported a DOI.",
                verbatim_quote="The study was registered with a DOI.",
            ),
        ),
    )
    assert unsupported_entity.verdict == "ABSTAIN"
    assert unsupported_entity.code == "unsupported_entity"
    unsupported_topic = assess_claims(
        "What are the section titles of the exoplanet transit study?",
        (
            ClaimEvidence(
                passage_key="e" * 64,
                document_version_key="f" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The methods section is described.",
                verbatim_quote="The methods section describes the protocol.",
            ),
        ),
    )
    assert unsupported_topic.verdict == "ABSTAIN"
    assert unsupported_topic.code == "insufficient_query_coverage"


def test_unscoped_generic_question_abstains_when_multiple_documents_match() -> None:
    candidate = EvidenceCandidate(
        passage_key="1" * 64,
        document_version_key="2" * 64,
        title="Plyometric Training during Depth Jumping",
        text="The sample size was 21 participants.",
    )
    other = EvidenceCandidate(
        passage_key="3" * 64,
        document_version_key="4" * 64,
        title="Complex Training in Cricketers",
        text="The sample size was 37 participants.",
    )
    decision = assess_retrieval("What is the sample size?", (candidate, other))
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "ambiguous_question"


def test_scoped_generic_question_answers_from_its_single_document() -> None:
    candidate = EvidenceCandidate(
        passage_key="1" * 64,
        document_version_key="2" * 64,
        title="Plyometric Training during Depth Jumping",
        text="The sample size was 21 participants.",
    )
    decision = assess_retrieval("What is the sample size?", (candidate,))
    assert decision.verdict == "ANSWERABLE"


def test_numeric_question_rejects_a_qualitative_quote_and_accepts_a_number() -> None:
    qualitative = assess_claims(
        "What was its sample size?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The study had a relatively small sample size.",
                verbatim_quote="This study was conducted with a relatively small sample size.",
            ),
        ),
    )
    assert qualitative.verdict == "ABSTAIN"
    assert qualitative.code == "insufficient_numeric_evidence"
    numeric = assess_claims(
        "What was its sample size?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The sample size was 21.",
                verbatim_quote="The sample size was 21 participants.",
            ),
        ),
    )
    assert numeric.verdict == "ANSWERABLE"
    assert numeric.accepted_claim_indexes == (0,)


def test_contracted_value_question_still_requires_a_number() -> None:
    qualitative = assess_claims(
        "What's its sample size?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title="Plyometric Training during Depth Jumping",
                claim_text="The study had a relatively small sample size.",
                verbatim_quote="This study was conducted with a relatively small sample size.",
            ),
        ),
    )
    assert qualitative.verdict == "ABSTAIN"
    assert qualitative.code == "insufficient_numeric_evidence"


def test_title_only_entity_and_facets_cannot_supply_a_finding() -> None:
    decision = assess_claims(
        "What did the Depth Jumping study report about muscle-tendon behavior?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title=(
                    "Plyometric Training Favors Optimizing Muscle-Tendon Behavior "
                    "during Depth Jumping"
                ),
                claim_text="Participants completed depth jumping training.",
                verbatim_quote=("Participants completed 12 weeks of depth jumping training."),
            ),
        ),
    )
    assert decision.verdict == "ABSTAIN"


def test_study_qualifier_is_validated_against_the_title_without_being_claim_text() -> None:
    depth = assess_claims(
        "What did the depth jumping study find about Achilles tendon stiffness?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title=(
                    "Plyometric Training Favors Optimizing Muscle-Tendon Behavior "
                    "during Depth Jumping"
                ),
                claim_text="Achilles tendon stiffness was quantified.",
                verbatim_quote=(
                    "The Achilles tendon stiffness was quantified using ultrasonography."
                ),
            ),
        ),
    )
    assert depth.verdict == "ANSWERABLE"
    wrong_study = assess_claims(
        "What did the female cricketers study find about upper limb power?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title="Complex Training in Male Cricketers",
                claim_text="Upper limb power improved.",
                verbatim_quote="Upper limb power improved in male cricketers.",
            ),
        ),
    )
    assert wrong_study.verdict == "ABSTAIN"
    assert wrong_study.code == "insufficient_query_coverage"


def test_claim_gate_does_not_score_the_document_title() -> None:
    decision = assess_claims(
        "What was the effect of knee flexion angle on muscle activity?",
        (
            ClaimEvidence(
                passage_key="1" * 64,
                document_version_key="2" * 64,
                document_title=(
                    "Impact of Starting Knee Flexion Angle on Muscle Activity and "
                    "Performance during Plyometrics without Jumping"
                ),
                claim_text="Participants refrained from training before testing.",
                verbatim_quote=(
                    "Additionally, participants were instructed to refrain from engaging "
                    "in physical training for a period of 24 h preceding the testing session."
                ),
            ),
        ),
    )
    assert decision.verdict == "ABSTAIN"


def test_absent_salient_qualifier_is_a_hard_abstention() -> None:
    candidate = EvidenceCandidate(
        passage_key="1" * 64,
        document_version_key="2" * 64,
        title="Complex Training in Male Cricketers",
        text="Upper limb power improved in male cricketers after four weeks of complex training.",
    )
    decision = assess_retrieval(
        "What did the female cricketers study find about upper limb power?", (candidate,)
    )
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "insufficient_query_coverage"
    assert "female" in decision.detail


def test_metadata_route_requires_every_topical_qualifier() -> None:
    candidate = EvidenceCandidate(
        passage_key="1" * 64,
        document_version_key="2" * 64,
        title="Achilles Tendon Stiffness during Depth Jumping",
        text="Achilles tendon stiffness increased during depth jumping.",
    )
    decision = assess_retrieval("What is the DOI of the patellar tendon study?", (candidate,))
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "insufficient_query_coverage"


def test_common_training_noise_words_do_not_hide_a_missing_qualifier() -> None:
    candidate = EvidenceCandidate(
        passage_key="1" * 64,
        document_version_key="2" * 64,
        title="Plyometric Training Favors Optimizing Muscle-Tendon Behavior",
        text="Achilles tendon stiffness increased during depth jumping.",
    )
    decision = assess_retrieval(
        "What is patellar tendon stiffness after depth jumping?", (candidate,)
    )
    assert decision.verdict == "ABSTAIN"
    assert decision.code == "insufficient_query_coverage"


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
