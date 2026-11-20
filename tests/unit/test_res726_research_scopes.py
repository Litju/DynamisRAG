"""RES-726: browsing, hard source boundaries, and soft comparison referents are distinct."""

from typing import cast

from dynamisrag.agent.answerability import (
    CONVERSATIONAL_REFERENCE_REASON,
    ClaimEvidence,
    EvidenceCandidate,
    assess_claims,
    assess_retrieval,
    build_query_profile,
    salient_terms,
)
from dynamisrag.agent.graph import (
    ConversationalResearchOrchestrator,
    LiteratureAcquisitionPort,
    ResearchAgentState,
    _comparison_anchor_present,  # pyright: ignore[reportPrivateUsage]
    _reject_unanchored_comparison_claims,  # pyright: ignore[reportPrivateUsage]
    _thread_excluded_versions,  # pyright: ignore[reportPrivateUsage]
    _thread_research_scopes,  # pyright: ignore[reportPrivateUsage]
)
from dynamisrag.agent.persistence import InMemoryAgentConversationStore

A, B = "a" * 64, "b" * 64


def test_automatic_comparison_followup_keeps_reference_without_blocking_discovery() -> None:
    for question in (
        "How does its measurement compare with force plates?",
        "How does it compare with a criterion system?",
        "Compare those studies with a new force plate validation.",
    ):
        hard_query_scope, references, all_required = _thread_research_scopes(question, (), (A,))
        assert hard_query_scope == []  # new papers are still discoverable
        assert references == [A]  # original study cannot silently change
        # A singular reference ("its", "that study") requires its one anchor;
        # "Compare those studies" below names the set and requires all of them.
        assert all_required == (question.startswith("Compare those studies"))


def test_restricted_comparison_only_uses_the_explicit_user_allowlist() -> None:
    scope, references, _ = _thread_research_scopes(
        "How does its reliability compare with the other selected study?", (A, B), (A,)
    )
    assert scope == [A, B]
    assert references == [A]


def test_ordinary_pronoun_followups_remain_anchored_to_prior_study() -> None:
    assert _thread_research_scopes("What was its sample size?", (), (A,)) == ([A], [], False)
    assert _thread_research_scopes("What were its results?", (A, B), (A,)) == ([A], [], False)
    assert _thread_research_scopes("Compare the methods of these papers.", (), ()) == (
        [],
        [],
        True,
    )


def test_plural_references_require_every_referenced_study() -> None:
    # "those studies" names the SET: the answer may not silently drop one side.
    _, anchors, all_required = _thread_research_scopes(
        "How do those studies compare with force plates?", (), (A, B)
    )
    assert anchors == [A, B]
    assert all_required
    seen_a_only = [{"document_version_key": A}, {"document_version_key": "c" * 64}]
    assert not _comparison_anchor_present((A, B), seen_a_only, all_required=True)
    assert _comparison_anchor_present(
        (A, B), [*seen_a_only, {"document_version_key": B}], all_required=True
    )
    # A singular reference keeps its any-one-of semantics.
    assert _comparison_anchor_present((A,), seen_a_only)


def test_comparison_evidence_must_include_the_referenced_study() -> None:
    original = {"document_version_key": A}
    unrelated = {"document_version_key": B}
    assert not _comparison_anchor_present((A,), [unrelated])
    assert _comparison_anchor_present((A,), [original, unrelated])
    assert _comparison_anchor_present((), [unrelated])


def test_discovery_router_uses_hard_allowlist_not_automatic_comparison_reference() -> None:
    orchestrator = object.__new__(ConversationalResearchOrchestrator)
    # Routing only tests that an acquisition port is configured; none may run here.
    orchestrator._acquisition = cast("LiteratureAcquisitionPort", object())  # pyright: ignore[reportPrivateUsage]
    state: ResearchAgentState = {
        "literature_request": {"searches": []},
        "acquisition_attempted": False,
        "scoped_version_keys": [],
        "allowed_version_keys": [],
        "comparison_anchor_version_keys": [A],
    }
    assert orchestrator._route_after_tools(state) == "acquire_external_literature"  # pyright: ignore[reportPrivateUsage]

    # A deliberately restricted conversation must not browse the broader corpus.
    state["allowed_version_keys"] = [A]
    assert orchestrator._route_after_tools(state) == "finalize_durable_response"  # pyright: ignore[reportPrivateUsage]


def test_final_scientific_claims_cannot_drop_the_original_comparison_study() -> None:
    verified = [{"document_version_key": B, "verified": True, "claim_text": "new finding"}]
    rejected: list[dict[str, object]] = []
    _reject_unanchored_comparison_claims(verified, rejected, (A,))
    assert verified == []
    assert rejected[0]["rejection_reason"] == "answerability_gate:referenced_study_not_cited"
    assert rejected[0]["verified"] is False

    anchored = [
        {"document_version_key": A, "verified": True},
        {"document_version_key": B, "verified": True},
    ]
    rejected = []
    _reject_unanchored_comparison_claims(anchored, rejected, (A,))
    assert len(anchored) == 2 and rejected == []


# ---------------------------------------------------------------------------
# RES-726: the relevance gate and the reference checks must not fight each other
# ---------------------------------------------------------------------------

FOLLOW_UP = "How do those measurements compare with force plates and other reference systems?"


def _candidate(key: str, doc: str, title: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        passage_key=key * 64, document_version_key=doc * 64, title=title, text=text
    )


ANCHOR = _candidate(
    "a",
    "a",
    "Reliability of a linear position transducer for measuring bar velocity",
    "Between-session reliability of mean bar velocity measured with the linear position "
    "transducer was excellent (ICC = 0.95) across loads in resistance training.",
)
ANCHOR_APPLICATION = _candidate(
    "c",
    "a",
    "Reliability of a linear position transducer for measuring bar velocity",
    "Linear position transducer velocity measurements allow coaches to monitor neuromuscular "
    "fatigue and adjust daily training loads in resistance training programmes.",
)
FORCE_PLATE = _candidate(
    "b",
    "b",
    "Force plate reference systems for strength and power assessment",
    "Force plates remain the reference standard system for direct ground reaction force "
    "measurement during the countermovement jump in strength and power assessment.",
)


def test_framing_determiners_never_trigger_the_salient_absent_everywhere_abstention() -> None:
    # "other", "additional" and friends frame how much evidence is wanted; a
    # passage never needs to contain them. Previously "other reference systems"
    # hard-abstained every candidate set, blocking legitimate comparisons.
    for word in ("other", "another", "additional", "further", "extra", "similar", "any"):
        profile = build_query_profile(f"How do the measurements compare with {word} systems?")
        assert (
            word not in profile.facets
            or dict(zip(profile.facets, profile.facet_weights, strict=True))[word] < 1.0
        )
        assert word not in salient_terms(profile)
    primary = build_query_profile("Find additional primary research on velocity training.")
    assert "addition" not in salient_terms(primary)
    decision = assess_retrieval(
        "Find additional primary research on velocity training.",
        (
            _candidate(
                "d",
                "d",
                "Primary research on velocity-based training",
                "This primary research study measured velocity-based training outcomes directly.",
            ),
        ),
    )
    assert decision.answerable, decision.to_payload()["detail"]


def test_comparison_followup_gate_admits_the_referenced_study_and_new_evidence() -> None:
    decision = assess_retrieval(
        FOLLOW_UP,
        (ANCHOR, ANCHOR_APPLICATION, FORCE_PLATE),
        referenced_document_keys=[A],
    )
    assert decision.answerable, decision.to_payload()["detail"]
    accepted = set(decision.accepted_passage_keys)
    assert {ANCHOR.passage_key, FORCE_PLATE.passage_key} <= accepted
    # The anchor is admitted by reference and the audit says so; accepted and
    # rejected passage sets stay disjoint.
    anchor_assessment = next(
        a for a in decision.assessments if a.candidate.passage_key == ANCHOR.passage_key
    )
    assert anchor_assessment.admissible
    assert CONVERSATIONAL_REFERENCE_REASON in anchor_assessment.reasons
    assert not (accepted & set(decision.rejected_passage_keys))


def test_a_lone_referenced_study_cannot_answer_a_comparison_by_itself() -> None:
    decision = assess_retrieval(FOLLOW_UP, (ANCHOR,), referenced_document_keys=[A])
    assert not decision.answerable
    # Both abstention codes route the run to literature discovery at the graph
    # layer (`evidence_gate:` prefix), never to a one-sided answer.
    assert decision.code in {"insufficient_query_coverage", "insufficient_cross_study_evidence"}


def test_several_referenced_studies_can_answer_each_other_by_reference() -> None:
    second = _candidate(
        "e",
        "b",
        "Criterion validity of a linear position transducer",
        "Compared with the criterion force plate, the linear position transducer agreed.",
    )
    decision = assess_retrieval(
        "How do those studies compare with each other?",
        (ANCHOR, second),
        referenced_document_keys=[A, B],
    )
    assert decision.answerable, decision.to_payload()["detail"]
    assert {ANCHOR.passage_key, second.passage_key} <= set(decision.accepted_passage_keys)


def test_referenced_study_claims_join_the_final_citation_gate() -> None:
    anchor_claim = ClaimEvidence(
        passage_key="a" * 64,
        document_version_key=A,
        document_title="Reliability of a linear position transducer for measuring bar velocity",
        claim_text="The referenced study reported excellent reliability of the transducer.",
        verbatim_quote="Between-session reliability of mean bar velocity was excellent.",
    )
    other_claim = ClaimEvidence(
        passage_key="b" * 64,
        document_version_key=B,
        document_title="Force plate reference systems for strength and power assessment",
        claim_text="Force plates are the reference standard for ground reaction force.",
        verbatim_quote="Force plates remain the reference standard system for direct ground "
        "reaction force measurement.",
    )
    decision = assess_claims(FOLLOW_UP, (anchor_claim, other_claim), referenced_document_keys=[A])
    assert decision.answerable, decision.to_payload()["detail"]
    assert decision.accepted_claim_indexes == (0, 1)
    lone = assess_claims(FOLLOW_UP, (anchor_claim,), referenced_document_keys=[A])
    assert not lone.answerable
    # Either fail-closed reason routes the run to discovery, never to a
    # one-sided comparison answer.
    assert lone.code in {"insufficient_query_coverage", "insufficient_cross_study_evidence"}


# ---------------------------------------------------------------------------
# RES-726: conversational scope corrections ("Exclude that paper ...")
# ---------------------------------------------------------------------------


def test_exclusion_directive_requires_both_a_directive_verb_and_a_study_reference() -> None:
    exclude = _thread_excluded_versions
    assert exclude("Exclude that paper from the next analysis.", (A,)) == [A]
    assert exclude("Please remove those papers from the analysis.", (A, B)) == [A, B]
    # An ordinary methods question is not a scope correction, no matter that it
    # contains "exclude" and a study reference (RES-726 review).
    assert exclude("Did that study exclude injured participants?", (A,)) == []
    # A negated directive must not exclude anything.
    for question in (
        "Do not exclude that paper from the next analysis.",
        "Don\u2019t exclude that paper.",
        "Do not ever exclude that paper.",
        "Did the review exclude that study?",
        "Why exclude that paper?",
        "Exclude that study\u2019s control group.",
    ):
        assert exclude(question, (A,)) == []
    assert exclude("Could you please drop that paper?", (A,)) == [A]
    # A bare negation is a question, not a scope correction.
    assert exclude("What did the study NOT report about force plates?", (A,)) == []
    # A directive without a study target does not exclude anything.
    assert exclude("Remove outliers from the dataset first.", (A,)) == []
    # A singular directive with several active studies is ambiguous: guessing
    # would silently drop the wrong study's evidence, so nothing is excluded.
    assert exclude("Exclude that paper from the next analysis.", (A, B)) == []
    # Without a conversational referent there is nothing to exclude.
    assert exclude("Exclude that paper from the next analysis.", ()) == []


def test_the_exclusion_list_never_silently_discards_a_correction() -> None:
    # The cap matches the maximum restricted allowlist (25): a correction
    # beyond the twelfth study must still be honoured (RES-726 review).
    store = InMemoryAgentConversationStore()
    thread = store.create_thread(owner_id="cap-owner", title="Exclusion cap")
    keys = tuple(f"{index:064x}" for index in range(1, 27))
    store.update_thread_context(
        thread.thread_id,
        user_id="cap-owner",
        active_version_keys=(),
        active_passage_keys=(),
        conversation_summary="",
        excluded_version_keys=keys[:25],
    )
    persisted = store.get_thread(thread.thread_id, user_id="cap-owner")
    assert len(persisted.excluded_version_keys) == 25


# ---------------------------------------------------------------------------
# RES-726 review regressions: admission must not vouch, bypass or transfer
# ---------------------------------------------------------------------------


def test_two_references_cannot_vouch_for_a_missing_named_comparator() -> None:
    second = _candidate(
        "e",
        "b",
        "Criterion validity of a linear position transducer",
        "Compared with the criterion force plate, the linear position transducer agreed.",
    )
    question = "Compare those studies with the XZ-77 reference system."
    decision = assess_retrieval(question, (ANCHOR, second), referenced_document_keys=[A, B])
    assert not decision.answerable
    assert decision.code == "unsupported_entity"
    anchor_claim = ClaimEvidence(
        passage_key="a" * 64,
        document_version_key=A,
        document_title="Reliability of a linear position transducer",
        claim_text="The referenced study reported excellent reliability.",
        verbatim_quote="Between-session reliability of mean bar velocity was excellent.",
    )
    claims = assess_claims(question, (anchor_claim,), referenced_document_keys=[A, B])
    assert not claims.answerable
    # Either fail-closed reason routes the run to discovery; the named
    # comparator is never vouched for by the referenced studies.
    assert claims.code in {"unsupported_entity", "insufficient_query_coverage"}


def test_a_single_study_cannot_answer_a_comparison_alone_at_the_retrieval_gate() -> None:
    # The anchor's own passages cover "force plates", but a comparison answer
    # still needs two sides: discovery must run BEFORE final verification
    # abstains on one document (RES-726 review).
    anchor_forceplate = _candidate(
        "f",
        "a",
        "The Validity and Reliability of the Force Plates and the Linear Position Transducer",
        "The Force Plates and the Linear Position Transducer demonstrated good reliability "
        "for daily measurements of countermovement depth and velocity (ICCs > 0.70).",
    )
    decision = assess_retrieval(
        "Compare that study with force plates.", (anchor_forceplate,), referenced_document_keys=[A]
    )
    assert not decision.answerable
    assert decision.code == "insufficient_cross_study_evidence"


def test_inventory_wording_cannot_bypass_the_comparison_contract() -> None:
    # "compare ... with the available studies" is a comparison; routing it to
    # the inventory/metadata branch bypassed the two-sided requirement
    # (RES-726 review).
    assert (
        build_query_profile("How does its jump height compare with the available studies?").kind
        == "comparison"
    )
    assert build_query_profile("What papers are available in the corpus?").kind == "inventory"


def test_claim_admission_never_transfers_between_quotations_of_one_passage() -> None:
    # Two verified quotations of the SAME passage: one relevant, one not.
    # The irrelevant quotation must stay rejected even though the relevant
    # one shares its passage key (RES-726 review). The referenced study's
    # claim keeps the comparison two-sided.
    anchor_claim = ClaimEvidence(
        passage_key="d" * 64,
        document_version_key=A,
        document_title="Reliability of a linear position transducer",
        claim_text="The referenced study reported excellent reliability.",
        verbatim_quote="Between-session reliability of mean bar velocity was excellent.",
    )
    relevant = ClaimEvidence(
        passage_key="a" * 64,
        document_version_key=B,
        document_title="Measurement procedures in resistance training",
        claim_text="Force plates are the reference standard for ground reaction force.",
        verbatim_quote="Force plates remain the reference standard system for direct ground "
        "reaction force measurement.",
    )
    irrelevant = ClaimEvidence(
        passage_key="a" * 64,
        document_version_key=B,
        document_title="Measurement procedures in resistance training",
        claim_text="Participants were recruited from local athletics clubs.",
        verbatim_quote="Participants were recruited from local athletics clubs.",
    )
    decision = assess_claims(
        FOLLOW_UP, (anchor_claim, relevant, irrelevant), referenced_document_keys=[A]
    )
    assert decision.answerable, decision.to_payload()["detail"]
    # Only the referenced study's claim and the relevant quotation are accepted;
    # the irrelevant quotation's index stays out even though it shares the
    # relevant one's passage key.
    assert decision.accepted_claim_indexes == (0, 1)


def test_more_than_keeps_its_quantitative_cutoff_weight() -> None:
    # "more than 10 weeks" is a real criterion a passage must meet;
    # bare "more" is only framing (RES-726 review).
    cutoff = build_query_profile("Compare force plate protocols lasting more than 10 weeks.")
    assert "more" in salient_terms(cutoff)
    framing = build_query_profile("Are there more studies about velocity-based training?")
    assert "more" not in salient_terms(framing)
