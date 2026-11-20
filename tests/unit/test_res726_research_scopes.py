"""RES-726: browsing, hard source boundaries, and soft comparison referents are distinct."""

from typing import cast

from dynamisrag.agent.graph import (
    ConversationalResearchOrchestrator,
    LiteratureAcquisitionPort,
    ResearchAgentState,
    _comparison_anchor_present,  # pyright: ignore[reportPrivateUsage]
    _reject_unanchored_comparison_claims,  # pyright: ignore[reportPrivateUsage]
    _thread_research_scopes,  # pyright: ignore[reportPrivateUsage]
)

A, B = "a" * 64, "b" * 64


def test_automatic_comparison_followup_keeps_reference_without_blocking_discovery() -> None:
    for question in (
        "How does its measurement compare with force plates?",
        "How does it compare with a criterion system?",
        "Compare those studies with a new force plate validation.",
    ):
        hard_query_scope, references = _thread_research_scopes(question, (), (A,))
        assert hard_query_scope == []  # new papers are still discoverable
        assert references == [A]  # original study cannot silently change


def test_restricted_comparison_only_uses_the_explicit_user_allowlist() -> None:
    scope, references = _thread_research_scopes(
        "How does its reliability compare with the other selected study?", (A, B), (A,)
    )
    assert scope == [A, B]
    assert references == [A]


def test_ordinary_pronoun_followups_remain_anchored_to_prior_study() -> None:
    assert _thread_research_scopes("What was its sample size?", (), (A,)) == ([A], [])
    assert _thread_research_scopes("What were its results?", (A, B), (A,)) == ([A], [])
    assert _thread_research_scopes("Compare the methods of these papers.", (), ()) == ([], [])


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
