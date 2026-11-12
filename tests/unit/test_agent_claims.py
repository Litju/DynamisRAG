"""Layered claim verification: a source-exact quote must not launder an unsupported proposition.

Covers the pure assessors (numeric provenance, lexical semantic support), the canonical
``verify_claims`` tool contract (each layer reported separately), and an end-to-end run in which a
scripted model proposes a mix of faithful, reversed, overstated, mis-numbered and fabricated
claims against the same real passage.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from dynamisrag.agent.claims import assess_numeric_provenance, assess_semantic_support
from dynamisrag.agent.contracts import (
    CandidateClaimInput,
    DataScope,
    VerifyClaimsInput,
)
from dynamisrag.agent.persistence import RunStatus
from tests.agent_stack import build_stack, scripted_runtime, tool_call_message

QUOTE = (
    "During depth jumping, Achilles tendon stiffness increased by 24.5% (p < 0.01) while "
    "gastrocnemius medialis fascicle length behavior shifted toward isometric contraction."
)


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        ("Achilles tendon stiffness rose by 24.5% during depth jumping.", "VERIFIED"),
        ("Stiffness increased by 24.5% (p < 0.01).", "VERIFIED"),
        ("Achilles tendon stiffness was higher after depth jumping.", "NOT_APPLICABLE"),
        ("Achilles tendon stiffness increased by 42.5%.", "FAILED"),
        ("Achilles tendon stiffness increased by 24.5 percent.", "VERIFIED"),
        ("Stiffness increased by 12 weeks.", "FAILED"),
        ("The 1,200 participants improved.", "FAILED"),
    ],
)
def test_numeric_provenance_is_checked_against_the_quoted_span(claim: str, expected: str) -> None:
    assert assess_numeric_provenance(claim, QUOTE)[0] == expected


def test_a_number_from_elsewhere_in_the_passage_does_not_count() -> None:
    # "12 weeks" is real in the passage but NOT in the quoted sentence: wrong-sentence provenance.
    assert assess_numeric_provenance("Training lasted 12 weeks.", QUOTE)[0] == "FAILED"


def test_absolute_value_restated_as_a_percentage_is_caught() -> None:
    result, reasons = assess_numeric_provenance(
        "Jump height increased by 36.8%.", "Jump height increased from 31.2 to 36.8 cm."
    )
    assert result == "FAILED" and "percentage" in reasons[0]


@pytest.mark.parametrize(
    ("claim", "status", "reason"),
    [
        (QUOTE, "IDENTICAL_TO_QUOTE", None),
        ("Achilles tendon stiffness increased by 24.5%", "IDENTICAL_TO_QUOTE", None),
        (
            "Achilles tendon stiffness rose by 24.5% during depth jumping.",
            "SUPPORTED_BY_LEXICAL_HEURISTIC",
            None,
        ),
        (
            "Achilles tendon stiffness decreased by 24.5% during depth jumping.",
            "UNSUPPORTED",
            "reversed_direction",
        ),
        (
            "Depth jumping caused Achilles tendon stiffness to increase by 24.5%.",
            "UNSUPPORTED",
            "causal_language_not_in_quote",
        ),
        (
            "Depth jumping always increases Achilles tendon stiffness.",
            "UNSUPPORTED",
            "universal_language_not_in_quote",
        ),
        (
            "Quadriceps hypertrophy was greater in sprinters.",
            "UNSUPPORTED",
            "content_not_in_quote",
        ),
        (
            "Achilles tendon stiffness did not increase during depth jumping.",
            "UNSUPPORTED",
            "negation_mismatch",
        ),
    ],
)
def test_semantic_support_rejects_overreach_and_accepts_faithful_wording(
    claim: str, status: str, reason: str | None
) -> None:
    assessment = assess_semantic_support(claim, QUOTE)
    assert assessment.status == status
    if reason is not None:
        assert any(r.startswith(reason) for r in assessment.reasons)


@pytest.mark.parametrize(
    ("claim", "quote", "reason"),
    [
        # Antonyms outside any short word list (found by independent review).
        (
            "Treatment lowers mortality in elderly patients.",
            "Treatment raises mortality in elderly patients.",
            "reversed_direction",
        ),
        ("Exercise elevates cortisol.", "Exercise suppresses cortisol.", "reversed_direction"),
        # Swapped roles: same words, opposite assignment.
        (
            "A decreased and B increased.",
            "A increased and B decreased.",
            "reversed_direction",
        ),
        # A fragment that strips the negation from its own sentence is not "the quote".
        (
            "significant increase in mortality",
            "There was no significant increase in mortality.",
            "negation_mismatch",
        ),
        # A hedged finding restated as a definite one.
        (
            "Training increases tendon stiffness.",
            "Training may increase tendon stiffness in some participants.",
            "hedge_dropped",
        ),
    ],
)
def test_review_found_bypasses_are_now_rejected(claim: str, quote: str, reason: str) -> None:
    assessment = assess_semantic_support(claim, quote)
    assert assessment.status == "UNSUPPORTED", assessment
    assert reason in assessment.reasons


def test_a_percentage_restated_as_a_count_is_caught() -> None:
    result, reasons = assess_numeric_provenance(
        "5 patients responded to treatment.", "5% of patients responded to treatment."
    )
    assert result == "FAILED" and "quote_gives_percentage_but_claim_does_not" in reasons[0]


def test_faithful_paraphrases_still_pass_after_hardening() -> None:
    for claim in (
        "Achilles tendon stiffness rose by 24.5% during depth jumping.",
        "Depth jumping raised Achilles tendon stiffness by 24.5%.",
    ):
        assert assess_semantic_support(claim, QUOTE).status != "UNSUPPORTED", claim


def test_dropped_negation_in_the_other_direction_is_also_caught() -> None:
    quote = "There was no significant difference in tendon stiffness between groups."
    assessment = assess_semantic_support("Tendon stiffness differed between groups.", quote)
    assert assessment.status == "UNSUPPORTED"
    assert "negation_mismatch" in assessment.reasons


# ---------------------------------------------------------------------------
# Tool-level: layers are reported separately
# ---------------------------------------------------------------------------


def _verify(stack_claims: list[tuple[str, str]]) -> list[dict[str, Any]]:
    stack = build_stack()
    version_key, passage_keys, span_keys = stack.studies["PMC5263151"]
    scope = DataScope(user_id="u", thread_id="t", run_id="r")
    # passage 1 holds the 24.5% sentence
    claims = tuple(
        CandidateClaimInput(
            claim_text=text,
            passage_key=passage_keys[1],
            span_key=span_keys[1],
            verbatim_quote=quote,
        )
        for text, quote in stack_claims
    )
    assert version_key
    env = stack.registry.verify_claims(VerifyClaimsInput(question="q", claims=claims), scope)
    assert env.ok and env.payload is not None
    return list(env.payload["items"])


def test_exact_quote_with_unsupported_proposition_is_rejected_but_quote_layer_stays_true() -> None:
    items = _verify(
        [
            ("Achilles tendon stiffness decreased by 24.5% during depth jumping.", QUOTE),
            ("Achilles tendon stiffness rose by 24.5% during depth jumping.", QUOTE),
            ("Achilles tendon stiffness increased by 42.5%.", QUOTE),
            (QUOTE, QUOTE),
        ]
    )
    reversed_, faithful, wrong_number, identical = items

    # The quotation is exact in all four: layer 1 is independent of the others.
    assert all(i["quote_verified"] for i in items)
    assert reversed_["verified"] is False
    assert reversed_["semantic_support"] == "UNSUPPORTED"
    assert reversed_["rejection_reason"].startswith("claim_not_supported_by_quoted_span")
    assert wrong_number["verified"] is False
    assert wrong_number["numeric_provenance"] == "FAILED"
    assert wrong_number["rejection_reason"].startswith("unverified_numerical_value_in_claim")

    assert faithful["verified"] is True
    assert faithful["numeric_provenance"] == "VERIFIED"
    assert faithful["semantic_support"] == "SUPPORTED_BY_LEXICAL_HEURISTIC"
    assert identical["verified"] is True and identical["semantic_support"] == "IDENTICAL_TO_QUOTE"
    for item in items:  # nothing here is expert-approved or non-provisional
        assert item["expert_review"] == "NOT_REVIEWED"
        assert item["origin"] == "MODEL_PROPOSED" and item["review_state"] == "PROPOSED"


def test_fabricated_quote_fails_layer_one() -> None:
    items = _verify([("Stiffness rose.", "Achilles tendon stiffness doubled in all participants.")])
    assert items[0]["quote_verified"] is False and items[0]["verified"] is False
    assert items[0]["rejection_reason"] == "verbatim_quote_absent_from_canonical_span"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _evidence(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    text = "\n".join(str(m.content) for m in messages)
    marker = "CANDIDATE_EVIDENCE_JSON:"
    return json.loads(text[text.index(marker) + len(marker) :].strip())


def test_run_keeps_only_supported_claims_and_reports_every_rejection() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if any("CANDIDATE_EVIDENCE_JSON" in str(m.content) for m in messages):
            item = next(e for e in _evidence(messages) if "24.5%" in e["verbatim_quote"])
            base = {"passage_key": item["passage_key"], "span_key": item["span_key"]}
            quote = item["verbatim_quote"]
            claims = [
                {**base, "verbatim_quote": quote, "claim_text": text}
                for text in (
                    "Achilles tendon stiffness rose by 24.5% during depth jumping.",  # faithful
                    "Achilles tendon stiffness decreased by 24.5% during depth jumping.",
                    "Depth jumping caused Achilles tendon stiffness to increase by 24.5%.",
                    "Achilles tendon stiffness increased by 42.5% during depth jumping.",
                )
            ] + [
                {
                    **base,
                    "verbatim_quote": "Achilles tendon stiffness doubled in every participant.",
                    "claim_text": "Stiffness doubled.",
                }
            ]
            return AIMessage(content=json.dumps({"claims": claims, "abstain_reason": None}))
        if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return AIMessage(content="ok")
        return tool_call_message(("hybrid_search", {"query": "achilles tendon stiffness"}))

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    thread = stack.store.create_thread(owner_id="u", title="t")
    run = stack.orchestrator.execute_run(
        thread_id=thread.thread_id,
        user_id="u",
        question="What happened to Achilles tendon stiffness during depth jumping?",
        idempotency_key="adv",
    )

    assert run.status == RunStatus.COMPLETED
    assert [c["claim_text"] for c in run.verified_citations] == [
        "Achilles tendon stiffness rose by 24.5% during depth jumping."
    ]
    reasons = sorted(str(r["rejection_reason"]).split(":")[0] for r in run.rejected_citations)
    assert reasons == [
        "claim_not_supported_by_quoted_span",
        "claim_not_supported_by_quoted_span",
        "unverified_numerical_value_in_claim",
        "verbatim_quote_absent_from_canonical_span",
    ]
    answer = run.answer_markdown or ""
    assert "rose by 24.5%" in answer and "decreased by 24.5%" not in answer
    assert "4 proposed claim(s) were **rejected**" in answer
    assert "expert review `NOT_REVIEWED`" in answer and "MODEL_PROPOSED" in answer
    # The record never claims more than "proposed".
    assert all(c["review_state"] == "PROPOSED" for c in run.verified_citations)


def test_run_whose_every_claim_is_unsupported_abstains() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if any("CANDIDATE_EVIDENCE_JSON" in str(m.content) for m in messages):
            item = _evidence(messages)[0]
            return AIMessage(
                content=json.dumps(
                    {
                        "claims": [
                            {
                                "passage_key": item["passage_key"],
                                "span_key": item["span_key"],
                                "verbatim_quote": item["verbatim_quote"],
                                "claim_text": "Quadriceps hypertrophy was greater in sprinters.",
                            }
                        ]
                    }
                )
            )
        if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return AIMessage(content="ok")
        return tool_call_message(("hybrid_search", {"query": "achilles tendon stiffness"}))

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    thread = stack.store.create_thread(owner_id="u", title="t")
    run = stack.orchestrator.execute_run(
        thread_id=thread.thread_id,
        user_id="u",
        question="What happened to Achilles tendon stiffness during depth jumping?",
        idempotency_key="all-bad",
    )
    assert run.status == RunStatus.ABSTAINED
    assert run.abstention_reason == "canonical_citation_verification_failed"
    assert run.verified_citations == () and len(run.rejected_citations) == 1
