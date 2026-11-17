# ruff: noqa: RUF001 - the exact acceptance question contains an en dash
"""RES-638: broad multi-aspect relevance, acronym inference and model-chosen literature queries."""

from __future__ import annotations

from typing import cast

from dynamisrag.agent.answerability import (
    ClaimEvidence,
    EvidenceCandidate,
    acronym_expansions,
    assess_claims,
    assess_retrieval,
    build_query_profile,
)
from dynamisrag.corpus.research import build_query_ladder, infer_acronyms, model_query_levels

INITIAL = (
    "What does the scientific literature establish about force–velocity profiling using "
    "linear position transducers in resistance training? Evaluate measurement validity, "
    "reliability, applications and limitations."
)
FOLLOW_UP = "How do LPT measurements compare with reference measurement systems?"


def _c(key: str, doc: str, title: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        passage_key=key * 64, document_version_key=doc * 64, title=title, text=text
    )


def test_enumerated_aspects_are_any_one_not_all_and_framing_words_carry_no_topic() -> None:
    profile = build_query_profile(INITIAL)
    assert profile.aspects == ("validi", "reliab", "applic", "limita")
    assert "us" not in profile.facets  # "using" is a question word, not a topic
    reliability = _c(
        "a",
        "1",
        "Reliability of a linear position transducer for force-velocity profiles",
        "Force-velocity profiling in resistance training with the linear position transducer "
        "showed excellent between-session reliability of velocity measurement.",
    )
    off_topic = _c("b", "2", "Sprint mechanics", "Sprint times improved after resisted sprints.")
    decision = assess_retrieval(INITIAL, (reliability, off_topic))
    assert decision.answerable
    assert decision.accepted_passage_keys == (reliability.passage_key,)
    payload = decision.to_payload()
    assert payload["aspect_coverage"] == {"validi": 0, "reliab": 1, "applic": 0, "limita": 0}
    assert set(cast("list[str]", payload["uncovered_aspects"])) == {"applic", "limita", "validi"}


def test_a_core_topic_absent_everywhere_still_abstains() -> None:
    unrelated = _c("c", "3", "Hamstring injury", "Validity and reliability of hamstring tests.")
    assert not assess_retrieval(INITIAL, (unrelated,)).answerable


def test_aspect_claims_need_an_aspect_or_value_in_the_quote_not_the_title() -> None:
    title = "Validity of a linear position transducer for force-velocity profiling"
    claims = (
        ClaimEvidence("a" * 64, "1" * 64, title, "x", "The device showed an ICC of 0.98."),
        ClaimEvidence("b" * 64, "1" * 64, title, "x", "Participants were recruited locally."),
    )
    decision = assess_claims(INITIAL, claims)
    assert decision.answerable
    assert decision.accepted_claim_indexes == (0,)


def test_acronym_is_inferred_from_the_earlier_turn_and_matches_its_expansion() -> None:
    assert infer_acronyms(FOLLOW_UP, INITIAL) == {"LPT": "linear position transducer"}
    assert acronym_expansions(FOLLOW_UP, [INITIAL])["lpt"] == ("linear", "positi", "transd")
    passages = (
        _c(
            "d",
            "4",
            "Concurrent validity of a linear position transducer",
            "Compared with the criterion force plate, the linear position transducer agreed.",
        ),
        _c("e", "5", "Bench press velocity", "The LPT agreed with the gold standard system."),
    )
    decision = assess_retrieval(FOLLOW_UP, passages, context=INITIAL)
    assert decision.answerable
    assert set(decision.accepted_passage_keys) == {p.passage_key for p in passages}


def test_fixed_rule_ladder_uses_whole_phrases_and_reference_wording() -> None:
    ladder = " ".join(q.europe_pmc() for q in build_query_ladder(INITIAL))
    assert 'TITLE_ABS:"linear position transducer"' in ladder
    assert 'TITLE_ABS:"position transducers"' not in ladder
    follow = build_query_ladder(FOLLOW_UP, INITIAL)[0].europe_pmc()
    assert 'TITLE_ABS:"LPT"' in follow
    assert 'TITLE_ABS:"gold standard"' in follow


def test_model_queries_are_validated_never_repaired_into_query_syntax() -> None:
    injected = 'x" OR "y'
    levels = model_query_levels(
        [[["linear position transducer", injected], ["force plate"]], [["AND:("]]]
    )
    assert len(levels) == 1
    assert levels[0].groups == (("linear position transducer",), ("force plate",))
