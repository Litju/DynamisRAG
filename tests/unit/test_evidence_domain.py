"""Mutation-sensitive scientific contracts, independent of a database or LLM."""

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from dynamisrag.domain.evidence import (
    Claim,
    EvidenceGraph,
    EvidenceLink,
    EvidenceMember,
    EvidenceSet,
    Relation,
)


def member() -> EvidenceMember:
    return EvidenceMember(
        passage_key="a" * 64,
        version_key="b" * 64,
        span_keys=("c" * 64,),
        provenance_sha256="d" * 64,
    )


def graph(m: EvidenceMember | None = None) -> EvidenceGraph:
    m = m or member()
    evidence_set = EvidenceSet(
        question_id="method-1",
        question="How was the angle measured\u03b2",
        corpus_id="synthetic-fixture-v1",
        selection_context="test fixture",
        members=(m,),
    )
    claim = Claim(
        proposition="The angle was measured using a goniometer.",
        category="METHODOLOGICAL",
        origin="HUMAN_AUTHORED",
        origin_reference="synthetic-test",
        authoring_process="test-v1",
    )
    link = EvidenceLink(
        claim_key=claim.key,
        evidence_set_key=evidence_set.key,
        member=m,
        span_key=m.span_keys[0],
        relation=Relation.CONTEXTUAL,
        rationale="Synthetic example",
        authoring_process="test-v1",
    )
    return EvidenceGraph(evidence_set=evidence_set, claims=(claim,), links=(link,))


def test_roundtrip_and_repeat_identity() -> None:
    first = graph()
    assert EvidenceGraph.model_validate_json(first.model_dump_json()) == first == graph()
    assert first.claims[0].key == graph().claims[0].key
    with pytest.raises(ValidationError):
        first.claims[0].proposition = "changed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposition", "Different meaning"),
        ("population", "children"),
        ("condition", "loaded"),
        ("comparator", "unloaded"),
        ("outcome", "power"),
        ("relationship", "decreases"),
        ("numerical_qualifiers", ["20"]),
        ("units", "degrees"),
        ("temporal_context", "after 6 weeks"),
        ("applicability", ["trained adults"]),
        ("limitations", ["small sample"]),
        ("origin", "MODEL_PROPOSED"),
        ("review_state", "AWAITING_REVIEW"),
    ],
)
def test_meaning_and_provenance_are_identity_inputs(field: str, value: object) -> None:
    original = graph().claims[0]
    changed = Claim.model_validate(original.model_dump(mode="json") | {field: value})
    assert changed.key != original.key


def test_membership_order_and_changes_matter() -> None:
    original = graph().evidence_set
    other = member().model_copy(update={"passage_key": "e" * 64})
    one = EvidenceSet.model_validate(original.model_dump() | {"members": (member(), other)})
    two = EvidenceSet.model_validate(original.model_dump() | {"members": (other, member())})
    assert len({original.key, one.key, two.key}) == 3
    with pytest.raises(ValidationError, match="duplicate passage"):
        EvidenceSet.model_validate(original.model_dump() | {"members": (member(), member())})
    with pytest.raises(ValidationError, match="duplicate source"):
        EvidenceMember.model_validate(member().model_dump() | {"span_keys": ("c" * 64, "c" * 64)})


@pytest.mark.parametrize("factory", [lambda: graph().claims[0], lambda: graph().links[0]])
def test_review_and_identity_forgery_fail_closed(
    factory: Callable[[], Claim | EvidenceLink],
) -> None:
    obj = factory()
    for field, value in (
        ("review_state", "REVIEWED"),
        ("review_state", "EXPERT_APPROVED"),
        ("key", "f" * 64),
        ("reviewer", "invented"),
    ):
        with pytest.raises(ValidationError):
            type(obj).model_validate(obj.model_dump(mode="json") | {field: value})


def test_link_interpretations_are_versioned_and_do_not_approve() -> None:
    link = graph().links[0]
    keys = set[str]()
    for relation in ("SUPPORTS", "CONTRADICTS", "CONTEXTUAL", "INSUFFICIENT"):
        changed = EvidenceLink.model_validate(link.model_dump(mode="json") | {"relation": relation})
        assert changed.review_state == "PROPOSED"
        keys.add(changed.key)
    assert len(keys) == 4
    with pytest.raises(ValidationError, match="span"):
        EvidenceLink.model_validate(link.model_dump() | {"span_key": "f" * 64})


def test_identity_preserves_unicode_whitespace_and_separator_boundaries() -> None:
    from dynamisrag.domain.identity import evidence_identity

    assert evidence_identity("claim-v1", {"b": "ß", "a": "A"}) == evidence_identity(
        "claim-v1", {"a": "A", "b": "ß"}
    )
    assert evidence_identity("claim-v1", ["a\x1fb", "c"]) != evidence_identity(
        "claim-v1", ["a", "b\x1fc"]
    )
    claim = graph().claims[0]
    for text in (claim.proposition + " ", "é", "e\u0301"):
        assert Claim.model_validate(claim.model_dump() | {"proposition": text}).key != claim.key
