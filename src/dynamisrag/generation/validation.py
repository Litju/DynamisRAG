"""All-or-nothing structural binding from typed proposals to canonical evidence."""

from __future__ import annotations

from dataclasses import dataclass

from dynamisrag.domain.evidence import (
    Claim,
    EvidenceGraph,
    EvidenceLink,
    EvidenceMember,
    EvidenceSet,
    Relation,
)
from dynamisrag.generation.provider import DraftSynthesis


@dataclass(frozen=True)
class VerifiedSpan:
    span_key: str
    text: str
    paragraph_text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class VerifiedSource:
    member: EvidenceMember
    text: str
    spans: tuple[VerifiedSpan, ...]


def build_graph(
    evidence_set: EvidenceSet,
    sources: tuple[VerifiedSource, ...],
    candidate: DraftSynthesis,
    *,
    model_identity: str,
    policy_sha256: str,
    allowed_spans: dict[str, frozenset[str]],
) -> EvidenceGraph:
    """Revalidate bytes and identities, then reject the entire graph on any bad citation."""
    evidence_set = EvidenceSet.model_validate_json(evidence_set.model_dump_json())
    draft = DraftSynthesis.model_validate_json(candidate.model_dump_json())
    source_by_key = {s.member.passage_key: s for s in sources}
    member_by_key = {m.passage_key: m for m in evidence_set.members}
    if (
        len(source_by_key) != len(sources)
        or set(source_by_key) != set(member_by_key)
        or any(source_by_key[k].member != member_by_key[k] for k in member_by_key)
    ):
        raise ValueError("materialized source inventory differs from EvidenceSet")
    claims: list[Claim] = []
    links: list[EvidenceLink] = []
    seen: set[tuple[str, str]] = set()
    for proposed in draft.claims:
        supports = [r for r in proposed.evidence if r.relation == Relation.SUPPORTS]
        if not any(r.exact_quote == proposed.proposition for r in supports):
            raise ValueError("claim proposition must be an exact supporting source quote")
        refs = {(r.passage_key, r.span_key, r.relation) for r in proposed.evidence}
        if len(refs) != len(proposed.evidence):
            raise ValueError("duplicate claim evidence reference")
        claim = Claim(
            proposition=proposed.proposition,
            category=proposed.category,
            origin="MODEL_PROPOSED",
            origin_reference=model_identity,
            authoring_process=f"structured-grounded-extractive-v1:{policy_sha256}",
        )
        for ref in proposed.evidence:
            source = source_by_key.get(ref.passage_key)
            if (
                source is None
                or ref.span_key not in member_by_key[ref.passage_key].span_keys
                or ref.span_key not in allowed_spans.get(ref.passage_key, frozenset())
            ):
                raise ValueError("unknown or nonmember evidence reference")
            span = next((s for s in source.spans if s.span_key == ref.span_key), None)
            if (
                span is None
                or span.paragraph_text[span.start_char : span.end_char] != span.text
                or not _complete_source_sentence(span, ref.exact_quote)
            ):
                raise ValueError("quote is absent from the cited canonical source span")
            if ref.relation == Relation.INSUFFICIENT:
                raise ValueError("insufficient evidence cannot be linked as a supported claim")
            links.append(
                EvidenceLink(
                    claim_key=claim.key,
                    evidence_set_key=evidence_set.key,
                    member=source.member,
                    span_key=span.span_key,
                    relation=ref.relation,
                    rationale=ref.exact_quote,
                    authoring_process=f"structured-grounded-extractive-v1:{policy_sha256}",
                )
            )
        identity = (claim.proposition, claim.category)
        if identity in seen:
            raise ValueError("duplicate claim identity")
        seen.add(identity)
        claims.append(claim)
    if not claims:
        raise ValueError("abstention has no EvidenceGraph")
    return EvidenceGraph(evidence_set=evidence_set, claims=tuple(claims), links=tuple(links))


def _complete_source_sentence(span: VerifiedSpan, quote: str) -> bool:
    if quote != quote.strip() or not quote.endswith((".", "!", "?")):
        return False
    start = span.paragraph_text.find(quote, span.start_char, span.end_char)
    if start < span.start_char or start + len(quote) > span.end_char:
        return False
    before = span.paragraph_text[:start].rstrip()
    after = span.paragraph_text[start + len(quote) :]
    return (not before or before.endswith((".", "!", "?"))) and (not after or after[0].isspace())
