"""Deterministic rank-preserving selection over independently verified candidates."""

from collections import Counter

from dynamisrag.domain.evidence import EvidenceMember
from dynamisrag.evidence.contracts import SelectionDecision, SelectionRequest
from dynamisrag.search.retrieval import PassageProvenance
from dynamisrag.search.scientific import ScientificResponse


def redundant(source: PassageProvenance, previous: PassageProvenance) -> bool:
    if source.document_version_key != previous.document_version_key:
        return False
    if source.content_sha256 == previous.content_sha256:
        return True
    return any(
        a.paragraph_key == b.paragraph_key
        and max(a.start_char, b.start_char) < min(a.end_char, b.end_char)
        for a in source.source_spans
        for b in previous.source_spans
    )


def choose(
    request: SelectionRequest,
    response: ScientificResponse,
    verified: dict[str, EvidenceMember],
) -> tuple[tuple[PassageProvenance, ...], tuple[SelectionDecision, ...]]:
    policy = request.policy
    selected: list[PassageProvenance] = []
    decisions: dict[str, SelectionDecision] = {}
    docs: Counter[str] = Counter()
    sections: Counter[tuple[str, str | None]] = Counter()
    document_bytes: Counter[str] = Counter()
    used = 0
    bm25 = {c.provenance.passage_key: c for c in response.bm25.candidates}
    dense = {c.provenance.passage_key: c for c in response.dense.candidates}
    fused = {h.passage_key: h for h in response.fusion.hits}

    # Include lane-only candidates; they were retrieved but never enter the fused window.
    for key in sorted(bm25.keys() | dense.keys()):
        source = (bm25.get(key) or dense[key]).provenance
        hit = fused.get(key)
        decisions[key] = SelectionDecision(
            passage_key=key,
            fused_rank=hit.final_rank if hit else None,
            rrf_score=hit.rrf_score if hit else None,
            bm25_rank=bm25[key].rank if key in bm25 else None,
            bm25_score=bm25[key].raw_score if key in bm25 else None,
            dense_rank=dense[key].rank if key in dense else None,
            dense_score=dense[key].raw_score if key in dense else None,
            version_key=source.document_version_key,
            document_key=source.document_canonical_key,
            section_key=source.section_key,
            disposition="NOT_SELECTED" if key in verified else "SOURCE_INVALID",
            reason=(
                "canonical_mismatch"
                if key not in verified
                else "outside_authorized_scope"
                if request.allowed_versions
                and source.document_version_key not in request.allowed_versions
                else "outside_fused_window"
            ),
        )
    for relaxed in (False, True):
        if relaxed and not policy.relax_preferences:
            break
        for hit in response.fusion.hits:
            key, source = hit.passage_key, hit.provenance
            old = decisions[key]
            if relaxed and old.disposition != "DIVERSITY_DEFERRED":
                continue
            if key not in verified or old.reason == "outside_authorized_scope":
                continue
            doc = source.document_canonical_key
            section = (source.document_version_key, source.section_key)
            size = len(source.text.encode("utf-8"))
            disposition, reason = "SELECTED", "soft_quota_relaxed" if relaxed else "ranked"
            if any(redundant(source, p) for p in selected):
                disposition, reason = "DUPLICATE", "same_version_redundancy"
            elif len(selected) >= policy.max_passages:
                disposition, reason = "BUDGET_EXCEEDED", "passage_budget"
            elif (
                (doc not in docs and len(docs) >= policy.max_documents)
                or docs[doc] >= policy.max_per_document
                or sections[section] >= policy.max_per_section
            ):
                disposition, reason = "BUDGET_EXCEEDED", "hard_quota"
            elif (
                used + size > policy.max_bytes
                or document_bytes[doc] + size > policy.max_document_bytes
            ):
                disposition, reason = "BUDGET_EXCEEDED", "byte_budget"
            elif not relaxed and (
                (policy.prefer_document_diversity and docs[doc] >= policy.soft_per_document)
                or (
                    policy.prefer_section_diversity and sections[section] >= policy.soft_per_section
                )
            ):
                disposition, reason = "DIVERSITY_DEFERRED", "soft_quota"
            decisions[key] = SelectionDecision.model_validate(
                old.model_dump() | {"disposition": disposition, "reason": reason}
            )
            if disposition == "SELECTED":
                selected.append(source)
                docs[doc] += 1
                sections[section] += 1
                document_bytes[doc] += size
                used += size
    ordered = sorted(decisions.values(), key=lambda d: (d.fused_rank or 1000, d.passage_key))
    return tuple(selected), tuple(ordered)
