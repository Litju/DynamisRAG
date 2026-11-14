"""Structured evidence relevance and answerability gate (RES-638).

This module answers a narrower question than natural-language inference:

    *Does the retrieved, canonically verified evidence contain the entities and
    the query facets the user asked about, and do the verified quotations support
    the requested proposition strongly enough to attempt synthesis?*

It deliberately separates four dimensions that the previous pipeline conflated:

1. **Lexical similarity** - weighted query-term coverage over a candidate's
   title, section and passage text. A single incidental word match is not
   relevance.
2. **Source authenticity** - the caller must mark a candidate canonical. This
   gate never upgrades an unverified passage.
3. **Query relevance** - entity/designation coverage plus weighted facet
   coverage. An unsupported entity (for example, a planet absent from the
   corpus) is a hard failure even when incidental words match.
4. **Claim support** - the existing deterministic quotation/numeric/lexical
   checks in :mod:`dynamisrag.agent.claims`; this gate adds only *question*
   relevance, never semantic entailment.

**Honest limits.** The gate is a calibrated lexical/structural heuristic. A pass
means "the evidence addresses the asked-about entities and facets and the
verified quotation survives the existing checks". It is labelled
``SUPPORTED_BY_STRUCTURED_EVIDENCE_GATE`` and is **not** natural-language
entailment, scientific validation or expert review. Thresholds are calibrated
in ``tests/fixtures/answerability/calibration.json`` and checked against a
disjoint held-out set; production code never branches on a question string.

Decision rules are corpus-adaptive where possible (candidate frequencies, query
kind) and question-agnostic. No question-specific blocklist is used.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from dynamisrag.generation.scientific import scientific_terms

__all__ = [
    "CALIBRATED_MIN_CLAIM_COVERAGE",
    "CALIBRATED_MIN_COMPARISON_COVERAGE",
    "CALIBRATED_MIN_RELEVANCE",
    "GATE_VERSION",
    "AnswerabilityDecision",
    "CandidateAssessment",
    "EvidenceCandidate",
    "QueryKind",
    "QueryProfile",
    "assess_claims",
    "assess_retrieval",
    "build_query_profile",
]

GATE_VERSION: Final = "structured-evidence-gate-v1"

# Calibrated on tests/fixtures/answerability/calibration.json; the calibration test
# recomputes the separating threshold and fails if this constant drifts.
CALIBRATED_MIN_RELEVANCE: Final[float] = 0.60
CALIBRATED_MIN_COMPARISON_COVERAGE: Final[float] = 0.30
CALIBRATED_MIN_CLAIM_COVERAGE: Final[float] = 0.60
AMBIGUITY_MARGIN: Final[float] = 0.08
MAX_REPORTED_TERMS: Final = 8

QueryKind = Literal["factual", "comparison", "inventory", "metadata"]
Verdict = Literal["ANSWERABLE", "ABSTAIN"]
ClaimSupport = Literal["SUPPORTED_BY_STRUCTURED_EVIDENCE_GATE", "NOT_ASSESSED"]

_GENERIC_QUERY_TERMS: Final[frozenset[str]] = frozenset(
    {
        "about",
        "according",
        "after",
        "available",
        "before",
        "change",
        "changed",
        "changes",
        "across",
        "answer",
        "compare",
        "comparison",
        "concern",
        "concerns",
        "corpus",
        "describe",
        "detail",
        "details",
        "difference",
        "differences",
        "discuss",
        "does",
        "effect",
        "effects",
        "evidence",
        "explain",
        "finding",
        "findings",
        "happen",
        "happened",
        "happens",
        "influence",
        "information",
        "intact",
        "keep",
        "list",
        "paper",
        "papers",
        "please",
        "question",
        "report",
        "reported",
        "reports",
        "recommend",
        "recommended",
        "research",
        "result",
        "results",
        "same",
        "show",
        "shown",
        "shows",
        "specifically",
        "state",
        "study",
        "studies",
        "summarize",
        "summary",
        "tell",
        "thread",
        "versus",
        "work",
    }
)

# Meta facets are extracted structurally by the comparison/metadata tools rather
# than by a passage sentence; they are not required lexical coverage for those
# question kinds.
_META_FACETS: Final[frozenset[str]] = frozenset(
    {
        "approach",
        "characteristic",
        "design",
        "method",
        "methodology",
        "outcome",
        "population",
        "procedure",
        "protocol",
        "sample",
    }
)

_QUESTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "between",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "many",
        "may",
        "might",
        "of",
        "on",
        "or",
        "much",
        "several",
        "should",
        "than",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
    }
)

# A small, general scientific abbreviation/lexicon normaliser. These are domain
# concepts, not question-specific rules; both the question and the evidence pass
# through the same map, so a paraphrase such as "muscle activity" and "EMG" can
# agree without either being special-cased.
_CONCEPT_ALIASES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"\b(?:electromyograph\w*|semg|emg)\b", re.IGNORECASE),
        "emg",
    ),
    (
        re.compile(
            r"\b(?:muscle\s+activation|muscle\s+activity|muscular\s+activity|"
            r"myoelectric\s+activity)\b",
            re.IGNORECASE,
        ),
        "muscle_activity",
    ),
    (
        re.compile(
            r"\b(?:vertical\s+jump|jump\s+height|jumping\s+height|"
            r"countermovement\s+jump|cmj)\b",
            re.IGNORECASE,
        ),
        "jump_height",
    ),
    (
        re.compile(r"\b(?:depth\s+jump(?:ing|s)?|drop\s+jump(?:ing|s)?|dj)\b", re.IGNORECASE),
        "depth_jump",
    ),
    (
        re.compile(r"\b(?:one\s+repetition\s+maximum|1\s*rm)\b", re.IGNORECASE),
        "1rm",
    ),
    (
        re.compile(r"\b(?:rate\s+of\s+force\s+development|rfd)\b", re.IGNORECASE),
        "rfd",
    ),
    (
        re.compile(r"\b(?:maximal\s+oxygen\s+uptake|vo2\s*max|vo2max)\b", re.IGNORECASE),
        "vo2max",
    ),
    (
        re.compile(r"\b(?:range\s+of\s+motion|rom)\b", re.IGNORECASE),
        "range_of_motion",
    ),
    (
        re.compile(r"\b(?:randomi[sz]ed\s+controlled\s+trial|rct)\b", re.IGNORECASE),
        "randomized_controlled_trial",
    ),
    (
        re.compile(
            r"\b(?:participants?|subjects?|athletes?|players?|cricketers?|adults?|"
            r"individuals?|men|women)\b",
            re.IGNORECASE,
        ),
        "population_member",
    ),
)

_DESIGNATION: Final = re.compile(
    r"\b[A-Z][A-Za-z]*(?:[-\u2013]?\d+[A-Za-z0-9]*(?:[-\u2013][A-Za-z0-9]+)*)\b"
)
_PROPER_PHRASE: Final = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")
_SENTENCE_START: Final = re.compile(r"(?:^|[.!?]\s+|\n)\s*([A-Z][a-z]+)")
_YEAR: Final = re.compile(r"^(?:19|20)\d{2}$")

_COMPARISON: Final = re.compile(
    r"\b(?:compare|comparison|compared|versus|vs\.?|differ(?:ence|ences|ent)?|"
    r"across\s+(?:the\s+)?(?:papers|studies)|between\s+(?:the\s+)?(?:papers|studies))\b",
    re.IGNORECASE,
)
_INVENTORY: Final = re.compile(
    r"\b(?:which|what)\s+(?:papers|studies|articles)\s+(?:are|exist|available)\b|"
    r"\blist\s+(?:the\s+)?(?:available\s+)?(?:papers|studies)\b|"
    r"\bavailable\s+(?:papers|studies)\b",
    re.IGNORECASE,
)
_METADATA: Final = re.compile(
    r"\b(?:identifiers?|doi|pmid|pmcid|section\s+titles?|metadata|authors?|journal)\b",
    re.IGNORECASE,
)
_SINGLE_VALUE: Final = re.compile(
    r"\b(?:how\s+(?:many|much|long|far|fast|often)|what\s+(?:is|was|were)\s+the\s+"
    r"(?:value|number|rate|period|duration|interval|angle|height|force|stiffness|"
    r"percentage|percent|ratio|frequency|volume|load|distance|time))\b",
    re.IGNORECASE,
)

_WORD: Final = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")


@dataclass(frozen=True)
class QueryProfile:
    """Question-derived requirements. No corpus or question-specific branches."""

    question: str
    kind: QueryKind
    entities: tuple[tuple[str, ...], ...]
    facets: tuple[str, ...]
    facet_weights: tuple[float, ...]
    meta_facets: tuple[str, ...]

    @property
    def minimum_coverage(self) -> float:
        return (
            CALIBRATED_MIN_COMPARISON_COVERAGE
            if self.kind == "comparison"
            else CALIBRATED_MIN_RELEVANCE
        )

    @property
    def total_weight(self) -> float:
        return sum(self.facet_weights)

    @property
    def entity_stems(self) -> frozenset[str]:
        return frozenset(part for entity in self.entities for part in entity)


@dataclass(frozen=True)
class EvidenceCandidate:
    """One canonical retrieval or claim candidate, as the caller observed it."""

    passage_key: str
    document_version_key: str
    title: str
    text: str
    section_title: str | None = None
    rank: int = 0
    canonical_verified: bool = True


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: EvidenceCandidate
    lexical_similarity: float
    query_relevance: float
    entity_support: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    missing_entities: tuple[str, ...]
    admissible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnswerabilityDecision:
    verdict: Verdict
    code: str
    detail: str
    gate_version: str
    question_kind: QueryKind
    claim_support: ClaimSupport
    required_coverage: float
    comparison_document_count: int
    assessments: tuple[CandidateAssessment, ...]
    accepted_passage_keys: tuple[str, ...]
    rejected_passage_keys: tuple[str, ...]
    contradictory: bool = False
    ambiguous: bool = False

    @property
    def answerable(self) -> bool:
        return self.verdict == "ANSWERABLE"

    def to_payload(self) -> dict[str, object]:
        """Bounded, JSON-safe audit payload (no source text, no credentials)."""
        return {
            "verdict": self.verdict,
            "code": self.code,
            "detail": self.detail,
            "gate_version": self.gate_version,
            "question_kind": self.question_kind,
            "claim_support": self.claim_support,
            "required_coverage": round(self.required_coverage, 4),
            "comparison_document_count": self.comparison_document_count,
            "accepted_passage_keys": list(self.accepted_passage_keys),
            "rejected_passage_keys": list(self.rejected_passage_keys),
            "contradictory": self.contradictory,
            "ambiguous": self.ambiguous,
            "candidates": [
                {
                    "passage_key": a.candidate.passage_key,
                    "lexical_similarity": round(a.lexical_similarity, 4),
                    "query_relevance": round(a.query_relevance, 4),
                    "entity_support": round(a.entity_support, 4),
                    "matched_terms": list(a.matched_terms),
                    "missing_terms": list(a.missing_terms),
                    "missing_entities": list(a.missing_entities),
                    "admissible": a.admissible,
                    "reasons": list(a.reasons),
                }
                for a in self.assessments
            ],
        }


def _fold(text: str) -> str:
    return (
        unicodedata.normalize("NFKC", text)
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .casefold()
    )


def _stem(token: str) -> str:
    if token.isdigit():
        return token
    if len(token) <= 4:
        return token
    if token.endswith("ies"):
        return _stem(token[:-3] + "y")
    if token.endswith("ation"):
        return _stem(token[:-5] + "at")
    if token.endswith("ing"):
        return _stem(token[:-3])
    if token.endswith("ed"):
        return _stem(token[:-2])
    if token.endswith("s") and not token.endswith("ss"):
        return _stem(token[:-1])
    # A bounded prefix collapses common inflection pairs that the suffix ladder
    # cannot (complete/completed, participant/participants) without pretending to
    # be a linguistic stemmer. It is deterministic and applied to both sides.
    return token[:6]


_META_STEMS: Final[frozenset[str]] = frozenset(_stem(term) for term in _META_FACETS)
_GENERIC_STEMS: Final[frozenset[str]] = frozenset(_stem(term) for term in _GENERIC_QUERY_TERMS)
_UP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "increase", "increased", "increases", "higher", "greater", "improved",
        "improvement", "improves", "improve", "enhanced", "larger", "more", "stronger",
        "longer", "faster", "rise", "rises", "rose", "elevated", "grew", "growth",
    }
)  # fmt: skip
_DOWN_WORDS: Final[frozenset[str]] = frozenset(
    {
        "decrease", "decreased", "decreases", "reduced", "reduction", "lower", "lowers",
        "declined", "less", "smaller", "slower", "shorter", "weaker", "impaired",
        "diminished", "dropped", "fell",
    }
)  # fmt: skip
_DIRECTION_STEMS: Final[frozenset[str]] = frozenset(_stem(word) for word in _UP_WORDS | _DOWN_WORDS)


def _surface_terms(text: str) -> set[str]:
    """Tokenised, stemmed terms plus general scientific concept aliases."""
    folded = _fold(text)
    terms: set[str] = set()
    for raw in _WORD.findall(folded):
        parts = [part for part in re.split(r"[.\-_]", raw) if part]
        for part in parts:
            if len(part) > 1 and part not in _QUESTION_WORDS:
                terms.add(_stem(part))
        if len(parts) > 1:
            terms.add(_stem("".join(parts)))
    for pattern, concept in _CONCEPT_ALIASES:
        if pattern.search(folded):
            terms.add(concept)
    return terms


def _weighted_facets(
    question: str,
    *,
    comparison: bool,
    entities: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    stems = scientific_terms(question)
    entity_stems = frozenset(
        part for entity in entities for part in (entity + tuple(_stem(part) for part in entity))
    )
    terms: dict[str, float] = {}
    for raw in stems:
        term = _stem(_fold(raw).replace(",", "."))
        if not term or term in _QUESTION_WORDS:
            continue
        if len(term) == 1 and not term.isdigit():
            continue
        if term in entity_stems:
            continue
        if term in _GENERIC_STEMS or term in _META_STEMS:
            weight = 0.25
        elif term.isdigit():
            weight = 1.5
        elif len(term) >= 8:
            weight = 1.25
        else:
            weight = 1.0
        terms[term] = max(terms.get(term, 0.0), weight)
    folded_question = _fold(question)
    for pattern, concept in _CONCEPT_ALIASES:
        if pattern.search(folded_question) and concept not in entity_stems:
            terms[concept] = max(terms.get(concept, 0.0), 1.25)
    if comparison:
        # Meta facets are resolved structurally, so they must not decide retrieval.
        terms = {term: weight for term, weight in terms.items() if term not in _META_STEMS}
    ordered = tuple(sorted(terms.items(), key=lambda item: (-item[1], item[0])))
    return tuple(term for term, _ in ordered), tuple(weight for _, weight in ordered)


def _entities(question: str) -> tuple[tuple[str, ...], ...]:
    """Designations and proper phrases that must appear in supporting evidence."""
    candidates: list[str] = []
    for match in _DESIGNATION.findall(question):
        candidates.append(match)
    for match in _PROPER_PHRASE.findall(question):
        candidates.append(match)
    starts = {m.group(1) for m in _SENTENCE_START.finditer(question)}
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", question):
        word = match.group(0)
        if word in starts or word in _QUESTION_WORDS:
            continue
        candidates.append(word)
    entities: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for phrase in candidates:
        folded = _fold(phrase)
        if _YEAR.match(folded):
            continue
        parts = tuple(
            part
            for raw in re.split(r"[\s\-\u2013]+", folded)
            for part in re.split(r"[.\-_]", raw)
            if len(part) > 1 and part not in _QUESTION_WORDS
        )
        if not parts or all(part.isdigit() and len(part) < 3 for part in parts):
            continue
        key = "".join(parts)
        if key in seen:
            continue
        seen.add(key)
        entities.append(parts)
    # "Kepler-22b" also yields the bare capitalized word "Kepler"; keep only the
    # most specific entity when one entity's parts are a subset of another's.
    return tuple(
        entity
        for entity in entities
        if not any(entity != other and set(entity) < set(other) for other in entities)
    )


def build_query_profile(question: str) -> QueryProfile:
    """Derive the entities and weighted facets a candidate must support."""
    cleaned = " ".join(question.split())
    if _INVENTORY.search(cleaned):
        kind: QueryKind = "inventory"
    elif _COMPARISON.search(cleaned):
        kind = "comparison"
    elif _METADATA.search(cleaned):
        kind = "metadata"
    else:
        kind = "factual"
    entities = _entities(cleaned)
    facets, weights = _weighted_facets(cleaned, comparison=kind == "comparison", entities=entities)
    meta = tuple(term for term in facets if term in _META_STEMS)
    return QueryProfile(
        question=cleaned[:2000],
        kind=kind,
        entities=entities,
        facets=facets,
        facet_weights=weights,
        meta_facets=meta,
    )


def _with_corpus_specificity(profile: QueryProfile, term_sets: Sequence[set[str]]) -> QueryProfile:
    """A query facet absent from every candidate is more likely unsupported.

    This is corpus-adaptive rather than a question blocklist: a term can only be
    penalised when none of the retrieved candidates contains it. Generic query
    words already carry a low base weight, so the penalty mainly exposes missing
    named qualifiers (for example a population or intervention the corpus does
    not mention).
    """
    if not term_sets:
        return profile
    adjusted: list[float] = []
    for term, weight in zip(profile.facets, profile.facet_weights, strict=True):
        folded = term.replace("_", "")
        present = any(
            term in terms or folded in {t.replace("_", "") for t in terms} for terms in term_sets
        )
        adjusted.append(weight * (2.0 if not present else 1.0))
    return replace(profile, facet_weights=tuple(adjusted))


def _candidate_terms(candidate: EvidenceCandidate) -> set[str]:
    return _surface_terms(
        " ".join(
            part
            for part in (
                candidate.title,
                candidate.section_title or "",
                candidate.text,
            )
            if part
        )
    )


def _entity_present(entity: tuple[str, ...], terms: set[str], joined: set[str]) -> bool:
    return all(part in terms or _stem(part) in terms for part in entity) or (
        "".join(entity) in joined
    )


def _assess_candidate(profile: QueryProfile, candidate: EvidenceCandidate) -> CandidateAssessment:
    terms = _candidate_terms(candidate)
    joined = {term.replace("_", "") for term in terms}
    missing_entities = tuple(
        " ".join(entity)
        for entity in profile.entities
        if not _entity_present(entity, terms, joined)
    )
    entity_support = (
        1.0
        if not profile.entities
        else (len(profile.entities) - len(missing_entities)) / len(profile.entities)
    )
    matched: list[str] = []
    missing: list[str] = []
    matched_weight = 0.0
    for term, weight in zip(profile.facets, profile.facet_weights, strict=True):
        if term in terms or term.replace("_", "") in joined:
            matched.append(term)
            matched_weight += weight
        else:
            missing.append(term)
    coverage = matched_weight / profile.total_weight if profile.total_weight else 0.0
    reasons: list[str] = []
    if not candidate.canonical_verified:
        reasons.append("source_not_canonical")
    if missing_entities:
        reasons.append("missing_entity")
    if profile.total_weight and coverage < profile.minimum_coverage:
        reasons.append("insufficient_query_facet_coverage")
    admissible = (
        candidate.canonical_verified
        and not missing_entities
        and bool(profile.total_weight)
        and coverage >= profile.minimum_coverage
    )
    return CandidateAssessment(
        candidate=candidate,
        lexical_similarity=coverage,
        query_relevance=coverage if not missing_entities else coverage * 0.5,
        entity_support=entity_support,
        matched_terms=tuple(matched[:MAX_REPORTED_TERMS]),
        missing_terms=tuple(missing[:MAX_REPORTED_TERMS]),
        missing_entities=missing_entities,
        admissible=admissible,
        reasons=tuple(reasons),
    )


def _documents(assessments: Iterable[CandidateAssessment]) -> int:
    return len(
        {
            a.candidate.document_version_key
            for a in assessments
            if a.admissible and a.candidate.document_version_key
        }
    )


def _decision(
    *,
    profile: QueryProfile,
    verdict: Verdict,
    code: str,
    detail: str,
    assessments: Sequence[CandidateAssessment],
    accepted: Sequence[CandidateAssessment] = (),
    contradictory: bool = False,
    ambiguous: bool = False,
    claim_support: ClaimSupport = "NOT_ASSESSED",
    comparison_document_count: int | None = None,
) -> AnswerabilityDecision:
    return AnswerabilityDecision(
        verdict=verdict,
        code=code,
        detail=detail[:400],
        gate_version=GATE_VERSION,
        question_kind=profile.kind,
        claim_support=claim_support,
        required_coverage=profile.minimum_coverage,
        comparison_document_count=(
            _documents(assessments)
            if comparison_document_count is None
            else comparison_document_count
        ),
        assessments=tuple(assessments),
        accepted_passage_keys=tuple(a.candidate.passage_key for a in accepted),
        rejected_passage_keys=tuple(
            a.candidate.passage_key for a in assessments if not a.admissible
        ),
        contradictory=contradictory,
        ambiguous=ambiguous,
    )


def assess_retrieval(  # noqa: PLR0911 - explicit fail-closed branches
    question: str,
    candidates: Sequence[EvidenceCandidate],
    *,
    authorized_document_count: int | None = None,
) -> AnswerabilityDecision:
    """Reject irrelevant retrieval candidates and decide whether to attempt synthesis."""
    profile = build_query_profile(question)
    canonical_terms = [
        _candidate_terms(candidate) for candidate in candidates if candidate.canonical_verified
    ]
    profile = _with_corpus_specificity(profile, canonical_terms)
    assessments = tuple(_assess_candidate(profile, candidate) for candidate in candidates)
    admissible = tuple(a for a in assessments if a.admissible)
    if not candidates:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="no_retrieval_candidates",
            detail="Retrieval returned no candidate passages.",
            assessments=assessments,
        )
    if profile.kind in {"inventory", "metadata"}:
        canonical = tuple(a for a in assessments if a.candidate.canonical_verified)
        if canonical:
            return _decision(
                profile=profile,
                verdict="ANSWERABLE",
                code="inventory_or_metadata_route",
                detail="A corpus inventory/metadata question does not synthesize a finding claim.",
                assessments=assessments,
                accepted=canonical,
            )
    if not profile.facets and not profile.entities:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="ambiguous_question",
            detail="The question has no checkable entities or content facets.",
            assessments=assessments,
        )
    if profile.kind == "comparison":
        canonical = tuple(
            a for a in assessments if a.candidate.canonical_verified and not a.missing_entities
        )
        if authorized_document_count is not None and authorized_document_count < 2:
            return _decision(
                profile=profile,
                verdict="ABSTAIN",
                code="insufficient_cross_study_evidence",
                detail="The authorized scope contains fewer than two documents.",
                assessments=assessments,
                accepted=admissible,
                comparison_document_count=authorized_document_count,
            )
        salient = tuple(
            term
            for term, weight in zip(profile.facets, profile.facet_weights, strict=True)
            if weight >= 1.0 and term not in _GENERIC_STEMS
        )
        if canonical and (not salient or any(a.admissible for a in canonical)):
            return _decision(
                profile=profile,
                verdict="ANSWERABLE",
                code="comparison_structurally_answerable",
                detail=(
                    "Canonical evidence is available for structural comparison; design, "
                    "population and outcome facets are extracted by the comparison tool."
                ),
                assessments=assessments,
                accepted=canonical,
            )
    if not admissible:
        missing_entity = any(a.missing_entities for a in assessments)
        noncanonical = not any(a.candidate.canonical_verified for a in assessments)
        if missing_entity:
            code, detail = (
                "unsupported_entity",
                "Retrieved passages do not contain the question's named entities.",
            )
        elif noncanonical:
            code, detail = (
                "source_not_canonical",
                "No retrieved passage passed canonical source verification.",
            )
        else:
            code, detail = (
                "insufficient_query_coverage",
                "Retrieved passages do not cover enough of the question's facets.",
            )
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code=code,
            detail=detail,
            assessments=assessments,
        )
    # Near-ties between documents with different facets are ambiguous, not a
    # licence to let the top RRF rank silently choose one.
    top = sorted(admissible, key=lambda a: (-a.query_relevance, a.candidate.rank))
    if (
        profile.kind == "factual"
        and bool(_SINGLE_VALUE.search(profile.question))
        and len(top) > 1
        and top[0].candidate.document_version_key != top[1].candidate.document_version_key
        and abs(top[0].query_relevance - top[1].query_relevance) <= AMBIGUITY_MARGIN
        and set(top[0].matched_terms) != set(top[1].matched_terms)
    ):
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="ambiguous_evidence",
            detail="Top admissible passages support different facets at near-equal relevance.",
            assessments=assessments,
            accepted=top[:1],
            ambiguous=True,
        )
    return _decision(
        profile=profile,
        verdict="ANSWERABLE",
        code="evidence_relevant_and_answerable",
        detail=(
            "Admissible evidence covers the question's entities and facets; this is a "
            "lexical/structural gate, not semantic entailment."
        ),
        assessments=assessments,
        accepted=admissible,
    )


def _direction_terms(text: str) -> tuple[int, ...]:
    folded = _fold(text)
    sequence: list[int] = []
    for word in re.findall(r"[a-z]+", folded):
        if word in _UP_WORDS:
            sequence.append(1)
        elif word in _DOWN_WORDS:
            sequence.append(-1)
    return tuple(sequence)


@dataclass(frozen=True)
class ClaimEvidence:
    """One verified citation, reduced to the fields this gate needs."""

    passage_key: str
    document_version_key: str
    document_title: str
    claim_text: str
    verbatim_quote: str


def _claim_candidate(claim: ClaimEvidence) -> EvidenceCandidate:
    return EvidenceCandidate(
        passage_key=claim.passage_key,
        document_version_key=claim.document_version_key,
        title=claim.document_title,
        text=f"{claim.claim_text}\n{claim.verbatim_quote}",
    )


def assess_claims(  # noqa: PLR0911 - explicit fail-closed branches
    question: str,
    claims: Sequence[ClaimEvidence],
    *,
    comparison_document_count: int = 0,
) -> AnswerabilityDecision:
    """Final gate before synthesis/finalisation, after canonical claim verification.

    It answers only whether the *verified quotations* address the asked-about
    entities/facets. It cannot and does not establish entailment.
    """
    profile = build_query_profile(question)
    if not claims:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="no_verified_claims",
            detail="No claim survived canonical quotation verification.",
            assessments=(),
        )
    claim_candidates = tuple(_claim_candidate(c) for c in claims)
    profile = _with_corpus_specificity(
        profile, [_candidate_terms(candidate) for candidate in claim_candidates]
    )
    assessments = tuple(_assess_candidate(profile, candidate) for candidate in claim_candidates)
    admissible = tuple(a for a in assessments if a.admissible)
    if profile.kind in {"inventory", "metadata"}:
        canonical = tuple(a for a in assessments if a.candidate.canonical_verified)
        if canonical:
            return _decision(
                profile=profile,
                verdict="ANSWERABLE",
                code="inventory_or_metadata_route",
                detail="A corpus inventory/metadata question does not synthesize a finding claim.",
                assessments=assessments,
                accepted=canonical,
            )
    if profile.kind == "comparison":
        if comparison_document_count < 2:
            return _decision(
                profile=profile,
                verdict="ABSTAIN",
                code="insufficient_cross_study_evidence",
                detail="Fewer than two studies were available for the requested comparison.",
                assessments=assessments,
                accepted=admissible,
                comparison_document_count=comparison_document_count,
            )
        # The comparison tool extracts design/population/result facets structurally
        # from complete documents; a passage-level claim is not its only evidence.
        return _decision(
            profile=profile,
            verdict="ANSWERABLE",
            code="comparison_structurally_answerable",
            detail="At least two admissible studies were compared structurally.",
            assessments=assessments,
            accepted=admissible,
            comparison_document_count=comparison_document_count,
        )
    if not admissible:
        missing_entity = any(a.missing_entities for a in assessments)
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="claim_not_answerable" if not missing_entity else "unsupported_entity",
            detail=(
                "Verified quotations do not contain the question's named entities."
                if missing_entity
                else "Verified quotations cover too little of the question's facets."
            ),
            assessments=assessments,
        )
    # Contradiction is measured on direction words over shared outcome facets.
    # It is deliberately conservative: only opposite directions on the same
    # outcome in different documents are reported.
    seen: dict[tuple[str, ...], list[tuple[str, tuple[int, ...]]]] = {}
    for assessment, claim in zip(assessments, claims, strict=True):
        if not assessment.admissible:
            continue
        outcome = tuple(
            sorted(set(assessment.matched_terms) - set(profile.entity_stems) - _DIRECTION_STEMS)
        )
        direction = _direction_terms(claim.claim_text)
        if not outcome or not direction:
            continue
        seen.setdefault(outcome, []).append((claim.document_version_key, direction))
    contradictory = any(
        len({doc for doc, _ in entries}) > 1
        and any(
            (a[1][0] > 0) != (b[1][0] > 0) for i, a in enumerate(entries) for b in entries[i + 1 :]
        )
        for entries in seen.values()
    )
    if contradictory:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="contradictory_evidence",
            detail=(
                "Admissible studies report opposite directions on the same outcome; the "
                "corpus does not support a single definitive proposition."
            ),
            assessments=assessments,
            accepted=admissible,
            contradictory=True,
        )
    return _decision(
        profile=profile,
        verdict="ANSWERABLE",
        code="claims_relevant_and_answerable",
        detail=(
            "Verified quotations cover the question's entities and facets; the gate is "
            "lexical/structural, not entailment."
        ),
        assessments=assessments,
        accepted=admissible,
        contradictory=contradictory,
        claim_support="SUPPORTED_BY_STRUCTURED_EVIDENCE_GATE",
    )
