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
    "CALIBRATED_MIN_COMPARISON_COVERAGE",
    "CALIBRATED_MIN_RELEVANCE",
    "GATE_VERSION",
    "AnswerabilityDecision",
    "CandidateAssessment",
    "EvidenceCandidate",
    "QueryKind",
    "QueryProfile",
    "acronym_expansions",
    "assess_claims",
    "assess_retrieval",
    "build_query_profile",
    "salient_terms",
    "stem_terms",
    "surface_terms",
]

GATE_VERSION: Final = "structured-evidence-gate-v1"

# Calibrated on tests/fixtures/answerability/calibration.json; the calibration test
# recomputes the separating threshold and fails if this constant drifts.
CALIBRATED_MIN_RELEVANCE: Final[float] = 0.60
CALIBRATED_MIN_COMPARISON_COVERAGE: Final[float] = 0.30
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
        "behavior",
        "behaviors",
        "behaviour",
        "behaviours",
        "both",
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
        "find",
        "finding",
        "findings",
        "found",
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
        "participant",
        "participants",
        "please",
        "player",
        "players",
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
        "subject",
        "subjects",
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
        "three",
        "two",
        "versus",
        "work",
        # Literature-question framing ("what does the literature establish ... evaluate ...").
        "assess",
        "establish",
        "established",
        "evaluate",
        "literature",
        "scientific",
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

# Inventory/metadata questions are answered from document identity and section
# structure rather than a finding sentence, so these facets are deliberately not
# required to appear in a passage. Named entities and topical qualifiers still
# are: "DOI of the Kepler-22b paper" must not pass merely because the corpus has
# some paper with a DOI.
_METADATA_ROUTE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "author",
        "authors",
        "doi",
        "identifier",
        "identifiers",
        "journal",
        "metadata",
        "pmcid",
        "pmid",
        "section",
        "sections",
        "title",
        "titles",
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
        "using",
        "via",
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
        # Measurement-science wording for the instrument a device is validated against.
        re.compile(
            r"\b(?:reference\s+(?:[a-z]+\s+)?(?:standards?|systems?|methods?|devices?|"
            r"instruments?|measures?)|criterion(?:\s+(?:measures?|methods?|devices?|"
            r"standards?|systems?))?|gold[\s-]+standards?|concurrent\s+validity)\b",
            re.IGNORECASE,
        ),
        "reference_standard",
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

_SUBSUMING_CONCEPTS: Final[frozenset[str]] = frozenset({"reference_standard"})

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
    r"\b(?:how\s+(?:many|much|long|far|fast|often|large|small|heavy|tall|old)|"
    r"what(?:'s|\u2019s| is| was| were)\s+"
    r"(?:(?:the|its|their|this|that)\s+)?"
    r"(?:value|number|rate|period|duration|interval|angle|height|force|stiffness|"
    r"percentage|percent|ratio|frequency|volume|load|distance|time|size|length|"
    r"mass|weight|age|count|proportion|sample\s+size|sample\s+count))\b",
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
    # Enumerated aspects ("validity, reliability, applications and limitations"): a passage needs
    # the core topic plus ANY one aspect; which aspects have evidence is reported, not assumed.
    aspects: tuple[str, ...] = ()
    # Acronym facet -> stems of its expansion ("lpt" -> linear, positi, transd), both directions.
    expansions: tuple[tuple[str, tuple[str, ...]], ...] = ()

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
    accepted_claim_indexes: tuple[int, ...] = ()
    contradictory: bool = False
    ambiguous: bool = False
    aspect_coverage: tuple[tuple[str, int], ...] = ()

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
            "accepted_claim_indexes": list(self.accepted_claim_indexes),
            "contradictory": self.contradictory,
            "ambiguous": self.ambiguous,
            "aspect_coverage": dict(self.aspect_coverage),
            "uncovered_aspects": [a for a, n in self.aspect_coverage if n == 0],
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


def _stem(token: str) -> str:  # noqa: PLR0911 - suffix ladder is explicit
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
_METADATA_ROUTE_STEMS: Final[frozenset[str]] = frozenset(
    _stem(term) for term in _METADATA_ROUTE_TERMS
)
# Measurement nouns carry content for retrieval ("sample size" must reach the
# methods passage) but do not identify *which* study is meant. They are used
# only by the ambiguity check, not to down-weight the query.
_GENERIC_MEASUREMENT_TERMS: Final[frozenset[str]] = frozenset({"number", "sample", "size", "value"})
_GENERIC_MEASUREMENT_STEMS: Final[frozenset[str]] = frozenset(
    _stem(term) for term in _GENERIC_MEASUREMENT_TERMS
)
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
# Relationship verbs are not directions. A question asking whether something was
# "affected" is satisfied by a result sentence reporting "greater"/"increased",
# but those words must not be treated as an up/down contradiction signal.
_EFFECT_WORDS: Final[frozenset[str]] = frozenset(
    {
        "affect",
        "affected",
        "affects",
        "association",
        "associated",
        "change",
        "changed",
        "changes",
        "correlate",
        "correlated",
        "difference",
        "differences",
        "effect",
        "effects",
        "impact",
        "impacts",
        "influence",
        "influenced",
        "influences",
        "relate",
        "related",
    }
)
_EFFECT_STEMS: Final[frozenset[str]] = frozenset(_stem(word) for word in _EFFECT_WORDS)


def stem_terms(terms: Iterable[str]) -> set[str]:
    """Canonicalise retrieval terms with the gate's deterministic stemmer.

    Exposed so lexical scoring and the answerability gate agree that
    "completed" and "complete", or "weeks" and "week", are the same facet.
    """
    return {term if "_" in term else _stem(_fold(term).replace(",", ".")) for term in terms if term}


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


def surface_terms(text: str) -> set[str]:
    """Public evidence-side term set: aliases plus deterministic stems."""
    return _surface_terms(text)


def _weighted_facets(  # noqa: PLR0912 - explicit facet weighting branches
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
        if not term or term in _QUESTION_WORDS or _fold(raw) in _QUESTION_WORDS:
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
        match = pattern.search(folded_question)
        if match is not None and concept in _SUBSUMING_CONCEPTS:
            # The concept replaces its own wording: "reference measurement systems" is met by a
            # "criterion" or "gold standard" passage, not by the words "reference" and "system".
            for word in _WORD.findall(match.group(0)):
                terms.pop(_stem(word), None)
        if match is not None and concept not in entity_stems:
            # A generic population noun ("participants", "subjects") is a
            # reference, not a study qualifier. The qualifier words ("female",
            # "volleyball", "cricketers") still carry full weight.
            weight = 0.5 if concept == "population_member" else 1.25
            terms[concept] = max(terms.get(concept, 0.0), weight)
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


_ENUM_WORD: Final = r"[a-z][a-z\-]*"
_ENUMERATION: Final = re.compile(
    rf"((?:{_ENUM_WORD}\s+){{0,2}}{_ENUM_WORD}(?:\s*,\s*(?:{_ENUM_WORD}\s+){{0,2}}{_ENUM_WORD})+"
    rf"\s*,?\s+(?:and|or)\s+(?:{_ENUM_WORD}\s+){{0,2}}{_ENUM_WORD})\s*$"
)
_ACRONYM_DEFINITION: Final = re.compile(
    r"((?:[A-Za-z][\w-]*\s+){1,5}?[A-Za-z][\w-]*)\s*\(\s*([A-Z][A-Z0-9]{1,7})s?\s*\)"
)
_ACRONYM_TOKEN: Final = re.compile(r"\b([A-Z]{2,6})s?\b")


def _enumerated_aspects(question: str) -> tuple[str, ...]:
    """Heads of a 3+ item enumeration clause, e.g. "validity, reliability, ... and limitations"."""
    folded = _fold(question)
    for clause in re.split(r"[.?!;:]", folded):
        match = _ENUMERATION.search(clause.strip())
        if match is None:
            continue
        items = [i for i in re.split(r"\s*,\s*|\s+(?:and|or)\s+", match.group(1)) if i.strip()]
        heads = tuple(
            dict.fromkeys(
                _stem(item.split()[-1])
                for item in items
                if item.split() and item.split()[-1] not in _QUESTION_WORDS
            )
        )
        if len(heads) >= 3:
            return heads
    return ()


def _initials(words: Sequence[str]) -> str:
    return "".join(word[0] for word in words).upper()


def acronym_expansions(question: str, texts: Iterable[str] = ()) -> dict[str, tuple[str, ...]]:
    """Acronym stem -> expansion stems, from explicit "phrase (ACR)" definitions in the question,
    prior turns or evidence, or from an exact-initials match against earlier words.

    Only an exact initial-letter match is accepted ("LPT" = "linear position transducer").
    """
    found: dict[str, tuple[str, ...]] = {}
    sources = [question, *texts]
    for source in sources:
        for match in _ACRONYM_DEFINITION.finditer(source):
            acronym = match.group(2)
            words = [w for w in re.split(r"[\s\-]+", match.group(1)) if w][-len(acronym) :]
            if len(words) == len(acronym) and _initials(words) == acronym:
                found.setdefault(_stem(_fold(acronym)), tuple(_stem(_fold(w)) for w in words))
    joined = " ".join(sources)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", joined)
    for match in _ACRONYM_TOKEN.finditer(question):
        acronym = match.group(1)
        key = _stem(_fold(acronym))
        if key in found:
            continue
        for start in range(len(words) - len(acronym) + 1):
            window = words[start : start + len(acronym)]
            if any(w.isupper() or w.lower() in _QUESTION_WORDS for w in window):
                continue
            if _initials(window) == acronym:
                found[key] = tuple(_stem(_fold(w)) for w in window)
                break
    return found


def _with_expansions(profile: QueryProfile, texts: Iterable[str]) -> QueryProfile:
    expansions = acronym_expansions(profile.question, texts)
    if not expansions:
        return profile
    return replace(profile, expansions=tuple(sorted(expansions.items())))


def _facet_present(profile: QueryProfile, term: str, terms: set[str], joined: set[str]) -> bool:
    """Facet match, where an acronym and its expansion satisfy each other."""
    if _term_present(term, terms, joined):
        return True
    for acronym, stems in profile.expansions:
        if term == acronym and stems and all(stem in terms for stem in stems):
            return True
        if term in stems and acronym in terms:
            return True
    return False


def build_query_profile(question: str, context: str = "") -> QueryProfile:
    """Derive the entities and weighted facets a candidate must support.

    ``context`` is prior user text for multi-turn references. It is used only
    when the current question has no checkable content of its own (for example
    "What about it?"): the current turn remains the required proposition, so a
    prior entity cannot silently become a mandatory entity for a follow-up that
    asks about a different facet.
    """
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
    if not facets and not entities and context.strip():
        return build_query_profile(context)
    meta = tuple(term for term in facets if term in _META_STEMS)
    aspects = (
        tuple(a for a in _enumerated_aspects(cleaned) if a in facets) if kind == "factual" else ()
    )
    profile = QueryProfile(
        question=cleaned[:2000],
        kind=kind,
        entities=entities,
        facets=facets,
        facet_weights=weights,
        meta_facets=meta,
        aspects=aspects if len(aspects) >= 3 else (),
    )
    return _with_expansions(profile, [context] if context.strip() else [])


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
            term in terms
            or folded in {t.replace("_", "") for t in terms}
            or _facet_present(profile, term, terms, {t.replace("_", "") for t in terms})
            for terms in term_sets
        )
        adjusted.append(weight * (2.0 if not present else 1.0))
    return replace(profile, facet_weights=tuple(adjusted))


def _term_present(term: str, terms: set[str], joined: set[str]) -> bool:
    if term in terms or term.replace("_", "") in joined:
        return True
    if "_" in term:
        parts = [part for part in term.split("_") if part not in {"of", "the"}]
        if parts and all(part in terms or part.replace("_", "") in joined for part in parts):
            return True
    return False


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
    aspect_weights: list[float] = []
    matched_aspect = 0.0
    for term, weight in zip(profile.facets, profile.facet_weights, strict=True):
        present = _facet_present(profile, term, terms, joined)
        (matched if present else missing).append(term)
        if term in profile.aspects:
            aspect_weights.append(weight)
            matched_aspect = max(matched_aspect, weight if present else 0.0)
        elif present:
            matched_weight += weight
    if aspect_weights:
        # Core topic plus the best single aspect: a reliability passage need not also report
        # applications. Missing aspects are reported by the decision, never silently assumed.
        unit = max(aspect_weights)
        core_total = profile.total_weight - sum(aspect_weights)
        coverage = (matched_weight + min(matched_aspect, unit)) / (core_total + unit)
    else:
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


def _candidate_has_all(candidate: EvidenceCandidate, terms: Sequence[str]) -> bool:
    surface = _candidate_terms(candidate)
    joined = {term.replace("_", "") for term in surface}
    return all(_term_present(term, surface, joined) for term in terms)


def _missing_from_every_candidate(
    profile: QueryProfile, term_sets: Sequence[set[str]]
) -> tuple[str, ...]:
    """Salient query terms absent from every canonical candidate.

    Weighted coverage can still pass at 0.60 when one high-weight qualifier (for
    example "female" or "patellar") is missing everywhere. That is a topic
    change, not partial relevance, so it is a hard abstention.
    """
    missing: list[str] = []
    for term in _salient_terms(profile):
        if term in profile.aspects:
            continue  # an aspect without evidence is a reported gap, not a topic change
        present = False
        for terms in term_sets:
            joined = {candidate_term.replace("_", "") for candidate_term in terms}
            if _facet_present(profile, term, terms, joined):
                present = True
                break
            if term in _DIRECTION_STEMS and terms & _DIRECTION_STEMS:
                present = True
                break
            if term in _EFFECT_STEMS and terms & (_EFFECT_STEMS | _DIRECTION_STEMS):
                present = True
                break
        if not present:
            missing.append(term)
    return tuple(missing)


def _structural_topic_terms(profile: QueryProfile) -> tuple[str, ...]:
    """Topical qualifiers that structure/metadata evidence must still support.

    Metadata terms such as ``doi`` or ``section title`` live in document identity,
    not necessarily in passage text, so they are excluded here. Generic query
    words carry no topical content. Everything else is a real qualifier.
    """
    return tuple(
        term
        for term, weight in zip(profile.facets, profile.facet_weights, strict=True)
        if weight >= 1.0 and term not in _GENERIC_STEMS and term not in _METADATA_ROUTE_STEMS
    )


def _structural_route_indexes(
    profile: QueryProfile, assessments: Sequence[CandidateAssessment]
) -> tuple[tuple[int, ...], str | None]:
    """Validate inventory/metadata evidence against entities and topical qualifiers.

    Returns the accepted indexes plus an abstention code, or ``None`` when the
    structural route is supported. This is a *relevance* check over canonical
    candidates, not entailment: the metadata tool still reads identity fields.
    """
    canonical = tuple(
        index
        for index, assessment in enumerate(assessments)
        if assessment.candidate.canonical_verified
    )
    if not canonical:
        return (), "source_not_canonical"
    supported = tuple(index for index in canonical if not assessments[index].missing_entities)
    if profile.entities and not supported:
        return (), "unsupported_entity"
    topics = _structural_topic_terms(profile)
    if topics:
        topical = tuple(
            index for index in supported if _candidate_has_all(assessments[index].candidate, topics)
        )
        if not topical:
            return (), "insufficient_query_coverage"
        return topical, None
    return supported, None


def _structural_route_decision(
    *,
    profile: QueryProfile,
    assessments: Sequence[CandidateAssessment],
    claims: bool = False,
) -> AnswerabilityDecision | None:
    indexes, code = _structural_route_indexes(profile, assessments)
    if code is not None:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code=code,
            detail=(
                "Inventory/metadata evidence does not support the question's named "
                "entities or topical qualifier."
            ),
            assessments=assessments,
        )
    accepted = tuple(assessments[index] for index in indexes)
    return _decision(
        profile=profile,
        verdict="ANSWERABLE",
        code="inventory_or_metadata_route",
        detail=(
            "A corpus inventory/metadata question does not synthesize a finding claim; "
            "accepted evidence still covers its named entities and topical qualifier."
        ),
        assessments=assessments,
        accepted=accepted,
        accepted_claim_indexes=indexes if claims else (),
    )


_STUDY_NOUNS: Final[frozenset[str]] = frozenset(
    {"article", "articles", "paper", "papers", "study", "studies", "trial", "trials"}
)
_STUDY_QUALIFIER_STOP: Final[frozenset[str]] = _QUESTION_WORDS | {
    "available",
    "both",
    "did",
    "does",
    "earlier",
    "find",
    "finding",
    "findings",
    "previous",
    "prior",
    "report",
    "reported",
    "same",
    "three",
    "two",
}


def _study_qualifier_terms(question: str) -> tuple[str, ...]:
    """Terms immediately before "study/paper/article/trial", as query qualifiers.

    "What did the depth jumping study find?" yields ("depth", "jump"). These
    terms identify the study; they are not claim content. A claim may use the
    document title to prove the study identity, but not to supply the finding.
    """
    tokens = re.findall(r"[a-z0-9]+", _fold(question))
    for index, token in enumerate(tokens):
        if token not in _STUDY_NOUNS:
            continue
        prior = tokens[max(0, index - 5) : index]
        terms: list[str] = []
        for word in reversed(prior):
            if word in _STUDY_QUALIFIER_STOP or len(word) <= 1:
                break
            term = _stem(word)
            if term not in terms:
                terms.insert(0, term)
            if len(terms) >= 3:
                break
        return tuple(terms)
    return ()


def _title_supports_qualifier(title: str, qualifier: Sequence[str]) -> bool:
    terms = _surface_terms(title)
    joined = {term.replace("_", "") for term in terms}
    return all(term in terms or term.replace("_", "") in joined for term in qualifier)


def _drop_qualifier_facets(profile: QueryProfile, qualifier: Sequence[str]) -> QueryProfile:
    joined = "".join(qualifier)
    kept = [
        (term, weight)
        for term, weight in zip(profile.facets, profile.facet_weights, strict=True)
        if not (term.replace("_", "") and term.replace("_", "") in joined)
    ]
    return replace(
        profile,
        facets=tuple(term for term, _ in kept),
        facet_weights=tuple(weight for _, weight in kept),
    )


def _salient_terms(profile: QueryProfile) -> tuple[str, ...]:
    return tuple(
        term
        for term, weight in zip(profile.facets, profile.facet_weights, strict=True)
        if weight >= 1.0 and term not in _GENERIC_STEMS
    )


def salient_terms(profile: QueryProfile) -> tuple[str, ...]:
    """Public view of the topical terms a question actually asks about."""
    return _salient_terms(profile)


def _documents(assessments: Iterable[CandidateAssessment]) -> int:
    """Count distinct documents in the supplied set, independent of lexical coverage."""
    return len(
        {a.candidate.document_version_key for a in assessments if a.candidate.document_version_key}
    )


def _decision(
    *,
    profile: QueryProfile,
    verdict: Verdict,
    code: str,
    detail: str,
    assessments: Sequence[CandidateAssessment],
    accepted: Sequence[CandidateAssessment] = (),
    accepted_claim_indexes: tuple[int, ...] = (),
    contradictory: bool = False,
    ambiguous: bool = False,
    claim_support: ClaimSupport = "NOT_ASSESSED",
    comparison_document_count: int | None = None,
) -> AnswerabilityDecision:
    coverage: list[tuple[str, int]] = []
    for aspect in profile.aspects:
        count = 0
        for assessment in accepted:
            terms = _candidate_terms(assessment.candidate)
            if _facet_present(profile, aspect, terms, {t.replace("_", "") for t in terms}):
                count += 1
        coverage.append((aspect, count))
    return AnswerabilityDecision(
        aspect_coverage=tuple(coverage),
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
        accepted_claim_indexes=accepted_claim_indexes,
        contradictory=contradictory,
        ambiguous=ambiguous,
    )


def assess_retrieval(  # noqa: PLR0911, PLR0912 - explicit fail-closed branches
    question: str,
    candidates: Sequence[EvidenceCandidate],
    *,
    context: str = "",
    authorized_document_count: int | None = None,
) -> AnswerabilityDecision:
    """Reject irrelevant retrieval candidates and decide whether to attempt synthesis."""
    profile = build_query_profile(question, context)
    profile = _with_expansions(profile, [context, *(f"{c.title} {c.text}" for c in candidates)])
    canonical_terms = [
        _candidate_terms(candidate) for candidate in candidates if candidate.canonical_verified
    ]
    profile = _with_corpus_specificity(profile, canonical_terms)
    assessments = tuple(_assess_candidate(profile, candidate) for candidate in candidates)
    admissible = tuple(a for a in assessments if a.admissible)
    if (
        profile.kind in {"factual", "comparison"}
        and canonical_terms
        and not any(assessment.missing_entities for assessment in assessments)
    ):
        missing = _missing_from_every_candidate(profile, canonical_terms)
        if missing:
            return _decision(
                profile=profile,
                verdict="ABSTAIN",
                code="insufficient_query_coverage",
                detail=(
                    "A salient query qualifier is absent from every canonical candidate: "
                    + ", ".join(missing[:MAX_REPORTED_TERMS])
                ),
                assessments=assessments,
            )
    if not candidates:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="no_retrieval_candidates",
            detail="Retrieval returned no candidate passages.",
            assessments=assessments,
        )
    if profile.kind in {"inventory", "metadata"}:
        structural = _structural_route_decision(profile=profile, assessments=assessments)
        if structural is not None:
            return structural
    if not profile.facets and not profile.entities:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="ambiguous_question",
            detail="The question has no checkable entities or content facets.",
            assessments=assessments,
        )
    if (
        profile.kind == "factual"
        and not profile.entities
        and not (set(_salient_terms(profile)) - _GENERIC_MEASUREMENT_STEMS)
        and len(
            {
                a.candidate.document_version_key
                for a in assessments
                if a.candidate.canonical_verified
            }
        )
        > 1
    ):
        # "What is the sample size?" has no study reference. A scoped follow-up
        # supplies one through its single active document; an unscoped question
        # that spans several studies does not, and choosing the top-ranked one
        # would silently change the subject.
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="ambiguous_question",
            detail="The question does not identify which study it asks about.",
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
        salient = _salient_terms(profile)
        if canonical and not salient:
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
        if canonical and any(a.admissible for a in canonical):
            # When the question names salient facets, an irrelevant document is
            # not made comparable merely by existing in the authorized scope.
            return _decision(
                profile=profile,
                verdict="ANSWERABLE",
                code="comparison_structurally_answerable",
                detail=(
                    "At least one canonical document covers the named comparison facets; "
                    "the comparison tool extracts the structural design/population facets."
                ),
                assessments=assessments,
                accepted=tuple(a for a in canonical if a.admissible),
            )
    if not admissible:
        missing_entity = all(a.missing_entities for a in assessments)
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
    # The title identifies the study and is allowed to satisfy study-context
    # facets. It is not allowed to supply the finding: _claim_has_finding_signal
    # separately requires entity, direction/effect, value or non-title evidence
    # in the quotation itself.
    return EvidenceCandidate(
        passage_key=claim.passage_key,
        document_version_key=claim.document_version_key,
        title=claim.document_title,
        text=claim.verbatim_quote,
    )


def _claim_has_finding_signal(
    assessment: CandidateAssessment, claim: ClaimEvidence, profile: QueryProfile
) -> bool:
    """A quotation must do more than repeat its own document title.

    Entity support is necessary but not sufficient: the title can name the
    entity while the quotation reports no finding. The quotation must contain a
    direction/effect word, a value answer, or an asked-about facet absent from
    the title.
    """
    if profile.aspects:
        # Enumerated multi-aspect question: the quotation itself must name an asked-about
        # aspect or report a value/direction; the title may not supply the finding.
        quote = _surface_terms(claim.verbatim_quote)
        quote_joined = {term.replace("_", "") for term in quote}
        return (
            any(_facet_present(profile, a, quote, quote_joined) for a in profile.aspects)
            or bool(_direction_terms(claim.verbatim_quote))
            or bool(re.search(r"\d", claim.verbatim_quote))
        )
    missing_salient = (
        (set(assessment.missing_terms) & set(_salient_terms(profile)))
        - _DIRECTION_STEMS
        - _EFFECT_STEMS
    )
    if missing_salient:
        return False
    if _direction_terms(claim.verbatim_quote) or (
        _surface_terms(claim.verbatim_quote) & _EFFECT_STEMS
    ):
        return True
    if _SINGLE_VALUE.search(profile.question) and re.search(r"\d", claim.verbatim_quote):
        return True
    quote_terms = _surface_terms(claim.verbatim_quote)
    title_terms = _surface_terms(claim.document_title)
    quote_joined = {term.replace("_", "") for term in quote_terms}
    title_joined = {term.replace("_", "") for term in title_terms}
    return any(
        _term_present(term, quote_terms, quote_joined)
        and not _term_present(term, title_terms, title_joined)
        for term in profile.facets
    )


def _aspect_claim_assessment(
    profile: QueryProfile, assessment: CandidateAssessment
) -> CandidateAssessment:
    terms = _candidate_terms(assessment.candidate)
    joined = {term.replace("_", "") for term in terms}
    aspects = [a for a in profile.aspects if _facet_present(profile, a, terms, joined)]
    core = [
        (term, weight)
        for term, weight in zip(profile.facets, profile.facet_weights, strict=True)
        if term not in profile.aspects
    ]
    total = sum(weight for _, weight in core)
    matched = sum(weight for term, weight in core if _facet_present(profile, term, terms, joined))
    core_coverage = matched / total if total else 0.0
    admissible = (
        assessment.candidate.canonical_verified
        and not assessment.missing_entities
        and bool(aspects)
        and core_coverage >= CALIBRATED_MIN_COMPARISON_COVERAGE
    )
    reasons = tuple(r for r in assessment.reasons if r != "insufficient_query_facet_coverage")
    if not aspects:
        reasons = (*reasons, "no_asked_aspect_in_quotation")
    if core_coverage < CALIBRATED_MIN_COMPARISON_COVERAGE:
        reasons = (*reasons, "insufficient_core_topic_coverage")
    return replace(assessment, admissible=admissible, reasons=reasons)


def assess_claims(  # noqa: PLR0911, PLR0912 - explicit fail-closed branches
    question: str,
    claims: Sequence[ClaimEvidence],
    *,
    context: str = "",
) -> AnswerabilityDecision:
    """Final gate before synthesis/finalisation, after canonical claim verification.

    It answers only whether the *verified quotations* address the asked-about
    entities/facets. It cannot and does not establish entailment.
    """
    profile = build_query_profile(question, context)
    if not claims:
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="no_verified_claims",
            detail="No claim survived canonical quotation verification.",
            assessments=(),
        )
    qualifier = _study_qualifier_terms(profile.question) if profile.kind == "factual" else ()
    if qualifier:
        entity_stems = {_stem(part) for entity in profile.entities for part in entity}
        if not any(term in entity_stems for term in qualifier):
            if not all(
                _title_supports_qualifier(claim.document_title, qualifier) for claim in claims
            ):
                return _decision(
                    profile=profile,
                    verdict="ABSTAIN",
                    code="insufficient_query_coverage",
                    detail=(
                        "The named study qualifier is absent from the verified document "
                        "title: " + ", ".join(qualifier[:MAX_REPORTED_TERMS])
                    ),
                    assessments=(),
                )
            profile = _drop_qualifier_facets(profile, qualifier)
    claim_candidates = tuple(_claim_candidate(c) for c in claims)
    profile = _with_expansions(
        profile, [context, *(f"{c.document_title} {c.verbatim_quote}" for c in claims)]
    )
    profile = _with_corpus_specificity(
        profile, [_candidate_terms(candidate) for candidate in claim_candidates]
    )
    assessments = tuple(_assess_candidate(profile, candidate) for candidate in claim_candidates)
    if profile.aspects:
        # A broad multi-aspect question: the caller admits only quotations from passages that
        # passed the retrieval gate (core topic + an aspect). One quotation is one finding, so it
        # must address an asked-about aspect and keep part of the core topic; it need not restate
        # the whole question. Which aspects stay without a quotation is reported.
        assessments = tuple(
            _aspect_claim_assessment(profile, assessment) for assessment in assessments
        )
    if profile.kind == "factual":
        assessments = tuple(
            assessment
            if _claim_has_finding_signal(assessment, claim, profile)
            else replace(
                assessment,
                admissible=False,
                reasons=(*assessment.reasons, "title_only_relevance"),
            )
            for assessment, claim in zip(assessments, claims, strict=True)
        )
    admissible = tuple(a for a in assessments if a.admissible)
    admissible_indexes = tuple(
        index for index, assessment in enumerate(assessments) if assessment.admissible
    )
    if (
        admissible
        and profile.kind in {"factual", "comparison"}
        and not profile.aspects
        and not any(assessment.missing_entities for assessment in assessments)
    ):
        missing = _missing_from_every_candidate(
            profile, [_candidate_terms(candidate) for candidate in claim_candidates]
        )
        if missing:
            return _decision(
                profile=profile,
                verdict="ABSTAIN",
                code="insufficient_query_coverage",
                detail=(
                    "A salient query qualifier is absent from every verified quotation: "
                    + ", ".join(missing[:MAX_REPORTED_TERMS])
                ),
                assessments=assessments,
            )
    if profile.kind in {"inventory", "metadata"}:
        structural = _structural_route_decision(
            profile=profile, assessments=assessments, claims=True
        )
        if structural is not None:
            return structural
    if (
        profile.kind == "factual"
        and not profile.entities
        and not (set(_salient_terms(profile)) - _GENERIC_MEASUREMENT_STEMS)
        and _documents(assessments) > 1
    ):
        return _decision(
            profile=profile,
            verdict="ABSTAIN",
            code="ambiguous_question",
            detail="The verified quotations span several studies without a study reference.",
            assessments=assessments,
        )
    if profile.kind == "factual" and _SINGLE_VALUE.search(profile.question):
        numeric = tuple(
            index
            for index in admissible_indexes
            if re.search(r"\d", assessments[index].candidate.text)
        )
        if not numeric:
            return _decision(
                profile=profile,
                verdict="ABSTAIN",
                code="insufficient_numeric_evidence",
                detail=("The question asks for a value, but no admissible quotation contains one."),
                assessments=assessments,
            )
    if profile.kind == "comparison":
        salient = _salient_terms(profile)
        comparable = (
            admissible
            if salient
            else tuple(a for a in assessments if a.candidate.canonical_verified)
        )
        comparable_indexes = (
            admissible_indexes
            if salient
            else tuple(
                index
                for index, assessment in enumerate(assessments)
                if assessment.candidate.canonical_verified
            )
        )
        admissible_documents = _documents(comparable)
        if admissible_documents < 2:
            return _decision(
                profile=profile,
                verdict="ABSTAIN",
                code="insufficient_cross_study_evidence",
                detail=(
                    "Fewer than two documents have verified quotations that answer the "
                    "requested comparison facets."
                ),
                assessments=assessments,
                accepted=comparable,
                accepted_claim_indexes=comparable_indexes,
                comparison_document_count=admissible_documents,
            )
        # The comparison tool extracts design/population/result facets structurally
        # from complete documents; a passage-level claim is not its only evidence.
        return _decision(
            profile=profile,
            verdict="ANSWERABLE",
            code="comparison_structurally_answerable",
            detail="At least two admissible studies were compared structurally.",
            assessments=assessments,
            accepted=comparable,
            accepted_claim_indexes=comparable_indexes,
            comparison_document_count=admissible_documents,
        )
    if not admissible:
        missing_entity = all(a.missing_entities for a in assessments)
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
            accepted_claim_indexes=admissible_indexes,
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
        accepted_claim_indexes=admissible_indexes,
        contradictory=contradictory,
        claim_support="SUPPORTED_BY_STRUCTURED_EVIDENCE_GATE",
    )
