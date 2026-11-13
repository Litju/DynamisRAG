"""Layered, deterministic claim assessment for RES-637 citation verification.

A model-proposed finding has four *independent* properties and this module owns the two that
are about the **claim wording** (the other two are canonical quotation identity, checked against
the stored source spans in ``ScientificToolRegistry.verify_claims``, and expert review, which
nothing in RES-637 sets):

1. ``assess_numeric_provenance`` — every number in the claim must occur in the *quoted span*
   itself (not merely somewhere in the passage), and a number the claim presents as a percentage
   must be a percentage in the quote. This catches "right number, wrong sentence" and
   "absolute value restated as a percent".
2. ``assess_semantic_support`` — does the quote support the claim's *wording*? A deterministic
   lexical-consistency heuristic over content-term coverage, negation parity, direction of change
   (increase vs decrease), and unsupported causal/universal language.

**Honest limits.** The semantic check is a heuristic gate, not natural-language inference. It
rejects clear overreach (reversed effect, dropped negation, unsupported causal claim, content the
quote does not contain). It cannot certify that a subtle paraphrase is faithful, so a pass is
labelled ``SUPPORTED_BY_LEXICAL_HEURISTIC`` — never "scientifically supported" — and every finding
stays ``MODEL_PROPOSED / PROPOSED`` with ``expert_review = NOT_REVIEWED``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "SEMANTIC_ASSESSOR",
    "SEMANTIC_COVERAGE_THRESHOLD",
    "NumericProvenance",
    "SemanticAssessment",
    "SemanticSupport",
    "assess_numeric_provenance",
    "assess_semantic_support",
]

SEMANTIC_ASSESSOR: Final = "lexical-consistency-v1"
SEMANTIC_COVERAGE_THRESHOLD: Final[float] = 0.6

NumericProvenance = Literal["VERIFIED", "FAILED", "NOT_APPLICABLE"]
SemanticSupport = Literal[
    "IDENTICAL_TO_QUOTE", "SUPPORTED_BY_LEXICAL_HEURISTIC", "UNSUPPORTED", "NOT_ASSESSED"
]

_NUMBER: Final = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*(?:%|percent(?![a-z])|per\s+cent(?![a-z])))?",
    flags=re.IGNORECASE,
)
_WORD: Final = re.compile(r"[a-z][a-z-]{3,}")
_STOP: Final[frozenset[str]] = frozenset(
    {
        "that", "this", "with", "from", "were", "have", "been", "which", "their", "there",
        "these", "those", "than", "then", "also", "such", "into", "between", "during",
        "after", "before", "while", "over", "under", "among", "other", "about", "within",
        "study", "studies", "results", "result", "found", "show", "shows", "showed", "shown",
        "paper", "authors", "reported", "reports", "participants",
    }
)  # fmt: skip
_NEGATION: Final[frozenset[str]] = frozenset(
    {"no", "not", "without", "never", "none", "neither", "nor", "cannot", "failed", "fail", "lack"}
)
_UP: Final[frozenset[str]] = frozenset(
    {
        "increase", "increased", "increases", "increasing", "higher", "greater", "improved",
        "improvement", "improves", "improve", "gain", "gains", "gained", "rise", "rises", "rose",
        "raised", "raises", "raise", "elevated", "elevates", "elevate", "enhanced", "enhances",
        "enhance", "boosted", "boosts", "promoted", "promotes", "accelerated", "accelerates",
        "amplified", "amplifies", "upregulated", "upregulates", "larger", "more", "stiffer",
        "faster", "stronger", "longer", "taller", "growth", "grew", "expanded", "expands",
    }
)  # fmt: skip
_DOWN: Final[frozenset[str]] = frozenset(
    {
        "decrease", "decreased", "decreases", "decreasing", "reduced", "reduction", "reduces",
        "reduce", "lower", "lowers", "lowered", "declined", "decline", "declines", "drop",
        "dropped", "drops", "less", "diminished", "diminishes", "worse", "worsened", "worsens",
        "impaired", "impairs", "smaller", "slower", "slowed", "slows", "fewer", "loss",
        "suppressed", "suppresses", "suppress", "inhibited", "inhibits", "inhibit", "attenuated",
        "attenuates", "blunted", "blunts", "dampened", "abolished", "abolishes", "downregulated",
        "downregulates", "weaker", "shorter", "shrank", "shrinks", "diminution",
    }
)  # fmt: skip
_HEDGE: Final[frozenset[str]] = frozenset(
    {
        "may", "might", "could", "suggest", "suggests", "suggested", "possibly", "possible",
        "potentially", "potential", "trend", "trended", "appears", "appeared", "likely",
        "tentatively", "preliminary", "approximately", "about",
    }
)  # fmt: skip
_CAUSAL: Final[tuple[str, ...]] = (
    "caused",
    "causes",
    "causing",
    "leads to",
    "led to",
    "results in",
    "resulted in",
    "because of",
    "due to",
    "proves",
    "proved",
)
_UNIVERSAL: Final[frozenset[str]] = frozenset({"always", "every", "all", "never", "everyone"})


def _canonical_number(whole: str, fraction: str | None) -> str:
    text = whole.replace(",", "")
    if fraction is not None:
        text = f"{text}.{fraction.rstrip('0')}" if fraction.rstrip("0") else text
    return text


def _numbers(text: str) -> tuple[set[str], set[str]]:
    """Return (numbers written WITHOUT a percent sign, numbers written as a percentage)."""
    plain: set[str] = set()
    percent: set[str] = set()
    for match in _NUMBER.finditer(text):
        value = _canonical_number(match.group(1), match.group(2))
        (percent if match.group(3) else plain).add(value)
    return plain, percent


def assess_numeric_provenance(claim_text: str, quote: str) -> tuple[NumericProvenance, list[str]]:
    """Check the claim's numbers (and their percent-ness, in BOTH directions) against the quote."""
    claim_plain, claim_pct = _numbers(claim_text)
    if not claim_plain and not claim_pct:
        return "NOT_APPLICABLE", []
    quote_plain, quote_pct = _numbers(quote)
    quote_all = quote_plain | quote_pct
    missing = sorted((claim_plain | claim_pct) - quote_all)
    if missing:
        return "FAILED", [f"number_not_in_quoted_span:{','.join(missing)}"]
    # "36.8%" claimed where the quote says "36.8 cm" (absolute restated as a percentage)...
    as_pct = sorted(claim_pct - quote_pct)
    if as_pct:
        return "FAILED", [f"claimed_as_percentage_but_not_in_quote:{','.join(as_pct)}"]
    # ...and "5 patients" claimed where the quote says "5% of patients" (percentage as a count).
    as_count = sorted((claim_plain & quote_pct) - quote_plain)
    if as_count:
        return "FAILED", [f"quote_gives_percentage_but_claim_does_not:{','.join(as_count)}"]
    return "VERIFIED", []


def _stem(word: str) -> str:
    return word[:6]


def _content_stems(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:'t)?", text.lower()))


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _polarity_sequence(text: str) -> list[int]:
    """Ordered +1/-1 direction-of-change words (so swapped roles reverse the sequence)."""
    sequence: list[int] = []
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _UP:
            sequence.append(1)
        elif word in _DOWN:
            sequence.append(-1)
    return sequence


def _is_ordered_subsequence(needle: list[int], haystack: list[int]) -> bool:
    iterator = iter(haystack)
    return all(any(item == candidate for candidate in iterator) for item in needle)


@dataclass(frozen=True)
class SemanticAssessment:
    status: SemanticSupport
    score: float
    reasons: tuple[str, ...]


def assess_semantic_support(claim_text: str, quote: str) -> SemanticAssessment:
    """Lexical-consistency check of a claim against the quote it cites (see module limits)."""
    claim_norm, quote_norm = _normalise(claim_text), _normalise(quote)
    claim_tok, quote_tok = _tokens(claim_text), _tokens(quote)
    if claim_norm == quote_norm:
        return SemanticAssessment("IDENTICAL_TO_QUOTE", 1.0, ())
    if claim_norm in quote_norm and not (quote_tok & (_NEGATION | _HEDGE)) - claim_tok:
        # A fragment of the quote only counts as the quote itself when it does not strip a
        # negation or hedge that the full sentence carries ("no significant increase" -> "...").
        return SemanticAssessment("IDENTICAL_TO_QUOTE", 1.0, ())

    claim_stems = _content_stems(claim_text)
    quote_stems = _content_stems(quote)
    coverage = len(claim_stems & quote_stems) / len(claim_stems) if claim_stems else 0.0
    reasons: list[str] = []
    if claim_stems and coverage < SEMANTIC_COVERAGE_THRESHOLD:
        missing = sorted(claim_stems - quote_stems)[:6]
        reasons.append(f"content_not_in_quote:{','.join(missing)}")

    claim_neg = bool(claim_tok & _NEGATION) or "n't" in claim_text.lower()
    quote_neg = bool(quote_tok & _NEGATION) or "n't" in quote.lower()
    if claim_neg != quote_neg:
        reasons.append("negation_mismatch")

    claim_seq, quote_seq = _polarity_sequence(claim_text), _polarity_sequence(quote)
    if claim_seq:
        if not quote_seq:
            reasons.append("direction_not_in_quote")
        elif not _is_ordered_subsequence(claim_seq, quote_seq) or (
            len(claim_seq) == len(quote_seq) and claim_seq != quote_seq
        ):
            reasons.append("reversed_direction")
    if (quote_tok & _HEDGE) - claim_tok:
        reasons.append("hedge_dropped")

    claim_lower, quote_lower = claim_text.lower(), quote.lower()
    if any(p in claim_lower and p not in quote_lower for p in _CAUSAL):
        reasons.append("causal_language_not_in_quote")
    if (claim_tok & _UNIVERSAL) - quote_tok:
        reasons.append("universal_language_not_in_quote")

    if reasons:
        return SemanticAssessment("UNSUPPORTED", round(coverage, 3), tuple(reasons))
    if not claim_stems:
        return SemanticAssessment("UNSUPPORTED", 0.0, ("no_assessable_content",))
    return SemanticAssessment("SUPPORTED_BY_LEXICAL_HEURISTIC", round(coverage, 3), ())
