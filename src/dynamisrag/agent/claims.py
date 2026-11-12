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

_NUMBER: Final = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*%)?")
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
        "improvement", "improves", "gain", "gains", "rise", "rose", "elevated", "enhanced",
        "enhance", "larger", "more", "stiffer", "faster",
    }
)  # fmt: skip
_DOWN: Final[frozenset[str]] = frozenset(
    {
        "decrease", "decreased", "decreases", "decreasing", "reduced", "reduction", "reduces",
        "lower", "declined", "decline", "drop", "dropped", "less", "diminished", "worse",
        "impaired", "smaller", "slower", "fewer", "loss",
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
    """Return (all canonical numbers, numbers written as a percentage)."""
    plain: set[str] = set()
    percent: set[str] = set()
    for match in _NUMBER.finditer(text):
        value = _canonical_number(match.group(1), match.group(2))
        plain.add(value)
        if match.group(3):
            percent.add(value)
    for match in re.finditer(
        r"(\d+)(?:\.(\d+))?\s*(?:percent|per cent)(?![a-z])", text, flags=re.I
    ):
        value = _canonical_number(match.group(1), match.group(2))
        plain.add(value)
        percent.add(value)
    return plain, percent


def assess_numeric_provenance(claim_text: str, quote: str) -> tuple[NumericProvenance, list[str]]:
    """Check that the claim's numbers (and percent-ness) are those of the quoted span."""
    claim_all, claim_pct = _numbers(claim_text)
    if not claim_all:
        return "NOT_APPLICABLE", []
    quote_all, quote_pct = _numbers(quote)
    missing = sorted(claim_all - quote_all)
    if missing:
        return "FAILED", [f"number_not_in_quoted_span:{','.join(missing)}"]
    not_pct = sorted(claim_pct - quote_pct)
    if not_pct:
        return "FAILED", [f"claimed_as_percentage_but_not_in_quote:{','.join(not_pct)}"]
    return "VERIFIED", []


def _stem(word: str) -> str:
    return word[:6]


def _content_stems(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:'t)?", text.lower()))


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True)
class SemanticAssessment:
    status: SemanticSupport
    score: float
    reasons: tuple[str, ...]


def assess_semantic_support(claim_text: str, quote: str) -> SemanticAssessment:
    """Lexical-consistency check of a claim against the quote it cites (see module limits)."""
    claim_norm, quote_norm = _normalise(claim_text), _normalise(quote)
    if claim_norm == quote_norm or claim_norm in quote_norm:
        return SemanticAssessment("IDENTICAL_TO_QUOTE", 1.0, ())

    claim_stems = _content_stems(claim_text)
    quote_stems = _content_stems(quote)
    coverage = len(claim_stems & quote_stems) / len(claim_stems) if claim_stems else 0.0
    reasons: list[str] = []
    if claim_stems and coverage < SEMANTIC_COVERAGE_THRESHOLD:
        missing = sorted(claim_stems - quote_stems)[:6]
        reasons.append(f"content_not_in_quote:{','.join(missing)}")

    claim_tok, quote_tok = _tokens(claim_text), _tokens(quote)
    claim_neg = bool(claim_tok & _NEGATION) or "n't" in claim_text.lower()
    quote_neg = bool(quote_tok & _NEGATION) or "n't" in quote.lower()
    if claim_neg != quote_neg:
        reasons.append("negation_mismatch")

    claim_up, claim_down = bool(claim_tok & _UP), bool(claim_tok & _DOWN)
    quote_up, quote_down = bool(quote_tok & _UP), bool(quote_tok & _DOWN)
    if claim_up and not claim_down and not quote_up:
        reasons.append("reversed_direction" if quote_down else "direction_not_in_quote")
    if claim_down and not claim_up and not quote_down:
        reasons.append("reversed_direction" if quote_up else "direction_not_in_quote")

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
