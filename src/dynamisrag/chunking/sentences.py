"""Deterministic scientific sentence segmentation (RES-134).

Sentence splitting is used only when a single Paragraph exceeds
``max_tokens``; it is never applied to paragraphs that already fit. The
splitter is a small explicit scientific policy — no spaCy, no heavyweight NLP
stack — that returns exact character spans into the canonical normalized
paragraph text. Paragraph text is never normalized or rewritten again: the
spans are used directly as PassageSourceSpan provenance.

The policy is deterministic and total: the same text always yields the same
spans. It handles the representative scientific cases — common abbreviations
(``Fig.``, ``Eq.``, ``Dr.``, ``et al.``, ``e.g.``, ``i.e.``, ``vs.``,
``No.`` ...), decimals, initials, ellipses and sentence-ending punctuation
followed by quotes or brackets — and never splits inside a Unicode code
point.

A sentence that still exceeds the hard maximum is split by
:func:`split_oversized_sentence` at no-whitespace lexical cluster boundaries
(``sci-sent-1.1`` semantics), so attached punctuation stays with its cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dynamisrag.chunking.tokens import LEXICAL_TOKEN_PATTERN

__all__ = ["SentenceSpan", "split_oversized_sentence", "split_sentences"]


@dataclass(frozen=True)
class SentenceSpan:
    """One exact sentence span into canonical paragraph text.

    ``text[start:end]`` is the exact canonical substring; the text is never
    rewritten. ``start`` is the first non-whitespace character of the
    sentence and ``end`` is past the terminator and any closing
    quotes/brackets.
    """

    start: int
    end: int
    text: str


_SENTENCE_TERMINATORS: Final[frozenset[str]] = frozenset({".", "!", "?"})
"""Characters that can end a sentence."""

_CLOSING_CHARACTERS: Final[frozenset[str]] = frozenset(
    {'"', "'", ")", "]", "}", "\u201d", "\u2019", "\u00bb"}
)
"""Quotes and brackets that belong to the sentence they follow."""

_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "al",
        "app",
        "approx",
        "assn",
        "bros",
        "ca",
        "cf",
        "ch",
        "chap",
        "co",
        "corp",
        "dept",
        "dr",
        "ed",
        "eds",
        "eq",
        "eqs",
        "esp",
        "etc",
        "fig",
        "figs",
        "gen",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "nos",
        "ord",
        "pp",
        "prof",
        "ref",
        "refs",
        "rev",
        "sc",
        "sec",
        "secs",
        "seq",
        "sp",
        "spp",
        "st",
        "suppl",
        "trans",
        "var",
        "viz",
        "vol",
        "vols",
        "vs",
    }
)
"""Lowercase word stems whose trailing period is not a sentence boundary.

The set is matched against the letter run immediately before the period. The
classic tradeoff is deliberate and deterministic: a sentence that genuinely
ends with one of these words (e.g. ``... the answer is no.``) is not split
there, while the scientific abbreviation reading (``No. 5``, ``et al.``,
``Fig. 2``) is always preserved.
"""

_DOTTED_ABBREVIATIONS: Final[frozenset[str]] = frozenset({"e.g", "i.e"})
"""Dotted abbreviations whose final period is never a sentence boundary.

Only the always-mid-sentence Latin abbreviations are listed. Other dotted
forms (``p.m.``, ``a.m.``, ``U.S.``) can end a sentence, so they are not
treated as abbreviation tails here.
"""


def _is_initial(text: str, i: int) -> bool:
    """Decide whether the period at ``i`` follows a single-letter initial.

    An initial is a single uppercase letter (``A. Smith``, ``B. Jones``,
    ``J. R. R. Tolkien``). The tradeoff is deliberate: a single-letter word at
    a sentence end followed by a capitalized word (``... the variable is X.
    Next``) reads as an initial and does not split, while the spec-required
    initial reading always holds. A following abbreviation word (``X.
    Fig. ...``) is not a name, so the period stays a boundary.
    """
    n = len(text)
    if not (i >= 1 and text[i - 1].isupper() and text[i - 1].isalpha()):
        return False
    if i >= 2 and text[i - 2].isalpha():
        return False
    if i + 1 < n and not text[i + 1].isspace():
        return False
    if i + 1 >= n:
        return True
    j = i + 1
    while j < n and text[j].isspace():
        j += 1
    if j >= n or not text[j].isupper():
        return False
    k = j
    while k < n and text[k].isalpha():
        k += 1
    return text[j:k].lower() not in _ABBREVIATIONS


def _is_within_token(text: str, i: int) -> bool:
    """The period at ``i`` is part of a larger token, not a boundary.

    Covers ellipses (``...``), period acronyms (``U.S.``, ``U.K.``), decimal
    numbers (``3.14``), never-boundary dotted abbreviations (``e.g.``,
    ``i.e.``) and lowercase-followed tokens (``www.example.com``, ``v1.2``).
    """
    n = len(text)
    return (
        (i + 1 < n and text[i + 1] == ".")
        or (i + 2 < n and text[i + 1].isupper() and text[i + 1].isalpha() and text[i + 2] == ".")
        or (i >= 2 and text[i - 2] == "." and text[i - 1].isupper() and text[i - 1].isalpha())
        or (i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit())
        or (i >= 3 and text[i - 3 : i].lower() in _DOTTED_ABBREVIATIONS)
        or (i + 1 < n and text[i + 1].islower())
    )


def _is_period_sentence_end(text: str, i: int) -> bool:
    """Decide whether the period at ``i`` ends a sentence.

    A period is not a sentence boundary when it is part of a larger token
    (see :func:`_is_within_token`), a single-letter initial (``A. Smith``) or
    a known scientific abbreviation word.
    """
    if _is_within_token(text, i):
        return False
    if _is_initial(text, i):
        return False
    j = i
    while j > 0 and text[j - 1].isalpha():
        j -= 1
    return text[j:i].lower() not in _ABBREVIATIONS


def split_sentences(text: str) -> tuple[SentenceSpan, ...]:
    """Split ``text`` into deterministic exact sentence spans.

    Returns spans in document order whose ``text`` values are exact substrings
    of ``text``. Adjacent spans reconstruct the original text up to the
    whitespace runs between sentences.
    """
    spans: list[SentenceSpan] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        if text[i] in _SENTENCE_TERMINATORS and (
            text[i] != "." or _is_period_sentence_end(text, i)
        ):
            j = i + 1
            while j < n and text[j] in _SENTENCE_TERMINATORS:
                j += 1
            k = j
            while k < n and text[k] in _CLOSING_CHARACTERS:
                k += 1
            s = start
            while s < k and text[s].isspace():
                s += 1
            spans.append(SentenceSpan(s, k, text[s:k]))
            start = k
            while start < n and text[start].isspace():
                start += 1
            i = start
        else:
            i += 1
    s = start
    while s < n and text[s].isspace():
        s += 1
    if s < n:
        spans.append(SentenceSpan(s, n, text[s:n]))
    return tuple(spans)


def split_oversized_sentence(
    text: str, start: int, end: int, max_tokens: int
) -> tuple[SentenceSpan, ...]:
    """Split one over-long sentence at deterministic lexical cluster boundaries.

    Each lexical token counts as one token. Tokens that touch without
    intervening whitespace — a word and its attached punctuation, a number
    and its decimal point, a parenthesized run — form one no-whitespace lexical
    cluster and are never separated: attached punctuation stays with its
    lexical cluster, so no punctuation-only chunk is produced when the
    cluster fits. Greedy packing never exceeds ``max_tokens`` and never
    splits inside a Unicode code point. A single cluster longer than
    ``max_tokens`` is the one case where the hard ceiling forces a split
    inside a cluster, and it then splits at token boundaries. Chunk text is
    the exact substring from the first to the last token of the chunk.
    """
    tokens: list[tuple[int, int]] = [
        (match.start(), match.end()) for match in LEXICAL_TOKEN_PATTERN.finditer(text, start, end)
    ]
    if not tokens:
        return (SentenceSpan(start, end, text[start:end]),)
    clusters: list[list[tuple[int, int]]] = []
    for token in tokens:
        if clusters and clusters[-1][-1][1] == token[0]:
            clusters[-1].append(token)
        else:
            clusters.append([token])
    spans: list[SentenceSpan] = []
    chunk: list[tuple[int, int]] = []
    chunk_tokens = 0
    for cluster in clusters:
        cluster_tokens = len(cluster)
        if chunk and chunk_tokens + cluster_tokens > max_tokens:
            spans.append(_span(text, chunk))
            chunk = []
            chunk_tokens = 0
        if cluster_tokens > max_tokens:
            # The hard ceiling forces a split inside the cluster itself.
            for token in cluster:
                if chunk_tokens + 1 > max_tokens:
                    spans.append(_span(text, chunk))
                    chunk = []
                    chunk_tokens = 0
                chunk.append(token)
                chunk_tokens += 1
            continue
        chunk.extend(cluster)
        chunk_tokens += cluster_tokens
    if chunk:
        spans.append(_span(text, chunk))
    return tuple(spans)


def _span(text: str, chunk: list[tuple[int, int]]) -> SentenceSpan:
    """Build the exact-substring span of one packed chunk of tokens."""
    return SentenceSpan(chunk[0][0], chunk[-1][1], text[chunk[0][0] : chunk[-1][1]])
