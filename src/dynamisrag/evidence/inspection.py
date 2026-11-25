"""Validated mapping from a verified citation span to passage and paragraph coordinates.

A verified citation carries three coordinates and two identities:

* ``span_key`` identifies one immutable source-span row;
* ``paragraph_key`` (derived from the version key and source anchor) identifies the paragraph;
* ``start_char``/``end_char`` are **paragraph-relative** offsets, not passage-relative.

The browser must never call ``text.indexOf(quote)``: repeated quotations are common, and
the first textual match can be the wrong occurrence. This module maps the stored span to
the exact passage offset by replaying the same reconstruction used for canonical source
verification, then proves the mapping by exact character equality. Any mismatch is
reported and **no highlight** is emitted. There is no fallback search for another match.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final, Literal
from uuid import UUID

from dynamisrag.db.canonical import (
    PassageProjectionRecords,
    PassageSourceSpanLineage,
)
from dynamisrag.db.evidence import InvalidProvenanceError, source_fingerprint

__all__ = [
    "HighlightMapping",
    "HighlightMappingError",
    "HighlightReason",
    "locate_verified_quote",
]

HighlightReason = Literal[
    "incomplete_request",
    "invalid_span_key",
    "invalid_coordinates",
    "invalid_quote",
    "span_not_found",
    "paragraph_identity_mismatch",
    "quote_outside_span",
    "span_coordinate_mismatch",
    "provenance_invalid",
    "passage_mapping_mismatch",
]

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_QUOTE_CHARS: Final = 650


class HighlightMappingError(ValueError):
    """The stored coordinates do not identify the requested quotation. Fail closed."""

    def __init__(self, reason: HighlightReason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class HighlightMapping:
    """An exact, validated quotation location in both coordinate systems."""

    span_key: str
    paragraph_key: str
    paragraph_source_anchor: str
    paragraph_text: str
    start_char: int
    end_char: int
    passage_start_char: int
    passage_end_char: int
    quote_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "span_key": self.span_key,
            "paragraph_key": self.paragraph_key,
            "paragraph_source_anchor": self.paragraph_source_anchor,
            "paragraph_text": self.paragraph_text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "passage_start_char": self.passage_start_char,
            "passage_end_char": self.passage_end_char,
            "quote_sha256": self.quote_sha256,
        }


def _require_request(
    *,
    span_key: str | None,
    start_char: int | None,
    end_char: int | None,
    quote: str | None,
) -> tuple[str, int, int, str]:
    if span_key is None or start_char is None or end_char is None or quote is None or not quote:
        raise HighlightMappingError("incomplete_request")
    if not _SHA256.fullmatch(span_key):
        raise HighlightMappingError("invalid_span_key")
    if not 0 <= start_char < end_char or end_char - start_char > _MAX_QUOTE_CHARS:
        raise HighlightMappingError("invalid_coordinates")
    if len(quote) > _MAX_QUOTE_CHARS or quote != quote.strip():
        raise HighlightMappingError("invalid_quote")
    return span_key, start_char, end_char, quote


def _reconstruct(
    lineages: tuple[PassageSourceSpanLineage, ...],
) -> tuple[list[str], dict[UUID, tuple[int, int]]]:
    """Replay canonical passage reconstruction and record each paragraph's part origin."""
    parts: list[str] = []
    starts: dict[UUID, tuple[int, int]] = {}
    previous_paragraph_id: UUID | None = None
    previous_end = 0
    for lineage in lineages:
        span, paragraph = lineage.span, lineage.paragraph
        if paragraph.id == previous_paragraph_id:
            if (
                span.start_char < previous_end
                or paragraph.text[previous_end : span.start_char].strip()
            ):
                raise HighlightMappingError("provenance_invalid")
            parts[-1] += paragraph.text[previous_end : span.end_char]
        else:
            parts.append(paragraph.text[span.start_char : span.end_char])
            starts[paragraph.id] = (len(parts) - 1, span.start_char)
        previous_paragraph_id, previous_end = paragraph.id, span.end_char
    return parts, starts


def locate_verified_quote(
    record: PassageProjectionRecords,
    *,
    span_key: str | None,
    start_char: int | None,
    end_char: int | None,
    quote: str | None,
) -> HighlightMapping:
    """Map one verified citation to passage coordinates or raise a stated reason.

    The caller must have already verified the quote; this function independently
    re-checks canonical provenance, span membership and exact character equality.
    """
    span_key, start_char, end_char, quote = _require_request(
        span_key=span_key,
        start_char=start_char,
        end_char=end_char,
        quote=quote,
    )
    try:
        source_fingerprint(record)
    except InvalidProvenanceError as error:
        raise HighlightMappingError("provenance_invalid") from error

    lineage = next(
        (item for item in record.source_spans if item.span.span_key == span_key),
        None,
    )
    if lineage is None:
        raise HighlightMappingError("span_not_found")
    span, paragraph = lineage.span, lineage.paragraph
    if (
        span.paragraph_id != paragraph.id
        or paragraph.document_version_id != record.version.id
        or span.document_version_id != record.version.id
    ):
        raise HighlightMappingError("paragraph_identity_mismatch")
    if not 0 <= span.start_char < span.end_char <= len(paragraph.text):
        raise HighlightMappingError("provenance_invalid")
    if not (span.start_char <= start_char < end_char <= span.end_char):
        raise HighlightMappingError("quote_outside_span")
    if paragraph.text[start_char:end_char] != quote:
        raise HighlightMappingError("span_coordinate_mismatch")

    parts, starts = _reconstruct(record.source_spans)
    if "\n\n".join(parts) != record.passage.text:
        raise HighlightMappingError("passage_mapping_mismatch")
    origin = starts.get(paragraph.id)
    if origin is None:
        raise HighlightMappingError("passage_mapping_mismatch")
    part_index, part_start_char = origin
    passage_start = sum(len(part) for part in parts[:part_index]) + 2 * part_index
    passage_start += start_char - part_start_char
    passage_end = passage_start + (end_char - start_char)
    if record.passage.text[passage_start:passage_end] != quote:
        raise HighlightMappingError("passage_mapping_mismatch")

    return HighlightMapping(
        span_key=span_key,
        paragraph_key=paragraph.paragraph_key,
        paragraph_source_anchor=paragraph.source_anchor,
        paragraph_text=paragraph.text,
        start_char=start_char,
        end_char=end_char,
        passage_start_char=passage_start,
        passage_end_char=passage_end,
        quote_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
    )
