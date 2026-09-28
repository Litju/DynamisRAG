"""Deterministic text extraction and normalization for JATS parsing (RES-133).

One explicit pipeline is used for every textual value the parser produces —
titles, paragraphs, section titles, captions, reference raw text and
metadata text fields — so the same source bytes always yield the same
normalized text:

    controlled text extraction (inline-markup flattening)
    -> deterministic whitespace normalization

Extraction never reserializes XML and never introduces spaces that were
not present: element text and tails are concatenated in document order,
exactly as ``itertext`` would, except that ``<alternatives>`` contributes
a single deterministic representation instead of duplicating MathML + TeX +
textual forms.
"""

from __future__ import annotations

from collections.abc import Iterator
from xml.etree.ElementTree import Element

__all__ = [
    "XLINK_HREF",
    "XML_LANG",
    "element_text",
    "first_local",
    "iter_local",
    "local_name",
    "normalize_language",
    "normalize_text",
]

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
"""The ``xml:lang`` attribute in its expanded ElementTree form."""

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
"""The ``xlink:href`` attribute in its expanded ElementTree form."""

_GRAPHIC_TAGS = frozenset({"graphic", "media", "inline-graphic", "inline-media"})
"""Elements that carry no narrative text (their ``xlink:href`` is asset
locator provenance, not text)."""

_ALTERNATIVES_TAG = "alternatives"
"""JATS wrapper offering multiple representations of one inline object."""


def local_name(tag: str) -> str:
    """Return an ElementTree tag's local name, stripping any namespace.

    PMC material includes JATS-style and historically related NLM tagging,
    so elements are always handled by local name rather than by assuming one
    exact namespace arrangement.
    """
    return tag.rsplit("}", 1)[-1]


def normalize_text(raw: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends.

    Deterministic and total: XML formatting whitespace (indentation, newlines
    between tags) disappears, while meaningful punctuation and inline textual
    order are preserved exactly as extracted.
    """
    return " ".join(raw.split())


def normalize_language(raw: str | None) -> str:
    """Normalize a source ``xml:lang`` value to the canonical language
    contract.

    Only the primary subtag is kept where it is compatible with the existing
    ``LanguageCode``: ``en`` -> ``en``, ``en-US`` -> ``en``, ``pt-BR`` ->
    ``pt``. An absent or unmappable value yields the explicit ISO 639
    "undetermined" code ``und`` — language is never guessed from text.
    """
    if not raw:
        return "und"
    subtag = raw.strip().replace("_", "-").split("-", 1)[0].lower()
    if len(subtag) in (2, 3) and subtag.isascii() and subtag.isalpha():
        return subtag
    return "und"


def element_text(element: Element) -> str:
    """Return the element's text content with inline markup flattened.

    Text and tails are concatenated in document order — identical to
    ``"".join(element.itertext())`` for ordinary content — except that:

    * ``<alternatives>`` contributes its first non-graphic representation
      only, so MathML + TeX + textual alternatives are never concatenated
      into duplicate paragraph text;
    * graphic/media elements contribute no text at all.
    """
    parts: list[str] = []
    _collect_text(element, parts)
    return "".join(parts)


def _collect_text(element: Element, parts: list[str]) -> None:
    if local_name(element.tag) in _GRAPHIC_TAGS:
        return
    if local_name(element.tag) == _ALTERNATIVES_TAG:
        for child in element:
            if local_name(child.tag) in _GRAPHIC_TAGS:
                continue
            _collect_text(child, parts)
            return
        return
    parts.append(element.text or "")
    for child in element:
        _collect_text(child, parts)
        parts.append(child.tail or "")


def iter_local(element: Element, name: str) -> Iterator[Element]:
    """Yield direct children of ``element`` whose local name is ``name``."""
    for child in element:
        if local_name(child.tag) == name:
            yield child


def first_local(element: Element, name: str) -> Element | None:
    """Return the first direct child with the given local name, or ``None``."""
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None
