"""Deterministic native JATS parser (RES-133).

``JatsParser`` turns exact JATS XML bytes into a deterministic
:class:`ParsedJatsArticle`: a typed, normalized, fully anchored
representation of the article's scientific structure. The parser is as close
to a pure function as practical:

    same XML bytes + same parser/normalizer revision
        -> same normalized parsed representation
        -> same content fingerprint

No clock, database calls, random ids, network or object-store calls happen
inside the parser; surrogate UUIDs belong to canonical materialization
(persistence), never to semantic parsing. The parser never mutates the
source bytes and never performs retrieval chunking — paragraphs are source
structure, passages are RES-134's retrieval structure.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any, Final
from xml.etree.ElementTree import Element, ParseError

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring as safe_fromstring

from dynamisrag.domain.identity import digest, normalize_doi
from dynamisrag.domain.values import ParagraphRegion, RevisionTag
from dynamisrag.jats.anchors import AnchorIndex, build_anchor_index
from dynamisrag.jats.errors import JatsMissingRequiredMetadata, JatsParseError, JatsParseWarning
from dynamisrag.jats.text import (
    XLINK_HREF,
    XML_LANG,
    element_text,
    first_local,
    iter_local,
    local_name,
    normalize_language,
    normalize_text,
)

__all__ = [
    "JATS_NORMALIZER_REVISION",
    "JATS_PARSER_REVISION",
    "JatsParser",
    "ParsedCitation",
    "ParsedFigure",
    "ParsedJatsArticle",
    "ParsedParagraph",
    "ParsedSection",
    "ParsedTable",
]

JATS_PARSER_REVISION: Final[RevisionTag] = "jats-1.0"
"""Revision of the JATS parsing semantics. Any behavior change that alters
the canonical parse result must bump this tag."""

JATS_NORMALIZER_REVISION: Final[RevisionTag] = "norm-1.0"
"""Revision of the text/metadata normalization semantics. Any behavior change
that alters normalized content must bump this tag."""

_DOI_VALUE_FORMAT = r"10\.[0-9]{4,9}/\S+"
_PMID_VALUE_FORMAT = r"[0-9]{1,10}"
_PMCID_VALUE_FORMAT = r"PMC[0-9]{1,12}"
"""Identifier value formats, mirrored from the canonical value contracts."""

_CITATION_TAGS = frozenset({"mixed-citation", "element-citation"})
"""The JATS citation element kinds, handled identically."""

_EXCLUDED_PARAGRAPH_ANCESTORS = frozenset(
    {"ref", "ref-list", "table-wrap", "table-wrap-foot", "fig", "license", "caption", "title-group"}
)
"""Ancestors whose ``<p>`` descendants are not narrative paragraphs.

These ``<p>`` elements belong to structures the canonical model represents
independently (references, table/figure captions, licenses, titles), so
extracting them as paragraphs would double-count their text.
"""

_REGION_TAGS = frozenset({"front", "body", "back"})
"""The JATS top-level article regions a paragraph can belong to."""


@dataclass(frozen=True)
class ParsedSection:
    """One canonical section: hierarchy position, title and provenance."""

    ordinal: int
    depth: int
    title: str | None
    semantic_type: str | None
    source_anchor: str
    structural_path: str
    content_fingerprint: str
    parent_anchor: str | None = None


@dataclass(frozen=True)
class ParsedParagraph:
    """One canonical source paragraph: region, order, anchor and text."""

    ordinal: int
    region: ParagraphRegion
    source_anchor: str
    section_anchor: str | None
    text: str
    content_sha256: str


@dataclass(frozen=True)
class ParsedCitation:
    """One canonical bibliographic reference."""

    ordinal: int
    source_reference_id: str | None
    source_anchor: str
    doi: str | None
    pmid: str | None
    pmcid: str | None
    title: str | None
    year: int | None
    raw_reference_text: str


@dataclass(frozen=True)
class ParsedTable:
    """One canonical table: label, caption, structured representation."""

    ordinal: int
    source_anchor: str
    section_anchor: str | None
    label: str | None
    caption: str | None
    structured_representation: Mapping[str, Any]
    content_fingerprint: str


@dataclass(frozen=True)
class ParsedFigure:
    """One canonical figure: label, caption, asset locator."""

    ordinal: int
    source_anchor: str
    section_anchor: str | None
    label: str | None
    caption: str | None
    asset_locator: str | None
    content_fingerprint: str


@dataclass(frozen=True)
class ParsedJatsArticle:
    """The deterministic normalized parse result for one JATS article.

    Equality is structural: two parses of the same bytes under the same
    parser/normalizer revisions produce equal values, which is what makes the
    content fingerprint a pure function of the source.
    """

    title: str
    language: str
    article_type: str | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    metadata: Mapping[str, Any]
    sections: tuple[ParsedSection, ...]
    paragraphs: tuple[ParsedParagraph, ...]
    citations: tuple[ParsedCitation, ...]
    tables: tuple[ParsedTable, ...]
    figures: tuple[ParsedFigure, ...]
    content_fingerprint: str
    warnings: tuple[JatsParseWarning, ...]


class JatsParser:
    """Parses exact JATS XML bytes into a deterministic canonical structure."""

    def parse(self, xml_bytes: bytes) -> ParsedJatsArticle:
        """Parse one JATS article. Raises :class:`JatsParseError` (narrow
        hierarchy) for malformed XML, a non-article root or a missing
        required title."""
        root = self._parse_xml(xml_bytes)
        if local_name(root.tag) != "article":
            raise JatsParseError(
                f"expected a JATS <article> root element, found <{local_name(root.tag)}>"
            )
        anchors = build_anchor_index(root)
        parents = {child: parent for parent in root.iter() for child in parent}
        warnings: list[JatsParseWarning] = [
            JatsParseWarning(
                code="duplicate-xml-id",
                message=(
                    f"source @id {duplicate_id!r} occurs more than once and is not "
                    "globally unique; every occurrence uses a structural-path anchor"
                ),
            )
            for duplicate_id in anchors.duplicate_ids
        ]

        title, article_meta = self._parse_title(root)
        language, raw_xml_lang = self._parse_language(root)
        article_type = root.get("article-type") or None
        doi, pmid, pmcid = self._parse_identifiers(article_meta, warnings)
        metadata = self._parse_metadata(root, article_meta, article_type, raw_xml_lang)
        sections = self._parse_sections(root, anchors)
        paragraphs = self._parse_paragraphs(root, anchors, parents)
        citations = self._parse_citations(root, anchors, warnings)
        tables = self._parse_tables(root, anchors, parents)
        figures = self._parse_figures(root, anchors, parents)
        fingerprint = self._content_fingerprint(
            title=title,
            language=language,
            metadata=metadata,
            sections=sections,
            paragraphs=paragraphs,
            citations=citations,
            tables=tables,
            figures=figures,
        )
        return ParsedJatsArticle(
            title=title,
            language=language,
            article_type=article_type,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            metadata=metadata,
            sections=sections,
            paragraphs=paragraphs,
            citations=citations,
            tables=tables,
            figures=figures,
            content_fingerprint=fingerprint,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Safe XML loading
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml(xml_bytes: bytes) -> Element:
        """Parse bytes with the safe XML stack: no DTD downloads, no external
        entities, no network schema resolution."""
        try:
            return safe_fromstring(xml_bytes)
        except ParseError as error:
            raise JatsParseError(f"source bytes are not well-formed XML: {error}") from error
        except DefusedXmlException as error:
            raise JatsParseError(
                f"source bytes contain forbidden XML constructs ({type(error).__name__})"
            ) from error

    # ------------------------------------------------------------------
    # Front matter: title, language, identifiers, metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_title(root: Element) -> tuple[str, Element]:
        """Extract the normalized article title and the ``article-meta``
        element. A missing title is a canonicalization error — no placeholder
        is ever invented."""
        front = first_local(root, "front")
        article_meta = first_local(front, "article-meta") if front is not None else None
        if article_meta is None:
            raise JatsMissingRequiredMetadata(
                "article has no front/article-meta; a title is required"
            )
        title_group = first_local(article_meta, "title-group")
        title_element = (
            first_local(title_group, "article-title") if title_group is not None else None
        )
        if title_element is None:
            raise JatsMissingRequiredMetadata(
                "article has no front/article-meta/title-group/article-title; a title is required"
            )
        title = normalize_text(element_text(title_element))
        if not title:
            raise JatsMissingRequiredMetadata("article-title is present but empty")
        return title, article_meta

    @staticmethod
    def _parse_language(root: Element) -> tuple[str, str | None]:
        """Normalize the source ``xml:lang``; ``und`` when absent or
        unmappable. The raw value is preserved separately in metadata."""
        raw = root.get(XML_LANG)
        if raw is None:
            front = first_local(root, "front")
            raw = front.get(XML_LANG) if front is not None else None
        return normalize_language(raw), (raw or None)

    @staticmethod
    def _parse_identifiers(
        article_meta: Element, warnings: list[JatsParseWarning]
    ) -> tuple[str | None, str | None, str | None]:
        """Parse explicit ``article-meta/article-id`` identifiers. Malformed
        optional identifiers are ignored (the raw values remain in metadata)
        and reported as parser diagnostics."""
        doi: str | None = None
        pmid: str | None = None
        pmcid: str | None = None
        for article_id in iter_local(article_meta, "article-id"):
            pub_type = (article_id.get("pub-id-type") or "").strip().lower()
            value = normalize_text(element_text(article_id))
            if not value:
                continue
            if pub_type == "doi":
                normalized = normalize_doi(value)
                if normalized is not None and re.fullmatch(_DOI_VALUE_FORMAT, normalized):
                    doi = doi or normalized
                else:
                    warnings.append(
                        JatsParseWarning(
                            code="malformed-identifier",
                            message=f"ignoring malformed article DOI {value!r}",
                        )
                    )
            elif pub_type == "pmid":
                if re.fullmatch(_PMID_VALUE_FORMAT, value):
                    pmid = pmid or value
                else:
                    warnings.append(
                        JatsParseWarning(
                            code="malformed-identifier",
                            message=f"ignoring malformed article PMID {value!r}",
                        )
                    )
            elif pub_type in ("pmcid", "pmc"):
                candidate = value.upper()
                if re.fullmatch(_PMCID_VALUE_FORMAT, candidate):
                    pmcid = pmcid or candidate
                else:
                    warnings.append(
                        JatsParseWarning(
                            code="malformed-identifier",
                            message=f"ignoring malformed article PMCID {value!r}",
                        )
                    )
        return doi, pmid, pmcid

    @staticmethod
    def _parse_metadata(
        root: Element,
        article_meta: Element,
        article_type: str | None,
        raw_xml_lang: str | None,
    ) -> dict[str, Any]:
        """Build the normalized, deterministically serializable version
        metadata. Fields with universal canonical meaning (title, language,
        document identifiers) stay first-class and are not buried here; the
        raw source identifier elements are preserved for provenance."""
        metadata: dict[str, Any] = {}
        if article_type:
            metadata["article_type"] = article_type
        article_ids = JatsParser._parse_article_ids(article_meta)
        if article_ids:
            metadata["article_ids"] = article_ids
        journal = JatsParser._parse_journal(article_meta)
        if journal:
            metadata["journal"] = journal
        dates = JatsParser._parse_pub_dates(article_meta)
        if dates:
            metadata["publication_dates"] = dates
        for field_name in ("volume", "issue", "fpage", "lpage", "elocation-id"):
            value = _optional_text(article_meta, field_name)
            if value:
                metadata[field_name] = value
        contributors = JatsParser._parse_contributors(article_meta)
        if contributors:
            metadata["contributors"] = contributors
        affiliations = JatsParser._parse_affiliations(article_meta)
        if affiliations:
            metadata["affiliations"] = affiliations
        keywords = JatsParser._parse_keywords(article_meta)
        if keywords:
            metadata["keywords"] = keywords
        subjects = JatsParser._parse_subjects(article_meta)
        if subjects:
            metadata["subjects"] = subjects
        if raw_xml_lang:
            metadata["raw_xml_lang"] = raw_xml_lang
        dtd_version = root.get("dtd-version")
        if dtd_version:
            metadata["jats_dtd_version"] = dtd_version
        return metadata

    @staticmethod
    def _parse_article_ids(article_meta: Element) -> list[dict[str, str]]:
        """The raw ``article-id`` elements, preserved verbatim for provenance
        (including identifiers whose values turned out to be malformed)."""
        return [
            {
                "pub_id_type": article_id.get("pub-id-type") or "",
                "value": normalize_text(element_text(article_id)),
            }
            for article_id in iter_local(article_meta, "article-id")
        ]

    @staticmethod
    def _parse_journal(article_meta: Element) -> dict[str, Any]:
        journal_meta = first_local(article_meta, "journal-meta")
        if journal_meta is None:
            return {}
        journal: dict[str, Any] = {}
        journal_title = first_local(journal_meta, "journal-title")
        if journal_title is not None:
            title = normalize_text(element_text(journal_title))
            if title:
                journal["title"] = title
        identifiers: list[dict[str, str]] = []
        for issn in iter_local(journal_meta, "issn"):
            value = normalize_text(element_text(issn))
            if value:
                identifiers.append({"type": issn.get("pub-type") or "print", "value": value})
        for journal_id in iter_local(journal_meta, "journal-id"):
            value = normalize_text(element_text(journal_id))
            if value:
                identifiers.append(
                    {
                        "type": journal_id.get("journal-id-type") or "publisher",
                        "value": value,
                    }
                )
        if identifiers:
            journal["identifiers"] = identifiers
        publisher = first_local(journal_meta, "publisher")
        if publisher is not None:
            publisher_name = first_local(publisher, "publisher-name")
            if publisher_name is not None:
                name = normalize_text(element_text(publisher_name))
                if name:
                    journal["publisher"] = name
        return journal

    @staticmethod
    def _parse_pub_dates(article_meta: Element) -> list[dict[str, str]]:
        dates: list[dict[str, str]] = []
        for pub_date in iter_local(article_meta, "pub-date"):
            entry: dict[str, str] = {}
            date_type = pub_date.get("pub-type") or pub_date.get("date-type")
            if date_type:
                entry["type"] = date_type
            for field_name in ("year", "month", "day"):
                value = _optional_text(pub_date, field_name)
                if value:
                    entry[field_name] = value
            iso_date = pub_date.get("iso-8601-date")
            if iso_date:
                entry["iso_8601"] = iso_date
            dates.append(entry)
        return dates

    @staticmethod
    def _parse_keywords(article_meta: Element) -> list[str]:
        keywords: list[str] = []
        for kwd_group in iter_local(article_meta, "kwd-group"):
            for kwd in iter_local(kwd_group, "kwd"):
                value = normalize_text(element_text(kwd))
                if value:
                    keywords.append(value)
        return keywords

    @staticmethod
    def _parse_subjects(article_meta: Element) -> list[str]:
        subjects: list[str] = []
        for subj_group in iter_local(article_meta, "subj-group"):
            for subject in iter_local(subj_group, "subject"):
                value = normalize_text(element_text(subject))
                if value:
                    subjects.append(value)
        return subjects

    @staticmethod
    def _parse_contributors(article_meta: Element) -> list[dict[str, Any]]:
        """Parse common JATS contributor structures into normalized metadata.
        No author identities are invented and no author table is built."""
        contributors: list[dict[str, Any]] = []
        for contrib_group in iter_local(article_meta, "contrib-group"):
            for contrib in iter_local(contrib_group, "contrib"):
                contributors.append(JatsParser._parse_contrib(contrib))
        return contributors

    @staticmethod
    def _parse_contrib(contrib: Element) -> dict[str, Any]:
        entry: dict[str, Any] = {"type": contrib.get("contrib-type") or "author"}
        JatsParser._apply_contrib_name(contrib, entry)
        orcid = JatsParser._parse_orcid(contrib)
        if orcid:
            entry["orcid"] = orcid
        affiliation_refs = JatsParser._parse_affiliation_refs(contrib)
        if affiliation_refs:
            entry["affiliation_refs"] = affiliation_refs
        return entry

    @staticmethod
    def _apply_contrib_name(contrib: Element, entry: dict[str, Any]) -> None:
        name = first_local(contrib, "name")
        if name is not None:
            surname = first_local(name, "surname")
            if surname is not None:
                value = normalize_text(element_text(surname))
                if value:
                    entry["surname"] = value
            given_names = first_local(name, "given-names")
            if given_names is not None:
                value = normalize_text(element_text(given_names))
                if value:
                    entry["given_names"] = value
            return
        collab = first_local(contrib, "collab")
        if collab is not None:
            value = normalize_text(element_text(collab))
            if value:
                entry["collab"] = value

    @staticmethod
    def _parse_orcid(contrib: Element) -> str | None:
        for contrib_id in iter_local(contrib, "contrib-id"):
            if (contrib_id.get("contrib-id-type") or "").strip().lower() == "orcid":
                value = normalize_text(element_text(contrib_id))
                if value:
                    return value
        return None

    @staticmethod
    def _parse_affiliation_refs(contrib: Element) -> list[str]:
        affiliation_refs: list[str] = []
        for xref in iter_local(contrib, "xref"):
            if (xref.get("ref-type") or "").strip().lower() == "aff":
                rid = xref.get("rid")
                if rid:
                    affiliation_refs.append(rid)
        return affiliation_refs

    @staticmethod
    def _parse_affiliations(article_meta: Element) -> list[dict[str, Any | None]]:
        """Parse article-level affiliations, preserving the source affiliation
        id and a flattened human-readable text. No organization resolution."""
        affiliations: list[dict[str, Any | None]] = []
        for aff in article_meta.iter():
            if local_name(aff.tag) != "aff":
                continue
            parts: list[str] = []
            institution = first_local(aff, "institution")
            if institution is None:
                wrap = first_local(aff, "institution-wrap")
                institution = first_local(wrap, "institution") if wrap is not None else None
            if institution is not None:
                value = normalize_text(element_text(institution))
                if value:
                    parts.append(value)
            addr_line = first_local(aff, "addr-line")
            if addr_line is not None:
                value = normalize_text(element_text(addr_line))
                if value:
                    parts.append(value)
            country = first_local(aff, "country")
            if country is not None:
                value = normalize_text(element_text(country))
                if value:
                    parts.append(value)
            affiliations.append({"id": aff.get("id"), "text": ", ".join(parts) or None})
        return affiliations

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _parse_sections(self, root: Element, anchors: AnchorIndex) -> tuple[ParsedSection, ...]:
        """Parse recursively nested ``body/sec`` into canonical sections with
        deterministic ordinals, depths and numeric structural paths."""
        body = first_local(root, "body")
        if body is None:
            return ()
        sections: list[ParsedSection] = []
        ordinals = count(1)
        self._walk_sections(
            body,
            parent_path="",
            depth=0,
            parent_anchor=None,
            sections=sections,
            ordinals=ordinals,
            anchors=anchors,
        )
        return tuple(sections)

    def _walk_sections(
        self,
        element: Element,
        *,
        parent_path: str,
        depth: int,
        parent_anchor: str | None,
        sections: list[ParsedSection],
        ordinals: Iterator[int],
        anchors: AnchorIndex,
    ) -> None:
        sibling_index = 0
        for child in element:
            if local_name(child.tag) != "sec":
                continue
            sibling_index += 1
            path = f"{parent_path}.{sibling_index}" if parent_path else str(sibling_index)
            title_element = first_local(child, "title")
            title = None
            if title_element is not None:
                normalized = normalize_text(element_text(title_element))
                if normalized:
                    title = normalized
            anchor = anchors.anchor_for(child)
            if anchor is None:
                continue
            sections.append(
                ParsedSection(
                    ordinal=next(ordinals),
                    depth=depth,
                    title=title,
                    semantic_type=child.get("sec-type") or None,
                    source_anchor=anchor,
                    structural_path=path,
                    content_fingerprint=self._section_fingerprint(child, anchors),
                    parent_anchor=parent_anchor,
                )
            )
            self._walk_sections(
                child,
                parent_path=path,
                depth=depth + 1,
                parent_anchor=anchor,
                sections=sections,
                ordinals=ordinals,
                anchors=anchors,
            )

    @staticmethod
    def _section_fingerprint(section: Element, anchors: AnchorIndex) -> str:
        """Deterministic fingerprint of a section's normalized content: its
        title, semantic type and every anchored descendant paragraph's
        anchor + normalized text, in document order."""
        parts: list[str] = []
        title_element = first_local(section, "title")
        parts.append(
            normalize_text(element_text(title_element)) if title_element is not None else ""
        )
        parts.append(section.get("sec-type") or "")
        for descendant in section.iter():
            if local_name(descendant.tag) != "p":
                continue
            anchor = anchors.anchor_for(descendant)
            if anchor is not None:
                parts.append(f"{anchor}\x1f{normalize_text(element_text(descendant))}")
        return digest(*parts)

    # ------------------------------------------------------------------
    # Paragraphs
    # ------------------------------------------------------------------

    def _parse_paragraphs(
        self, root: Element, anchors: AnchorIndex, parents: Mapping[Element, Element]
    ) -> tuple[ParsedParagraph, ...]:
        """Extract narrative ``<p>`` elements from the front (abstract), body
        and back regions, in document order, with a global deterministic
        ordinal. Paragraphs belonging to independently represented structures
        (references, captions, licenses, titles) are not double-counted;
        purely empty paragraphs after normalization are skipped."""
        paragraphs: list[ParsedParagraph] = []
        ordinals = count(1)
        for element in root.iter():
            if local_name(element.tag) != "p":
                continue
            if self._has_excluded_ancestor(element, parents):
                continue
            region = self._region_of(element, parents)
            if region is None:
                continue
            text = normalize_text(element_text(element))
            if not text:
                continue
            anchor = anchors.anchor_for(element)
            if anchor is None:
                continue
            paragraphs.append(
                ParsedParagraph(
                    ordinal=next(ordinals),
                    region=region,
                    source_anchor=anchor,
                    section_anchor=self._section_anchor_for(element, parents, anchors),
                    text=text,
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
        return tuple(paragraphs)

    @staticmethod
    def _has_excluded_ancestor(element: Element, parents: Mapping[Element, Element]) -> bool:
        current = parents.get(element)
        while current is not None:
            if local_name(current.tag) in _EXCLUDED_PARAGRAPH_ANCESTORS:
                return True
            current = parents.get(current)
        return False

    @staticmethod
    def _region_of(element: Element, parents: Mapping[Element, Element]) -> ParagraphRegion | None:
        current = parents.get(element)
        while current is not None:
            name = local_name(current.tag)
            if name == "front":
                return ParagraphRegion.FRONT
            if name == "body":
                return ParagraphRegion.BODY
            if name == "back":
                return ParagraphRegion.BACK
            current = parents.get(current)
        return None

    @staticmethod
    def _section_anchor_for(
        element: Element, parents: Mapping[Element, Element], anchors: AnchorIndex
    ) -> str | None:
        current = parents.get(element)
        while current is not None:
            if local_name(current.tag) == "sec":
                return anchors.anchor_for(current)
            current = parents.get(current)
        return None

    # ------------------------------------------------------------------
    # Citations / references
    # ------------------------------------------------------------------

    def _parse_citations(
        self, root: Element, anchors: AnchorIndex, warnings: list[JatsParseWarning]
    ) -> tuple[ParsedCitation, ...]:
        """Parse ``back/ref-list/ref`` (tolerantly: any ``ref``) including
        ``mixed-citation`` and ``element-citation`` structures. Explicit
        identifiers are recognized; nothing is inferred from free text and
        citations are never resolved to Documents."""
        citations: list[ParsedCitation] = []
        ordinals = count(1)
        for element in root.iter():
            if local_name(element.tag) != "ref":
                continue
            anchor = anchors.anchor_for(element)
            if anchor is None:
                continue
            citation_element = None
            for child in element:
                if local_name(child.tag) in _CITATION_TAGS:
                    citation_element = child
                    break
            source = citation_element if citation_element is not None else element
            doi, pmid, pmcid = self._parse_citation_ids(source, warnings, anchor)
            title_element = first_local(source, "article-title")
            title = None
            if title_element is not None:
                normalized = normalize_text(element_text(title_element))
                if normalized:
                    title = normalized
            year = self._parse_year(source, warnings, anchor)
            citations.append(
                ParsedCitation(
                    ordinal=next(ordinals),
                    source_reference_id=element.get("id") or None,
                    source_anchor=anchor,
                    doi=doi,
                    pmid=pmid,
                    pmcid=pmcid,
                    title=title,
                    year=year,
                    raw_reference_text=normalize_text(element_text(source)),
                )
            )
        return tuple(citations)

    @staticmethod
    def _parse_citation_ids(
        source: Element, warnings: list[JatsParseWarning], anchor: str | None
    ) -> tuple[str | None, str | None, str | None]:
        doi: str | None = None
        pmid: str | None = None
        pmcid: str | None = None
        for element in source.iter():
            if local_name(element.tag) != "pub-id":
                continue
            pub_type = (element.get("pub-id-type") or "").strip().lower()
            value = normalize_text(element_text(element))
            if not value:
                continue
            if pub_type == "doi":
                normalized = normalize_doi(value)
                if normalized is not None and re.fullmatch(_DOI_VALUE_FORMAT, normalized):
                    doi = doi or normalized
                else:
                    warnings.append(
                        JatsParseWarning(
                            code="malformed-identifier",
                            message=f"ignoring malformed reference DOI {value!r}",
                            source_anchor=anchor,
                        )
                    )
            elif pub_type == "pmid":
                if re.fullmatch(_PMID_VALUE_FORMAT, value):
                    pmid = pmid or value
                else:
                    warnings.append(
                        JatsParseWarning(
                            code="malformed-identifier",
                            message=f"ignoring malformed reference PMID {value!r}",
                            source_anchor=anchor,
                        )
                    )
            elif pub_type in ("pmcid", "pmc"):
                candidate = value.upper()
                if re.fullmatch(_PMCID_VALUE_FORMAT, candidate):
                    pmcid = pmcid or candidate
                else:
                    warnings.append(
                        JatsParseWarning(
                            code="malformed-identifier",
                            message=f"ignoring malformed reference PMCID {value!r}",
                            source_anchor=anchor,
                        )
                    )
        return doi, pmid, pmcid

    @staticmethod
    def _parse_year(
        source: Element, warnings: list[JatsParseWarning], anchor: str | None
    ) -> int | None:
        year_element = first_local(source, "year")
        if year_element is None:
            return None
        raw = normalize_text(element_text(year_element))
        if raw.isdigit() and 1000 <= int(raw) <= 2200:
            return int(raw)
        warnings.append(
            JatsParseWarning(
                code="invalid-reference-year",
                message=(
                    f"ignoring implausible reference year {raw!r}; "
                    "the raw reference text is preserved"
                ),
                source_anchor=anchor,
            )
        )
        return None

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _parse_tables(
        self, root: Element, anchors: AnchorIndex, parents: Mapping[Element, Element]
    ) -> tuple[ParsedTable, ...]:
        """Parse each ``table-wrap`` as one canonical table — never every
        descendant ``<table>`` separately."""
        tables: list[ParsedTable] = []
        ordinals = count(1)
        for element in root.iter():
            if local_name(element.tag) != "table-wrap":
                continue
            anchor = anchors.anchor_for(element)
            if anchor is None:
                continue
            label_element = first_local(element, "label")
            label = None
            if label_element is not None:
                normalized = normalize_text(element_text(label_element))
                if normalized:
                    label = normalized
            caption_element = first_local(element, "caption")
            caption = None
            if caption_element is not None:
                normalized = normalize_text(element_text(caption_element))
                if normalized:
                    caption = normalized
            structured = self._table_structure(element)
            tables.append(
                ParsedTable(
                    ordinal=next(ordinals),
                    source_anchor=anchor,
                    section_anchor=self._section_anchor_for(element, parents, anchors),
                    label=label,
                    caption=caption,
                    structured_representation=structured,
                    content_fingerprint=sha256_json(structured),
                )
            )
        return tuple(tables)

    @staticmethod
    def _table_structure(table_wrap: Element) -> dict[str, Any]:
        """Deterministic JSON representation of a textual table: thead/tbody/
        tfoot distinction, row order, cell order, header vs data cells,
        normalized cell text and span attributes. Image-only tables preserve
        the graphic locator instead of fabricating cells."""
        graphic = first_local(table_wrap, "graphic")
        graphic_locator = graphic.get(XLINK_HREF) if graphic is not None else None
        table_element = first_local(table_wrap, "table")
        if table_element is None:
            return {
                "image_only": True,
                "graphic_locator": graphic_locator,
                "head": None,
                "body": None,
                "foot": None,
            }
        head = JatsParser._rows(table_element, "thead")
        body = JatsParser._rows(table_element, "tbody")
        foot = JatsParser._rows(table_element, "tfoot")
        if head is None and body is None and foot is None:
            body = JatsParser._rows(table_element, None)
        return {
            "image_only": False,
            "graphic_locator": graphic_locator,
            "head": head,
            "body": body,
            "foot": foot,
        }

    @staticmethod
    def _rows(container: Element, section_tag: str | None) -> list[dict[str, Any]] | None:
        section = first_local(container, section_tag) if section_tag is not None else container
        if section is None:
            return None
        rows: list[dict[str, Any]] = []
        for child in section:
            if local_name(child.tag) != "tr":
                continue
            cells: list[dict[str, Any]] = []
            for cell in child:
                name = local_name(cell.tag)
                if name not in ("td", "th"):
                    continue
                cells.append(
                    {
                        "text": normalize_text(element_text(cell)),
                        "header": name == "th",
                        "rowspan": _span_value(cell.get("rowspan")),
                        "colspan": _span_value(cell.get("colspan")),
                    }
                )
            rows.append({"cells": cells})
        return rows

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _parse_figures(
        self, root: Element, anchors: AnchorIndex, parents: Mapping[Element, Element]
    ) -> tuple[ParsedFigure, ...]:
        """Parse each ``fig`` as one canonical figure. Asset locators are
        read from explicit ``xlink:href`` attributes; assets are never
        downloaded and no multimodal parsing happens."""
        figures: list[ParsedFigure] = []
        ordinals = count(1)
        for element in root.iter():
            if local_name(element.tag) != "fig":
                continue
            anchor = anchors.anchor_for(element)
            if anchor is None:
                continue
            label_element = first_local(element, "label")
            label = None
            if label_element is not None:
                normalized = normalize_text(element_text(label_element))
                if normalized:
                    label = normalized
            caption_element = first_local(element, "caption")
            caption = None
            if caption_element is not None:
                normalized = normalize_text(element_text(caption_element))
                if normalized:
                    caption = normalized
            asset_locator = None
            for child in element:
                if local_name(child.tag) in ("graphic", "media"):
                    href = child.get(XLINK_HREF)
                    if href:
                        asset_locator = href
                        break
            figures.append(
                ParsedFigure(
                    ordinal=next(ordinals),
                    source_anchor=anchor,
                    section_anchor=self._section_anchor_for(element, parents, anchors),
                    label=label,
                    caption=caption,
                    asset_locator=asset_locator,
                    content_fingerprint=digest(label or "", caption or "", asset_locator or ""),
                )
            )
        return tuple(figures)

    # ------------------------------------------------------------------
    # Content fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def _content_fingerprint(
        *,
        title: str,
        language: str,
        metadata: Mapping[str, Any],
        sections: tuple[ParsedSection, ...],
        paragraphs: tuple[ParsedParagraph, ...],
        citations: tuple[ParsedCitation, ...],
        tables: tuple[ParsedTable, ...],
        figures: tuple[ParsedFigure, ...],
    ) -> str:
        """SHA-256 over the canonical JSON serialization of the normalized
        scientific representation. Surrogate ids, database ids, timestamps
        and parser warnings never participate; a semantic text change alters
        the fingerprint, formatting-only XML whitespace does not."""
        payload = {
            "title": title,
            "language": language,
            "metadata": metadata,
            "sections": [
                {
                    "ordinal": section.ordinal,
                    "depth": section.depth,
                    "title": section.title,
                    "semantic_type": section.semantic_type,
                    "source_anchor": section.source_anchor,
                    "structural_path": section.structural_path,
                    "content_fingerprint": section.content_fingerprint,
                }
                for section in sections
            ],
            "paragraphs": [
                {
                    "ordinal": paragraph.ordinal,
                    "region": paragraph.region.value,
                    "source_anchor": paragraph.source_anchor,
                    "section_anchor": paragraph.section_anchor,
                    "text": paragraph.text,
                    "content_sha256": paragraph.content_sha256,
                }
                for paragraph in paragraphs
            ],
            "citations": [
                {
                    "ordinal": citation.ordinal,
                    "source_reference_id": citation.source_reference_id,
                    "source_anchor": citation.source_anchor,
                    "doi": citation.doi,
                    "pmid": citation.pmid,
                    "pmcid": citation.pmcid,
                    "title": citation.title,
                    "year": citation.year,
                    "raw_reference_text": citation.raw_reference_text,
                }
                for citation in citations
            ],
            "tables": [
                {
                    "ordinal": table.ordinal,
                    "source_anchor": table.source_anchor,
                    "section_anchor": table.section_anchor,
                    "label": table.label,
                    "caption": table.caption,
                    "structured_representation": table.structured_representation,
                    "content_fingerprint": table.content_fingerprint,
                }
                for table in tables
            ],
            "figures": [
                {
                    "ordinal": figure.ordinal,
                    "source_anchor": figure.source_anchor,
                    "section_anchor": figure.section_anchor,
                    "label": figure.label,
                    "caption": figure.caption,
                    "asset_locator": figure.asset_locator,
                    "content_fingerprint": figure.content_fingerprint,
                }
                for figure in figures
            ],
        }
        return sha256_json(payload)


def sha256_json(payload: object) -> str:
    """SHA-256 over the deterministic canonical JSON serialization of
    ``payload``: sorted keys, compact separators, UTF-8."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_text(parent: Element, tag: str) -> str | None:
    """Normalized text of the first direct child with the given local name,
    or ``None`` when absent or empty after normalization."""
    element = first_local(parent, tag)
    if element is None:
        return None
    value = normalize_text(element_text(element))
    return value or None


def _span_value(raw: str | None) -> int:
    """Parse a table span attribute; malformed values fall back to 1 (the
    source cell occupies exactly one grid position)."""
    if raw is None:
        return 1
    try:
        value = int(raw.strip())
    except ValueError:
        return 1
    return value if value >= 1 else 1
