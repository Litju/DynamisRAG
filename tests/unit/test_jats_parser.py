"""Deterministic JATS parsing proven without database or network (RES-133).

The parser is a pure function of the source bytes: every test here runs
against small synthetic JATS fixtures and needs no infrastructure.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Final

import pytest

from dynamisrag.domain.contracts import SourceArtifact
from dynamisrag.domain.values import IdentifierNamespace, ParagraphRegion
from dynamisrag.jats import (
    JATS_NORMALIZER_REVISION,
    JATS_PARSER_REVISION,
    JatsCanonicalImporter,
    JatsMissingRequiredMetadata,
    JatsParseError,
    JatsParser,
    JatsSourceIntegrityError,
    JatsSourcePmcidConflict,
    ParsedJatsArticle,
)
from dynamisrag.jats.importer import candidate_identifiers
from dynamisrag.jats.text import normalize_language, normalize_text
from tests._support import JATS_FULL_ARTICLE as _FULL_ARTICLE
from tests._support import JATS_SPARSE_ARTICLE as _SPARSE_ARTICLE

_DEEP_SECTIONS: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Deep nesting</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <sec id="s1"><title>Level one</title>
      <sec id="s1-1"><title>Level two</title>
        <sec id="s1-1-1"><title>Level three</title>
          <sec id="s1-1-1-1"><title>Level four</title><p>Deep paragraph.</p></sec>
        </sec>
      </sec>
    </sec>
    <sec><p>Section without a title.</p></sec>
  </body>
</article>
"""

_DUPLICATE_IDS: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Duplicate ids</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <sec id="dup"><title>First owner of dup</title><p>Paragraph in first dup section.</p></sec>
    <sec id="dup"><title>Second owner of dup</title><p>Paragraph in second dup section.</p></sec>
    <sec><p>Paragraph with no id.</p></sec>
  </body>
</article>
"""

_IMAGE_ONLY_TABLE: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group><article-title>Image table</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <table-wrap id="T9">
      <label>Table 9</label>
      <caption>An image-only table.</caption>
      <graphic xlink:href="table9.png"/>
    </table-wrap>
  </body>
</article>
"""

_FIGURE_WITHOUT_GRAPHIC: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Bare figure</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <fig id="F0">
      <label>Figure 0</label>
      <caption>A figure without any graphic.</caption>
    </fig>
  </body>
</article>
"""

_MINIMAL_REFERENCE: Final[bytes] = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Minimal reference</article-title></title-group>
    </article-meta>
  </front>
  <body><p>Body.</p></body>
  <back>
    <ref-list>
      <ref><mixed-citation>Just free text, no identifiers at all.</mixed-citation></ref>
    </ref-list>
  </back>
</article>
"""


def _parse(xml: bytes) -> ParsedJatsArticle:
    return JatsParser().parse(xml)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_parser_is_deterministic() -> None:
    first = _parse(_FULL_ARTICLE)
    second = _parse(_FULL_ARTICLE)

    assert first == second
    assert first.content_fingerprint == second.content_fingerprint


def test_text_normalization_is_deterministic() -> None:
    assert normalize_text("  a \n\t b  c ") == "a b c"
    assert normalize_text("a <tag>b</tag> c") == "a <tag>b</tag> c"
    assert normalize_text("") == ""


def test_inline_markup_is_flattened_without_introducing_spaces() -> None:
    parsed = _parse(_FULL_ARTICLE)

    assert parsed.title == "A synthetic study of things and numbers"
    paragraph_texts = [paragraph.text for paragraph in parsed.paragraphs]
    assert "This is very important. It has two sentences." in paragraph_texts
    # <alternatives> contributes the single TeX representation, not a
    # duplication of every alternative form.
    assert "Second abstract paragraph with a^2 + b^2 math." in paragraph_texts


def test_paragraphs_preserve_inline_textual_order() -> None:
    xml = b"""<article><front><article-meta><title-group>
    <article-title>t</article-title></title-group></article-meta></front>
    <body><p>Prefix <italic>middle</italic> suffix.</p></body></article>"""
    parsed = _parse(xml)

    assert [paragraph.text for paragraph in parsed.paragraphs] == ["Prefix middle suffix."]


# ---------------------------------------------------------------------------
# Source anchors
# ---------------------------------------------------------------------------


def test_source_anchors_use_unique_xml_ids() -> None:
    parsed = _parse(_FULL_ARTICLE)

    assert parsed.sections[0].source_anchor == "jats:#sec1"
    assert parsed.sections[1].source_anchor == "jats:#sec1-1"
    assert parsed.sections[2].source_anchor == "jats:#sec2"
    assert parsed.citations[0].source_anchor == "jats:#R1"
    assert parsed.citations[1].source_anchor == "jats:#R2"
    assert parsed.tables[0].source_anchor == "jats:#T1"
    assert parsed.figures[0].source_anchor == "jats:#F1"


def test_source_anchors_are_unique_within_one_document() -> None:
    parsed = _parse(_FULL_ARTICLE)

    anchors = [section.source_anchor for section in parsed.sections]
    anchors += [paragraph.source_anchor for paragraph in parsed.paragraphs]
    anchors += [citation.source_anchor for citation in parsed.citations]
    anchors += [table.source_anchor for table in parsed.tables]
    anchors += [figure.source_anchor for figure in parsed.figures]

    assert len(anchors) == len(set(anchors))


def test_missing_xml_ids_fall_back_to_structural_paths() -> None:
    parsed = _parse(_SPARSE_ARTICLE)

    assert parsed.paragraphs[0].source_anchor == "jats:/body[1]/p[1]"


def test_duplicate_xml_ids_use_deterministic_path_fallback() -> None:
    parsed = _parse(_DUPLICATE_IDS)

    assert parsed.warnings[0].code == "duplicate-xml-id"
    first, second, third = parsed.sections
    # A duplicated @id is not globally unique: every occurrence — including
    # the first — uses its own structural-path anchor, never jats:#dup.
    assert first.source_anchor == "jats:/body[1]/sec[1]"
    assert second.source_anchor == "jats:/body[1]/sec[2]"
    assert third.source_anchor == "jats:/body[1]/sec[3]"
    assert all(section.source_anchor != "jats:#dup" for section in parsed.sections)
    # The fallback is deterministic across parses.
    reparsed = _parse(_DUPLICATE_IDS)
    assert [section.source_anchor for section in reparsed.sections] == [
        section.source_anchor for section in parsed.sections
    ]


def test_unique_xml_ids_still_use_the_jats_fragment_anchor() -> None:
    parsed = _parse(_DUPLICATE_IDS)

    # The third section has no @id at all, and the two duplicated sections
    # fall back to paths — but a genuinely unique @id elsewhere in the same
    # document still yields the jats:#id form.
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Unique ids</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <sec id="dup"><title>First dup</title><p>One.</p></sec>
    <sec id="dup"><title>Second dup</title><p>Two.</p></sec>
    <sec id="unique"><title>Unique</title><p>Three.</p></sec>
  </body>
</article>
"""
    parsed = _parse(xml)

    assert parsed.warnings[0].code == "duplicate-xml-id"
    first, second, third = parsed.sections
    assert first.source_anchor == "jats:/body[1]/sec[1]"
    assert second.source_anchor == "jats:/body[1]/sec[2]"
    assert third.source_anchor == "jats:#unique"


def test_structural_paths_trace_into_the_source_document() -> None:
    parsed = _parse(_FULL_ARTICLE)

    # body direct paragraph, nested section paragraph, list paragraph
    anchors = {paragraph.source_anchor for paragraph in parsed.paragraphs}
    assert "jats:/body[1]/p[1]" in anchors
    assert "jats:/body[1]/sec[1]/p[1]" in anchors
    assert "jats:/body[1]/sec[1]/sec[1]/p[1]" in anchors
    assert "jats:/body[1]/sec[1]/sec[1]/list[1]/list-item[1]/p[1]" in anchors


# ---------------------------------------------------------------------------
# Metadata, identifiers, language
# ---------------------------------------------------------------------------


def test_metadata_normalization_is_stable() -> None:
    first = _parse(_FULL_ARTICLE)
    second = _parse(_FULL_ARTICLE)

    assert first.metadata == second.metadata
    metadata = first.metadata
    assert metadata["article_type"] == "research-article"
    assert metadata["journal"]["title"] == "Journal of Synthetic Studies"
    assert metadata["journal"]["publisher"] == "Synthetic Press"
    assert metadata["journal"]["identifiers"] == [
        {"type": "print", "value": "1234-5678"},
        {"type": "electronic", "value": "8765-4321"},
    ]
    assert metadata["publication_dates"] == [
        {"type": "epub", "year": "2024", "month": "5", "day": "15"}
    ]
    assert metadata["volume"] == "19"
    assert metadata["issue"] == "5"
    assert metadata["elocation-id"] == "e03089012"
    assert metadata["raw_xml_lang"] == "en-US"
    assert metadata["jats_dtd_version"] == "1.3"
    assert metadata["keywords"] == ["synthetic", "testing"]


def test_contributors_and_affiliations_are_preserved() -> None:
    parsed = _parse(_FULL_ARTICLE)

    contributors = parsed.metadata["contributors"]
    assert contributors[0] == {
        "type": "author",
        "surname": "Smith",
        "given_names": "Jane A.",
        "orcid": "0000-0002-1825-0097",
        "affiliation_refs": ["aff1"],
    }
    assert contributors[1] == {"type": "author", "collab": "The Synthetic Consortium"}
    assert parsed.metadata["affiliations"] == [
        {
            "id": "aff1",
            "text": "Department of Synthetics, Example University, United States",
        }
    ]


def test_identifier_extraction_and_normalization() -> None:
    parsed = _parse(_FULL_ARTICLE)

    assert parsed.doi == "10.1371/journal.pone.03089012"
    assert parsed.pmid == "38888888"
    assert parsed.pmcid == "PMC123456"


def test_identifier_normalization_accepts_common_doi_spellings() -> None:
    xml = b"""<article><front><article-meta>
    <article-id pub-id-type="doi">https://doi.org/10.1371/JOURNAL.PONE.03089012</article-id>
    <title-group><article-title>t</article-title></title-group>
    </article-meta></front><body><p>b</p></body></article>"""
    parsed = _parse(xml)

    assert parsed.doi == "10.1371/journal.pone.03089012"


def test_malformed_optional_identifier_is_ignored_and_warns() -> None:
    xml = b"""<article><front><article-meta>
    <article-id pub-id-type="doi">not-a-doi</article-id>
    <article-id pub-id-type="pmid">12345678901</article-id>
    <title-group><article-title>t</article-title></title-group>
    </article-meta></front><body><p>b</p></body></article>"""
    parsed = _parse(xml)

    assert parsed.doi is None
    assert parsed.pmid is None
    assert [warning.code for warning in parsed.warnings] == [
        "malformed-identifier",
        "malformed-identifier",
    ]
    # The raw values remain in metadata for provenance.
    assert parsed.metadata["article_ids"] == [
        {"pub_id_type": "doi", "value": "not-a-doi"},
        {"pub_id_type": "pmid", "value": "12345678901"},
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en", "en"),
        ("en-US", "en"),
        ("pt-BR", "pt"),
        ("EN", "en"),
        ("de", "de"),
        ("und", "und"),
        (None, "und"),
        ("", "und"),
        ("x1", "und"),
        ("toolong", "und"),
    ],
)
def test_language_normalization(raw: str | None, expected: str) -> None:
    assert normalize_language(raw) == expected


def test_missing_language_normalizes_to_und() -> None:
    parsed = _parse(_SPARSE_ARTICLE)

    assert parsed.language == "und"
    assert "raw_xml_lang" not in parsed.metadata


def test_language_is_read_from_the_article_element() -> None:
    parsed = _parse(_FULL_ARTICLE)

    assert parsed.language == "en"
    assert parsed.metadata["raw_xml_lang"] == "en-US"


# ---------------------------------------------------------------------------
# Content fingerprint
# ---------------------------------------------------------------------------


def test_content_fingerprint_is_deterministic() -> None:
    first = _parse(_FULL_ARTICLE)
    second = _parse(_FULL_ARTICLE)

    assert first.content_fingerprint == second.content_fingerprint
    assert len(first.content_fingerprint) == 64


def test_semantic_text_change_alters_the_fingerprint() -> None:
    baseline = _parse(_FULL_ARTICLE)
    changed = _parse(_FULL_ARTICLE.replace(b"Intro paragraph one.", b"Intro paragraph two."))

    assert changed.content_fingerprint != baseline.content_fingerprint


def test_formatting_whitespace_change_does_not_alter_the_fingerprint() -> None:
    baseline = _parse(_FULL_ARTICLE)
    reformatted = _parse(
        _FULL_ARTICLE.replace(
            b"<p>Intro paragraph one.</p>", b"<p>\n        Intro paragraph one.\n    </p>"
        )
    )

    assert reformatted.content_fingerprint == baseline.content_fingerprint


def test_semantic_metadata_change_alters_the_fingerprint() -> None:
    baseline = _parse(_FULL_ARTICLE)
    changed = _parse(_FULL_ARTICLE.replace(b"<volume>19</volume>", b"<volume>20</volume>"))

    assert changed.content_fingerprint != baseline.content_fingerprint


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_section_hierarchy_paths_and_depths() -> None:
    parsed = _parse(_FULL_ARTICLE)

    assert [(s.structural_path, s.depth, s.title) for s in parsed.sections] == [
        ("1", 0, "Introduction"),
        ("1.1", 1, "Background"),
        ("2", 0, "Methods"),
    ]
    assert [s.ordinal for s in parsed.sections] == [1, 2, 3]
    assert parsed.sections[1].parent_anchor == parsed.sections[0].source_anchor
    assert parsed.sections[0].parent_anchor is None


def test_deeply_nested_sections_get_canonical_paths() -> None:
    parsed = _parse(_DEEP_SECTIONS)

    assert [(s.structural_path, s.depth) for s in parsed.sections] == [
        ("1", 0),
        ("1.1", 1),
        ("1.1.1", 2),
        ("1.1.1.1", 3),
        ("2", 0),
    ]
    # A section without a title is allowed; none is invented.
    assert parsed.sections[4].title is None


def test_semantic_type_is_preserved_from_sec_type() -> None:
    xml = b"""<article><front><article-meta><title-group>
    <article-title>t</article-title></title-group></article-meta></front>
    <body><sec sec-type="methods"><title>M</title><p>p</p></sec></body></article>"""
    parsed = _parse(xml)

    assert parsed.sections[0].semantic_type == "methods"


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------


def test_paragraph_ordering_regions_text_and_hash() -> None:
    parsed = _parse(_FULL_ARTICLE)

    paragraphs = parsed.paragraphs
    assert [paragraph.ordinal for paragraph in paragraphs] == list(range(1, len(paragraphs) + 1))
    assert [paragraph.region for paragraph in paragraphs[:2]] == [
        ParagraphRegion.FRONT,
        ParagraphRegion.FRONT,
    ]
    assert paragraphs[2].region is ParagraphRegion.BODY
    # Abstract paragraphs are front-region paragraphs, not a fake section.
    assert paragraphs[0].section_anchor is None
    # Body paragraphs point at their nearest owning section.
    assert paragraphs[3].section_anchor == "jats:#sec1"
    assert paragraphs[4].section_anchor == "jats:#sec1-1"
    for paragraph in paragraphs:
        assert (
            paragraph.content_sha256 == hashlib.sha256(paragraph.text.encode("utf-8")).hexdigest()
        )


def test_caption_and_reference_paragraphs_are_not_double_counted() -> None:
    parsed = _parse(_FULL_ARTICLE)

    texts = [paragraph.text for paragraph in parsed.paragraphs]
    assert "A synthetic figure caption." not in texts
    assert "A synthetic table." not in texts
    # The figure caption lives on the figure, the table caption on the table.
    assert parsed.figures[0].caption == "A synthetic figure caption."
    assert parsed.tables[0].caption == "A synthetic table."


def test_empty_paragraphs_are_skipped() -> None:
    xml = b"""<article><front><article-meta><title-group>
    <article-title>t</article-title></title-group></article-meta></front>
    <body><p>   </p><p>Real paragraph.</p></body></article>"""
    parsed = _parse(xml)

    assert [paragraph.text for paragraph in parsed.paragraphs] == ["Real paragraph."]


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_citation_extraction_from_mixed_and_element_citations() -> None:
    parsed = _parse(_FULL_ARTICLE)

    first, second = parsed.citations
    assert first.ordinal == 1
    assert first.source_reference_id == "R1"
    assert first.doi == "10.1016/j.cell.2020.01.001"
    assert first.pmid == "31900000"
    assert first.pmcid is None
    assert first.year is None
    assert "Smith J, 2020. A cited study." in first.raw_reference_text
    assert second.ordinal == 2
    assert second.source_reference_id == "R2"
    assert second.pmcid == "PMC7000000"
    assert second.title == "Another cited work"
    assert second.year == 2019


def test_implausible_reference_year_warns_and_preserves_raw_text() -> None:
    xml = b"""<article><front><article-meta><title-group>
    <article-title>t</article-title></title-group></article-meta></front>
    <body><p>b</p></body>
    <back><ref-list><ref id="R9"><element-citation>
    <article-title>Weird year</article-title><year>9999</year>
    </element-citation></ref></ref-list></back></article>"""
    parsed = _parse(xml)

    assert parsed.citations[0].year is None
    # No space is introduced between elements that have none in the source.
    assert parsed.citations[0].raw_reference_text == "Weird year9999"
    assert parsed.warnings[0].code == "invalid-reference-year"


def test_unresolved_citation_with_minimal_data() -> None:
    parsed = _parse(_MINIMAL_REFERENCE)

    citation = parsed.citations[0]
    assert citation.source_reference_id is None
    assert citation.doi is None
    assert citation.pmid is None
    assert citation.pmcid is None
    assert citation.title is None
    assert citation.year is None
    assert citation.raw_reference_text == "Just free text, no identifiers at all."


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_table_structured_representation() -> None:
    parsed = _parse(_FULL_ARTICLE)

    table = parsed.tables[0]
    assert table.label == "Table 1"
    assert table.caption == "A synthetic table."
    assert table.section_anchor == "jats:#sec2"
    structured = table.structured_representation
    assert structured["image_only"] is False
    assert structured["head"] == [
        {
            "cells": [
                {"text": "Group", "header": True, "rowspan": 2, "colspan": 1},
                {"text": "Values", "header": True, "rowspan": 1, "colspan": 2},
            ]
        }
    ]
    assert structured["body"] == [
        {
            "cells": [
                {"text": "Control", "header": False, "rowspan": 1, "colspan": 1},
                {"text": "1", "header": False, "rowspan": 1, "colspan": 1},
                {"text": "2", "header": False, "rowspan": 1, "colspan": 1},
            ]
        },
        {
            "cells": [
                {"text": "Treated", "header": False, "rowspan": 1, "colspan": 1},
                {"text": "3", "header": False, "rowspan": 1, "colspan": 1},
                {"text": "4", "header": False, "rowspan": 1, "colspan": 1},
            ]
        },
    ]
    assert structured["foot"] is None


def test_image_only_table_preserves_graphic_locator_without_fabricating_cells() -> None:
    parsed = _parse(_IMAGE_ONLY_TABLE)

    table = parsed.tables[0]
    structured = table.structured_representation
    assert structured["image_only"] is True
    assert structured["graphic_locator"] == "table9.png"
    assert structured["head"] is None
    assert structured["body"] is None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_figure_extraction() -> None:
    parsed = _parse(_FULL_ARTICLE)

    figure = parsed.figures[0]
    assert figure.label == "Figure 1"
    assert figure.caption == "A synthetic figure caption."
    assert figure.asset_locator == "figure1.png"
    assert figure.section_anchor == "jats:#sec2"


def test_figure_without_graphic_has_no_asset_locator() -> None:
    parsed = _parse(_FIGURE_WITHOUT_GRAPHIC)

    figure = parsed.figures[0]
    assert figure.label == "Figure 0"
    assert figure.caption == "A figure without any graphic."
    assert figure.asset_locator is None


# ---------------------------------------------------------------------------
# Sparse article
# ---------------------------------------------------------------------------


def test_sparse_article_behavior() -> None:
    parsed = _parse(_SPARSE_ARTICLE)

    assert parsed.title == "A sparse article"
    assert parsed.language == "und"
    assert parsed.doi is None
    assert parsed.pmid is None
    assert parsed.pmcid == "PMC22222222"
    assert parsed.sections == ()
    assert len(parsed.paragraphs) == 1
    assert parsed.paragraphs[0].text == "Only paragraph."
    assert parsed.citations == ()
    assert parsed.tables == ()
    assert parsed.figures == ()
    assert parsed.warnings == ()


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_malformed_xml_raises_a_narrow_parse_error() -> None:
    with pytest.raises(JatsParseError, match="not well-formed XML"):
        _parse(b"<article><front>")


def test_non_article_root_raises_a_narrow_parse_error() -> None:
    with pytest.raises(JatsParseError, match="article"):
        _parse(
            b"<not-an-article><title-group><article-title>t</article-title></title-group></not-an-article>"
        )


def test_missing_title_raises_a_missing_required_metadata_error() -> None:
    with pytest.raises(JatsMissingRequiredMetadata, match="title"):
        _parse(
            b"<article><front><article-meta></article-meta></front><body><p>b</p></body></article>"
        )


def test_empty_title_raises_a_missing_required_metadata_error() -> None:
    with pytest.raises(JatsMissingRequiredMetadata, match="title"):
        _parse(
            b"<article><front><article-meta><title-group><article-title>  </article-title>"
            b"</title-group></article-meta></front><body><p>b</p></body></article>"
        )


def test_forbidden_xml_constructs_are_rejected() -> None:
    with pytest.raises(JatsParseError):
        _parse(
            b'<?xml version="1.0"?><!DOCTYPE article [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b"<article><front><article-meta><title-group><article-title>&xxe;</article-title>"
            b"</title-group></article-meta></front><body><p>b</p></body></article>"
        )


# ---------------------------------------------------------------------------
# Source integrity (importer-level, proven without a database)
# ---------------------------------------------------------------------------


def _artifact(content: bytes, **overrides: object) -> SourceArtifact:
    values: dict[str, object] = {
        "source_system": "europe_pmc",
        "source_external_id": "PMC123456",
        "source_uri": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fulltextXML",
        "media_type": "application/xml",
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
        "retrieved_at": datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC),
        "storage_uri": "file:///dynamisrag-artifacts/sha256/PMC123456.xml",
    }
    values.update(overrides)
    return SourceArtifact(**values)  # type: ignore[arg-type]


def test_source_integrity_mismatch_is_rejected_before_parsing() -> None:
    artifact = _artifact(_FULL_ARTICLE, content_sha256="0" * 64)

    with pytest.raises(JatsSourceIntegrityError, match="do not match"):
        JatsCanonicalImporter.parse_verified(artifact, _FULL_ARTICLE)


def test_source_integrity_byte_size_mismatch_is_rejected() -> None:
    artifact = _artifact(_FULL_ARTICLE, byte_size=1)

    with pytest.raises(JatsSourceIntegrityError, match="do not match"):
        JatsCanonicalImporter.parse_verified(artifact, _FULL_ARTICLE)


def test_parser_does_not_mutate_source_bytes() -> None:
    xml = bytearray(_FULL_ARTICLE)
    _parse(bytes(xml))

    assert bytes(xml) == _FULL_ARTICLE


# ---------------------------------------------------------------------------
# Europe PMC source PMCID provenance (importer-level, proven without a database)
# ---------------------------------------------------------------------------


def test_source_pmcid_mismatch_is_rejected_before_materialization() -> None:
    """An explicit XML PMCID that differs from the acquired artifact PMCID is
    a fatal source/identity conflict — never two aliases of one Document."""
    xml = _FULL_ARTICLE.replace(
        b'<article-id pub-id-type="pmcid">PMC123456</article-id>',
        b'<article-id pub-id-type="pmcid">PMC999999</article-id>',
    )
    artifact = _artifact(xml)
    parsed = _parse(xml)
    assert parsed.pmcid == "PMC999999"

    with pytest.raises(JatsSourcePmcidConflict, match="PMC999999"):
        JatsCanonicalImporter.validate_source_pmcid(artifact, parsed)


def test_source_pmcid_match_is_accepted() -> None:
    artifact = _artifact(_FULL_ARTICLE)
    parsed = _parse(_FULL_ARTICLE)

    JatsCanonicalImporter.validate_source_pmcid(artifact, parsed)


def test_source_pmcid_omitted_from_xml_is_accepted() -> None:
    """The artifact PMCID remains a known identifier when the XML omits it."""
    xml = _FULL_ARTICLE.replace(
        b'<article-id pub-id-type="pmcid">PMC123456</article-id>\n      ', b""
    )
    artifact = _artifact(xml)
    parsed = _parse(xml)
    assert parsed.pmcid is None

    JatsCanonicalImporter.validate_source_pmcid(artifact, parsed)


def test_source_pmcid_validation_ignores_non_europe_pmc_artifacts() -> None:
    artifact = _artifact(_FULL_ARTICLE, source_system="arxiv")
    parsed = _parse(_FULL_ARTICLE)

    JatsCanonicalImporter.validate_source_pmcid(artifact, parsed)


# ---------------------------------------------------------------------------
# Parser/normalizer revisions and candidate identifiers
# ---------------------------------------------------------------------------


def test_parser_revisions_are_explicit_and_valid() -> None:
    assert JATS_PARSER_REVISION == "jats-1.0"
    assert JATS_NORMALIZER_REVISION == "norm-1.0"


def test_artifact_pmcid_is_a_known_identifier_even_when_xml_omits_it() -> None:
    artifact = _artifact(_SPARSE_ARTICLE)
    parsed = _parse(_SPARSE_ARTICLE)

    candidates = candidate_identifiers(artifact, parsed)

    assert (IdentifierNamespace.PMCID, "PMC22222222") in candidates


def test_candidate_identifiers_include_parsed_identifiers() -> None:
    artifact = _artifact(_FULL_ARTICLE)
    parsed = _parse(_FULL_ARTICLE)

    candidates = candidate_identifiers(artifact, parsed)
    assert (IdentifierNamespace.DOI, "10.1371/journal.pone.03089012") in candidates
    assert (IdentifierNamespace.PMID, "38888888") in candidates
    assert (IdentifierNamespace.PMCID, "PMC123456") in candidates
