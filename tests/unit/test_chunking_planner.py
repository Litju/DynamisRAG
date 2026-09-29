"""Unit tests for the structure-aware passage planner.

Infrastructure-free: the planner is a pure function of the canonical source
structure and the frozen chunker configuration.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

from dynamisrag.chunking.config import ChunkerConfig
from dynamisrag.chunking.errors import ChunkingError
from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.domain.contracts import DocumentVersion, Paragraph, Section
from dynamisrag.domain.identity import passage_key
from dynamisrag.domain.values import ParagraphRegion


def _make_version(*, source_artifact_key: str = "a" * 64) -> DocumentVersion:
    return DocumentVersion(
        document_id=uuid4(),
        document_canonical_key="doi:10.1038/nature12373",
        source_artifact_id=uuid4(),
        source_artifact_key=source_artifact_key,
        parser_revision="jats-1.0",
        normalizer_revision="norm-1.0",
        content_fingerprint="b" * 64,
        title="A study",
        language="en",
        created_at=datetime(2026, 9, 29, tzinfo=UTC),
    )


def _make_section(
    version: DocumentVersion,
    structural_path: str = "1",
    *,
    title: str | None = None,
    source_anchor: str | None = None,
) -> Section:
    return Section(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=int(structural_path.rsplit(".", 1)[-1]),
        depth=structural_path.count("."),
        title=title,
        source_anchor=source_anchor,
        structural_path=structural_path,
    )


def _make_paragraph(
    version: DocumentVersion,
    ordinal: int,
    text: str,
    *,
    section: Section | None = None,
    region: ParagraphRegion = ParagraphRegion.BODY,
    source_anchor: str | None = None,
) -> Paragraph:
    return Paragraph(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=ordinal,
        region=region,
        source_anchor=source_anchor or f"jats:/body[1]/p[{ordinal}]",
        text=text,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        section_id=section.id if section is not None else None,
    )


def _text_with_tokens(count: int, *, start: int = 0) -> str:
    return " ".join(f"w{index}" for index in range(start, start + count))


def _small_config() -> ChunkerConfig:
    return ChunkerConfig(target_tokens=4, max_tokens=6, min_tokens=2)


# ---------------------------------------------------------------------------
# Structural grouping
# ---------------------------------------------------------------------------


def test_same_section_adjacent_paragraphs_group_into_one_passage() -> None:
    version = _make_version()
    section = _make_section(version, "1")
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(1), section=section),
        _make_paragraph(version, 1, _text_with_tokens(1), section=section),
        _make_paragraph(version, 2, _text_with_tokens(1), section=section),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [section], paragraphs)

    assert len(manifest.passages) == 1
    passage = manifest.passages[0]
    assert len(passage.source_spans) == 3
    assert passage.section.section_key == section.section_key
    assert passage.section.structural_path == "1"
    assert passage.section.title == section.title
    assert passage.section.source_anchor == section.source_anchor


def test_different_sections_never_group() -> None:
    version = _make_version()
    first = _make_section(version, "1")
    second = _make_section(version, "2")
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(2), section=first),
        _make_paragraph(version, 1, _text_with_tokens(2), section=second),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [first, second], paragraphs)

    assert len(manifest.passages) == 2
    assert manifest.passages[0].section.section_key == first.section_key
    assert manifest.passages[1].section.section_key == second.section_key


def test_nested_section_boundary_starts_a_new_group() -> None:
    version = _make_version()
    parent = _make_section(version, "1")
    child = _make_section(version, "1.1")
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(2), section=parent),
        _make_paragraph(version, 1, _text_with_tokens(2), section=child),
        _make_paragraph(version, 2, _text_with_tokens(2), section=parent),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [parent, child], paragraphs)

    assert [passage.section.structural_path for passage in manifest.passages] == [
        "1",
        "1.1",
        "1",
    ]


def test_front_body_back_boundaries_never_merge() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(2), region=ParagraphRegion.FRONT),
        _make_paragraph(version, 1, _text_with_tokens(2), region=ParagraphRegion.BODY),
        _make_paragraph(version, 2, _text_with_tokens(2), region=ParagraphRegion.BACK),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert len(manifest.passages) == 3
    assert all(passage.section.section_key is None for passage in manifest.passages)


def test_sectionless_paragraphs_pack_with_adjacent_sectionless_paragraphs() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(2)),
        _make_paragraph(version, 1, _text_with_tokens(2)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert len(manifest.passages) == 1
    assert len(manifest.passages[0].source_spans) == 2
    section = manifest.passages[0].section
    assert section.section_key is None
    assert section.structural_path is None
    assert section.source_anchor is None
    assert section.title is None


def test_sectionless_paragraph_does_not_merge_with_a_sectioned_one() -> None:
    version = _make_version()
    section = _make_section(version, "1")
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(2)),
        _make_paragraph(version, 1, _text_with_tokens(2), section=section),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [section], paragraphs)

    assert len(manifest.passages) == 2
    assert manifest.passages[0].section.section_key is None
    assert manifest.passages[1].section.section_key == section.section_key


def test_paragraphs_are_processed_in_ordinal_order() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 2, _text_with_tokens(1)),
        _make_paragraph(version, 0, _text_with_tokens(1)),
        _make_paragraph(version, 1, _text_with_tokens(1)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert [span.paragraph_source_anchor for span in manifest.passages[0].source_spans] == [
        "jats:/body[1]/p[0]",
        "jats:/body[1]/p[1]",
        "jats:/body[1]/p[2]",
    ]


def test_input_from_another_version_is_rejected() -> None:
    version = _make_version()
    other = _make_version(source_artifact_key="c" * 64)
    paragraph = _make_paragraph(other, 0, _text_with_tokens(2))

    chunker = StructureAwareChunker(_small_config())
    try:
        chunker.plan(version, [], [paragraph])
    except ChunkingError as error:
        assert "belongs to version" in str(error)
    else:
        raise AssertionError("expected ChunkingError")


def test_paragraph_with_missing_section_is_rejected() -> None:
    """A section-owned paragraph whose section is not among the supplied
    same-version sections is an explicit failure — never silently
    downgraded to sectionless passage metadata."""
    version = _make_version()
    section = _make_section(version, "1")
    paragraph = _make_paragraph(version, 0, _text_with_tokens(2), section=section)

    chunker = StructureAwareChunker(_small_config())
    try:
        chunker.plan(version, [], [paragraph])
    except ChunkingError as error:
        assert "not among the supplied sections" in str(error)
    else:
        raise AssertionError("expected ChunkingError")


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def test_packing_flushes_at_target_before_adding_an_independent_unit() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(3)),
        _make_paragraph(version, 1, _text_with_tokens(3)),
        _make_paragraph(version, 2, _text_with_tokens(3)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert [passage.token_count for passage in manifest.passages] == [6, 3]


def test_hard_max_is_never_exceeded() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(4)),
        _make_paragraph(version, 1, _text_with_tokens(4)),
        _make_paragraph(version, 2, _text_with_tokens(4)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert [passage.token_count for passage in manifest.passages] == [4, 4, 4]
    assert all(passage.token_count <= 6 for passage in manifest.passages)


def test_small_tail_merges_backward_within_the_group() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(5)),
        _make_paragraph(version, 1, _text_with_tokens(1)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert len(manifest.passages) == 1
    assert manifest.passages[0].token_count == 6


def test_short_final_passage_is_retained_when_merge_would_exceed_max() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(5)),
        _make_paragraph(version, 1, _text_with_tokens(2)),
    ]
    config = ChunkerConfig(target_tokens=4, max_tokens=6, min_tokens=3)

    manifest = StructureAwareChunker(config).plan(version, [], paragraphs)

    assert [passage.token_count for passage in manifest.passages] == [5, 2]


def test_oversized_paragraph_is_sentence_split() -> None:
    version = _make_version()
    text = (
        f"{_text_with_tokens(3)}. {_text_with_tokens(3, start=3)}. {_text_with_tokens(3, start=6)}."
    )
    paragraph = _make_paragraph(version, 0, text)

    manifest = StructureAwareChunker(_small_config()).plan(version, [], [paragraph])

    assert [passage.token_count for passage in manifest.passages] == [4, 4, 4]
    assert [passage.text for passage in manifest.passages] == [
        "w0 w1 w2.",
        "w3 w4 w5.",
        "w6 w7 w8.",
    ]


def test_single_huge_sentence_splits_at_token_boundaries() -> None:
    version = _make_version()
    paragraph = _make_paragraph(version, 0, _text_with_tokens(7) + ".")

    manifest = StructureAwareChunker(_small_config()).plan(version, [], [paragraph])

    assert [passage.token_count for passage in manifest.passages] == [6, 2]
    assert " ".join(passage.text for passage in manifest.passages) == _text_with_tokens(7) + "."


def test_hard_split_keeps_terminal_punctuation_with_its_word() -> None:
    """The max boundary lands immediately before the terminal period: the
    period stays with its word instead of becoming a punctuation-only
    passage."""
    version = _make_version()
    paragraph = _make_paragraph(version, 0, "one two.")
    config = ChunkerConfig(target_tokens=2, max_tokens=2, min_tokens=1)

    manifest = StructureAwareChunker(config).plan(version, [], [paragraph])

    assert [passage.text for passage in manifest.passages] == ["one", "two."]


def test_oversized_paragraph_fragments_reconstruct_the_exact_source_substring() -> None:
    """Same-paragraph fragments are reconstructed from the exact canonical
    Paragraph.text substring between the first fragment's start and the last
    fragment's end — never a synthetic fixed-space join."""
    version = _make_version()
    text = "w0 w1 w2.  w3 w4 w5. w6 w7 w8."
    paragraph = _make_paragraph(version, 0, text)
    config = ChunkerConfig(target_tokens=10, max_tokens=10, min_tokens=2)

    manifest = StructureAwareChunker(config).plan(version, [], [paragraph])

    assert len(manifest.passages) == 2
    first = manifest.passages[0]
    assert len(first.source_spans) == 2
    assert first.text == "w0 w1 w2.  w3 w4 w5."
    assert first.text == text[first.source_spans[0].start_char : first.source_spans[-1].end_char]
    assert "  " in first.text


def test_oversized_paragraph_punctuation_stays_attached_to_its_cluster() -> None:
    version = _make_version()
    text = "alpha, beta gamma. delta, epsilon zeta. eta, theta iota."
    paragraph = _make_paragraph(version, 0, text)
    config = ChunkerConfig(target_tokens=13, max_tokens=13, min_tokens=2)

    manifest = StructureAwareChunker(config).plan(version, [], [paragraph])

    assert len(manifest.passages) == 2
    first = manifest.passages[0]
    assert [text[span.start_char : span.end_char] for span in first.source_spans] == [
        "alpha, beta gamma.",
        "delta, epsilon zeta.",
    ]
    assert first.text == "alpha, beta gamma. delta, epsilon zeta."
    second = manifest.passages[1]
    assert [text[span.start_char : span.end_char] for span in second.source_spans] == [
        "eta, theta iota.",
    ]


def test_oversized_paragraph_without_sentence_splitting_is_rejected() -> None:
    version = _make_version()
    paragraph = _make_paragraph(version, 0, _text_with_tokens(10))
    config = ChunkerConfig(
        target_tokens=4, max_tokens=6, min_tokens=2, split_long_paragraphs_by_sentence=False
    )

    chunker = StructureAwareChunker(config)
    try:
        chunker.plan(version, [], [paragraph])
    except ChunkingError as error:
        assert "split_long_paragraphs_by_sentence is disabled" in str(error)
    else:
        raise AssertionError("expected ChunkingError")


def test_spans_do_not_overlap_and_preserve_source_order() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(3)),
        _make_paragraph(version, 1, _text_with_tokens(3)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    for passage in manifest.passages:
        for first, second in zip(passage.source_spans, passage.source_spans[1:], strict=False):
            if first.paragraph_key == second.paragraph_key:
                assert first.end_char <= second.start_char
            assert first.source_order < second.source_order


def test_multiple_paragraphs_are_joined_with_a_blank_line() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, "First paragraph."),
        _make_paragraph(version, 1, "Second paragraph."),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert len(manifest.passages) == 1
    assert manifest.passages[0].text == "First paragraph.\n\nSecond paragraph."


def test_paragraph_text_is_never_rewritten() -> None:
    version = _make_version()
    text = "A  sentence   with  odd    spacing."
    paragraph = _make_paragraph(version, 0, text)

    manifest = StructureAwareChunker(_small_config()).plan(version, [], [paragraph])

    assert manifest.passages[0].text == text
    span = manifest.passages[0].source_spans[0]
    assert (span.start_char, span.end_char) == (0, len(text))


# ---------------------------------------------------------------------------
# Passage identity and text construction
# ---------------------------------------------------------------------------


def test_ordinals_are_zero_based_document_order() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(4)),
        _make_paragraph(version, 1, _text_with_tokens(4)),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert [passage.ordinal for passage in manifest.passages] == [0, 1]


def test_passage_key_binds_version_revision_and_ordinal() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, _text_with_tokens(4)),
        _make_paragraph(version, 1, _text_with_tokens(4)),
    ]
    chunker = StructureAwareChunker(_small_config())

    manifest = chunker.plan(version, [], paragraphs)

    for passage in manifest.passages:
        assert passage.passage_key == passage_key(
            version.version_key, chunker.chunker_revision, passage.ordinal
        )


def test_content_sha256_is_the_text_digest() -> None:
    version = _make_version()
    paragraph = _make_paragraph(version, 0, "Some text.")

    manifest = StructureAwareChunker(_small_config()).plan(version, [], [paragraph])

    assert manifest.passages[0].content_sha256 == hashlib.sha256(b"Some text.").hexdigest()


def test_primary_source_anchor_is_the_first_span_paragraph_anchor() -> None:
    version = _make_version()
    paragraphs = [
        _make_paragraph(version, 0, "First."),
        _make_paragraph(version, 1, "Second."),
    ]

    manifest = StructureAwareChunker(_small_config()).plan(version, [], paragraphs)

    assert manifest.passages[0].primary_source_anchor == paragraphs[0].source_anchor


def test_empty_document_produces_an_empty_manifest() -> None:
    version = _make_version()

    manifest = StructureAwareChunker(_small_config()).plan(version, [], [])

    assert manifest.passages == ()
