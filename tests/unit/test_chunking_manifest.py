"""Unit tests for the deterministic passage manifest.

Infrastructure-free: the manifest is a frozen value with a canonical
serialization. The tests construct manifests directly.
"""

from __future__ import annotations

import hashlib
import re

from dynamisrag.chunking.config import (
    ALGORITHM_REVISION,
    MANIFEST_SCHEMA_REVISION,
    ChunkerConfig,
    chunker_revision,
    config_sha256,
)
from dynamisrag.chunking.manifest import (
    ManifestPassage,
    ManifestSourceSpan,
    PassageManifest,
    SectionMetadata,
)


def _section() -> SectionMetadata:
    return SectionMetadata(
        section_key="a" * 64,
        structural_path="1",
        source_anchor="jats:/body[1]/sec[1]",
        title="Introduction",
    )


def _span(
    source_order: int = 0,
    *,
    paragraph_key: str = "b" * 64,
    paragraph_source_anchor: str = "jats:/body[1]/p[1]",
    start_char: int = 0,
    end_char: int = 10,
) -> ManifestSourceSpan:
    return ManifestSourceSpan(
        source_order=source_order,
        paragraph_key=paragraph_key,
        paragraph_source_anchor=paragraph_source_anchor,
        start_char=start_char,
        end_char=end_char,
    )


def _passage(
    ordinal: int = 0,
    *,
    text: str = "A passage of text.",
    token_count: int = 3,
    section: SectionMetadata | None = None,
    spans: tuple[ManifestSourceSpan, ...] | None = None,
) -> ManifestPassage:
    return ManifestPassage(
        ordinal=ordinal,
        passage_key=f"{'c' * 63}{ordinal}",
        text=text,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=token_count,
        primary_source_anchor="jats:/body[1]/p[1]",
        section=section if section is not None else SectionMetadata(None, None, None, None),
        source_spans=spans if spans is not None else (_span(),),
    )


def _manifest(
    passages: tuple[ManifestPassage, ...] | None = None,
    *,
    config: ChunkerConfig | None = None,
) -> PassageManifest:
    config = config if config is not None else ChunkerConfig()
    return PassageManifest(
        schema_revision=MANIFEST_SCHEMA_REVISION,
        document_version_key="d" * 64,
        chunker_revision=chunker_revision(config),
        algorithm_revision=config.algorithm_revision,
        config_sha256=config_sha256(config),
        config=config,
        passages=passages if passages is not None else (_passage(),),
    )


def test_repeated_construction_produces_byte_identical_manifests() -> None:
    first = _manifest()
    second = _manifest()

    assert first.manifest_bytes == second.manifest_bytes
    assert first.manifest_sha256 == second.manifest_sha256


def test_manifest_sha256_is_the_bytes_digest() -> None:
    manifest = _manifest()

    assert manifest.manifest_sha256 == hashlib.sha256(manifest.manifest_bytes).hexdigest()


def test_semantic_text_change_changes_the_manifest() -> None:
    baseline = _manifest((_passage(text="Original text."),))
    changed = _manifest((_passage(text="Changed text."),))

    assert changed.manifest_bytes != baseline.manifest_bytes
    assert changed.manifest_sha256 != baseline.manifest_sha256


def test_config_change_changes_the_manifest() -> None:
    baseline = _manifest()
    changed = _manifest(config=ChunkerConfig(max_tokens=400))

    assert changed.manifest_bytes != baseline.manifest_bytes
    assert changed.config_sha256 != baseline.config_sha256


def test_manifest_excludes_timestamps_and_random_ids() -> None:
    manifest = _manifest(
        (
            _passage(
                section=_section(),
                spans=(_span(start_char=0, end_char=10), _span(1, start_char=11, end_char=20)),
            ),
        )
    )
    payload = manifest.manifest_bytes

    assert b"row_created_at" not in payload
    assert b"created_at" not in payload
    assert b"retrieved_at" not in payload
    # No hyphenated surrogate UUIDs anywhere in the manifest (the unhyphenated
    # hex digests are the deterministic semantic keys, which belong there).
    assert (
        re.search(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", payload) is None
    )


def test_manifest_carries_the_semantic_identity_fields() -> None:
    config = ChunkerConfig()
    manifest = _manifest(config=config)

    assert manifest.document_version_key == "d" * 64
    assert manifest.schema_revision == MANIFEST_SCHEMA_REVISION
    assert manifest.algorithm_revision == ALGORITHM_REVISION
    assert manifest.config_sha256 == config_sha256(config)
    assert manifest.chunker_revision == chunker_revision(config)
    assert manifest.config.target_tokens == 350


def test_manifest_passage_entries_carry_only_semantic_fields() -> None:
    manifest = _manifest(
        (
            _passage(
                ordinal=0,
                spans=(_span(0, start_char=0, end_char=5), _span(1, start_char=6, end_char=10)),
            ),
            _passage(
                ordinal=1,
                text="Second.",
                token_count=1,
                spans=(_span(0, start_char=0, end_char=7),),
            ),
        )
    )

    for passage in manifest.passages:
        assert passage.passage_key
        assert passage.text
        assert passage.content_sha256 == hashlib.sha256(passage.text.encode("utf-8")).hexdigest()
        assert passage.token_count > 0
        assert passage.primary_source_anchor
        for span in passage.source_spans:
            assert span.paragraph_key
            assert span.paragraph_source_anchor
            assert span.end_char > span.start_char >= 0


def test_manifest_section_metadata_is_semantic() -> None:
    manifest = _manifest(
        (
            _passage(section=_section()),
            _passage(ordinal=1, text="Sectionless."),
        )
    )

    section_passage, sectionless_passage = manifest.passages
    assert section_passage.section.section_key == "a" * 64
    assert section_passage.section.structural_path == "1"
    assert section_passage.section.source_anchor == "jats:/body[1]/sec[1]"
    assert section_passage.section.title == "Introduction"
    assert sectionless_passage.section.section_key is None
    assert sectionless_passage.section.structural_path is None
    assert sectionless_passage.section.title is None


def test_empty_manifest_is_valid() -> None:
    manifest = _manifest(())

    assert manifest.passages == ()
    assert manifest.manifest_bytes
    assert manifest.manifest_sha256 == hashlib.sha256(manifest.manifest_bytes).hexdigest()
