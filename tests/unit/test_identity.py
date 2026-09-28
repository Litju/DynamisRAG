"""The deterministic identity primitives must be pure and total.

These tests are infrastructure-free: identity is a pure function of its
inputs, so the whole contract can be proven without a database.

Every entity key derives from semantic parent keys — never from surrogate
database ids — so these tests compose the pure functions exactly the way the
contracts do: an artifact key feeds a document version key, which feeds the
section/passage/citation/table/figure keys below it.
"""

from __future__ import annotations

import re

import pytest

from dynamisrag.domain.identity import (
    citation_key,
    digest,
    document_canonical_key,
    document_table_key,
    document_version_key,
    figure_key,
    normalize_doi,
    passage_key,
    section_key,
    source_artifact_key,
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

_CONTENT_SHA = "a" * 64
_ARTIFACT_KEY = source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA)
_DOCUMENT_KEY = document_canonical_key(doi="10.1038/nature12373", pmid=None, pmcid=None, title=None)
_VERSION_KEY = document_version_key(
    _DOCUMENT_KEY, _ARTIFACT_KEY, "jats-1.2", "norm-v3", _CONTENT_SHA
)


def test_digest_is_lowercase_sha256_hex() -> None:
    value = digest("part-a", "part-b")

    assert _SHA256_HEX.fullmatch(value)


def test_digest_is_deterministic() -> None:
    assert digest("a", "b", "c") == digest("a", "b", "c")


def test_digest_changes_when_any_part_changes() -> None:
    baseline = digest("a", "b", "c")

    assert baseline != digest("x", "b", "c")
    assert baseline != digest("a", "x", "c")
    assert baseline != digest("a", "b", "x")
    assert baseline != digest("a", "b")
    assert baseline != digest("a", "b", "c", "d")


def test_digest_separates_parts_unambiguously() -> None:
    """The unit separator must keep ('ab','c') distinct from ('a','bc')."""
    assert digest("ab", "c") != digest("a", "bc")


def test_normalize_doi_strips_resolver_prefixes_and_lowercases() -> None:
    assert normalize_doi("10.1038/nature12373") == "10.1038/nature12373"
    assert normalize_doi("doi:10.1038/Nature12373") == "10.1038/nature12373"
    assert normalize_doi("https://doi.org/10.1038/Nature12373") == "10.1038/nature12373"
    assert normalize_doi(" http://dx.doi.org/10.1038/nature12373 ") == "10.1038/nature12373"


def test_normalize_doi_passes_none_through() -> None:
    assert normalize_doi(None) is None


def test_source_artifact_key_is_deterministic() -> None:
    first = source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA)
    second = source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA)

    assert first == second
    assert _SHA256_HEX.fullmatch(first)


def test_source_artifact_key_changes_with_each_identity_input() -> None:
    baseline = source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA)

    assert baseline != source_artifact_key("other_source", "PMC123456", _CONTENT_SHA)
    assert baseline != source_artifact_key("europe_pmc", "PMC654321", _CONTENT_SHA)
    assert baseline != source_artifact_key("europe_pmc", "PMC123456", "b" * 64)


def test_document_canonical_key_prefers_doi_over_everything_else() -> None:
    key = document_canonical_key(
        doi="10.1038/nature12373", pmid="12345", pmcid="PMC123456", title="A title"
    )

    assert key == "doi:10.1038/nature12373"


def test_document_canonical_key_normalizes_doi_forms_to_one_identity() -> None:
    bare = document_canonical_key(doi="10.1038/nature12373", pmid=None, pmcid=None, title=None)
    prefixed = document_canonical_key(
        doi="doi:10.1038/Nature12373", pmid=None, pmcid=None, title=None
    )
    resolved = document_canonical_key(
        doi="https://doi.org/10.1038/NATURE12373", pmid=None, pmcid=None, title=None
    )

    assert bare == prefixed == resolved


def test_document_canonical_key_falls_back_pmid_pmcid_then_title() -> None:
    assert (
        document_canonical_key(doi=None, pmid="12345", pmcid="PMC123456", title="T") == "pmid:12345"
    )
    assert (
        document_canonical_key(doi=None, pmid=None, pmcid="PMC123456", title="T")
        == "pmcid:PMC123456"
    )
    title_key = document_canonical_key(doi=None, pmid=None, pmcid=None, title="A title")
    assert title_key.startswith("title:")
    assert _SHA256_HEX.fullmatch(title_key.removeprefix("title:"))


def test_document_canonical_key_changes_with_title() -> None:
    first = document_canonical_key(doi=None, pmid=None, pmcid=None, title="Title one")
    second = document_canonical_key(doi=None, pmid=None, pmcid=None, title="Title two")

    assert first != second


def test_document_canonical_key_rejects_a_work_with_no_identity() -> None:
    with pytest.raises(ValueError, match="doi, pmid, pmcid, title"):
        document_canonical_key(doi=None, pmid=None, pmcid=None, title=None)


def test_document_version_key_is_deterministic() -> None:
    first = document_version_key(_DOCUMENT_KEY, _ARTIFACT_KEY, "parser-1", "norm-1", _CONTENT_SHA)
    second = document_version_key(_DOCUMENT_KEY, _ARTIFACT_KEY, "parser-1", "norm-1", _CONTENT_SHA)

    assert first == second
    assert _SHA256_HEX.fullmatch(first)


def test_document_version_key_changes_with_each_identity_input() -> None:
    baseline = document_version_key(
        _DOCUMENT_KEY, _ARTIFACT_KEY, "parser-1", "norm-1", _CONTENT_SHA
    )

    assert baseline != document_version_key(
        "pmid:12345", _ARTIFACT_KEY, "parser-1", "norm-1", _CONTENT_SHA
    )
    assert baseline != document_version_key(
        _DOCUMENT_KEY, "b" * 64, "parser-1", "norm-1", _CONTENT_SHA
    )
    assert baseline != document_version_key(
        _DOCUMENT_KEY, _ARTIFACT_KEY, "parser-2", "norm-1", _CONTENT_SHA
    )
    assert baseline != document_version_key(
        _DOCUMENT_KEY, _ARTIFACT_KEY, "parser-1", "norm-2", _CONTENT_SHA
    )
    assert baseline != document_version_key(
        _DOCUMENT_KEY, _ARTIFACT_KEY, "parser-1", "norm-1", "c" * 64
    )


def test_document_version_key_ignores_nothing_but_semantic_parents() -> None:
    """Surrogate ids are not inputs at all: only the parent keys feed the digest.

    Two independently computed views of the same logical document — different
    database, different insertion run, different surrogate ids — carry the
    same semantic parents and therefore the same version key.
    """
    rebuilt = document_version_key(
        document_canonical_key(doi="10.1038/nature12373", pmid=None, pmcid=None, title=None),
        source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA),
        "jats-1.2",
        "norm-v3",
        _CONTENT_SHA,
    )

    assert rebuilt == _VERSION_KEY


def test_section_key_is_deterministic_and_path_sensitive() -> None:
    first = section_key(_VERSION_KEY, "1.2.3")
    second = section_key(_VERSION_KEY, "1.2.3")

    assert first == second
    assert _SHA256_HEX.fullmatch(first)
    assert first != section_key(_VERSION_KEY, "1.2.4")
    assert first != section_key(
        document_version_key(_DOCUMENT_KEY, _ARTIFACT_KEY, "jats-1.3", "norm-v3", _CONTENT_SHA),
        "1.2.3",
    )


def test_passage_key_is_deterministic_and_input_sensitive() -> None:
    first = passage_key(_VERSION_KEY, "chunker-1", 7)
    second = passage_key(_VERSION_KEY, "chunker-1", 7)

    assert first == second
    assert _SHA256_HEX.fullmatch(first)
    assert first != passage_key(_VERSION_KEY, "chunker-2", 7)
    assert first != passage_key(_VERSION_KEY, "chunker-1", 8)
    assert first != passage_key("b" * 64, "chunker-1", 7)


def test_citation_key_is_deterministic_and_input_sensitive() -> None:
    first = citation_key(_VERSION_KEY, 3, "ref-3", "Smith et al., 2020")
    second = citation_key(_VERSION_KEY, 3, "ref-3", "Smith et al., 2020")

    assert first == second
    assert _SHA256_HEX.fullmatch(first)
    assert first != citation_key(_VERSION_KEY, 4, "ref-3", "Smith et al., 2020")
    assert first != citation_key(_VERSION_KEY, 3, "ref-4", "Smith et al., 2020")
    assert first != citation_key(_VERSION_KEY, 3, "ref-3", "Doe et al., 2021")
    assert first != citation_key("b" * 64, 3, "ref-3", "Smith et al., 2020")


def test_citation_key_treats_missing_optional_inputs_stably() -> None:
    """Unresolved citations still get a stable identity."""
    unresolved = citation_key(_VERSION_KEY, 1, None, None)

    assert unresolved == citation_key(_VERSION_KEY, 1, None, None)
    assert unresolved != citation_key(_VERSION_KEY, 2, None, None)


def test_document_table_key_is_deterministic_and_input_sensitive() -> None:
    first = document_table_key(_VERSION_KEY, 1, "Table 1", "Caption", "anchor-1")
    second = document_table_key(_VERSION_KEY, 1, "Table 1", "Caption", "anchor-1")

    assert first == second
    assert _SHA256_HEX.fullmatch(first)
    assert first != document_table_key(_VERSION_KEY, 2, "Table 1", "Caption", "anchor-1")
    assert first != document_table_key(_VERSION_KEY, 1, "Table 2", "Caption", "anchor-1")
    assert first != document_table_key("b" * 64, 1, "Table 1", "Caption", "anchor-1")


def test_figure_key_is_deterministic_and_input_sensitive() -> None:
    first = figure_key(_VERSION_KEY, 2, "Figure 2", "Caption", "anchor-2")
    second = figure_key(_VERSION_KEY, 2, "Figure 2", "Caption", "anchor-2")

    assert first == second
    assert _SHA256_HEX.fullmatch(first)
    assert first != figure_key(_VERSION_KEY, 3, "Figure 2", "Caption", "anchor-2")
    assert first != figure_key(_VERSION_KEY, 2, "Figure 3", "Caption", "anchor-2")
    assert first != figure_key("b" * 64, 2, "Figure 2", "Caption", "anchor-2")
