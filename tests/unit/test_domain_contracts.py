"""The canonical document contracts are frozen, strict and self-identifying.

Infrastructure-free: every invariant here is a property of the pydantic
models and the pure identity functions, so no database is required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict, Unpack
from uuid import UUID

import pytest
from pydantic import ValidationError

from dynamisrag.domain.contracts import (
    Citation,
    Document,
    DocumentTable,
    DocumentVersion,
    Figure,
    Passage,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.identity import (
    citation_key,
    document_table_key,
    document_version_key,
    figure_key,
    passage_key,
    section_key,
    source_artifact_key,
)
from dynamisrag.domain.values import DocumentType

_ARTIFACT_ID = UUID("11111111-1111-4111-8111-111111111111")
_DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONTENT_SHA = "a" * 64
_NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)


class _ArtifactKwargs(TypedDict, total=False):
    source_system: str
    source_external_id: str
    source_uri: str
    media_type: str
    content_sha256: str
    byte_size: int
    retrieved_at: datetime
    storage_uri: str
    license_name: str | None
    license_uri: str | None


class _DocumentKwargs(TypedDict, total=False):
    document_type: DocumentType
    doi: str | None
    pmid: str | None
    pmcid: str | None
    title: str | None


class _VersionKwargs(TypedDict, total=False):
    document_id: UUID
    source_artifact_id: UUID
    parser_revision: str
    normalizer_revision: str
    content_fingerprint: str
    title: str
    language: str
    versioned_metadata: dict[str, Any]
    created_at: datetime


def _artifact_kwargs(**overrides: Unpack[_ArtifactKwargs]) -> _ArtifactKwargs:
    kwargs: _ArtifactKwargs = {
        "source_system": "europe_pmc",
        "source_external_id": "PMC123456",
        "source_uri": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fulltextXML",
        "media_type": "application/xml",
        "content_sha256": _CONTENT_SHA,
        "byte_size": 2048,
        "retrieved_at": _NOW,
        "storage_uri": "s3://dynamisrag-artifacts/europe_pmc/PMC123456.xml",
    }
    kwargs.update(overrides)
    return kwargs


def _document_kwargs(**overrides: Unpack[_DocumentKwargs]) -> _DocumentKwargs:
    kwargs: _DocumentKwargs = {
        "document_type": DocumentType.JOURNAL_ARTICLE,
        "doi": "10.1038/nature12373",
        "pmid": "23656234",
        "pmcid": "PMC3656234",
        "title": "A foundational study",
    }
    kwargs.update(overrides)
    return kwargs


def _version_kwargs(**overrides: Unpack[_VersionKwargs]) -> _VersionKwargs:
    kwargs: _VersionKwargs = {
        "document_id": _DOCUMENT_ID,
        "source_artifact_id": _ARTIFACT_ID,
        "parser_revision": "jats-1.2",
        "normalizer_revision": "norm-v3",
        "content_fingerprint": _CONTENT_SHA,
        "title": "A foundational study",
        "language": "en",
        "versioned_metadata": {"journal": "Nature", "volume": "497"},
        "created_at": _NOW,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# SourceArtifact
# ---------------------------------------------------------------------------


def test_source_artifact_derives_its_identity_key() -> None:
    artifact = SourceArtifact(**_artifact_kwargs())

    expected = source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA)
    assert artifact.artifact_key == expected
    assert artifact.artifact_key != ""


def test_source_artifact_is_frozen() -> None:
    artifact = SourceArtifact(**_artifact_kwargs())

    with pytest.raises(ValidationError, match="frozen"):
        artifact.storage_uri = "s3://other"


def test_source_artifact_rejects_unknown_fields() -> None:
    supplied = _artifact_kwargs() | {"unexpected_field": "x"}
    with pytest.raises(ValidationError, match=r"unexpected keyword|extra_forbidden|extra"):
        SourceArtifact.model_validate(supplied)


@pytest.mark.parametrize(
    "bad_hash",
    [
        "not-a-hash",
        "A" * 64,  # uppercase is invalid
        "a" * 63,  # too short
        "a" * 65,  # too long
        "g" * 64,  # non-hex character
        "",
    ],
)
def test_source_artifact_rejects_invalid_sha256(bad_hash: str) -> None:
    with pytest.raises(ValidationError, match=r"content_sha256|pattern"):
        SourceArtifact(**_artifact_kwargs(content_sha256=bad_hash))


def test_source_artifact_rejects_negative_byte_size() -> None:
    with pytest.raises(ValidationError, match="byte_size"):
        SourceArtifact(**_artifact_kwargs(byte_size=-1))


def test_source_artifact_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match=r"retrieved_at|aware"):
        SourceArtifact(
            **_artifact_kwargs(retrieved_at=datetime(2026, 9, 27, 12, 0, 0))  # noqa: DTZ001 - naive datetime is the point of this test
        )


def test_source_artifact_key_cannot_be_supplied_by_the_caller() -> None:
    """The derived identity must not be spoofable through the constructor."""
    supplied = _artifact_kwargs() | {"artifact_key": "0" * 64}
    artifact = SourceArtifact.model_validate(supplied)

    assert artifact.artifact_key == source_artifact_key("europe_pmc", "PMC123456", _CONTENT_SHA)


def test_source_artifact_accepts_optional_license() -> None:
    artifact = SourceArtifact(
        **_artifact_kwargs(
            license_name="CC BY 4.0", license_uri="https://creativecommons.org/licenses/by/4.0/"
        )
    )

    assert artifact.license_name == "CC BY 4.0"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def test_document_derives_canonical_key_from_strongest_identifier() -> None:
    document = Document(**_document_kwargs())

    assert document.canonical_key == "doi:10.1038/nature12373"


def test_document_normalizes_doi_at_the_boundary() -> None:
    document = Document(**_document_kwargs(doi="https://doi.org/10.1038/Nature12373"))

    assert document.doi == "10.1038/nature12373"
    assert document.canonical_key == "doi:10.1038/nature12373"


def test_document_identity_falls_back_through_pmid_and_pmcid() -> None:
    assert (
        Document(**_document_kwargs(doi=None, pmid="23656234", pmcid="PMC3656234")).canonical_key
        == "pmid:23656234"
    )
    assert Document(**_document_kwargs(doi=None, pmid=None, pmcid="PMC3656234")).canonical_key == (
        "pmcid:PMC3656234"
    )


def test_document_without_any_identity_input_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"canonical_key|doi, pmid, pmcid, title"):
        Document(**_document_kwargs(doi=None, pmid=None, pmcid=None, title=None))


def test_document_is_frozen_and_strict() -> None:
    document = Document(**_document_kwargs())

    with pytest.raises(ValidationError, match="frozen"):
        document.title = "changed"
    supplied = _document_kwargs() | {"extra_field": 1}
    with pytest.raises(ValidationError):
        Document.model_validate(supplied)


@pytest.mark.parametrize("bad", ["not-a-pmid", "12345678901", "PMC123456"])
def test_document_rejects_malformed_pmid(bad: str) -> None:
    with pytest.raises(ValidationError, match="pmid"):
        Document(**_document_kwargs(doi=None, pmid=bad, pmcid=None))


@pytest.mark.parametrize("bad", ["123456", "pmc123456", "PMC"])
def test_document_rejects_malformed_pmcid(bad: str) -> None:
    with pytest.raises(ValidationError, match="pmcid"):
        Document(**_document_kwargs(doi=None, pmid=None, pmcid=bad))


# ---------------------------------------------------------------------------
# DocumentVersion
# ---------------------------------------------------------------------------


def test_document_version_derives_its_identity_key() -> None:
    version = DocumentVersion(**_version_kwargs())

    expected = document_version_key(_DOCUMENT_ID, _ARTIFACT_ID, "jats-1.2", "norm-v3", _CONTENT_SHA)
    assert version.version_key == expected


def test_document_version_key_changes_with_processing_configuration() -> None:
    baseline = DocumentVersion(**_version_kwargs())
    reparsed = DocumentVersion(**_version_kwargs(parser_revision="jats-1.3"))

    assert baseline.version_key != reparsed.version_key


def test_document_version_defaults_metadata_to_empty_mapping() -> None:
    version = DocumentVersion(**_version_kwargs(versioned_metadata={}))

    assert version.versioned_metadata == {}


@pytest.mark.parametrize(
    "bad_revision", ["", ".leading-dot", "-leading-dash", "has space", "x" * 65]
)
def test_document_version_rejects_invalid_revision_tags(bad_revision: str) -> None:
    with pytest.raises(ValidationError, match=r"parser_revision|normalizer_revision"):
        DocumentVersion(**_version_kwargs(parser_revision=bad_revision))


def test_document_version_rejects_invalid_content_fingerprint() -> None:
    with pytest.raises(ValidationError, match="content_fingerprint"):
        DocumentVersion(**_version_kwargs(content_fingerprint="ZZZ"))


def test_document_version_is_frozen() -> None:
    version = DocumentVersion(**_version_kwargs())

    with pytest.raises(ValidationError, match="frozen"):
        version.title = "changed"


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


def test_section_derives_its_identity_key() -> None:
    section = Section(document_version_id=_DOCUMENT_ID, ordinal=2, depth=1, structural_path="1.2")

    assert section.section_key == section_key(_DOCUMENT_ID, "1.2")


@pytest.mark.parametrize("bad_path", ["", "1.", ".1", "1..2", "a.b", "1.2.3."])
def test_section_rejects_non_canonical_structural_paths(bad_path: str) -> None:
    with pytest.raises(ValidationError, match="structural_path"):
        Section(document_version_id=_DOCUMENT_ID, ordinal=1, depth=0, structural_path=bad_path)


def test_section_accepts_root_and_nested_paths() -> None:
    root = Section(document_version_id=_DOCUMENT_ID, ordinal=1, depth=0, structural_path="1")
    nested = Section(
        document_version_id=_DOCUMENT_ID,
        ordinal=2,
        depth=2,
        structural_path="1.2.3",
        parent_section_id=root.id,
    )

    assert nested.parent_section_id == root.id


def test_section_rejects_negative_ordinal_and_depth() -> None:
    with pytest.raises(ValidationError, match="ordinal"):
        Section(document_version_id=_DOCUMENT_ID, ordinal=-1, depth=0, structural_path="1")
    with pytest.raises(ValidationError, match="depth"):
        Section(document_version_id=_DOCUMENT_ID, ordinal=0, depth=-1, structural_path="1")


def test_section_is_frozen() -> None:
    section = Section(document_version_id=_DOCUMENT_ID, ordinal=0, depth=0, structural_path="1")

    with pytest.raises(ValidationError, match="frozen"):
        section.title = "changed"


# ---------------------------------------------------------------------------
# Passage
# ---------------------------------------------------------------------------


def test_passage_derives_its_identity_key() -> None:
    passage = Passage(
        document_version_id=_DOCUMENT_ID,
        chunker_revision="chunker-7",
        ordinal=4,
        text="A canonical passage of scientific content.",
        content_sha256=_CONTENT_SHA,
    )

    assert passage.passage_key == passage_key(_DOCUMENT_ID, "chunker-7", 4)


def test_passage_identity_is_independent_of_section_provenance() -> None:
    """Section is provenance, not identity: (version, chunker, ordinal) is the key."""
    first = Passage(
        document_version_id=_DOCUMENT_ID,
        chunker_revision="chunker-7",
        ordinal=4,
        text="one",
        content_sha256="b" * 64,
    )
    second = Passage(
        document_version_id=_DOCUMENT_ID,
        chunker_revision="chunker-7",
        ordinal=4,
        text="two",
        content_sha256="c" * 64,
    )

    assert first.passage_key == second.passage_key


def test_passage_rejects_empty_text_and_negative_token_count() -> None:
    base: dict[str, Any] = {
        "document_version_id": _DOCUMENT_ID,
        "chunker_revision": "chunker-7",
        "ordinal": 0,
        "content_sha256": _CONTENT_SHA,
    }
    with pytest.raises(ValidationError, match="text"):
        Passage(**base, text="")
    with pytest.raises(ValidationError, match="token_count"):
        Passage(**base, text="x", token_count=-1)


def test_passage_accepts_unknown_token_count() -> None:
    passage = Passage(
        document_version_id=_DOCUMENT_ID,
        chunker_revision="chunker-7",
        ordinal=0,
        text="x",
        content_sha256=_CONTENT_SHA,
        token_count=None,
    )

    assert passage.token_count is None


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def test_unresolved_citation_is_valid() -> None:
    citation = Citation(document_version_id=_DOCUMENT_ID, ordinal=0)

    assert citation.doi is None
    assert citation.pmid is None
    assert citation.resolved_document_id is None
    assert citation.citation_key == citation_key(_DOCUMENT_ID, 0, None, None)


def test_citation_derives_its_identity_key() -> None:
    citation = Citation(
        document_version_id=_DOCUMENT_ID,
        ordinal=3,
        source_reference_id="ref-3",
        doi="10.1016/j.cell.2020.01.001",
        pmid="31900000",
        pmcid="PMC7000000",
        title="A cited study",
        year=2020,
        raw_reference_text="Doe et al. (2020). A cited study. Cell.",
        resolved_document_id=_DOCUMENT_ID,
    )

    assert citation.citation_key == citation_key(
        _DOCUMENT_ID, 3, "ref-3", "Doe et al. (2020). A cited study. Cell."
    )


@pytest.mark.parametrize("bad_year", [999, 2201, 0, -1])
def test_citation_rejects_implausible_years(bad_year: int) -> None:
    with pytest.raises(ValidationError, match="year"):
        Citation(document_version_id=_DOCUMENT_ID, ordinal=0, year=bad_year)


def test_citation_is_frozen() -> None:
    citation = Citation(document_version_id=_DOCUMENT_ID, ordinal=0)

    with pytest.raises(ValidationError, match="frozen"):
        citation.year = 2021


# ---------------------------------------------------------------------------
# DocumentTable / Figure
# ---------------------------------------------------------------------------


def test_document_table_derives_its_identity_key() -> None:
    table = DocumentTable(
        document_version_id=_DOCUMENT_ID,
        ordinal=1,
        label="Table 1",
        caption="Baseline characteristics",
        source_anchor="table-wrap-1",
    )

    assert table.document_table_key == document_table_key(
        _DOCUMENT_ID, 1, "Table 1", "Baseline characteristics", "table-wrap-1"
    )


def test_document_table_defaults_structured_representation_to_empty_mapping() -> None:
    table = DocumentTable(document_version_id=_DOCUMENT_ID, ordinal=1)

    assert table.structured_representation == {}


def test_figure_derives_its_identity_key() -> None:
    figure = Figure(
        document_version_id=_DOCUMENT_ID,
        ordinal=2,
        label="Figure 2",
        caption="Pathway diagram",
        source_anchor="fig-2",
        asset_locator="s3://dynamisrag-artifacts/figures/fig-2.png",
    )

    assert figure.figure_key == figure_key(_DOCUMENT_ID, 2, "Figure 2", "Pathway diagram", "fig-2")


@pytest.mark.parametrize("contract", [DocumentTable, Figure])
def test_structural_objects_are_frozen(contract: type[DocumentTable] | type[Figure]) -> None:
    instance = contract(document_version_id=_DOCUMENT_ID, ordinal=0)

    with pytest.raises(ValidationError, match="frozen"):
        instance.ordinal = 9
