"""Persistence invariants of the canonical document model, proven against
live PostgreSQL 18.

Every test runs inside a transaction that is rolled back at teardown, so the
canonical database stays clean and the immutability rules are never fought
with DELETE-based cleanup. The tests exercise the schema through the same
primitives the application will use (``dynamisrag.db.canonical``), and where
an invariant is enforced by the database itself, the prohibited write is
attempted directly to prove the database — not the Python layer — rejects it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.db import (
    SemanticParentKeyError,
    create_database_engine,
    get_citation_resolutions,
    get_document,
    get_document_identifiers,
    get_document_version,
    get_section,
    get_source_artifact,
    insert_citation,
    insert_citation_resolution,
    insert_document,
    insert_document_identifier,
    insert_document_table,
    insert_document_version,
    insert_figure,
    insert_passage,
    insert_section,
    insert_source_artifact,
    list_citations,
    list_document_tables,
    list_document_versions,
    list_figures,
    list_passages,
    list_sections,
)
from dynamisrag.db.models import (
    CitationRecord,
    CitationResolutionRecord,
    DocumentIdentifierRecord,
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    PassageRecord,
    SectionRecord,
    SourceArtifactRecord,
)
from dynamisrag.domain.contracts import (
    Citation,
    CitationResolution,
    Document,
    DocumentIdentifier,
    DocumentTable,
    DocumentVersion,
    Figure,
    Passage,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.identity import (
    citation_resolution_key,
    document_version_key,
    source_artifact_key,
)
from dynamisrag.domain.values import DocumentType, IdentifierNamespace
from tests._support import ALEMBIC_INI, REPO_ROOT, build_settings

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
_CONTENT_SHA = "a" * 64
_ARTIFACT_SHA = "b" * 64
_ARTIFACT_KEY = source_artifact_key("europe_pmc", "PMC123456", _ARTIFACT_SHA)
_DOCUMENT_KEY = "doi:10.1038/nature12373"
_VERSION_KEY = document_version_key(
    _DOCUMENT_KEY, _ARTIFACT_KEY, "jats-1.2", "norm-v3", _CONTENT_SHA
)

_CANONICAL_TABLES = (
    "source_artifact",
    "document",
    "document_identifier",
    "document_version",
    "section",
    "passage",
    "citation",
    "citation_resolution",
    "document_table",
    "figure",
)


@pytest.fixture
def db_session(live_settings: Settings) -> Iterator[Session]:
    """A session whose transaction is always rolled back.

    Rollback (never DELETE) keeps the canonical tables pristine between
    tests and doubles as proof that the suite needs no deletion lifecycle.
    """
    engine = create_database_engine(live_settings)
    session = Session(bind=engine)
    transaction = session.begin()
    try:
        yield session
    finally:
        transaction.rollback()
        session.close()
        engine.dispose()


def _make_artifact(**overrides: Any) -> SourceArtifact:
    kwargs: dict[str, Any] = {
        "source_system": "europe_pmc",
        "source_external_id": "PMC123456",
        "source_uri": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fulltextXML",
        "media_type": "application/xml",
        "content_sha256": _ARTIFACT_SHA,
        "byte_size": 2048,
        "retrieved_at": _NOW,
        "storage_uri": "s3://dynamisrag-artifacts/europe_pmc/PMC123456.xml",
    }
    kwargs.update(overrides)
    return SourceArtifact(**kwargs)


def _make_document(**overrides: Any) -> Document:
    kwargs: dict[str, Any] = {
        "document_type": DocumentType.JOURNAL_ARTICLE,
        "doi": "10.1038/nature12373",
        "pmid": "23656234",
        "pmcid": "PMC3656234",
        "title": "A foundational study",
    }
    kwargs.update(overrides)
    return Document(**kwargs)


def _make_version(
    document: Document, artifact: SourceArtifact, **overrides: Any
) -> DocumentVersion:
    kwargs: dict[str, Any] = {
        "document_id": document.id,
        "document_canonical_key": document.canonical_key,
        "source_artifact_id": artifact.id,
        "source_artifact_key": artifact.artifact_key,
        "parser_revision": "jats-1.2",
        "normalizer_revision": "norm-v3",
        "content_fingerprint": _CONTENT_SHA,
        "title": "A foundational study",
        "language": "en",
        "versioned_metadata": {"journal": "Nature", "volume": "497"},
        "created_at": _NOW,
    }
    kwargs.update(overrides)
    return DocumentVersion(**kwargs)


def _make_section(
    version: DocumentVersion,
    structural_path: str = "1",
    ordinal: int = 1,
    *,
    parent: Section | None = None,
    **overrides: Any,
) -> Section:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "version_key": version.version_key,
        "ordinal": ordinal,
        "depth": structural_path.count("."),
        "title": f"Section {structural_path}",
        "structural_path": structural_path,
    }
    if parent is not None:
        kwargs["parent_section_id"] = parent.id
    kwargs.update(overrides)
    return Section(**kwargs)


def _make_passage(
    version: DocumentVersion,
    *,
    section: Section | None = None,
    chunker_revision: str = "chunker-7",
    **overrides: Any,
) -> Passage:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "version_key": version.version_key,
        "chunker_revision": chunker_revision,
        "ordinal": 0,
        "text": "A canonical passage of scientific content.",
        "content_sha256": _CONTENT_SHA,
        "section_id": section.id if section else None,
    }
    kwargs.update(overrides)
    return Passage(**kwargs)


def _make_citation(version: DocumentVersion, **overrides: Any) -> Citation:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "version_key": version.version_key,
        "ordinal": 0,
    }
    kwargs.update(overrides)
    return Citation(**kwargs)


def _make_table(
    version: DocumentVersion, *, section: Section | None = None, **overrides: Any
) -> DocumentTable:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "version_key": version.version_key,
        "ordinal": 1,
        "label": "Table 1",
        "caption": "Baseline characteristics",
        "source_anchor": "table-wrap-1",
        "section_id": section.id if section else None,
    }
    kwargs.update(overrides)
    return DocumentTable(**kwargs)


def _make_figure(
    version: DocumentVersion, *, section: Section | None = None, **overrides: Any
) -> Figure:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "version_key": version.version_key,
        "ordinal": 1,
        "label": "Figure 1",
        "caption": "Study overview",
        "source_anchor": "fig-1",
        "asset_locator": "s3://dynamisrag-artifacts/figures/fig-1.png",
        "section_id": section.id if section else None,
    }
    kwargs.update(overrides)
    return Figure(**kwargs)


def _make_resolution(
    citation: Citation, document: Document, **overrides: Any
) -> CitationResolution:
    kwargs: dict[str, Any] = {
        "citation_id": citation.id,
        "citation_key": citation.citation_key,
        "resolved_document_id": document.id,
        "resolved_document_canonical_key": document.canonical_key,
        "resolver_revision": "resolver-1",
        "resolved_at": _NOW,
    }
    kwargs.update(overrides)
    return CitationResolution(**kwargs)


def _full_graph(
    session: Session,
) -> tuple[
    SourceArtifact,
    Document,
    DocumentVersion,
    Section,
    Section,
    Passage,
    Citation,
    DocumentTable,
    Figure,
    Document,
    CitationResolution,
]:
    """Insert one of every canonical entity and return the domain objects."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    root = _make_section(version, structural_path="1", ordinal=1)
    child = _make_section(version, structural_path="1.2", ordinal=2, parent=root)
    passage = _make_passage(version)
    citation = _make_citation(version)
    table = _make_table(version, section=root)
    figure = _make_figure(version, section=child)
    cited = _make_document(doi=None, pmid=None, pmcid=None, title="The cited work")
    resolution = _make_resolution(citation, cited)

    insert_source_artifact(session, artifact)
    insert_document(session, document)
    insert_document(session, cited)
    insert_document_version(session, version)
    insert_section(session, root)
    insert_section(session, child)
    insert_passage(session, passage)
    insert_citation(session, citation)
    insert_document_table(session, table)
    insert_figure(session, figure)
    insert_citation_resolution(session, resolution)
    return (
        artifact,
        document,
        version,
        root,
        child,
        passage,
        citation,
        table,
        figure,
        cited,
        resolution,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def _assert_resolution_round_trip(
    db_session: Session, cited: Document, resolution: CitationResolution
) -> None:
    read_cited = get_document(db_session, cited.id)
    assert read_cited is not None
    assert read_cited.canonical_key == cited.canonical_key

    resolutions = get_citation_resolutions(db_session, resolution.citation_id)
    assert [item.id for item in resolutions] == [resolution.id]
    assert resolutions[0].resolved_document_id == cited.id
    assert resolutions[0].resolver_revision == "resolver-1"
    assert resolutions[0].resolved_at == _NOW
    assert resolutions[0].resolution_key == resolution.resolution_key


def test_full_graph_round_trip(db_session: Session) -> None:
    """One of every canonical entity can be inserted and read back intact."""
    (
        artifact,
        document,
        version,
        root,
        child,
        passage,
        citation,
        table,
        figure,
        cited,
        resolution,
    ) = _full_graph(db_session)

    read_artifact = get_source_artifact(db_session, artifact.id)
    assert read_artifact is not None
    assert read_artifact.source_system == artifact.source_system
    assert read_artifact.source_external_id == artifact.source_external_id
    assert read_artifact.content_sha256 == artifact.content_sha256
    assert read_artifact.artifact_key == artifact.artifact_key
    assert read_artifact.retrieved_at == artifact.retrieved_at
    assert read_artifact.row_created_at is not None

    read_document = get_document(db_session, document.id)
    assert read_document is not None
    assert read_document.canonical_key == document.canonical_key
    assert read_document.document_type == document.document_type.value
    assert read_document.title == document.title

    read_version = get_document_version(db_session, version.id)
    assert read_version is not None
    assert read_version.version_key == version.version_key
    assert read_version.title == version.title
    assert read_version.versioned_metadata == version.versioned_metadata
    assert read_version.created_at == version.created_at

    read_root = get_section(db_session, root.id)
    assert read_root is not None
    assert read_root.section_key == root.section_key
    assert read_root.parent_section_id is None

    read_child = get_section(db_session, child.id)
    assert read_child is not None
    assert read_child.parent_section_id == root.id
    # The composite parent foreign key stores the parent's document version
    # explicitly; it must agree with the child's own version.
    assert read_child.parent_document_version_id == version.id

    passages = list_passages(db_session, version.id, passage.chunker_revision)
    assert [item.id for item in passages] == [passage.id]
    assert passages[0].text == passage.text
    assert passages[0].content_sha256 == passage.content_sha256

    citations = list_citations(db_session, version.id)
    assert [item.id for item in citations] == [citation.id]

    tables = list_document_tables(db_session, version.id)
    assert [item.id for item in tables] == [table.id]
    assert tables[0].section_id == root.id
    assert tables[0].structured_representation == table.structured_representation

    figures = list_figures(db_session, version.id)
    assert [item.id for item in figures] == [figure.id]
    assert figures[0].section_id == child.id
    assert figures[0].asset_locator == figure.asset_locator

    _assert_resolution_round_trip(db_session, cited, resolution)


def test_list_sections_returns_every_inserted_section(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    root = _make_section(version, structural_path="1", ordinal=1)
    child = _make_section(version, structural_path="1.2", ordinal=2, parent=root)
    insert_section(db_session, root)
    insert_section(db_session, child)

    sections = list_sections(db_session, version.id)
    assert {item.id for item in sections} == {root.id, child.id}


# ---------------------------------------------------------------------------
# Identity: duplicate deterministic identities are rejected
# ---------------------------------------------------------------------------


def test_duplicate_source_artifact_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    insert_source_artifact(db_session, artifact)
    duplicate = _make_artifact()

    with pytest.raises(IntegrityError, match="uq_source_artifact_artifact_key"):
        insert_source_artifact(db_session, duplicate)


def test_same_external_object_with_different_bytes_is_a_different_artifact(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    insert_source_artifact(db_session, artifact)
    reacquired = _make_artifact(content_sha256="c" * 64, byte_size=4096)

    record = insert_source_artifact(db_session, reacquired)

    assert record.id != artifact.id
    assert record.artifact_key != artifact.artifact_key


def test_duplicate_document_identity_is_rejected(db_session: Session) -> None:
    document = _make_document()
    insert_document(db_session, document)
    duplicate = _make_document()

    with pytest.raises(IntegrityError, match="uq_document_canonical_key"):
        insert_document(db_session, duplicate)


def test_duplicate_document_version_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    duplicate = _make_version(document, artifact)

    with pytest.raises(IntegrityError, match="uq_document_version_version_key"):
        insert_document_version(db_session, duplicate)


def test_duplicate_section_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    insert_section(db_session, _make_section(version, structural_path="1", ordinal=1))
    duplicate = _make_section(version, structural_path="1", ordinal=5)

    with pytest.raises(IntegrityError, match="uq_section_document_version_section_key"):
        insert_section(db_session, duplicate)


def test_sections_in_different_versions_may_share_a_path(db_session: Session) -> None:
    """Structural paths are version-scoped: the same path in two versions is fine."""
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    version_two = _make_version(document, artifact, parser_revision="jats-1.3")
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)
    insert_document_version(db_session, version_two)
    insert_section(db_session, _make_section(version_one, structural_path="1", ordinal=1))
    record = insert_section(db_session, _make_section(version_two, structural_path="1", ordinal=1))

    sections = list_sections(db_session, version_two.id)
    assert [item.id for item in sections] == [record.id]
    assert record.structural_path == "1"


def test_duplicate_passage_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    insert_passage(db_session, _make_passage(version, ordinal=0))
    duplicate = _make_passage(version, ordinal=0, text="A different passage, same identity")

    with pytest.raises(IntegrityError, match="uq_passage_version_chunker_ordinal"):
        insert_passage(db_session, duplicate)


def test_duplicate_citation_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    insert_citation(db_session, _make_citation(version, ordinal=0))
    duplicate = _make_citation(version, ordinal=0)

    with pytest.raises(IntegrityError, match="uq_citation_document_version_ordinal"):
        insert_citation(db_session, duplicate)


def test_duplicate_document_table_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    insert_document_table(db_session, _make_table(version, ordinal=1))
    duplicate = _make_table(version, ordinal=1)

    with pytest.raises(IntegrityError, match="uq_document_table_document_version_key"):
        insert_document_table(db_session, duplicate)


def test_duplicate_figure_identity_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    insert_figure(db_session, _make_figure(version, ordinal=1))
    duplicate = _make_figure(version, ordinal=1)

    with pytest.raises(IntegrityError, match="uq_figure_document_version_key"):
        insert_figure(db_session, duplicate)


def test_repeated_persist_of_one_artifact_identity_collides_not_duplicates(
    db_session: Session,
) -> None:
    """A repeated persist of the same deterministic artifact identity is a
    unique-constraint collision — never an ambiguous second row, even when
    the surrogate key differs."""
    artifact = _make_artifact()
    first = insert_source_artifact(db_session, artifact)
    repersist = _make_artifact()
    assert repersist.artifact_key == artifact.artifact_key
    assert repersist.id != artifact.id

    # The savepoint absorbs the constraint violation so the outer transaction
    # stays usable and the row count can be asserted.
    with (
        db_session.begin_nested(),
        pytest.raises(IntegrityError, match="uq_source_artifact_artifact_key"),
    ):
        insert_source_artifact(db_session, repersist)

    count = db_session.scalar(select(func.count()).select_from(SourceArtifactRecord))
    assert count == 1
    assert get_source_artifact(db_session, first.id) is not None


# ---------------------------------------------------------------------------
# Document identifiers: aliases are separate from stable identity
# ---------------------------------------------------------------------------


def test_creation_time_identifiers_are_persisted_as_aliases(db_session: Session) -> None:
    """Every strong identifier known at creation is persisted as an immutable
    alias, so the alias table is the complete record of known identifiers."""
    document = _make_document()

    insert_document(db_session, document)

    aliases = get_document_identifiers(db_session, document.id)
    assert {(item.namespace, item.normalized_value) for item in aliases} == {
        ("doi", "10.1038/nature12373"),
        ("pmid", "23656234"),
        ("pmcid", "PMC3656234"),
    }


def test_enrichment_attaches_an_alias_without_changing_canonical_identity(
    db_session: Session,
) -> None:
    """A newly discovered DOI joins the document as an alias.

    The document's canonical identity was fixed at creation from the then
    known PMCID; enrichment must not silently re-identify the same logical
    work, so the stored canonical key is byte-for-byte unchanged afterwards.
    """
    document = _make_document(doi=None, pmid=None, pmcid="PMC3656234")
    insert_document(db_session, document)
    before = get_document(db_session, document.id)
    assert before is not None
    assert before.canonical_key == "pmcid:PMC3656234"

    insert_document_identifier(
        db_session,
        DocumentIdentifier(
            document_id=document.id,
            namespace=IdentifierNamespace.DOI,
            normalized_value="10.1038/nature12373",
        ),
    )

    after = get_document(db_session, document.id)
    assert after is not None
    assert after.canonical_key == before.canonical_key == "pmcid:PMC3656234"
    aliases = get_document_identifiers(db_session, document.id)
    assert {(item.namespace, item.normalized_value) for item in aliases} == {
        ("pmcid", "PMC3656234"),
        ("doi", "10.1038/nature12373"),
    }


def test_same_alias_cannot_be_attached_to_two_documents(db_session: Session) -> None:
    """(namespace, normalized_value) is globally unique: one DOI identifies
    one document, so the same alias on a second document is rejected."""
    first = _make_document(doi="10.1038/nature12373")
    second = _make_document(doi=None, pmid=None, pmcid=None, title="An unrelated work")
    insert_document(db_session, first)
    insert_document(db_session, second)

    with pytest.raises(IntegrityError, match="uq_document_identifier_namespace_value"):
        insert_document_identifier(
            db_session,
            DocumentIdentifier(
                document_id=second.id,
                namespace=IdentifierNamespace.DOI,
                normalized_value="10.1038/nature12373",
            ),
        )


def test_title_only_document_holds_a_provisional_identity_with_no_aliases(
    db_session: Session,
) -> None:
    """A title-digest identity is provisional: no alias exists for it, and
    the canonical key is explicitly weaker than any alias-based identity."""
    document = _make_document(doi=None, pmid=None, pmcid=None, title="A foundational study")

    insert_document(db_session, document)

    read = get_document(db_session, document.id)
    assert read is not None
    assert read.canonical_key.startswith("title:")
    assert get_document_identifiers(db_session, document.id) == []


def test_database_rejects_malformed_document_identifier_value(
    db_session: Session,
) -> None:
    """The CHECK constraint mirrors the domain namespace/value validation for
    any writer that bypasses the contracts."""
    document = _make_document()
    insert_document(db_session, document)
    record = DocumentIdentifierRecord(
        id=uuid4(),
        document_id=document.id,
        namespace="doi",
        normalized_value="not-a-doi",
    )
    db_session.add(record)

    with pytest.raises(IntegrityError, match="ck_document_identifier_value_format"):
        db_session.flush()


def test_database_rejects_unknown_identifier_namespace(db_session: Session) -> None:
    document = _make_document()
    insert_document(db_session, document)
    record = DocumentIdentifierRecord(
        id=uuid4(),
        document_id=document.id,
        namespace="isbn",
        normalized_value="978-3-16-148410-0",
    )
    db_session.add(record)

    with pytest.raises(IntegrityError, match="ck_document_identifier_namespace"):
        db_session.flush()


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_multiple_versions_belong_to_one_document(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    version_two = _make_version(
        document, artifact, parser_revision="jats-1.3", content_fingerprint="d" * 64
    )
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)
    insert_document_version(db_session, version_two)

    versions = list_document_versions(db_session, document.id)
    assert {item.id for item in versions} == {version_one.id, version_two.id}
    assert versions[0].version_key != versions[1].version_key


def test_previous_version_is_unchanged_when_a_newer_one_is_inserted(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)

    before = get_document_version(db_session, version_one.id)
    assert before is not None
    snapshot = (
        before.title,
        before.language,
        before.parser_revision,
        before.normalizer_revision,
        before.content_fingerprint,
        before.versioned_metadata,
        before.created_at,
        before.version_key,
        before.row_created_at,
    )

    version_two = _make_version(
        document, artifact, parser_revision="jats-1.3", content_fingerprint="e" * 64
    )
    insert_document_version(db_session, version_two)

    after = get_document_version(db_session, version_one.id)
    assert after is not None
    assert (
        after.title,
        after.language,
        after.parser_revision,
        after.normalizer_revision,
        after.content_fingerprint,
        after.versioned_metadata,
        after.created_at,
        after.version_key,
        after.row_created_at,
    ) == snapshot


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_nested_section_hierarchy_round_trips(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    root = _make_section(version, structural_path="1", ordinal=1)
    child = _make_section(version, structural_path="1.2", ordinal=2, parent=root)
    grandchild = _make_section(version, structural_path="1.2.3", ordinal=3, parent=child)
    insert_section(db_session, root)
    insert_section(db_session, child)
    insert_section(db_session, grandchild)

    read_root = get_section(db_session, root.id)
    read_child = get_section(db_session, child.id)
    read_grandchild = get_section(db_session, grandchild.id)
    assert read_root is not None and read_child is not None and read_grandchild is not None
    assert read_root.parent_section_id is None
    assert read_child.parent_section_id == root.id
    assert read_grandchild.parent_section_id == child.id
    assert read_grandchild.depth == 2


def test_section_parent_from_another_document_version_is_rejected(
    db_session: Session,
) -> None:
    """The composite parent foreign key rejects a cross-version parent.

    The persistence mapper can never produce this row (it derives
    ``parent_document_version_id`` from the section's own version), so the
    prohibited row is built directly to prove the database enforces the
    invariant for any writer.
    """
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    version_two = _make_version(document, artifact, parser_revision="jats-1.3")
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)
    insert_document_version(db_session, version_two)
    root = _make_section(version_one, structural_path="1", ordinal=1)
    insert_section(db_session, root)

    cross_version_child = SectionRecord(
        id=uuid4(),
        document_version_id=version_two.id,
        parent_section_id=root.id,
        parent_document_version_id=version_two.id,
        ordinal=1,
        depth=1,
        structural_path="1",
        section_key="f" * 64,
    )
    db_session.add(cross_version_child)

    with pytest.raises(IntegrityError, match="fk_section_parent"):
        db_session.flush()


# ---------------------------------------------------------------------------
# Passage
# ---------------------------------------------------------------------------


def test_multiple_chunker_revisions_coexist_for_one_version(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    insert_passage(db_session, _make_passage(version, chunker_revision="chunker-7", ordinal=0))
    insert_passage(db_session, _make_passage(version, chunker_revision="chunker-7", ordinal=1))
    insert_passage(db_session, _make_passage(version, chunker_revision="chunker-8", ordinal=0))

    revision_seven = list_passages(db_session, version.id, "chunker-7")
    revision_eight = list_passages(db_session, version.id, "chunker-8")
    assert [item.ordinal for item in revision_seven] == [0, 1]
    assert [item.ordinal for item in revision_eight] == [0]
    assert revision_seven[0].passage_key != revision_eight[0].passage_key


def test_passage_with_section_from_the_same_version_succeeds(
    db_session: Session,
) -> None:
    """A passage may own a section of its own document version."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    section = _make_section(version, structural_path="2", ordinal=2)
    insert_section(db_session, section)
    passage = _make_passage(version, section=section, ordinal=0)

    record = insert_passage(db_session, passage)

    assert record.section_id == section.id
    assert record.section_document_version_id == version.id


def test_passage_without_a_section_remains_valid(db_session: Session) -> None:
    """Section ownership is optional: a sectionless passage is a valid
    first-class record."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)

    record = insert_passage(db_session, _make_passage(version, ordinal=0))

    assert record.section_id is None
    assert record.section_document_version_id is None


def test_passage_with_section_from_another_document_version_is_rejected(
    db_session: Session,
) -> None:
    """The composite section foreign key rejects a cross-version section.

    The persistence mapper can never produce this row (it derives
    ``section_document_version_id`` from the passage's own version), so the
    prohibited row is built directly to prove the database enforces the
    invariant for any writer.
    """
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    version_two = _make_version(document, artifact, parser_revision="jats-1.3")
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)
    insert_document_version(db_session, version_two)
    section = _make_section(version_one, structural_path="1", ordinal=1)
    insert_section(db_session, section)

    cross_version_passage = PassageRecord(
        id=uuid4(),
        document_version_id=version_two.id,
        section_id=section.id,
        section_document_version_id=version_two.id,
        chunker_revision="chunker-7",
        ordinal=0,
        text="A canonical passage of scientific content.",
        content_sha256=_CONTENT_SHA,
        passage_key="f" * 64,
    )
    db_session.add(cross_version_passage)

    with pytest.raises(IntegrityError, match="fk_passage_section"):
        db_session.flush()


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def test_unresolved_citation_is_a_valid_first_class_record(
    db_session: Session,
) -> None:
    """An unresolved citation is permanently valid: no resolution state lives
    on the canonical citation, so nothing forecloses resolving it later."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    citation = _make_citation(version, ordinal=0)

    record = insert_citation(db_session, citation)

    assert record.doi is None
    assert record.pmid is None
    assert record.pmcid is None
    assert record.year is None
    assert record.citation_key == citation.citation_key


def _citation_graph(db_session: Session) -> tuple[Citation, Document]:
    """Insert a document with one unresolved citation; return it and a
    separate cited document for it to resolve to."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    citation = _make_citation(version, ordinal=0)
    insert_citation(db_session, citation)
    cited = _make_document(doi=None, pmid=None, pmcid=None, title="The cited work")
    insert_document(db_session, cited)
    return citation, cited


def test_unresolved_citation_can_be_resolved_later_without_updating_the_citation(
    db_session: Session,
) -> None:
    """The full append-only resolution flow: persist an unresolved citation,
    append a resolution later, and prove the canonical citation is untouched."""
    citation, cited = _citation_graph(db_session)

    before = db_session.get(CitationRecord, citation.id)
    assert before is not None
    snapshot = (
        before.ordinal,
        before.source_reference_id,
        before.doi,
        before.pmid,
        before.pmcid,
        before.title,
        before.year,
        before.raw_reference_text,
        before.citation_key,
        before.row_created_at,
    )
    assert get_citation_resolutions(db_session, citation.id) == []

    resolution = _make_resolution(citation, cited)
    record = insert_citation_resolution(db_session, resolution)

    assert record.resolved_document_id == cited.id
    assert record.resolution_key == resolution.resolution_key

    after = db_session.get(CitationRecord, citation.id)
    assert after is not None
    assert (
        after.ordinal,
        after.source_reference_id,
        after.doi,
        after.pmid,
        after.pmcid,
        after.title,
        after.year,
        after.raw_reference_text,
        after.citation_key,
        after.row_created_at,
    ) == snapshot

    resolutions = get_citation_resolutions(db_session, citation.id)
    assert [item.id for item in resolutions] == [record.id]
    assert resolutions[0].resolved_document_id == cited.id


def test_citation_resolution_with_unknown_resolved_document_is_rejected(
    db_session: Session,
) -> None:
    citation, _ = _citation_graph(db_session)
    resolution = _make_resolution(
        citation, _make_document(doi=None, pmid=None, pmcid=None, title="X")
    )
    object.__setattr__(resolution, "resolved_document_id", uuid4())

    with pytest.raises(IntegrityError, match="fk_citation_resolution_resolved_document"):
        insert_citation_resolution(db_session, resolution)


def test_repeated_identical_resolution_is_rejected_not_duplicated(
    db_session: Session,
) -> None:
    """The deterministic resolution key makes an identical re-resolution a
    unique-constraint collision — never an ambiguous second row."""
    citation, cited = _citation_graph(db_session)
    first = insert_citation_resolution(db_session, _make_resolution(citation, cited))
    repeat = _make_resolution(citation, cited)
    assert repeat.resolution_key == first.resolution_key
    assert repeat.id != first.id

    with (
        db_session.begin_nested(),
        pytest.raises(IntegrityError, match="uq_citation_resolution_resolution_key"),
    ):
        insert_citation_resolution(db_session, repeat)

    count = db_session.scalar(select(func.count()).select_from(CitationResolutionRecord))
    assert count == 1


def test_new_resolver_revision_coexists_as_a_separate_resolution(
    db_session: Session,
) -> None:
    """Resolution records are versionable: a new resolver revision appends a
    new, coexisting resolution instead of replacing the previous one."""
    citation, cited = _citation_graph(db_session)
    insert_citation_resolution(
        db_session, _make_resolution(citation, cited, resolver_revision="resolver-1")
    )
    insert_citation_resolution(
        db_session, _make_resolution(citation, cited, resolver_revision="resolver-2")
    )

    resolutions = get_citation_resolutions(db_session, citation.id)
    assert [item.resolver_revision for item in resolutions] == ["resolver-1", "resolver-2"]
    assert resolutions[0].resolution_key != resolutions[1].resolution_key


# ---------------------------------------------------------------------------
# Structural objects
# ---------------------------------------------------------------------------


def test_table_and_figure_belong_to_sections_of_their_own_version(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    section = _make_section(version, structural_path="2", ordinal=2)
    insert_section(db_session, section)
    table = _make_table(version, ordinal=1, section=section)
    figure = _make_figure(version, ordinal=1, section=section)

    table_record = insert_document_table(db_session, table)
    figure_record = insert_figure(db_session, figure)

    assert table_record.section_document_version_id == version.id
    assert figure_record.section_document_version_id == version.id


def test_table_with_section_from_another_version_is_rejected(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    version_two = _make_version(document, artifact, parser_revision="jats-1.3")
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)
    insert_document_version(db_session, version_two)
    section = _make_section(version_one, structural_path="1", ordinal=1)
    insert_section(db_session, section)

    cross_version_table = DocumentTableRecord(
        id=uuid4(),
        document_version_id=version_two.id,
        section_id=section.id,
        section_document_version_id=version_two.id,
        ordinal=1,
        document_table_key="f" * 64,
    )
    db_session.add(cross_version_table)

    with pytest.raises(IntegrityError, match="fk_document_table_section"):
        db_session.flush()


def test_figure_with_section_from_another_version_is_rejected(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version_one = _make_version(document, artifact)
    version_two = _make_version(document, artifact, parser_revision="jats-1.3")
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version_one)
    insert_document_version(db_session, version_two)
    section = _make_section(version_one, structural_path="1", ordinal=1)
    insert_section(db_session, section)

    cross_version_figure = FigureRecord(
        id=uuid4(),
        document_version_id=version_two.id,
        section_id=section.id,
        section_document_version_id=version_two.id,
        ordinal=1,
        figure_key="f" * 64,
    )
    db_session.add(cross_version_figure)

    with pytest.raises(IntegrityError, match="fk_figure_section"):
        db_session.flush()


# ---------------------------------------------------------------------------
# Hash validation at the database layer
# ---------------------------------------------------------------------------


def _raw_artifact(content_sha256: str) -> SourceArtifactRecord:
    """Build an artifact row directly, bypassing domain hash validation.

    The domain contracts already reject malformed hashes (proven in the unit
    suite); these tests prove the database CHECK constraint independently.
    """
    return SourceArtifactRecord(
        id=uuid4(),
        source_system="europe_pmc",
        source_external_id="PMC999999",
        source_uri="https://example.org/fulltextXML",
        media_type="application/xml",
        content_sha256=content_sha256,
        byte_size=10,
        retrieved_at=_NOW,
        storage_uri="s3://dynamisrag-artifacts/raw.xml",
        artifact_key="f" * 64,
    )


@pytest.mark.parametrize("bad_hash", ["A" * 64, "a" * 63, "g" * 64, "not-a-hash"])
def test_database_rejects_invalid_sha256_on_source_artifact(
    db_session: Session, bad_hash: str
) -> None:
    db_session.add(_raw_artifact(bad_hash))

    with pytest.raises(IntegrityError, match="ck_source_artifact_content_sha256_hex"):
        db_session.flush()


def test_database_rejects_invalid_sha256_on_document_version(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    document = _make_document()
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    record = DocumentVersionRecord(
        id=uuid4(),
        document_id=document.id,
        source_artifact_id=artifact.id,
        parser_revision="jats-1.2",
        normalizer_revision="norm-v3",
        content_fingerprint="A" * 64,
        title="A foundational study",
        language="en",
        versioned_metadata={},
        created_at=_NOW,
        version_key="f" * 64,
    )
    db_session.add(record)

    with pytest.raises(IntegrityError, match="ck_document_version_content_fingerprint_hex"):
        db_session.flush()


def test_database_rejects_invalid_sha256_on_passage(db_session: Session) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    record = PassageRecord(
        id=uuid4(),
        document_version_id=version.id,
        chunker_revision="chunker-7",
        ordinal=0,
        text="A canonical passage of scientific content.",
        content_sha256="a" * 63,
        passage_key="f" * 64,
    )
    db_session.add(record)

    with pytest.raises(IntegrityError, match="ck_passage_content_sha256_hex"):
        db_session.flush()


# ---------------------------------------------------------------------------
# Immutability: the database itself rejects prohibited mutations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", _CANONICAL_TABLES)
def test_database_rejects_update_on_canonical_tables(db_session: Session, table: str) -> None:
    """In-place mutation is rejected by the database, not by Python objects.

    The UPDATE sets a column to its own value: the values are irrelevant,
    the operation itself is what the append-only schema prohibits.
    """
    _full_graph(db_session)

    # The table name comes from the hardcoded _CANONICAL_TABLES tuple, never
    # from input; the whole point is to issue raw SQL against PostgreSQL.
    with pytest.raises(SQLAlchemyError, match="append-only"):
        db_session.execute(text(f"UPDATE {table} SET row_created_at = row_created_at"))  # noqa: S608


@pytest.mark.parametrize("table", _CANONICAL_TABLES)
def test_database_rejects_delete_on_canonical_tables(db_session: Session, table: str) -> None:
    """Append-only means no deletes either; tombstoning is a later issue."""
    _full_graph(db_session)

    # See the UPDATE test above: hardcoded table name, raw SQL by design.
    with pytest.raises(SQLAlchemyError, match="append-only"):
        db_session.execute(text(f"DELETE FROM {table}"))  # noqa: S608


# ---------------------------------------------------------------------------
# Semantic parent key consistency: FK-valid but key-invalid graphs are refused
# ---------------------------------------------------------------------------


def _insert_version_graph(db_session: Session) -> tuple[SourceArtifact, Document, DocumentVersion]:
    """Insert one artifact, one document and one valid version; return them."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    return artifact, document, version


def _assert_semantic_mismatch(
    excinfo: pytest.ExceptionInfo[SemanticParentKeyError],
    *,
    parent_id: UUID,
    expected_key: str,
    received_key: str,
) -> None:
    """The error must identify the relationship, the referenced parent id, the
    parent's persisted key and the key the child declared."""
    assert excinfo.value.parent_id == parent_id
    assert excinfo.value.expected_key == expected_key
    assert excinfo.value.received_key == received_key


def test_document_version_with_wrong_document_canonical_key_is_rejected(
    db_session: Session,
) -> None:
    """FK-valid but semantically inconsistent: the version references document
    A by id while carrying document B's canonical key. Persistence must refuse
    the write before the inconsistent row can be committed."""
    artifact = _make_artifact()
    document = _make_document()
    other = _make_document(doi=None, pmid=None, pmcid=None, title="An unrelated work")
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document(db_session, other)
    version = _make_version(document, artifact, document_canonical_key=other.canonical_key)

    with pytest.raises(SemanticParentKeyError, match=r"document_version.document_id") as excinfo:
        insert_document_version(db_session, version)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=document.id,
        expected_key=document.canonical_key,
        received_key=other.canonical_key,
    )
    count = db_session.scalar(select(func.count()).select_from(DocumentVersionRecord))
    assert count == 0


def test_document_version_with_wrong_source_artifact_key_is_rejected(
    db_session: Session,
) -> None:
    """The version references artifact A by id while carrying artifact B's
    key — refused before the write."""
    artifact = _make_artifact()
    other_artifact = _make_artifact(content_sha256="c" * 64, byte_size=4096)
    document = _make_document()
    insert_source_artifact(db_session, artifact)
    insert_source_artifact(db_session, other_artifact)
    insert_document(db_session, document)
    version = _make_version(document, artifact, source_artifact_key=other_artifact.artifact_key)

    with pytest.raises(
        SemanticParentKeyError, match=r"document_version.source_artifact_id"
    ) as excinfo:
        insert_document_version(db_session, version)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=artifact.id,
        expected_key=artifact.artifact_key,
        received_key=other_artifact.artifact_key,
    )
    count = db_session.scalar(select(func.count()).select_from(DocumentVersionRecord))
    assert count == 0


def test_section_with_wrong_version_key_is_rejected(db_session: Session) -> None:
    """The section references version A by id while carrying version B's key —
    refused before the write."""
    _, _, version = _insert_version_graph(db_session)
    section = _make_section(version, version_key="f" * 64)

    with pytest.raises(SemanticParentKeyError, match=r"section.document_version_id") as excinfo:
        insert_section(db_session, section)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=version.id,
        expected_key=version.version_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(SectionRecord))
    assert count == 0


def test_passage_with_wrong_version_key_is_rejected(db_session: Session) -> None:
    """The passage references version A by id while carrying version B's key —
    refused before the write."""
    _, _, version = _insert_version_graph(db_session)
    passage = _make_passage(version, version_key="f" * 64)

    with pytest.raises(SemanticParentKeyError, match=r"passage.document_version_id") as excinfo:
        insert_passage(db_session, passage)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=version.id,
        expected_key=version.version_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(PassageRecord))
    assert count == 0


def test_citation_with_wrong_version_key_is_rejected(db_session: Session) -> None:
    """The citation references version A by id while carrying version B's key —
    refused before the write."""
    _, _, version = _insert_version_graph(db_session)
    citation = _make_citation(version, version_key="f" * 64)

    with pytest.raises(SemanticParentKeyError, match=r"citation.document_version_id") as excinfo:
        insert_citation(db_session, citation)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=version.id,
        expected_key=version.version_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(CitationRecord))
    assert count == 0


def test_document_table_with_wrong_version_key_is_rejected(db_session: Session) -> None:
    """The table references version A by id while carrying version B's key —
    refused before the write."""
    _, _, version = _insert_version_graph(db_session)
    table = _make_table(version, version_key="f" * 64)

    with pytest.raises(
        SemanticParentKeyError, match=r"document_table.document_version_id"
    ) as excinfo:
        insert_document_table(db_session, table)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=version.id,
        expected_key=version.version_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(DocumentTableRecord))
    assert count == 0


def test_figure_with_wrong_version_key_is_rejected(db_session: Session) -> None:
    """The figure references version A by id while carrying version B's key —
    refused before the write."""
    _, _, version = _insert_version_graph(db_session)
    figure = _make_figure(version, version_key="f" * 64)

    with pytest.raises(SemanticParentKeyError, match=r"figure.document_version_id") as excinfo:
        insert_figure(db_session, figure)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=version.id,
        expected_key=version.version_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(FigureRecord))
    assert count == 0


def test_citation_resolution_with_wrong_citation_key_is_rejected(
    db_session: Session,
) -> None:
    """The resolution references citation A by id while carrying citation B's
    key — refused before the write."""
    citation, cited = _citation_graph(db_session)
    resolution = _make_resolution(citation, cited, citation_key="f" * 64)

    with pytest.raises(SemanticParentKeyError, match=r"citation_resolution.citation_id") as excinfo:
        insert_citation_resolution(db_session, resolution)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=citation.id,
        expected_key=citation.citation_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(CitationResolutionRecord))
    assert count == 0


def test_citation_resolution_with_wrong_resolved_document_key_is_rejected(
    db_session: Session,
) -> None:
    """The resolution references document A by id while carrying document B's
    canonical key — refused before the write."""
    citation, cited = _citation_graph(db_session)
    resolution = _make_resolution(citation, cited, resolved_document_canonical_key="f" * 64)

    with pytest.raises(
        SemanticParentKeyError, match=r"citation_resolution.resolved_document_id"
    ) as excinfo:
        insert_citation_resolution(db_session, resolution)
    _assert_semantic_mismatch(
        excinfo,
        parent_id=cited.id,
        expected_key=cited.canonical_key,
        received_key="f" * 64,
    )
    count = db_session.scalar(select(func.count()).select_from(CitationResolutionRecord))
    assert count == 0


def test_valid_graph_with_correct_semantic_parent_keys_persists(
    db_session: Session,
) -> None:
    """The positive counterpart: a correctly keyed graph persists through the
    same boundary, and every child reads back bound to its referenced parent."""
    (
        artifact,
        _document,
        version,
        root,
        _child,
        passage,
        citation,
        _table,
        _figure,
        cited,
        resolution,
    ) = _full_graph(db_session)

    assert get_document_version(db_session, version.id) is not None
    assert get_section(db_session, root.id) is not None
    assert list_passages(db_session, version.id, passage.chunker_revision)
    assert list_citations(db_session, version.id)
    assert list_document_tables(db_session, version.id)
    assert list_figures(db_session, version.id)
    assert get_citation_resolutions(db_session, citation.id)
    assert get_document(db_session, cited.id) is not None
    assert get_source_artifact(db_session, artifact.id) is not None
    assert resolution.resolution_key == citation_resolution_key(
        citation.citation_key, cited.canonical_key, "resolver-1"
    )


# ---------------------------------------------------------------------------
# Cross-database semantic reproducibility
# ---------------------------------------------------------------------------


_FullGraphObjects = tuple[
    SourceArtifact,
    Document,
    DocumentVersion,
    Section,
    Section,
    Passage,
    Citation,
    DocumentTable,
    Figure,
    Document,
    CitationResolution,
]


def _canonical_identities(objects: _FullGraphObjects) -> dict[str, Any]:
    """The canonical identities of a persisted semantic graph, by entity."""
    (
        artifact,
        document,
        version,
        root,
        child,
        passage,
        citation,
        table,
        figure,
        _cited,
        _resolution,
    ) = objects
    return {
        "artifact_key": artifact.artifact_key,
        "document_canonical_key": document.canonical_key,
        "version_key": version.version_key,
        "section_keys": (root.section_key, child.section_key),
        "passage_key": passage.passage_key,
        "citation_key": citation.citation_key,
        "document_table_key": table.document_table_key,
        "figure_key": figure.figure_key,
    }


def _stored_canonical_identities(session: Session, objects: _FullGraphObjects) -> dict[str, Any]:
    """Read the same identities back out of the database, by entity."""
    (
        artifact,
        document,
        version,
        root,
        child,
        passage,
        citation,
        table,
        figure,
        _cited,
        _resolution,
    ) = objects
    read_artifact = get_source_artifact(session, artifact.id)
    read_document = get_document(session, document.id)
    read_version = get_document_version(session, version.id)
    read_root = get_section(session, root.id)
    read_child = get_section(session, child.id)
    read_passages = list_passages(session, version.id, passage.chunker_revision)
    read_citations = list_citations(session, version.id)
    read_tables = list_document_tables(session, version.id)
    read_figures = list_figures(session, version.id)
    assert read_artifact is not None
    assert read_document is not None
    assert read_version is not None
    assert read_root is not None
    assert read_child is not None
    assert [item.id for item in read_passages] == [passage.id]
    assert [item.id for item in read_citations] == [citation.id]
    assert [item.id for item in read_tables] == [table.id]
    assert [item.id for item in read_figures] == [figure.id]
    return {
        "artifact_key": read_artifact.artifact_key,
        "document_canonical_key": read_document.canonical_key,
        "version_key": read_version.version_key,
        "section_keys": (read_root.section_key, read_child.section_key),
        "passage_key": read_passages[0].passage_key,
        "citation_key": read_citations[0].citation_key,
        "document_table_key": read_tables[0].document_table_key,
        "figure_key": read_figures[0].figure_key,
    }


def test_same_semantic_graph_produces_identical_canonical_identities_across_databases(
    live_settings: Settings,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core invariant against two independently constructed databases.

    The same semantic graph is persisted twice: once into the canonical
    database (inside the rolled-back test transaction) and once into a
    throwaway scratch database that is created, migrated to head, used and
    dropped within this test. The two runs assign different random surrogate
    ids; every canonical identity — computed before the write and read back
    from each database after it — must be identical.
    """
    first = _full_graph(db_session)
    first_identities = _canonical_identities(first)
    assert _stored_canonical_identities(db_session, first) == first_identities

    database = f"dynamisrag_identity_probe_{uuid4().hex[:12]}"
    maintenance_dsn = _dsn_with_database(str(live_settings.database_url), "postgres")
    with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        scratch_dsn = _dsn_with_database(str(live_settings.database_url), database)
        monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", scratch_dsn)
        config = Config(str(ALEMBIC_INI))
        config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        command.upgrade(config, "head")
        engine = create_database_engine(build_settings(database_url=scratch_dsn))
        try:
            with Session(bind=engine) as scratch_session:
                with scratch_session.begin():
                    second = _full_graph(scratch_session)
                stored_identities = _stored_canonical_identities(scratch_session, second)
        finally:
            engine.dispose()
    finally:
        with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database))
            )

    # Different database instance, different insertion run, different random
    # surrogate ids — identical canonical identities.
    assert first[2].id != second[2].id
    assert _canonical_identities(second) == first_identities
    assert stored_identities == first_identities


def _dsn_with_database(dsn: str, database: str) -> str:
    """Return ``dsn`` pointing at a different database on the same server."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{database}"))
