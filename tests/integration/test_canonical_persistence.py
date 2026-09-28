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
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.db import (
    create_database_engine,
    get_document,
    get_document_version,
    get_section,
    get_source_artifact,
    insert_citation,
    insert_document,
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
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    PassageRecord,
    SectionRecord,
    SourceArtifactRecord,
)
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
from dynamisrag.domain.values import DocumentType

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
_CONTENT_SHA = "a" * 64
_ARTIFACT_SHA = "b" * 64

_CANONICAL_TABLES = (
    "source_artifact",
    "document",
    "document_version",
    "section",
    "passage",
    "citation",
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
        "source_artifact_id": artifact.id,
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
    version: DocumentVersion, *, chunker_revision: str = "chunker-7", **overrides: Any
) -> Passage:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "chunker_revision": chunker_revision,
        "ordinal": 0,
        "text": "A canonical passage of scientific content.",
        "content_sha256": _CONTENT_SHA,
    }
    kwargs.update(overrides)
    return Passage(**kwargs)


def _make_citation(
    version: DocumentVersion,
    *,
    resolved_document: Document | None = None,
    **overrides: Any,
) -> Citation:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
        "ordinal": 0,
        "resolved_document_id": resolved_document.id if resolved_document else None,
    }
    kwargs.update(overrides)
    return Citation(**kwargs)


def _make_table(
    version: DocumentVersion, *, section: Section | None = None, **overrides: Any
) -> DocumentTable:
    kwargs: dict[str, Any] = {
        "document_version_id": version.id,
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
        "ordinal": 1,
        "label": "Figure 1",
        "caption": "Study overview",
        "source_anchor": "fig-1",
        "asset_locator": "s3://dynamisrag-artifacts/figures/fig-1.png",
        "section_id": section.id if section else None,
    }
    kwargs.update(overrides)
    return Figure(**kwargs)


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

    insert_source_artifact(session, artifact)
    insert_document(session, document)
    insert_document_version(session, version)
    insert_section(session, root)
    insert_section(session, child)
    insert_passage(session, passage)
    insert_citation(session, citation)
    insert_document_table(session, table)
    insert_figure(session, figure)
    return artifact, document, version, root, child, passage, citation, table, figure


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


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
    assert read_document.doi == document.doi

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


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def test_unresolved_citation_is_a_valid_first_class_record(
    db_session: Session,
) -> None:
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
    assert record.resolved_document_id is None
    assert record.citation_key == citation.citation_key


def test_resolved_citation_references_a_known_document(db_session: Session) -> None:
    artifact = _make_artifact()
    citing = _make_document(doi="10.1038/nature12373")
    cited = _make_document(doi=None, pmid=None, pmcid=None, title="The cited work")
    version = _make_version(citing, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, citing)
    insert_document(db_session, cited)
    insert_document_version(db_session, version)
    citation = _make_citation(version, ordinal=0, resolved_document=cited)

    record = insert_citation(db_session, citation)

    assert record.resolved_document_id == cited.id


def test_citation_with_unknown_resolved_document_is_rejected(
    db_session: Session,
) -> None:
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(db_session, artifact)
    insert_document(db_session, document)
    insert_document_version(db_session, version)
    citation = _make_citation(version, ordinal=0, resolved_document_id=uuid4())

    with pytest.raises(IntegrityError, match="fk_citation_resolved_document"):
        insert_citation(db_session, citation)


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
