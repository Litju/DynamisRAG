"""JATS canonical materialization proven against live PostgreSQL 18.

Every test runs inside a transaction that is rolled back at teardown, so the
canonical database stays clean. The tests exercise the full RES-133 path —
acquired SourceArtifact bytes -> JatsParser -> JatsCanonicalImporter ->
canonical rows — and prove first-import counts, idempotent reparse,
identifier enrichment, identifier conflict and paragraph ownership.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.db import (
    create_database_engine,
    get_document,
    get_document_identifiers,
    get_document_version,
    get_document_version_by_key,
    insert_document,
    insert_source_artifact,
    list_citations,
    list_document_tables,
    list_document_versions,
    list_figures,
    list_paragraphs,
    list_sections,
)
from dynamisrag.db.models import (
    CitationRecord,
    DocumentIdentifierRecord,
    DocumentRecord,
    DocumentTableRecord,
    DocumentVersionRecord,
    FigureRecord,
    ParagraphRecord,
    PassageRecord,
    SectionRecord,
)
from dynamisrag.domain.contracts import Citation, Document, SourceArtifact
from dynamisrag.domain.identity import document_version_key
from dynamisrag.domain.values import DocumentType
from dynamisrag.jats import (
    JatsCanonicalImporter,
    JatsDocumentIdentityConflict,
    JatsImportCounts,
    JatsImportResult,
    JatsParser,
    JatsSourcePmcidConflict,
)
from tests._support import (
    ALEMBIC_INI,
    JATS_FULL_ARTICLE,
    JATS_SPARSE_ARTICLE,
    REPO_ROOT,
    build_settings,
)

pytestmark = pytest.mark.integration

_NOW: Final[datetime] = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
_ARTIFACT_SHA: Final[str] = "b" * 64
_SPARSE_SHA: Final[str] = "c" * 64

_FULL_COUNTS: Final[JatsImportCounts] = JatsImportCounts(
    sections=3,
    paragraphs=7,
    citations=2,
    tables=1,
    figures=1,
)


@pytest.fixture
def db_session(live_settings: Settings) -> Iterator[Session]:
    """A session whose transaction is always rolled back (never DELETE)."""
    engine = create_database_engine(live_settings)
    session = Session(bind=engine)
    transaction = session.begin()
    try:
        yield session
    finally:
        transaction.rollback()
        session.close()
        engine.dispose()


def _make_artifact(
    content: bytes,
    *,
    pmcid: str = "PMC123456",
    content_sha256: str | None = None,
) -> SourceArtifact:
    return SourceArtifact(
        source_system="europe_pmc",
        source_external_id=pmcid,
        source_uri=f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
        media_type="application/xml",
        content_sha256=content_sha256 or hashlib.sha256(content).hexdigest(),
        byte_size=len(content),
        retrieved_at=_NOW,
        storage_uri=f"file:///dynamisrag-artifacts/sha256/{pmcid}.xml",
    )


def _import(session: Session, artifact: SourceArtifact, content: bytes) -> JatsImportResult:
    return JatsCanonicalImporter(session).import_artifact(artifact, content)


# ---------------------------------------------------------------------------
# Full first import
# ---------------------------------------------------------------------------


def test_full_first_import_materializes_the_canonical_graph(
    db_session: Session,
) -> None:
    """One synthetic acquired SourceArtifact + one JATS document yields the
    expected Document, aliases, Version and child rows with exact counts and
    stable anchors."""
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)

    result = _import(db_session, artifact, JATS_FULL_ARTICLE)

    assert result.created is True
    assert result.counts == _FULL_COUNTS
    _assert_document(db_session, result.document)
    version = _assert_version(db_session, result, artifact)
    sections = _assert_sections(db_session, version)
    _assert_paragraphs(db_session, version, sections)
    _assert_citations(db_session, version)
    _assert_tables(db_session, version, sections)
    _assert_figures(db_session, version, sections)

    # No Passage rows are created: paragraphs are source structure.
    passage_count = db_session.scalar(
        select(func.count())
        .select_from(PassageRecord)
        .where(PassageRecord.document_version_id == version.id)
    )
    assert passage_count == 0


def _assert_document(db_session: Session, document: Document) -> None:
    read = get_document(db_session, document.id)
    assert read is not None
    assert read.canonical_key == "doi:10.1371/journal.pone.03089012"
    assert read.document_type == DocumentType.JOURNAL_ARTICLE.value
    assert read.title == "A synthetic study of things and numbers"
    aliases = get_document_identifiers(db_session, document.id)
    assert {(alias.namespace, alias.normalized_value) for alias in aliases} == {
        ("doi", "10.1371/journal.pone.03089012"),
        ("pmid", "38888888"),
        ("pmcid", "PMC123456"),
    }


def _assert_version(
    db_session: Session, result: JatsImportResult, artifact: SourceArtifact
) -> DocumentVersionRecord:
    version = get_document_version(db_session, result.version.id)
    assert version is not None
    assert version.version_key == result.version.version_key
    assert version.title == "A synthetic study of things and numbers"
    assert version.language == "en"
    assert version.parser_revision == "jats-1.0"
    assert version.normalizer_revision == "norm-1.0"
    assert version.source_artifact_id == artifact.id
    assert version.content_fingerprint == JatsParser().parse(JATS_FULL_ARTICLE).content_fingerprint
    assert version.versioned_metadata == JatsParser().parse(JATS_FULL_ARTICLE).metadata
    return version


def _assert_sections(
    db_session: Session, version: DocumentVersionRecord
) -> Sequence[SectionRecord]:
    sections = list_sections(db_session, version.id)
    assert [(section.structural_path, section.depth) for section in sections] == [
        ("1", 0),
        ("1.1", 1),
        ("2", 0),
    ]
    assert [section.source_anchor for section in sections] == [
        "jats:#sec1",
        "jats:#sec1-1",
        "jats:#sec2",
    ]
    assert sections[1].parent_section_id == sections[0].id
    return sections


def _assert_paragraphs(
    db_session: Session, version: DocumentVersionRecord, sections: Sequence[SectionRecord]
) -> None:
    paragraphs = list_paragraphs(db_session, version.id)
    assert len(paragraphs) == 7
    assert [paragraph.region for paragraph in paragraphs[:2]] == ["front", "front"]
    assert paragraphs[2].region == "body"
    assert paragraphs[2].section_id is None  # direct body paragraph
    assert paragraphs[3].section_id == sections[0].id
    assert paragraphs[0].source_anchor == "jats:/front[1]/article-meta[1]/abstract[1]/p[1]"
    assert paragraphs[0].paragraph_key  # deterministic, derived from version key + anchor


def _assert_citations(db_session: Session, version: DocumentVersionRecord) -> None:
    citations = list_citations(db_session, version.id)
    assert len(citations) == 2
    assert citations[0].source_anchor == "jats:#R1"
    assert citations[0].doi == "10.1016/j.cell.2020.01.001"
    assert citations[0].pmid == "31900000"
    assert citations[1].source_anchor == "jats:#R2"
    assert citations[1].pmcid == "PMC7000000"
    assert citations[1].year == 2019


def _assert_tables(
    db_session: Session, version: DocumentVersionRecord, sections: Sequence[SectionRecord]
) -> None:
    tables = list_document_tables(db_session, version.id)
    assert len(tables) == 1
    assert tables[0].source_anchor == "jats:#T1"
    assert tables[0].section_id == sections[2].id
    assert tables[0].structured_representation["image_only"] is False


def _assert_figures(
    db_session: Session, version: DocumentVersionRecord, sections: Sequence[SectionRecord]
) -> None:
    figures = list_figures(db_session, version.id)
    assert len(figures) == 1
    assert figures[0].source_anchor == "jats:#F1"
    assert figures[0].asset_locator == "figure1.png"
    assert figures[0].section_id == sections[2].id


# ---------------------------------------------------------------------------
# Repeated import (idempotency)
# ---------------------------------------------------------------------------


def test_repeated_import_is_idempotent(db_session: Session) -> None:
    """Same source + same revisions: created=False, same Document id, same
    DocumentVersion id, identical row counts, no duplicate aliases/children."""
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)

    first = _import(db_session, artifact, JATS_FULL_ARTICLE)
    assert first.created is True

    second = _import(db_session, artifact, JATS_FULL_ARTICLE)

    assert second.created is False
    assert second.document.id == first.document.id
    assert second.version.id == first.version.id
    assert second.version.version_key == first.version.version_key
    assert second.counts == first.counts

    version = get_document_version(db_session, first.version.id)
    assert version is not None
    assert list_sections(db_session, version.id)
    assert len(list_paragraphs(db_session, version.id)) == 7
    assert len(list_citations(db_session, version.id)) == 2
    assert len(list_document_tables(db_session, version.id)) == 1
    assert len(list_figures(db_session, version.id)) == 1

    aliases = get_document_identifiers(db_session, first.document.id)
    assert len(aliases) == 3  # no duplicate aliases

    document_count = db_session.scalar(select(func.count()).select_from(DocumentRecord))
    assert document_count == 1
    version_count = db_session.scalar(select(func.count()).select_from(DocumentVersionRecord))
    assert version_count == 1


def test_reparse_of_the_same_artifact_in_a_fresh_transaction_is_idempotent(
    live_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotency across transactions: commit the first import, then reparse
    in a new transaction and prove the same version row is returned.

    Runs against a throwaway scratch database so the canonical development
    database stays clean.
    """
    database = f"dynamisrag_jats_probe_{uuid4().hex[:12]}"
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
            artifact = _make_artifact(JATS_SPARSE_ARTICLE, pmcid="PMC22222222")
            with Session(bind=engine) as session:
                with session.begin():
                    insert_source_artifact(session, artifact)
                    first = _import(session, artifact, JATS_SPARSE_ARTICLE)
                assert first.created is True

            with Session(bind=engine) as session, session.begin():
                second = _import(session, artifact, JATS_SPARSE_ARTICLE)
                assert second.created is False
                assert second.document.id == first.document.id
                assert second.version.id == first.version.id
                assert second.counts == JatsImportCounts(
                    sections=0, paragraphs=1, citations=0, tables=0, figures=0
                )
                version = get_document_version(session, first.version.id)
                assert version is not None
                assert len(list_paragraphs(session, version.id)) == 1
        finally:
            engine.dispose()
    finally:
        with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database))
            )


# ---------------------------------------------------------------------------
# Identifier enrichment and conflict
# ---------------------------------------------------------------------------


def test_identifier_enrichment_appends_alias_without_reidentifying(
    db_session: Session,
) -> None:
    """An existing PMCID-identified Document + a newly parsed DOI: same
    Document canonical key, DOI alias appended, no replacement Document.

    The second reparse after enrichment is the regression proof: the
    enriched Document must reconstruct with its original persisted canonical
    key (not the DOI), so the same DocumentVersion is found instead of a
    duplicate graph being materialized.
    """
    existing = Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        doi=None,
        pmid=None,
        pmcid="PMC123456",
        title="A synthetic study of things and numbers",
    )
    insert_document(db_session, existing)
    before = get_document(db_session, existing.id)
    assert before is not None
    assert before.canonical_key == "pmcid:PMC123456"

    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)
    result = _import(db_session, artifact, JATS_FULL_ARTICLE)

    assert result.created is True
    assert result.document.id == existing.id
    assert result.document.canonical_key == "pmcid:PMC123456"

    after = get_document(db_session, existing.id)
    assert after is not None
    assert after.canonical_key == "pmcid:PMC123456"  # never re-identified
    aliases = get_document_identifiers(db_session, existing.id)
    assert {(alias.namespace, alias.normalized_value) for alias in aliases} == {
        ("pmcid", "PMC123456"),
        ("doi", "10.1371/journal.pone.03089012"),
        ("pmid", "38888888"),
    }
    document_count = db_session.scalar(select(func.count()).select_from(DocumentRecord))
    assert document_count == 1

    # Reparse after enrichment: the persisted canonical key is preserved, so
    # the same DocumentVersion is returned — no duplicate graph.
    reparsed = _import(db_session, artifact, JATS_FULL_ARTICLE)

    assert reparsed.created is False
    assert reparsed.document.id == existing.id
    assert reparsed.document.canonical_key == "pmcid:PMC123456"
    assert reparsed.version.id == result.version.id
    assert reparsed.version.version_key == result.version.version_key
    assert reparsed.counts == result.counts

    version_count = db_session.scalar(select(func.count()).select_from(DocumentVersionRecord))
    assert version_count == 1
    paragraph_count = db_session.scalar(select(func.count()).select_from(ParagraphRecord))
    assert paragraph_count == 7
    alias_count = len(get_document_identifiers(db_session, existing.id))
    assert alias_count == 3  # no duplicate aliases


def test_identifier_conflict_fails_without_a_partial_graph(
    db_session: Session,
) -> None:
    """DOI and PMCID already resolve to different Documents: explicit
    conflict error, no partial canonical graph."""
    doi_document = Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        doi="10.1371/journal.pone.03089012",
        title="A synthetic study of things and numbers",
    )
    pmcid_document = Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        pmcid="PMC123456",
        title="A synthetic study of things and numbers",
    )
    insert_document(db_session, doi_document)
    insert_document(db_session, pmcid_document)

    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)

    with pytest.raises(JatsDocumentIdentityConflict, match="different documents"):
        _import(db_session, artifact, JATS_FULL_ARTICLE)

    # No partial canonical graph: no versions, sections, paragraphs,
    # citations, tables or figures were materialized.
    assert db_session.scalar(select(func.count()).select_from(DocumentVersionRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(SectionRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(ParagraphRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentIdentifierRecord)) == 2


# ---------------------------------------------------------------------------
# Europe PMC source PMCID provenance
# ---------------------------------------------------------------------------


def test_artifact_pmcid_matching_the_xml_is_valid(db_session: Session) -> None:
    """artifact PMC123 + XML PMC123: the explicit XML PMCID confirms the
    acquired provenance and the import succeeds."""
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)

    result = _import(db_session, artifact, JATS_FULL_ARTICLE)

    assert result.created is True
    assert result.document.canonical_key == "doi:10.1371/journal.pone.03089012"
    aliases = get_document_identifiers(db_session, result.document.id)
    assert ("pmcid", "PMC123456") in {
        (alias.namespace, alias.normalized_value) for alias in aliases
    }


def test_artifact_pmcid_conflicting_with_the_xml_is_rejected(db_session: Session) -> None:
    """artifact PMC123 + XML PMC999: explicit fatal source/identity conflict
    raised before canonical materialization.

    No partial graph is materialized, and the two PMCIDs are never attached
    as aliases of one Document.
    """
    xml = JATS_FULL_ARTICLE.replace(
        b'<article-id pub-id-type="pmcid">PMC123456</article-id>',
        b'<article-id pub-id-type="pmcid">PMC999999</article-id>',
    )
    artifact = _make_artifact(xml)
    insert_source_artifact(db_session, artifact)

    with pytest.raises(JatsSourcePmcidConflict, match="PMC999999"):
        _import(db_session, artifact, xml)

    assert db_session.scalar(select(func.count()).select_from(DocumentRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentIdentifierRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentVersionRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(SectionRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(ParagraphRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(CitationRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentTableRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(FigureRecord)) == 0


# ---------------------------------------------------------------------------
# Savepoint atomicity
# ---------------------------------------------------------------------------


def test_mid_materialization_failure_rolls_back_only_the_import_savepoint(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure halfway through materialization rolls back the importer's
    savepoint only: no partial canonical graph remains, and the caller's
    outer transaction stays usable.

    The failure is forced at the citation insert — after the Document, its
    aliases, the DocumentVersion, the Sections and the Paragraphs have all
    been flushed — so it is genuinely mid-materialization.
    """
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)

    def failing_insert_citation(session: Session, citation: Citation) -> None:
        raise RuntimeError("forced mid-materialization failure")

    monkeypatch.setattr("dynamisrag.jats.importer.insert_citation", failing_insert_citation)

    with pytest.raises(RuntimeError, match="forced mid-materialization failure"):
        _import(db_session, artifact, JATS_FULL_ARTICLE)

    # The imported graph is gone entirely: every canonical row flushed before
    # the failure is rolled back with the savepoint.
    assert db_session.scalar(select(func.count()).select_from(DocumentRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentIdentifierRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentVersionRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(SectionRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(ParagraphRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(CitationRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(DocumentTableRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(FigureRecord)) == 0

    # The outer transaction remains usable: an unrelated row can still be
    # written and read.
    unrelated = Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        title="An unrelated document",
    )
    insert_document(db_session, unrelated)
    read = get_document(db_session, unrelated.id)
    assert read is not None
    assert read.title == "An unrelated document"


# ---------------------------------------------------------------------------
# Paragraph ownership
# ---------------------------------------------------------------------------


def test_paragraph_with_same_version_section_succeeds(db_session: Session) -> None:
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)
    result = _import(db_session, artifact, JATS_FULL_ARTICLE)

    paragraphs = list_paragraphs(db_session, result.version.id)
    owned = [paragraph for paragraph in paragraphs if paragraph.section_id is not None]
    # intro, background, list and methods paragraphs; the two abstract
    # paragraphs and the direct body paragraph are sectionless.
    assert len(owned) == 4
    sections = {section.id for section in list_sections(db_session, result.version.id)}
    assert {paragraph.section_id for paragraph in owned} <= sections


def test_paragraph_with_section_from_another_version_is_rejected(
    db_session: Session,
) -> None:
    """The composite section foreign key rejects a cross-version section."""
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)
    first = _import(db_session, artifact, JATS_FULL_ARTICLE)
    changed = JATS_FULL_ARTICLE.replace(b"Intro paragraph one.", b"Intro paragraph two.")
    changed_artifact = _make_artifact(changed)
    insert_source_artifact(db_session, changed_artifact)
    second = _import(db_session, changed_artifact, changed)
    assert second.created is True
    assert second.version.id != first.version.id

    section = list_sections(db_session, first.version.id)[0]
    cross_version_paragraph = ParagraphRecord(
        id=uuid4(),
        document_version_id=second.version.id,
        section_id=section.id,
        section_document_version_id=second.version.id,
        ordinal=99,
        region="body",
        source_anchor="jats:/body[1]/p[99]",
        text="A paragraph owned by a section of another version.",
        content_sha256="d" * 64,
        paragraph_key="e" * 64,
    )
    db_session.add(cross_version_paragraph)

    with pytest.raises(IntegrityError, match="fk_paragraph_section"):
        db_session.flush()


# ---------------------------------------------------------------------------
# Sparse article
# ---------------------------------------------------------------------------


def test_sparse_article_import(db_session: Session) -> None:
    artifact = _make_artifact(JATS_SPARSE_ARTICLE, pmcid="PMC22222222")
    insert_source_artifact(db_session, artifact)

    result = _import(db_session, artifact, JATS_SPARSE_ARTICLE)

    assert result.created is True
    assert result.counts == JatsImportCounts(
        sections=0, paragraphs=1, citations=0, tables=0, figures=0
    )
    assert result.document.canonical_key == "pmcid:PMC22222222"
    assert result.version.language == "und"
    assert result.version.title == "A sparse article"
    paragraph = list_paragraphs(db_session, result.version.id)[0]
    assert paragraph.text == "Only paragraph."
    assert paragraph.region == "body"
    assert paragraph.section_id is None


def test_artifact_pmcid_identifies_the_document_when_the_xml_omits_it(
    db_session: Session,
) -> None:
    """The acquired SourceArtifact PMCID is strong provenance: even when the
    XML does not repeat it, it is a known Document identifier."""
    xml = JATS_FULL_ARTICLE.replace(
        b'<article-id pub-id-type="pmcid">PMC123456</article-id>\n      ', b""
    )
    artifact = _make_artifact(xml)
    insert_source_artifact(db_session, artifact)

    result = _import(db_session, artifact, xml)

    assert result.created is True
    # DOI is the strongest parsed identifier, so it fixes the canonical key;
    # the artifact PMCID is attached as an alias.
    assert result.document.canonical_key == "doi:10.1371/journal.pone.03089012"
    aliases = get_document_identifiers(db_session, result.document.id)
    assert {(alias.namespace, alias.normalized_value) for alias in aliases} == {
        ("doi", "10.1371/journal.pone.03089012"),
        ("pmid", "38888888"),
        ("pmcid", "PMC123456"),
    }


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------


def test_version_key_is_the_deterministic_identity(db_session: Session) -> None:
    artifact = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact)
    result = _import(db_session, artifact, JATS_FULL_ARTICLE)

    expected_key = document_version_key(
        result.document.canonical_key,
        artifact.artifact_key,
        "jats-1.0",
        "norm-1.0",
        JatsParser().parse(JATS_FULL_ARTICLE).content_fingerprint,
    )
    assert result.version.version_key == expected_key
    assert get_document_version_by_key(db_session, expected_key) is not None


def test_different_bytes_produce_a_different_version(db_session: Session) -> None:
    """The same PMCID with different bytes is a different artifact and a
    different version of the same logical Document."""
    artifact_one = _make_artifact(JATS_FULL_ARTICLE)
    insert_source_artifact(db_session, artifact_one)
    first = _import(db_session, artifact_one, JATS_FULL_ARTICLE)

    changed = JATS_FULL_ARTICLE.replace(b"Intro paragraph one.", b"Intro paragraph two.")
    artifact_two = _make_artifact(changed)
    insert_source_artifact(db_session, artifact_two)
    second = _import(db_session, artifact_two, changed)

    assert second.created is True
    assert second.document.id == first.document.id  # same logical document
    assert second.version.id != first.version.id
    assert second.version.content_fingerprint != first.version.content_fingerprint
    versions = list_document_versions(db_session, first.document.id)
    assert len(versions) == 2


def _dsn_with_database(dsn: str, database: str) -> str:
    """Return ``dsn`` pointing at a different database on the same server."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{database}"))
