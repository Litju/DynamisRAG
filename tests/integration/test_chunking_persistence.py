"""Persistence invariants of deterministic structure-aware chunking (RES-134),
proven against live PostgreSQL 18.

The tests exercise the full flow: canonical DocumentVersion + Sections +
Paragraphs -> pure StructureAwareChunker -> PassageManifest ->
PassageMaterializer -> Passage + PassageSourceSpan rows. Idempotency,
multiple chunker revisions, savepoint atomicity and cross-database manifest
determinism are proven here.
"""

from __future__ import annotations

import hashlib
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
from sqlalchemy.orm import Session

from dynamisrag.chunking.config import ChunkerConfig
from dynamisrag.chunking.materializer import PassageMaterializer
from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.config import Settings
from dynamisrag.db import (
    create_database_engine,
    insert_document,
    insert_document_version,
    insert_paragraph,
    insert_section,
    insert_source_artifact,
    list_passage_source_spans,
    list_passages,
)
from dynamisrag.db.models import (
    PassageSourceSpanRecord,
)
from dynamisrag.domain.contracts import (
    Document,
    DocumentVersion,
    Paragraph,
    Passage,
    Section,
    SourceArtifact,
)
from dynamisrag.domain.identity import document_version_key, source_artifact_key
from dynamisrag.domain.values import DocumentType, ParagraphRegion
from tests._support import ALEMBIC_INI, REPO_ROOT, build_settings

pytestmark = pytest.mark.integration


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


_NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
_ARTIFACT_SHA = "b" * 64
_ARTIFACT_KEY = source_artifact_key("europe_pmc", "PMC123456", _ARTIFACT_SHA)
_DOCUMENT_KEY = "doi:10.1038/nature12373"
_CONTENT_SHA = "a" * 64
_VERSION_KEY = document_version_key(
    _DOCUMENT_KEY, _ARTIFACT_KEY, "jats-1.0", "norm-1.0", _CONTENT_SHA
)


def _make_artifact() -> SourceArtifact:
    return SourceArtifact(
        source_system="europe_pmc",
        source_external_id="PMC123456",
        source_uri="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fulltextXML",
        media_type="application/xml",
        content_sha256=_ARTIFACT_SHA,
        byte_size=2048,
        retrieved_at=_NOW,
        storage_uri="s3://dynamisrag-artifacts/europe_pmc/PMC123456.xml",
    )


def _make_document() -> Document:
    return Document(
        document_type=DocumentType.JOURNAL_ARTICLE,
        doi="10.1038/nature12373",
        pmid="23656234",
        pmcid="PMC3656234",
        title="A foundational study",
    )


def _make_version(document: Document, artifact: SourceArtifact) -> DocumentVersion:
    return DocumentVersion(
        document_id=document.id,
        document_canonical_key=document.canonical_key,
        source_artifact_id=artifact.id,
        source_artifact_key=artifact.artifact_key,
        parser_revision="jats-1.0",
        normalizer_revision="norm-1.0",
        content_fingerprint=_CONTENT_SHA,
        title="A foundational study",
        language="en",
        versioned_metadata={"journal": "Nature"},
        created_at=_NOW,
    )


def _make_section(
    version: DocumentVersion,
    structural_path: str,
    *,
    title: str,
) -> Section:
    return Section(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=int(structural_path.rsplit(".", 1)[-1]),
        depth=structural_path.count("."),
        title=title,
        source_anchor=f"jats:/body[1]/sec[{structural_path}]",
        structural_path=structural_path,
    )


def _make_paragraph(
    version: DocumentVersion,
    ordinal: int,
    text_value: str,
    *,
    section: Section | None = None,
    region: ParagraphRegion = ParagraphRegion.BODY,
) -> Paragraph:
    return Paragraph(
        document_version_id=version.id,
        version_key=version.version_key,
        ordinal=ordinal,
        region=region,
        source_anchor=f"jats:/body[1]/p[{ordinal}]",
        text=text_value,
        content_sha256=hashlib.sha256(text_value.encode("utf-8")).hexdigest(),
        section_id=section.id if section is not None else None,
    )


_OVERSIZED_TEXT = (
    "The first sentence of the oversized paragraph has several tokens in it. "
    "The second sentence of the oversized paragraph also has several tokens. "
    "The third sentence of the oversized paragraph has several tokens too."
)


def _sample_graph(session: Session) -> tuple[DocumentVersion, list[Section], list[Paragraph]]:
    """Insert one canonical graph: three sections, five paragraphs (one
    oversized), one sectionless paragraph. Semantic keys are deterministic;
    surrogate ids are fresh random values."""
    artifact = _make_artifact()
    document = _make_document()
    version = _make_version(document, artifact)
    insert_source_artifact(session, artifact)
    insert_document(session, document)
    insert_document_version(session, version)

    sections = [
        _make_section(version, "1", title="Introduction"),
        _make_section(version, "2", title="Methods"),
        _make_section(version, "3", title="Results"),
    ]
    for section in sections:
        insert_section(session, section)

    paragraphs = [
        _make_paragraph(version, 0, "Introduction paragraph one.", section=sections[0]),
        _make_paragraph(version, 1, "Introduction paragraph two.", section=sections[0]),
        _make_paragraph(version, 2, "Methods paragraph.", section=sections[1]),
        _make_paragraph(version, 3, _OVERSIZED_TEXT, section=sections[1]),
        _make_paragraph(version, 4, "Sectionless body paragraph."),
    ]
    for paragraph in paragraphs:
        insert_paragraph(session, paragraph)
    return version, sections, paragraphs


def _span_rows(db_session: Session, passage_ids: list[UUID]) -> list[PassageSourceSpanRecord]:
    return list(list_passage_source_spans(db_session, passage_ids))


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------


def test_chunking_produces_passages_and_exact_ordered_spans(db_session: Session) -> None:
    version, sections, paragraphs = _sample_graph(db_session)
    chunker = StructureAwareChunker()
    manifest = chunker.plan(version, sections, paragraphs)

    result = PassageMaterializer(db_session).materialize(version, manifest)

    assert result.created is True
    assert len(result.passages) == len(manifest.passages)
    assert len(result.source_spans) == sum(
        len(passage.source_spans) for passage in manifest.passages
    )
    for passage, planned in zip(result.passages, manifest.passages, strict=True):
        assert passage.passage_key == planned.passage_key
        assert passage.text == planned.text
        assert passage.content_sha256 == planned.content_sha256
        assert passage.token_count == planned.token_count
        assert passage.ordinal == planned.ordinal
        assert passage.source_anchor == planned.primary_source_anchor

    persisted_spans = _span_rows(db_session, [passage.id for passage in result.passages])
    assert len(persisted_spans) == len(result.source_spans)
    for span in persisted_spans:
        assert span.document_version_id == version.id
        assert span.end_char > span.start_char >= 0
        assert span.span_key


def test_materialized_spans_reference_consistent_same_version_parents(
    db_session: Session,
) -> None:
    """Every persisted span's passage and paragraph belong to the span's
    document version — the composite foreign keys make this structural."""
    version, sections, paragraphs = _sample_graph(db_session)
    chunker = StructureAwareChunker()
    manifest = chunker.plan(version, sections, paragraphs)
    result = PassageMaterializer(db_session).materialize(version, manifest)

    passages = {passage.id: passage for passage in result.passages}
    paragraphs_by_id = {paragraph.id: paragraph for paragraph in paragraphs}
    for span in _span_rows(db_session, [passage.id for passage in result.passages]):
        assert span.passage_id in passages
        assert span.paragraph_id in paragraphs_by_id
        assert passages[span.passage_id].document_version_id == version.id
        assert paragraphs_by_id[span.paragraph_id].document_version_id == version.id


def test_oversized_paragraph_is_split_into_sentence_spans(db_session: Session) -> None:
    version, sections, paragraphs = _sample_graph(db_session)
    config = ChunkerConfig(target_tokens=10, max_tokens=15, min_tokens=5)
    chunker = StructureAwareChunker(config)
    manifest = chunker.plan(version, sections, paragraphs)
    result = PassageMaterializer(db_session).materialize(version, manifest)

    oversized = paragraphs[3]
    oversized_spans = [
        span for span in result.source_spans if span.paragraph_key == oversized.paragraph_key
    ]
    assert len(oversized_spans) == 3
    for span in oversized_spans:
        assert span.start_char < span.end_char <= len(oversized.text)
        assert oversized.text[span.start_char : span.end_char]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_repeated_materialization_is_idempotent(db_session: Session) -> None:
    version, sections, paragraphs = _sample_graph(db_session)
    chunker = StructureAwareChunker()
    manifest = chunker.plan(version, sections, paragraphs)
    materializer = PassageMaterializer(db_session)

    first = materializer.materialize(version, manifest)
    second = materializer.materialize(version, manifest)

    assert first.created is True
    assert second.created is False
    assert [passage.id for passage in first.passages] == [passage.id for passage in second.passages]
    assert first.manifest.manifest_bytes == second.manifest.manifest_bytes
    assert first.manifest.manifest_sha256 == second.manifest.manifest_sha256
    # Same persisted span ids, compared in deterministic span-key order (the
    # two lists are produced in different orders: insertion vs passage_id).
    first_span_ids = [span.id for span in sorted(first.source_spans, key=lambda s: s.span_key)]
    second_span_ids = [span.id for span in sorted(second.source_spans, key=lambda s: s.span_key)]
    assert first_span_ids == second_span_ids

    assert len(list_passages(db_session, version.id, manifest.chunker_revision)) == len(
        first.passages
    )
    span_count = db_session.scalar(select(func.count()).select_from(PassageSourceSpanRecord))
    assert span_count == len(first.source_spans)


def test_existing_set_reconstructing_to_a_different_manifest_fails_explicitly(
    db_session: Session,
) -> None:
    """A persisted passage set that contradicts the manifest being
    materialized is an explicit failure — never silently returned."""
    from dynamisrag.chunking.errors import ChunkerRevisionConflictError
    from dynamisrag.db import insert_passage

    version, sections, paragraphs = _sample_graph(db_session)
    materializer = PassageMaterializer(db_session)
    manifest = StructureAwareChunker().plan(version, sections, paragraphs)
    materializer.materialize(version, manifest)

    # Tamper: insert an extra passage row under the same chunker revision
    # that the chunker never produced. The existing set now reconstructs to a
    # different manifest and must fail explicitly.
    bogus = Passage(
        document_version_id=version.id,
        version_key=version.version_key,
        chunker_revision=manifest.chunker_revision,
        ordinal=len(manifest.passages),
        text="A passage that the chunker never produced.",
        content_sha256="a" * 64,
    )
    insert_passage(db_session, bogus)

    with pytest.raises(ChunkerRevisionConflictError, match="immutable"):
        materializer.materialize(version, manifest)


# ---------------------------------------------------------------------------
# Multiple chunker revisions
# ---------------------------------------------------------------------------


def test_different_chunker_revisions_coexist_on_one_version(db_session: Session) -> None:
    version, sections, paragraphs = _sample_graph(db_session)
    materializer = PassageMaterializer(db_session)
    manifest_a = StructureAwareChunker(
        ChunkerConfig(target_tokens=300, max_tokens=400, min_tokens=50)
    ).plan(version, sections, paragraphs)
    manifest_b = StructureAwareChunker(
        ChunkerConfig(target_tokens=300, max_tokens=450, min_tokens=50)
    ).plan(version, sections, paragraphs)
    assert manifest_a.chunker_revision != manifest_b.chunker_revision

    result_a = materializer.materialize(version, manifest_a)
    result_b = materializer.materialize(version, manifest_b)

    assert result_a.created is True
    assert result_b.created is True
    assert len(list_passages(db_session, version.id, manifest_a.chunker_revision)) == len(
        result_a.passages
    )
    assert len(list_passages(db_session, version.id, manifest_b.chunker_revision)) == len(
        result_b.passages
    )

    # Set A remains immutable after B coexists.
    again_a = materializer.materialize(version, manifest_a)
    assert again_a.created is False
    assert [passage.id for passage in again_a.passages] == [
        passage.id for passage in result_a.passages
    ]


# ---------------------------------------------------------------------------
# Savepoint atomicity
# ---------------------------------------------------------------------------


def test_forced_failure_leaves_no_partial_passage_set(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-chunk persistence failure rolls the whole passage set back through
    the savepoint; the outer transaction remains usable."""
    import dynamisrag.chunking.materializer as materializer_module

    version, sections, paragraphs = _sample_graph(db_session)
    chunker = StructureAwareChunker()
    manifest = chunker.plan(version, sections, paragraphs)

    calls = {"count": 0}
    original = materializer_module.insert_passage_source_span

    def failing(session: Session, span: Any) -> PassageSourceSpanRecord:
        calls["count"] += 1
        if calls["count"] > 2:
            raise RuntimeError("forced mid-chunk failure")
        return original(session, span)

    monkeypatch.setattr(materializer_module, "insert_passage_source_span", failing)
    materializer = PassageMaterializer(db_session)
    with pytest.raises(RuntimeError, match="forced mid-chunk failure"):
        materializer.materialize(version, manifest)

    assert list_passages(db_session, version.id, manifest.chunker_revision) == []
    assert db_session.scalar(select(func.count()).select_from(PassageSourceSpanRecord)) == 0

    # The outer transaction remains usable.
    assert db_session.execute(text("SELECT 1")).scalar_one() == 1

    # A clean re-run succeeds.
    monkeypatch.setattr(materializer_module, "insert_passage_source_span", original)
    result = materializer.materialize(version, manifest)
    assert result.created is True
    assert len(result.passages) == len(manifest.passages)


# ---------------------------------------------------------------------------
# Cross-database determinism
# ---------------------------------------------------------------------------


def _dsn_with_database(dsn: str, database: str) -> str:
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{database}"))


def test_identical_semantic_graph_produces_identical_manifests_across_databases(
    live_settings: Settings,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core determinism invariant: the same canonical graph chunked under
    the same configuration produces byte-identical manifests in two
    independently constructed databases with different surrogate UUIDs."""
    version, sections, paragraphs = _sample_graph(db_session)
    chunker = StructureAwareChunker()
    manifest_a = chunker.plan(version, sections, paragraphs)

    database = f"dynamisrag_chunking_probe_{uuid4().hex[:12]}"
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
                    version_b, sections_b, paragraphs_b = _sample_graph(scratch_session)
                manifest_b = chunker.plan(version_b, sections_b, paragraphs_b)
        finally:
            engine.dispose()
    finally:
        with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database))
            )

    # Different database instance, different surrogate ids — identical
    # semantic output.
    assert version_b.id != version.id
    assert version_b.version_key == version.version_key
    assert manifest_b.manifest_bytes == manifest_a.manifest_bytes
    assert manifest_b.manifest_sha256 == manifest_a.manifest_sha256
    assert [p.passage_key for p in manifest_b.passages] == [
        p.passage_key for p in manifest_a.passages
    ]
    assert [p.text for p in manifest_b.passages] == [p.text for p in manifest_a.passages]
    assert [p.token_count for p in manifest_b.passages] == [
        p.token_count for p in manifest_a.passages
    ]
    spans_a = [
        (span.paragraph_key, span.paragraph_source_anchor, span.start_char, span.end_char)
        for passage in manifest_a.passages
        for span in passage.source_spans
    ]
    spans_b = [
        (span.paragraph_key, span.paragraph_source_anchor, span.start_char, span.end_char)
        for passage in manifest_b.passages
        for span in passage.source_spans
    ]
    assert spans_b == spans_a
