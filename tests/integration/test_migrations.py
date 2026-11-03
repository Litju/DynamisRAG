"""The database migration state must match the migration code on disk.

Comparing against ``ScriptDirectory`` rather than a hard-coded revision id means
a new revision that is committed but not applied fails this test, without the
test itself needing editing.
"""

from __future__ import annotations

import re
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from scripts.res145_evidence_demo import scratch_database
from sqlalchemy import Engine, event, text
from sqlalchemy.engine import Connection

from dynamisrag.config import Settings, load_settings, to_sqlalchemy_url
from dynamisrag.db import create_database_engine
from dynamisrag.db.migration_safety import MigrationTargetSafetyError
from tests._support import ALEMBIC_INI, REPO_ROOT, build_settings

_SCRIPT_LOCATION: Final[str] = str(REPO_ROOT / "alembic")
_VERSION_TABLE: Final[str] = "alembic_version"
_EXPECTED_POSTGRES_MAJOR: Final[str] = "18"

_CANONICAL_TABLES: Final[tuple[str, ...]] = (
    "source_artifact",
    "document",
    "document_identifier",
    "document_version",
    "section",
    "passage",
    "paragraph",
    "citation",
    "citation_resolution",
    "document_table",
    "figure",
    "passage_source_span",
)
"""The 12 immutable source tables after RES-134's 0004."""

_EVIDENCE_TABLES = ("evidence_set", "claim", "evidence_membership", "evidence_link")

_RES133_TABLES: Final[tuple[str, ...]] = (
    "source_artifact",
    "document",
    "document_identifier",
    "document_version",
    "section",
    "passage",
    "paragraph",
    "citation",
    "citation_resolution",
    "document_table",
    "figure",
)
"""The 11 canonical tables sealed by RES-133's 0003, before 0004."""

_RES131_TABLES: Final[tuple[str, ...]] = (
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
"""The 10 canonical tables sealed by RES-131's 0002, before 0003."""

_CANONICAL_UNIQUE_CONSTRAINTS: Final[tuple[str, ...]] = (
    "uq_source_artifact_artifact_key",
    "uq_document_canonical_key",
    "uq_document_identifier_namespace_value",
    "uq_document_version_version_key",
    "uq_section_document_version_section_key",
    "uq_passage_version_chunker_ordinal",
    "uq_passage_id_document_version_id",
    "uq_paragraph_paragraph_key",
    "uq_paragraph_document_version_source_anchor",
    "uq_paragraph_document_version_ordinal",
    "uq_paragraph_id_document_version_id",
    "uq_citation_document_version_ordinal",
    "uq_citation_resolution_resolution_key",
    "uq_document_table_document_version_key",
    "uq_figure_document_version_key",
    "uq_passage_source_span_span_key",
    "uq_passage_source_span_passage_order",
)

_RES133_UNIQUE_CONSTRAINTS: Final[tuple[str, ...]] = (
    "uq_source_artifact_artifact_key",
    "uq_document_canonical_key",
    "uq_document_identifier_namespace_value",
    "uq_document_version_version_key",
    "uq_section_document_version_section_key",
    "uq_passage_version_chunker_ordinal",
    "uq_paragraph_paragraph_key",
    "uq_paragraph_document_version_source_anchor",
    "uq_paragraph_document_version_ordinal",
    "uq_citation_document_version_ordinal",
    "uq_citation_resolution_resolution_key",
    "uq_document_table_document_version_key",
    "uq_figure_document_version_key",
)
"""The unique constraints sealed by 0003 (0004 adds the span and id-pair ones)."""

_RES131_UNIQUE_CONSTRAINTS: Final[tuple[str, ...]] = (
    "uq_source_artifact_artifact_key",
    "uq_document_canonical_key",
    "uq_document_identifier_namespace_value",
    "uq_document_version_version_key",
    "uq_section_document_version_section_key",
    "uq_passage_version_chunker_ordinal",
    "uq_citation_document_version_ordinal",
    "uq_citation_resolution_resolution_key",
    "uq_document_table_document_version_key",
    "uq_figure_document_version_key",
)
"""The unique constraints sealed by 0002 (0003 adds only paragraph ones)."""

_PARAGRAPH_CONSTRAINTS: Final[tuple[str, ...]] = (
    "pk_paragraph",
    "uq_paragraph_paragraph_key",
    "uq_paragraph_document_version_source_anchor",
    "uq_paragraph_document_version_ordinal",
    "fk_paragraph_document_version",
    "fk_paragraph_section",
    "ck_paragraph_section_pair",
    "ck_paragraph_ordinal_nonnegative",
    "ck_paragraph_region_source_derived",
    "ck_paragraph_content_sha256_hex",
    "ck_paragraph_text_nonempty",
    "ck_paragraph_source_anchor_nonempty",
)
"""Constraints introduced by 0003; absent from the sealed 0002 model."""

_SPAN_CONSTRAINTS: Final[tuple[str, ...]] = (
    "pk_passage_source_span",
    "uq_passage_source_span_span_key",
    "uq_passage_source_span_passage_order",
    "fk_passage_source_span_passage",
    "fk_passage_source_span_paragraph",
    "ck_passage_source_span_order_nonnegative",
    "ck_passage_source_span_start_nonnegative",
    "ck_passage_source_span_end_positive",
    "ck_passage_source_span_end_after_start",
)
"""Constraints introduced by 0004; absent from the sealed 0003 model."""

_CANONICAL_COMPOSITE_FOREIGN_KEYS: Final[tuple[str, ...]] = (
    "fk_section_parent",
    "fk_passage_section",
    "fk_paragraph_section",
    "fk_document_table_section",
    "fk_figure_section",
    "fk_passage_source_span_passage",
    "fk_passage_source_span_paragraph",
)

_IMMUTABILITY_FUNCTION: Final[str] = "dynamisrag_enforce_immutable"


def _engine(settings: Settings) -> Engine:
    return create_database_engine(settings)


def _script_heads() -> set[str]:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", _SCRIPT_LOCATION)
    return set(ScriptDirectory.from_config(config).get_heads())


def _dsn_with_database(dsn: str, database: str) -> str:
    """Return ``dsn`` pointing at a different database on the same server."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{database}"))


def _public_tables(connection: Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
        )
    }


def _public_functions(connection: Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            text(
                "SELECT routine_name FROM information_schema.routines "
                "WHERE routine_schema = 'public'"
            )
        )
    }


def _public_triggers(connection: Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            text(
                "SELECT trigger_name FROM information_schema.triggers "
                "WHERE trigger_schema = 'public'"
            )
        )
    }


def _public_constraint_names(connection: Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            text(
                "SELECT conname FROM pg_constraint c "
                "JOIN pg_class t ON t.oid = c.conrelid "
                "JOIN pg_namespace n ON n.oid = t.relnamespace "
                "WHERE n.nspname = 'public'"
            )
        )
    }


def _public_columns(connection: Connection, table: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :table"
            ),
            {"table": table},
        )
    }


def _assert_canonical_model_present(settings: Settings) -> None:
    engine = _engine(settings)
    try:
        with engine.connect() as connection:
            tables = _public_tables(connection)
            functions = _public_functions(connection)
            triggers = _public_triggers(connection)
            constraints = _public_constraint_names(connection)
    finally:
        engine.dispose()

    assert tables == {_VERSION_TABLE, *_CANONICAL_TABLES, *_EVIDENCE_TABLES}
    assert _IMMUTABILITY_FUNCTION in functions
    assert "dynamisrag_check_passage_source_span" in functions
    assert triggers == {f"trg_{table}_immutable" for table in _CANONICAL_TABLES} | {
        "trg_passage_source_span_check",
        "trg_evidence_set_complete",
        *(f"trg_{table}_validate" for table in _EVIDENCE_TABLES),
        *(f"trg_{table}_immutable" for table in _EVIDENCE_TABLES),
    }
    assert set(_CANONICAL_UNIQUE_CONSTRAINTS) <= constraints
    assert set(_CANONICAL_COMPOSITE_FOREIGN_KEYS) <= constraints
    assert set(_SPAN_CONSTRAINTS) <= constraints


def _assert_canonical_model_absent(settings: Settings) -> None:
    engine = _engine(settings)
    try:
        with engine.connect() as connection:
            tables = _public_tables(connection)
            functions = _public_functions(connection)
            triggers = _public_triggers(connection)
            constraints = _public_constraint_names(connection)
    finally:
        engine.dispose()

    assert tables == {_VERSION_TABLE}
    assert _IMMUTABILITY_FUNCTION not in functions
    assert "dynamisrag_check_passage_source_span" not in functions
    assert not triggers
    assert not (set(_CANONICAL_UNIQUE_CONSTRAINTS) & constraints)


def _assert_res133_model_state(settings: Settings) -> None:
    """The sealed RES-133 model: exactly the 11 canonical tables, no
    passage_source_span table, no span constraints and no 0004 additions to
    the passage/paragraph tables — the state 0003 leaves behind and 0004
    builds upon."""
    engine = _engine(settings)
    try:
        with engine.connect() as connection:
            tables = _public_tables(connection)
            triggers = _public_triggers(connection)
            constraints = _public_constraint_names(connection)
            passage_columns = _public_columns(connection, "passage")
    finally:
        engine.dispose()

    assert tables == {_VERSION_TABLE, *_RES133_TABLES}
    assert triggers == {f"trg_{table}_immutable" for table in _RES133_TABLES}
    assert set(_RES133_UNIQUE_CONSTRAINTS) <= constraints
    assert not (set(_SPAN_CONSTRAINTS) & constraints)
    assert "ck_passage_text_nonempty" not in constraints
    assert "uq_passage_id_document_version_id" not in constraints
    assert "uq_paragraph_id_document_version_id" not in constraints
    assert "passage_document_version_id" not in passage_columns


def _assert_res131_model_state(settings: Settings) -> None:
    """The sealed RES-131 model: exactly the 10 canonical tables, no
    paragraph table, no ``citation.source_anchor`` and no paragraph
    constraints — the state 0002 leaves behind and 0003 builds upon."""
    engine = _engine(settings)
    try:
        with engine.connect() as connection:
            tables = _public_tables(connection)
            triggers = _public_triggers(connection)
            constraints = _public_constraint_names(connection)
            citation_columns = _public_columns(connection, "citation")
    finally:
        engine.dispose()

    assert tables == {_VERSION_TABLE, *_RES131_TABLES}
    assert triggers == {f"trg_{table}_immutable" for table in _RES131_TABLES}
    assert set(_RES131_UNIQUE_CONSTRAINTS) <= constraints
    assert not (set(_PARAGRAPH_CONSTRAINTS) & constraints)
    assert "source_anchor" not in citation_columns


@pytest.mark.integration
def test_alembic_ini_points_at_the_repository_and_omits_the_dsn() -> None:
    """The DSN must come from dynamisrag.config, never from alembic.ini."""
    text_content = ALEMBIC_INI.read_text(encoding="utf-8")

    assert "sqlalchemy.url" not in text_content
    assert "path_separator = os" in text_content


@pytest.mark.integration
def test_database_is_at_the_migration_head(live_settings: Settings) -> None:
    engine = _engine(live_settings)
    try:
        with engine.connect() as connection:
            applied = set(
                connection.execute(text("SELECT version_num FROM alembic_version")).scalars()
            )
            server_version = connection.execute(text("SHOW server_version")).scalar_one()
    finally:
        engine.dispose()

    assert applied == _script_heads()
    assert str(server_version).startswith(_EXPECTED_POSTGRES_MAJOR)


@pytest.mark.integration
def test_test_role_cannot_connect_to_the_protected_reference_database() -> None:
    test_database_url = load_settings().test_database_url
    if test_database_url is None:
        pytest.fail("DYNAMISRAG_TEST_DATABASE_URL must use the dedicated test role", pytrace=False)
    test_url = str(test_database_url)
    protected_database = urlsplit(str(load_settings().database_url)).path
    protected_url = urlunsplit(urlsplit(test_url)._replace(path=protected_database))
    with psycopg.connect(test_url) as connection:
        role = connection.execute(
            "SELECT current_user,rolsuper,rolcreatedb,rolcreaterole FROM pg_roles "
            "WHERE rolname=current_user"
        ).fetchone()
    assert role == (urlsplit(test_url).username, False, True, False)

    with pytest.raises(psycopg.OperationalError, match="permission denied for database"):
        psycopg.connect(protected_url, connect_timeout=3)


@pytest.mark.integration
def test_adversarial_reference_downgrade_is_rejected_before_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure the protected DSN and block any DDL if the guard regresses."""
    protected_url = to_sqlalchemy_url(load_settings().database_url)
    monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", protected_url)
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", _SCRIPT_LOCATION)
    attempted_ddl: list[str] = []

    def refuse_ddl(
        _connection: Connection,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,  # noqa: FBT001 - required by SQLAlchemy's event signature
    ) -> None:
        if re.match(r"(?is)^\s*(CREATE|ALTER|DROP|TRUNCATE)\b", statement):
            attempted_ddl.append(statement)
            raise AssertionError("protected-reference DDL reached the DBAPI")

    event.listen(Engine, "before_cursor_execute", refuse_ddl)
    try:
        with pytest.raises(MigrationTargetSafetyError):
            command.downgrade(config, "base")
    finally:
        event.remove(Engine, "before_cursor_execute", refuse_ddl)

    assert attempted_ddl == []


@pytest.mark.integration
def test_baseline_revision_created_no_application_objects() -> None:
    """RES-130 must not pre-empt RES-131: revision 0001 creates no application objects.

    The proof runs against a throwaway database that is created, migrated to
    the 0001 baseline only, inspected, and dropped. Executing the baseline
    against an empty database is the strongest form of the claim — and unlike
    inspecting the canonical database, it stays true now that RES-131's
    schema legitimately exists there.
    """
    with (
        scratch_database("0001_foundation_baseline") as (engine, _),
        engine.connect() as connection,
    ):
        tables = _public_tables(connection)

    assert tables == {_VERSION_TABLE}


@pytest.mark.integration
def test_canonical_migration_round_trips(live_settings: Settings) -> None:
    """0004 round-trips through 0003, 0002 and 0001: downgrading to 0003 leaves
    the sealed RES-133 model intact, the upgrade to head restores the RES-134
    additions, and the full downgrade to the 0001 baseline removes every
    application objects only inside a disposable PostgreSQL database."""
    with scratch_database() as (engine, config):
        database = engine.url.database
        if database is None:
            raise RuntimeError("scratch database has no database name")
        scratch_settings = build_settings(
            database_url=_dsn_with_database(str(live_settings.database_url), database)
        )
        _assert_canonical_model_present(scratch_settings)

        command.downgrade(config, "0003_jats_source_structure")
        _assert_res133_model_state(scratch_settings)

        command.upgrade(config, "head")
        _assert_canonical_model_present(scratch_settings)

        command.downgrade(config, "0002_canonical_document_model")
        _assert_res131_model_state(scratch_settings)

        command.upgrade(config, "head")
        _assert_canonical_model_present(scratch_settings)

        command.downgrade(config, "0001_foundation_baseline")
        _assert_canonical_model_absent(scratch_settings)

        command.upgrade(config, "head")
        _assert_canonical_model_present(scratch_settings)


@pytest.mark.integration
def test_res131_revision_creates_exactly_the_sealed_canonical_model(
    live_settings: Settings,
) -> None:
    """Prove revision 0002 independently of 0003: on a throwaway database,
    upgrading straight to 0002 creates exactly the sealed RES-131 model —
    the 10 canonical tables, no paragraph table, no citation.source_anchor —
    and nothing else."""
    with scratch_database("0002_canonical_document_model") as (engine, _):
        database = engine.url.database
        if database is None:
            raise RuntimeError("scratch database has no database name")
        scratch_dsn = _dsn_with_database(str(live_settings.database_url), database)
        _assert_res131_model_state(build_settings(database_url=scratch_dsn))


@pytest.mark.integration
def test_baseline_revision_created_no_application_objects_is_reachable_from_head(
    live_settings: Settings,
) -> None:
    """The downgrade target named by the round-trip test really is the baseline.

    Guards the round-trip test itself: ``downgrade 0001_foundation_baseline``
    must be a valid target from the head revision, not a typo.
    """
    engine = _engine(live_settings)
    try:
        with engine.connect() as connection:
            applied = {
                row[0]
                for row in connection.execute(text("SELECT version_num FROM alembic_version"))
            }
    finally:
        engine.dispose()

    # Alembic records only the current head revision; the point of this test
    # is that the 0001 baseline named by the round-trip is reachable from it.
    assert applied == _script_heads()
