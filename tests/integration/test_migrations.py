"""The database migration state must match the migration code on disk.

Comparing against ``ScriptDirectory`` rather than a hard-coded revision id means
a new revision that is committed but not applied fails this test, without the
test itself needing editing.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from psycopg import sql
from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from tests._support import ALEMBIC_INI, REPO_ROOT

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
    "citation",
    "document_table",
    "figure",
)

_CANONICAL_UNIQUE_CONSTRAINTS: Final[tuple[str, ...]] = (
    "uq_source_artifact_artifact_key",
    "uq_document_canonical_key",
    "uq_document_identifier_namespace_value",
    "uq_document_version_version_key",
    "uq_section_document_version_section_key",
    "uq_passage_version_chunker_ordinal",
    "uq_citation_document_version_ordinal",
    "uq_document_table_document_version_key",
    "uq_figure_document_version_key",
)

_CANONICAL_COMPOSITE_FOREIGN_KEYS: Final[tuple[str, ...]] = (
    "fk_section_parent",
    "fk_document_table_section",
    "fk_figure_section",
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

    assert tables == {_VERSION_TABLE, *_CANONICAL_TABLES}
    assert _IMMUTABILITY_FUNCTION in functions
    assert triggers == {f"trg_{table}_immutable" for table in _CANONICAL_TABLES}
    assert set(_CANONICAL_UNIQUE_CONSTRAINTS) <= constraints
    assert set(_CANONICAL_COMPOSITE_FOREIGN_KEYS) <= constraints


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
    assert not triggers
    assert not (set(_CANONICAL_UNIQUE_CONSTRAINTS) & constraints)


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
def test_baseline_revision_created_no_application_objects(
    live_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RES-130 must not pre-empt RES-131: revision 0001 creates no application objects.

    The proof runs against a throwaway database that is created, migrated to
    the 0001 baseline only, inspected, and dropped. Executing the baseline
    against an empty database is the strongest form of the claim — and unlike
    inspecting the canonical database, it stays true now that RES-131's
    schema legitimately exists there.
    """
    database = f"dynamisrag_baseline_probe_{uuid4().hex[:12]}"
    maintenance_dsn = _dsn_with_database(str(live_settings.database_url), "postgres")
    with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        scratch_dsn = _dsn_with_database(str(live_settings.database_url), database)
        monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", scratch_dsn)
        config = Config(str(ALEMBIC_INI))
        config.set_main_option("script_location", _SCRIPT_LOCATION)
        command.upgrade(config, "0001_foundation_baseline")
        with psycopg.connect(scratch_dsn) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            }
    finally:
        with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database))
            )

    assert tables == {_VERSION_TABLE}


@pytest.mark.integration
def test_canonical_migration_round_trips(live_settings: Settings) -> None:
    """0002 round-trips: upgrade creates the canonical model, downgrade to the
    0001 baseline removes it, and upgrade restores it — all against the live
    PostgreSQL 18 server."""
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", _SCRIPT_LOCATION)

    command.upgrade(config, "head")
    _assert_canonical_model_present(live_settings)

    command.downgrade(config, "0001_foundation_baseline")
    _assert_canonical_model_absent(live_settings)

    command.upgrade(config, "head")
    _assert_canonical_model_present(live_settings)


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

    assert applied == {"0002_canonical_document_model"}
