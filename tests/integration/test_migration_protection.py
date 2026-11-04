from __future__ import annotations

import argparse
import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import Engine, event
from sqlalchemy.engine import Connection

from dynamisrag.config import load_settings
from dynamisrag.db.migration_safety import MAINTENANCE_ROLE, MigrationTargetSafetyError
from tests._support import ALEMBIC_INI, REPO_ROOT

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class ProtectedDecoy:
    database: str
    oid: int
    owner: str
    owner_url: str
    owner_comment: str
    database_comment: str
    maintenance_member: bool


def _urls() -> tuple[str, str, str]:
    settings = load_settings()
    if settings.test_database_url is None or settings.test_admin_database_url is None:
        pytest.fail(
            "Set DYNAMISRAG_TEST_DATABASE_URL and DYNAMISRAG_TEST_ADMIN_DATABASE_URL "
            "for the isolated PostgreSQL test cluster",
            pytrace=False,
        )
    test_url = str(settings.test_database_url)
    admin_url = str(settings.test_admin_database_url)
    app_database = urlsplit(str(settings.database_url)).path.lstrip("/")
    if (
        urlsplit(test_url).path.lstrip("/") == app_database
        or urlsplit(admin_url).path.lstrip("/") == app_database
        or urlsplit(test_url).path.lstrip("/") != "postgres"
        or urlsplit(admin_url).path.lstrip("/") != "postgres"
        or (urlsplit(test_url).hostname, urlsplit(test_url).port)
        != (urlsplit(admin_url).hostname, urlsplit(admin_url).port)
    ):
        pytest.fail("The test and admin DSNs must not name the application database", pytrace=False)
    return test_url, admin_url, app_database


@contextmanager
def _protected_decoy(*, maintenance_member: bool = False) -> Generator[ProtectedDecoy, None, None]:
    _, admin_url, _ = _urls()
    suffix = uuid4().hex
    database = f"dynamisrag_reference_decoy_{suffix}"
    owner = f"dynamisrag_decoy_{suffix[:20]}"
    password = uuid4().hex * 2
    owner_comment = f"res583-decoy-owner:{suffix}"
    database_comment = f"res583-protected-decoy:{suffix}"
    admin_parts = urlsplit(admin_url)
    host = admin_parts.hostname or "localhost"
    if ":" in host:
        host = f"[{host}]"
    port = f":{admin_parts.port}" if admin_parts.port is not None else ""
    owner_parts = admin_parts._replace(
        netloc=f"{owner}:{password}@{host}{port}", path=f"/{database}"
    )
    owner_url = urlunsplit(owner_parts)
    oid: int | None = None
    created_role = False
    created_database = False
    try:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            identity = admin.execute(
                "SELECT current_database(), current_user, session_user, r.rolsuper "
                "FROM pg_roles AS r WHERE r.rolname=current_user"
            ).fetchone()
            if identity is None or identity[0] != "postgres" or identity[1] != identity[2]:
                raise RuntimeError("The isolated test administrator identity could not be read")
            if not identity[3]:
                raise RuntimeError(
                    "Protected-decoy tests require the disposable cluster administrator"
                )
            admin.execute(
                sql.SQL(
                    "CREATE ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                    "NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}"
                ).format(sql.Identifier(owner), sql.Literal(password))
            )
            created_role = True
            admin.execute(
                sql.SQL("COMMENT ON ROLE {} IS {}").format(
                    sql.Identifier(owner), sql.Literal(owner_comment)
                )
            )
            admin.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(database), sql.Identifier(owner)
                )
            )
            created_database = True
            row = admin.execute(
                "SELECT oid, pg_get_userbyid(datdba) FROM pg_database WHERE datname=%s",
                (database,),
            ).fetchone()
            if row is None or row[1] != owner:
                raise RuntimeError("Protected decoy database owner verification failed")
            if not isinstance(row[0], int):
                raise RuntimeError("Protected decoy database OID is invalid")
            oid = row[0]
            admin.execute(
                sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                    sql.Identifier(database), sql.Literal(database_comment)
                )
            )
            admin.execute(
                sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(database)
                )
            )
            admin.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(database), sql.Identifier(owner)
                )
            )
            if maintenance_member:
                admin.execute(
                    sql.SQL("GRANT {} TO {}").format(
                        sql.Identifier(MAINTENANCE_ROLE), sql.Identifier(owner)
                    )
                )
        with psycopg.connect(owner_url, autocommit=True) as decoy:
            decoy.execute("CREATE TABLE migration_guard_sentinel (value integer PRIMARY KEY)")
            decoy.execute("INSERT INTO migration_guard_sentinel VALUES (583)")
        yield ProtectedDecoy(
            database, oid, owner, owner_url, owner_comment, database_comment, maintenance_member
        )
    finally:
        _cleanup_protected_decoy(
            admin_url=admin_url,
            database=database if created_database else None,
            oid=oid,
            owner=owner if created_role else None,
            owner_comment=owner_comment,
            database_comment=database_comment,
            maintenance_member=maintenance_member,
        )


def _cleanup_protected_decoy(
    *,
    admin_url: str,
    database: str | None,
    oid: int | None,
    owner: str | None,
    owner_comment: str,
    database_comment: str,
    maintenance_member: bool,
) -> None:
    with psycopg.connect(admin_url, autocommit=True) as admin:
        if database is not None:
            current = admin.execute(
                """
                SELECT d.oid, pg_get_userbyid(d.datdba),
                       shobj_description(d.oid, 'pg_database'),
                       (SELECT count(*) FROM pg_stat_activity AS a
                        WHERE a.datname=d.datname AND a.pid<>pg_backend_pid())
                FROM pg_database AS d WHERE d.datname=%s
                """,
                (database,),
            ).fetchone()
            if (
                current is None
                or oid is None
                or current[0] != oid
                or current[1] != owner
                or current[2] != database_comment
                or current[3] != 0
            ):
                raise RuntimeError(
                    f"Leaving protected decoy {database!r}; exact identity or idle state changed"
                )
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        if owner is None:
            return
        membership = admin.execute(
            """
            SELECT parent.rolname, m.admin_option
            FROM pg_auth_members AS m
            JOIN pg_roles AS parent ON parent.oid=m.roleid
            JOIN pg_roles AS member ON member.oid=m.member
            WHERE member.rolname=%s
            """,
            (owner,),
        ).fetchall()
        if maintenance_member:
            if membership != [(MAINTENANCE_ROLE, False)]:
                raise RuntimeError(f"Leaving protected-decoy role {owner!r}; membership changed")
            admin.execute(
                sql.SQL("REVOKE {} FROM {}").format(
                    sql.Identifier(MAINTENANCE_ROLE), sql.Identifier(owner)
                )
            )
        current = admin.execute(
            """
            SELECT oid, rolsuper, shobj_description(oid, 'pg_authid'),
                   (SELECT count(*) FROM pg_auth_members AS m
                    WHERE m.member=pg_roles.oid OR m.roleid=pg_roles.oid),
                   (SELECT count(*) FROM pg_database WHERE datdba=pg_roles.oid)
            FROM pg_roles WHERE rolname=%s
            """,
            (owner,),
        ).fetchone()
        if (
            current is None
            or current[1]
            or current[2] != owner_comment
            or current[3] != 0
            or current[4] != 0
        ):
            raise RuntimeError(f"Leaving protected-decoy role {owner!r}; exact identity changed")
        admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(owner)))


def _database_state(url: str) -> tuple[tuple[str, ...], tuple[int, ...], str | None]:
    with psycopg.connect(url) as connection:
        tables = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
            ).fetchall()
        )
        sentinel = tuple(
            int(row[0])
            for row in connection.execute(
                "SELECT value FROM migration_guard_sentinel ORDER BY value"
            ).fetchall()
        )
        marker = connection.execute(
            "SELECT shobj_description(oid, 'pg_database') FROM pg_database "
            "WHERE datname=current_database()"
        ).fetchone()
    return tables, sentinel, None if marker is None else str(marker[0])


def test_test_role_cannot_connect_to_a_protected_decoy() -> None:
    test_url, _, _ = _urls()
    with _protected_decoy() as decoy:
        test_parts = urlsplit(test_url)
        decoy_url = urlunsplit(test_parts._replace(path=f"/{decoy.database}"))
        with pytest.raises(psycopg.OperationalError, match="permission denied for database"):
            psycopg.connect(decoy_url, connect_timeout=3)


def test_exact_maintenance_target_string_cannot_authorize_a_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _protected_decoy() as decoy:
        before = _database_state(decoy.owner_url)
        monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", decoy.owner_url)
        config = Config(str(ALEMBIC_INI))
        config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        config.cmd_opts = argparse.Namespace(x=[f"maintenance_target={decoy.database}:{decoy.oid}"])
        attempted_ddl: list[str] = []

        def refuse_ddl(
            _connection: Connection,
            _cursor: object,
            statement: str,
            _parameters: object,
            _context: object,
            _executemany: bool,  # noqa: FBT001 - required by SQLAlchemy's event signature
        ) -> None:
            if re.match(r"(?is)^\s*(CREATE|ALTER|DROP|TRUNCATE|REINDEX|GRANT|REVOKE)\b", statement):
                attempted_ddl.append(statement)
                raise AssertionError("protected-decoy DDL reached the DBAPI")

        event.listen(Engine, "before_cursor_execute", refuse_ddl)
        try:
            with pytest.raises(MigrationTargetSafetyError):
                command.downgrade(config, "base")
        finally:
            event.remove(Engine, "before_cursor_execute", refuse_ddl)

        assert attempted_ddl == []
        assert _database_state(decoy.owner_url) == before


def test_dba_granted_maintenance_role_authorizes_only_its_exact_owned_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _protected_decoy(maintenance_member=True) as decoy:
        monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", decoy.owner_url)
        config = Config(str(ALEMBIC_INI))
        config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        config.cmd_opts = argparse.Namespace(x=[f"maintenance_target={decoy.database}:{decoy.oid}"])
        command.upgrade(config, "0001_foundation_baseline")
        tables, sentinel, marker = _database_state(decoy.owner_url)
        assert set(tables) == {"alembic_version", "migration_guard_sentinel"}
        assert sentinel == (583,)
        assert marker == decoy.database_comment


def test_offline_sql_places_the_role_guard_before_migration_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import StringIO

    from dynamisrag.db.migration_safety import offline_migration_guard_sql

    with _protected_decoy() as decoy:
        monkeypatch.setenv("DYNAMISRAG_DATABASE_URL", decoy.owner_url)
        output = StringIO()
        config = Config(str(ALEMBIC_INI), output_buffer=output)
        config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        config.cmd_opts = argparse.Namespace(x=[f"maintenance_target={decoy.database}:{decoy.oid}"])
        command.upgrade(config, "head", sql=True)
        generated = output.getvalue()
        guard_at = generated.index("DO $dynamisrag_migration_guard$")
        first_ddl = re.search(r"(?im)^CREATE TABLE\b", generated)
        assert first_ddl is not None and guard_at < first_ddl.start()

        before = _database_state(decoy.owner_url)
        with (
            psycopg.connect(decoy.owner_url) as connection,
            pytest.raises(
                psycopg.errors.RaiseException, match="Refusing offline-generated migration"
            ),
        ):
            connection.execute(
                sql.SQL(offline_migration_guard_sql(f"{decoy.database}:{decoy.oid}"))
            )
        assert _database_state(decoy.owner_url) == before
