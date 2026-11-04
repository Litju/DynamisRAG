"""Idempotently provision the dedicated test role on an isolated PostgreSQL cluster."""

from __future__ import annotations

import re
import sys
from typing import Any
from urllib.parse import urlsplit

import psycopg
from psycopg import Connection, sql
from pydantic import ValidationError

from dynamisrag.config import load_settings
from dynamisrag.db.migration_safety import MAINTENANCE_ROLE, TEST_DATABASE_ROLE

_ROLE_MARKER = "dynamisrag-managed-test-role-v1"
_MAINTENANCE_MARKER = "dynamisrag-managed-maintenance-role-v1"
_SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class TestRoleProvisioningError(RuntimeError):
    """Raised when an existing role or target cluster is unsafe to change."""


def _require_identifier(value: str, *, label: str) -> None:
    if _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise TestRoleProvisioningError(f"{label} must be a lowercase PostgreSQL identifier")


def _database_owner(connection: Connection[Any], database: str) -> tuple[int, str, bool] | None:
    return connection.execute(
        """
        SELECT oid, pg_get_userbyid(datdba), datallowconn
        FROM pg_database WHERE datname=%s AND NOT datistemplate
        """,
        (database,),
    ).fetchone()


def _ensure_maintenance_role(connection: Connection[Any]) -> None:
    row = connection.execute(
        """
        SELECT oid, rolcanlogin, rolsuper, rolcreatedb, rolcreaterole,
               rolreplication, rolbypassrls,
               shobj_description(oid, 'pg_authid')
        FROM pg_roles WHERE rolname=%s
        """,
        (MAINTENANCE_ROLE,),
    ).fetchone()
    if row is None:
        connection.execute(
            sql.SQL(
                "CREATE ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS NOINHERIT"
            ).format(sql.Identifier(MAINTENANCE_ROLE))
        )
        connection.execute(
            sql.SQL("COMMENT ON ROLE {} IS {}").format(
                sql.Identifier(MAINTENANCE_ROLE), sql.Literal(_MAINTENANCE_MARKER)
            )
        )
        return

    (
        oid,
        can_login,
        is_superuser,
        can_createdb,
        can_createrole,
        can_replicate,
        bypass_rls,
        marker,
    ) = row
    memberships = connection.execute(
        "SELECT count(*) FROM pg_auth_members WHERE roleid=%s OR member=%s",
        (oid, oid),
    ).fetchone()[0]
    if marker not in (None, _MAINTENANCE_MARKER) or any(
        (can_login, is_superuser, can_createdb, can_createrole, can_replicate, bypass_rls)
    ):
        raise TestRoleProvisioningError(
            f"Existing role {MAINTENANCE_ROLE!r} has conflicting configuration; no changes made"
        )
    if marker is None and memberships:
        raise TestRoleProvisioningError(
            f"Unmarked role {MAINTENANCE_ROLE!r} has memberships; no changes made"
        )
    if marker is None:
        connection.execute(
            sql.SQL("COMMENT ON ROLE {} IS {}").format(
                sql.Identifier(MAINTENANCE_ROLE), sql.Literal(_MAINTENANCE_MARKER)
            )
        )


def _ensure_test_role(
    connection: Connection[Any],
    *,
    role_name: str,
    password: str,
    protected_database: str,
    application_role: str,
) -> None:
    _require_identifier(role_name, label="test role")
    _require_identifier(protected_database, label="protected database")
    _require_identifier(application_role, label="application role")
    if role_name in {MAINTENANCE_ROLE, application_role}:
        raise TestRoleProvisioningError(
            "The test role must be distinct from the application and maintenance roles"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", password):
        raise TestRoleProvisioningError("The test-role password must be a 64-character hex value")

    protected = _database_owner(connection, protected_database)
    if protected is None or not protected[2] or protected[1] != application_role:
        raise TestRoleProvisioningError(
            "The application database is missing, not owned by the application role, "
            "is a template, or disallows connections; "
            "no role or ACL changes made"
        )

    row = connection.execute(
        """
        SELECT oid, rolsuper, rolcreatedb, rolcreaterole, rolreplication,
               rolbypassrls, rolcanlogin, shobj_description(oid, 'pg_authid')
        FROM pg_roles WHERE rolname=%s
        """,
        (role_name,),
    ).fetchone()
    created = row is None
    role_oid: int | None = None
    if row is not None:
        role_oid, is_superuser, _, can_createrole, can_replicate, bypass_rls, _, marker = row
        owns_protected = protected[1] == role_name
        memberships = connection.execute(
            "SELECT count(*) FROM pg_auth_members WHERE member=%s",
            (role_oid,),
        ).fetchone()[0]
        is_managed = marker == _ROLE_MARKER
        safe_legacy = (
            marker is None
            and not is_superuser
            and not can_createrole
            and not can_replicate
            and not bypass_rls
            and not memberships
        )
        if owns_protected or (not is_managed and not safe_legacy):
            raise TestRoleProvisioningError(
                f"Existing role {role_name!r} is not a safe dedicated test role; no changes made"
            )
        if is_managed:
            member_roles = connection.execute(
                """
                SELECT parent.rolname FROM pg_auth_members AS m
                JOIN pg_roles AS parent ON parent.oid=m.roleid
                WHERE m.member=%s
                """,
                (role_oid,),
            ).fetchall()
            for (parent_role,) in member_roles:
                connection.execute(
                    sql.SQL("REVOKE {} FROM {}").format(
                        sql.Identifier(parent_role), sql.Identifier(role_name)
                    )
                )

    password_sql = sql.Literal(password)
    role_identifier = sql.Identifier(role_name)
    if created:
        connection.execute(
            sql.SQL(
                "CREATE ROLE {} WITH LOGIN CREATEDB NOSUPERUSER NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}"
            ).format(role_identifier, password_sql)
        )
    else:
        connection.execute(
            sql.SQL(
                "ALTER ROLE {} WITH LOGIN CREATEDB NOSUPERUSER NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}"
            ).format(role_identifier, password_sql)
        )
    connection.execute(
        sql.SQL("COMMENT ON ROLE {} IS {}").format(role_identifier, sql.Literal(_ROLE_MARKER))
    )

    # Public CONNECT would override a role-specific REVOKE, so remove it only on
    # this explicitly configured test cluster's application database.
    protected_identifier = sql.Identifier(protected_database)
    connection.execute(
        sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(protected_identifier)
    )
    connection.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            protected_identifier, sql.Identifier(protected[1])
        )
    )
    connection.execute(
        sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(
            protected_identifier, role_identifier
        )
    )

    if connection.execute(
        "SELECT has_database_privilege(%s, %s, 'CONNECT')",
        (role_name, protected_database),
    ).fetchone()[0]:
        raise TestRoleProvisioningError(
            "The test role can still connect to the application database"
        )


def provision_test_role(
    connection: Connection[Any],
    *,
    role_name: str,
    password: str,
    protected_database: str,
    application_role: str,
) -> None:
    """Provision an isolated test role; never connect to the protected database itself."""
    _require_identifier(role_name, label="test role")
    _require_identifier(protected_database, label="protected database")
    admin = connection.execute(
        """
        SELECT current_database(), current_user, session_user,
               r.rolsuper, r.rolcreaterole
        FROM pg_roles AS r WHERE r.rolname=current_user
        """
    ).fetchone()
    if admin is None or admin[0] == protected_database or admin[1] != admin[2]:
        raise TestRoleProvisioningError(
            "Connect to a separate administrative database, not the application database"
        )
    if not admin[3] and not admin[4]:
        raise TestRoleProvisioningError(
            "The provisioning connection requires a PostgreSQL administrator role"
        )
    app_owner = _database_owner(connection, protected_database)
    if app_owner is None or app_owner[1] != application_role:
        raise TestRoleProvisioningError(
            "The application database is not owned by the configured application role"
        )
    if not admin[3] and admin[1] != application_role:
        raise TestRoleProvisioningError(
            "A non-superuser provisioner must own the application database"
        )
    with connection.transaction():
        _ensure_maintenance_role(connection)
        _ensure_test_role(
            connection,
            role_name=role_name,
            password=password,
            protected_database=protected_database,
            application_role=application_role,
        )


def main() -> int:
    settings = load_settings()
    if settings.test_database_url is None or settings.test_admin_database_url is None:
        raise TestRoleProvisioningError(
            "Set DYNAMISRAG_TEST_DATABASE_URL and DYNAMISRAG_TEST_ADMIN_DATABASE_URL for "
            "a disposable PostgreSQL cluster"
        )
    test_dsn = str(settings.test_database_url)
    admin_dsn = str(settings.test_admin_database_url)
    application = urlsplit(str(settings.database_url))
    test = urlsplit(test_dsn)
    admin = urlsplit(admin_dsn)
    role_name = test.username
    password = test.password
    application_database = application.path.lstrip("/")
    if (
        role_name != TEST_DATABASE_ROLE
        or application.username is None
        or password is None
        or test.path.lstrip("/") == application_database
        or admin.path.lstrip("/") == application_database
        or test.path.lstrip("/") != "postgres"
        or admin.path.lstrip("/") != "postgres"
        or admin.path.lstrip("/") in {"template0", "template1"}
        or (admin.hostname, admin.port) != (test.hostname, test.port)
        or admin.username == TEST_DATABASE_ROLE
    ):
        raise TestRoleProvisioningError(
            "Test and admin DSNs must target the same isolated cluster, different from "
            "the application database, using the dedicated dynamisrag_test login"
        )
    with psycopg.connect(admin_dsn) as connection:
        provision_test_role(
            connection,
            role_name=role_name,
            password=password,
            protected_database=application_database,
            application_role=application.username or "",
        )
    with psycopg.connect(test_dsn) as connection:
        identity = connection.execute(
            """
            SELECT current_database(), session_user, current_user,
                   r.rolsuper, r.rolcreatedb, r.rolcreaterole,
                   has_database_privilege(current_user, %s, 'CONNECT')
            FROM pg_roles AS r WHERE r.rolname=current_user
            """,
            (application_database,),
        ).fetchone()
        if identity != (
            "postgres",
            TEST_DATABASE_ROLE,
            TEST_DATABASE_ROLE,
            False,
            True,
            False,
            False,
        ):
            raise TestRoleProvisioningError(
                "Provisioned test credentials failed the least-privilege connection check"
            )
    sys.stdout.write(
        "Provisioned the isolated test role and rejected it from the application DB.\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TestRoleProvisioningError as error:
        sys.stderr.write(f"Test role provisioning refused: {error}\n")
        raise SystemExit(2) from None
    except ValidationError as error:
        sys.stderr.write(
            f"Test role provisioning configuration is invalid ({error.error_count()} setting(s)); "
            "values were withheld\n"
        )
        raise SystemExit(2) from None
    except psycopg.Error as error:
        sys.stderr.write(
            f"Test role provisioning failed with PostgreSQL SQLSTATE "
            f"{error.sqlstate or 'unknown'}; "
            "connection details were withheld\n"
        )
        raise SystemExit(2) from None
