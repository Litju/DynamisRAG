from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from scripts.provision_test_role import (
    TestRoleProvisioningError as ProvisioningError,
)
from scripts.provision_test_role import (
    provision_test_role,
)

from dynamisrag.config import load_settings
from dynamisrag.db.migration_safety import MAINTENANCE_ROLE

pytestmark = pytest.mark.integration


def _configuration() -> tuple[str, str, str, str]:
    settings = load_settings()
    if settings.test_admin_database_url is None:
        pytest.fail("DYNAMISRAG_TEST_ADMIN_DATABASE_URL is required", pytrace=False)
    admin_url = str(settings.test_admin_database_url)
    app = urlsplit(str(settings.database_url))
    admin = urlsplit(admin_url)
    if admin.path.lstrip("/") != "postgres" or admin.path.lstrip("/") == app.path.lstrip("/"):
        pytest.fail(
            "Role provisioning tests require the isolated cluster's postgres database",
            pytrace=False,
        )
    if app.username is None:
        pytest.fail("DYNAMISRAG_DATABASE_URL must name the application owner role", pytrace=False)
    return admin_url, app.path.lstrip("/"), app.username, str(settings.test_database_url)


def _drop_role(connection: psycopg.Connection[Any], role: str, comment: str) -> None:
    state = connection.execute(
        """
        SELECT shobj_description(oid, 'pg_authid'),
               (SELECT count(*) FROM pg_auth_members AS m
                WHERE m.member=pg_roles.oid OR m.roleid=pg_roles.oid),
               (SELECT count(*) FROM pg_database WHERE datdba=pg_roles.oid)
        FROM pg_roles WHERE rolname=%s
        """,
        (role,),
    ).fetchone()
    if state is None:
        return
    if state != (comment, 0, 0):
        raise RuntimeError(f"Leaving provisioning test role {role!r}; its identity changed")
    connection.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


def _role_state(connection: psycopg.Connection[Any], role: str, app: str) -> tuple[object, ...]:
    row = connection.execute(
        """
        SELECT r.rolsuper, r.rolcreatedb, r.rolcreaterole, r.rolreplication,
               r.rolbypassrls, r.rolcanlogin, r.rolinherit,
               shobj_description(r.oid, 'pg_authid'),
               (SELECT count(*) FROM pg_auth_members AS m WHERE m.member=r.oid),
               has_database_privilege(r.rolname, %s, 'CONNECT')
        FROM pg_roles AS r WHERE r.rolname=%s
        """,
        (app, role),
    ).fetchone()
    if row is None:
        raise AssertionError(f"provisioned test role {role!r} is missing")
    return row


def test_role_provisioning_handles_fresh_legacy_and_conflicting_roles() -> None:
    admin_url, app_database, app_role, test_url = _configuration()
    test_password = urlsplit(test_url).password
    if test_password is None:
        pytest.fail("DYNAMISRAG_TEST_DATABASE_URL needs a password", pytrace=False)
    names = [f"dynamisrag_test_probe_{uuid4().hex[:16]}" for _ in range(3)]
    fresh_role, legacy_role, conflict_role = names
    foreign_comment = f"unmanaged-role:{uuid4().hex}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin_identity = admin.execute(
            "SELECT current_database(), r.rolsuper FROM pg_roles AS r WHERE r.rolname=current_user"
        ).fetchone()
        assert admin_identity == ("postgres", True)
        try:
            provision_test_role(
                admin,
                role_name=fresh_role,
                password=test_password,
                protected_database=app_database,
                application_role=app_role,
            )
            expected = (
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                "dynamisrag-managed-test-role-v1",
                0,
                False,
            )
            assert _role_state(admin, fresh_role, app_database) == expected

            # Reprovisioning repairs an incomplete managed role and removes
            # any accidental membership without changing its identity.
            admin.execute(
                sql.SQL("ALTER ROLE {} NOCREATEDB INHERIT").format(sql.Identifier(fresh_role))
            )
            admin.execute(
                sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(MAINTENANCE_ROLE), sql.Identifier(fresh_role)
                )
            )
            provision_test_role(
                admin,
                role_name=fresh_role,
                password=test_password,
                protected_database=app_database,
                application_role=app_role,
            )
            assert _role_state(admin, fresh_role, app_database) == expected

            # A safe role left by the earlier provisioning script is adopted.
            admin.execute(
                sql.SQL(
                    "CREATE ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                    "NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(legacy_role))
            )
            provision_test_role(
                admin,
                role_name=legacy_role,
                password=test_password,
                protected_database=app_database,
                application_role=app_role,
            )
            assert _role_state(admin, legacy_role, app_database) == expected

            # An unmanaged privileged role is refused and left unchanged.
            admin.execute(
                sql.SQL("CREATE ROLE {} WITH LOGIN SUPERUSER").format(sql.Identifier(conflict_role))
            )
            admin.execute(
                sql.SQL("COMMENT ON ROLE {} IS {}").format(
                    sql.Identifier(conflict_role), sql.Literal(foreign_comment)
                )
            )
            before = admin.execute(
                "SELECT rolsuper, rolcreatedb, rolcreaterole, "
                "shobj_description(oid, 'pg_authid') FROM pg_roles WHERE rolname=%s",
                (conflict_role,),
            ).fetchone()
            with pytest.raises(ProvisioningError, match="no changes made"):
                provision_test_role(
                    admin,
                    role_name=conflict_role,
                    password=test_password,
                    protected_database=app_database,
                    application_role=app_role,
                )
            after = admin.execute(
                "SELECT rolsuper, rolcreatedb, rolcreaterole, "
                "shobj_description(oid, 'pg_authid') FROM pg_roles WHERE rolname=%s",
                (conflict_role,),
            ).fetchone()
            assert after == before
        finally:
            _drop_role(admin, fresh_role, "dynamisrag-managed-test-role-v1")
            _drop_role(admin, legacy_role, "dynamisrag-managed-test-role-v1")
            _drop_role(admin, conflict_role, foreign_comment)
