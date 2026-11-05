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
from dynamisrag.db.test_cluster import (
    TEST_POSTGRES_HOST_PORT,
    ClusterIsolationError,
    require_test_cluster_identity,
    require_test_cluster_system_identifier,
    validate_test_database_dsns,
    verify_test_cluster_system_identifier,
    verify_test_compose_resources,
)

pytestmark = pytest.mark.integration


def _configuration() -> tuple[str, str, str, str, str]:
    settings = load_settings()
    if settings.test_admin_database_url is None or settings.test_database_url is None:
        pytest.fail(
            "DYNAMISRAG_TEST_ADMIN_DATABASE_URL and DYNAMISRAG_TEST_DATABASE_URL are required",
            pytrace=False,
        )
    admin_url = str(settings.test_admin_database_url)
    test_url = str(settings.test_database_url)
    try:
        app_database, app_role = validate_test_database_dsns(
            str(settings.database_url),
            test_url,
            admin_url,
            test_host_port=TEST_POSTGRES_HOST_PORT,
        )
        system_identifier = require_test_cluster_system_identifier(
            settings.test_cluster_system_identifier
        )
        verify_test_compose_resources()
        with psycopg.connect(admin_url) as admin:
            verify_test_cluster_system_identifier(admin, system_identifier)
            require_test_cluster_identity(admin, system_identifier, expected_database="postgres")
    except RuntimeError as error:
        pytest.fail(str(error), pytrace=False)
    return admin_url, app_database, app_role, test_url, system_identifier


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
    admin_url, app_database, app_role, test_url, system_identifier = _configuration()
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
                expected_test_cluster_system_identifier=system_identifier,
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
                expected_test_cluster_system_identifier=system_identifier,
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
                expected_test_cluster_system_identifier=system_identifier,
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
                    expected_test_cluster_system_identifier=system_identifier,
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


def test_test_role_cannot_forge_the_cluster_identity_marker() -> None:
    _, _, _, test_url, system_identifier = _configuration()
    with psycopg.connect(test_url, autocommit=True) as test_role:
        require_test_cluster_identity(
            test_role,
            system_identifier,
            expected_database="postgres",
            expected_user="dynamisrag_test",
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            test_role.execute(
                sql.SQL("COMMENT ON ROLE {} IS {}").format(
                    sql.Identifier("dynamisrag_test_cluster_identity"),
                    sql.Literal("dynamisrag-test-cluster-system-identifier:1"),
                )
            )
        require_test_cluster_identity(
            test_role,
            system_identifier,
            expected_database="postgres",
            expected_user="dynamisrag_test",
        )


def test_provisioning_refuses_a_mismatched_cluster_identity_before_changes() -> None:
    admin_url, app_database, app_role, _, system_identifier = _configuration()
    mismatched = "1" if system_identifier != "1" else "2"
    probe_role = f"dynamisrag_test_mismatch_{uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        before = admin.execute(
            "SELECT shobj_description(oid, 'pg_authid') FROM pg_roles "
            "WHERE rolname='dynamisrag_test_cluster_identity'"
        ).fetchone()
        with pytest.raises(ClusterIsolationError, match="different system identifier"):
            provision_test_role(
                admin,
                role_name=probe_role,
                password="0" * 64,
                protected_database=app_database,
                application_role=app_role,
                expected_test_cluster_system_identifier=mismatched,
            )
        assert admin.execute(
            "SELECT count(*) FROM pg_roles WHERE rolname=%s", (probe_role,)
        ).fetchone() == (0,)
        after = admin.execute(
            "SELECT shobj_description(oid, 'pg_authid') FROM pg_roles "
            "WHERE rolname='dynamisrag_test_cluster_identity'"
        ).fetchone()
        assert after == before
