"""Fail-closed target authorization for online Alembic migrations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import LiteralString, cast

from sqlalchemy import Connection, String, literal, text
from sqlalchemy.dialects import postgresql

DISPOSABLE_DATABASE_MARKER_PREFIX = "dynamisrag-disposable:"
TEST_DATABASE_ROLE = "dynamisrag_test"
MAINTENANCE_ROLE = "dynamisrag_migration_operators"
_DISPOSABLE_MARKER = re.compile(r"dynamisrag-disposable:[0-9a-f]{32}")


class MigrationTargetSafetyError(RuntimeError):
    """Raised before Alembic configures or executes migrations on an unsafe target."""


@dataclass(frozen=True, slots=True)
class DatabaseTargetIdentity:
    database: str
    oid: int
    current_user: str
    session_user: str
    owner: str
    is_superuser: bool
    can_create_database: bool
    can_create_role: bool
    can_replicate: bool
    can_bypass_rls: bool
    comment: str | None
    direct_maintenance_member: bool


def migration_target_is_allowed(
    target: DatabaseTargetIdentity, maintenance_target: str | None
) -> bool:
    """Allow a marked test database or an exact target authorized by a DBA-granted role."""
    owner_session = target.current_user == target.session_user == target.owner
    least_privileged = (
        not target.is_superuser
        and not target.can_create_role
        and not target.can_replicate
        and not target.can_bypass_rls
    )
    disposable = (
        owner_session
        and target.database.startswith("dynamisrag_test_")
        and target.current_user == TEST_DATABASE_ROLE
        and least_privileged
        and target.can_create_database
        and not target.direct_maintenance_member
        and target.comment is not None
        and _DISPOSABLE_MARKER.fullmatch(target.comment) is not None
    )
    maintenance = (
        owner_session
        and target.current_user != TEST_DATABASE_ROLE
        and least_privileged
        and not target.can_create_database
        and target.direct_maintenance_member
        and maintenance_target == f"{target.database}:{target.oid}"
    )
    return disposable or maintenance


def require_migration_target(connection: Connection, maintenance_target: str | None = None) -> None:
    """Verify the live database, owner, role privileges, marker, and DBA role membership."""
    row = (
        connection.execute(
            text(
                """
                SELECT current_database() AS database,
                       d.oid,
                       current_user AS current_user,
                       session_user AS session_user,
                       pg_get_userbyid(d.datdba) AS owner,
                       r.rolsuper AS is_superuser,
                       r.rolcreatedb AS can_create_database,
                       r.rolcreaterole AS can_create_role,
                       r.rolreplication AS can_replicate,
                       r.rolbypassrls AS can_bypass_rls,
                       shobj_description(d.oid, 'pg_database') AS comment,
                       EXISTS (
                           SELECT 1
                           FROM pg_auth_members AS m
                           JOIN pg_roles AS ops ON ops.oid = m.roleid
                           WHERE m.member = r.oid
                             AND ops.rolname = :maintenance_role
                             AND NOT m.admin_option
                       ) AS direct_maintenance_member
                FROM pg_database AS d
                JOIN pg_roles AS r ON r.rolname = current_user
                WHERE d.datname = current_database()
                """
            ),
            {"maintenance_role": MAINTENANCE_ROLE},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise MigrationTargetSafetyError("Could not verify the connected PostgreSQL database")

    target = DatabaseTargetIdentity(**row)
    if not migration_target_is_allowed(target, maintenance_target):
        raise MigrationTargetSafetyError(
            f"Refusing online Alembic migration for database {target.database!r} "
            f"(OID {target.oid}); use a marked disposable test database or an exact target "
            f"authorized to the {MAINTENANCE_ROLE} role. A CLI target string alone "
            "is not authority."
        )


def offline_migration_guard_sql(maintenance_target: str | None) -> LiteralString:
    """Render the same authorization check ahead of any offline-generated DDL."""
    target_literal = str(
        literal(maintenance_target or "", type_=String()).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )

    return cast(
        LiteralString,
        f"""
DO $dynamisrag_migration_guard$
DECLARE
    target_name text;
    target_oid oid;
    connected_role name;
    session_role name;
    database_owner name;
    role_is_superuser boolean;
    role_can_createdb boolean;
    role_can_createrole boolean;
    role_can_replicate boolean;
    role_can_bypass_rls boolean;
    database_comment text;
    direct_maintenance_member boolean;
BEGIN
    SELECT current_database(), d.oid, current_user, session_user,
           pg_get_userbyid(d.datdba), r.rolsuper, r.rolcreatedb,
           r.rolcreaterole, r.rolreplication, r.rolbypassrls,
           shobj_description(d.oid, 'pg_database'),
           EXISTS (
               SELECT 1 FROM pg_auth_members AS m
               JOIN pg_roles AS ops ON ops.oid=m.roleid
               WHERE m.member=r.oid AND ops.rolname='{MAINTENANCE_ROLE}'
                 AND NOT m.admin_option
           )
    INTO target_name, target_oid, connected_role, session_role, database_owner,
         role_is_superuser, role_can_createdb, role_can_createrole,
         role_can_replicate, role_can_bypass_rls, database_comment,
         direct_maintenance_member
    FROM pg_database AS d
    JOIN pg_roles AS r ON r.rolname=current_user
    WHERE d.datname=current_database();

    IF NOT FOUND OR NOT COALESCE((
        connected_role=session_role AND connected_role=database_owner
        AND NOT role_is_superuser AND NOT role_can_createrole
        AND NOT role_can_replicate AND NOT role_can_bypass_rls
        AND (
            (connected_role='{TEST_DATABASE_ROLE}' AND role_can_createdb
             AND left(target_name, 16)='dynamisrag_test_'
             AND NOT direct_maintenance_member
             AND COALESCE(database_comment ~ '^dynamisrag-disposable:[0-9a-f]{{32}}$', false))
            OR
            (connected_role<>'{TEST_DATABASE_ROLE}' AND NOT role_can_createdb
             AND direct_maintenance_member
             AND {target_literal}=target_name || ':' || target_oid::text)
        )
    ), false) THEN
        RAISE EXCEPTION 'Refusing offline-generated migration for unauthorized database % (OID %)',
            current_database(), target_oid;
    END IF;
END
$dynamisrag_migration_guard$;
""".strip(),  # noqa: S608 - only SQLAlchemy-escaped literals are interpolated.
    )
