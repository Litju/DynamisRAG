"""Fail-closed target checks for online Alembic migrations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import Connection, text

DISPOSABLE_DATABASE_MARKER_PREFIX = "dynamisrag-disposable:"
_DISPOSABLE_MARKER = re.compile(r"dynamisrag-disposable:[0-9a-f]{32}")


class MigrationTargetSafetyError(RuntimeError):
    """Raised before Alembic can execute DDL against an unapproved database."""


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
    comment: str | None


def migration_target_is_allowed(
    target: DatabaseTargetIdentity, maintenance_target: str | None
) -> bool:
    """Allow a marked test database or an exact, owner-authorized maintenance target."""
    disposable = (
        target.current_user == target.session_user == target.owner
        and not target.is_superuser
        and target.can_create_database
        and not target.can_create_role
        and target.comment is not None
        and _DISPOSABLE_MARKER.fullmatch(target.comment) is not None
    )
    maintenance = (
        maintenance_target == f"{target.database}:{target.oid}"
        and target.current_user == target.session_user == target.owner
    )
    return disposable or maintenance


def require_migration_target(connection: Connection, maintenance_target: str | None = None) -> None:
    """Check the live server identity before Alembic configures or runs migrations."""
    row = (
        connection.execute(
            text(
                """
            SELECT current_database() AS "database",
                   d.oid,
                   current_user AS "current_user",
                   session_user AS "session_user",
                   pg_get_userbyid(d.datdba) AS owner,
                   r.rolsuper AS is_superuser,
                   r.rolcreatedb AS can_create_database,
                   r.rolcreaterole AS can_create_role,
                   shobj_description(d.oid, 'pg_database') AS comment
            FROM pg_database AS d
            JOIN pg_roles AS r ON r.rolname = current_user
            WHERE d.datname = current_database()
            """
            )
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
            f"(OID {target.oid}); use a marked disposable database or explicitly "
            "authorize its exact name and OID for maintenance."
        )
