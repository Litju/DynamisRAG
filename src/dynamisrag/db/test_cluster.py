"""Fail-closed identity checks for the disposable local integration cluster."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from psycopg import Connection

TEST_DATABASE_ROLE = "dynamisrag_test"
TEST_CLUSTER_IDENTITY_ROLE = "dynamisrag_test_cluster_identity"
TEST_CLUSTER_IDENTITY_PREFIX = "dynamisrag-test-cluster-system-identifier:"
TEST_COMPOSE_PROJECT_NAME = "dynamisrag-integration"
TEST_POSTGRES_VOLUME_NAME = "dynamisrag-integration_postgres-data"
TEST_POSTGRES_HOST_PORT = 55433
_SYSTEM_IDENTIFIER = re.compile(r"^[1-9][0-9]{0,19}$")
_REPO_ROOT = Path(__file__).resolve().parents[3]


class ClusterIsolationError(RuntimeError):
    """Raised before a test or provisioner can use an unattested cluster."""


def require_test_cluster_system_identifier(value: str | None) -> str:
    """Require one unsigned PostgreSQL system identifier, never a guessed value."""
    if value is None or _SYSTEM_IDENTIFIER.fullmatch(value) is None or int(value) > 2**64 - 1:
        raise ClusterIsolationError(
            "Set DYNAMISRAG_TEST_CLUSTER_SYSTEM_IDENTIFIER to the exact system identifier "
            "read from the isolated integration cluster"
        )
    return value


def test_cluster_identity_comment(system_identifier: str) -> str:
    return TEST_CLUSTER_IDENTITY_PREFIX + require_test_cluster_system_identifier(system_identifier)


def test_cluster_identity_matches(
    comment: str | None, expected_system_identifier: str | None
) -> bool:
    if (
        expected_system_identifier is None
        or _SYSTEM_IDENTIFIER.fullmatch(expected_system_identifier) is None
        or int(expected_system_identifier) > 2**64 - 1
    ):
        return False
    return comment == TEST_CLUSTER_IDENTITY_PREFIX + expected_system_identifier


def validate_test_database_dsns(
    application_url: str,
    test_url: str,
    admin_url: str,
    *,
    test_host_port: int,
) -> tuple[str, str]:
    """Reject ambiguous or non-loopback DSNs before opening any connection."""
    try:
        application, test, admin = map(urlsplit, (application_url, test_url, admin_url))
        endpoints = [(parts.hostname, parts.port) for parts in (application, test, admin)]
    except ValueError as error:
        raise ClusterIsolationError("The integration PostgreSQL DSNs are ambiguous") from error

    if (
        test_host_port == 55432
        or any(
            parts.scheme not in {"postgresql", "postgresql+psycopg"}
            or parts.hostname != "127.0.0.1"
            or parts.port != test_host_port
            or parts.query
            or parts.fragment
            for parts in (application, test, admin)
        )
        or any(endpoint != ("127.0.0.1", test_host_port) for endpoint in endpoints)
        or application.path.lstrip("/") in {"", "postgres"}
        or test.path.lstrip("/") != "postgres"
        or admin.path.lstrip("/") != "postgres"
        or test.username != TEST_DATABASE_ROLE
        or admin.username is None
        or admin.username == TEST_DATABASE_ROLE
        or application.username is None
    ):
        raise ClusterIsolationError(
            "Application, test, and admin DSNs must use the isolated 127.0.0.1 test port "
            f"({test_host_port}), with the test/admin connections to postgres"
        )
    return application.path.lstrip("/"), application.username


def verify_test_cluster_system_identifier(
    connection: Connection[Any], expected_system_identifier: str
) -> None:
    """Check the immutable cluster identifier using the isolated DBA connection."""
    expected = require_test_cluster_system_identifier(expected_system_identifier)
    admin = connection.execute(
        """
        SELECT current_database(), current_user, session_user, r.rolsuper
        FROM pg_roles AS r WHERE r.rolname=current_user
        """
    ).fetchone()
    if admin is None or admin[0] != "postgres" or admin[1] != admin[2] or not admin[3]:
        raise ClusterIsolationError(
            "Cluster identity provisioning requires a superuser session on the postgres database"
        )
    actual = connection.execute(
        "SELECT system_identifier::text FROM pg_control_system()"
    ).fetchone()
    if actual is None or actual[0] != expected:
        raise ClusterIsolationError(
            "The administrative DSN reached a PostgreSQL cluster with a different system "
            "identifier; no provisioning changes were made"
        )


def require_test_cluster_identity(
    connection: Connection[Any],
    expected_system_identifier: str | None,
    *,
    expected_database: str | None = None,
    expected_user: str | None = None,
) -> None:
    """Verify the DBA-owned cluster marker through a role the test login cannot alter."""
    expected = require_test_cluster_system_identifier(expected_system_identifier)
    row = connection.execute(
        """
        SELECT current_database(), session_user, current_user,
               identity.rolcanlogin, identity.rolsuper, identity.rolcreatedb,
               identity.rolcreaterole, identity.rolreplication, identity.rolbypassrls,
               shobj_description(identity.oid, 'pg_authid'),
               (SELECT count(*) FROM pg_auth_members AS m
                WHERE m.member=identity.oid OR m.roleid=identity.oid)
        FROM pg_roles AS identity
        WHERE identity.rolname=%s
        """,
        (TEST_CLUSTER_IDENTITY_ROLE,),
    ).fetchone()
    if (
        row is None
        or (expected_database is not None and row[0] != expected_database)
        or (expected_user is not None and (row[1] != expected_user or row[2] != expected_user))
        or row[3:9] != (False, False, False, False, False, False)
        or row[9] != test_cluster_identity_comment(expected)
        or row[10] != 0
    ):
        raise ClusterIsolationError(
            "The connected PostgreSQL cluster is unmarked, mismatched, or has an altered "
            "cluster-identity role"
        )


def require_isolated_compose_identity(
    project_name: str, service_name: str, postgres_volume_name: str, host_port: str
) -> None:
    """Require the dedicated Compose project, volume, and loopback port."""
    if (
        project_name != TEST_COMPOSE_PROJECT_NAME
        or service_name != "postgres"
        or postgres_volume_name != TEST_POSTGRES_VOLUME_NAME
        or host_port != str(TEST_POSTGRES_HOST_PORT)
    ):
        raise ClusterIsolationError(
            "Integration tests require the dedicated dynamisrag-integration Compose project, "
            "its namespaced postgres volume, and host port 55433"
        )


def verify_test_compose_resources() -> None:
    """Confirm the live PostgreSQL container is the dedicated project and volume."""
    compose = [
        "docker",
        "compose",
        "--project-name",
        TEST_COMPOSE_PROJECT_NAME,
        "--file",
        str(_REPO_ROOT / "compose.yaml"),
        "--file",
        str(_REPO_ROOT / "compose.integration.yaml"),
    ]
    try:
        container_id = subprocess.run(  # noqa: S603 - fixed Compose command and project name
            [*compose, "ps", "--quiet", "postgres"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        if not container_id or "\n" in container_id:
            raise ClusterIsolationError(
                "The isolated Compose project must have exactly one running PostgreSQL container"
            )
        inspected = json.loads(
            subprocess.run(  # noqa: S603 - fixed Docker command and container id argument
                ["docker", "inspect", container_id],  # noqa: S607 - executable is the fixed Docker CLI
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        )
        if not isinstance(inspected, list):
            raise ClusterIsolationError("Docker returned an ambiguous container identity")
        inspected_list = cast(list[object], inspected)
        if len(inspected_list) != 1 or not isinstance(inspected_list[0], dict):
            raise ClusterIsolationError("Docker returned an ambiguous container identity")
        container = cast(dict[str, Any], inspected_list[0])
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        IndexError,
        KeyError,
    ) as error:
        raise ClusterIsolationError(
            "Could not verify the isolated Docker Compose PostgreSQL container and volume"
        ) from error

    labels = cast(dict[str, str], container.get("Config", {}).get("Labels") or {})
    if not container.get("State", {}).get("Running"):
        raise ClusterIsolationError("The isolated PostgreSQL Compose container is not running")
    mounts = cast(list[dict[str, Any]], container.get("Mounts", []))
    volumes: list[str | None] = [
        mount.get("Name")
        for mount in mounts
        if mount.get("Type") == "volume" and mount.get("Destination") == "/var/lib/postgresql"
    ]
    bindings = cast(
        list[dict[str, str]],
        container.get("HostConfig", {}).get("PortBindings", {}).get("5432/tcp") or [],
    )
    host_ports = {(binding.get("HostIp"), binding.get("HostPort")) for binding in bindings}
    volume_name = volumes[0] if len(volumes) == 1 else ""
    host_port = "55433" if host_ports == {("127.0.0.1", "55433")} else ""
    require_isolated_compose_identity(
        str(labels.get("com.docker.compose.project", "")),
        str(labels.get("com.docker.compose.service", "")),
        volume_name or "",
        host_port,
    )
