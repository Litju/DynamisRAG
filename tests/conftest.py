"""Fixtures shared by the unit and integration suites."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

import psycopg
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.res145_evidence_demo import scratch_database

from dynamisrag.application import create_app
from dynamisrag.config import Settings, load_settings
from dynamisrag.db.test_cluster import (
    TEST_DATABASE_ROLE,
    TEST_POSTGRES_HOST_PORT,
    require_test_cluster_identity,
    require_test_cluster_system_identifier,
    validate_test_database_dsns,
    verify_test_cluster_system_identifier,
    verify_test_compose_resources,
)
from tests._support import build_settings

_MISSING_ENVIRONMENT_HINT = (
    "Integration tests need the isolated Compose cluster, DYNAMISRAG_TEST_DATABASE_URL, "
    "DYNAMISRAG_TEST_ADMIN_DATABASE_URL, and DYNAMISRAG_TEST_CLUSTER_SYSTEM_IDENTIFIER. "
    "From PowerShell run Copy-Item .env.example .env, then use the dynamisrag-integration "
    "Compose project, volume, and port documented in docs/migration-safety.md; "
    "then run scripts/init-test-role.sh"
)


@pytest.fixture
def offline_settings() -> Settings:
    """Settings that point at closed loopback ports: every dependency is down."""
    return build_settings()


@pytest.fixture
def offline_client(offline_settings: Settings) -> Iterator[TestClient]:
    """Test client for an application whose dependencies are all unreachable.

    Entered as a context manager so the ASGI lifespan runs, which also exercises
    engine disposal and probe-client cleanup on shutdown.
    """
    with TestClient(create_app(offline_settings)) as client:
        yield client


@pytest.fixture
def live_settings() -> Iterator[Settings]:
    """Use a fresh marked database on the dedicated non-superuser test cluster.

    Fails loudly when the configuration is absent or invalid. An explicit
    integration run must never degrade into a silent skip, so the actionable
    setup hint is attached and the raw validation traceback suppressed. Each
    integration test gets its own database; the default ``uv run pytest`` run
    is unaffected because ``-m 'not integration'``
    deselects these tests before this fixture is ever requested.
    """
    try:
        load_settings()
    except ValidationError as error:
        pytest.fail(
            f"{_MISSING_ENVIRONMENT_HINT} ({error.error_count()} invalid setting(s))",
            pytrace=False,
        )
    with ExitStack() as stack:
        try:
            stack.enter_context(scratch_database())
        except RuntimeError as error:
            pytest.fail(f"{_MISSING_ENVIRONMENT_HINT} ({error})", pytrace=False)
        yield load_settings()


@pytest.fixture(scope="session", autouse=True)
def require_isolated_integration_cluster(request: pytest.FixtureRequest) -> None:
    """Preflight every explicit integration run before any test contacts a service."""
    if not any(item.get_closest_marker("integration") for item in request.session.items):
        return
    try:
        settings = load_settings()
        if settings.test_database_url is None or settings.test_admin_database_url is None:
            raise RuntimeError("the test and admin PostgreSQL DSNs are required")
        system_identifier = require_test_cluster_system_identifier(
            settings.test_cluster_system_identifier
        )
        validate_test_database_dsns(
            str(settings.database_url),
            str(settings.test_database_url),
            str(settings.test_admin_database_url),
            test_host_port=TEST_POSTGRES_HOST_PORT,
        )
        verify_test_compose_resources()
        with psycopg.connect(str(settings.test_admin_database_url)) as admin:
            verify_test_cluster_system_identifier(admin, system_identifier)
            require_test_cluster_identity(admin, system_identifier, expected_database="postgres")
        with psycopg.connect(str(settings.test_database_url)) as test:
            require_test_cluster_identity(
                test,
                system_identifier,
                expected_database="postgres",
                expected_user=TEST_DATABASE_ROLE,
            )
    except RuntimeError as error:
        pytest.fail(f"{_MISSING_ENVIRONMENT_HINT} ({error})", pytrace=False)
    except ValidationError as error:
        pytest.fail(
            f"{_MISSING_ENVIRONMENT_HINT} ({error.error_count()} invalid setting(s))",
            pytrace=False,
        )
    except psycopg.Error as error:
        pytest.fail(
            f"{_MISSING_ENVIRONMENT_HINT} (PostgreSQL preflight failed with SQLSTATE "
            f"{error.sqlstate or 'unknown'})",
            pytrace=False,
        )
