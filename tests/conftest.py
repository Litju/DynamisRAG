"""Fixtures shared by the unit and integration suites."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.res145_evidence_demo import scratch_database

from dynamisrag.application import create_app
from dynamisrag.config import Settings, load_settings
from tests._support import build_settings

_MISSING_ENVIRONMENT_HINT = (
    "Integration tests need isolated live services and DYNAMISRAG_TEST_DATABASE_URL. From "
    "PowerShell in the repository root run: Copy-Item .env.example .env; "
    "start a separate PostgreSQL test cluster; run scripts/init-test-role.sh; "
    "verify DYNAMISRAG_TEST_DATABASE_URL points to its postgres database"
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
