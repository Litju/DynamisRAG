"""Fixtures shared by the unit and integration suites."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dynamisrag.application import create_app
from dynamisrag.config import Settings, load_settings
from tests._support import build_settings

_MISSING_ENVIRONMENT_HINT = (
    "Integration tests need the live services and a populated environment. From "
    "PowerShell in the repository root run: Copy-Item .env.example .env; "
    "docker compose up -d --wait; uv run alembic upgrade head"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip integration tests with an actionable message when infra is absent.

    A bare ``pydantic.ValidationError`` about a missing DSN would be technically
    correct but unhelpful. This only fires when *configuration* is absent, which
    cannot happen in CI because the workflow creates ``.env`` first; a wrong DSN
    with valid configuration still fails the tests loudly.
    """
    integration_items = [item for item in items if item.get_closest_marker("integration")]
    if not integration_items:
        return
    try:
        load_settings()
    except ValidationError as error:
        reason = f"{_MISSING_ENVIRONMENT_HINT} ({error.error_count()} invalid setting(s))"
        for item in integration_items:
            item.add_marker(pytest.mark.skip(reason=reason))


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
def live_settings() -> Settings:
    """Settings for the real local stack, read from the environment and ``.env``."""
    return load_settings()
