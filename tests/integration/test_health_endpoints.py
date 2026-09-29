"""Liveness and readiness against the live PostgreSQL 18 and OpenSearch 3.x stack."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from dynamisrag import __version__
from dynamisrag.application import create_app
from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.probe import POSTGRES_DEPENDENCY_NAME, check_postgres
from dynamisrag.health.models import CheckStatus, LivenessReport, ReadinessReport
from dynamisrag.health.router import LIVENESS_PATH, READINESS_PATH, SERVICE_NAME
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.opensearch import OPENSEARCH_DEPENDENCY_NAME, OpenSearchProbe

_EXPECTED_POSTGRES_MAJOR: Final[str] = "18"
_EXPECTED_OPENSEARCH_MAJOR: Final[str] = "3."


@pytest.fixture
def live_client(live_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(live_settings)) as client:
        yield client


@pytest.mark.integration
def test_liveness_reports_up(live_client: TestClient) -> None:
    response = live_client.get(LIVENESS_PATH)

    assert response.status_code == 200
    report = LivenessReport.model_validate(response.json())
    assert report.status is CheckStatus.UP
    assert report.service == SERVICE_NAME
    assert report.version == __version__


@pytest.mark.integration
def test_readiness_reports_200_and_both_dependencies_healthy(live_client: TestClient) -> None:
    response = live_client.get(READINESS_PATH)

    assert response.status_code == 200
    report = ReadinessReport.model_validate(response.json())
    assert report.status is CheckStatus.UP
    assert list(report.dependencies) == [
        POSTGRES_DEPENDENCY_NAME,
        OPENSEARCH_DEPENDENCY_NAME,
    ]
    for check in report.dependencies.values():
        assert check.status is CheckStatus.UP
        assert check.detail is None
        assert check.version


@pytest.mark.integration
def test_postgres_probe_reports_the_pinned_major_version(live_settings: Settings) -> None:
    engine = create_database_engine(live_settings)
    try:
        check = check_postgres(engine)
    finally:
        engine.dispose()

    assert check.status is CheckStatus.UP
    assert check.version is not None
    assert check.version.startswith(_EXPECTED_POSTGRES_MAJOR)


@pytest.mark.integration
def test_opensearch_probe_reports_the_pinned_major_version(live_settings: Settings) -> None:
    client = OpenSearchClient(live_settings)
    try:
        check = OpenSearchProbe(client).check()
    finally:
        client.close()

    assert check.status is CheckStatus.UP
    assert check.version is not None
    assert check.version.startswith(_EXPECTED_OPENSEARCH_MAJOR)


@pytest.mark.integration
def test_opensearch_credentials_are_required(live_settings: Settings) -> None:
    """A wrong password must produce a clean `down` verdict, not a 500."""
    with_wrong_password = live_settings.model_copy(
        update={"opensearch_password": SecretStr("definitely-not-the-password-1A")}
    )
    client = OpenSearchClient(with_wrong_password)
    try:
        check = OpenSearchProbe(client).check()
    finally:
        client.close()

    assert check.status is CheckStatus.DOWN
    assert check.detail is not None
    assert check.detail.startswith("AuthenticationFailed")
