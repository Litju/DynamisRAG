"""Liveness and readiness behaviour with every dependency unreachable."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dynamisrag import __version__
from dynamisrag.health.models import (
    CheckStatus,
    DependencyCheck,
    LivenessReport,
    ReadinessReport,
)
from dynamisrag.health.router import LIVENESS_PATH, READINESS_PATH, SERVICE_NAME


def test_liveness_succeeds_while_every_dependency_is_down(
    offline_client: TestClient,
) -> None:
    """Liveness must never depend on infrastructure, or an outage would cause a
    restart loop instead of a traffic drain."""
    response = offline_client.get(LIVENESS_PATH)

    assert response.status_code == 200
    report = LivenessReport.model_validate(response.json())
    assert report.status is CheckStatus.UP
    assert report.service == SERVICE_NAME
    assert report.version == __version__


def test_health_responses_are_not_cacheable(offline_client: TestClient) -> None:
    for path in (LIVENESS_PATH, READINESS_PATH):
        assert offline_client.get(path).headers["cache-control"] == "no-store"


def test_readiness_fails_with_503_and_names_every_dependency(
    offline_client: TestClient,
) -> None:
    response = offline_client.get(READINESS_PATH)

    assert response.status_code == 503
    report = ReadinessReport.model_validate(response.json())
    assert report.status is CheckStatus.DOWN
    assert list(report.dependencies) == ["postgres", "opensearch"]
    for check in report.dependencies.values():
        assert check.status is CheckStatus.DOWN
        assert check.detail
        assert check.version is None


def test_readiness_body_is_identical_in_shape_for_success_and_failure(
    offline_client: TestClient,
) -> None:
    """A failure must stay machine-readable, not collapse into a stack trace."""
    response = offline_client.get(READINESS_PATH)

    assert response.headers["content-type"].startswith("application/json")
    assert response.status_code == 503
    parsed = ReadinessReport.model_validate(response.json())
    assert isinstance(parsed, ReadinessReport)


def test_unknown_route_is_404(offline_client: TestClient) -> None:
    assert offline_client.get("/does-not-exist").status_code == 404


def test_openapi_document_is_served_and_documents_both_health_paths(
    offline_client: TestClient,
) -> None:
    schema = offline_client.get("/openapi.json").json()

    assert LIVENESS_PATH in schema["paths"]
    assert READINESS_PATH in schema["paths"]
    assert "503" in schema["paths"][READINESS_PATH]["get"]["responses"]


def test_dependency_check_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra_forbidden"):
        DependencyCheck.model_validate(
            {"name": "postgres", "status": "up", "latency_ms": 1, "unexpected": True}
        )
