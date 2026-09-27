"""Liveness and readiness endpoints.

The router is built by a factory rather than declared at import time so that
each application instance closes over its own, immutable resources. Nothing is
stored in module-level mutable state, which keeps the endpoints deterministic
and lets tests exercise several differently-configured apps in one process.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter, Response, status
from sqlalchemy import Engine

from dynamisrag import __version__
from dynamisrag.config import Settings
from dynamisrag.db.probe import check_postgres
from dynamisrag.health.models import (
    CheckStatus,
    DependencyCheck,
    LivenessReport,
    ReadinessReport,
)
from dynamisrag.search.opensearch import OpenSearchProbe

__all__ = [
    "LIVENESS_PATH",
    "READINESS_PATH",
    "SERVICE_NAME",
    "build_health_router",
]

LIVENESS_PATH: Final[str] = "/healthz"
READINESS_PATH: Final[str] = "/readyz"
SERVICE_NAME: Final[str] = "dynamisrag"

_NO_STORE: Final[str] = "no-store"
"""Health responses must never be cached by a proxy or a browser."""


def build_health_router(
    *,
    settings: Settings,
    engine: Engine,
    opensearch: OpenSearchProbe,
) -> APIRouter:
    """Create a router bound to one application's resources.

    Readiness checks run in a fixed order -- PostgreSQL, then OpenSearch -- so
    the dependency mapping in the response body has a stable key order.
    """
    router = APIRouter(tags=["health"])

    @router.get(
        LIVENESS_PATH,
        response_model=LivenessReport,
        summary="Liveness probe",
        description=(
            "Reports that the process is running. Touches no infrastructure, so a "
            "dependency outage can never cause an orchestrator to restart a healthy "
            "process."
        ),
    )
    def liveness(response: Response) -> LivenessReport:
        response.headers["Cache-Control"] = _NO_STORE
        return LivenessReport(
            status=CheckStatus.UP,
            service=SERVICE_NAME,
            version=__version__,
            environment=settings.environment.value,
        )

    @router.get(
        READINESS_PATH,
        response_model=ReadinessReport,
        summary="Readiness probe",
        description=(
            "Verifies every required dependency explicitly. Returns HTTP 503 when "
            "PostgreSQL or OpenSearch is unreachable, with a per-dependency verdict "
            "in the body."
        ),
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessReport}},
    )
    def readiness(response: Response) -> ReadinessReport:
        response.headers["Cache-Control"] = _NO_STORE
        postgres_check: DependencyCheck = check_postgres(engine)
        opensearch_check: DependencyCheck = opensearch.check()
        dependencies = {
            postgres_check.name: postgres_check,
            opensearch_check.name: opensearch_check,
        }
        all_up = all(item.status is CheckStatus.UP for item in dependencies.values())
        report = ReadinessReport(
            status=CheckStatus.UP if all_up else CheckStatus.DOWN,
            service=SERVICE_NAME,
            version=__version__,
            environment=settings.environment.value,
            dependencies=dependencies,
        )
        if report.status is not CheckStatus.UP:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return report

    return router
