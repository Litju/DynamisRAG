"""Wire-format models for the liveness and readiness endpoints.

These models are the published contract of the health surface, so they are
pydantic models rather than ad-hoc dictionaries: field order, types and the
serialised representation are therefore deterministic.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CheckStatus",
    "DependencyCheck",
    "LivenessReport",
    "ReadinessReport",
]


class CheckStatus(StrEnum):
    """Outcome of a single liveness or readiness verdict."""

    UP = "up"
    DOWN = "down"


class DependencyCheck(BaseModel):
    """Result of probing one required dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: CheckStatus
    latency_ms: int = Field(ge=0)
    version: str | None = None
    detail: str | None = None
    """Populated only when ``status`` is :attr:`CheckStatus.DOWN`."""


class LivenessReport(BaseModel):
    """Body of ``GET /healthz``.

    Liveness answers "is this process running?". It is intentionally
    infrastructure-free so a database outage can never trigger a restart loop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CheckStatus
    service: str
    version: str
    environment: str


class ReadinessReport(BaseModel):
    """Body of ``GET /readyz``.

    Readiness answers "can this process serve traffic right now?", which means
    every required dependency must be reachable. The HTTP status is ``200`` when
    ``status`` is :attr:`CheckStatus.UP` and ``503`` otherwise; the body is
    identical in both cases so failures stay machine-readable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CheckStatus
    service: str
    version: str
    environment: str
    dependencies: dict[str, DependencyCheck]
    """Keyed by dependency name, always populated in declaration order."""
