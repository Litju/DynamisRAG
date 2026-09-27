"""PostgreSQL connectivity probe backing the readiness endpoint."""

from __future__ import annotations

from time import perf_counter
from typing import Final

from sqlalchemy import String, cast, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from dynamisrag.health.models import CheckStatus, DependencyCheck

__all__ = ["POSTGRES_DEPENDENCY_NAME", "check_postgres"]

POSTGRES_DEPENDENCY_NAME: Final[str] = "postgres"
"""Stable dependency key used in the readiness payload."""

_PROBE_QUERY = select(cast(func.current_setting("server_version"), String).label("server_version"))
"""Single round trip that both proves the session is usable and reports the version."""


def check_postgres(engine: Engine) -> DependencyCheck:
    """Run ``SELECT current_setting('server_version')`` against ``engine``.

    Returns a :class:`~dynamisrag.health.models.DependencyCheck` in every case;
    a failure is reported as ``down`` with a bounded, credential-free detail
    string rather than being raised, so readiness degrades cleanly.
    """
    started_at = perf_counter()
    try:
        with engine.connect() as connection:
            row = connection.execute(_PROBE_QUERY).one()
        server_version: str = str(row.server_version)
    except SQLAlchemyError as error:
        return DependencyCheck(
            name=POSTGRES_DEPENDENCY_NAME,
            status=CheckStatus.DOWN,
            latency_ms=_elapsed_ms(started_at),
            detail=_describe(error),
        )
    return DependencyCheck(
        name=POSTGRES_DEPENDENCY_NAME,
        status=CheckStatus.UP,
        latency_ms=_elapsed_ms(started_at),
        version=server_version,
    )


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _describe(error: SQLAlchemyError) -> str:
    """Render a driver error as ``ExcName: message``.

    ``type(error).__name__`` is always safe. The message is truncated so a
    pathological driver error cannot dominate the readiness payload.
    """
    message = " ".join(str(error).split())
    if len(message) > _MAX_DETAIL_LENGTH:
        message = f"{message[:_MAX_DETAIL_LENGTH]}..."
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


_MAX_DETAIL_LENGTH: Final[int] = 240
