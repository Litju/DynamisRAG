"""PostgreSQL probe behaviour with no server reachable.

The positive path against a live PostgreSQL 18 server lives in
``tests/integration``; what is verified here is that an unreachable database
degrades into a `down` verdict instead of an unhandled exception.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy import Engine
from sqlalchemy.pool import QueuePool

from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from dynamisrag.db.probe import POSTGRES_DEPENDENCY_NAME, check_postgres
from dynamisrag.health.models import CheckStatus
from tests._support import UNREACHABLE_HOST, build_settings

_MAX_ACCEPTABLE_DETAIL_LENGTH: Final[int] = 300
"""The readiness payload must stay small even when a driver error is verbose."""


def _engine(settings: Settings) -> Engine:
    return create_database_engine(settings)


def test_unreachable_database_reports_down_with_a_diagnostic_detail() -> None:
    engine = _engine(build_settings())

    check = check_postgres(engine)

    assert check.name == POSTGRES_DEPENDENCY_NAME
    assert check.status is CheckStatus.DOWN
    assert check.version is None
    assert check.detail is not None
    assert check.detail.startswith("OperationalError:")
    assert check.latency_ms >= 0
    engine.dispose()


def test_probe_detail_is_bounded_and_never_raises() -> None:
    """A misconfigured DSN must not turn the readiness endpoint into a 500, and
    the reported detail must not dominate the payload."""
    engine = _engine(build_settings(database_url=f"postgresql://user@{UNREACHABLE_HOST}/missing"))

    for _ in range(2):
        check = check_postgres(engine)
        assert check.status is CheckStatus.DOWN
        assert check.detail is not None
        assert len(check.detail) <= _MAX_ACCEPTABLE_DETAIL_LENGTH

    engine.dispose()


def test_engine_is_built_for_psycopg_and_connects_lazily() -> None:
    """Creating an engine must not attempt a connection: the application factory
    has to work before infrastructure exists."""
    engine = _engine(build_settings())

    assert engine.url.drivername == "postgresql+psycopg"
    assert isinstance(engine.pool, QueuePool)
    assert engine.pool.size() == 5
    engine.dispose()
