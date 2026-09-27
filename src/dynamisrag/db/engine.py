"""SQLAlchemy engine construction for PostgreSQL 18 via psycopg 3."""

from __future__ import annotations

from typing import Final

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import QueuePool

from dynamisrag.config import Settings, to_sqlalchemy_url

__all__ = ["create_database_engine"]

_POOL_SIZE: Final[int] = 5
_MAX_OVERFLOW: Final[int] = 5
_POOL_TIMEOUT_SECONDS: Final[int] = 5
_POOL_RECYCLE_SECONDS: Final[int] = 1800
"""Recycle below the usual one-hour idle limit of managed PostgreSQL proxies."""

_CONNECT_TIMEOUT_SECONDS: Final[int] = 5
"""Driver-level connect timeout; bounds how long a cold probe can block."""


def create_database_engine(settings: Settings) -> Engine:
    """Build a synchronous psycopg 3 engine from validated settings.

    The engine is created eagerly but connects lazily. Connection attempts
    carry an explicit driver timeout and ``pool_pre_ping`` so a stale pooled
    socket surfaces as a probe failure rather than as a request-time error.
    """
    return create_engine(
        to_sqlalchemy_url(settings.database_url),
        poolclass=QueuePool,
        pool_size=_POOL_SIZE,
        max_overflow=_MAX_OVERFLOW,
        pool_timeout=_POOL_TIMEOUT_SECONDS,
        pool_recycle=_POOL_RECYCLE_SECONDS,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": _CONNECT_TIMEOUT_SECONDS,
            "application_name": "dynamisrag",
        },
    )
