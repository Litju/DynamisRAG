"""PostgreSQL access layer (SQLAlchemy 2.x over psycopg 3)."""

from __future__ import annotations

from dynamisrag.db.engine import create_database_engine
from dynamisrag.db.probe import POSTGRES_DEPENDENCY_NAME, check_postgres

__all__ = [
    "POSTGRES_DEPENDENCY_NAME",
    "check_postgres",
    "create_database_engine",
]
