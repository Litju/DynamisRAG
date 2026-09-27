"""Alembic environment for DynamisRAG.

The database URL is never read from ``alembic.ini`` or from the command line:
it comes from :func:`dynamisrag.config.load_settings`, so migrations and the
running application can never disagree about the target database. The DSN is
handed straight to :func:`sqlalchemy.create_engine` rather than stored via
``Config.set_main_option``, because ``configparser`` would try to interpolate
``%`` characters that appear in generated passwords.

Both offline (``--sql``) and online modes are wired. Offline mode needs no
DBAPI, which is what makes ``uv run alembic upgrade head --sql`` usable for
reviewing a migration before it touches a database.
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Final

from alembic import context
from sqlalchemy import Connection, MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from dynamisrag.config import load_settings, to_sqlalchemy_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata: Final[MetaData | None] = None
"""No application schema exists yet.

RES-131 introduces the scientific schema. Until models exist there is no
``MetaData`` to compare against, so ``alembic revision --autogenerate`` is
unavailable and every revision is hand-written. This is a deliberate,
documented state rather than an oversight.
"""


def database_url() -> str:
    """Return the SQLAlchemy URL of the configured PostgreSQL database."""
    return to_sqlalchemy_url(load_settings().database_url)


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Render migration SQL to stdout without opening a connection."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live PostgreSQL server.

    ``NullPool`` keeps migration runs free of pooled state, so a run never
    inherits a connection left behind by a previous process and never leaves
    one behind itself.
    """
    engine: Engine = create_engine(database_url(), poolclass=NullPool)
    try:
        with engine.connect() as connection:
            _run_migrations(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
